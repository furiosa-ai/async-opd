"""TP weight gathering for Megatron trainer backend.

Provides utilities to:
  - Convert Megatron param names to HuggingFace format
  - Gather TP-sharded weights to rank-0 as a full HF-format state dict
  - Build HF-format param metadata without gathering (used at init time)

Supports both TP=1 (single rank, no communication) and TP>1 (all-gather
across the TP process group to reconstruct full tensors on rank 0).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


# ---------------------------------------------------------------------------
# Name conversion table: Megatron → HuggingFace
# ---------------------------------------------------------------------------

# Each entry is (regex_pattern, replacement_template).
# Patterns are tried in order; first match wins.
# Layer-indexed patterns use a named group `n` for the layer index.
_MEGATRON_TO_HF_PATTERNS: list[tuple[str, str]] = [
    # Attention linear_qkv (output-parallel, sharded dim=0)
    (
        r"^decoder\.layers\.(?P<n>\d+)\.self_attention\.linear_qkv\.weight$",
        r"model.layers.\g<n>.self_attn.qkv_proj.weight",
    ),
    # Attention output projection (input-parallel, sharded dim=1)
    (
        r"^decoder\.layers\.(?P<n>\d+)\.self_attention\.linear_proj\.weight$",
        r"model.layers.\g<n>.self_attn.o_proj.weight",
    ),
    # MLP gate+up (output-parallel, sharded dim=0)
    (
        r"^decoder\.layers\.(?P<n>\d+)\.mlp\.linear_fc1\.weight$",
        r"model.layers.\g<n>.mlp.gate_up_proj.weight",
    ),
    # MLP down (input-parallel, sharded dim=1)
    (
        r"^decoder\.layers\.(?P<n>\d+)\.mlp\.linear_fc2\.weight$",
        r"model.layers.\g<n>.mlp.down_proj.weight",
    ),
    # Input layernorm — fused (TE: linear_qkv.layer_norm_weight)
    (
        r"^decoder\.layers\.(?P<n>\d+)\.self_attention\.linear_qkv\.layer_norm_weight$",
        r"model.layers.\g<n>.input_layernorm.weight",
    ),
    # Input layernorm — separate (non-TE: input_layernorm.weight)
    (
        r"^decoder\.layers\.(?P<n>\d+)\.input_layernorm\.weight$",
        r"model.layers.\g<n>.input_layernorm.weight",
    ),
    # Post-attention layernorm — fused (TE: linear_fc1.layer_norm_weight)
    (
        r"^decoder\.layers\.(?P<n>\d+)\.mlp\.linear_fc1\.layer_norm_weight$",
        r"model.layers.\g<n>.post_attention_layernorm.weight",
    ),
    # Post-attention layernorm — separate (non-TE: pre_mlp_layernorm.weight)
    (
        r"^decoder\.layers\.(?P<n>\d+)\.pre_mlp_layernorm\.weight$",
        r"model.layers.\g<n>.post_attention_layernorm.weight",
    ),
    # QKV bias (Qwen2.5)
    (
        r"^decoder\.layers\.(?P<n>\d+)\.self_attention\.linear_qkv\.bias$",
        r"model.layers.\g<n>.self_attn.qkv_proj.bias",
    ),
    # Embedding
    (
        r"^embedding\.word_embeddings\.weight$",
        r"model.embed_tokens.weight",
    ),
    # Final layernorm
    (
        r"^decoder\.final_layernorm\.weight$",
        r"model.norm.weight",
    ),
    # QK layernorm (Qwen3)
    (
        r"^decoder\.layers\.(?P<n>\d+)\.self_attention\.q_layernorm\.weight$",
        r"model.layers.\g<n>.self_attn.q_norm.weight",
    ),
    (
        r"^decoder\.layers\.(?P<n>\d+)\.self_attention\.k_layernorm\.weight$",
        r"model.layers.\g<n>.self_attn.k_norm.weight",
    ),
    # LM head
    (
        r"^output_layer\.weight$",
        r"lm_head.weight",
    ),
]

# Pre-compile for performance
_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat), repl) for pat, repl in _MEGATRON_TO_HF_PATTERNS
]

# Expose the raw table for introspection
MEGATRON_TO_HF_NAME_MAP: dict[str, str] = {
    pat: repl for pat, repl in _MEGATRON_TO_HF_PATTERNS
}


# ---------------------------------------------------------------------------
# TP-sharding metadata
# ---------------------------------------------------------------------------

# Maps a substring of the Megatron param name → concat dimension.
# These are the params that are split across TP ranks.
_TP_SHARDED_DIMS: dict[str, int] = {
    "linear_qkv.weight": 0,   # output-parallel → concat on dim 0
    "linear_qkv.bias": 0,     # QKV bias (Qwen2) — same sharding as weight
    "linear_proj.weight": 1,  # input-parallel  → concat on dim 1
    "linear_fc1.weight": 0,   # output-parallel → concat on dim 0
    "linear_fc2.weight": 1,   # input-parallel  → concat on dim 1
    "word_embeddings.weight": 0,  # vocab-parallel → concat on dim 0
    "output_layer.weight": 0,     # vocab-parallel → concat on dim 0
}


def _tp_concat_dim(megatron_name: str) -> int | None:
    """Return the concatenation dimension for a TP-sharded param, or None."""
    for substr, dim in _TP_SHARDED_DIMS.items():
        if megatron_name.endswith(substr):
            return dim
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def megatron_to_hf_name(megatron_name: str) -> str:
    """Convert a single Megatron param name to HuggingFace format.

    Raises ValueError if no pattern matches.
    """
    for pattern, repl in _COMPILED_PATTERNS:
        m = pattern.match(megatron_name)
        if m is not None:
            return pattern.sub(repl, megatron_name)
    raise ValueError(
        f"megatron_to_hf_name: no pattern matched '{megatron_name}'"
    )


def gather_tp_weights(
    model: torch.nn.Module,
    tp_group,  # torch.distributed ProcessGroup or None
) -> dict[str, Tensor]:
    """Gather TP-sharded weights to rank-0 as a full HF-format state dict.

    MVP behaviour (full model per rank):
        If tp_group is None or tp_size == 1, each rank already holds the
        full model.  We simply return the HF state dict without any
        collective communication.

    True TP behaviour (future):
        Each rank holds a shard.  We all-gather across tp_group on the
        appropriate dimension and concatenate on rank-0.

    Returns:
        On rank 0: dict mapping HF param name → full CPU tensor.
        On other ranks: empty dict (all work is done on rank 0).
    """
    tp_size = 1 if tp_group is None else torch.distributed.get_world_size(group=tp_group)
    tp_rank = 0 if tp_group is None else torch.distributed.get_rank(group=tp_group)

    # ---- MVP path: full model on every rank, no gathering needed ----------
    if tp_size == 1:
        sd: dict[str, Tensor] = {}
        raw = model.state_dict()
        for meg_name, tensor in raw.items():
            # Skip _extra_state entries (Megatron/TE internal, often None)
            if tensor is None or "_extra_state" in meg_name:
                continue
            # Try Megatron conversion; fall back to the name as-is.
            try:
                hf_name = megatron_to_hf_name(meg_name)
            except ValueError:
                hf_name = meg_name
            sd[hf_name] = tensor.cpu()
        return sd

    # ---- True TP path: sharded model, all-gather -------------------------
    # Only rank 0 of the TP group accumulates the full dict.
    result: dict[str, Tensor] = {}

    for meg_name, param in model.named_parameters():
        concat_dim = _tp_concat_dim(meg_name)

        if concat_dim is not None:
            # All ranks participate in the gather; rank-0 receives all shards.
            shards = [torch.empty_like(param) for _ in range(tp_size)]
            torch.distributed.all_gather(shards, param, group=tp_group)
            if tp_rank == 0:
                if meg_name.endswith("mlp.linear_fc1.weight"):
                    # Megatron's gated MLP ColumnParallelLinear stores each TP
                    # shard as [gate_shard, up_shard].  Reconstruct HF/vLLM's
                    # fused layout as [all_gate, all_up], not
                    # [gate0, up0, gate1, up1].
                    local_half = shards[0].size(0) // 2
                    gate = torch.cat([s[:local_half] for s in shards], dim=0)
                    up = torch.cat([s[local_half:] for s in shards], dim=0)
                    full = torch.cat([gate, up], dim=0)
                else:
                    full = torch.cat(shards, dim=concat_dim)
            else:
                continue  # non-rank-0 skips accumulation
        else:
            # Not TP-sharded: all ranks have the same copy; rank 0 takes it.
            if tp_rank != 0:
                continue
            full = param.detach()

        try:
            hf_name = megatron_to_hf_name(meg_name)
        except ValueError:
            hf_name = meg_name

        result[hf_name] = full.cpu()

    return result


def build_megatron_weights_info(
    model: torch.nn.Module,
    tp_size: int,
) -> list[tuple[str, tuple, str]]:
    """Compute HF-format param metadata WITHOUT gathering actual tensors.

    Called eagerly during trainer __init__ so get_weights_info() returns
    immediately (no collective communication).

    Args:
        model:   The (possibly sharded) model on this rank.
        tp_size: Tensor-parallel world size (1 = no TP / MVP path).

    Returns:
        List of (hf_name, full_shape, dtype_str) tuples, where full_shape
        reflects the unsharded size (i.e. sharded dim multiplied by tp_size).

    The output format matches what build_weight_merge_map() in weight_merge.py
    expects for its trainer_weights_info argument.
    """
    info: list[tuple[str, tuple, str]] = []

    for meg_name, param in model.named_parameters():
        try:
            hf_name = megatron_to_hf_name(meg_name)
        except ValueError:
            hf_name = meg_name

        shape = list(param.shape)

        if tp_size > 1:
            concat_dim = _tp_concat_dim(meg_name)
            if concat_dim is not None:
                shape[concat_dim] *= tp_size

        dtype_str = str(param.dtype).replace("torch.", "")
        info.append((hf_name, tuple(shape), dtype_str))

    return info

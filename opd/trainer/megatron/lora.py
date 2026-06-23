"""TP-aware LoRA for Megatron-Core GPTModel.

Adds LoRA adapters to mcore ColumnParallelLinear / RowParallelLinear via
forward hooks — no modification to mcore internals.

For fused layers (linear_qkv = q+k+v, linear_fc1 = gate+up), separate
LoRA adapters are used for each sub-projection so that vLLM's native LoRA
can consume them directly (it expects per-projection q/k/v adapters).

TP-aware sharding:
  - ColumnParallelLinear: LoRA A replicated, LoRA B sharded on dim 0
  - RowParallelLinear: LoRA A sharded on dim 1, LoRA B replicated

Weight sync extracts LoRA A/B with HF-format names and TP-gathers them
for transfer to vLLM rollout.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear, RowParallelLinear,
    )


# ---------------------------------------------------------------------------
# LoRA hook functions
# ---------------------------------------------------------------------------

def _column_parallel_lora_hook(module, input, output):
    """Forward hook for ColumnParallelLinear with LoRA.

    For fused layers (qkv, gate_up), applies separate LoRA per sub-projection
    and concatenates the deltas.
    """
    x = input[0]  # [seq, batch, hidden] or [batch, seq, hidden]
    out_tensor, out_bias = output

    if hasattr(module, '_lora_sub_projs'):
        # Fused layer: separate LoRA per sub-projection
        deltas = []
        for A, B, scale in module._lora_sub_projs:
            delta = (x @ A.T) @ B.T * scale
            deltas.append(delta)
        delta = torch.cat(deltas, dim=-1)
    else:
        # Single projection
        delta = (x @ module.lora_A.T) @ module.lora_B.T * module._lora_scale
    return (out_tensor + delta, out_bias)


def _row_parallel_lora_hook(module, input, output):
    """Forward hook for RowParallelLinear with LoRA.

    LoRA A is sharded (matches input sharding). Intermediate is all-reduced
    across TP before multiplying by LoRA B (replicated).
    """
    x_shard = input[0]  # sharded input [seq, batch, shard_in]
    out_tensor, out_bias = output

    intermediate = x_shard @ module.lora_A.T  # [*, rank] per shard
    # All-reduce across TP to reconstruct full intermediate
    if module._lora_tp_size > 1:
        torch.distributed.all_reduce(intermediate, group=module._lora_tp_group)
    delta = intermediate @ module.lora_B.T * module._lora_scale
    return (out_tensor + delta, out_bias)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_megatron_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    target_modules: list[str] | None = None,
    tp_size: int = 1,
    tp_group=None,
    dtype: torch.dtype = torch.bfloat16,
    hf_config=None,
) -> dict[str, nn.Parameter]:
    """Add LoRA adapters to a Megatron GPTModel.

    Args:
        model: mcore GPTModel (with decoder.layers)
        rank: LoRA rank (r)
        alpha: LoRA alpha (scaling = alpha / rank)
        target_modules: which modules to target. Default: all 4 linear types.
            Options: "linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"
        tp_size: tensor parallel world size
        tp_group: torch.distributed process group for TP
        dtype: dtype for LoRA parameters

    Returns:
        Dict of {param_name: nn.Parameter} for all LoRA params (for optimizer).
    """
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear, RowParallelLinear,
    )

    if target_modules is None:
        target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

    scale = alpha / rank
    lora_params = {}
    device = next(model.parameters()).device

    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    # Precompute QKV split sizes from HF config (avoids ambiguous heuristics)
    qkv_sizes_per_tp = None
    if hf_config is not None:
        num_heads = hf_config.num_attention_heads
        num_kv = getattr(hf_config, "num_key_value_heads", num_heads)
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // num_heads)
        # Per-TP-rank sizes
        q_per_tp = (num_heads // tp_size) * head_dim
        kv_per_tp = (num_kv // tp_size) * head_dim
        qkv_sizes_per_tp = [q_per_tp, kv_per_tp, kv_per_tp]

    for layer_idx, layer in enumerate(model.decoder.layers):
        for mod_name, mod in layer.named_modules():
            short_name = mod_name.split(".")[-1]
            if short_name not in target_modules:
                continue

            if isinstance(mod, ColumnParallelLinear):
                _add_column_parallel_lora(
                    mod, mod_name, layer_idx, rank, scale, tp_size, dtype, device,
                    lora_params, qkv_sizes_per_tp=qkv_sizes_per_tp)
            elif isinstance(mod, RowParallelLinear):
                _add_row_parallel_lora(
                    mod, mod_name, layer_idx, rank, scale, tp_size, tp_group,
                    dtype, device, lora_params)

    n_lora = sum(p.numel() for p in lora_params.values())
    n_base = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[Megatron-LoRA] {len(lora_params)} LoRA params "
          f"({n_lora/1e6:.1f}M trainable, {n_base/1e6:.1f}M frozen, "
          f"rank={rank}, alpha={alpha})", flush=True)

    return lora_params


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Mapping from mcore fused module names to HF sub-projection names + sizes.
# Each entry: (hf_suffixes, size_fn) where size_fn(mod) → list of output sizes per sub-proj.
_FUSED_MODULES = {
    "linear_qkv": {
        # q_proj, k_proj, v_proj — sizes depend on GQA config
        "hf_names": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    },
    "linear_fc1": {
        # gate_proj, up_proj — equal halves
        "hf_names": ["mlp.gate_proj", "mlp.up_proj"],
    },
}

# Non-fused modules: direct 1:1 mapping
_SIMPLE_MODULES = {
    "linear_proj": "self_attn.o_proj",
    "linear_fc2": "mlp.down_proj",
}



def _gate_up_split_sizes(mod) -> list[int]:
    """Compute per-TP-rank sizes for gate and up from a fused gate_up linear."""
    total = mod.output_size_per_partition
    half = total // 2
    return [half, half]


def _add_column_parallel_lora(mod, mod_name, layer_idx, rank, scale,
                              tp_size, dtype, device, lora_params,
                              qkv_sizes_per_tp=None):
    """Add LoRA to a ColumnParallelLinear (qkv or gate_up or standalone)."""
    short_name = mod_name.split(".")[-1]
    in_features = mod.input_size  # full (not sharded)

    if short_name in _FUSED_MODULES:
        # Fused: separate LoRA per sub-projection
        info = _FUSED_MODULES[short_name]
        hf_names = info["hf_names"]

        if short_name == "linear_qkv":
            if qkv_sizes_per_tp is not None:
                sizes = qkv_sizes_per_tp
            else:
                sizes = _compute_qkv_sizes(mod, tp_size)
            mod._lora_qkv_sizes = sizes
        else:
            sizes = _gate_up_split_sizes(mod)

        sub_projs = []
        for i, (hf_name, out_size) in enumerate(zip(hf_names, sizes)):
            # A: [rank, in_features] — replicated across TP
            A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            # B: [out_size_per_tp, rank] — sharded (each TP rank has its portion)
            B = nn.Parameter(torch.zeros(out_size, rank, dtype=dtype, device=device))

            A.requires_grad = True
            B.requires_grad = True

            param_prefix = f"decoder.layers.{layer_idx}.{hf_name}"
            A_name = f"{param_prefix}.lora_A"
            B_name = f"{param_prefix}.lora_B"

            # Register as module attributes for state_dict access
            setattr(mod, f"_lora_A_{i}", A)
            setattr(mod, f"_lora_B_{i}", B)
            lora_params[A_name] = A
            lora_params[B_name] = B

            sub_projs.append((A, B, scale))

        mod._lora_sub_projs = sub_projs
        mod.register_forward_hook(_column_parallel_lora_hook)
    else:
        # Non-fused column parallel (shouldn't happen with current targets)
        out_size = mod.output_size_per_partition
        A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        B = nn.Parameter(torch.zeros(out_size, rank, dtype=dtype, device=device))
        A.requires_grad = True
        B.requires_grad = True

        hf_name = _SIMPLE_MODULES.get(short_name, mod_name)
        param_prefix = f"decoder.layers.{layer_idx}.{hf_name}"
        setattr(mod, "lora_A", A)
        setattr(mod, "lora_B", B)
        mod._lora_scale = scale
        lora_params[f"{param_prefix}.lora_A"] = A
        lora_params[f"{param_prefix}.lora_B"] = B
        mod.register_forward_hook(_column_parallel_lora_hook)


def _add_row_parallel_lora(mod, mod_name, layer_idx, rank, scale,
                           tp_size, tp_group, dtype, device, lora_params):
    """Add LoRA to a RowParallelLinear (o_proj or down_proj)."""
    short_name = mod_name.split(".")[-1]
    shard_in = mod.input_size_per_partition  # sharded input size
    out_features = mod.output_size  # full output

    # A: [rank, shard_in] — sharded (each TP rank has different portion)
    A = nn.Parameter(torch.empty(rank, shard_in, dtype=dtype, device=device))
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    # B: [out_features, rank] — replicated (same on all TP ranks)
    B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
    A.requires_grad = True
    B.requires_grad = True

    hf_name = _SIMPLE_MODULES.get(short_name, mod_name)
    param_prefix = f"decoder.layers.{layer_idx}.{hf_name}"

    setattr(mod, "lora_A", A)
    setattr(mod, "lora_B", B)
    mod._lora_scale = scale
    mod._lora_tp_size = tp_size
    mod._lora_tp_group = tp_group
    lora_params[f"{param_prefix}.lora_A"] = A
    lora_params[f"{param_prefix}.lora_B"] = B
    mod.register_forward_hook(_row_parallel_lora_hook)


def _compute_qkv_sizes(mod, tp_size: int) -> list[int]:
    """Compute per-TP-rank output sizes for [q, k, v] in a fused qkv linear.

    Uses the total output_size_per_partition and tries GQA ratios first
    (since most modern models use GQA), then falls back to MHA equal split.
    """
    total = mod.output_size_per_partition

    if hasattr(mod, '_lora_qkv_sizes'):
        return mod._lora_qkv_sizes

    # GQA: total = q + 2*kv where q = kv * heads_per_group
    # So total = kv * (heads_per_group + 2)
    # Try common heads_per_group ratios first (GQA is more common than MHA)
    for hpg in [2, 4, 7, 8, 16, 32]:
        kv = total // (hpg + 2)
        if kv > 0 and kv * (hpg + 2) == total:
            return [kv * hpg, kv, kv]

    # MHA fallback: q=k=v equal split
    if total % 3 == 0:
        s = total // 3
        return [s, s, s]

    raise ValueError(
        f"Cannot infer QKV split sizes from total={total}. "
        f"Please set hf_config on the model."
    )


# ---------------------------------------------------------------------------
# LoRA state dict extraction (for weight sync)
# ---------------------------------------------------------------------------

def gather_lora_state_dict(
    model: nn.Module,
    lora_params: dict[str, nn.Parameter],
    tp_size: int = 1,
    tp_rank: int = 0,
    tp_group=None,
    pp_size: int = 1,
    pp_rank: int = 0,
    layers_per_stage: int = 0,
) -> dict[str, torch.Tensor]:
    """Extract LoRA state dict with HF-format names, TP-gathered.

    Returns dict suitable for vLLM's native LoRA (TensorLoRARequest).
    Names follow: model.layers.{global_idx}.{hf_module}.lora_A.weight

    For TP>1: gathers sharded LoRA params across TP ranks.
    Only returns non-empty on TP-rank-0.
    """
    layer_offset = pp_rank * layers_per_stage if pp_size > 1 else 0
    result = {}

    for param_name, param in lora_params.items():
        # param_name: "decoder.layers.{local_idx}.{hf_module}.lora_{A|B}"
        m = re.match(r"decoder\.layers\.(\d+)\.(.*)\.(lora_[AB])", param_name)
        if not m:
            continue
        local_idx = int(m.group(1))
        hf_module = m.group(2)
        lora_part = m.group(3)
        global_idx = local_idx + layer_offset
        hf_name = f"model.layers.{global_idx}.{hf_module}.{lora_part}.weight"

        if tp_size > 1:
            tensor = _gather_lora_param(param, param_name, hf_module, lora_part,
                                        tp_size, tp_rank, tp_group)
            if tp_rank != 0:
                continue
        else:
            tensor = param.detach()

        result[hf_name] = tensor.to(torch.bfloat16).cpu()

    return result


def _gather_lora_param(param, param_name, hf_module, lora_part,
                       tp_size, tp_rank, tp_group):
    """Gather a TP-sharded LoRA parameter across TP ranks.

    ColumnParallelLinear:
      - lora_A: replicated → just take rank-0's copy
      - lora_B: sharded on dim 0 → all-gather and concat on dim 0

    RowParallelLinear:
      - lora_A: sharded on dim 1 → all-gather and concat on dim 1
      - lora_B: replicated → just take rank-0's copy
    """
    # Determine if this is column-parallel or row-parallel from hf_module name
    is_col = any(s in hf_module for s in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"])
    is_row = any(s in hf_module for s in ["o_proj", "down_proj"])

    if is_col:
        if lora_part == "lora_A":
            # Replicated — rank 0 has the canonical copy
            return param.detach()
        else:
            # lora_B: sharded on dim 0 → gather
            return _all_gather_concat(param, dim=0, tp_size=tp_size, tp_group=tp_group)
    elif is_row:
        if lora_part == "lora_A":
            # Sharded on dim 1 → gather
            return _all_gather_concat(param, dim=1, tp_size=tp_size, tp_group=tp_group)
        else:
            # lora_B: replicated — rank 0 has the canonical copy
            return param.detach()
    else:
        return param.detach()


def _all_gather_concat(tensor, dim, tp_size, tp_group):
    """All-gather a tensor across TP and concatenate along dim."""
    shards = [torch.empty_like(tensor) for _ in range(tp_size)]
    torch.distributed.all_gather(shards, tensor, group=tp_group)
    return torch.cat(shards, dim=dim)


def build_lora_weights_info(
    lora_params: dict[str, nn.Parameter],
    tp_size: int = 1,
    pp_size: int = 1,
    pp_rank: int = 0,
    layers_per_stage: int = 0,
) -> list[tuple[str, torch.Size, torch.dtype]]:
    """Build (name, shape, dtype) list for LoRA weight sync.

    Returns HF-format names with full (TP-gathered) shapes.
    """
    layer_offset = pp_rank * layers_per_stage if pp_size > 1 else 0
    info = []

    for param_name, param in lora_params.items():
        m = re.match(r"decoder\.layers\.(\d+)\.(.*)\.(lora_[AB])", param_name)
        if not m:
            continue
        local_idx = int(m.group(1))
        hf_module = m.group(2)
        lora_part = m.group(3)
        global_idx = local_idx + layer_offset
        hf_name = f"model.layers.{global_idx}.{hf_module}.{lora_part}.weight"

        # Compute full (gathered) shape
        shape = list(param.shape)
        if tp_size > 1:
            is_col = any(s in hf_module for s in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"])
            is_row = any(s in hf_module for s in ["o_proj", "down_proj"])
            if is_col and lora_part == "lora_B":
                shape[0] *= tp_size  # gather on dim 0
            elif is_row and lora_part == "lora_A":
                shape[1] *= tp_size  # gather on dim 1

        info.append((hf_name, torch.Size(shape), torch.bfloat16))

    return info

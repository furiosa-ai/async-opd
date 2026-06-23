"""HF-to-Megatron config conversion, GPTModel factory, and weight loading.

Converts a HuggingFace ``PretrainedConfig`` into a Megatron-Core
``TransformerConfig``, provides a ``model_provider`` closure for
``megatron.training.get_model()``, and loads HF checkpoint weights
into a Megatron ``GPTModel``.

Supports 3D parallelism: TP (tensor), PP (pipeline), DP (data).
Adapted from veRL's mcore config/model/loader, extended for PP layer
filtering and full TP+PP weight sharding.

Supports dense architectures: Qwen3, Qwen2, LLaMA.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from opd.utils.config import resolve_trust_remote_code

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformers import PretrainedConfig

log = logging.getLogger(__name__)


def _has_te() -> bool:
    """Check if Transformer Engine is available."""
    try:
        import transformer_engine  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Config conversion
# ---------------------------------------------------------------------------


def _detect_qkv_bias(hf_config: PretrainedConfig) -> bool:
    """Detect whether the model uses bias in QKV projections.

    Qwen2 always uses QKV bias. LLaMA / Qwen3 typically do not.
    """
    arch = hf_config.architectures[0] if hasattr(hf_config, "architectures") else ""
    if "Qwen2" in arch and "Qwen3" not in arch:
        return True
    return getattr(hf_config, "attention_bias", False)


def _detect_qk_layernorm(hf_config: PretrainedConfig) -> bool:
    """Detect whether the model uses QK layernorm (Qwen3-specific)."""
    arch = hf_config.architectures[0] if hasattr(hf_config, "architectures") else ""
    return "Qwen3" in arch


def hf_to_mcore_config(
    model_path: str,
    tp_size: int = 1,
    dp_size: int = 1,
    pp_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool | None = None,
    **override_kwargs,
) -> TransformerConfig:
    """Convert a HuggingFace model config to a Megatron TransformerConfig.

    Args:
        model_path: HuggingFace model name or local path.
        tp_size: Tensor-parallel world size (1 for Phase 1).
        dp_size: Data-parallel world size (unused in config, kept for API).
        dtype: Parameter dtype (default bf16).
        **override_kwargs: Extra kwargs forwarded to TransformerConfig.

    Returns:
        A fully-constructed ``TransformerConfig``.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig as _TC

    hf_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=resolve_trust_remote_code(
            trust_remote_code,
            context="Megatron config loading",
        ),
    )
    return hf_config_to_mcore_config(hf_config, tp_size=tp_size, dp_size=dp_size,
                                     pp_size=pp_size, dtype=dtype, **override_kwargs)


def hf_config_to_mcore_config(
    hf_config: PretrainedConfig,
    tp_size: int = 1,
    dp_size: int = 1,
    pp_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    **override_kwargs,
) -> TransformerConfig:
    """Convert an already-loaded HF config to a Megatron TransformerConfig.

    This is the workhorse; ``hf_to_mcore_config`` is a convenience wrapper
    that loads the HF config from *model_path* first.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig as _TC

    qkv_bias = _detect_qkv_bias(hf_config)
    qk_layernorm = _detect_qk_layernorm(hf_config)

    args: dict = {
        # Architecture
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": getattr(hf_config, "num_key_value_heads",
                                    hf_config.num_attention_heads),
        "ffn_hidden_size": hf_config.intermediate_size,
        "kv_channels": getattr(hf_config, "head_dim", None),
        "layernorm_epsilon": getattr(hf_config, "rms_norm_eps", 1e-6),
        # Activation / normalization
        "activation_func": F.silu,
        "gated_linear_unit": True,
        "normalization": "RMSNorm",
        # Bias
        "add_bias_linear": False,
        "add_qkv_bias": qkv_bias,
        # QK layernorm (Qwen3)
        "qk_layernorm": qk_layernorm,
        # Dropout (typically 0 for these models)
        "attention_dropout": getattr(hf_config, "attention_dropout", 0.0),
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        # Dtype
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,
        # Parallel sizes
        "tensor_model_parallel_size": tp_size,
        "pipeline_model_parallel_size": pp_size,
        # Sequence parallelism requires Transformer Engine fused layernorms.
        # Default: enabled with TP>1. Without TE, raises error unless
        # explicitly overridden: sequence_parallel=False in config.
        "sequence_parallel": tp_size > 1,
        # Activation recomputation (gradient checkpointing)
        # "selective": recompute attention softmax only (best memory/speed tradeoff)
        # "full" + "uniform": recompute all layers (max memory savings, slower)
        # None: disabled
        "recompute_granularity": "selective",
        # Misc
        "variable_seq_lengths": True,
        "use_cpu_initialization": False,
        # allgather dispatcher doesn't support variable_seq_lengths;
        # use alltoall (only matters for MoE, harmless for dense).
        "moe_token_dispatcher_type": "alltoall",
    }

    # Apply caller overrides
    args.update(override_kwargs)

    # Validate: Megatron backend requires Transformer Engine for fused kernels
    # (fused layernorms, fused attention, sequence parallelism with TP>1).
    # Opt out via use_transformer_engine: false in config.
    # The trainer sets _require_te=True; tests/other callers skip the check.
    if override_kwargs.pop("_require_te", False) and not _has_te():
        raise ImportError(
            "Transformer Engine is required for the Megatron backend but not installed. "
            "Install with: pip install transformer-engine[pytorch]\n"
            "Or set training.trainer.use_transformer_engine: false in your config "
            "to use standard PyTorch kernels (slower, no sequence parallelism)."
        )

    # Strip keys not supported by this megatron-core version
    removed = [k for k in list(args) if not hasattr(_TC, k)]
    if removed:
        warnings.warn(
            f"hf_config_to_mcore_config: removing unsupported TransformerConfig "
            f"keys for megatron-core 0.16.1: {removed}",
            stacklevel=2,
        )
        for k in removed:
            args.pop(k)

    return _TC(**args)


# ---------------------------------------------------------------------------
# Model provider
# ---------------------------------------------------------------------------


def _get_layer_spec(config, use_te: bool = True):
    """Get the GPT decoder block spec, preferring TE when available.

    Args:
        config: TransformerConfig.
        use_te: If True, try Transformer Engine spec (fused kernels).
                If False, use local spec (pure-PyTorch, avoids TE's
                fused_attn_fwd which fails on systems with both CUDA 12/13).
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

    if use_te:
        try:
            return get_gpt_decoder_block_spec(config, use_transformer_engine=True)
        except Exception:
            pass
    return get_gpt_decoder_block_spec(config, use_transformer_engine=False)


def make_model_provider(
    config: TransformerConfig,
    hf_config: PretrainedConfig,
    max_seq_len: int | None = None,
):
    """Return a ``model_provider`` closure for ``megatron.training.get_model()``.

    Args:
        config: Megatron TransformerConfig (from ``hf_to_mcore_config``).
        hf_config: HuggingFace PretrainedConfig (for vocab_size, rope_theta, etc.).
        max_seq_len: Override max sequence length. Defaults to
            ``hf_config.max_position_embeddings``.

    Returns:
        A callable ``model_provider(pre_process, post_process) -> GPTModel``.
    """
    vocab_size = hf_config.vocab_size
    seq_len = max_seq_len or hf_config.max_position_embeddings
    rope_theta = getattr(hf_config, "rope_theta", 10000.0)

    # Build rope_scaling kwargs if present
    rope_scaling_kwargs = {}
    if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling is not None:
        rope_scaling_kwargs["seq_len_interpolation_factor"] = (
            hf_config.rope_scaling.get("factor", 1.0)
        )

    def model_provider(pre_process=True, post_process=True) -> GPTModel:
        from megatron.core.models.gpt.gpt_model import GPTModel as _GPTModel

        layer_spec = _get_layer_spec(config)
        share_embeddings = bool(getattr(hf_config, "tie_word_embeddings", False))
        model = _GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=seq_len,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=share_embeddings,
            position_embedding_type="rope",
            rotary_base=rope_theta,
            **rope_scaling_kwargs,
        )
        return model

    return model_provider


# ---------------------------------------------------------------------------
# HF weight loading
# ---------------------------------------------------------------------------


def _resolve_model_dir(model_path: str) -> Path:
    """Resolve *model_path* to a local directory.

    If *model_path* is already a local directory, return it directly.
    If it looks like an absolute or relative filesystem path (starts with
    ``/``, ``./``, ``../``, or ``~``) but doesn't exist, raise
    ``FileNotFoundError``.  Otherwise treat it as a HuggingFace Hub model
    ID (e.g. ``Qwen/Qwen2.5-0.5B-Instruct``) and resolve via
    ``snapshot_download``.
    """
    p = Path(model_path)
    if p.is_dir():
        return p
    # If it looks like a filesystem path, don't try the hub
    if model_path.startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"Local model directory not found: {model_path}")
    # Assume HuggingFace Hub model ID
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path))


def _load_hf_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load a HuggingFace state dict from *model_path* (safetensors or .bin).

    *model_path* can be a local directory or a HuggingFace Hub model ID.
    Tries safetensors first (preferred), then falls back to PyTorch .bin files.
    Loads all shards if the checkpoint is split across multiple files.
    """
    model_dir = _resolve_model_dir(model_path)

    # --- safetensors path ---------------------------------------------------
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if safetensor_files:
        import safetensors.torch

        sd: dict[str, torch.Tensor] = {}
        for f in safetensor_files:
            sd.update(safetensors.torch.load_file(str(f), device="cpu"))
        return sd

    # --- PyTorch .bin path --------------------------------------------------
    bin_files = sorted(model_dir.glob("*.bin"))
    if bin_files:
        sd = {}
        for f in bin_files:
            sd.update(torch.load(str(f), map_location="cpu", weights_only=True))
        return sd

    raise FileNotFoundError(
        f"No safetensors or .bin checkpoint files found in {model_dir}"
    )


def _interleave_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    is_bias: bool = False,
) -> torch.Tensor:
    """Interleave Q, K, V into Megatron's fused QKV layout for GQA.

    Megatron expects the fused QKV weight to be arranged per query-group::

        [Q_group0, K_group0, V_group0, Q_group1, K_group1, V_group1, ...]

    where ``Q_group_i`` contains ``heads_per_group`` Q heads and ``K_group_i``,
    ``V_group_i`` each contain 1 KV head.

    This matches veRL's ``_broadcast_tp_shard_tensor_qkv`` interleaving
    (``verl/models/mcore/loader.py:288-381``).

    Args:
        q: Q weight/bias from HF. Shape ``[num_heads * head_dim, hidden]``
           (weight) or ``[num_heads * head_dim]`` (bias).
        k: K weight/bias. Shape ``[num_kv_heads * head_dim, hidden]`` or
           ``[num_kv_heads * head_dim]``.
        v: V weight/bias (same shape as *k*).
        num_heads: Total number of query attention heads.
        num_kv_heads: Number of key/value heads (GQA groups).
        head_dim: Dimension per head.
        is_bias: If True, inputs are 1-D bias vectors instead of 2-D weights.

    Returns:
        Fused QKV tensor in Megatron's interleaved layout.
    """
    heads_per_group = num_heads // num_kv_heads

    if is_bias:
        q = q.view(num_heads, head_dim)
        k = k.view(num_kv_heads, head_dim)
        v = v.view(num_kv_heads, head_dim)
    else:
        q = q.view(num_heads, head_dim, -1)
        k = k.view(num_kv_heads, head_dim, -1)
        v = v.view(num_kv_heads, head_dim, -1)

    chunks = []
    for group_idx in range(num_kv_heads):
        q_start = group_idx * heads_per_group
        q_end = q_start + heads_per_group
        chunks.append(q[q_start:q_end])
        chunks.append(k[group_idx : group_idx + 1])
        chunks.append(v[group_idx : group_idx + 1])

    fused = torch.cat(chunks, dim=0)
    if is_bias:
        return fused.view(-1)
    return fused.view(-1, fused.size(-1))


def load_hf_weights_to_mcore(
    mcore_model: GPTModel,
    hf_model_path: str,
    tp_rank: int = 0,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    trust_remote_code: bool | None = None,
) -> None:
    """Load HuggingFace checkpoint weights into a Megatron GPTModel.

    Supports both TP=1 (full weights loaded directly) and TP>1 (weights
    are sharded per rank before loading).

    With PP>1, only the layers assigned to this PP stage are loaded.
    Stage assignment: layers are split evenly across PP ranks.
    Stage 0 also gets embedding (pre_process), last stage gets output_layer
    (post_process).

    The function handles:
      - QKV fusion with per-group interleaving (GQA-aware) + TP sharding
      - gate+up fusion for SwiGLU MLP + TP sharding
      - o_proj / down_proj column-parallel TP sharding
      - Embedding / output_layer vocab-parallel TP sharding
      - QKV bias (Qwen2) with TP sharding
      - q_norm / k_norm (Qwen3) — replicated, not sharded
      - LayerNorm fusion into Megatron's linear_qkv / linear_fc1

    Args:
        mcore_model: A Megatron ``GPTModel`` instance (on GPU).
        hf_model_path: Path or HuggingFace model name for the checkpoint.
        tp_rank: Tensor-parallel rank (0 for TP=1).
        tp_size: Tensor-parallel world size (1 for no TP).
        pp_rank: Pipeline-parallel rank (0 for PP=1).
        pp_size: Pipeline-parallel world size (1 for no PP).
    """
    hf_config = AutoConfig.from_pretrained(
        hf_model_path,
        trust_remote_code=resolve_trust_remote_code(
            trust_remote_code,
            context="Megatron HF weight loading",
        ),
    )
    num_heads = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
    head_dim = hf_config.hidden_size // num_heads
    # Some models (e.g. Qwen2.5) may override head_dim explicitly
    if hasattr(hf_config, "head_dim") and hf_config.head_dim is not None:
        head_dim = hf_config.head_dim
    hidden_size = hf_config.hidden_size
    intermediate_size = hf_config.intermediate_size

    hf_sd = _load_hf_state_dict(hf_model_path)

    # Collect model parameter names to detect fused vs separate layernorm
    model_param_names = set(name for name, _ in mcore_model.named_parameters())
    model_state_names = set(mcore_model.state_dict().keys())

    mcore_sd: dict[str, torch.Tensor] = {}

    # Pre-compute TP shard sizes
    heads_per_group = num_heads // num_kv_heads
    groups_per_tp = num_kv_heads // tp_size
    rows_per_qkv_tp = groups_per_tp * (heads_per_group + 2) * head_dim

    # PP layer assignment: split layers evenly across PP ranks
    num_layers = hf_config.num_hidden_layers
    if pp_size > 1:
        layers_per_stage = num_layers // pp_size
        pp_layer_start = pp_rank * layers_per_stage
        pp_layer_end = pp_layer_start + layers_per_stage
        # Last stage gets any remainder layers
        if pp_rank == pp_size - 1:
            pp_layer_end = num_layers
    else:
        pp_layer_start = 0
        pp_layer_end = num_layers

    for layer_idx in range(pp_layer_start, pp_layer_end):
        hf_pre = f"model.layers.{layer_idx}"
        # Megatron layer index is local to this PP stage
        local_layer_idx = layer_idx - pp_layer_start
        mc_pre = f"decoder.layers.{local_layer_idx}"

        # --- QKV fusion with interleaving -----------------------------------
        q = hf_sd[f"{hf_pre}.self_attn.q_proj.weight"]
        k = hf_sd[f"{hf_pre}.self_attn.k_proj.weight"]
        v = hf_sd[f"{hf_pre}.self_attn.v_proj.weight"]
        qkv = _interleave_qkv(q, k, v, num_heads, num_kv_heads, head_dim)
        if tp_size > 1:
            qkv = qkv[tp_rank * rows_per_qkv_tp : (tp_rank + 1) * rows_per_qkv_tp]
        mcore_sd[f"{mc_pre}.self_attention.linear_qkv.weight"] = qkv

        # QKV bias (Qwen2 has bias; Qwen3 / LLaMA do not)
        qb_key = f"{hf_pre}.self_attn.q_proj.bias"
        if qb_key in hf_sd:
            qb = hf_sd[qb_key]
            kb = hf_sd[f"{hf_pre}.self_attn.k_proj.bias"]
            vb = hf_sd[f"{hf_pre}.self_attn.v_proj.bias"]
            qkv_bias = _interleave_qkv(
                qb, kb, vb, num_heads, num_kv_heads, head_dim, is_bias=True,
            )
            if tp_size > 1:
                qkv_bias = qkv_bias[tp_rank * rows_per_qkv_tp : (tp_rank + 1) * rows_per_qkv_tp]
            mcore_sd[f"{mc_pre}.self_attention.linear_qkv.bias"] = qkv_bias

        # --- gate + up fusion (SwiGLU) -------------------------------------
        gate = hf_sd[f"{hf_pre}.mlp.gate_proj.weight"]
        up = hf_sd[f"{hf_pre}.mlp.up_proj.weight"]
        if tp_size > 1:
            # Megatron's gated ColumnParallelLinear stores the local gate shard
            # followed by the local up shard.  Shard each projection first;
            # slicing the concatenated [gate, up] tensor would give rank 0 all
            # gate rows and rank 1 all up rows for TP=2.
            shard = intermediate_size // tp_size
            gate = gate[tp_rank * shard : (tp_rank + 1) * shard]
            up = up[tp_rank * shard : (tp_rank + 1) * shard]
        gate_up = torch.cat([gate, up], dim=0)
        mcore_sd[f"{mc_pre}.mlp.linear_fc1.weight"] = gate_up

        # --- Simple renames (with TP sharding) ------------------------------
        o_proj = hf_sd[f"{hf_pre}.self_attn.o_proj.weight"]
        if tp_size > 1:
            shard = o_proj.shape[1] // tp_size  # num_heads * head_dim / TP
            o_proj = o_proj[:, tp_rank * shard : (tp_rank + 1) * shard]
        mcore_sd[f"{mc_pre}.self_attention.linear_proj.weight"] = o_proj

        down_proj = hf_sd[f"{hf_pre}.mlp.down_proj.weight"]
        if tp_size > 1:
            shard = intermediate_size // tp_size
            down_proj = down_proj[:, tp_rank * shard : (tp_rank + 1) * shard]
        mcore_sd[f"{mc_pre}.mlp.linear_fc2.weight"] = down_proj

        # --- Layer norms ----------------------------------------------------
        # With Transformer Engine, layernorms are fused into the linear layers:
        #   self_attention.linear_qkv.layer_norm_weight
        #   mlp.linear_fc1.layer_norm_weight
        # Without TE (local spec), they are separate:
        #   input_layernorm.weight
        #   pre_mlp_layernorm.weight
        # We detect which layout the model uses and map accordingly.
        fused_ln_key = f"{mc_pre}.self_attention.linear_qkv.layer_norm_weight"
        separate_ln_key = f"{mc_pre}.input_layernorm.weight"
        if fused_ln_key in model_param_names:
            mcore_sd[fused_ln_key] = hf_sd[f"{hf_pre}.input_layernorm.weight"]
            mcore_sd[f"{mc_pre}.mlp.linear_fc1.layer_norm_weight"] = (
                hf_sd[f"{hf_pre}.post_attention_layernorm.weight"]
            )
        else:
            mcore_sd[separate_ln_key] = hf_sd[f"{hf_pre}.input_layernorm.weight"]
            mcore_sd[f"{mc_pre}.pre_mlp_layernorm.weight"] = (
                hf_sd[f"{hf_pre}.post_attention_layernorm.weight"]
            )

        # --- q_norm / k_norm (Qwen3) ---------------------------------------
        qn_key = f"{hf_pre}.self_attn.q_norm.weight"
        if qn_key in hf_sd:
            mcore_sd[f"{mc_pre}.self_attention.q_layernorm.weight"] = hf_sd[qn_key]
            mcore_sd[f"{mc_pre}.self_attention.k_layernorm.weight"] = (
                hf_sd[f"{hf_pre}.self_attn.k_norm.weight"]
            )

    # --- Embeddings, final layernorm, LM head -------------------------------
    # With PP: first stage (pre_process) gets embedding, last stage
    # (post_process) gets output_layer + final layernorm.
    is_first_stage = pp_rank == 0
    is_last_stage = pp_rank == pp_size - 1

    if is_first_stage:
        embed = hf_sd["model.embed_tokens.weight"]
        if tp_size > 1:
            vocab_shard = embed.shape[0] // tp_size
            embed = embed[tp_rank * vocab_shard : (tp_rank + 1) * vocab_shard]
        mcore_sd["embedding.word_embeddings.weight"] = embed

    if is_last_stage:
        # Some tied-embedding Megatron layouts allocate no output-layer
        # parameter on the single-stage rank; in that case the output layer
        # reads directly from ``embedding.word_embeddings.weight``.
        if "output_layer.weight" in model_state_names:
            lm_head = hf_sd.get("lm_head.weight", hf_sd["model.embed_tokens.weight"])
            if tp_size > 1:
                vocab_shard = lm_head.shape[0] // tp_size
                lm_head = lm_head[tp_rank * vocab_shard : (tp_rank + 1) * vocab_shard]
            mcore_sd["output_layer.weight"] = lm_head

        mcore_sd["decoder.final_layernorm.weight"] = hf_sd["model.norm.weight"]

    # --- Load into model ----------------------------------------------------
    _load_state_dict_into_mcore(mcore_model, mcore_sd)

    log.info(
        "Loaded %d HF tensors into Megatron GPTModel (%d layers)",
        len(mcore_sd),
        hf_config.num_hidden_layers,
    )


def _load_state_dict_into_mcore(
    model: GPTModel,
    converted_sd: dict[str, torch.Tensor],
) -> None:
    """Load converted weights into a Megatron GPTModel, handling dtype + device.

    Megatron GPTModel param names do NOT have a ``module.`` prefix (unlike
    models wrapped with DDP).  This function matches converted_sd keys
    against the model's ``state_dict().keys()`` directly.
    """
    model_sd = model.state_dict()

    # Verify all converted keys exist in model
    missing_in_model = set(converted_sd.keys()) - set(model_sd.keys())
    missing_in_converted = set(model_sd.keys()) - set(converted_sd.keys())

    if missing_in_model:
        raise RuntimeError(
            f"Keys in converted state dict not found in Megatron model: "
            f"{sorted(missing_in_model)[:10]}... ({len(missing_in_model)} total)"
        )
    if missing_in_converted:
        log.warning(
            "Keys in Megatron model not found in converted state dict "
            "(will keep random init): %s",
            sorted(missing_in_converted)[:10],
        )

    # Copy weights with dtype/device matching
    for name, param in model.named_parameters():
        if name in converted_sd:
            src = converted_sd[name]
            if src.shape != param.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"converted {src.shape} vs model {param.shape}"
                )
            param.data.copy_(src.to(dtype=param.dtype, device=param.device))

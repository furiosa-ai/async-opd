"""FP8 quantization utilities for rollout vLLM workers.

Provides blockwise FP8 (e4m3) quantization with 128x128 blocks, vLLM monkey-patches
to preserve weight_loader attributes during process_weights_after_loading, and a
critical patch to gpu_worker.Worker.update_weights that routes bf16 NCCL tensors
through quantization instead of naive param.copy_().

Ported from verl-opd (verl/utils/vllm/vllm_fp8_utils.py) with adaptations for
our pipeline's IPC architecture. The patch_update_weights_for_fp8() function is
novel — verl-opd uses collective_rpc to access model_runner directly, which our
pipeline cannot do across the EngineCore subprocess boundary.
"""

from dataclasses import dataclass, field
from unittest.mock import patch

import torch
import vllm

try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
except ImportError as e:
    raise ImportError("FP8 quantization not available — requires vllm with "
                      "LinearBase and FusedMoE layers") from e


# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

FP8_BLOCK_QUANT_KWARGS = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


# ------------------------------------------------------------------ #
#  FP8 detection state + helpers                                      #
# ------------------------------------------------------------------ #

@dataclass
class _FP8State:
    """Cache of fp8 parameter names to avoid repeated module traversal."""
    seen_params: set = field(default_factory=set)
    fp8_param_names: set = field(default_factory=set)


_fp8_state = _FP8State()


def get_module_from_param_name(model, name: str):
    """Traverse model hierarchy to find the module owning a parameter."""
    path_parts = name.split(".")
    module_path = path_parts[:-1]
    # Map unfused names to fused names (e.g. q_proj -> qkv_proj)
    packed_modules_mapping = model.packed_modules_mapping
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names_list in packed_modules_mapping.items()
        for original_name in original_names_list
    }
    if module_path[-1] in reversed_mapping:
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module = model
    try:
        for part in module_path:
            if isinstance(current_module, FusedMoE):
                return current_module
            elif isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError) as e:
        print(f"WARNING: Could not find module for parameter '{name}': {e}")
    return current_module


def is_fp8_weight(name, model):
    """Check if a named parameter is an FP8-quantized weight."""
    if name not in _fp8_state.seen_params:
        _fp8_state.seen_params.add(name)
        if name.endswith("weight"):
            module = get_module_from_param_name(model, name)
            if (isinstance(module, LinearBase) and module.weight.dtype == torch.float8_e4m3fn) or (
                isinstance(module, FusedMoE)
                and module.w13_weight.dtype == torch.float8_e4m3fn
                and module.w2_weight.dtype == torch.float8_e4m3fn
            ):
                _fp8_state.fp8_param_names.add(name)
    return name in _fp8_state.fp8_param_names


# ------------------------------------------------------------------ #
#  Blockwise FP8 quantization                                        #
# ------------------------------------------------------------------ #

def scaled_fp8_blockwise(data_hp, weight_block_size):
    """Quantize a 2D bf16/fp32 tensor to FP8 e4m3 with blockwise scaling.

    Args:
        data_hp: High-precision 2D weight tensor [out_features, in_features]
        weight_block_size: [block_rows, block_cols], typically [128, 128]

    Returns:
        (fp8_data, scales) where fp8_data has dtype float8_e4m3fn and
        scales is fp32 with shape [n_block_rows, n_block_cols].
    """
    assert len(data_hp.shape) == 2, "Only 2D tensors supported"
    block_size0, block_size1 = weight_block_size
    assert data_hp.shape[0] % block_size0 == 0, (
        f"dim0 {data_hp.shape[0]} must be divisible by {block_size0}")
    assert data_hp.shape[1] % block_size1 == 0, (
        f"dim1 {data_hp.shape[1]} must be divisible by {block_size1}")
    assert block_size0 == block_size1

    max_dtype = torch.finfo(torch.float8_e4m3fn).max
    original_shape = data_hp.shape
    blk_m = data_hp.shape[0] // block_size0
    blk_n = data_hp.shape[1] // block_size1

    # Reshape to [blk_m, block_size0, blk_n, block_size1]
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)
    # Permute to [blk_m, blk_n, block_size0, block_size1] and flatten blocks
    data_hp = data_hp.permute(0, 2, 1, 3).to(torch.float32).contiguous().flatten(start_dim=2)

    # Per-block max absolute value
    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)
    scale_fp = max_dtype / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)
    descale_fp = torch.reciprocal(scale_fp)

    # Scale, clamp, cast to fp8
    data_lp = torch.clamp(data_hp * scale_fp, min=-max_dtype, max=max_dtype)
    fp_data = data_lp.to(torch.float8_e4m3fn)

    # Reshape back to original [out_features, in_features]
    fp_data = fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
    fp_data = fp_data.permute(0, 2, 1, 3).reshape(original_shape)

    return fp_data, descale_fp


# ------------------------------------------------------------------ #
#  Weight quantization                                                #
# ------------------------------------------------------------------ #

def scaled_fp8_per_tensor(data_hp):
    """Quantize a tensor to FP8 e4m3 with a single per-tensor scale.

    Args:
        data_hp: High-precision tensor (any shape)

    Returns:
        (fp8_data, scale) where scale is a scalar fp32 tensor (dequant scale).
    """
    max_dtype = torch.finfo(torch.float8_e4m3fn).max
    data_f32 = data_hp.to(torch.float32)
    max_abs = data_f32.abs().max()
    scale_fp = max_dtype / max_abs if max_abs > 0 else torch.tensor(1.0)
    data_lp = torch.clamp(data_f32 * scale_fp, min=-max_dtype, max=max_dtype)
    fp8_data = data_lp.to(torch.float8_e4m3fn)
    descale = torch.tensor(1.0 / scale_fp.item(), dtype=torch.float32)
    return fp8_data, descale


def quant_weights(weights, model, quant_config, dtype=torch.bfloat16):
    """Quantize bf16 weights to fp8 + scales for fp8 parameters.

    Non-fp8 weights are passed through unchanged.
    Supports both per-tensor (weight_block_size=None) and blockwise quantization.

    Args:
        weights: list of (name, tensor) pairs
        model: vLLM model instance
        quant_config: vLLM Fp8Config with optional weight_block_size
        dtype: dtype to cast weights to before quantization

    Returns:
        list of (name, tensor) pairs with fp8 weights + scale tensors added
    """
    block_size = getattr(quant_config, "weight_block_size", None)
    weights_quantized = []
    for k, v in weights:
        if not is_fp8_weight(k, model):
            weights_quantized.append((k, v))
            continue
        if block_size is not None:
            # Blockwise quantization
            param_lp, param_scale = scaled_fp8_blockwise(
                v.to(dtype), weight_block_size=block_size)
            param_scale = param_scale.squeeze(-1)
            weights_quantized.append((k, param_lp))
            weights_quantized.append((k + "_scale_inv", param_scale))
        else:
            # Per-tensor quantization
            param_lp, param_scale = scaled_fp8_per_tensor(v.to(dtype))
            weights_quantized.append((k, param_lp))
            weights_quantized.append((k + "_scale", param_scale.reshape(1)))
    return weights_quantized


# ------------------------------------------------------------------ #
#  vLLM process_weights_after_loading patches                         #
# ------------------------------------------------------------------ #

def _process_weights_after_loading_linear(self, layer):
    """Patched process_weights_after_loading for Fp8LinearMethod (vllm 0.16).

    Fixes online blockwise quantization: vLLM 0.16's blockwise path assumes
    pre-quantized fp8 checkpoints and produces -inf scales when given bf16
    weights. We detect this case (block_quant + weight is NOT fp8) and apply
    our own scaled_fp8_blockwise quantization before calling the original.

    Also preserves weight_loader attributes for weight refit support.
    """
    # Fix: online blockwise quantization from bf16
    # vLLM 0.16 block_quant path assumes is_checkpoint_fp8_serialized=True,
    # but hf_overrides forces this even for bf16 checkpoints. The weight data
    # arrives as bf16 (cast to fp8 container without scaling), so weight_scale_inv
    # is uninitialized (zeros) → process_fp8_weight_block_strategy → -inf scales.
    # Fix: detect bf16 data in fp8 container and quantize properly first.
    if (self.block_quant
            and hasattr(layer, "weight")
            and layer.weight.dtype == torch.float8_e4m3fn
            and hasattr(layer, "weight_scale_inv")):
        # Check if scales are invalid: uninitialized (zeros), contain inf/nan,
        # or are negative (scales are dequantization factors, must be positive)
        scale_data = layer.weight_scale_inv.data
        if scale_data.abs().sum() == 0 or not scale_data.isfinite().all() or (scale_data <= 0).any():
            # Online quantization: weight was cast to fp8 without proper scaling.
            # The fp8 weight data is fine (bf16 values truncated to fp8 precision),
            # but scales are garbage. Re-quantize from the fp8 values to get
            # properly scaled fp8 + correct blockwise scales.
            bf16_approx = layer.weight.data.float()
            fp8_data, scales = scaled_fp8_blockwise(
                bf16_approx, weight_block_size=list(self.weight_block_size))
            layer.weight.data.copy_(fp8_data)
            layer.weight_scale_inv.data.copy_(scales.squeeze(-1))
            print(f"[FP8] Fixed blockwise FP8 scales for {next((n for n, p in layer._parameters.items() if p is layer.weight), '?')} (online quantization)", flush=True)

    # Save weight_loader refs before original replaces parameters
    saved_loaders = {}
    for name, param in layer.named_parameters(prefix="", recurse=False):
        if hasattr(param, "weight_loader"):
            saved_loaders[name] = (param.weight_loader, type(param))

    # Run the original — handles block_quant, marlin, per-tensor, everything
    _original_process_weights_after_loading_linear(self, layer)

    # Restore weight_loader and add subclass_type to new parameters
    for name, param in layer.named_parameters(prefix="", recurse=False):
        if name in saved_loaders:
            loader, orig_type = saved_loaders[name]
            if not hasattr(param, "weight_loader"):
                param.weight_loader = loader
            if not hasattr(param, "subclass_type"):
                param.subclass_type = orig_type


def _process_weights_after_loading_moe(self, layer):
    """Patched process_weights_after_loading for Fp8MoEMethod (vllm 0.16).

    Wraps the original: runs it first, then restores weight_loader attributes
    and adds subclass_type for weight refit support.
    """
    saved_loaders = {}
    for name, param in layer.named_parameters(prefix="", recurse=False):
        if hasattr(param, "weight_loader"):
            saved_loaders[name] = (param.weight_loader, type(param))

    _original_process_weights_after_loading_moe(self, layer)

    for name, param in layer.named_parameters(prefix="", recurse=False):
        if name in saved_loaders:
            loader, orig_type = saved_loaders[name]
            if not hasattr(param, "weight_loader"):
                param.weight_loader = loader
            if not hasattr(param, "subclass_type"):
                param.subclass_type = orig_type


def check_fp8_hardware_support():
    """Check that at least one visible GPU supports native FP8 (sm_89+).

    Raises RuntimeError if no FP8-capable GPU is found. Without FP8 hardware,
    vLLM falls back to Marlin (int32 packed weights) which is incompatible
    with our bf16 NCCL → fp8 quantization weight sync path.
    """
    import torch
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        if major * 10 + minor >= 89:
            return  # At least one FP8-capable GPU
    raise RuntimeError(
        "FP8 rollout quantization requires GPUs with compute capability >= 8.9 "
        "(Ada Lovelace, Hopper, or Blackwell). Current GPUs only support "
        "Marlin FP8 fallback which is incompatible with NCCL weight sync. "
        "Remove 'quantization: fp8' from config or use FP8-capable hardware.")


_fp8_patches_applied = False
_original_process_weights_after_loading_linear = None
_original_process_weights_after_loading_moe = None


def apply_vllm_fp8_patches():
    """Monkey-patch vLLM's process_weights_after_loading to preserve weight_loader.

    Must be called BEFORE LLM/AsyncLLM construction so the patch takes effect
    during model loading in the EngineCore subprocess. Idempotent.
    """
    import vllm
    assert vllm.__version__.startswith("0.16"), f"FP8 patches written for vLLM 0.16, got {vllm.__version__}"
    global _fp8_patches_applied, _original_process_weights_after_loading_linear
    global _original_process_weights_after_loading_moe
    if _fp8_patches_applied:
        return
    _fp8_patches_applied = True
    print("[FP8] Applying vLLM FP8 patches for blockwise quantization", flush=True)
    # Save originals for fallback in non-block-quant paths
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod
    _original_process_weights_after_loading_linear = Fp8LinearMethod.process_weights_after_loading
    _original_process_weights_after_loading_moe = Fp8MoEMethod.process_weights_after_loading
    patcher1 = patch(
        "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading",
        _process_weights_after_loading_linear,
    )
    patcher1.start()
    patcher2 = patch(
        "vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading",
        _process_weights_after_loading_moe,
    )
    patcher2.start()


# ------------------------------------------------------------------ #
#  gpu_worker.Worker.update_weights monkey-patch (Option C)           #
# ------------------------------------------------------------------ #

_original_update_weights = None  # saved reference to original method


def patch_update_weights_for_fp8():
    """Monkey-patch gpu_worker.Worker.update_weights to handle FP8 models.

    For non-fp8 models: delegates to original update_weights (no change).
    For fp8 models: wraps the receive flow to:
      1. Monkey-patch param classes ONCE before receive (O(N))
      2. Use a thin per-param callback that quantizes + loads (O(1) per param)
      3. Restore param classes ONCE after receive (O(N))

    Must be called BEFORE LLM/AsyncLLM construction so the patch takes
    effect inside the EngineCore subprocess.
    """
    import vllm
    assert vllm.__version__.startswith("0.16"), f"FP8 patches written for vLLM 0.16, got {vllm.__version__}"
    from vllm.v1.worker.gpu_worker import Worker

    global _original_update_weights
    _original_update_weights = Worker.update_weights

    def _fp8_update_weights(self_worker, update_info: dict) -> None:
        if self_worker.weight_transfer_engine is None:
            raise RuntimeError("Weight transfer not configured.")

        typed_update_info = self_worker.weight_transfer_engine.parse_update_info(update_info)

        # Check if this is an fp8 model
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config
        quant_config = self_worker.model_runner.vllm_config.quant_config
        is_fp8 = isinstance(quant_config, Fp8Config)

        if not is_fp8 or typed_update_info.is_checkpoint_format:
            return _original_update_weights(self_worker, update_info)

        model = self_worker.model_runner.model
        params_dict = dict(model.named_parameters())

        def load_weights_fp8(weights):
            """Quantize received bf16 weights to fp8 + scales, copy directly into params."""
            quantized_pairs = quant_weights(weights, model, quant_config)
            for name, tensor in quantized_pairs:
                if name in params_dict:
                    params_dict[name].data.copy_(tensor)
                else:
                    print(f"WARNING: FP8 load: param {name} not found in model")

        self_worker.weight_transfer_engine.receive_weights(
            typed_update_info, load_weights=load_weights_fp8)

    Worker.update_weights = _fp8_update_weights


# ------------------------------------------------------------------ #
#  128-alignment startup check                                        #
# ------------------------------------------------------------------ #

def check_fp8_alignment(model):
    """Verify all linear layer weights have dimensions divisible by 128.

    FP8 blockwise quantization with [128, 128] blocks requires this.
    Called via apply_model() after vLLM model load.

    Raises ValueError with details if any dimension is misaligned.
    """
    misaligned = []
    for name, param in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if param.dtype == torch.float8_e4m3fn and len(param.shape) == 2:
            d0, d1 = param.shape
            if d0 % 128 != 0 or d1 % 128 != 0:
                misaligned.append(f"  {name}: shape={list(param.shape)} "
                                  f"(d0 % 128 = {d0 % 128}, d1 % 128 = {d1 % 128})")
    if misaligned:
        raise ValueError(
            "FP8 blockwise quantization requires all linear weight dimensions "
            "to be divisible by 128. Misaligned parameters:\n"
            + "\n".join(misaligned))
    n_fp8 = sum(1 for _, p in model.named_parameters()
                if p.dtype == torch.float8_e4m3fn and len(p.shape) == 2)
    print(f"[FP8] Alignment check passed: {n_fp8} fp8 weight tensors, all 128-aligned", flush=True)

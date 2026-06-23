"""Weight merge map for trainer → vLLM parameter mapping."""


def build_weight_merge_map(trainer_weights_info, vllm_params_info):
    """Build a mapping from trainer state_dict → vLLM internal params.

    vLLM merges some HF checkpoint weights:
      q_proj + k_proj + v_proj → qkv_proj  (cat dim=0)
      gate_proj + up_proj → gate_up_proj    (cat dim=0)

    Returns a list of (vllm_name, sources) where sources is either:
      - [(trainer_name,)] for direct 1:1 mapping
      - [(name1,), (name2,), ...] for concatenation (in order)
    """
    trainer_names = {name for name, _, _ in trainer_weights_info}
    merge_map = []

    # Skip FP8 scale/quantization params — these are created during
    # quantization on the receive side, not transferred via NCCL
    _skip_suffixes = ("_scale", "_scale_inv", "_zero_point", "_amax")

    for vllm_name, vllm_shape, vllm_dtype in vllm_params_info:
        if any(vllm_name.endswith(s) for s in _skip_suffixes):
            continue
        if "qkv_proj" in vllm_name:
            if vllm_name in trainer_names:
                # Trainer already has fused qkv_proj (native Megatron model)
                merge_map.append((vllm_name, [vllm_name]))
            else:
                # q_proj + k_proj + v_proj → qkv_proj
                parts = vllm_name.split("qkv_proj")
                base = parts[0]   # e.g. "model.layers.0.self_attn."
                suffix = parts[1]  # e.g. ".weight" or ".bias"
                sources = [
                    base + "q_proj" + suffix,
                    base + "k_proj" + suffix,
                    base + "v_proj" + suffix,
                ]
                merge_map.append((vllm_name, sources))
        elif "gate_up_proj" in vllm_name:
            if vllm_name in trainer_names:
                # Trainer already has fused gate_up_proj (native Megatron model)
                merge_map.append((vllm_name, [vllm_name]))
            else:
                # gate_proj + up_proj → gate_up_proj
                parts = vllm_name.split("gate_up_proj")
                base = parts[0]
                suffix = parts[1]
                sources = [
                    base + "gate_proj" + suffix,
                    base + "up_proj" + suffix,
                ]
                merge_map.append((vllm_name, sources))
        else:
            # Direct 1:1 mapping — vLLM name should exist in trainer
            if vllm_name in trainer_names:
                merge_map.append((vllm_name, [vllm_name]))
            else:
                print(f"[Pipeline] WARNING: vLLM param {vllm_name} "
                      f"not found in trainer state_dict", flush=True)

    return merge_map

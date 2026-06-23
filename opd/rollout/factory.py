"""Rollout backend factory — maps backend name to worker_main function and capabilities."""


def _get_streaming_worker_main():
    """Lazy import of streaming worker main to avoid top-level vLLM import."""
    from opd.rollout.vllm.streaming import vllm_streaming_rollout_worker_main
    return vllm_streaming_rollout_worker_main


def get_rollout_backend(name: str) -> dict:
    """Return worker_main function and capability flags for a rollout backend."""
    if name == "vllm":
        from opd.rollout.vllm.batch import vllm_batch_rollout_worker_main
        return {
            "worker_main": vllm_batch_rollout_worker_main,
            "streaming_worker_main": _get_streaming_worker_main,
            "supports_streaming": True,
            "supports_nccl": True,
        }
    elif name == "hf":
        from opd.rollout.hf import hf_rollout_worker_main
        return {
            "worker_main": hf_rollout_worker_main,
            "supports_streaming": False,
            "supports_nccl": False,
        }
    elif name == "dummy":
        from opd.rollout.dummy import dummy_rollout_worker_main
        return {
            "worker_main": dummy_rollout_worker_main,
            "supports_streaming": False,
            "supports_nccl": False,
        }
    else:
        raise ValueError(f"Unknown rollout backend: '{name}'. Available: vllm, hf, dummy")

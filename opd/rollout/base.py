"""Engine-agnostic base class for rollout workers.

This module has ZERO vllm/transformers/sglang imports.
All engine-specific code lives in backend subpackages (e.g., opd.rollout.vllm).
"""

import os
import sys

from opd.launch_specs import RolloutLaunchSpec


class BaseRolloutWorker:
    """Common rollout worker interface for @ray.remote wrapping.

    Subclasses implement __init__ (model load) and run() (command loop).
    """

    def __init__(self, config: dict):
        self.launch_spec = config if isinstance(config, RolloutLaunchSpec) else None
        if isinstance(config, RolloutLaunchSpec):
            config = config.merged_config()

        self.model_path = config["model_path"]
        self.gpu_ids = config["gpu_ids"]
        self.tp_size = config["tp_size"]
        self.gpu_memory_utilization = config["gpu_memory_utilization"]
        self.max_response_length = config["max_response_length"]
        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.top_k = config["top_k"]
        self.max_num_seqs = config["max_num_seqs"]
        self.worker_id = config["worker_id"]
        self.use_weight_transfer = config["use_weight_transfer"]
        self.max_model_len = config["max_model_len"]
        self.max_num_batched_tokens = config["max_num_batched_tokens"]
        self.enforce_eager = config.get("enforce_eager", True)
        self.dtype = config["dtype"]
        self.quantization = config["quantization"]
        self.native_lora = config["native_lora"]
        self.lora_rank = config["lora_rank"]
        self.lora_cfg = config["lora_cfg"]
        self.block_size = config["block_size"]
        self.max_logprobs = config["max_logprobs"]
        self.trust_remote_code = config.get("trust_remote_code")
        self._lora_request = None  # set after first LoRA sync

    @staticmethod
    def _setup_env(gpu_ids, extra_env=None):
        """Set CUDA_VISIBLE_DEVICES and PATH for subprocess."""
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        # else: Ray manages CUDA_VISIBLE_DEVICES via num_gpus allocation
        bin_dir = os.path.dirname(sys.executable)
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + ":" + os.environ.get("PATH", "")
        for k, v in (extra_env or {}).items():
            os.environ[k] = v

    def run(self, cmd_queue, result_queue, prompt_queue=None):
        raise NotImplementedError

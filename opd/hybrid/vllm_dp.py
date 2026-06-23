"""vLLM data-parallel adapter for fused_hybrid_sync trainer ranks."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any

from opd.hybrid.vllm_spmd import FusedHybridVLLMSPMDAdapter
from opd.hybrid.weight_export import BucketedInprocessWeightUpdater, build_weight_checksum_plan
from opd.rollout.vllm.utils import get_params_info_fn
from opd.utils.config import resolve_trust_remote_code


_DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
)


@contextmanager
def _local_vllm_env():
    """Hide torchrun/FSDP env while constructing a TP=1 local vLLM engine."""
    saved = {key: os.environ.get(key) for key in _DISTRIBUTED_ENV_KEYS}
    saved_v1 = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    try:
        for key in _DISTRIBUTED_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if saved_v1 is None:
            os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
        else:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = saved_v1


class FusedHybridVLLMDPAdapter(FusedHybridVLLMSPMDAdapter):
    """One local TP=1 vLLM replica colocated with each FSDP rank."""

    rollout_parallelism = "data_parallel"

    def __init__(
        self,
        *,
        model_path: str,
        dp_size: int,
        max_response_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_num_seqs: int,
        max_model_len: int | None,
        max_num_batched_tokens: int | None,
        gpu_memory_utilization: float,
        dtype: str,
        seed: int,
        sleep_level: int,
        bucket_mb: int,
        weight_update_backend: str,
        debug_full_state_sync: bool,
        verify_weight_checksum: bool,
        trainer_weights_info: list | None,
        rank: int,
        world_size: int,
        rank_info: dict[str, Any],
        max_logprobs: int = 1,
    ):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.dp_size = int(dp_size)
        self.tp_size = 1
        self.max_response_length = int(max_response_length)
        self.max_logprobs = int(max_logprobs)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.sleep_level = int(sleep_level)
        self.rollout_version = 0
        self._awake = True
        self._weights_awake = True
        self._fast_mc_installed = False

        from vllm import LLM

        kwargs = dict(
            tensor_parallel_size=1,
            trust_remote_code=resolve_trust_remote_code(context="fused hybrid DP vLLM model loading"),
            enforce_eager=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enable_sleep_mode=True,
            disable_custom_all_reduce=True,
            seed=seed + int(rank),
        )
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        with _local_vllm_env():
            self.llm = LLM(model_path, **kwargs)
        self.info = self._probe_info()
        self.vllm_params_info = self._get_vllm_params_info()
        checksum_plan = None
        if verify_weight_checksum:
            checksum_plan = build_weight_checksum_plan(
                trainer_weights_info or [],
                self.vllm_params_info,
            )
        self.updater = BucketedInprocessWeightUpdater(
            bucket_mb=bucket_mb,
            debug_full_state_sync=debug_full_state_sync,
            backend=weight_update_backend,
            verify_checksum=verify_weight_checksum,
            checksum_plan=checksum_plan,
            vllm_params_info=self.vllm_params_info,
            tp_size=1,
        )

    def _get_vllm_params_info(self) -> list:
        # Keep the method local for import-cycle clarity; behavior matches SPMD.
        result = self.llm.apply_model(get_params_info_fn)
        if (
            isinstance(result, list)
            and result
            and isinstance(result[0], tuple)
            and len(result[0]) == 3
            and isinstance(result[0][0], str)
        ):
            return result
        if isinstance(result, list):
            result = result[0] if result else []
        return result or []

    def _add_parallelism_metrics(self, metrics: dict) -> dict:
        super()._add_parallelism_metrics(metrics)
        metrics["fused_hybrid_weight_update_replicated_dp"] = True
        return metrics

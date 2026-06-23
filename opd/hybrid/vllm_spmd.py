"""vLLM SPMD adapter used by the fused_hybrid_sync trainer ranks."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import torch

from opd.hybrid.weight_export import (
    BucketedInprocessWeightUpdater,
    build_weight_checksum_plan,
    cuda_memory_snapshot,
)
from opd.rollout.vllm.fast_mc import (
    align_single_pass_capture,
    install_fast_mc_add_request_patch_rpc,
    install_fast_mc_patch,
    install_fast_mc_patch_rpc,
    pop_single_pass_mc_from_engine_or_core,
    pop_single_pass_mc_from_llm,
)
from opd.rollout.vllm.utils import (
    extract_student_logprobs as _extract_student_logprobs,
    extract_student_topk_support as _extract_student_topk_support,
    get_params_info_fn,
)
from opd.utils.config import resolve_trust_remote_code


class FusedHybridVLLMSPMDAdapter:
    """Rank-local handle to a vLLM external-launcher shard.

    Every FSDP rank constructs this adapter and participates in the same vLLM
    SPMD calls. Rank 0 returns public generation/telemetry results to the
    coordinator; all ranks execute the underlying collectives.
    """

    rollout_parallelism = "spmd_tp"

    def __init__(
        self,
        *,
        model_path: str,
        tp_size: int,
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
        self.tp_size = int(tp_size)
        self.dp_size = 1
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

        # external_launcher reads torchrun-like metadata in some vLLM versions.
        # FSDP already initialized the same rank/world; keep these env vars
        # aligned rather than creating a separate process topology.
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", "0")
        if rank_info.get("fsdp_master_addr"):
            os.environ.setdefault("MASTER_ADDR", str(rank_info["fsdp_master_addr"]))
        if rank_info.get("fsdp_master_port"):
            os.environ.setdefault("MASTER_PORT", str(rank_info["fsdp_master_port"]))
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        from vllm import LLM

        kwargs = dict(
            tensor_parallel_size=self.tp_size,
            distributed_executor_backend="external_launcher",
            trust_remote_code=resolve_trust_remote_code(context="fused hybrid SPMD vLLM model loading"),
            enforce_eager=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enable_sleep_mode=True,
            disable_custom_all_reduce=True,
            seed=seed,
        )
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens
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
            tp_size=self.tp_size,
        )

    def _probe_info(self) -> dict:
        pc = self.llm.llm_engine.vllm_config.parallel_config
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "rollout_parallelism": self.rollout_parallelism,
            "dp_size": int(self.dp_size),
            "distributed_executor_backend": pc.distributed_executor_backend,
            "tensor_parallel_size": pc.tensor_parallel_size,
            "pipeline_parallel_size": pc.pipeline_parallel_size,
            "vllm_world_size": pc.world_size,
        }

    def _get_vllm_params_info(self) -> list:
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

    def sleep(self, reason: str = "") -> dict:
        if not self._awake:
            return {"status": "slept", "already": True, "reason": reason}
        t0 = time.monotonic()
        self.llm.sleep(level=self.sleep_level)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._awake = False
        self._weights_awake = False
        return {
            "status": "slept",
            "reason": reason,
            "sleep_level": self.sleep_level,
            "sleep_seconds": time.monotonic() - t0,
            "memory": cuda_memory_snapshot(f"after_sleep:{reason}"),
        }

    def wake_weights(self) -> dict:
        if self._weights_awake:
            return {"status": "awake_weights", "already": True}
        t0 = time.monotonic()
        self.llm.wake_up(tags=["weights"])
        self._weights_awake = True
        return {
            "status": "awake_weights",
            "wake_seconds": time.monotonic() - t0,
            "memory": cuda_memory_snapshot("after_wake_weights"),
        }

    def wake_kv_cache(self) -> dict:
        if self._awake:
            return {"status": "awake", "already": True}
        t0 = time.monotonic()
        # If weights were not separately woken, full wake also wakes them.
        tags = ["kv_cache"] if self._weights_awake else None
        self.llm.wake_up(tags=tags)
        self._awake = True
        self._weights_awake = True
        return {
            "status": "awake",
            "wake_seconds": time.monotonic() - t0,
            "memory": cuda_memory_snapshot("after_wake_kv_cache"),
        }

    def refresh_weights(self, state_dict: dict[str, torch.Tensor], actor_version: int) -> dict:
        self.wake_weights()
        telemetry = self.updater.update(self.llm, state_dict)
        self.rollout_version = int(actor_version)
        metrics = telemetry.to_metrics()
        self._add_parallelism_metrics(metrics)
        metrics["fused_hybrid_actor_version"] = int(actor_version)
        metrics["fused_hybrid_rollout_version"] = int(self.rollout_version)
        return metrics

    def refresh_named_tensors(
        self,
        named_tensors,
        actor_version: int,
        *,
        full_state_materialized: bool = False,
    ) -> dict:
        self.wake_weights()
        telemetry = self.updater.update_from_named_tensors(
            self.llm,
            named_tensors,
            full_state_materialized=full_state_materialized,
        )
        self.rollout_version = int(actor_version)
        metrics = telemetry.to_metrics()
        self._add_parallelism_metrics(metrics)
        metrics["fused_hybrid_actor_version"] = int(actor_version)
        metrics["fused_hybrid_rollout_version"] = int(self.rollout_version)
        return metrics

    def _add_parallelism_metrics(self, metrics: dict) -> dict:
        metrics["fused_hybrid_rollout_parallelism"] = self.rollout_parallelism
        metrics["fused_hybrid_rollout_dp_size"] = int(self.dp_size)
        metrics["fused_hybrid_rollout_tp_size"] = int(self.tp_size)
        return metrics

    def rollout_mode(self) -> dict:
        self.wake_kv_cache()
        return {
            "status": "rollout_ready",
            "rollout_version": int(self.rollout_version),
            "memory": cuda_memory_snapshot("rollout_mode"),
            "rollout_parallelism": self.rollout_parallelism,
            "rollout_tp_size": int(self.tp_size),
            "rollout_dp_size": int(self.dp_size),
        }

    def generate(
        self,
        batch: dict,
        *,
        return_logprobs: bool = False,
        response_topk_k: int = 0,
        max_response_length: int | None = None,
        mc_n_total_samples: int = 0,
    ) -> dict:
        from vllm import SamplingParams

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        max_response_length = int(max_response_length or self.max_response_length)
        mc_n_total_samples = int(mc_n_total_samples or 0)

        if int(input_ids.size(0)) == 0:
            return self._empty_generate_result(
                input_ids,
                max_response_length,
                return_logprobs=return_logprobs,
                response_topk_k=response_topk_k,
                mc_n_total_samples=mc_n_total_samples,
            )

        prompt_lengths: list[int] = []
        prompt_token_lists: list[list[int]] = []
        for i in range(input_ids.size(0)):
            mask = attention_mask[i].bool()
            ids = input_ids[i][mask].detach().cpu().tolist()
            prompt_token_lists.append(ids)
            prompt_lengths.append(len(ids))

        mc_request_ids = None
        if mc_n_total_samples > 0:
            self._ensure_fast_mc_patch()
            mc_request_ids = [f"mc-{uuid.uuid4().hex}" for _ in prompt_token_lists]
            prompts = [
                {
                    "prompt_token_ids": ids,
                    "mc_request_id": request_id,
                    "mc_n_total_samples": mc_n_total_samples,
                }
                for ids, request_id in zip(prompt_token_lists, mc_request_ids, strict=False)
            ]
        else:
            prompts = [{"prompt_token_ids": ids} for ids in prompt_token_lists]

        sp = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k if self.top_k > 0 else -1,
            max_tokens=max_response_length,
            detokenize=False,
            logprobs=max(1 if return_logprobs else 0, response_topk_k) or None,
            ignore_eos=batch.get("ignore_eos", False),
        )
        outputs = self.llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)

        batch_size = len(outputs)
        max_prompt_len = input_ids.size(1)
        total_len = max_prompt_len + max_response_length
        full_ids = torch.zeros(batch_size, total_len, dtype=torch.long)
        full_mask = torch.zeros(batch_size, total_len, dtype=torch.bool)
        responses = torch.zeros(batch_size, max_response_length, dtype=torch.long)
        response_lengths = []
        student_logprobs = (
            torch.zeros(batch_size, max_response_length, dtype=torch.float32)
            if return_logprobs else None
        )
        query_indices_response = [] if response_topk_k > 0 else None
        query_logprobs_response = [] if response_topk_k > 0 else None
        mc_query_indices_response = [] if mc_n_total_samples > 0 else None
        mc_query_old_logprobs_response = [] if mc_n_total_samples > 0 else None

        for i, out in enumerate(outputs):
            resp_ids = list(out.outputs[0].token_ids)
            resp_len = min(len(resp_ids), max_response_length)
            response_lengths.append(resp_len)
            p_len = prompt_lengths[i]
            pad_len = max_prompt_len - p_len
            full_ids[i, :max_prompt_len] = input_ids[i].detach().cpu()
            full_mask[i, pad_len:max_prompt_len] = True
            for j in range(resp_len):
                full_ids[i, max_prompt_len + j] = resp_ids[j]
                full_mask[i, max_prompt_len + j] = True
                responses[i, j] = resp_ids[j]
            if return_logprobs and student_logprobs is not None:
                _extract_student_logprobs(out, student_logprobs, i, resp_len)
            if response_topk_k > 0:
                topk_idx, topk_logps = _extract_student_topk_support(
                    out, resp_len=resp_len, topk_k=response_topk_k
                )
                query_indices_response.append(topk_idx)
                query_logprobs_response.append(topk_logps)

        full_token_lists = []
        for i, out in enumerate(outputs):
            p_len = prompt_lengths[i]
            pad_len = max_prompt_len - p_len
            prompt = input_ids[i][pad_len:].detach().cpu().tolist()
            resp = list(out.outputs[0].token_ids)[:max_response_length]
            full_token_lists.append(prompt + resp)

        if mc_n_total_samples > 0:
            captured = self._pop_single_pass_mc_sequences(mc_request_ids, mc_n_total_samples)
            aligned = []
            for entry, out in zip(captured, outputs, strict=False):
                resp_len = min(len(out.outputs[0].token_ids), max_response_length)
                finish_reason = getattr(out.outputs[0], "finish_reason", None)
                aligned.append(
                    align_single_pass_capture(
                        entry,
                        resp_len,
                        finish_reason,
                        expected_token_ids=list(out.outputs[0].token_ids)[:resp_len],
                    )
                )
            mc_query_indices_response = [entry["mc_query_indices_response"] for entry in aligned]
            mc_query_old_logprobs_response = [
                entry["mc_query_old_logprobs_response"] for entry in aligned
            ]

        result = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "responses": responses,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "response_lengths": torch.tensor(response_lengths, dtype=torch.long),
            "full_token_lists": full_token_lists,
        }
        if student_logprobs is not None:
            result["student_logprobs"] = student_logprobs
        if query_indices_response is not None:
            result["query_indices_response"] = query_indices_response
            result["query_logprobs_response"] = query_logprobs_response
        if mc_query_indices_response is not None:
            result["mc_query_indices_response"] = mc_query_indices_response
            result["mc_query_old_logprobs_response"] = mc_query_old_logprobs_response
        return result

    def _empty_generate_result(
        self,
        input_ids: torch.Tensor,
        max_response_length: int,
        *,
        return_logprobs: bool,
        response_topk_k: int,
        mc_n_total_samples: int,
    ) -> dict:
        batch_size = 0
        max_prompt_len = int(input_ids.size(1))
        total_len = max_prompt_len + int(max_response_length)
        result = {
            "input_ids": torch.zeros(batch_size, total_len, dtype=torch.long),
            "attention_mask": torch.zeros(batch_size, total_len, dtype=torch.bool),
            "responses": torch.zeros(batch_size, int(max_response_length), dtype=torch.long),
            "prompt_lengths": torch.zeros(batch_size, dtype=torch.long),
            "response_lengths": torch.zeros(batch_size, dtype=torch.long),
            "full_token_lists": [],
        }
        if return_logprobs:
            result["student_logprobs"] = torch.zeros(
                batch_size, int(max_response_length), dtype=torch.float32
            )
        if response_topk_k > 0:
            result["query_indices_response"] = []
            result["query_logprobs_response"] = []
        if mc_n_total_samples > 0:
            result["mc_query_indices_response"] = []
            result["mc_query_old_logprobs_response"] = []
        return result

    def _ensure_fast_mc_patch(self) -> None:
        if self._fast_mc_installed:
            return
        if self.tp_size > 1:
            engine_core = self.llm.llm_engine.engine_core
            results = install_fast_mc_patch_rpc(
                engine_core,
                fallback_topk_k=max(1, self.max_logprobs),
            )
            self._fast_mc_installed = bool(results) and all(results)
            if self._fast_mc_installed:
                install_fast_mc_add_request_patch_rpc(self.llm, engine_core)
        else:
            self._fast_mc_installed = install_fast_mc_patch(
                self.llm,
                fallback_topk_k=max(1, self.max_logprobs),
            )
        if not self._fast_mc_installed:
            raise RuntimeError("Failed to install fused hybrid rollout fast MC patch")

    def _pop_single_pass_mc_sequences(self, capture_request_ids, n_total_samples):
        if self.tp_size > 1:
            return pop_single_pass_mc_from_engine_or_core(
                self.llm.llm_engine.engine_core,
                capture_request_ids,
                n_total_samples,
            )
        return pop_single_pass_mc_from_llm(
            self.llm,
            capture_request_ids,
            n_total_samples,
        )

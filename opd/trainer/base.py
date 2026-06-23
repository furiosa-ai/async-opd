"""BaseBackend ABC — shared infrastructure for FSDP and Megatron backends.

Extracts truly identical methods from both backends so subclasses only
implement backend-specific logic (model loading, distributed training,
weight gather, etc.).
"""

import math
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod

import torch

from opd.loss.kl import chunked_lm_head_gather
from opd.launch_specs import (
    TrainerLaunchSpec,
    algorithm_mode,
)
from opd.trainer.config import build_kl_config_from_algorithm_payload
from opd.utils.trace import timer


# ===================== Module-level helpers =====================

GLOBAL_MINI_PLAN_METADATA_KEYS = (
    "_use_global_mini_plan",
    "_mini_slices",
    "_global_mini_slices",
    "_rank_source_ranges",
    "_global_batch_size",
    "_configured_global_mini_batch_size",
    "_common_micro_counts",
)

def vllm_trainer_send(iterator, wt_group, packed=False):
    """Send weights via vLLM NCCL — compatible with both 0.16 and 0.17 APIs."""
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLWeightTransferEngine,
    )
    try:
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs,
        )
        args = NCCLTrainerSendWeightsArgs(group=wt_group, packed=packed)
        NCCLWeightTransferEngine.trainer_send_weights(iterator, args)
    except ImportError:
        # vLLM 0.16 API: group is a separate kwarg
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator, group=wt_group, packed=packed
        )


def build_lr_scheduler(optimizer, optim_cfg, total_steps):
    """Build LR scheduler with optional warmup + cosine/linear decay.

    Config keys (all optional):
        lr_warmup_steps_ratio: float  — fraction of total_steps for warmup (default: 0)
        lr_decay_style: str           — "cosine" or "linear" (default: None = constant)
        min_lr: float                 — minimum LR at end of decay (default: 0)
    """
    warmup_ratio = optim_cfg.get("lr_warmup_steps_ratio", 0.0)
    decay_style = optim_cfg.get("lr_decay_style", None)
    min_lr = float(optim_cfg.get("min_lr", 0.0))
    base_lr = float(optim_cfg["lr"])

    warmup_steps = int(total_steps * warmup_ratio)

    if min_lr >= base_lr:
        min_lr = 0.0
    lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step):
        # Warmup phase
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        # Decay phase
        if decay_style is None or total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        if decay_style == "cosine":
            return lr_ratio + (1.0 - lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif decay_style == "linear":
            return lr_ratio + (1.0 - lr_ratio) * (1.0 - progress)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===================== BaseBackend ABC =====================

class BaseBackend(ABC):
    """Abstract base for training backends (FSDP, Megatron, etc.).

    Holds common fields and methods that are identical across backends.
    Subclasses must implement the abstract methods/properties below.
    """

    def __init__(self, config: dict, rank_info: dict | None = None):
        existing_launch_spec = getattr(self, "launch_spec", None)
        if existing_launch_spec is not None:
            self.launch_spec = existing_launch_spec
        else:
            self.launch_spec = config if isinstance(config, TrainerLaunchSpec) else None
        algorithm_launch = None
        if isinstance(config, TrainerLaunchSpec):
            launch_spec = config
            algorithm_launch = launch_spec.static.algorithm
            config = launch_spec.merged_config()
            rank_info = launch_spec.rank_payload()
        else:
            algorithm_launch = config["algorithm"]

        self.rank = rank_info["fsdp_rank"]
        self.world_size = rank_info["fsdp_world_size"]
        self._model_path = config["model_path"]
        self._dtype_str = config["dtype"]
        self.micro_batch_size = config["micro_batch_size"]
        self.max_response_length = config["max_response_length"]

        # Algorithm / KL config
        self.algorithm_launch = algorithm_launch
        self._algo_mode = algorithm_mode(self.algorithm_launch)  # opd/sft/grpo — consumed for completeness
        self.kl_chunk_size = config["kl_chunk_size"]
        self.kl_config = build_kl_config_from_algorithm_payload(self.algorithm_launch)
        self.mini_batch_size = config["mini_batch_size"]
        self.use_sequence_packing = config["use_sequence_packing"]
        self.total_steps = config["total_steps"]
        self.optim_cfg = config["optim"]
        self._wt_group = None

        # Loss mode: "kl" (default OPD), "sft" (supervised fine-tuning),
        # "grpo" (group relative policy optimization).
        # Subclasses (SFTTrainer, GRPOTrainer) set these before calling super().
        # MegatronTrainer reads them to dispatch loss in forward_step.
        self.loss_mode = getattr(self, 'loss_mode', 'kl')

        # Default: collective weights_info same as regular (FSDP uses HF names)
        self._collective_weights_info = None  # set by subclass if different

        # Composition: mode trainer reference (OPDTrainer, SFTTrainer, etc.)
        # Set by trainer.run() → backend.run(trainer=self)
        self._trainer = None
        self.teacher_artifact_queue = None
        if self.launch_spec is not None:
            self.teacher_artifact_queue = getattr(
                self.launch_spec.runtime, "teacher_artifact_queue", None
            )
        elif isinstance(config, dict):
            self.teacher_artifact_queue = config.get("teacher_artifact_queue")
        self.teacher_artifact_mode = config.get("teacher_artifact_mode", "legacy")
        self.teacher_model_path = config.get("teacher_model_path") or self._model_path
        self._trust_remote_code = config.get("trust_remote_code")
        self.teacher_hidden_dtype = config.get("teacher_hidden_dtype", "bfloat16")
        self.teacher_hidden_semantics = config.get("teacher_hidden_semantics", "lm_head_input")
        self.teacher_hidden_recompute_materialization = config.get("teacher_hidden_recompute_materialization", "lazy")
        self.teacher_artifact_buffer = None
        self._teacher_recompute_head = None
        if self.teacher_artifact_queue is not None:
            from opd.trainer.teacher_artifact_buffer import TrainerTeacherArtifactBuffer
            self.teacher_artifact_buffer = TrainerTeacherArtifactBuffer(
                self.teacher_artifact_queue,
                max_batches=4,
            )

        # Dispatch table for command loop — subclasses extend in __init__
        self._command_handlers = {
            "train": self._handle_train,
            "init_weight_transfer": self._handle_init_weight_transfer,
            "sync_weights": self._handle_sync_weights,
            "sync_weights_collective": self._handle_sync_weights_collective,
            "get_weights_info": self._handle_get_weights_info,
            "get_collective_weights_info": self._handle_get_collective_weights_info,
            "get_clean_state_dict": self._handle_get_clean_state_dict,
            "save_checkpoint": self._handle_save_checkpoint,
            "load_checkpoint": self._handle_load_checkpoint,
            "shutdown_weight_transfer": self._handle_shutdown_weight_transfer,
            "hybrid_init_rollout": self._handle_hybrid_init_rollout,
            "hybrid_rollout_mode": self._handle_hybrid_rollout_mode,
            "hybrid_generate": self._handle_hybrid_generate,
            "hybrid_generate_cached": self._handle_hybrid_generate_cached,
            "hybrid_start_teacher_cached": self._handle_hybrid_start_teacher_cached,
            "hybrid_resolve_teacher_cached": self._handle_hybrid_resolve_teacher_cached,
            "hybrid_train_cached": self._handle_hybrid_train_cached,
            "hybrid_release_rollout": self._handle_hybrid_release_rollout,
            "hybrid_refresh_weights": self._handle_hybrid_refresh_weights,
            "hybrid_prepare_train": self._handle_hybrid_prepare_train,
            "memory_snapshot": self._handle_memory_snapshot,
        }

        self._hybrid_generation_cache = {}
        self._hybrid_teacher_cache = {}
        self._hybrid_teacher_futures = {}
        self._hybrid_teacher_clients = {}

    def _build_lr_scheduler(self, total_steps: int):
        """Build a backend-compatible LR scheduler.

        Most backends expose a normal ``torch.optim.Optimizer``. Specialized
        backends can override this when their optimizer is not a PyTorch
        optimizer subclass.
        """
        return build_lr_scheduler(self.optimizer, self.optim_cfg, total_steps)

    # -------------------- DP rank (overridden by Megatron) --------------------

    @property
    def dp_rank(self) -> int:
        """Data-parallel rank for batch splitting. Defaults to self.rank.

        Megatron overrides this to use mpu.get_data_parallel_rank() so that
        TP ranks within the same DP group see the same batch shard.
        """
        return self.rank

    @property
    def dp_world_size(self) -> int:
        """Data-parallel world size for batch splitting. Defaults to self.world_size.

        Megatron overrides this to use mpu.get_data_parallel_world_size().
        """
        return self.world_size

    # -------------------- Abstract interface --------------------

    @property
    @abstractmethod
    def use_distributed(self) -> bool:
        """Whether multi-rank distributed training is active."""
        ...

    @abstractmethod
    def _get_log_prefix(self) -> str:
        """Return log prefix, e.g. 'Trainer-FSDP' or 'Trainer-Megatron'."""
        ...

    @abstractmethod
    def _build_weights_info(self):
        """Build (name, shape, dtype) list for NCCL weight sync."""
        ...

    @abstractmethod
    def _train_step_impl(self, batch) -> dict:
        """Execute one training step on the given batch. Returns metrics dict."""
        ...

    def _run_train_step(self, batch, loss_fn):
        """Execute one training step with a pluggable loss function.

        This is the composition interface: mode trainers provide loss_fn,
        backends provide the training loop (forward/backward/optimizer).

        Args:
            batch: dict of tensors, already rank-split and truncated.
                Must include "input_ids", "attention_mask", "response_mask".
                May include mode-specific keys that get passed through to loss_fn.
            loss_fn: callable(logits, micro_batch_dict) -> (loss, n_tokens, extras_dict)
                logits: [B, S, V] tensor from model forward
                micro_batch_dict: dict of tensors for this micro-batch (on device)
                Returns: (loss_tensor, n_tokens_int, extras_dict)

        Returns:
            dict with metrics: kl_loss (avg loss), n_tokens, grad_norm, lr, plus extras.
        """
        raise NotImplementedError("Subclass must implement _run_train_step")

    # -------------------- Global mini-batch planning helpers --------------------

    @staticmethod
    def _balanced_range(total: int, index: int, parts: int) -> tuple[int, int]:
        """Return this part's half-open range for an even-as-possible split."""
        if parts <= 0:
            raise ValueError(f"parts must be positive, got {parts}")
        if index < 0 or index >= parts:
            raise ValueError(f"index {index} outside [0, {parts})")
        if total < 0:
            raise ValueError(f"total must be non-negative, got {total}")

        base = total // parts
        rem = total % parts
        start = index * base + min(index, rem)
        length = base + (1 if index < rem else 0)
        return start, start + length

    @staticmethod
    def _split_span_for_micro_steps(start: int, end: int,
                                    n_micro: int) -> list[tuple[int, int]]:
        """Split a local mini span using the historical ceil-chunk rule."""
        if n_micro <= 0:
            raise ValueError(f"n_micro must be positive, got {n_micro}")
        length = end - start
        if length < 0:
            raise ValueError(f"invalid span [{start}, {end})")
        actual_mbs = (length + n_micro - 1) // n_micro
        return [
            (start + idx * actual_mbs, min(start + (idx + 1) * actual_mbs, end))
            for idx in range(n_micro)
        ]

    @staticmethod
    def _global_mini_ranges(global_bs: int, mini_batch_size: int) -> list[tuple[int, int]]:
        """Return global optimizer-step ranges preserving configured mini size."""
        if global_bs < 0:
            raise ValueError(f"global_bs must be non-negative, got {global_bs}")
        if global_bs == 0:
            return []
        if mini_batch_size <= 0 or mini_batch_size >= global_bs:
            return [(0, global_bs)]
        return [
            (start, min(start + mini_batch_size, global_bs))
            for start in range(0, global_bs, mini_batch_size)
        ]

    @staticmethod
    def _common_micro_count(global_mini_len: int, world_size: int,
                            micro_batch_size: int) -> int:
        """Return the deterministic microstep count shared by all ranks."""
        if global_mini_len <= 0:
            return 0
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if micro_batch_size <= 0:
            raise ValueError(
                f"micro_batch_size must be positive, got {micro_batch_size}")
        max_local_shard = (global_mini_len + world_size - 1) // world_size
        return max(1, (max_local_shard + micro_batch_size - 1) // micro_batch_size)

    @staticmethod
    def _validate_global_mini_plan_feasible(
        global_bs: int,
        mini_batch_size: int,
        world_size: int,
        micro_batch_size: int,
    ) -> None:
        """Fail fast if global-mini-first cannot run uniform FSDP collectives."""
        for mini_idx, (gs, ge) in enumerate(
            BaseBackend._global_mini_ranges(global_bs, mini_batch_size)
        ):
            mini_len = ge - gs
            common_n_micro = BaseBackend._common_micro_count(
                mini_len, world_size, micro_batch_size
            )
            for candidate_rank in range(world_size):
                shard_s, shard_e = BaseBackend._balanced_range(
                    mini_len, candidate_rank, world_size
                )
                local_len = shard_e - shard_s
                if local_len <= 0:
                    raise RuntimeError(
                        "global-mini-first cannot safely shard mini "
                        f"{mini_idx} [{gs}, {ge}) across world_size={world_size}: "
                        f"rank {candidate_rank} would receive zero samples"
                    )
                if local_len < common_n_micro:
                    raise RuntimeError(
                        "global-mini-first cannot create non-empty microsteps "
                        f"for mini {mini_idx}: rank {candidate_rank} "
                        f"local_len={local_len}, common_n_micro={common_n_micro}, "
                        f"micro_batch_size={micro_batch_size}"
                    )

    def _supports_global_mini_plan(self) -> bool:
        """Whether this backend can consume explicit global-mini metadata.

        V1 is intentionally FSDP-scoped. Tests that bind this method onto a
        lightweight namespace can opt in via ``_force_global_mini_plan_support``.
        """
        if bool(getattr(self, "_force_global_mini_plan_support", False)):
            return True
        return type(self).__name__ == "FSDPBackend"

    def _should_use_global_mini_plan(self, global_bs: int,
                                     mini_batch_size: int,
                                     world_size: int) -> bool:
        """Gate global-mini-first to OPD/FSDP cases the rank-first path mishandles."""
        if not self._supports_global_mini_plan():
            return False
        if world_size <= 1 or mini_batch_size <= 0:
            return False

        per_rank_bs = global_bs // world_size
        per_rank_mini_bs = max(1, mini_batch_size // world_size)
        rank_first_unsafe = (
            global_bs % world_size != 0
            or mini_batch_size % world_size != 0
            or (
                per_rank_mini_bs > 0
                and per_rank_mini_bs < per_rank_bs
                and per_rank_bs % per_rank_mini_bs != 0
            )
        )
        return rank_first_unsafe

    def _build_rank_mini_plan(self, global_bs: int, mini_batch_size: int,
                              rank: int, world_size: int,
                              micro_batch_size: int | None = None) -> dict:
        """Build source and local slices for global-mini-first sharding.

        The plan first forms configured global mini-batches, then shards each
        mini-batch across ranks with balanced remainder distribution. Each rank's
        shards are concatenated locally; ``local_mini_slices`` records the
        resulting offsets for the train loop.
        """
        if micro_batch_size is not None:
            self._validate_global_mini_plan_feasible(
                global_bs, mini_batch_size, world_size, micro_batch_size
            )

        global_mini_slices = self._global_mini_ranges(global_bs, mini_batch_size)
        rank_source_ranges = []
        local_mini_slices = []
        common_micro_counts = []
        local_cursor = 0

        for gs, ge in global_mini_slices:
            mini_len = ge - gs
            rel_s, rel_e = self._balanced_range(mini_len, rank, world_size)
            src = (gs + rel_s, gs + rel_e)
            rank_source_ranges.append(src)

            local_len = src[1] - src[0]
            local_mini_slices.append((local_cursor, local_cursor + local_len))
            local_cursor += local_len

            if micro_batch_size is not None:
                common_micro_counts.append(
                    self._common_micro_count(mini_len, world_size, micro_batch_size)
                )

        return {
            "global_mini_slices": global_mini_slices,
            "rank_source_ranges": rank_source_ranges,
            "local_mini_slices": local_mini_slices,
            "common_micro_counts": common_micro_counts,
            "local_batch_size": local_cursor,
        }

    @abstractmethod
    def _get_state_dict_for_sync(self):
        """Return state_dict for weight sync, or None if model is directly accessible."""
        ...

    @abstractmethod
    def _gather_full_state_dict(self):
        """Gather full model state dict for checkpointing (all ranks participate if distributed)."""
        ...

    @abstractmethod
    def _gather_full_optim_state_dict(self):
        """Gather full optimizer state dict for checkpointing (all ranks participate if distributed)."""
        ...

    # -------------------- Command loop --------------------

    def run(self, cmd_queue, result_queue, trainer=None) -> None:
        """Command loop — blocks until shutdown.

        Args:
            trainer: Optional mode trainer (OPDTrainer, SFTTrainer, etc.).
                When provided, _handle_train delegates to trainer.train_step()
                instead of self._train_step_impl().
        """
        if trainer is not None:
            self._trainer = trainer
            # Register any extra command handlers from the trainer
            if hasattr(trainer, 'command_handlers'):
                self._command_handlers.update(trainer.command_handlers())
        while True:
            if self.rank == 0:
                cmd = cmd_queue.get()
                t_recv = time.monotonic()
            else:
                cmd = None
                t_recv = None

            if self.use_distributed:
                cmd_name_list = [cmd[0] if cmd else None]
                torch.distributed.broadcast_object_list(cmd_name_list, src=0)
                cmd_name = cmd_name_list[0]
                if self.rank == 0:
                    cmd_name = cmd[0]
            else:
                cmd_name = cmd[0]

            if cmd_name == "shutdown":
                self._wait_async_save()  # ensure last checkpoint is on disk
                if self.rank == 0:
                    print(f"[{self._get_log_prefix()}] shutting down", flush=True)
                break

            handler = self._command_handlers.get(cmd_name)
            if handler:
                result = handler(cmd, t_recv, result_queue)
                if result == "_break":
                    break
            else:
                if self.rank == 0:
                    print(f"[{self._get_log_prefix()}] unknown command: {cmd_name}", flush=True)

        if self.use_distributed:
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                except Exception:
                    pass  # TCPStore may already be torn down by exiting peers

    # -------------------- Shared command handlers --------------------

    def _handle_train(self, cmd, t_recv, result_queue):
        from opd.data.batch_utils import broadcast_batch as _broadcast_batch

        rank = self.rank
        direct_artifact_metrics = None
        if rank == 0:
            payload = cmd[1]
            if payload.get("_direct_teacher_artifacts"):
                try:
                    payload = self._materialize_direct_teacher_artifacts(payload)
                    direct_artifact_metrics = payload.pop(
                        "_direct_teacher_artifact_metrics", None
                    )
                    status = {"status": "ok"}
                except Exception as e:
                    status = {"status": "error", "reason": str(e)}
            else:
                status = {"status": "ok"}
        else:
            payload = None
            status = None

        if self.use_distributed:
            status_list = [status]
            torch.distributed.broadcast_object_list(status_list, src=0)
            status = status_list[0]
            if status.get("status") != "ok":
                raise RuntimeError(status.get("reason", "trainer direct batch error"))
            batch = _broadcast_batch(payload if rank == 0 else None,
                                     rank, self.world_size, self.device)
        else:
            if status.get("status") != "ok":
                raise RuntimeError(status.get("reason", "trainer direct batch error"))
            batch = payload
        self._execute_train_batch(
            batch,
            t_recv,
            result_queue,
            direct_artifact_metrics=direct_artifact_metrics,
        )

    def _execute_train_batch(
        self,
        batch,
        t_recv,
        result_queue,
        *,
        direct_artifact_metrics=None,
    ):
        """Run one already-materialized train batch.

        The legacy train handler calls this after broadcasting a global batch.
        Fused-hybrid cached DP calls it with rank-local, pre-sharded batches.
        """

        rank = self.rank
        # Rebuild scheduler on first train call to account for mini-batch splitting.
        if self._scheduler_needs_rebuild:
            global_bs = int(batch.get("_global_batch_size", batch["input_ids"].size(0)))
            dp_ws = getattr(self, 'dp_world_size', self.world_size)
            if self._should_use_global_mini_plan(
                global_bs, self.mini_batch_size, dp_ws
            ):
                n_mini_est = len(
                    self._global_mini_ranges(global_bs, self.mini_batch_size)
                )
            else:
                per_rank_bs = global_bs // dp_ws if dp_ws > 1 else global_bs
                # mini_batch_size is global — compute per-rank, then n_mini
                per_rank_mini_bs = max(1, self.mini_batch_size // dp_ws) if self.mini_batch_size > 0 else 0
                if per_rank_mini_bs > 0 and per_rank_mini_bs < per_rank_bs:
                    n_mini_est = per_rank_bs // per_rank_mini_bs
                else:
                    n_mini_est = 1
            if n_mini_est > 1:
                adj_total = self.total_steps * n_mini_est
                self.scheduler = self._build_lr_scheduler(adj_total)
                if rank == 0:
                    print(f"[{self._get_log_prefix()}] scheduler adjusted: {self.total_steps} train calls x "
                          f"{n_mini_est} mini-batches = {adj_total} optimizer steps", flush=True)
            self._scheduler_needs_rebuild = False
        send_mono = batch.pop("_send_mono", None) if rank == 0 else None
        with timer() as t:
            if self._trainer is not None:
                metrics = self._trainer.train_step(batch, self)
            else:
                metrics = self._train_step_impl(batch)
        if rank == 0:
            metrics["train_seconds"] = t["elapsed"]
            t["send_mono"] = send_mono
            t["queue_recv"] = t_recv
            metrics["timing"] = t
            if direct_artifact_metrics is not None:
                metrics["teacher_artifacts"] = direct_artifact_metrics
            result_queue.put({"metrics": metrics})

    def _materialize_direct_teacher_artifacts(self, payload: dict) -> dict:
        if self.teacher_artifact_buffer is None:
            raise RuntimeError("direct teacher artifact buffer is not configured")
        buffer_id = int(payload["teacher_buffer_id"])
        expected = int(payload["expected_samples"])
        timeout_s = float(payload.get("teacher_artifact_timeout_s", 300.0))
        self.teacher_artifact_buffer.wait_complete(
            buffer_id,
            expected_count=expected,
            timeout_s=timeout_s,
        )
        teacher_output = self.teacher_artifact_buffer.assemble_canonical(
            buffer_id,
            payload,
            teacher_recompute_head=self._get_teacher_recompute_head(),
        )
        from opd.worker.proxy import QueueTrainerProxy
        batch = QueueTrainerProxy._build_train_batch(payload, teacher_output)
        for key in (
            "logical_batch_id", "gen_weight_version", "expected_samples",
            "_send_mono",
        ):
            if key in payload:
                batch[key] = payload[key]
        metrics = self.teacher_artifact_buffer.last_metrics
        metrics["coordinator_teacher_artifact_bytes"] = 0
        metrics["trainer_teacher_artifact_recv_bytes"] = metrics.get("recv_bytes", 0)
        batch["_direct_teacher_artifact_metrics"] = metrics
        self.teacher_artifact_buffer.pop(buffer_id)
        return batch

    def _get_teacher_recompute_head(self):
        if self.teacher_artifact_mode != "hidden_recompute":
            return None
        if self._teacher_recompute_head is None:
            from opd.trainer.teacher_recompute import TeacherRecomputeHead
            self._teacher_recompute_head = TeacherRecomputeHead(
                model_path=self.teacher_model_path,
                device=self.device,
                dtype_name=self.teacher_hidden_dtype,
                hidden_semantics=self.teacher_hidden_semantics,
                chunk_size=getattr(self, "kl_chunk_size", 128),
                trust_remote_code=self._trust_remote_code,
                materialization=self.teacher_hidden_recompute_materialization,
            )
        return self._teacher_recompute_head

    def _handle_sync_weights(self, cmd, t_recv, result_queue):
        merge_map = cmd[1] if self.rank == 0 else None
        t0 = time.time()

        state_dict = self._get_state_dict_for_sync()
        # When state_dict is None and distributed, _vllm_send_merged_weights
        # calls _get_clean_state_dict which may involve collectives (e.g.,
        # all_gather for TP-sharded Megatron). All ranks must participate
        # in the gather, even though only rank 0 sends to vLLM.
        if state_dict is None and getattr(self, 'use_distributed', False):
            state_dict = self._get_clean_state_dict()
        if self.rank == 0:
            if self._wt_group is not None:
                self._vllm_send_merged_weights(merge_map, self._wt_group, state_dict=state_dict)
                dt = time.time() - t0
                print(f"[{self._get_log_prefix()}] sent merged weights in {dt:.2f}s", flush=True)
                result_queue.put({"status": "synced_nccl",
                                  "sync_seconds": dt})
            else:
                print(f"[{self._get_log_prefix()}] WARNING: weight transfer not initialized",
                      flush=True)
                result_queue.put({"status": "error", "sync_seconds": 0})

    def _handle_sync_weights_collective(self, cmd, t_recv, result_queue):
        """Broadcast weights via Ray collective (Ray backend only).

        Uses de-fused HF names (q_proj, k_proj, v_proj, gate_proj, up_proj)
        because vLLM's model.load_weights() expects standard checkpoint format.
        """
        group_name = cmd[1] if self.rank == 0 else None
        t0 = time.time()

        # All ranks participate in gather (TP all-gather + PP broadcast)
        state_dict = self._get_clean_state_dict()

        if self.rank == 0:
            # Send fused names (matching model.named_parameters on rollout)
            # Rollout does direct param copy, not load_weights
            from opd.worker.ray_weight_sync import _trainer_broadcast_weights
            _trainer_broadcast_weights(
                state_dict, self.weights_info, group_name=group_name,
            )
            dt = time.time() - t0
            print(f"[{self._get_log_prefix()}] broadcast weights in {dt:.2f}s",
                  flush=True)
            result_queue.put({"status": "synced_collective",
                              "sync_seconds": dt})

    def _handle_save_checkpoint(self, cmd, t_recv, result_queue):
        save_info = cmd[1] if self.rank == 0 else None
        if self.use_distributed:
            # Broadcast save_info to all ranks so they can participate in gather
            save_info_list = [save_info]
            torch.distributed.broadcast_object_list(save_info_list, src=0)
            save_info = save_info_list[0]
        sd = self._gather_full_state_dict()
        optim_sd = self._gather_full_optim_state_dict()
        if self.rank == 0:
            t_start = time.monotonic()
            self._save_checkpoint(
                checkpoint_dir=save_info["checkpoint_dir"],
                step=save_info["step"], state_dict=sd,
                save_optimizer=save_info.get("save_optimizer", True),
                optim_state_dict=optim_sd,
            )
            t_end = time.monotonic()
            result_queue.put({"status": "saved",
                              "checkpoint_dir": save_info["checkpoint_dir"],
                              "mono_start": t_start, "mono_end": t_end})
        self._post_save_cleanup()

    def _handle_init_weight_transfer(self, cmd, t_recv, result_queue):
        if self.rank == 0:
            init_info = cmd[1]
            # Ensure correct CUDA device before NCCL init — trainer_init()
            # calls torch.cuda.current_device() internally.
            torch.cuda.set_device(self.device)
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLWeightTransferEngine,
            )
            self._wt_group = NCCLWeightTransferEngine.trainer_init(init_info)
            print(f"[{self._get_log_prefix()}] vLLM weight transfer engine ready", flush=True)
            result_queue.put({"status": "ok"})

    def _handle_get_weights_info(self, cmd, t_recv, result_queue):
        if self.rank == 0:
            result_queue.put({"weights_info": self.weights_info})

    def _handle_get_collective_weights_info(self, cmd, t_recv, result_queue):
        if self.rank == 0:
            wi = self._collective_weights_info or self.weights_info
            result_queue.put({"weights_info": wi})

    def _handle_get_clean_state_dict(self, cmd, t_recv, result_queue):
        """Gather weights and return the state dict (for Ray collective broadcast).

        All ranks participate in the gather (TP all-gather + PP broadcast).
        Only rank 0 returns the result.
        """
        sd = self._get_clean_state_dict()
        if self.rank == 0:
            # Move to CPU to avoid GPU memory pressure during broadcast
            sd_cpu = {k: v.cpu() for k, v in sd.items()}
            result_queue.put({"state_dict": sd_cpu})

    def _handle_load_checkpoint(self, cmd, t_recv, result_queue):
        load_info = cmd[1] if self.rank == 0 else None
        if self.use_distributed:
            load_info_list = [load_info]
            torch.distributed.broadcast_object_list(load_info_list, src=0)
            load_info = load_info_list[0]
        step = self._load_checkpoint(load_info["checkpoint_dir"])
        if self.rank == 0:
            result_queue.put({"status": "loaded", "step": step})


    def _ensure_hybrid_adapter(self):
        raise RuntimeError(
            f"{self._get_log_prefix()} does not implement fused_hybrid_sync adapter"
        )

    def _handle_hybrid_init_rollout(self, cmd, t_recv, result_queue):
        adapter = self._ensure_hybrid_adapter()
        if self.rank == 0:
            result_queue.put({"status": "hybrid_rollout_ready", "info": adapter.info})

    def _handle_hybrid_rollout_mode(self, cmd, t_recv, result_queue):
        actor_version = int(cmd[1].get("actor_version", 0)) if self.rank == 0 else None
        if self.use_distributed:
            version_list = [actor_version]
            torch.distributed.broadcast_object_list(version_list, src=0)
            actor_version = int(version_list[0])
        adapter = self._ensure_hybrid_adapter()
        metrics = {}
        if adapter.rollout_version != actor_version:
            metrics = self._refresh_hybrid_adapter_weights(actor_version)
            metrics = self._finalize_hybrid_checksum_metrics(metrics)
        status = adapter.rollout_mode()
        if self.rank == 0:
            result_queue.put({"status": "rollout_ready", "metrics": metrics, **status})

    def _handle_hybrid_generate(self, cmd, t_recv, result_queue):
        from opd.data.batch_utils import broadcast_batch as _broadcast_batch
        from opd.hybrid.dp_merge import (
            merge_indexed_generation_payloads,
            shard_indices_for_rank,
            slice_batch_for_indices,
        )

        if self.rank == 0:
            payload = cmd[1]
            batch = payload["batch"]
            options = payload.get("options", {})
        else:
            batch = None
            options = None
        if self.use_distributed:
            options_list = [options]
            torch.distributed.broadcast_object_list(options_list, src=0)
            options = options_list[0]
            batch = _broadcast_batch(batch if self.rank == 0 else None,
                                     self.rank, self.world_size, self.device)
        options = options or {}
        adapter = self._ensure_hybrid_adapter()
        rollout_parallelism = getattr(adapter, "rollout_parallelism", "spmd_tp")
        mc_n_total_samples = int(options.get("mc_n_total_samples", 0) or 0)
        if rollout_parallelism == "data_parallel":
            batch_size = int(batch["input_ids"].size(0))
            local_indices = shard_indices_for_rank(batch_size, self.rank, self.world_size)
            local_batch = slice_batch_for_indices(batch, local_indices)
            t0 = time.monotonic()
            local_result = adapter.generate(
                local_batch,
                return_logprobs=bool(options.get("return_logprobs", False)),
                response_topk_k=int(options.get("response_topk_k", 0)),
                max_response_length=options.get("max_response_length"),
                mc_n_total_samples=mc_n_total_samples,
            )
            timing = {
                "generate_seconds": time.monotonic() - t0,
                "worker_id": int(self.rank),
                "rank": int(self.rank),
                "host": os.uname().nodename if hasattr(os, "uname") else "",
                "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
                "fused_hybrid": True,
                "fused_hybrid_rollout_parallelism": "data_parallel",
                "local_batch_size": len(local_indices),
                "global_batch_size": batch_size,
            }
            vllm_stats = local_result.pop("_vllm_stats", []) if local_result else []
            local_payload = {
                "rank": int(self.rank),
                "indices": local_indices,
                "result": local_result,
                "timing": timing,
                "vllm_stats": vllm_stats,
            }
            if self.use_distributed:
                # Only rank 0 needs the global generation payload.  Using
                # all_gather_object here would replicate every rank's response
                # tensors on every trainer rank; that is prohibitive for MC
                # losses where each token carries n-sample support tensors.
                gathered = [None for _ in range(self.world_size)] if self.rank == 0 else None
                torch.distributed.gather_object(
                    local_payload,
                    object_gather_list=gathered,
                    dst=0,
                )
            else:
                gathered = [local_payload]
            if self.rank == 0:
                result = merge_indexed_generation_payloads(gathered, batch_size)
                result_queue.put(result)
            return

        t0 = time.monotonic()
        result = adapter.generate(
            batch,
            return_logprobs=bool(options.get("return_logprobs", False)),
            response_topk_k=int(options.get("response_topk_k", 0)),
            max_response_length=options.get("max_response_length"),
            mc_n_total_samples=mc_n_total_samples,
        )
        timing = {
            "generate_seconds": time.monotonic() - t0,
            "worker_id": 0,
            "host": os.uname().nodename if hasattr(os, "uname") else "",
            "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
            "fused_hybrid": True,
            "fused_hybrid_rollout_parallelism": rollout_parallelism,
        }
        result["timing"] = timing
        result["_worker_timings"] = [timing]
        result["_vllm_stats"] = {0: []}
        if self.rank == 0:
            result_queue.put(result)

    def _hybrid_dp_scatter_plan(self, batch: dict) -> list[dict]:
        from opd.hybrid.dp_merge import (
            indices_from_ranges,
            split_batch_for_dp_ranks,
        )

        batch_size = int(batch["input_ids"].size(0))
        use_global_mini_plan = bool(
            self._should_use_global_mini_plan(
                batch_size,
                self.mini_batch_size,
                self.world_size,
            )
        )
        indices_by_rank = []
        metadata_by_rank = []
        if use_global_mini_plan:
            for rank in range(self.world_size):
                plan = self._build_rank_mini_plan(
                    global_bs=batch_size,
                    mini_batch_size=self.mini_batch_size,
                    rank=rank,
                    world_size=self.world_size,
                    micro_batch_size=self.micro_batch_size,
                )
                indices_by_rank.append(indices_from_ranges(plan["rank_source_ranges"]))
                metadata_by_rank.append({
                    "_use_global_mini_plan": True,
                    "_mini_slices": list(plan["local_mini_slices"]),
                    "_global_mini_slices": list(plan["global_mini_slices"]),
                    "_rank_source_ranges": list(plan["rank_source_ranges"]),
                    "_common_micro_counts": list(plan["common_micro_counts"]),
                })
        else:
            per_rank = batch_size // max(self.world_size, 1)
            for rank in range(self.world_size):
                start = rank * per_rank
                end = start + per_rank
                # This path mirrors the historical trainer rank-first split.  In
                # normal FSDP fused-DP configs, unsafe non-divisible cases are
                # routed through the global-mini plan above.
                indices_by_rank.append(list(range(start, end)))
                metadata_by_rank.append({"_use_global_mini_plan": False})

        shards = split_batch_for_dp_ranks(
            batch,
            self.world_size,
            indices_by_rank=indices_by_rank,
        )
        for rank, shard in enumerate(shards):
            shard.update(metadata_by_rank[rank])
        return shards

    def _indexed_global_long_tensor(self, values, indices, global_size: int) -> torch.Tensor:
        device = self.device if torch.cuda.is_available() else torch.device("cpu")
        out = torch.full((int(global_size),), -1, dtype=torch.long, device=device)
        index_list = [int(i) for i in indices]
        if index_list:
            idx = torch.tensor(index_list, dtype=torch.long, device=device)
            out[idx] = values.detach().to(device=device, dtype=torch.long)
        if self.use_distributed:
            torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.MAX)
        return out.cpu()

    def _all_rank_timing_summary(self, local: dict) -> list[dict]:
        device = self.device if torch.cuda.is_available() else torch.device("cpu")
        t = torch.tensor(
            [
                float(local.get("rank", self.rank)),
                float(local.get("generate_seconds", 0.0)),
                float(local.get("local_batch_size", 0)),
                float(local.get("global_batch_size", 0)),
            ],
            dtype=torch.float64,
            device=device,
        )
        if self.use_distributed:
            gathered = [torch.zeros_like(t) for _ in range(self.world_size)]
            torch.distributed.all_gather(gathered, t)
        else:
            gathered = [t]
        timings = []
        for item in gathered:
            vals = item.detach().cpu().tolist()
            timings.append({
                "worker_id": int(vals[0]),
                "rank": int(vals[0]),
                "generate_seconds": float(vals[1]),
                "local_batch_size": int(vals[2]),
                "global_batch_size": int(vals[3]),
                "fused_hybrid": True,
                "fused_hybrid_rollout_parallelism": "data_parallel",
            })
        return timings

    def _scatter_cached_dp_payload(self, payload: dict) -> dict:
        if self.rank == 0:
            batch = payload["batch"]
            options = payload.get("options", {}) or {}
            cache_id = f"fh-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
            shards = self._hybrid_dp_scatter_plan(batch)
            scatter_payloads = []
            for rank, shard in enumerate(shards):
                scatter_payloads.append({
                    "cache_id": cache_id,
                    "batch": shard,
                    "indices": list(shard.get("_global_sample_indices", [])),
                    "options": options,
                    "global_batch_size": int(batch["input_ids"].size(0)),
                    "rank": rank,
                })
        else:
            scatter_payloads = None
        if self.use_distributed:
            recv = [None]
            torch.distributed.scatter_object_list(recv, scatter_payloads, src=0)
            return recv[0]
        return scatter_payloads[0]

    def _handle_hybrid_generate_cached(self, cmd, t_recv, result_queue):
        if self.rank == 0:
            payload = cmd[1]
        else:
            payload = None
        adapter = self._ensure_hybrid_adapter()
        rollout_parallelism = getattr(adapter, "rollout_parallelism", "spmd_tp")
        if rollout_parallelism != "data_parallel":
            raise RuntimeError(
                "hybrid_generate_cached requires data_parallel rollout; "
                f"got {rollout_parallelism!r}"
            )
        local_pack = self._scatter_cached_dp_payload(payload)
        cache_id = str(local_pack["cache_id"])
        local_batch = local_pack["batch"]
        local_indices = [int(i) for i in local_pack.get("indices", [])]
        options = local_pack.get("options", {}) or {}
        global_batch_size = int(local_pack["global_batch_size"])
        mc_n_total_samples = int(options.get("mc_n_total_samples", 0) or 0)

        t0 = time.monotonic()
        local_result = adapter.generate(
            local_batch,
            return_logprobs=bool(options.get("return_logprobs", False)),
            response_topk_k=int(options.get("response_topk_k", 0)),
            max_response_length=options.get("max_response_length"),
            mc_n_total_samples=mc_n_total_samples,
        )
        generate_seconds = time.monotonic() - t0
        vllm_stats = local_result.pop("_vllm_stats", []) if local_result else []
        cache_metadata = {
            "cache_id": cache_id,
            "indices": local_indices,
            "global_batch_size": global_batch_size,
            "options": options,
            "vllm_stats": vllm_stats,
            "_rank_sharded": True,
            "_global_sample_indices": local_indices,
            "_global_batch_size": global_batch_size,
            "_dp_rank": int(self.rank),
            "_dp_world_size": int(self.world_size),
        }
        for key in GLOBAL_MINI_PLAN_METADATA_KEYS:
            if key in local_batch:
                cache_metadata[key] = local_batch[key]
        self._hybrid_generation_cache[cache_id] = {
            "gen_output": local_result,
            "metadata": cache_metadata,
        }

        prompt_lengths = self._indexed_global_long_tensor(
            local_result["prompt_lengths"],
            local_indices,
            global_batch_size,
        )
        response_lengths = self._indexed_global_long_tensor(
            local_result["response_lengths"],
            local_indices,
            global_batch_size,
        )
        local_timing = {
            "rank": int(self.rank),
            "generate_seconds": generate_seconds,
            "local_batch_size": len(local_indices),
            "global_batch_size": global_batch_size,
        }
        worker_timings = self._all_rank_timing_summary(local_timing)
        if self.rank == 0:
            if (prompt_lengths < 0).any() or (response_lengths < 0).any():
                raise RuntimeError(
                    f"cached DP generation produced incomplete length summary for {cache_id}"
                )
            timing = {
                "generate_seconds": max(t["generate_seconds"] for t in worker_timings),
                "worker_id": 0,
                "host": os.uname().nodename if hasattr(os, "uname") else "",
                "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
                "fused_hybrid": True,
                "fused_hybrid_rollout_parallelism": "data_parallel",
                "fused_hybrid_dp_cached_generation": True,
                "global_batch_size": global_batch_size,
            }
            print(
                f"[FusedHybrid] cached_dp_generate cache_id={cache_id} "
                f"global_bs={global_batch_size}",
                flush=True,
            )
            result_queue.put({
                "_hybrid_cached": True,
                "cache_id": cache_id,
                "prompt_lengths": prompt_lengths,
                "response_lengths": response_lengths,
                "timing": timing,
                "_worker_timings": worker_timings,
                "_vllm_stats": {},
                "fused_hybrid_dp_cached_generation": True,
            })

    def _ensure_hybrid_teacher_client(self, teacher_addr: str):
        client = self._hybrid_teacher_clients.get(teacher_addr)
        if client is None:
            from opd.worker.teacher.client import TeacherClient
            client = TeacherClient(teacher_addr, n_workers=1)
            self._hybrid_teacher_clients[teacher_addr] = client
        return client

    @staticmethod
    def _append_eos_teacher_query_indices(query_indices_response, eos_token_id: int):
        out = []
        for q_idx in query_indices_response:
            if q_idx is None or q_idx.dim() != 2:
                out.append(q_idx)
                continue
            eos_col = torch.full(
                (q_idx.size(0), 1),
                int(eos_token_id),
                dtype=q_idx.dtype,
                device=q_idx.device,
            )
            out.append(torch.cat([q_idx, eos_col], dim=-1))
        return out

    def _score_cached_teacher_local(self, gen_output: dict, teacher_client, options: dict):
        from opd.data.batch_utils import (
            adapt_mc_response_samples,
            adapt_response_support,
            pad_teacher,
        )

        full_lists = gen_output["full_token_lists"]
        prompt_lengths = gen_output.get("prompt_lengths")
        uses_rollout_support_topk = bool(options.get("uses_rollout_support_topk", False))
        uses_multi_sample_policy_gradient_kl = bool(
            options.get("uses_multi_sample_policy_gradient_kl", False)
        )
        uses_multi_sample_forward_kl = bool(
            options.get("uses_multi_sample_forward_kl", False)
        )
        uses_mof_mc_candidates = bool(options.get("uses_mof_mc_candidates", False))
        uses_mof_generated_only = bool(options.get("uses_mof_generated_only", False))
        uses_mof_eos_aware = bool(options.get("uses_mof_eos_aware", False))
        uses_multi_sample_kl = (
            uses_multi_sample_policy_gradient_kl
            or uses_multi_sample_forward_kl
            or uses_mof_mc_candidates
        )
        query_indices_response = (
            gen_output.get("mc_query_indices_response")
            if (uses_multi_sample_policy_gradient_kl or uses_mof_mc_candidates)
            else gen_output.get("query_indices_response")
        )
        if uses_mof_mc_candidates and uses_mof_eos_aware:
            query_indices_response = self._append_eos_teacher_query_indices(
                query_indices_response,
                int(options["eos_token_id"]),
            )

        n = max(int(getattr(teacher_client, "n_workers", 1)), 1)
        chunk = max((len(full_lists) + n - 1) // n, 1)
        futures = []
        for i in range(0, len(full_lists), chunk):
            if uses_rollout_support_topk or uses_multi_sample_kl:
                batch_request_ids = [f"q-{uuid.uuid4().hex}" for _ in full_lists[i:i + chunk]]
                submit_kwargs = {
                    "prompt_lengths": prompt_lengths[i:i + chunk].tolist(),
                    "query_request_ids": batch_request_ids,
                }
                if uses_multi_sample_forward_kl:
                    submit_kwargs["teacher_mc_n_total_samples"] = int(
                        options.get("pg_kl_n_total_samples", 0) or 0
                    )
                else:
                    submit_kwargs["query_indices_response"] = query_indices_response[i:i + chunk]
                futures.append(teacher_client.submit(full_lists[i:i + chunk], **submit_kwargs))
            else:
                futures.append(teacher_client.submit(full_lists[i:i + chunk]))

        t_submit = time.time()
        all_logps, all_idx, all_token_logps = [], [], []
        teacher_mono_start = None
        teacher_mono_end = None
        for fut in futures:
            _, logps, indices, token_logps, ms, me = fut.result()
            all_logps.extend(logps)
            all_idx.extend(indices)
            all_token_logps.extend(token_logps)
            if ms is not None:
                teacher_mono_start = min(ms, teacher_mono_start or ms)
            if me is not None:
                teacher_mono_end = max(me, teacher_mono_end or me)
        teacher_seconds = time.time() - t_submit
        with timer() as t_pad:
            if uses_multi_sample_forward_kl:
                out = adapt_mc_response_samples(gen_output, all_idx, all_logps, None)
            elif uses_mof_mc_candidates:
                out = adapt_mc_response_samples(
                    gen_output,
                    query_indices_response,
                    all_logps,
                    None,
                )
                if out is not None and uses_mof_eos_aware:
                    out["eos_token_id"] = int(options["eos_token_id"])
            elif uses_multi_sample_policy_gradient_kl:
                out = adapt_mc_response_samples(
                    gen_output,
                    gen_output["mc_query_indices_response"],
                    all_logps,
                    gen_output["mc_query_old_logprobs_response"],
                )
            elif uses_rollout_support_topk:
                out = adapt_response_support(
                    gen_output,
                    gen_output["query_indices_response"],
                    all_logps,
                    gen_output.get("query_logprobs_response"),
                )
            elif uses_mof_generated_only:
                out = pad_teacher(gen_output, all_logps, all_idx, all_token_logps)
            else:
                out = pad_teacher(gen_output, all_logps, all_idx, all_token_logps)
                if out is not None and uses_mof_eos_aware:
                    out["eos_token_id"] = int(options["eos_token_id"])
        if out is None:
            raise RuntimeError("cached teacher scoring produced no teacher output")
        return out, {
            "teacher_seconds": teacher_seconds,
            "pad_seconds": t_pad["elapsed"],
            "teacher_mono_start": teacher_mono_start,
            "teacher_mono_end": teacher_mono_end,
            "n_prompts": len(full_lists),
            "total_tok": sum(len(tl) for tl in full_lists),
        }

    def _handle_hybrid_start_teacher_cached(self, cmd, t_recv, result_queue):
        payload = cmd[1] if self.rank == 0 else None
        if self.use_distributed:
            payload_list = [payload]
            torch.distributed.broadcast_object_list(payload_list, src=0)
            payload = payload_list[0]
        cache_id = str(payload["cache_id"])
        if cache_id not in self._hybrid_generation_cache:
            raise RuntimeError(f"missing cached generation handle {cache_id!r} on rank {self.rank}")
        teacher_addr = str(payload["teacher_addr"])
        teacher_options = payload.get("teacher_options", {}) or {}
        entry = self._hybrid_generation_cache[cache_id]
        client = self._ensure_hybrid_teacher_client(teacher_addr)
        holder = {"cache_id": cache_id}

        def _bg_score():
            try:
                teacher_output, metrics = self._score_cached_teacher_local(
                    entry["gen_output"],
                    client,
                    teacher_options,
                )
                holder["teacher_output"] = teacher_output
                holder["metrics"] = metrics
                holder["done"] = time.monotonic()
            except Exception as exc:  # pragma: no cover - propagated on resolve
                holder["error"] = exc
                holder["done"] = time.monotonic()

        thread = threading.Thread(target=_bg_score, daemon=True)
        holder["thread"] = thread
        self._hybrid_teacher_futures[cache_id] = holder
        thread.start()
        if self.rank == 0:
            result_queue.put({
                "status": "teacher_started",
                "cache_id": cache_id,
                "_hybrid_cached_teacher": True,
            })

    def _handle_hybrid_resolve_teacher_cached(self, cmd, t_recv, result_queue):
        payload = cmd[1] if self.rank == 0 else None
        if self.use_distributed:
            payload_list = [payload]
            torch.distributed.broadcast_object_list(payload_list, src=0)
            payload = payload_list[0]
        cache_id = str(payload["cache_id"])
        holder = self._hybrid_teacher_futures.get(cache_id)
        if holder is None:
            raise RuntimeError(f"missing cached teacher future {cache_id!r} on rank {self.rank}")
        thread = holder["thread"]
        t_join_start = time.monotonic()
        thread.join()
        wait_seconds = time.monotonic() - t_join_start
        if "error" in holder:
            raise RuntimeError(f"cached teacher scoring failed on rank {self.rank}: {holder['error']}") from holder["error"]
        self._hybrid_teacher_cache[cache_id] = holder["teacher_output"]
        metrics = holder.get("metrics", {})
        device = self.device if torch.cuda.is_available() else torch.device("cpu")
        local = torch.tensor(
            [
                float(metrics.get("teacher_seconds", 0.0)),
                float(metrics.get("pad_seconds", 0.0)),
                float(wait_seconds),
                float(metrics.get("n_prompts", 0)),
                float(metrics.get("total_tok", 0)),
            ],
            dtype=torch.float64,
            device=device,
        )
        if self.use_distributed:
            max_vals = local.clone()
            torch.distributed.all_reduce(max_vals, op=torch.distributed.ReduceOp.MAX)
            sum_vals = local.clone()
            torch.distributed.all_reduce(sum_vals, op=torch.distributed.ReduceOp.SUM)
        else:
            max_vals = local
            sum_vals = local
        if self.rank == 0:
            max_cpu = max_vals.detach().cpu().tolist()
            sum_cpu = sum_vals.detach().cpu().tolist()
            result_queue.put({
                "status": "teacher_resolved",
                "cache_id": cache_id,
                "_hybrid_cached_teacher": True,
                "_teacher_seconds": float(max_cpu[0]),
                "_pad_seconds": float(max_cpu[1]),
                "_teacher_wait_seconds": float(max_cpu[2]),
                "_resolve_end": time.monotonic(),
                "fused_hybrid_dp_cached_teacher": True,
                "fused_hybrid_dp_cached_teacher_prompts": int(sum_cpu[3]),
                "fused_hybrid_dp_cached_teacher_tokens": int(sum_cpu[4]),
            })

    def _handle_hybrid_train_cached(self, cmd, t_recv, result_queue):
        from opd.data.opd_payload import build_opd_train_batch

        payload = cmd[1] if self.rank == 0 else None
        if self.use_distributed:
            payload_list = [payload]
            torch.distributed.broadcast_object_list(payload_list, src=0)
            payload = payload_list[0]
        cache_id = str(payload["cache_id"])
        gen_entry = self._hybrid_generation_cache.get(cache_id)
        teacher_output = self._hybrid_teacher_cache.get(cache_id)
        if gen_entry is None:
            raise RuntimeError(f"missing cached generation handle {cache_id!r} on rank {self.rank}")
        if teacher_output is None:
            raise RuntimeError(f"missing cached teacher output {cache_id!r} on rank {self.rank}")
        local_gen = dict(gen_entry["gen_output"])
        metadata = dict(gen_entry.get("metadata", {}))
        for key, value in metadata.items():
            local_gen[key] = value
        batch = build_opd_train_batch(local_gen, teacher_output)
        for key, value in metadata.items():
            batch[key] = value
        batch["_rank_sharded"] = True
        batch["_send_mono"] = payload.get("send_mono")
        try:
            self._execute_train_batch(batch, t_recv, result_queue)
        finally:
            self._hybrid_generation_cache.pop(cache_id, None)
            self._hybrid_teacher_cache.pop(cache_id, None)
            self._hybrid_teacher_futures.pop(cache_id, None)

    def _handle_hybrid_release_rollout(self, cmd, t_recv, result_queue):
        reason = cmd[1].get("reason", "") if self.rank == 0 and len(cmd) > 1 else ""
        if self.use_distributed:
            reason_list = [reason]
            torch.distributed.broadcast_object_list(reason_list, src=0)
            reason = reason_list[0]
        adapter = self._ensure_hybrid_adapter()
        status = adapter.sleep(reason=reason)
        if self.rank == 0:
            result_queue.put(status)

    def _refresh_hybrid_adapter_weights(self, actor_version: int) -> dict:
        adapter = self._ensure_hybrid_adapter()
        sync_cfg = getattr(self, "_fused_hybrid_sync_cfg", {}) or {}
        weight_backend = str(sync_cfg.get("weight_update_backend", "bucketed_inprocess"))
        debug_full_state = bool(sync_cfg.get("debug_full_state_sync", False))
        if (
            weight_backend == "bucketed_inprocess"
            and not debug_full_state
            and hasattr(self, "_iter_hybrid_weight_tensors")
        ):
            try:
                return adapter.refresh_named_tensors(
                    self._iter_hybrid_weight_tensors(),
                    actor_version,
                    full_state_materialized=False,
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        sd = self._get_clean_state_dict()
        try:
            metrics = adapter.refresh_weights(sd, actor_version)
        finally:
            del sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return metrics

    def _finalize_hybrid_checksum_metrics(self, metrics: dict) -> dict:
        """Reduce and log fused-hybrid checksum metrics after rank-local refresh."""
        prefix = "fused_hybrid_weight_update_checksum"
        if not metrics.get(f"{prefix}_enabled"):
            return metrics
        target_local = metrics.get(f"{prefix}_target_local")
        source = metrics.get(f"{prefix}_source")
        if target_local is None or source is None:
            return metrics

        target = float(target_local)
        target_count = float(metrics.get(f"{prefix}_target_count_local", 0) or 0)
        rollout_parallelism = str(
            (getattr(self, "_fused_hybrid_sync_cfg", {}) or {}).get(
                "rollout_parallelism",
                "spmd_tp",
            )
        )
        if rollout_parallelism == "data_parallel":
            source = float(source)
            missing_sources = int(metrics.get(f"{prefix}_missing_source_count", 0) or 0)
            rel_err = abs(source - target) / max(abs(source), 1e-12)
            local = {
                "source": source,
                "target": target,
                "rel_error": rel_err,
                "ok": rel_err <= 1e-6 and missing_sources == 0,
                "source_count": int(metrics.get(f"{prefix}_source_count", 0) or 0),
                "target_count": int(target_count),
                "missing_source_count": missing_sources,
            }
            gathered = [local]
            if self.use_distributed:
                import torch.distributed as dist

                gathered = [None for _ in range(self.world_size)]
                dist.all_gather_object(gathered, local)
            ok_by_rank = [bool(item.get("ok")) for item in gathered]
            rel_errors = [float(item.get("rel_error", 0.0)) for item in gathered]
            ok = all(ok_by_rank)
            rel_error_max = max(rel_errors) if rel_errors else 0.0
            source_values = [float(item.get("source", 0.0)) for item in gathered]
            target_values = [float(item.get("target", 0.0)) for item in gathered]
            target_counts = [int(item.get("target_count", 0)) for item in gathered]
            source_counts = [int(item.get("source_count", 0)) for item in gathered]
            missing_by_rank = [int(item.get("missing_source_count", 0)) for item in gathered]
            missing_total = sum(missing_by_rank)
            metrics[f"{prefix}_source"] = local["source"]
            metrics[f"{prefix}_target"] = local["target"]
            metrics[f"{prefix}_target_count"] = local["target_count"]
            metrics[f"{prefix}_rel_error"] = rel_error_max
            metrics[f"{prefix}_ok"] = ok
            metrics[f"{prefix}_missing_source_count"] = missing_total
            metrics[f"{prefix}_replicated_dp"] = True
            metrics[f"{prefix}_rank_count"] = len(gathered)
            metrics[f"{prefix}_rank_ok"] = ok_by_rank
            metrics[f"{prefix}_all_ranks_ok"] = ok
            metrics[f"{prefix}_source_by_rank"] = source_values
            metrics[f"{prefix}_target_local_by_rank"] = target_values
            metrics[f"{prefix}_source_count_by_rank"] = source_counts
            metrics[f"{prefix}_target_count_by_rank"] = target_counts
            metrics[f"{prefix}_missing_source_count_by_rank"] = missing_by_rank
            metrics[f"{prefix}_rel_error_by_rank"] = rel_errors
            metrics[f"{prefix}_source_min"] = min(source_values)
            metrics[f"{prefix}_source_max"] = max(source_values)
            metrics[f"{prefix}_target_min"] = min(target_values)
            metrics[f"{prefix}_target_max"] = max(target_values)
            metrics[f"{prefix}_rel_error_max"] = rel_error_max
            if self.rank == 0:
                if ok:
                    print(
                        "[FusedHybrid] Weight checksum OK "
                        f"(data_parallel ranks={len(gathered)}, "
                        f"source_range=[{metrics[f'{prefix}_source_min']:.2f}, "
                        f"{metrics[f'{prefix}_source_max']:.2f}], "
                        f"target_range=[{metrics[f'{prefix}_target_min']:.2f}, "
                        f"{metrics[f'{prefix}_target_max']:.2f}], "
                        f"max_rel_err={rel_error_max:.2e})",
                        flush=True,
                    )
                else:
                    print(
                        "[FusedHybrid] WARNING: Weight checksum mismatch! "
                        f"data_parallel max_rel_err={rel_error_max:.2e} "
                        f"missing_sources_total={missing_total}",
                        flush=True,
                    )
            return metrics

        if self.use_distributed:
            import torch.distributed as dist

            device = self.device if torch.cuda.is_available() else torch.device("cpu")
            reduced = torch.tensor([target, target_count], dtype=torch.float64, device=device)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            target = float(reduced[0].item())
            target_count = int(reduced[1].item())

        source = float(source)
        missing_sources = int(metrics.get(f"{prefix}_missing_source_count", 0) or 0)
        rel_err = abs(source - target) / max(abs(source), 1e-12)
        ok = rel_err <= 1e-6 and missing_sources == 0
        metrics[f"{prefix}_target"] = target
        metrics[f"{prefix}_target_count"] = int(target_count)
        metrics[f"{prefix}_rel_error"] = rel_err
        metrics[f"{prefix}_ok"] = ok

        if self.rank == 0:
            if ok:
                print(
                    "[FusedHybrid] Weight checksum OK "
                    f"(source={source:.2f}, rollout={target:.2f}, "
                    f"rel_err={rel_err:.2e})",
                    flush=True,
                )
            else:
                print(
                    "[FusedHybrid] WARNING: Weight checksum mismatch! "
                    f"source={source:.6f} rollout={target:.6f} "
                    f"rel_err={rel_err:.2e} missing_sources={missing_sources}",
                    flush=True,
                )
        return metrics

    def _handle_hybrid_refresh_weights(self, cmd, t_recv, result_queue):
        actor_version = int(cmd[1].get("actor_version", 0)) if self.rank == 0 else None
        if self.use_distributed:
            version_list = [actor_version]
            torch.distributed.broadcast_object_list(version_list, src=0)
            actor_version = int(version_list[0])
        metrics = self._refresh_hybrid_adapter_weights(actor_version)
        metrics = self._finalize_hybrid_checksum_metrics(metrics)
        if self.rank == 0:
            result_queue.put({"status": "weights_refreshed", "metrics": metrics})

    def _handle_hybrid_prepare_train(self, cmd, t_recv, result_queue):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.use_distributed:
            torch.distributed.barrier()
        if self.rank == 0:
            result_queue.put({"status": "trainer_ready"})

    def _handle_memory_snapshot(self, cmd, t_recv, result_queue):
        label = cmd[1] if self.rank == 0 and cmd is not None and len(cmd) > 1 else ""
        if self.use_distributed:
            torch.distributed.barrier()
        if self.rank == 0:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                result_queue.put({
                    "status": "memory_snapshot",
                    "label": label,
                    "device": int(device),
                    "allocated": int(torch.cuda.memory_allocated(device)),
                    "reserved": int(torch.cuda.memory_reserved(device)),
                    "max_allocated": int(torch.cuda.max_memory_allocated(device)),
                })
            else:
                result_queue.put({
                    "status": "memory_snapshot",
                    "label": label,
                    "device": None,
                    "allocated": 0,
                    "reserved": 0,
                    "max_allocated": 0,
                })

    def _handle_shutdown_weight_transfer(self, cmd, t_recv, result_queue):
        """Destroy the vLLM weight transfer NCCL engine before rollout exits.

        Must be called before rollout processes shut down — otherwise the
        trainer's TCPStore gets EPOLLHUP from departed peers and crashes.
        """
        if self.rank == 0 and self._wt_group is not None:
            try:
                if hasattr(self._wt_group, 'shutdown'):
                    self._wt_group.shutdown()
            except Exception as e:
                print(f"[{self._get_log_prefix()}] weight transfer shutdown error: {e}",
                      flush=True)
            self._wt_group = None
            print(f"[{self._get_log_prefix()}] weight transfer engine shut down", flush=True)
            result_queue.put({"status": "ok"})
        elif self.rank == 0:
            result_queue.put({"status": "ok"})

    def _async_write_checkpoint(self, state_dict, train_state, checkpoint_dir, step):
        """Write model weights and training state to disk in a background thread.

        Args:
            state_dict: CPU model state dict to save as model.pt
            train_state: Training state dict (optimizer, scheduler, step) or None
            checkpoint_dir: Directory to save into (must already exist)
            step: Step number for log message
        """
        def _write():
            model_path = os.path.join(checkpoint_dir, "model.pt")
            tmp_path = model_path + ".tmp"
            torch.save(state_dict, tmp_path)
            os.rename(tmp_path, model_path)
            if train_state is not None:
                state_path = os.path.join(checkpoint_dir, "training_state.pt")
                tmp_path = state_path + ".tmp"
                torch.save(train_state, tmp_path)
                os.rename(tmp_path, state_path)
            print(f"[{self._get_log_prefix()}] Checkpoint saved at step {step} -> {checkpoint_dir}",
                  flush=True)

        self._wait_async_save()

        import threading
        self._async_save_thread = threading.Thread(target=_write, daemon=True)
        self._async_save_thread.start()

    def _wait_async_save(self):
        """Block until the previous async checkpoint save completes."""
        t = getattr(self, "_async_save_thread", None)
        if t is not None and t.is_alive():
            t.join()
            self._async_save_thread = None

    def _post_save_cleanup(self):
        """Hook for post-checkpoint cleanup. Override in subclass if needed."""
        pass

    # -------------------- Weight sync helpers --------------------

    def _vllm_send_merged_weights(self, merge_map, wt_group, state_dict=None):
        """Send weights to rollout via vLLM's NCCL weight transfer engine.

        If merge_map is provided: merges q/k/v → qkv_proj etc. (TP=1 path).
        If merge_map is None: sends raw state dict keys (TP>1 checkpoint format path).
        """
        if state_dict is not None:
            sd = state_dict
        else:
            sd = self._get_clean_state_dict()

        device = self.device

        if merge_map is not None:
            def _merged_iter():
                for vllm_name, sources in merge_map:
                    if len(sources) == 1:
                        yield vllm_name, sd[sources[0]].to(device)
                    else:
                        tensors = [sd[s].to(device) for s in sources]
                        yield vllm_name, torch.cat(tensors, dim=0)
            vllm_trainer_send(_merged_iter(), wt_group, packed=False)
        else:
            # TP>1 / Megatron: send in weights_info order to guarantee
            # send order matches the update_info order on the rollout side.
            # State dict is de-fused (q/k/v, gate/up) by Megatron's
            # _get_clean_state_dict.
            def _raw_iter():
                for name, _, _ in self.weights_info:
                    if name in sd:
                        yield name, sd[name].to(device)
            vllm_trainer_send(_raw_iter(), wt_group, packed=False)

    def _merge_state_dict_for_vllm(self, merge_map, state_dict=None):
        """Merge state dict to match vLLM param names, return as CPU list.

        Same merge logic as _vllm_send_merged_weights but returns a list of
        (vllm_name, cpu_tensor) for queue-based transfer (TP>1 path).
        """
        sd = state_dict if state_dict is not None else self._get_clean_state_dict()
        merged = []
        for vllm_name, sources in merge_map:
            if len(sources) == 1:
                merged.append((vllm_name, sd[sources[0]].cpu()))
            else:
                tensors = [sd[s] for s in sources]
                merged.append((vllm_name, torch.cat(tensors, dim=0).cpu()))
        return merged

    def _get_clean_state_dict(self):
        """Get state dict with cleaned param names (no wrapper prefixes).

        Subclasses may override for backend-specific name cleaning.
        """
        sd = {}
        for name, param in self.model.state_dict().items():
            clean = name.replace("_orig_mod.", "")
            sd[clean] = param
        return sd

    # -------------------- Model setup helpers --------------------

    def _patch_gradient_checkpointing(self):
        """Fix gradient checkpointing for Qwen3 (and similar models)."""
        from torch.utils.checkpoint import checkpoint

        inner = getattr(self.model, "model", None)
        if inner is None or not hasattr(inner, "layers"):
            return

        for layer in inner.layers:
            orig_fwd = layer.forward

            def make_ckpt(fn):
                def ckpt_fwd(*a, **kw):
                    return checkpoint(fn, *a, use_reentrant=False, **kw)
                return ckpt_fwd

            layer.forward = make_ckpt(orig_fwd)

        print(f"[{self._get_log_prefix()}] Patched gradient checkpointing for "
              f"{type(inner).__name__} ({len(inner.layers)} layers)", flush=True)

    # -------------------- Batch preparation --------------------

    def _prepare_batch(self, batch):
        """Generic batch preparation: rank-split, mini-batch metadata, truncation.

        Works for any mode. Slices ALL tensors in the batch dict by rank,
        truncates trailing padding, and computes mini-batch metadata.

        Returns a dict with:
            All original tensor keys (rank-split and truncated),
            n_mini, mini_bs, seq_len, actual_max_len
        """
        import torch

        rank = getattr(self, 'dp_rank', self.rank)
        world_size = getattr(self, 'dp_world_size', self.world_size)
        mini_batch_size = self.mini_batch_size

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bs = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Rank split: slice all batch-dim tensors
        if world_size > 1:
            per_rank = bs // world_size
            s = rank * per_rank
            e = s + per_rank
            sliced = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.size(0) == bs:
                    sliced[k] = v[s:e]
                else:
                    sliced[k] = v
            batch = sliced
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            bs = per_rank

        # Mini-batch metadata — mini_batch_size is global, divide by world_size
        per_rank_mini_bs = max(1, mini_batch_size // world_size) if mini_batch_size > 0 else 0
        if per_rank_mini_bs > 0 and per_rank_mini_bs < bs:
            n_mini = bs // per_rank_mini_bs
            mini_bs = per_rank_mini_bs
        else:
            n_mini = 1
            mini_bs = bs

        # Truncate trailing padding
        nonzero_cols = attention_mask.nonzero(as_tuple=True)[1]
        if nonzero_cols.numel() > 0:
            actual_max_len = int(nonzero_cols.max().item()) + 2
        else:
            actual_max_len = seq_len
        actual_max_len = min(actual_max_len, seq_len)

        if actual_max_len < seq_len:
            truncated = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.size(1) == seq_len:
                    truncated[k] = v[:, :actual_max_len]
                else:
                    truncated[k] = v
            batch = truncated

        result = dict(batch)
        result["n_mini"] = n_mini
        result["mini_bs"] = mini_bs
        result["seq_len"] = seq_len
        result["actual_max_len"] = actual_max_len
        return result

    def _prepare_train_batch(self, batch):
        """Prepare a training batch: rank-split, mini-batch split, response mask, truncation.

        Returns a dict with prepared tensors and metadata:
            input_ids, attention_mask, teacher_topk_logps, teacher_topk_indices,
            response_mask, prompt_lengths, batch (possibly modified with rank-sliced extras),
            n_mini, mini_bs, seq_len, actual_max_len, max_prompt, orig_seq_len
        """
        rank = getattr(self, 'dp_rank', self.rank)
        world_size = getattr(self, 'dp_world_size', self.world_size)
        mini_batch_size = self.mini_batch_size
        max_response_length = self.max_response_length

        input_ids = batch["input_ids"]
        bs = input_ids.size(0)
        rank_sharded = bool(batch.get("_rank_sharded", False))
        global_bs = int(batch.get("_global_batch_size", bs)) if rank_sharded else bs

        # Canonical multi-sample Monte Carlo trainer handoff.
        multi_sample_mode = all(
            key in batch
            for key in (
                "mc_sample_indices",
                "mc_teacher_logprobs",
                "mc_valid_mask",
            )
        )

        should_plan = getattr(self, "_should_use_global_mini_plan", None)
        if rank_sharded:
            use_global_mini_plan = bool(batch.get("_use_global_mini_plan", False))
        else:
            use_global_mini_plan = bool(
                should_plan is not None
                and should_plan(global_bs, mini_batch_size, world_size)
            )
        mini_plan = None

        # With multi-rank, split batch across ranks.  The legacy path slices one
        # contiguous rank shard and then computes per-rank mini-batches.  For
        # non-divisible OPD/FSDP shapes, use the safer global-mini-first plan:
        # configured mini-batches are formed globally, each mini is balanced
        # across ranks, and this rank's shards are concatenated locally.
        if rank_sharded and use_global_mini_plan:
            mini_plan = {
                "local_mini_slices": [
                    tuple(x) for x in batch.get("_mini_slices", [])
                ],
                "global_mini_slices": [
                    tuple(x) for x in batch.get("_global_mini_slices", [])
                ],
                "rank_source_ranges": [
                    tuple(x) for x in batch.get("_rank_source_ranges", [])
                ],
                "common_micro_counts": list(batch.get("_common_micro_counts", [])),
                "local_batch_size": bs,
            }
            if not mini_plan["local_mini_slices"]:
                raise RuntimeError(
                    "pre-sharded global-mini batch missing _mini_slices metadata"
                )
        elif use_global_mini_plan:
            mini_plan = self._build_rank_mini_plan(
                global_bs=global_bs,
                mini_batch_size=mini_batch_size,
                rank=rank,
                world_size=world_size,
                micro_batch_size=self.micro_batch_size,
            )
            sliced = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.size(0) == global_bs:
                    pieces = [
                        v[s:e]
                        for s, e in mini_plan["rank_source_ranges"]
                        if e > s
                    ]
                    sliced[k] = torch.cat(pieces, dim=0) if pieces else v[:0]
                else:
                    sliced[k] = v
            batch = sliced
            bs = mini_plan["local_batch_size"]
        elif rank_sharded:
            # Fused-hybrid DP cached train already owns this rank's local shard.
            # Do not split a second time.
            pass
        elif world_size > 1:
            if bs % world_size != 0:
                print(f"WARNING: batch size {bs} not divisible by world_size {world_size}, "
                      f"dropping {bs % world_size} samples", flush=True)
            per_rank = bs // world_size
            rank_start = rank * per_rank
            rank_end = rank_start + per_rank
            sliced = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.size(0) == bs:
                    sliced[k] = v[rank_start:rank_end]
                else:
                    sliced[k] = v
            batch = sliced
            bs = per_rank

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_lengths = batch["prompt_lengths"]
        support_mode = "support_topk_logps" in batch
        teacher_topk_logps = batch["support_topk_logps"] if support_mode else batch.get("teacher_topk_logps")
        teacher_topk_indices = batch["support_topk_indices"] if support_mode else batch.get("teacher_topk_indices")
        teacher_hidden_states = batch.get("teacher_hidden_states")
        teacher_hidden_valid_mask = batch.get("teacher_hidden_valid_mask")
        teacher_valid_mask = (
            batch["support_valid_mask"] if support_mode
            else batch.get("teacher_valid_mask", teacher_hidden_valid_mask)
        )

        # Mini-batch metadata.  In the global-mini-first path, these are only
        # compatibility diagnostics; FSDP consumes the explicit slice metadata.
        if use_global_mini_plan:
            n_mini = len(mini_plan["local_mini_slices"])
            mini_bs = max(
                (e - s for s, e in mini_plan["local_mini_slices"]),
                default=0,
            )
        else:
            # Mini-batch splitting — mini_batch_size is global, divide by world_size
            per_rank_mini_bs = max(1, mini_batch_size // world_size) if mini_batch_size > 0 else 0
            if per_rank_mini_bs > 0 and per_rank_mini_bs < bs:
                assert bs % per_rank_mini_bs == 0, f"per-rank batch size {bs} not divisible by per_rank_mini_bs {per_rank_mini_bs} (global mini_batch_size={mini_batch_size}, world_size={world_size})"
                n_mini = bs // per_rank_mini_bs
                mini_bs = per_rank_mini_bs
            else:
                n_mini = 1
                mini_bs = bs

        seq_len = input_ids.size(1)
        orig_seq_len = batch.get("_orig_seq_len", seq_len)
        max_prompt = orig_seq_len - max_response_length
        response_mask = attention_mask.clone().bool()
        response_mask[:, :max_prompt] = False
        if multi_sample_mode:
            response_mask &= batch["mc_valid_mask"]
        elif teacher_valid_mask is not None:
            response_mask &= teacher_valid_mask

        # Truncate trailing zeros (unused response positions) to avoid wasting compute.
        nonzero_cols = attention_mask.nonzero(as_tuple=True)[1]
        if nonzero_cols.numel() > 0:
            actual_max_len = int(nonzero_cols.max().item()) + 2
        else:
            actual_max_len = seq_len
        actual_max_len = min(actual_max_len, seq_len)
        if actual_max_len < seq_len:
            truncated = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.size(1) == seq_len:
                    truncated[k] = v[:, :actual_max_len]
                else:
                    truncated[k] = v
            batch = truncated
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            prompt_lengths = batch["prompt_lengths"]
            teacher_topk_logps = batch["support_topk_logps"] if support_mode else batch.get("teacher_topk_logps")
            teacher_topk_indices = batch["support_topk_indices"] if support_mode else batch.get("teacher_topk_indices")
            teacher_hidden_states = batch.get("teacher_hidden_states")
            teacher_hidden_valid_mask = batch.get("teacher_hidden_valid_mask")
            teacher_valid_mask = (
                batch["support_valid_mask"] if support_mode
                else batch.get("teacher_valid_mask", teacher_hidden_valid_mask)
            )
            response_mask = response_mask[:, :actual_max_len]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_topk_logps": teacher_topk_logps,
            "teacher_topk_indices": teacher_topk_indices,
            "teacher_hidden_states": teacher_hidden_states,
            "teacher_hidden_valid_mask": teacher_hidden_valid_mask,
            "response_mask": response_mask,
            "prompt_lengths": prompt_lengths,
            "batch": batch,
            "n_mini": n_mini,
            "mini_bs": mini_bs,
            "seq_len": seq_len,
            "actual_max_len": actual_max_len,
            "max_prompt": max_prompt,
            "orig_seq_len": orig_seq_len,
        }
        if use_global_mini_plan:
            result.update({
                "_use_global_mini_plan": True,
                "_mini_slices": list(mini_plan["local_mini_slices"]),
                "_global_mini_slices": list(mini_plan["global_mini_slices"]),
                "_rank_source_ranges": list(mini_plan["rank_source_ranges"]),
                "_global_batch_size": global_bs,
                "_configured_global_mini_batch_size": mini_batch_size,
                "_common_micro_counts": list(mini_plan["common_micro_counts"]),
            })
        else:
            result["_use_global_mini_plan"] = False
        return result

    def _patch_model_for_chunked_kl(self):
        """Monkey-patch model forward to support chunked LM head KL computation.

        When called with _kl_args kwarg, the forward:
        1. Runs the transformer backbone (model.model) to get hidden states
        2. Calls chunked_lm_head_gather to compute log-softmax at given indices
        3. Returns gathered log-probs [B, S, K] instead of full logits [B, S, V]

        Returns True if patching succeeded, False if model architecture is unsupported.
        """
        model = self.model
        if not hasattr(model, 'model') or not hasattr(model, 'lm_head'):
            return False

        import types

        original_forward = model.forward

        def _patched_forward(self_model, *args, _kl_args=None, **kwargs):
            if _kl_args is None:
                return original_forward(*args, **kwargs)

            # Extract FA varlen kwargs for sequence packing
            fa_kwargs = {}
            for k in ('cu_seq_lens_q', 'cu_seq_lens_k', 'max_length_q', 'max_length_k'):
                if k in kwargs:
                    fa_kwargs[k] = kwargs.pop(k)

            outputs = self_model.model(
                input_ids=kwargs.get('input_ids', args[0] if args else None),
                attention_mask=kwargs.get('attention_mask'),
                position_ids=kwargs.get('position_ids'),
                use_cache=False,
                **fa_kwargs,
            )
            hidden_states = outputs[0]
            lm_weight = self_model.lm_head.weight

            mode = _kl_args['mode']
            chunk_size = _kl_args.get('chunk_size', 1024)
            return_values = bool(_kl_args.get('return_values', False))
            if mode == 'return_hidden':
                return {'hidden_states': hidden_states, 'lm_head_weight': lm_weight}
            if mode in ('forward_kl', 'reverse_kl', 'reverse_kl_rollout_student_topk', 'skewed_kl'):
                return chunked_lm_head_gather(hidden_states, lm_weight,
                                              _kl_args['indices'],
                                              chunk_size=chunk_size)
            elif mode == 'thunlp_opd_default_loss':
                return chunked_lm_head_gather(hidden_states[:, :-1], lm_weight,
                                              _kl_args['indices'][:, 1:],
                                              chunk_size=chunk_size)
            elif mode in ('multi_sample_policy_gradient_kl', 'multi_sample_forward_kl'):
                return chunked_lm_head_gather(hidden_states[:, :-1], lm_weight,
                                              _kl_args['indices'][:, 1:],
                                              chunk_size=chunk_size)
            elif mode in ('token_level_kl', 'policy_gradient_kl'):
                input_ids = kwargs.get('input_ids', args[0] if args else None)
                target_ids = input_ids[:, 1:].unsqueeze(-1)
                student_token_logps = chunked_lm_head_gather(
                    hidden_states[:, :-1], lm_weight, target_ids,
                    chunk_size=chunk_size)
                if return_values:
                    if not hasattr(self_model, "value_head"):
                        raise AttributeError(
                            "Chunked KL forward requested values, but model has no value_head.")
                    values = self_model.value_head(hidden_states).squeeze(-1)
                    return {
                        "student_token_logps": student_token_logps,
                        "values": values,
                    }
                return student_token_logps
            else:
                logits = self_model.lm_head(hidden_states)
                return logits

        model.forward = types.MethodType(_patched_forward, model)
        return True

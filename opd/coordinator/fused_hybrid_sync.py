"""Fused synchronous OPD coordinator with colocated FSDP trainer + vLLM rollout.

This scheduler keeps the OPD teacher as the existing independent service while
student FSDP ranks also host a vLLM external-launcher shard.  Weight refresh is
performed inside the student rank group with bucketed checkpoint-format loads,
so the path never creates the unsafe duplicate-GPU trainer-to-rollout NCCL
communicator used by process-separated rollout workers.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import random
import socket
import time
from collections.abc import Iterable
from types import SimpleNamespace

import numpy as np
import torch

from opd.coordinator.base import CoordinatorBase
from opd.coordinator.mode import build_step_off_backend
from opd.coordinator.opd_mode import OPDMode
from opd.worker.proxy import QueueTrainerProxy
from opd.worker.teacher.client import TeacherClient


class FusedHybridOPDMode(OPDMode):
    """OPD mode variant that sends generation to the fused trainer ranks."""

    @classmethod
    def from_coordinator(cls, coordinator):
        obj = cls(
            rollout_proxy=None,
            teacher_client=getattr(coordinator, "teacher_client", None),
            trainer_proxy=getattr(coordinator, "trainer_proxy", None),
            tracer=getattr(coordinator, "tracer", None),
            opd_config=getattr(coordinator, "opd_config", None),
            logger=getattr(coordinator, "logger", None),
            tokenizer=coordinator._init_tokenizer(),
            n_trainer_gpus=getattr(coordinator, "_n_trainer_gpus", 1),
            ray_teacher_actor=getattr(coordinator, "_ray_teacher_actor", None),
            teacher_trace_info=getattr(coordinator, "_teacher_trace_info", {}),
            teacher_artifact_queue=None,
        )
        teacher_port = getattr(coordinator, "teacher_port", None)
        obj._fused_teacher_addr = (
            f"tcp://127.0.0.1:{teacher_port}" if teacher_port is not None else None
        )
        return obj

    @property
    def _uses_cached_dp_rollout(self) -> bool:
        if self._opd_config is None:
            return False
        fhs = getattr(getattr(self._opd_config, "pipeline", None), "fused_hybrid_sync", None)
        return (
            fhs is not None
            and getattr(fhs, "rollout_parallelism", "spmd_tp") == "data_parallel"
            and not self.uses_direct_teacher_artifacts
        )

    def _cached_teacher_options(self) -> dict:
        opts = {
            "kl_loss_mode": self._opd_config.algorithm.opd.kl_loss_mode,
            "uses_rollout_support_topk": bool(self._uses_rollout_support_topk),
            "uses_multi_sample_policy_gradient_kl": bool(
                self._uses_multi_sample_policy_gradient_kl
            ),
            "uses_multi_sample_forward_kl": bool(self._uses_multi_sample_forward_kl),
            "uses_mof_mc_candidates": bool(self._uses_mof_mc_candidates),
            "uses_mof_generated_only": bool(self._uses_mof_generated_only),
            "uses_mof_eos_aware": bool(self._uses_mof_eos_aware),
            "pg_kl_n_total_samples": int(
                self._opd_config.algorithm.opd.pg_kl_n_total_samples
            ),
        }
        if self._uses_mof_eos_aware:
            opts["eos_token_id"] = int(self._mof_eos_token_id())
        return opts

    def async_generate(self, batch_dict):
        response_topk_k = (
            int(self._opd_config.algorithm.opd.rollout_student_topk_k)
            if self._uses_rollout_support_topk
            else 0
        )
        opts = {
            "return_logprobs": False,
            "response_topk_k": response_topk_k,
            "max_response_length": self._opd_config.data.max_response_length,
        }
        # OPDMode computes these as properties; keep the fused command explicit.
        if self._opd_config.algorithm.opd.kl_loss_mode == "policy_gradient_kl":
            opts["return_logprobs"] = bool(self._opd_config.algorithm.opd.use_importance_sampling)
        if self._uses_multi_sample_policy_gradient_kl or self._uses_mof_mc_candidates:
            opts["mc_n_total_samples"] = int(
                self._opd_config.algorithm.opd.pg_kl_n_total_samples
            )
        command = "hybrid_generate_cached" if self._uses_cached_dp_rollout else "hybrid_generate"
        self.trainer_proxy.submit_command_async(
            command,
            {"batch": batch_dict, "options": opts},
        )

    def wait_generate(self):
        result = self.trainer_proxy.collect_command()
        if not isinstance(result, dict):
            raise RuntimeError(
                f"fused hybrid generate returned non-dict result: {type(result).__name__}"
            )
        if result.get("_hybrid_cached") or "full_token_lists" in result:
            return result
        raise RuntimeError(
            "fused hybrid generate returned a non-generation result "
            f"(keys={sorted(result.keys())}); this usually means a prior "
            "trainer command result was not drained before rollout generation"
        )

    def async_teacher(self, gen_output, batch=None):
        if not (isinstance(gen_output, dict) and gen_output.get("_hybrid_cached")):
            return super().async_teacher(gen_output, batch=batch)
        teacher_addr = getattr(self, "_fused_teacher_addr", None)
        if not teacher_addr:
            raise RuntimeError("cached fused-hybrid teacher scoring requires teacher address")
        cache_id = gen_output["cache_id"]
        ack = self.trainer_proxy.submit_command(
            "hybrid_start_teacher_cached",
            {
                "cache_id": cache_id,
                "teacher_addr": teacher_addr,
                "teacher_options": self._cached_teacher_options(),
            },
        )
        if not isinstance(ack, dict) or ack.get("status") != "teacher_started":
            raise RuntimeError(f"cached teacher start failed: {ack!r}")

        def _resolve():
            return self.trainer_proxy.submit_command(
                "hybrid_resolve_teacher_cached",
                {"cache_id": cache_id},
            )

        return SimpleNamespace(get=_resolve)

    def async_train(self, gen_output, teacher_output):
        if (
            isinstance(gen_output, dict)
            and gen_output.get("_hybrid_cached")
            and isinstance(teacher_output, dict)
            and teacher_output.get("_hybrid_cached_teacher")
        ):
            self.trainer_proxy.submit_command_async(
                "hybrid_train_cached",
                {"cache_id": gen_output["cache_id"], "send_mono": time.monotonic()},
            )
            return
        return super().async_train(gen_output, teacher_output)


class FusedHybridRuntimeGroup:
    """Coordinator-side phase/version controller for fused student ranks."""

    def __init__(self, coordinator: CoordinatorBase):
        self.coordinator = coordinator
        self.rollout_version = 0
        self.phase = "init"

    def _cmd(self, name: str, payload: dict | None = None):
        if payload is None:
            payload = {}
        return self.coordinator.trainer_proxy.submit_command(name, payload)

    def enter_rollout_mode(self, actor_version: int) -> dict:
        self.phase = "rollout"
        print(f"[FusedHybrid] phase=rollout actor_version={actor_version}", flush=True)
        result = self._cmd("hybrid_rollout_mode", {"actor_version": int(actor_version)})
        self.rollout_version = int(result.get("rollout_version", actor_version))
        if self.rollout_version != int(actor_version):
            raise RuntimeError(
                f"stale fused rollout version: rollout={self.rollout_version}, "
                f"actor={actor_version}"
            )
        return result.get("metrics", {}) or {}

    def quiesce_rollout(self) -> None:
        self.phase = "rollout_quiesce"
        print("[FusedHybrid] sleep_rollout reason=after_generate", flush=True)
        self._cmd("hybrid_release_rollout", {"reason": "after_generate"})
        print("[FusedHybrid] phase=rollout_quiesced", flush=True)

    def enter_trainer_mode(self) -> None:
        self.phase = "trainer"
        print("[FusedHybrid] phase=trainer", flush=True)
        self._cmd("hybrid_prepare_train", {})

    def refresh_after_train(self, actor_version: int) -> dict:
        self.phase = "sync"
        print(f"[FusedHybrid] phase=sync actor_version={actor_version}", flush=True)
        result = self._cmd("hybrid_refresh_weights", {"actor_version": int(actor_version)})
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        self.rollout_version = int(metrics.get("fused_hybrid_rollout_version", actor_version))
        return metrics


class FusedHybridSyncScheduler:
    """Strict rollout → teacher → train → bucketed refresh loop."""

    def __init__(self, *, backend, runtime_group: FusedHybridRuntimeGroup,
                 total_steps: int, test_freq: int = -1, save_freq: int = 0,
                 tracer=None, tid_config=None, n_mini_per_step: int = 1,
                 refresh_policy: str = "after_train"):
        self.backend = backend
        self.runtime_group = runtime_group
        self.total_steps = int(total_steps)
        self.test_freq = int(test_freq)
        self.save_freq = int(save_freq)
        self.tracer = tracer
        self._n_mini_per_step = int(n_mini_per_step)
        self.refresh_policy = refresh_policy
        tc = tid_config or {}
        self.TID_ROLLOUT = tc.get("rollout", 10)
        self.TID_TEACHER = tc.get("teacher", 11)
        self.TID_TRAIN = tc.get("train", 12)
        self.TID_PIPELINE = tc.get("pipeline", 14)
        self.TID_EVAL = tc.get("eval", 13)

    def _next_batch(self):
        try:
            _, batch = next(self._data_iter)
            return batch
        except StopIteration:
            return None

    def _emit(self, name: str, cat: str, tid: int, start: float, end: float,
              args: dict | None = None):
        if self.tracer is not None:
            self.tracer.emit(name, cat=cat, tid=tid,
                             t_start=start, t_end=end, args=args)

    @staticmethod
    def _token_trace_args(payload: dict) -> dict:
        args = {}
        if "prompt_lengths" in payload:
            args["prompt_tok"] = int(payload["prompt_lengths"].sum())
            args["n_seqs"] = len(payload.get("prompt_lengths", []))
        if "response_lengths" in payload:
            args["gen_tok"] = int(payload["response_lengths"].sum())
            args.setdefault("n_seqs", len(payload.get("response_lengths", [])))
        return args

    def run(self, data_iter: Iterable, resume_step: int = 0) -> int:
        self._data_iter = data_iter
        be = self.backend
        runtime = self.runtime_group
        actor_version = resume_step * self._n_mini_per_step
        runtime.rollout_version = actor_version
        last_step = resume_step

        for step in range(resume_step + 1, self.total_steps + 1):
            batch = self._next_batch()
            if batch is None:
                break
            last_step = step
            timing: dict = {}
            pending_metrics: dict = {}

            t0 = time.monotonic()
            refresh_metrics = runtime.enter_rollout_mode(actor_version)
            pending_metrics.update(refresh_metrics)
            be.async_generate(batch)
            gen_out = be.wait_generate()
            gen_timing = gen_out.pop("timing", {})
            timing["generate_seconds"] = gen_timing.get("generate_seconds", 0)
            gen_args = {
                "step": step,
                "actor_version": actor_version,
                "phase": "rollout_generate",
                "fused_hybrid": True,
            }
            gen_args.update(self._token_trace_args(gen_out))
            self._emit("generate", "rollout", self.TID_ROLLOUT,
                       t0, time.monotonic(), gen_args)

            teacher_fut = be.async_teacher(gen_out)
            runtime.quiesce_rollout()

            teacher_out = be.resolve_teacher(teacher_fut, timing)
            if teacher_out is None:
                print(f"[FusedHybrid] step={step} teacher failed; skipping train", flush=True)
                continue

            train_start = time.monotonic()
            runtime.enter_trainer_mode()
            be.async_train(gen_out, teacher_out)
            train_result = be.wait_train()
            metrics = (train_result or {}).setdefault("metrics", {})
            for source in (gen_out, teacher_out):
                if isinstance(source, dict):
                    for key in (
                        "fused_hybrid_dp_cached_generation",
                        "fused_hybrid_dp_cached_teacher",
                        "fused_hybrid_dp_cached_teacher_prompts",
                        "fused_hybrid_dp_cached_teacher_tokens",
                    ):
                        if key in source:
                            metrics[key] = source[key]
            train_timing = metrics.get("timing", {})
            train_end = time.monotonic()
            train_args = {
                "step": step,
                "phase": "trainer_train",
                "fused_hybrid": True,
            }
            train_args.update(self._token_trace_args(gen_out))
            if "gen_tok" in train_args:
                train_args["resp_tok"] = train_args.pop("gen_tok")
            self._emit(
                "train",
                "train",
                self.TID_TRAIN,
                train_timing.get("mono_start", train_start),
                train_timing.get("mono_end", train_end),
                train_args,
            )
            n_optim = int(metrics.get("n_optim_steps") or 1)
            actor_version += n_optim

            if self.refresh_policy == "after_train":
                sync_start = time.monotonic()
                sync_metrics = runtime.refresh_after_train(actor_version)
                pending_metrics.update(sync_metrics)
                timing["sync_seconds"] = float(
                    sync_metrics.get("fused_hybrid_weight_update_duration_s", 0.0)
                )
                self._emit(
                    "sync_weights",
                    "sync",
                    self.TID_PIPELINE,
                    sync_start,
                    time.monotonic(),
                    {
                        "step": step,
                        "actor_version": actor_version,
                        "phase": "actor_to_rollout_bucketed_refresh",
                        "fused_hybrid": True,
                        "weight_update_backend": sync_metrics.get(
                            "fused_hybrid_weight_update_backend"
                        ),
                    },
                )
            else:
                timing["sync_seconds"] = float(
                    pending_metrics.get("fused_hybrid_weight_update_duration_s", 0.0)
                )

            metrics.update(pending_metrics)
            metrics["fused_hybrid_actor_version"] = actor_version
            metrics["fused_hybrid_rollout_version"] = runtime.rollout_version
            metrics["fused_hybrid_teacher_independent"] = True
            be.log_train_step(step, timing, gen_out, train_result)

            if self.save_freq > 0 and step % self.save_freq == 0:
                be.save_checkpoint(step)
                wait_checkpoint_save = getattr(be, "wait_checkpoint_save", None)
                if wait_checkpoint_save is not None:
                    result = wait_checkpoint_save()
                    status = result.get("status") if isinstance(result, dict) else None
                    print(
                        f"[FusedHybrid] checkpoint_save_result_drained "
                        f"step={step} status={status or 'unknown'}",
                        flush=True,
                    )
            if self.test_freq > 0 and step % self.test_freq == 0:
                if self.tracer is not None:
                    with self.tracer.span("eval", cat="eval", tid=self.TID_EVAL):
                        be.evaluate(step)
                else:
                    be.evaluate(step)

        return last_step


class FusedHybridSyncCoordinator(CoordinatorBase):
    """Local OPD coordinator for the fused_hybrid_sync scheduler."""

    def start(self):
        oc = self.opd_config
        if oc.deterministic:
            random.seed(oc.seed)
            np.random.seed(oc.seed)
            torch.manual_seed(oc.seed)
            print(f"[FusedHybrid] Deterministic mode: seed={oc.seed}", flush=True)

        self._setup_env()
        lora_dc = oc.trainer.lora
        if lora_dc is not None:
            raise ValueError("fused_hybrid_sync does not support LoRA in the MVP")
        lora_cfg = None
        self._native_lora = False
        self._apply_rollout_logprob_flags(oc.algorithm.opd.kl_loss_mode)

        if oc.pipeline.deployment != "local":
            raise ValueError("fused_hybrid_sync currently supports local deployment only")
        ctx = mp.get_context("spawn")

        student_gpu_list = self._gpu_ids("trainer").split(",")
        teacher_gpu_set = self._start_teacher_local(ctx, student_gpu_list)
        _ = teacher_gpu_set  # teacher remains independent; validation owns overlap policy.

        self._start_trainer_local(ctx, lora_cfg)
        self.trainer_proxy = QueueTrainerProxy(
            cmd_queue=self.trainer_cmd_queue,
            result_queue=self.trainer_result_queue,
            proc=self.trainer_proc,
            fsdp_procs=getattr(self, "_trainer_fsdp_procs", []),
        )

        print("[FusedHybrid] Waiting for fused student ranks to initialize vLLM...", flush=True)
        init = self.trainer_proxy.submit_command("hybrid_init_rollout")
        if not isinstance(init, dict) or init.get("status") != "hybrid_rollout_ready":
            raise RuntimeError(f"fused hybrid rollout init failed: {init!r}")
        self.hybrid_runtime = FusedHybridRuntimeGroup(self)

        if self._needs_teacher():
            self.teacher_client = TeacherClient(
                f"tcp://127.0.0.1:{self.teacher_port}", n_workers=1,
            )
        else:
            self.teacher_client = None

        host = socket.gethostname()
        self._trainer_trace_info = {"host": host, "gpu_ids": self._gpu_ids("trainer")}
        self._teacher_trace_info = {"host": host, "gpu_ids": self._gpu_ids("teacher")}
        self._rollout_worker_info = [
            {"host": host, "gpu_ids": gpu_id, "fused_hybrid": True}
            for gpu_id in student_gpu_list
        ]
        print("[FusedHybrid] All workers ready.", flush=True)
        atexit.register(self.shutdown)

    def run(self):
        mode_cls = getattr(self, "_mode_cls", None)
        if mode_cls is None:
            mode_cls = FusedHybridOPDMode
        self._mode = mode_cls.from_coordinator(self)

        resume_step, test_freq, eval_modes, val_before_train = self._prepare_run()
        backend = build_step_off_backend(self._mode, self)
        data_iter = iter(self._data_iterator())
        self._skip_data_for_resume(data_iter, resume_step, label="FusedHybrid")
        if hasattr(self._mode, "on_resume_skip_complete"):
            self._mode.on_resume_skip_complete()

        oc = self.opd_config
        mini_bs = oc.trainer.mini_batch_size or 0
        n_mini = max(self.batch_size // mini_bs, 1) if mini_bs > 0 else 1
        sched_test_freq = -1 if not (set(eval_modes) & {"inline"}) else test_freq

        scheduler = FusedHybridSyncScheduler(
            backend=backend,
            runtime_group=self.hybrid_runtime,
            total_steps=self.total_steps,
            test_freq=sched_test_freq,
            save_freq=self.save_freq,
            tracer=self.tracer,
            tid_config={
                "rollout": self.TID_ROLLOUT,
                "teacher": self.TID_TEACHER,
                "train": self.TID_TRAIN,
                "pipeline": self.TID_PIPELINE,
                "eval": self.TID_EVAL,
            },
            n_mini_per_step=n_mini,
            refresh_policy=oc.pipeline.fused_hybrid_sync.refresh_policy,
        )
        step = scheduler.run(data_iter, resume_step)
        print(f"[FusedHybrid] Done ({min(step, self.total_steps)} steps).", flush=True)

"""Async-sample step-off coordinator with strict logical-batch dispatch.

This scheduler keeps trainer-visible step-off semantics (full logical batches,
strict train order, same optimizer-step accounting) while using the streaming
vLLM/AsyncLLM rollout worker so individual completed samples can be teacher
scored immediately.
"""

from __future__ import annotations

import concurrent.futures
import queue
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

import torch

from opd.coordinator.base import CoordinatorBase
from opd.coordinator.opd_mode import OPDMode
from opd.data.batch_utils import pad_teacher, split_gen_teacher, stack_gen_output
from opd.trainer.teacher_artifact_buffer import estimate_tensor_bytes


@dataclass
class _LogicalBatch:
    logical_batch_id: int
    train_step: int
    expected: int
    gen_wv: int
    samples: dict[int, dict] = field(default_factory=dict)
    scored: dict[int, dict] = field(default_factory=dict)
    submitted_at: float = field(default_factory=time.monotonic)
    first_rollout_start_at: float | None = None
    last_rollout_end_at: float | None = None

    def add_sample(self, sample: dict) -> None:
        idx = int(sample["sample_in_batch_idx"])
        if idx in self.samples:
            raise RuntimeError(
                f"duplicate rollout sample for logical_batch={self.logical_batch_id} "
                f"idx={idx}"
            )
        self.samples[idx] = sample
        start = sample.get("rollout_mono_start")
        end = sample.get("rollout_mono_end")
        if isinstance(start, (int, float)):
            start = float(start)
            self.first_rollout_start_at = (
                start if self.first_rollout_start_at is None
                else min(self.first_rollout_start_at, start)
            )
        if isinstance(end, (int, float)):
            end = float(end)
            self.last_rollout_end_at = (
                end if self.last_rollout_end_at is None
                else max(self.last_rollout_end_at, end)
            )

    def add_scored(self, sample: dict) -> None:
        idx = int(sample["sample_in_batch_idx"])
        if idx in self.scored:
            raise RuntimeError(
                f"duplicate scored sample for logical_batch={self.logical_batch_id} "
                f"idx={idx}"
            )
        self.scored[idx] = sample

    @property
    def rollout_complete(self) -> bool:
        return len(self.samples) == self.expected

    @property
    def scored_complete(self) -> bool:
        return len(self.scored) == self.expected

    def ordered_scored(self) -> list[dict]:
        missing = [i for i in range(self.expected) if i not in self.scored]
        if missing:
            raise RuntimeError(
                f"logical_batch={self.logical_batch_id} missing scored indices {missing}"
            )
        return [self.scored[i] for i in range(self.expected)]

    def ordered_samples(self) -> list[dict]:
        missing = [i for i in range(self.expected) if i not in self.samples]
        if missing:
            raise RuntimeError(
                f"logical_batch={self.logical_batch_id} missing sample indices {missing}"
            )
        return [self.samples[i] for i in range(self.expected)]


@dataclass
class _TeacherFuture:
    future: concurrent.futures.Future
    sample_meta: list[tuple[int, int, Any]]
    submitted_at: float
    n_prompts: int
    total_tok: int


class _TeacherMicrobatcher:
    """Coordinator-owned metadata-preserving teacher microbatcher."""

    ARTIFACT_PUT_TIMEOUT_S = 120.0

    def __init__(self, *, score_fn, scoring_batch_size: int, tracer=None,
                 tid_teacher: int = 11, teacher_trace_info: dict | None = None,
                 artifact_queue=None, direct_transport: bool = False):
        self.score_fn = score_fn
        self.scoring_batch_size = max(int(scoring_batch_size or 1), 1)
        self.tracer = tracer
        self.tid_teacher = tid_teacher
        self.teacher_trace_info = teacher_trace_info or {}
        self.artifact_queue = artifact_queue
        self.direct_transport = bool(direct_transport)
        self.queue: deque[dict] = deque()
        self.pending: list[_TeacherFuture] = []
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stepoff-teacher"
        )

    def enqueue(self, sample: dict) -> None:
        self.queue.append(sample)
        if len(self.queue) >= self.scoring_batch_size:
            self.flush(self.scoring_batch_size, reason="scoring_batch")

    def flush(self, n: int | None = None, *, reason: str = "flush") -> None:
        if not self.queue:
            return
        if n is None:
            n = len(self.queue)
        batch = [self.queue.popleft() for _ in range(min(n, len(self.queue)))]
        sample_meta = [self._meta(s) for s in batch]
        n_prompts = sum(len(s.get("full_token_lists", [])) for s in batch) or len(batch)
        total_tok = sum(len(tl) for s in batch for tl in s.get("full_token_lists", []))
        t_submit = time.monotonic()
        if self.tracer is not None:
            self.tracer.instant(
                "teacher_microbatch_submit",
                cat="pipeline",
                tid=self.tid_teacher,
                args={"n_prompts": n_prompts, "reason": reason,
                      "logical_batch_ids": sorted({m[0] for m in sample_meta})},
            )
        fut = self.pool.submit(self._score_batch, batch, sample_meta)
        self.pending.append(_TeacherFuture(
            future=fut,
            sample_meta=sample_meta,
            submitted_at=t_submit,
            n_prompts=n_prompts,
            total_tok=total_tok,
        ))

    def flush_blocking_batch(self, logical_batch_id: int) -> None:
        """Flush queued samples for a train-blocking logical batch promptly."""
        if not any(s.get("logical_batch_id") == logical_batch_id for s in self.queue):
            return
        # Keep row order deterministic: flush the current queue prefix through the
        # last sample for the blocking batch, preserving metadata by row.
        last = 0
        for i, sample in enumerate(self.queue):
            if sample.get("logical_batch_id") == logical_batch_id:
                last = i
        self.flush(last + 1, reason="train_blocking")

    def poll(self) -> list[dict]:
        ready: list[dict] = []
        keep: list[_TeacherFuture] = []
        for rec in self.pending:
            if rec.future.done():
                samples, teacher_t0, teacher_t1 = rec.future.result()
                self._validate_result(rec.sample_meta, samples)
                if self.tracer is not None:
                    ts = teacher_t0 or rec.submitted_at
                    te = teacher_t1 or time.monotonic()
                    self.tracer.emit(
                        "teacher_score",
                        cat="teacher",
                        tid=self.tid_teacher,
                        t_start=ts,
                        t_end=te,
                        args={"n_prompts": rec.n_prompts,
                              "total_tok": rec.total_tok,
                              "logical_batch_ids": sorted({m[0] for m in rec.sample_meta}),
                              **self.teacher_trace_info},
                    )
                ready.extend(samples)
            else:
                keep.append(rec)
        self.pending = keep
        return ready

    def drain(self) -> list[dict]:
        self.flush(None, reason="drain")
        ready: list[dict] = []
        while self.pending:
            done = self.poll()
            if done:
                ready.extend(done)
            else:
                time.sleep(0.01)
        return ready

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=False)

    @staticmethod
    def _meta(sample: dict) -> tuple[int, int, Any]:
        return (
            int(sample["logical_batch_id"]),
            int(sample["sample_in_batch_idx"]),
            sample.get("sample_seq_id"),
        )

    def _score_batch(self, batch: list[dict], sample_meta: list[tuple[int, int, Any]]):
        t0 = time.monotonic()
        teacher_t0, teacher_t1 = self.score_fn(batch)
        t1 = time.monotonic()
        # score_fn mutates samples in place. Return teacher-side timestamps when
        # available; otherwise coordinator-side wall times.
        return batch, teacher_t0 or t0, teacher_t1 or t1

    def _send_artifacts_to_trainer(self, samples: list[dict]) -> list[dict]:
        """Send scored teacher artifacts to the trainer-side artifact queue.

        Direct-transport async step-off deliberately delays this call until the
        corresponding train command is dispatched.  Sending artifacts when
        teacher scoring finishes can let future batches fill the mp.Queue while
        the trainer is still busy with the current step, which turns large
        hidden-state artifacts into a silent scheduler deadlock.
        """
        if self.artifact_queue is None:
            raise RuntimeError("direct teacher artifact queue is not configured")
        acks = []
        for sample in samples:
            payload = self._extract_teacher_payload(sample)
            n_bytes = estimate_tensor_bytes(payload)
            envelope = {
                "schema_version": 1,
                "logical_batch_id": int(sample["logical_batch_id"]),
                "sample_in_batch_idx": int(sample["sample_in_batch_idx"]),
                "sample_seq_id": sample.get("sample_seq_id"),
                "train_step": int(sample.get("logical_batch_id", 0)) + 1,
                "gen_weight_version": int(sample.get("gen_wv", sample.get("weight_version", 0))),
                "n_expected": int(sample.get("n_expected", 0) or 0),
                "payload_kind": self._payload_kind(payload),
                "shape": self._shape_meta(payload),
                "dtype": self._dtype_meta(payload),
                "position_spec": {"alignment": "existing_pad_teacher"},
                "n_tokens": sum(len(tl) for tl in sample.get("full_token_lists", [])),
                "n_bytes": n_bytes,
                "payload": payload,
            }
            try:
                self.artifact_queue.put(
                    envelope,
                    timeout=self.ARTIFACT_PUT_TIMEOUT_S,
                )
            except queue.Full as exc:
                raise RuntimeError(
                    "timed out sending teacher artifact to trainer "
                    f"after {self.ARTIFACT_PUT_TIMEOUT_S:.0f}s: "
                    f"logical_batch_id={envelope['logical_batch_id']} "
                    f"sample_in_batch_idx={envelope['sample_in_batch_idx']} "
                    f"n_expected={envelope['n_expected']} "
                    f"n_bytes={n_bytes} payload_kind={envelope['payload_kind']}"
                ) from exc
            if self.tracer is not None:
                self.tracer.instant(
                    "teacher_artifact_send",
                    cat="teacher",
                    tid=self.tid_teacher,
                    args={"logical_batch_id": envelope["logical_batch_id"],
                          "sample_in_batch_idx": envelope["sample_in_batch_idx"],
                          "n_bytes": n_bytes,
                          "payload_kind": envelope["payload_kind"]},
                )
            acks.append({
                "logical_batch_id": envelope["logical_batch_id"],
                "sample_in_batch_idx": envelope["sample_in_batch_idx"],
                "sample_seq_id": envelope["sample_seq_id"],
                "gen_wv": envelope["gen_weight_version"],
                "weight_version": envelope["gen_weight_version"],
            })
        return acks

    def send_artifacts_to_trainer(self, samples: list[dict]) -> list[dict]:
        return self._send_artifacts_to_trainer(samples)

    @staticmethod
    def _extract_teacher_payload(sample: dict) -> dict:
        keys = (
            "teacher_topk_logps", "teacher_topk_indices", "teacher_token_logps",
            "teacher_query_logprobs_response", "teacher_mc_indices_response",
            "teacher_hidden_states", "teacher_hidden_token_ids", "teacher_hidden_metadata",
        )
        payload = {k: sample[k] for k in keys if k in sample}
        if "teacher_hidden_metadata" in payload and payload["teacher_hidden_metadata"]:
            meta0 = payload["teacher_hidden_metadata"][0]
            if isinstance(meta0, dict):
                payload["teacher_hidden_dtype"] = meta0.get("teacher_hidden_dtype")
                payload["teacher_hidden_semantics"] = meta0.get("teacher_hidden_semantics")
                payload["hidden_size"] = meta0.get("hidden_size")
        if not payload:
            raise RuntimeError("teacher sample did not contain teacher artifacts")
        return payload

    @staticmethod
    def _payload_kind(payload: dict) -> str:
        if "teacher_hidden_states" in payload:
            return "hidden_states"
        if "teacher_query_logprobs_response" in payload:
            if "teacher_mc_indices_response" in payload:
                return "mc_query_logprobs"
            return "support_query_logprobs"
        if "teacher_token_logps" in payload:
            return "token_logprobs"
        return "topk_logprobs"

    @staticmethod
    def _shape_meta(payload: dict) -> dict:
        meta = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                meta[key] = list(value.shape)
            elif isinstance(value, list):
                meta[key] = [
                    list(v.shape) if isinstance(v, torch.Tensor) else None
                    for v in value
                ]
        return meta

    @staticmethod
    def _dtype_meta(payload: dict) -> dict:
        meta = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                meta[key] = str(value.dtype)
            elif isinstance(value, list):
                meta[key] = [
                    str(v.dtype) if isinstance(v, torch.Tensor) else None
                    for v in value
                ]
        return meta

    def _validate_result(self, expected_meta: list[tuple[int, int, Any]], samples: list[dict]) -> None:
        if len(samples) != len(expected_meta):
            raise RuntimeError(
                f"teacher row count mismatch: got {len(samples)}, expected {len(expected_meta)}"
            )
        seen = set()
        for sample, expected in zip(samples, expected_meta, strict=True):
            actual = self._meta(sample)
            if actual != expected:
                raise RuntimeError(
                    f"teacher metadata mismatch: got {actual}, expected {expected}"
                )
            if actual in seen:
                raise RuntimeError(f"duplicate teacher metadata {actual}")
            seen.add(actual)


class StepOffAsyncScheduler:
    """Strict logical-batch step-off scheduler over AsyncLLM sample streaming."""

    WATCHDOG_INTERVAL_S = 60.0
    WATCHDOG_MAX_LOGS_PER_WAIT = 3

    def __init__(self, coordinator: "StepOffAsyncCoordinator", mode: OPDMode,
                 *, step_off: int, total_steps: int, test_freq: int = -1,
                 save_freq: int = 0, n_mini_per_step: int = 1):
        self.c = coordinator
        self.mode = mode
        self.step_off = int(step_off)
        self.capacity = max(self.step_off, 1)
        self.total_steps = int(total_steps)
        self.test_freq = test_freq
        self.save_freq = save_freq
        self.n_mini_per_step = max(int(n_mini_per_step), 1)
        self.tracer = coordinator.tracer
        self._teacher = _TeacherMicrobatcher(
            score_fn=mode.make_stream_score_fn(coordinator.teacher_client),
            scoring_batch_size=(coordinator.opd_config.teacher.scoring_batch_size or 8),
            tracer=self.tracer,
            tid_teacher=coordinator.TID_TEACHER,
            teacher_trace_info=getattr(coordinator, "_teacher_trace_info", {}),
            artifact_queue=getattr(coordinator, "teacher_artifact_queue", None),
            direct_transport=bool(
                getattr(coordinator, "_uses_direct_teacher_artifacts", lambda: False)()
            ),
        )
        self._train_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stepoff-train"
        )

    def run(self, data_iter, resume_step: int = 0) -> int:
        c = self.c
        sample_stream = c.rollout_proxy.sample_stream()
        prompt_sink = c.rollout_proxy.prompt_sink()

        self._start_autonomous_workers()

        self._data_iter = data_iter
        self._data_exhausted = False
        self._next_submit_id = resume_step
        self._next_train_id = resume_step
        self._weight_version = resume_step * self.n_mini_per_step
        self._last_n_optim_steps = self.n_mini_per_step
        self._inflight: OrderedDict[int, _LogicalBatch] = OrderedDict()
        self._pending_train: dict | None = None
        self._train_future: concurrent.futures.Future | None = None
        self._last_train_result: dict | None = None
        self._last_train_wait_log = 0.0
        self._last_drain_wait_log = 0.0
        self._train_wait_log_count = 0
        self._drain_wait_log_count = 0
        self._sample_stream = sample_stream
        self._prompt_sink = prompt_sink

        try:
            while self._next_train_id < self.total_steps:
                self._fill_capacity()
                if self._maybe_finish_train():
                    continue
                if (self._data_exhausted and not self._inflight
                        and self._pending_train is None):
                    break
                progressed = False
                progressed |= self._collect_one_sample(timeout=0.05)
                if self._maybe_finish_train():
                    continue
                progressed |= self._collect_teacher_ready()
                progressed |= self._maybe_dispatch_train()
                progressed |= self._maybe_finish_train()
                if not progressed:
                    self._maybe_log_train_wait()
                    # If the oldest batch is only waiting for a partial teacher
                    # microbatch, score it now instead of waiting for unrelated samples.
                    self._flush_train_blocking_teacher()
                    time.sleep(0.01)

            self._teacher.drain()
            self._collect_teacher_ready()
            self._finish_pending_train_blocking()
            return self._next_train_id
        finally:
            try:
                c.rollout_proxy.exit_autonomous()
            except Exception:
                pass
            self._teacher.shutdown()
            self._train_pool.shutdown(wait=True, cancel_futures=False)

    # ------------------------------------------------------------------ #
    # rollout submission / collection                                     #
    # ------------------------------------------------------------------ #

    def _start_autonomous_workers(self) -> None:
        n_workers = self.c.rollout_proxy.n_workers
        seq_len = self.c.max_prompt_length
        empty = {
            "input_ids": torch.empty((0, seq_len), dtype=torch.long),
            "attention_mask": torch.empty((0, seq_len), dtype=torch.bool),
        }
        if self.c._need_student_logprobs:
            empty["return_logprobs"] = True
        if getattr(self.c, "_rollout_support_topk_k", 0) > 0:
            empty["response_topk_k"] = getattr(self.c, "_rollout_support_topk_k", 0)
        if getattr(self.c, "_mc_n_total_samples", 0) > 0:
            empty["mc_n_total_samples"] = getattr(self.c, "_mc_n_total_samples", 0)
        self.c.rollout_proxy.enter_autonomous([dict(empty) for _ in range(n_workers)])

    def _next_batch(self) -> dict | None:
        try:
            _, batch = next(self._data_iter)
            return batch
        except StopIteration:
            self._data_exhausted = True
            return None

    def _fill_capacity(self) -> None:
        while (not self._data_exhausted
               and self._next_submit_id < self.total_steps
               and self._rollout_slots_used() < self.capacity):
            batch = self._next_batch()
            if batch is None:
                return
            self._submit_logical_batch(batch)

    def _rollout_slots_used(self) -> int:
        """Count submitted logical batches that occupy step-off rollout slots.

        Classic step-off keeps ``step_off`` rollout batches ahead of the
        currently pending train. A logical batch that has already been
        dispatched to the trainer no longer occupies a rollout slot, even
        though we keep its bookkeeping record until train completion and weight
        sync. Excluding that pending-train batch preserves the original
        staleness semantics: steady-state stale optimizer steps are
        ``step_off * n_optim_steps``.

        For step_off=0, keep the pending train counted so no next rollout is
        submitted before the current train/sync completes.
        """
        if self.step_off <= 0 or self._pending_train is None:
            return len(self._inflight)
        pending_rec = self._pending_train.get("rec")
        pending_lbid = (
            pending_rec.logical_batch_id if pending_rec is not None else None
        )
        return sum(
            1 for lbid in self._inflight
            if lbid != pending_lbid
        )

    def _submit_logical_batch(self, batch: dict) -> None:
        lbid = self._next_submit_id
        self._next_submit_id += 1
        train_step = lbid + 1
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        expected = int(input_ids.size(0))
        rec = _LogicalBatch(lbid, train_step, expected, self._weight_version)
        self._inflight[lbid] = rec
        max_prompt_len = int(input_ids.size(1))
        ground_truths = batch.get("ground_truths", [])
        t0 = time.monotonic()
        for i in range(expected):
            mask = attention_mask[i].bool()
            ids = input_ids[i][mask].tolist()
            prompt = {
                "prompt_ids": ids,
                "prompt_len": len(ids),
                "pad_len": max_prompt_len - len(ids),
                "input_ids_row": input_ids[i].clone(),
                "return_logprobs": bool(self.c._need_student_logprobs),
                "logical_batch_id": lbid,
                "sample_in_batch_idx": i,
                "sample_seq_id": f"{lbid}:{i}:{uuid.uuid4().hex[:8]}",
                "gen_wv": self._weight_version,
            }
            if getattr(self.c, "_rollout_support_topk_k", 0) > 0:
                prompt["response_topk_k"] = getattr(self.c, "_rollout_support_topk_k", 0)
            if getattr(self.c, "_mc_n_total_samples", 0) > 0:
                prompt["mc_n_total_samples"] = getattr(self.c, "_mc_n_total_samples", 0)
            if i < len(ground_truths):
                prompt["ground_truth"] = ground_truths[i]
            while not self._prompt_sink.put(prompt, timeout=1.0):
                pass
        if self.tracer is not None:
            self.tracer.emit(
                "async_logical_batch_submit",
                cat="pipeline",
                tid=self.c.TID_PIPELINE,
                t_start=t0,
                t_end=time.monotonic(),
                args={"logical_batch_id": lbid, "train_step": train_step,
                      "n_seqs": expected, "gen_wv": self._weight_version,
                      "inflight_logical_batches": len(self._inflight)},
            )

    def _collect_one_sample(self, timeout: float) -> bool:
        sample = self._sample_stream.get_sample(timeout=timeout)
        if sample is None:
            return False
        for key in ("logical_batch_id", "sample_in_batch_idx", "sample_seq_id", "gen_wv"):
            if key not in sample:
                raise RuntimeError(f"async rollout sample missing metadata key '{key}'")
        lbid = int(sample["logical_batch_id"])
        idx = int(sample["sample_in_batch_idx"])
        if lbid not in self._inflight:
            raise RuntimeError(f"received sample for unknown logical_batch={lbid}")
        rec = self._inflight[lbid]
        if int(sample["gen_wv"]) != rec.gen_wv:
            raise RuntimeError(
                f"sample gen_wv mismatch for logical_batch={lbid}: "
                f"{sample['gen_wv']} != {rec.gen_wv}"
            )
        sample["weight_version"] = rec.gen_wv
        sample["n_expected"] = rec.expected
        rec.add_sample(sample)
        self._teacher.enqueue(sample)
        if self.tracer is not None:
            rollout_t0 = sample.get("rollout_mono_start")
            rollout_t1 = sample.get("rollout_mono_end")
            if isinstance(rollout_t0, (int, float)) and isinstance(rollout_t1, (int, float)):
                prompt_len = sample.get("prompt_len")
                response_len = sample.get("response_len")
                if prompt_len is None and "prompt_lengths" in sample:
                    prompt_len = int(sample["prompt_lengths"][0].item())
                if response_len is None and "response_lengths" in sample:
                    response_len = int(sample["response_lengths"][0].item())
                self.tracer.emit(
                    "async_generate_sample",
                    cat="rollout",
                    tid=self.c.TID_ROLLOUT_BASE + int(sample.get("worker_id", 0)),
                    t_start=float(rollout_t0),
                    t_end=float(rollout_t1),
                    args={"logical_batch_id": lbid, "sample_in_batch_idx": idx,
                          "gen_wv": rec.gen_wv,
                          "request_id": sample.get("rollout_request_id"),
                          "prompt_len": prompt_len,
                          "response_len": response_len,
                          "host": sample.get("host"),
                          "gpu_ids": sample.get("gpu_ids")},
                )
            self.tracer.instant(
                "async_rollout_sample_ready",
                cat="rollout",
                tid=self.c.TID_ROLLOUT,
                args={"logical_batch_id": lbid, "sample_in_batch_idx": idx,
                      "gen_wv": rec.gen_wv,
                      "n_ready": len(rec.samples), "n_expected": rec.expected,
                      "worker_id": sample.get("worker_id", 0)},
            )
        if rec.rollout_complete and self.tracer is not None:
            # Preserve the familiar aggregate "generate" slice for async
            # step-off traces. It spans coordinator submission through the last
            # completed sample for this logical batch, so it includes prompt
            # queueing and true AsyncLLM request time without reintroducing a
            # full-batch generate barrier.
            self.tracer.emit(
                "generate",
                cat="rollout",
                tid=self.c.TID_ROLLOUT,
                t_start=rec.submitted_at,
                t_end=rec.last_rollout_end_at or time.monotonic(),
                args={"logical_batch_id": lbid, "train_step": rec.train_step,
                      "n_seqs": rec.expected, "gen_wv": rec.gen_wv,
                      "mode": "async_stepoff",
                      "span_basis": "logical_batch_submit_to_last_sample"},
            )
            self.tracer.instant(
                "async_logical_batch_rollout_ready",
                cat="pipeline",
                tid=self.c.TID_PIPELINE,
                args={"logical_batch_id": lbid, "n_seqs": rec.expected},
            )
        return True

    # ------------------------------------------------------------------ #
    # teacher / train                                                     #
    # ------------------------------------------------------------------ #

    def _collect_teacher_ready(self) -> bool:
        ready = self._teacher.poll()
        for sample in ready:
            lbid = int(sample["logical_batch_id"])
            if lbid not in self._inflight:
                raise RuntimeError(f"scored sample for unknown logical_batch={lbid}")
            self._inflight[lbid].add_scored(sample)
            if self.tracer is not None:
                self.tracer.instant(
                    "async_sample_scored",
                    cat="teacher",
                    tid=self.c.TID_TEACHER,
                    args={"logical_batch_id": lbid,
                          "sample_in_batch_idx": int(sample["sample_in_batch_idx"]),
                          "n_scored": len(self._inflight[lbid].scored),
                          "n_expected": self._inflight[lbid].expected},
                )
        return bool(ready)

    def _flush_train_blocking_teacher(self) -> None:
        rec = self._inflight.get(self._next_train_id)
        if rec is not None and rec.rollout_complete and not rec.scored_complete:
            self._teacher.flush_blocking_batch(rec.logical_batch_id)

    def _maybe_dispatch_train(self) -> bool:
        if self._pending_train is not None:
            return False
        rec = self._inflight.get(self._next_train_id)
        if rec is None or not rec.scored_complete:
            return False
        samples = rec.ordered_scored()
        direct_transport = bool(
            getattr(self.c, "_uses_direct_teacher_artifacts", lambda: False)()
        )
        with self._span("logical_batch_ready"):
            if direct_transport:
                gen_out = stack_gen_output(rec.ordered_samples())
                teacher_out = None
            else:
                gen_out, teacher_out = split_gen_teacher(samples, pad_teacher_fn=pad_teacher)
        if self.tracer is not None:
            self.tracer.instant(
                "scored_buffer_ready",
                cat="pipeline",
                tid=self.c.TID_PIPELINE,
                args={"logical_batch_id": rec.logical_batch_id,
                      "n_seqs": rec.expected},
            )
        timing = {}
        if direct_transport:
            self.mode.async_train_direct_teacher_artifacts(
                gen_out,
                teacher_buffer_id=rec.logical_batch_id,
                logical_batch_id=rec.logical_batch_id,
                gen_weight_version=rec.gen_wv,
                expected_samples=rec.expected,
            )
            # Start the trainer command first so rank 0 is ready to drain the
            # artifact queue.  Only then send this logical batch's artifacts.
            # This avoids pre-loading future batches into the mp.Queue while the
            # trainer is still busy, which can deadlock large hidden-state
            # artifact runs via queue/pipe backpressure.
            sent = self._teacher.send_artifacts_to_trainer(samples)
            rec.scored = {
                int(sample["sample_in_batch_idx"]): sample
                for sample in sent
            }
            if self.tracer is not None:
                self.tracer.instant(
                    "train_dispatch_from_teacher_buffer",
                    cat="pipeline",
                    tid=self.c.TID_PIPELINE,
                    args={"logical_batch_id": rec.logical_batch_id,
                          "n_seqs": rec.expected,
                          "gen_wv": rec.gen_wv},
                )
        else:
            self.mode.async_train(gen_out, teacher_out)
        dispatch_mono = time.monotonic()
        self._train_future = self._train_pool.submit(self.mode.wait_train)
        staleness = self._weight_version - rec.gen_wv
        self._pending_train = {
            "rec": rec,
            "gen_out": gen_out,
            "timing": timing,
            "staleness": staleness,
            "dispatch_mono": dispatch_mono,
        }
        self._last_train_wait_log = dispatch_mono
        self._train_wait_log_count = 0
        return True

    def _maybe_finish_train(self) -> bool:
        if self._pending_train is None or self._train_future is None:
            return False
        if not self._train_future.done():
            return False
        self._last_train_result = self._train_future.result()
        self._drain_submitted_rollouts_before_sync()
        self._complete_train_and_sync()
        return True

    def _finish_pending_train_blocking(self) -> None:
        while self._pending_train is not None:
            if self._train_future is not None and self._train_future.done():
                self._last_train_result = self._train_future.result()
                self._drain_submitted_rollouts_before_sync()
                self._complete_train_and_sync()
                return
            self._collect_one_sample(timeout=0.05)
            if self._train_future is not None and self._train_future.done():
                self._last_train_result = self._train_future.result()
                self._drain_submitted_rollouts_before_sync()
                self._complete_train_and_sync()
                return
            self._collect_teacher_ready()
            if self._train_future is not None and self._train_future.done():
                self._last_train_result = self._train_future.result()
                self._drain_submitted_rollouts_before_sync()
                self._complete_train_and_sync()
                return
            self._maybe_log_train_wait()
            time.sleep(0.01)

    def _drain_submitted_rollouts_before_sync(self) -> None:
        """Conservative phase-1 weight policy: no sync during active requests."""
        self._last_drain_wait_log = time.monotonic()
        self._drain_wait_log_count = 0
        while any(not rec.rollout_complete for rec in self._inflight.values()):
            self._maybe_log_drain_wait()
            self._collect_one_sample(timeout=0.05)
            self._collect_teacher_ready()
            self._flush_train_blocking_teacher()
        self._collect_teacher_ready()

    def _maybe_log_train_wait(self) -> None:
        if self._pending_train is None or self._train_future is None:
            return
        if self._train_future.done():
            return
        now = time.monotonic()
        if now - self._last_train_wait_log < self.WATCHDOG_INTERVAL_S:
            return
        if self._train_wait_log_count >= self.WATCHDOG_MAX_LOGS_PER_WAIT:
            return
        self._last_train_wait_log = now
        self._train_wait_log_count += 1
        rec: _LogicalBatch = self._pending_train["rec"]
        dispatch_mono = self._pending_train.get("dispatch_mono")
        age_s = now - dispatch_mono if isinstance(dispatch_mono, (int, float)) else 0.0
        cap_note = (
            " last_capped_watchdog_log=true"
            if self._train_wait_log_count >= self.WATCHDOG_MAX_LOGS_PER_WAIT
            else ""
        )
        print(
            "[StepOffAsync] waiting for train result "
            f"train_step={rec.train_step} logical_batch_id={rec.logical_batch_id} "
            f"watchdog_log={self._train_wait_log_count}/{self.WATCHDOG_MAX_LOGS_PER_WAIT} "
            f"age={age_s:.1f}s future_done={self._train_future.done()} "
            f"inflight={self._inflight_summary()}{cap_note}",
            flush=True,
        )

    def _maybe_log_drain_wait(self) -> None:
        now = time.monotonic()
        if now - self._last_drain_wait_log < self.WATCHDOG_INTERVAL_S:
            return
        if self._drain_wait_log_count >= self.WATCHDOG_MAX_LOGS_PER_WAIT:
            return
        self._last_drain_wait_log = now
        self._drain_wait_log_count += 1
        cap_note = (
            " last_capped_watchdog_log=true"
            if self._drain_wait_log_count >= self.WATCHDOG_MAX_LOGS_PER_WAIT
            else ""
        )
        print(
            "[StepOffAsync] waiting for submitted rollouts before sync "
            f"watchdog_log={self._drain_wait_log_count}/{self.WATCHDOG_MAX_LOGS_PER_WAIT} "
            f"next_train_id={self._next_train_id} "
            f"inflight={self._inflight_summary()}{cap_note}",
            flush=True,
        )

    def _inflight_summary(self, *, limit: int = 8) -> str:
        parts = []
        for lbid, rec in list(self._inflight.items())[:limit]:
            parts.append(
                f"{lbid}:rollout={len(rec.samples)}/{rec.expected},"
                f"scored={len(rec.scored)}/{rec.expected},gen_wv={rec.gen_wv}"
            )
        extra = len(self._inflight) - limit
        if extra > 0:
            parts.append(f"...+{extra}")
        return "[" + "; ".join(parts) + "]"

    def _complete_train_and_sync(self) -> None:
        assert self._pending_train is not None
        rec: _LogicalBatch = self._pending_train["rec"]
        gen_out = self._pending_train["gen_out"]
        timing = self._pending_train["timing"]
        result = self._last_train_result or {}
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        if self.tracer is not None:
            teacher_artifacts = metrics.get("teacher_artifacts") or {}
            for ev in teacher_artifacts.get("recv_events", []):
                self.tracer.instant(
                    "teacher_artifact_recv_trainer",
                    cat="train",
                    tid=self.c.TID_TRAIN,
                    args={"logical_batch_id": ev.get("logical_batch_id"),
                          "sample_in_batch_idx": ev.get("sample_in_batch_idx"),
                          "n_bytes": ev.get("n_bytes")},
                )
                self.tracer.instant(
                    "teacher_artifact_buffer_sample_ready",
                    cat="train",
                    tid=self.c.TID_TRAIN,
                    args={"logical_batch_id": ev.get("logical_batch_id"),
                          "sample_in_batch_idx": ev.get("sample_in_batch_idx")},
                )
            if teacher_artifacts:
                self.tracer.instant(
                    "teacher_artifact_buffer_logical_batch_ready",
                    cat="train",
                    tid=self.c.TID_TRAIN,
                    args={"logical_batch_id": teacher_artifacts.get("logical_batch_id"),
                          "ready_count": teacher_artifacts.get("ready_count"),
                          "expected_count": teacher_artifacts.get("expected_count"),
                          "trainer_teacher_artifact_recv_bytes": teacher_artifacts.get(
                              "trainer_teacher_artifact_recv_bytes", 0),
                          "coordinator_teacher_artifact_bytes": teacher_artifacts.get(
                              "coordinator_teacher_artifact_bytes", 0)},
                )
            tt = metrics.get("timing", {})
            t_train_start = tt.get("mono_start", time.monotonic())
            t_train_end = tt.get("mono_end", time.monotonic())
            prompt_tok = int(gen_out["prompt_lengths"].sum()) if "prompt_lengths" in gen_out else 0
            resp_tok = int(gen_out["response_lengths"].sum()) if "response_lengths" in gen_out else 0
            self.tracer.emit(
                "train",
                cat="train",
                tid=self.c.TID_TRAIN,
                t_start=t_train_start,
                t_end=t_train_end,
                args={"logical_batch_id": rec.logical_batch_id,
                      "prompt_tok": prompt_tok, "resp_tok": resp_tok,
                      "n_seqs": len(gen_out.get("prompt_lengths", [])),
                      **getattr(self.c, "_trainer_trace_info", {})},
            )
        n_optim = metrics.get("n_optim_steps") or 1
        self._last_n_optim_steps = int(n_optim)
        stale = self._pending_train["staleness"]
        timing["staleness_min"] = stale
        timing["staleness_max"] = stale
        timing["staleness_mean"] = stale
        timing["staleness_std"] = 0.0
        self.mode.log_train_step(rec.train_step, timing, gen_out, result)

        sync_start = time.monotonic()
        sync_seconds = self.c._sync_weights()
        timing["sync_seconds"] = sync_seconds
        if self.tracer is not None:
            self.tracer.emit("sync_weights", cat="sync", tid=self.c.TID_PIPELINE,
                             t_start=sync_start, t_end=time.monotonic())
        self._weight_version += self._last_n_optim_steps
        del self._inflight[rec.logical_batch_id]
        self._next_train_id += 1
        self._pending_train = None
        self._train_future = None
        self._last_train_result = None

        if (self.save_freq > 0 and rec.train_step % self.save_freq == 0):
            self.c._save_checkpoint(rec.train_step)
        if self.test_freq > 0 and rec.train_step % self.test_freq == 0:
            if self.tracer is not None:
                with self.tracer.span("eval", cat="eval", tid=self.c.TID_EVAL):
                    self.c._evaluate(rec.train_step)
            else:
                self.c._evaluate(rec.train_step)

    def _span(self, name: str):
        class _Noop:
            def __enter__(self_inner):
                return None
            def __exit__(self_inner, *args):
                return False
        if self.tracer is None:
            return _Noop()
        return self.tracer.span(name, cat="pipeline", tid=self.c.TID_PIPELINE)


class StepOffAsyncCoordinator(CoordinatorBase):
    """n-step-off coordinator using AsyncLLM sample completion."""

    def run(self):
        if getattr(self, "_mode_cls", None) is not OPDMode:
            raise ValueError("StepOffAsyncCoordinator currently supports OPD mode only")
        sync_mode = "NCCL" if self.use_nccl else "CPU"
        print(
            f"[Pipeline] async step_off={self.step_off}, steps={self.total_steps}, "
            f"epochs={self.total_epochs}, sync={sync_mode}",
            flush=True,
        )
        self._mode = self._mode_cls.from_coordinator(self)
        resume_step, test_freq, eval_modes, val_before_train = self._prepare_run()
        data_iter = iter(self._data_iterator())
        self._skip_data_for_resume(data_iter, resume_step)
        sched_test_freq = -1 if not (eval_modes & {"inline"}) else test_freq
        mini_bs = self.opd_config.trainer.mini_batch_size or 0
        n_mini = max(self.batch_size // mini_bs, 1) if mini_bs > 0 else 1
        scheduler = StepOffAsyncScheduler(
            self,
            self._mode,
            step_off=self.step_off,
            total_steps=self.total_steps,
            test_freq=sched_test_freq,
            save_freq=self.save_freq,
            n_mini_per_step=n_mini,
        )
        step = scheduler.run(data_iter, resume_step)
        if "post" in eval_modes and test_freq > 0:
            self._run_post_eval(self.tracer, test_freq, val_before_train)
        if "post_allgpu" in eval_modes and test_freq > 0:
            print("[Pipeline] Skipping post-eval (will use all-GPU eval after shutdown).",
                  flush=True)
        print(f"[Pipeline] Done ({min(step, self.total_steps)} steps).", flush=True)

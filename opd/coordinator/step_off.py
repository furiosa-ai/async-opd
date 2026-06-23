"""Step-off coordinator and scheduler.

StepOffScheduler drives the step loop while the backend provides all operations.
StepOffCoordinator wires up the scheduler with the pipeline components.
"""

import statistics
import time
from collections import deque

from opd.coordinator.base import CoordinatorBase
from opd.coordinator.mode import build_step_off_backend
from opd.utils.trace import timer


class StepOffScheduler:
    """N-step-off scheduling loop.

    The backend object provides pipeline operations:
      async_generate(batch), wait_generate() -> gen_dict,
      async_teacher(gen_out) -> future, resolve_teacher(future, timing) -> teacher_out,
      async_train(gen_out, teacher_out), wait_train() -> result_dict,
      sync_weights(), evaluate(step),
      log_train_step(step, timing, gen_out, train_result),
      save_checkpoint(step)
    """

    def __init__(self, backend, step_off, total_steps, test_freq=-1,
                 save_freq=0, tracer=None, tid_config=None,
                 n_mini_per_step=1):
        """
        Args:
            backend: object implementing the scheduler backend interface
            step_off: number of generations to queue ahead (0=sync, 1+=async overlap)
            total_steps: total training steps
            test_freq: evaluate every N steps (-1 = never)
            save_freq: save checkpoint every N steps (0 = never)
            tracer: optional Tracer for Perfetto trace events
            tid_config: optional dict of TID constants for tracing
            n_mini_per_step: estimated optimizer updates per step (for resume weight version init)
        """
        self.backend = backend
        self.step_off = step_off
        self.total_steps = total_steps
        self.test_freq = test_freq
        self.save_freq = save_freq
        self._n_mini_per_step = n_mini_per_step  # only for resume init
        self._last_n_optim_steps = None  # set from actual trainer result
        self.tracer = tracer
        # TID constants for tracing (defaults match StepOffCoordinator)
        tc = tid_config or {}
        self.TID_ROLLOUT_BASE = tc.get("rollout_base", 0)
        self.TID_ROLLOUT = tc.get("rollout", 10)
        self.TID_TRAIN = tc.get("train", 12)
        self.TID_PIPELINE = tc.get("pipeline", 13)
        self.TID_EVAL = tc.get("eval", 14)
        # Per-component trace metadata (host, gpu_ids)
        self._trainer_trace_info = tc.get("trainer_info", {})
        self._teacher_trace_info = tc.get("teacher_info", {})

    def _next_batch(self):
        try:
            _, batch = next(self._data_iter)
            return batch
        except StopIteration:
            self._data_exhausted = True
            return None

    def _send_gen(self):
        batch = self._next_batch()
        if batch is None:
            return False
        self.backend.async_generate(batch)
        self._gens_in_pipe += 1
        self._gen_wv_queue.append(self._weight_version)
        return True

    def _recv_gen(self):
        gen = self.backend.wait_generate()
        t_recv = time.monotonic()
        self._gens_in_pipe -= 1
        gen_secs = gen.pop("timing", {}).get("generate_seconds", 0)
        tr = self.tracer
        vllm_stats = gen.pop("_vllm_stats", {})
        if tr:
            worker_timings = gen.pop("_worker_timings", None)
            gen_start = gen_end = None
            # Token counts for trace args
            prompt_toks = int(gen["prompt_lengths"].sum()) if "prompt_lengths" in gen else 0
            resp_toks = int(gen["response_lengths"].sum()) if "response_lengths" in gen else 0
            n_seqs = len(gen.get("prompt_lengths", []))
            for wt in (worker_timings or []):
                wid = wt.get("worker_id", 0)
                t_start = wt.get("mono_start", t_recv - wt.get("generate_seconds", 0))
                t_end = wt.get("mono_end", t_recv)
                tr.emit(f"gen-w{wid}", cat=f"rollout-{wid}",
                        tid=self.TID_ROLLOUT_BASE + wid,
                        t_start=t_start, t_end=t_end,
                        args={"prompt_tok": prompt_toks, "gen_tok": resp_toks,
                              "n_seqs": n_seqs,
                              "host": wt.get("host", ""),
                              "gpu_ids": wt.get("gpu_ids", "")})
                gen_start = min(t_start, gen_start) if gen_start else t_start
                gen_end = max(t_end, gen_end) if gen_end else t_end
            if gen_start is not None:
                tr.emit("generate", cat="rollout", tid=self.TID_ROLLOUT,
                        t_start=gen_start, t_end=gen_end,
                        args={"prompt_tok": prompt_toks, "gen_tok": resp_toks,
                              "n_seqs": n_seqs})
            if gen_end and gen_end < t_recv:
                tr.emit("queue", cat="pipeline", tid=self.TID_PIPELINE,
                        t_start=gen_end, t_end=t_recv)
            # vLLM throughput counters
            for wid, samples in vllm_stats.items():
                for mono_t, gen_tps, prompt_tps, running, kv_pct in samples:
                    tr.counter(f"rollout-w{wid}", {
                        "gen_tok/s": gen_tps,
                        "prompt_tok/s": prompt_tps,
                        "running_reqs": running,
                        "kv_cache_%": kv_pct,
                    }, tid=self.TID_ROLLOUT_BASE + wid, t=mono_t)
            # Emit zero counter when worker is truly idle (no queued generate).
            # Use gen_end (worker's mono_end) as the idle transition timestamp.
            if gen_end and self._gens_in_pipe == 0:
                for wid in vllm_stats:
                    tr.counter(f"rollout-w{wid}", {
                        "gen_tok/s": 0, "prompt_tok/s": 0,
                        "running_reqs": 0, "kv_cache_%": 0,
                    }, tid=self.TID_ROLLOUT_BASE + wid, t=gen_end + 0.001)
        else:
            gen.pop("_worker_timings", None)
        self._last_gen_end = t_recv
        gen_wv = self._gen_wv_queue.popleft()
        return gen, gen_secs, gen_wv

    def _steps_until_eval(self, step):
        if self.test_freq <= 0:
            return float("inf")
        r = step % self.test_freq
        return 0 if r == 0 else self.test_freq - r

    def _finish_pending_train(self):
        """Wait for pending train result, emit trace, log metrics.

        Does NOT drain in-flight gens or sync weights — call _drain_and_sync()
        separately (typically after dispatching the next train so drain+sync
        overlap with training).
        """
        if self._pending_train is None:
            return
        pt = self._pending_train
        be = self.backend
        tr = self.tracer
        if self._cached_train_result is not None:
            pt_result = self._cached_train_result
            self._cached_train_result = None
        else:
            pt_result = be.wait_train()
        t_done = time.monotonic()
        if pt_result is not None and "metrics" not in pt_result:
            print(f"[StepOff] WARNING: wait_train returned unexpected result "
                  f"(keys={list(pt_result.keys())}): {pt_result}", flush=True)
        metrics = (pt_result or {}).get("metrics", {})
        if tr:
            teacher_artifacts = metrics.get("teacher_artifacts") or {}
            for ev in teacher_artifacts.get("recv_events", []):
                tr.instant(
                    "teacher_artifact_recv_trainer",
                    cat="train",
                    tid=self.TID_TRAIN,
                    args={"logical_batch_id": ev.get("logical_batch_id"),
                          "sample_in_batch_idx": ev.get("sample_in_batch_idx"),
                          "n_bytes": ev.get("n_bytes")},
                )
                tr.instant(
                    "teacher_artifact_buffer_sample_ready",
                    cat="train",
                    tid=self.TID_TRAIN,
                    args={"logical_batch_id": ev.get("logical_batch_id"),
                          "sample_in_batch_idx": ev.get("sample_in_batch_idx")},
                )
            if teacher_artifacts:
                tr.instant(
                    "teacher_artifact_buffer_logical_batch_ready",
                    cat="train",
                    tid=self.TID_TRAIN,
                    args={"logical_batch_id": teacher_artifacts.get("logical_batch_id"),
                          "ready_count": teacher_artifacts.get("ready_count"),
                          "expected_count": teacher_artifacts.get("expected_count"),
                          "trainer_teacher_artifact_recv_bytes": teacher_artifacts.get(
                              "trainer_teacher_artifact_recv_bytes", 0),
                          "coordinator_teacher_artifact_bytes": teacher_artifacts.get(
                              "coordinator_teacher_artifact_bytes", 0)},
                )
            train_timing = metrics.get("timing", {})
            t_send_mono = train_timing.get("send_mono")
            t_queue_recv = train_timing.get("queue_recv")
            t_train_start = train_timing.get("mono_start", t_done)
            t_train_end = train_timing.get("mono_end", t_done)
            if t_send_mono and t_queue_recv:
                tr.emit("queue_transit", cat="pipeline", tid=self.TID_PIPELINE,
                        t_start=t_send_mono, t_end=t_queue_recv)
            if t_queue_recv and t_queue_recv < t_train_start:
                tr.emit("recv_train", cat="pipeline", tid=self.TID_PIPELINE,
                        t_start=t_queue_recv, t_end=t_train_start)
            go = pt["gen_out"]
            train_prompt_tok = int(go["prompt_lengths"].sum()) if "prompt_lengths" in go else 0
            train_resp_tok = int(go["response_lengths"].sum()) if "response_lengths" in go else 0
            tr.emit("train", cat="train", tid=self.TID_TRAIN,
                    t_start=t_train_start, t_end=t_train_end,
                    args={"prompt_tok": train_prompt_tok, "resp_tok": train_resp_tok,
                          "n_seqs": len(go.get("prompt_lengths", [])),
                          **self._trainer_trace_info})
            if t_train_end < t_done:
                tr.emit("train_result", cat="pipeline", tid=self.TID_PIPELINE,
                        t_start=t_train_end, t_end=t_done)
        # Track actual optimizer steps from trainer result
        n_optim = metrics.get("n_optim_steps")
        if n_optim is None:
            print(f"[StepOff] WARNING: train result missing n_optim_steps, "
                  f"keys={list(metrics.keys())}, defaulting to 1", flush=True)
            n_optim = 1
        self._last_n_optim_steps = n_optim
        # Staleness stats (compatible with fully_async logging)
        staleness = pt.get("staleness", 0)
        pt["timing"]["staleness_min"] = staleness
        pt["timing"]["staleness_max"] = staleness
        pt["timing"]["staleness_mean"] = staleness
        pt["timing"]["staleness_std"] = 0.0
        be.log_train_step(pt["step"], pt["timing"], pt["gen_out"], pt_result)
        self._pending_train = None

    def _drain_gens(self):
        """Drain in-flight gens into buffer. Rollout-only, no trainer needed."""
        tr = self.tracer
        with timer() as t_drain:
            while self._gens_in_pipe > 0:
                self._gen_buffer.append(self._recv_gen())
        if tr and t_drain["elapsed"] > 0.0005:
            tr.emit("drain_gen", cat="pipeline", tid=self.TID_PIPELINE,
                    t_start=t_drain["mono_start"], t_end=t_drain["mono_end"])

    def _sync_and_log(self):
        """Sync weights + log the finished train step. Requires train done."""
        be = self.backend
        tr = self.tracer
        if tr:
            with tr.span("sync_weights", cat="sync", tid=self.TID_PIPELINE) as s:
                be.sync_weights()
        else:
            be.sync_weights()

    def _complete_pending_train(self):
        """Finish pending train + drain + sync (used for eval/final steps)."""
        had_pending = self._pending_train is not None
        self._finish_pending_train()
        self._drain_gens()
        if had_pending:
            self._sync_and_log()
            self._weight_version += self._last_n_optim_steps

    def run(self, data_iter, resume_step=0):
        """Run the scheduling loop.

        Args:
            data_iter: iterator yielding (index, batch_dict) pairs
            resume_step: step to resume from (0 = start fresh)
        """
        step_off = self.step_off
        be = self.backend
        tr = self.tracer

        self._gens_in_pipe = 0
        self._gen_buffer = deque()
        self._data_exhausted = False
        self._last_gen_end = 0.0
        self._pending_train = None
        self._cached_train_result = None
        self._data_iter = data_iter
        # Weight version tracks optimizer updates (not steps).
        self._weight_version = resume_step * self._n_mini_per_step
        self._gen_wv_queue = deque()  # weight version tag per in-flight gen

        # Warmup: queue step_off gens into rollout cmd queue
        # Skip if training is already complete (e.g. --resume for post-eval only)
        if resume_step < self.total_steps and step_off > 0:
            print(f"[StepOff] Warmup: queuing {step_off} rollout generations before first train step...", flush=True)
            for i in range(step_off):
                if not self._send_gen():
                    print(f"[StepOff] Warmup: data exhausted after {i}/{step_off} gens", flush=True)
                    break
                if (i + 1) % max(1, step_off // 4) == 0 or i == step_off - 1:
                    print(f"[StepOff] Warmup: queued {i + 1}/{step_off} gens", flush=True)

        step = resume_step
        for step in range(resume_step + 1, self.total_steps + 1):
            timing = {}
            if step_off == 0:
                self._complete_pending_train()

            # 1. Get gen
            if self._gen_buffer:
                gen_out, gen_secs, gen_wv = self._gen_buffer.popleft()
            elif self._gens_in_pipe > 0:
                gen_out, gen_secs, gen_wv = self._recv_gen()
            else:
                if not self._send_gen():
                    break
                gen_out, gen_secs, gen_wv = self._recv_gen()
            timing["generate_seconds"] = gen_secs

            # Trace: dispatch
            if tr:
                t_dispatch = time.monotonic()
                tr.emit("dispatch", cat="pipeline", tid=self.TID_PIPELINE,
                        t_start=self._last_gen_end, t_end=t_dispatch)

            # 2. Start teacher async
            tf = be.async_teacher(gen_out)

            # 3. Drain + finish previous train (overlaps with teacher scoring)
            had_pending = self._pending_train is not None
            is_eval_step = self.test_freq > 0 and step % self.test_freq == 0
            if step_off > 0 and had_pending:
                self._drain_gens()
                self._finish_pending_train()

            # 4. Resolve teacher
            teacher_out = be.resolve_teacher(tf, timing)

            # 5. Sync right after teacher, before next train
            if step_off > 0 and had_pending:
                self._sync_and_log()
                self._weight_version += self._last_n_optim_steps

            if teacher_out is None:
                print(f"[Step {step}] Teacher failed, skipping.", flush=True)
            else:
                # 6. Dispatch train async
                direct_train = getattr(be, "async_train_direct_teacher_output", None)
                if direct_train is not None:
                    direct_train(
                        gen_out,
                        teacher_out,
                        logical_batch_id=step - 1,
                        gen_weight_version=gen_wv,
                    )
                    if tr:
                        tr.instant(
                            "train_dispatch_from_teacher_buffer",
                            cat="pipeline",
                            tid=self.TID_PIPELINE,
                            args={"logical_batch_id": step - 1,
                                  "n_seqs": len(gen_out.get("prompt_lengths", [])),
                                  "gen_wv": gen_wv},
                        )
                else:
                    be.async_train(gen_out, teacher_out)
                staleness = self._weight_version - gen_wv
                self._pending_train = {
                    "step": step, "timing": timing, "gen_out": gen_out,
                    "staleness": staleness,
                }

            # 6. Checkpoint: wait for current train if save is due
            _saved = False
            if (step_off > 0 and had_pending
                    and self.save_freq > 0 and step % self.save_freq == 0
                    and not is_eval_step and self._pending_train is not None):
                self._cached_train_result = be.wait_train()
                be.save_checkpoint(step)
                _saved = True

            # 7. Feed next gen (after sync so it uses updated weights)
            if tr:
                with timer() as t_send:
                    sent = False
                    if (self._gens_in_pipe + len(self._gen_buffer) < step_off
                            and not self._data_exhausted
                            and self._steps_until_eval(step) > self._gens_in_pipe + len(self._gen_buffer)
                            and step + self._gens_in_pipe + len(self._gen_buffer) < self.total_steps):
                        self._send_gen()
                        sent = True
                if sent:
                    tr.emit("send_gen", cat="pipeline", tid=self.TID_PIPELINE,
                            t_start=t_send["mono_start"], t_end=t_send["mono_end"])
            else:
                if (self._gens_in_pipe + len(self._gen_buffer) < step_off
                        and not self._data_exhausted
                        and self._steps_until_eval(step) > self._gens_in_pipe + len(self._gen_buffer)
                        and step + self._gens_in_pipe + len(self._gen_buffer) < self.total_steps):
                    self._send_gen()

            # 8. Eval
            if is_eval_step:
                self._complete_pending_train()
                # Save after train completes, before eval
                if self.save_freq > 0 and step % self.save_freq == 0 and not _saved:
                    be.save_checkpoint(step)
                    _saved = True
                if tr:
                    with tr.span("eval", cat="eval", tid=self.TID_EVAL):
                        be.evaluate(step)
                else:
                    be.evaluate(step)
                # Refill pipeline
                while (self._gens_in_pipe + len(self._gen_buffer) < step_off
                       and not self._data_exhausted
                       and step + self._gens_in_pipe + len(self._gen_buffer) < self.total_steps):
                    if not self._send_gen():
                        break

            # 9. Save checkpoint — fallback for step_off=0 or step 1 (no had_pending)
            elif self.save_freq > 0 and step % self.save_freq == 0 and not _saved:
                # Drain pending train before save — otherwise the save result
                # ends up in the queue before collect_train() runs, and
                # _wait_checkpoint_save() consumes the train result by mistake.
                if self._pending_train is not None:
                    self._complete_pending_train()
                be.save_checkpoint(step)

        # Complete final pending train
        self._complete_pending_train()

        # Save final checkpoint if needed
        if self.save_freq > 0 and step > resume_step and step % self.save_freq != 0:
            be.save_checkpoint(step)

        return step


class StepOffCoordinator(CoordinatorBase):
    """Coordinator for n-step-off scheduling."""

    def run(self):
        """Run the training pipeline with StepOffScheduler."""
        step_off = self.step_off
        sync_mode = "NCCL" if self.use_nccl else "CPU"
        print(f"[Pipeline] step_off={step_off}, steps={self.total_steps}, "
              f"epochs={self.total_epochs}, sync={sync_mode}", flush=True)

        # Construct mode before _prepare_run() so eval-before-train works
        # (_prepare_run may call _evaluate(0) which delegates to self._mode)
        mode_cls = getattr(self, '_mode_cls', None)
        if mode_cls is None:
            raise RuntimeError(
                "StepOffCoordinator requires _mode_cls to be set. "
                "Use the factory (Coordinator or create_coordinator) to construct "
                "coordinators — it determines the correct mode from config."
            )
        self._mode = mode_cls.from_coordinator(self)

        resume_step, test_freq, eval_modes, val_before_train = self._prepare_run()

        backend = build_step_off_backend(self._mode, self)

        data_iter = iter(self._data_iterator())

        self._skip_data_for_resume(data_iter, resume_step)
        # Let mode clean up after resume skip (e.g. GRPOMode clears _gt_queue)
        if hasattr(self._mode, 'on_resume_skip_complete'):
            self._mode.on_resume_skip_complete()

        tid_config = {
            "rollout_base": self.TID_ROLLOUT_BASE,
            "rollout": self.TID_ROLLOUT,
            "train": self.TID_TRAIN,
            "pipeline": self.TID_PIPELINE,
            "eval": self.TID_EVAL,
            "trainer_info": getattr(self, '_trainer_trace_info', {}),
        }

        # Disable inline eval in scheduler if only post modes are used
        sched_test_freq = -1 if not (eval_modes & {"inline"}) else test_freq

        oc = self.opd_config
        mini_bs = oc.trainer.mini_batch_size or 0
        n_mini = max(self.batch_size // mini_bs, 1) if mini_bs > 0 else 1

        scheduler = StepOffScheduler(
            backend=backend,
            step_off=step_off,
            total_steps=self.total_steps,
            test_freq=sched_test_freq,
            save_freq=self.save_freq,
            tracer=self.tracer,
            tid_config=tid_config,
            n_mini_per_step=n_mini,
        )
        step = scheduler.run(data_iter, resume_step)

        if "post" in eval_modes and test_freq > 0:
            self._run_post_eval(self.tracer, test_freq, val_before_train)
        if "post_allgpu" in eval_modes and test_freq > 0:
            print("[Pipeline] Skipping post-eval (will use all-GPU eval after shutdown).",
                  flush=True)

        print(f"[Pipeline] Done ({min(step, self.total_steps)} steps).",
              flush=True)

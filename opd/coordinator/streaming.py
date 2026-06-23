"""Streaming pipeline coordinator for on-policy distillation.

StreamCoordinator implements the fully_async scheduling mode: three independent
loops (rollout collector, teacher scorer, train dispatcher) run as threads
connected by bounded queues. The main thread is a thin lifecycle coordinator
handling weight sync, eval, and shutdown.
"""

import os
import statistics
import threading
import time
from collections import deque

import torch

from opd.data.batch_utils import pad_teacher, split_gen_teacher
from opd.utils.staleness_queue import StalenessQueue
from opd.utils.trace import timer
from opd.coordinator.base import CoordinatorBase


class StreamCoordinator(CoordinatorBase):
    """Coordinator for the fully_async scheduling mode.

    Extracted from Coordinator for decomposition. The run() method implements
    the abstract pipeline: rollout → teacher scoring → training, with weight
    sync driven by the train thread via sync_request/sync_ack events.
    """

    # ------------------------------------------------------------------ #
    #  Entry point                                                        #
    # ------------------------------------------------------------------ #

    def run(self):
        """Run the fully_async scheduling mode.

        Three independent loops (rollout collector, teacher scorer, train dispatcher)
        run as threads connected by bounded queues. The main thread is a thin
        lifecycle coordinator handling weight sync, eval, and shutdown.
        """
        # Construct mode before _prepare_run() so eval-before-train works
        mode_cls = getattr(self, '_mode_cls', None)
        if mode_cls is None:
            raise RuntimeError(
                "StreamCoordinator requires _mode_cls to be set. "
                "Use the factory (Coordinator or create_coordinator) to construct "
                "coordinators — it determines the correct mode from config."
            )
        self._mode = mode_cls.from_coordinator(self)

        resume_step, test_freq, eval_modes, val_before_train = self._prepare_run()

        # Determine effective batch size from mode's stream_batch_multiplier
        G = self._mode.stream_batch_multiplier
        effective_batch_size = self.batch_size * G
        if G > 1:
            self._need_student_logprobs = True
            # Guard: KL-regularized GRPO not yet supported in streaming
            if hasattr(self._mode, 'grpo_kl_beta') and self._mode.grpo_kl_beta > 0:
                raise NotImplementedError(
                    f"GRPO streaming does not yet support grpo_kl_beta > 0 "
                    f"(got {self._mode.grpo_kl_beta}). Reference model scoring "
                    f"is not implemented for streaming stages. Use step-off "
                    f"scheduling or set grpo_kl_beta: 0."
                )

        self._max_pending = (self.staleness_threshold + 1) * effective_batch_size
        capacity_sem = threading.Semaphore(self._max_pending)
        print(f"[Pipeline] fully_async mode, steps={self.total_steps}, "
              f"staleness_threshold={self.staleness_threshold}, "
              f"max_pending={self._max_pending}",
              flush=True)

        tr = self.tracer

        # Cycling data iterator
        data_iter = self._cycling_data_iter()

        self._skip_data_for_resume(data_iter, resume_step)

        # --- Setup queues and threading events ---
        step_counter = [resume_step]  # mutable, single writer (train loop)
        # Weight version tracks optimizer updates (not steps).
        oc = self.opd_config
        mini_bs = oc.trainer.mini_batch_size or 0
        batch_size = oc.trainer.batch_size
        n_mini_per_step = max(batch_size // mini_bs, 1) if mini_bs > 0 else 1
        self._n_mini_per_step = n_mini_per_step
        # On resume, _prepare_run() did 1 weight sync → rollout version = 1.
        # Offset aligns rollout's version (0-based) with coordinator's (resume-based).
        # Rollout versions are scaled by n_mini_per_step in the RolloutCollector
        # to match step-off convention (optimizer-step units).
        self._rollout_version_offset = (resume_step * n_mini_per_step - 1) if resume_step > 0 else 0

        sample_queue = StalenessQueue(step_counter, self.staleness_threshold)
        scored_queue = StalenessQueue(step_counter, self.staleness_threshold)

        shutdown_event = threading.Event()
        pause_event = threading.Event()  # pauses collector + teacher threads
        collector_stopped = threading.Event()  # collector acks it has stopped reading
        sync_request = threading.Event()  # train -> coordinator: "need sync"
        sync_ack = threading.Event()      # coordinator -> train: "sync done"

        # Train results deque (thread-safe): train thread appends, coordinator pops
        train_results = deque()

        # Drop/use counters for resource waste tracking
        sample_stats = {"enqueued": 0}

        # Store instance attrs for extracted methods
        self._n_workers = self.rollout_proxy.n_workers
        self._rollout_span_starts = [time.monotonic()] * self._n_workers
        self._overall_rollout_start = self._rollout_span_starts[0]
        # Per-worker host/gpu info (populated from first sample results)
        self._worker_hosts = [""] * self._n_workers
        self._worker_gpu_ids = [""] * self._n_workers
        self._last_enqueued = [0]
        self._last_prompt_tok = 0
        self._last_gen_tok = 0
        self._pause_event = pause_event
        self._collector_stopped = collector_stopped
        self._shutdown_event = shutdown_event
        self._sync_request = sync_request
        self._sync_ack = sync_ack
        self._sample_stats = sample_stats
        self._step_counter = step_counter
        self._sample_queue = sample_queue
        self._scored_queue = scored_queue
        self._capacity_sem = capacity_sem

        # Start autonomous generation on all workers.
        # Note: For GRPO, seed prompts lack prompt_group_id/ground_truth
        # metadata (they bypass PromptFeeder). TrainDispatcher passes them
        # through directly (gid=None), and assemble_fn rejects them —
        # the capacity_sem release on the None-teacher path prevents deadlock.
        # PromptFeeder takes over with properly tagged prompts immediately.
        sub_batches = self._feed_sub_batches(data_iter, n_batches=1)
        if sub_batches is None:
            print("[Pipeline] No data available.", flush=True)
            return
        # Acquire capacity for seed prompts (they bypass PromptFeeder)
        n_seed = sum(sb["input_ids"].size(0) for sb in sub_batches)
        for _ in range(n_seed):
            capacity_sem.acquire()
        self.rollout_proxy.enter_autonomous(sub_batches)

        # --- Launch four loop threads (stage classes) ---
        from opd.streaming_stages import PromptFeeder, RolloutCollector, TeacherScorer, TrainDispatcher

        # Construct transport-agnostic stage interfaces
        self._prompt_sink = self.rollout_proxy.prompt_sink()
        sample_stream = self.rollout_proxy.sample_stream()

        feeder = PromptFeeder(
            data_iter=data_iter, prompt_sink=self._prompt_sink,
            shutdown_event=shutdown_event, pause_event=pause_event,
            need_student_logprobs=self._need_student_logprobs,
            rollout_support_topk_k=getattr(self, "_rollout_support_topk_k", 0),
            mc_n_total_samples=getattr(self, "_mc_n_total_samples", 0),
            capacity_sem=capacity_sem, sync_request=sync_request,
            tracer=tr, tid_rollout=self.TID_ROLLOUT)

        collector = RolloutCollector(
            sample_stream=sample_stream, sample_queue=sample_queue,
            shutdown_event=shutdown_event, pause_event=pause_event,
            collector_stopped=collector_stopped,
            weight_version_ref=[0],  # unused, kept for API compat
            tracer=tr, tid_rollout=self.TID_ROLLOUT,
            sample_stats=sample_stats, capacity_sem=capacity_sem,
            version_offset=self._rollout_version_offset,
            version_scale=n_mini_per_step)

        # Mode provides its own streaming score_fn and assemble_fn
        score_fn = self._mode.make_stream_score_fn(self.teacher_client)
        assemble_fn = self._mode.make_stream_assemble_fn(self.max_response_length)
        default_scoring_bs = oc.teacher.scoring_batch_size or 8 if oc.teacher else 8
        scoring_bs = effective_batch_size if effective_batch_size > self.batch_size else default_scoring_bs

        scorer = TeacherScorer(
            score_fn=score_fn, sample_queue=sample_queue,
            scored_queue=scored_queue, shutdown_event=shutdown_event,
            scoring_batch_size=scoring_bs,
            tracer=tr, tid_teacher=self.TID_TEACHER,
            teacher_trace_info=getattr(self, '_teacher_trace_info', {}))

        dispatcher = TrainDispatcher(
            scored_queue=scored_queue, shutdown_event=shutdown_event,
            sync_request=sync_request, sync_ack=sync_ack,
            step_counter=step_counter, train_results=train_results,
            batch_size=effective_batch_size,
            assemble_batch_fn=assemble_fn,
            async_train_fn=self._async_train,
            wait_train_fn=self._wait_train,
            tracer=tr, tid_train=self.TID_TRAIN,
            capacity_sem=capacity_sem,
            trainer_trace_info=getattr(self, '_trainer_trace_info', {}),
            group_size=G)

        feeder_thread = threading.Thread(
            target=feeder.run, daemon=True, name="prompt-feeder")
        collector_thread = threading.Thread(
            target=collector.run, daemon=True, name="rollout-collector")
        teacher_thread = threading.Thread(
            target=scorer.run, daemon=True, name="teacher-scorer")
        train_thread = threading.Thread(
            target=dispatcher.run, daemon=True, name="train-dispatch")

        self._collector_thread = collector_thread
        feeder_thread.start()
        collector_thread.start()
        teacher_thread.start()
        train_thread.start()

        print("[Pipeline] Fully async threads started: "
              f"feeder, collector, teacher, train", flush=True)

        # --- Main thread = lifecycle coordinator ---
        try:
            while True:
                # Wait for train loop to request sync
                self._sync_request.wait(timeout=1.0)
                if not self._sync_request.is_set():
                    if self._step_counter[0] >= self.total_steps:
                        # Wait briefly for the final sync_request — the train
                        # thread may have incremented step_counter but not yet
                        # set sync_request (race window).
                        self._sync_request.wait(timeout=5.0)
                        if not self._sync_request.is_set():
                            break
                        # fall through to sync handling below
                    else:
                        # Health check: detect dead rollout/trainer processes
                        dead = []
                        for i, p in enumerate(self.rollout_procs):
                            if not p.is_alive():
                                dead.append(f"Rollout-{i} (pid={p.pid}, exit={p.exitcode})")
                        if self.trainer_proc and not self.trainer_proc.is_alive():
                            dead.append(f"Trainer (pid={self.trainer_proc.pid}, exit={self.trainer_proc.exitcode})")
                        if dead:
                            print(f"\n{'='*60}", flush=True)
                            print(f"[Coordinator] FATAL: dead worker(s) detected: {', '.join(dead)}",
                                  flush=True)
                            print(f"[Coordinator] step={self._step_counter[0]}/{self.total_steps}, "
                                  f"shutting down pipeline", flush=True)
                            print(f"{'='*60}\n", flush=True)
                            shutdown_event.set()
                            break
                        continue

                # step_counter already incremented by train thread (1-indexed)
                step = self._step_counter[0]
                print(f"[Coordinator] sync requested for step {step}", flush=True)

                # 1-2. Pause collector thread + rollout workers
                self._pause_workers_for_sync(step)
                print(f"[Coordinator] workers paused for step {step}", flush=True)

                # 3. Sync weights
                with tr.span("sync_weights", cat="sync",
                             tid=self.TID_PIPELINE) as sw:
                    sw["step"] = step
                    self._sync_weights_paused()
                print(f"[Coordinator] weights synced for step {step}", flush=True)

                # Check if eval is due this step (before deciding to resume)
                eval_due = ("inline" in eval_modes and
                            test_freq > 0 and step % test_freq == 0)

                if eval_due:
                    # 4-5. Skip resume — workers stay paused for eval.
                    # Exit autonomous directly from paused state.
                    # (Avoids double abort/resume cycle that corrupts vLLM
                    # engine scheduler state.)
                    self.rollout_proxy.submit_command("exit_autonomous")
                    exit_msgs = self.rollout_proxy.drain_until_status(
                        "exited_autonomous", forward_target=sample_queue) or []
                    self._write_vllm_stats(exit_msgs)
                    # Release permits for cancelled in-flight requests
                    # (their results will never reach the collector/dispatcher)
                    for msg in exit_msgs:
                        n_cancelled = msg.get("n_cancelled", 0)
                        if n_cancelled > 0:
                            print(f"[Coordinator] releasing {n_cancelled} permits "
                                  f"for cancelled in-flight requests", flush=True)
                            for _ in range(n_cancelled):
                                capacity_sem.release()
                else:
                    # 4-5. Resume rollout workers + collector/teacher threads
                    print(f"[Coordinator] resuming workers for step {step}", flush=True)
                    self._resume_workers_for_sync()
                    print(f"[Coordinator] workers resumed for step {step}", flush=True)

                # 6. Log completed train steps from deque
                self._sync_request.clear()
                while train_results:
                    wv, train_result, gen_out, samples = train_results.popleft()
                    ev_sample = 0
                    ev_scored = 0
                    sq_depth = self._sample_queue.qsize()
                    scq_depth = self._scored_queue.qsize()
                    if tr is not None:
                        sem_free = self._capacity_sem._value
                        sem_max = self._max_pending
                        in_flight_pct = (sem_max - sem_free) / sem_max * 100 if sem_max > 0 else 0
                        tr.counter("queue_depth",
                                   {"sample_queue": sq_depth,
                                    "scored_queue": scq_depth,
                                    "in_flight_%": round(in_flight_pct, 1)},
                                   tid=self.TID_PIPELINE)
                    # Staleness stats from samples used in this train step.
                    # With keep-pause, a single sample can span multiple weight
                    # versions. Use token-weighted staleness: each segment's
                    # staleness is weighted by how many tokens it contributed.
                    # Sample weight_versions are in optimizer-step units
                    # (scaled by n_mini_per_step in RolloutCollector).
                    cur_wv = wv * self._n_mini_per_step
                    stalenesses = []
                    for s in samples:
                        bps = s.get("weight_breakpoints")
                        resp_len = s.get("response_lengths")
                        if resp_len is not None:
                            resp_len = resp_len.item() if hasattr(resp_len, 'item') else int(resp_len[0])
                        if bps and len(bps) > 1 and resp_len and resp_len > 0:
                            # Token-weighted staleness across breakpoints
                            weighted_sum = 0.0
                            for j in range(len(bps)):
                                seg_start = bps[j][0]
                                seg_end = bps[j + 1][0] if j + 1 < len(bps) else resp_len
                                seg_tokens = max(seg_end - seg_start, 0)
                                seg_staleness = cur_wv - bps[j][1]
                                weighted_sum += seg_tokens * seg_staleness
                            stalenesses.append(weighted_sum / resp_len)
                        else:
                            s_wv = s.get("weight_version", 0)
                            stalenesses.append(cur_wv - s_wv)
                    stale_min = round(min(stalenesses), 2)
                    stale_max = round(max(stalenesses), 2)
                    stale_mean = round(statistics.mean(stalenesses), 2)
                    stale_std = round(statistics.stdev(stalenesses), 2) if len(stalenesses) > 1 else 0.0
                    timing = {
                        "sync_seconds": sw.elapsed,
                        "sample_q_depth": sq_depth,
                        "scored_q_depth": scq_depth,
                        "evicted_sample_q": ev_sample,
                        "evicted_scored_q": ev_scored,
                        "staleness_min": stale_min,
                        "staleness_max": stale_max,
                        "staleness_mean": stale_mean,
                        "staleness_std": stale_std,
                    }
                    self._log_train_step(
                        wv, timing, gen_out or {}, train_result)
                    print(f"[Step {wv}] queue: sample_q={sq_depth} scored_q={scq_depth} "
                          f"| evicted: sample_q={ev_sample} scored_q={ev_scored} "
                          f"| staleness: min={stale_min} max={stale_max} "
                          f"mean={stale_mean:.1f} std={stale_std:.1f}",
                          flush=True)

                # 7. Check termination before unblocking dispatcher
                done = self._step_counter[0] >= self.total_steps
                if done:
                    if test_freq > 0 and "inline" in eval_modes:
                        if not eval_due:
                            # Workers are in autonomous mode — pause + exit first
                            self._pause_event.set()
                            if not self._collector_stopped.wait(timeout=10.0):
                                print("[Coordinator] WARNING: collector slow to ack final pause",
                                      flush=True)
                            self.rollout_proxy.submit_command("pause")
                            self._write_vllm_stats(
                                self.rollout_proxy.drain_until_status(
                                    "paused", forward_target=sample_queue))
                            self.rollout_proxy.submit_command("exit_autonomous")
                            self._write_vllm_stats(
                                self.rollout_proxy.drain_until_status(
                                    "exited_autonomous", forward_target=sample_queue))
                        # eval_due: workers already exited autonomous above
                        with tr.span("eval", cat="eval",
                                     tid=self.TID_EVAL) as ev:
                            ev["step"] = step
                            self._evaluate(step)
                    # Save checkpoint for final step if due
                    if self.save_freq > 0 and step % self.save_freq == 0:
                        self._save_checkpoint(step)
                        self._wait_checkpoint_save()
                    # Signal all threads to stop before breaking —
                    # prevents dispatcher from running an extra train step
                    # and teacher from scoring unnecessary samples.
                    self._shutdown_event.set()
                    break

                # 8. Eval (workers already exited autonomous above if eval_due)
                if eval_due:

                    with tr.span("eval", cat="eval", tid=self.TID_EVAL) as ev:
                        ev["step"] = step
                        self._evaluate(step)

                    # Re-enter autonomous (feeder thread handles continuous prompts)
                    re_batches = self._feed_sub_batches(data_iter, n_batches=1)
                    if re_batches:
                        # Acquire capacity for seed prompts (feeder gated on sync_request)
                        n_seed = sum(sb["input_ids"].size(0) for sb in re_batches)
                        for _ in range(n_seed):
                            capacity_sem.acquire()
                        # _feed_sub_batches already sets return_logprobs if needed
                        self.rollout_proxy.enter_autonomous(re_batches)

                    # Reset rollout span starts after eval re-enter
                    t_resume = time.monotonic()
                    for wid in range(self._n_workers):
                        self._rollout_span_starts[wid] = t_resume
                    self._overall_rollout_start = t_resume

                    self._pause_event.clear()

                # 9. Checkpoint (must be blocking — drain save result before
                # releasing dispatcher, otherwise _wait_train gets save result
                # instead of train result).
                if self.save_freq > 0 and step % self.save_freq == 0:
                    self._save_checkpoint(step)
                    self._wait_checkpoint_save()

                # 10. Unblock dispatcher for next step
                self._sync_ack.set()
                print(f"[Coordinator] sync_ack set for step {step}", flush=True)

        finally:
            # Shutdown: signal all threads to stop
            self._shutdown_event.set()
            # Unblock train thread if it's waiting on sync_ack
            self._sync_ack.set()
            # Send sentinels to shared prompt_queue (one per worker)
            if hasattr(self, '_prompt_sink') and self._prompt_sink is not None:
                self._prompt_sink.send_sentinel()

            feeder_thread.join(timeout=5)
            collector_thread.join(timeout=5)
            teacher_thread.join(timeout=5)
            train_thread.join(timeout=5)

            # Stop autonomous mode
            try:
                self._pause_event.set()
                time.sleep(0.1)
                self.rollout_proxy.submit_command("pause")
                self._write_vllm_stats(
                    self.rollout_proxy.drain_until_status("paused"))
                self.rollout_proxy.submit_command("exit_autonomous")
                self._write_vllm_stats(
                    self.rollout_proxy.drain_until_status("exited_autonomous"))
            except Exception as e:
                print(f"[Shutdown] Ignoring cleanup error: {e}", flush=True)

        step = self._step_counter[0]
        # Save final checkpoint if not already saved by the in-loop save
        if self.save_freq > 0 and self.run_dir:
            final_ckpt = os.path.join(self.run_dir, "checkpoints", f"step_{step}")
            if not os.path.exists(final_ckpt):
                self._save_checkpoint(step)
                self._wait_checkpoint_save()

        # Log sample usage stats
        enqueued = self._sample_stats["enqueued"]
        print(f"[Pipeline] Sample stats: enqueued={enqueued}", flush=True)

        # Post-training eval: load each checkpoint and eval
        if "post" in eval_modes and test_freq > 0:
            self._run_post_eval(tr, test_freq, val_before_train)
        if "post_allgpu" in eval_modes and test_freq > 0:
            print("[Pipeline] Skipping post-eval (will use all-GPU eval after shutdown).",
                  flush=True)

        print(f"[Pipeline] Done ({min(step, self.total_steps)} steps, fully_async).",
              flush=True)

    # ------------------------------------------------------------------ #
    #  Extracted helpers                                                  #
    # ------------------------------------------------------------------ #

    def _cycling_data_iter(self):
        """Yield data batches indefinitely, cycling through the dataset."""
        while True:
            yield from self._data_iterator()

    def _drain_result_queues(self, target_status, sq=None):
        """Drain rollout result queues until target_status seen from all workers.

        Delegates to rollout_proxy.drain_until_status(). Straggler data
        samples are forwarded to sq (sample_queue) if provided.

        Returns list of status dicts (one per worker) for the target_status messages.
        """
        if sq is None:
            sq = self._sample_queue
        return self.rollout_proxy.drain_until_status(
            target_status, forward_target=sq)

    def _write_vllm_stats(self, status_msgs):
        """Write vLLM throughput stats from worker status messages to trace."""
        tr = self.tracer
        for wid, msg in enumerate(status_msgs):
            for mono_t, gen_tps, prompt_tps, running, kv_pct in msg.get("_vllm_stats", []):
                tr.counter(f"rollout-w{wid}", {
                    "gen_tok/s": gen_tps,
                    "prompt_tok/s": prompt_tps,
                    "running_reqs": running,
                    "kv_cache_%": kv_pct,
                }, tid=self.TID_ROLLOUT_BASE + wid, t=mono_t)

    def _pause_workers_for_sync(self, step_label):
        """Pause collector thread, then pause rollout workers.

        Returns after all workers have acked 'paused'. The collector thread
        is guaranteed stopped (pause_event set + sleep for its poll timeout).
        Emits per-worker rollout spans covering the active generation interval.
        """
        self._pause_event.set()
        # Wait for collector thread to confirm it has stopped reading.
        # Without this, the collector can consume and discard a worker's
        # "paused" status message from the result queue before
        # drain_until_status reads it, causing a deadlock.
        if not self._collector_stopped.wait(timeout=10.0):
            # Collector didn't ack — check if it's still alive
            ct = getattr(self, '_collector_thread', None)
            if ct is not None and not ct.is_alive():
                print("[Coordinator] WARNING: collector thread died, proceeding with pause",
                      flush=True)
            else:
                print("[Coordinator] WARNING: collector slow to ack pause (>10s), proceeding",
                      flush=True)
        # Emit per-worker rollout spans (from last resume to now)
        t_pause = time.monotonic()
        n_produced = self._sample_stats["enqueued"] - self._last_enqueued[0]
        prompt_tok = self._sample_stats.get("prompt_tok", 0) - self._last_prompt_tok
        gen_tok = self._sample_stats.get("gen_tok", 0) - self._last_gen_tok
        gen_args = {"step": step_label, "wv": self._step_counter[0],
                    "n_prompts": n_produced, "prompt_tok": prompt_tok,
                    "gen_tok": gen_tok}
        for wid in range(self._n_workers):
            winfo = self._rollout_worker_info[wid] if hasattr(self, '_rollout_worker_info') and wid < len(self._rollout_worker_info) else {}
            self.tracer.emit(f"gen-w{wid}", cat=f"rollout-{wid}",
                    tid=self.TID_ROLLOUT_BASE + wid,
                    t_start=self._rollout_span_starts[wid], t_end=t_pause,
                    args={**gen_args, "host": winfo.get("host", ""), "gpu_ids": winfo.get("gpu_ids", "")})
        # Emit overall generate span covering all workers
        self.tracer.emit("generate", cat="rollout", tid=self.TID_ROLLOUT,
                t_start=self._overall_rollout_start, t_end=t_pause,
                args={**gen_args, "n_workers": self._n_workers})
        # Now we are the sole reader of rollout results
        with self.tracer.span("pause_rollout", cat="sync",
                         tid=self.TID_PIPELINE) as sp:
            sp["step"] = step_label
            self.rollout_proxy.submit_command("pause")
            status_msgs = self.rollout_proxy.drain_until_status(
                "paused", forward_target=self._sample_queue)
            self._write_vllm_stats(status_msgs)

    def _resume_workers_for_sync(self, sq=None):
        """Resume rollout workers without seed batch, then resume threads.

        Workers in keep mode retain their in-flight requests. New prompts
        arrive via prompt_queue once the feeder thread unblocks (after
        sync_request is cleared). This avoids competing with the feeder
        for semaphore permits — the root cause of prior deadlocks.
        """
        if sq is None:
            sq = self._sample_queue
        self.rollout_proxy.submit_command("resume", None)
        self.rollout_proxy.drain_until_status("resumed", forward_target=sq)
        # Reset rollout span starts to track next interval
        t_resume = time.monotonic()
        for wid in range(self._n_workers):
            self._rollout_span_starts[wid] = t_resume
        self._overall_rollout_start = t_resume
        self._last_enqueued[0] = self._sample_stats["enqueued"]
        self._last_prompt_tok = self._sample_stats.get("prompt_tok", 0)
        self._last_gen_tok = self._sample_stats.get("gen_tok", 0)
        self._pause_event.clear()

    # ------------------------------------------------------------------ #
    #  Async-specific pipeline helpers                                    #
    # ------------------------------------------------------------------ #

    def _sync_weights_paused(self):
        """Weight sync while rollout workers are paused.

        Delegates to base class _sync_weights_paused which uses
        weight_engine.sync_paused() with drain_until_status for
        interleaved data + status messages.
        """
        fwd = getattr(self, '_sample_queue', None)
        super()._sync_weights_paused(forward_target=fwd)

    def _feed_sub_batches(self, data_iter, n_batches=1):
        """Feed sub-batches to rollout workers for autonomous streaming generation.

        Args:
            data_iter: Iterator yielding (idx, batch) tuples.
            n_batches: Number of data batches to concatenate. Use n_batches=2
                at startup so workers have enough prompts to keep generating
                while the first training step runs.
        """
        n_workers = self.rollout_proxy.n_workers
        batches = []
        for _ in range(n_batches):
            try:
                _, batch = next(data_iter)
                batches.append(batch)
            except StopIteration:
                break
        if not batches:
            return None

        # Concatenate multiple batches if needed
        if len(batches) > 1:
            batch = {}
            for key in batches[0]:
                vals = [b[key] for b in batches]
                if isinstance(vals[0], torch.Tensor):
                    batch[key] = torch.cat(vals, dim=0)
                else:
                    batch[key] = vals[0]  # non-tensor keys (e.g. flags) — take first
        else:
            batch = batches[0]

        bs = batch["input_ids"].size(0)
        chunk = (bs + n_workers - 1) // n_workers
        sub_batches = []
        for i in range(n_workers):
            s, e = i * chunk, min((i + 1) * chunk, bs)
            sub = {k: v[s:e] if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
            if self._need_student_logprobs:
                sub["return_logprobs"] = True
            if getattr(self, "_rollout_support_topk_k", 0) > 0:
                sub["response_topk_k"] = self._rollout_support_topk_k
            if getattr(self, "_mc_n_total_samples", 0) > 0:
                sub["mc_n_total_samples"] = self._mc_n_total_samples
            sub_batches.append(sub)
        return sub_batches

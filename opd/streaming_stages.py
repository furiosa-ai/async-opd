"""Async pipeline stage classes for fully_async scheduling mode.

Four stages form the streaming pipeline:
  PromptFeeder → RolloutCollector → TeacherScorer → TrainDispatcher
"""

import queue
import time

import torch


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PromptFeeder — feeds individual prompts to rollout workers
# ---------------------------------------------------------------------------

class PromptFeeder:
    """Feeds prompts from data iterator to a PromptSink.

    Runs as a thread. Pulls batches from data_iter, unpacks into
    individual prompt dicts, and pushes to the sink.
    Workers pull from the underlying transport at their own pace (pull model).
    Backpressure via the sink's bounded capacity.
    """

    def __init__(self, data_iter, prompt_sink, shutdown_event, pause_event,
                 need_student_logprobs=False, rollout_support_topk_k=0,
                 mc_n_total_samples=0, capacity_sem=None,
                 sync_request=None, tracer=None, tid_rollout=10):
        self.data_iter = data_iter
        self.prompt_sink = prompt_sink
        self.shutdown_event = shutdown_event
        self.pause_event = pause_event
        self.need_student_logprobs = need_student_logprobs
        self.rollout_support_topk_k = rollout_support_topk_k
        self.mc_n_total_samples = mc_n_total_samples
        self.capacity_sem = capacity_sem
        self.sync_request = sync_request
        self.tracer = tracer
        self.tid_rollout = tid_rollout

    def _gated(self):
        """Check if feeder should pause (pause_event or sync_request)."""
        if self.pause_event.is_set():
            return True
        sr = self.sync_request
        if sr is not None and sr.is_set():
            return True
        return False

    def run(self):
        shutdown_event = self.shutdown_event
        prompt_sink = self.prompt_sink
        capacity_sem = self.capacity_sem
        tr = self.tracer
        tid = self.tid_rollout
        rl = self.need_student_logprobs
        response_topk_k = self.rollout_support_topk_k
        mc_n_total_samples = self.mc_n_total_samples
        group_counter = 0  # monotonic prompt_group_id

        while not shutdown_event.is_set():
            if self._gated():
                time.sleep(0.05)
                continue
            try:
                _, batch = next(self.data_iter)
            except StopIteration:
                time.sleep(0.1)
                continue
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            max_prompt_len = input_ids.size(1)
            grpo_n = batch.get("grpo_n_samples", 1)
            ground_truths = batch.get("ground_truths", [])
            batch_rl = rl or batch.get("return_logprobs", False)
            for i in range(input_ids.size(0)):
                if shutdown_event.is_set():
                    return
                mask = attention_mask[i].bool()
                ids = input_ids[i][mask].tolist()
                p_len = len(ids)
                pad_len = max_prompt_len - p_len
                gid = group_counter
                group_counter += 1
                gt = ground_truths[i] if i < len(ground_truths) else None
                for _g in range(grpo_n):
                    if shutdown_event.is_set():
                        return
                    # Rate limit: wait for pipeline capacity before feeding
                    if capacity_sem is not None:
                        if not capacity_sem.acquire(blocking=False):
                            while not shutdown_event.is_set():
                                if self._gated():
                                    time.sleep(0.05)
                                    continue
                                if capacity_sem.acquire(timeout=1.0):
                                    break
                        if shutdown_event.is_set():
                            return
                    if tr is not None and i % 64 == 0 and _g == 0:
                        tr.counter("pipeline", {"prompts_fed": i},
                                   tid=tid)
                    prompt_info = {
                        "prompt_ids": ids,
                        "prompt_len": p_len,
                        "pad_len": pad_len,
                        "input_ids_row": input_ids[i].clone(),
                        "return_logprobs": batch_rl,
                    }
                    if response_topk_k > 0:
                        prompt_info["response_topk_k"] = response_topk_k
                    if mc_n_total_samples > 0:
                        prompt_info["mc_n_total_samples"] = mc_n_total_samples
                    if grpo_n > 1:
                        prompt_info["prompt_group_id"] = gid
                    if gt is not None:
                        prompt_info["ground_truth"] = gt
                    while not shutdown_event.is_set():
                        if self._gated():
                            time.sleep(0.05)
                            continue
                        if prompt_sink.put(prompt_info, timeout=1.0):
                            break


# ---------------------------------------------------------------------------
# RolloutCollector — drains rollout sample stream into sample_queue
# ---------------------------------------------------------------------------

class RolloutCollector:
    """Collects completed samples from rollout workers.

    Runs as a thread. Polls a SampleStream for deserialized data samples
    and pushes them to sample_queue.
    """

    def __init__(self, sample_stream, sample_queue, shutdown_event,
                 pause_event, weight_version_ref,
                 tracer=None, tid_rollout=10, sample_stats=None,
                 capacity_sem=None, version_offset=0, version_scale=1,
                 collector_stopped=None, **_kwargs):
        self.sample_stream = sample_stream
        self.sample_queue = sample_queue
        self.shutdown_event = shutdown_event
        self.pause_event = pause_event
        self.collector_stopped = collector_stopped
        self.weight_version_ref = weight_version_ref
        self.tracer = tracer
        self.tid_rollout = tid_rollout
        self.sample_stats = sample_stats
        self.capacity_sem = capacity_sem
        self.version_offset = version_offset
        self.version_scale = version_scale

    def run(self):
        shutdown_event = self.shutdown_event
        pause_event = self.pause_event
        collector_stopped = self.collector_stopped
        sample_queue = self.sample_queue
        sample_stream = self.sample_stream
        tr = self.tracer
        sample_stats = self.sample_stats

        while not shutdown_event.is_set():
            if pause_event.is_set():
                if collector_stopped is not None:
                    collector_stopped.set()
                while pause_event.is_set() and not shutdown_event.is_set():
                    time.sleep(0.01)
                if collector_stopped is not None:
                    collector_stopped.clear()
                continue
            if shutdown_event.is_set():
                return
            result = sample_stream.get_sample(timeout=0.05)
            if result is None:
                continue
            scale = self.version_scale
            offset = self.version_offset
            wv = result.get("weight_version", 0) * scale + offset
            result["weight_version"] = wv
            if "weight_breakpoints" in result:
                result["weight_breakpoints"] = [
                    (tok, ver * scale + offset) for tok, ver in result["weight_breakpoints"]]
            sample_queue.put(result, weight_version=wv)
            if sample_stats is not None:
                sample_stats["enqueued"] += 1
                pl = result.get("prompt_lengths")
                rl = result.get("response_lengths")
                if pl is not None:
                    sample_stats["prompt_tok"] = sample_stats.get("prompt_tok", 0) + int(pl.sum())
                if rl is not None:
                    sample_stats["gen_tok"] = sample_stats.get("gen_tok", 0) + int(rl.sum())
                if tr is not None and sample_stats["enqueued"] % 64 == 0:
                    tr.counter("pipeline",
                               {"samples_collected": sample_stats["enqueued"]},
                               tid=self.tid_rollout)


# ---------------------------------------------------------------------------
# TeacherScorer — scores samples with teacher logprobs
# ---------------------------------------------------------------------------

class TeacherScorer:
    """Scores samples from sample_queue using teacher model.

    Runs as a thread. Accumulates scoring_batch_size samples,
    submits to teacher via ZMQ client, attaches logprobs,
    and pushes scored samples to scored_queue.
    Does NOT pause during weight sync — keeps scoring samples
    already in sample_queue so scored_queue stays warm.
    """

    def __init__(self, score_fn, sample_queue, scored_queue,
                 shutdown_event, scoring_batch_size=8,
                 tracer=None, tid_teacher=11, teacher_trace_info=None):
        self.score_fn = score_fn
        self.sample_queue = sample_queue
        self.scored_queue = scored_queue
        self.shutdown_event = shutdown_event
        self.scoring_batch_size = scoring_batch_size
        self.tracer = tracer
        self.tid_teacher = tid_teacher
        self._teacher_trace_info = teacher_trace_info or {}

    def run(self):
        shutdown_event = self.shutdown_event
        sample_queue = self.sample_queue
        scored_queue = self.scored_queue
        scoring_batch_size = self.scoring_batch_size
        tr = self.tracer

        print(f"[TeacherLoop] started, scoring_batch_size={scoring_batch_size}",
              flush=True)

        try:
         while not shutdown_event.is_set():
            batch_samples = []
            while len(batch_samples) < scoring_batch_size:
                if shutdown_event.is_set():
                    break
                try:
                    sample = sample_queue.get(timeout=0.5)
                    batch_samples.append(sample)
                except queue.Empty:
                    continue
            if not batch_samples:
                continue

            full_token_lists = []
            for s in batch_samples:
                full_token_lists.extend(s.get("full_token_lists", []))

            sq_size = sample_queue.qsize()
            n_label = len(full_token_lists) or len(batch_samples)
            print(f"[TeacherLoop] scoring {n_label} samples "
                  f"(sq={sq_size})", flush=True)
            if tr is not None:
                tr.counter("pipeline", {"sample_queue": sq_size},
                           tid=self.tid_teacher)
            t_local_start = time.monotonic()
            try:
                t_start, t_end = self.score_fn(batch_samples)
            except Exception as e:
                print(f"[TeacherLoop] Error: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue
            t_local_end = time.monotonic()
            print(f"[TeacherLoop] scored {n_label} samples",
                  flush=True)

            if tr is not None:
                # Use teacher-side timestamps for local teacher, coordinator-side for remote.
                ts = t_start if t_start and not self._teacher_trace_info.get("_remote") else t_local_start
                te = t_end if t_end and not self._teacher_trace_info.get("_remote") else t_local_end
                total_tok = sum(len(tl) for tl in full_token_lists)
                tr.emit("teacher_score", cat="teacher",
                        tid=self.tid_teacher, t_start=ts, t_end=te,
                        args={"n_prompts": len(full_token_lists),
                              "total_tok": total_tok,
                              **self._teacher_trace_info})

            for sample in batch_samples:
                wv = sample.get("weight_version", 0)
                scored_queue.put(sample, weight_version=wv)
        except Exception as e:
            print(f"[TeacherLoop] CRASHED: {e}", flush=True)
            import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# GRPO score_fn / assemble_fn factories
# ---------------------------------------------------------------------------


# GRPO score_fn and assemble_batch_fn are defined on GRPOMode
# (opd/coordinator/grpo_mode.py) via the make_stream_score_fn() and
# make_stream_assemble_fn() Protocol methods.


# ---------------------------------------------------------------------------
# TrainDispatcher — assembles batches and dispatches to trainer
# ---------------------------------------------------------------------------

class TrainDispatcher:
    """Assembles scored samples into batches and dispatches to trainer.

    Runs as a thread. Collects batch_size scored samples, assembles
    gen_output + teacher_output via assemble_batch_fn, sends to
    trainer via async_train_fn/wait_train_fn, signals sync_request.
    """

    def __init__(self, scored_queue, shutdown_event, sync_request, sync_ack,
                 step_counter, train_results, batch_size,
                 assemble_batch_fn, async_train_fn, wait_train_fn,
                 tracer=None, tid_train=12, capacity_sem=None,
                 trainer_trace_info=None, group_size=1):
        self.scored_queue = scored_queue
        self.shutdown_event = shutdown_event
        self.sync_request = sync_request
        self.sync_ack = sync_ack
        self.step_counter = step_counter
        self.train_results = train_results
        self.batch_size = batch_size
        self.group_size = group_size
        self.assemble_batch_fn = assemble_batch_fn
        self.async_train_fn = async_train_fn
        self.wait_train_fn = wait_train_fn
        self.tracer = tracer
        self.tid_train = tid_train
        self._trainer_trace_info = trainer_trace_info or {}
        self.capacity_sem = capacity_sem

    def run(self):
        shutdown_event = self.shutdown_event
        scored_queue = self.scored_queue
        batch_size = self.batch_size
        group_size = self.group_size
        tr = self.tracer
        needs_sync_ack = False
        # Persistent buffer for incomplete groups across batches (GRPO only)
        pending = {} if group_size > 1 else None

        print(f"[TrainLoop] started, batch_size={batch_size}, group_size={group_size}",
              flush=True)

        try:
         while not shutdown_event.is_set():
            if group_size > 1:
                # GRPO: collect complete groups to avoid splitting across batches.
                # Samples without prompt_group_id (seed prompts) are passed
                # through directly — assemble_fn rejects them and the capacity
                # release on the None-teacher path prevents deadlock.
                complete_samples = []
                while len(complete_samples) < batch_size:
                    if shutdown_event.is_set():
                        return
                    try:
                        sample = scored_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    gid = sample.get("prompt_group_id")
                    if gid is None:
                        # Seed prompt — pass through for capacity accounting
                        complete_samples.append(sample)
                        continue
                    pending.setdefault(gid, []).append(sample)
                    if len(pending[gid]) == group_size:
                        complete_samples.extend(pending.pop(gid))
                    if tr is not None and len(complete_samples) % 64 == 0 and len(complete_samples) > 0:
                        tr.counter("pipeline", {"scored_ready": len(complete_samples)},
                                   tid=self.tid_train)
                samples = complete_samples
            else:
                # OPD/SFT: collect individual samples as before
                samples = []
                while len(samples) < batch_size:
                    if shutdown_event.is_set():
                        return
                    try:
                        sample = scored_queue.get(timeout=0.5)
                        samples.append(sample)
                        if tr is not None and len(samples) % 64 == 0:
                            tr.counter("pipeline", {"scored_ready": len(samples)},
                                       tid=self.tid_train)
                    except queue.Empty:
                        continue

            if len(samples) < batch_size:
                continue

            if needs_sync_ack:
                self.sync_ack.wait()
                self.sync_ack.clear()
                if shutdown_event.is_set():
                    return

            if tr is not None:
                tr.counter("pipeline", {"scored_ready": batch_size},
                           tid=self.tid_train)

            gen_out, teacher_out = self.assemble_batch_fn(samples)

            if teacher_out is None:
                print("[TrainLoop] teacher padding failed, skipping batch", flush=True)
                # Release permits for consumed samples to prevent deadlock
                if self.capacity_sem is not None:
                    for _ in range(len(samples)):
                        self.capacity_sem.release()
                continue

            self.async_train_fn(gen_out, teacher_out)
            train_result = self.wait_train_fn()

            self.step_counter[0] += 1
            wv = self.step_counter[0]  # 1-indexed step number
            sample_ids = [s.get("worker_id", "?") for s in samples]
            weight_versions = [s.get("weight_version", 0) for s in samples]
            print(f"[TrainLoop] step={wv} weight_versions={weight_versions} "
                  f"workers={sample_ids}", flush=True)

            metrics = (train_result or {}).get("metrics", {})
            train_timing = metrics.get("timing", {})
            if tr is not None:
                train_tok = sum(sum(len(tl) for tl in s.get("full_token_lists", [])) for s in samples)
                tr.emit("train", cat="train", tid=self.tid_train,
                        t_start=train_timing.get("mono_start", time.monotonic()),
                        t_end=train_timing.get("mono_end", time.monotonic()),
                        args={"step": wv, "n_seqs": len(samples),
                              "total_tok": train_tok,
                              **self._trainer_trace_info})

            self.train_results.append((wv, train_result, gen_out, samples))
            # Gate feeder BEFORE releasing permits — prevents feeder from
            # grabbing permits during sync window (AReaL-exact bound).
            self.sync_request.set()
            needs_sync_ack = True

            # Release capacity for consumed samples.  Feeder is now gated
            # on sync_request, so only the coordinator can acquire these.
            if self.capacity_sem is not None:
                for _ in range(len(samples)):
                    self.capacity_sem.release()
        except Exception as e:
            print(f"[TrainLoop] CRASHED: {e}", flush=True)
            import traceback; traceback.print_exc()

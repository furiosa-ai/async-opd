"""OPDMode — on-policy distillation mode for coordinator composition.

Implements the CoordinatorMode protocol for the OPD pipeline:
data iteration, rollout generation, ZMQ teacher scoring, training dispatch,
and step logging. Receives explicit dependencies (no coordinator back-ref).

OPSDMode (self-distillation) is a subclass that overrides teacher scoring
to use rollout-based self-scoring with privileged prompts.
"""

import pickle
import threading
import time
import uuid
from collections import deque
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from opd.data.prompt import PromptDataset, collate_fn
from opd.data.batch_utils import pad_teacher, adapt_response_support
from opd.launch_specs import resolve_teacher_n_logprobs
from opd.trainer.opd import opd_trainer_main
from opd.trainer.teacher_artifact_buffer import estimate_tensor_bytes
from opd.utils.trace import timer


# Trace thread IDs — must match CoordinatorBase constants
TID_ROLLOUT = 10
TID_TEACHER = 11
TID_TRAIN = 12
TID_PIPELINE = 14


class OPDMode:
    """On-policy distillation mode.

    Encapsulates all OPD-specific pipeline operations: data loading,
    rollout generation, teacher scoring, training dispatch, and logging.

    Constructor takes explicit dependencies so it can be composed with
    any scheduler (StepOffScheduler, streaming stages, etc.) without
    a back-reference to the coordinator.
    """

    @classmethod
    def from_coordinator(cls, coordinator):
        """Construct an OPDMode from a coordinator's live state.

        Called in run() after start() — proxies, teacher_client, tracer are ready.
        """
        return cls(
            rollout_proxy=getattr(coordinator, 'rollout_proxy', None),
            teacher_client=getattr(coordinator, 'teacher_client', None),
            trainer_proxy=getattr(coordinator, 'trainer_proxy', None),
            tracer=getattr(coordinator, 'tracer', None),
            opd_config=getattr(coordinator, 'opd_config', None),
            logger=getattr(coordinator, 'logger', None),
            tokenizer=coordinator._init_tokenizer(),
            n_trainer_gpus=getattr(coordinator, '_n_trainer_gpus', 1),
            ray_teacher_actor=getattr(coordinator, '_ray_teacher_actor', None),
            teacher_trace_info=getattr(coordinator, '_teacher_trace_info', {}),
            teacher_artifact_queue=getattr(coordinator, 'teacher_artifact_queue', None),
        )

    def __init__(self, *, rollout_proxy, teacher_client, trainer_proxy,
                 tracer, config=None, opd_config=None, logger=None, tokenizer=None,
                 n_trainer_gpus=1, ray_teacher_actor=None,
                 teacher_trace_info=None, teacher_artifact_queue=None):
        """
        Args:
            rollout_proxy: QueueRolloutProxy for generation commands.
            teacher_client: TeacherClient for ZMQ scoring (None for OPSD).
            trainer_proxy: QueueTrainerProxy for training commands.
            tracer: Tracer for Perfetto spans.
            config: Full config dict with "teacher", "training", "data" keys.
            opd_config: Optional OPDConfig dataclass for typed access.
            logger: Optional JSONL/ClearML logger.
            tokenizer: Pre-configured tokenizer from coordinator (with padding_side="left").
            n_trainer_gpus: Number of trainer FSDP ranks (for token scaling).
            ray_teacher_actor: If set, teacher is a Ray actor (affects timestamps).
            teacher_trace_info: Extra trace args for teacher spans.
        """
        self.rollout_proxy = rollout_proxy
        self.teacher_client = teacher_client
        self.trainer_proxy = trainer_proxy
        self.tracer = tracer
        self.logger = logger
        self._opd_config = opd_config

        self._n_trainer_gpus = n_trainer_gpus
        self._ray_teacher_actor = ray_teacher_actor
        self._teacher_trace_info = teacher_trace_info or {}
        self._teacher_artifact_queue = teacher_artifact_queue

        self._tokenizer = tokenizer

    @property
    def uses_direct_teacher_artifacts(self):
        return (
            self._opd_config is not None
            and self._opd_config.algorithm.opd.teacher_artifact_mode in {"direct", "hidden_recompute"}
        )


    @property
    def uses_hidden_teacher_recompute(self):
        return (
            self._opd_config is not None
            and self._opd_config.algorithm.opd.teacher_artifact_mode == "hidden_recompute"
        )

    @property
    def _uses_rollout_support_topk(self):
        return (
            self._opd_config is not None
            and (
                self._opd_config.algorithm.opd.kl_loss_mode == "thunlp_opd_default_loss"
                or (
                    self._opd_config.algorithm.opd.kl_loss_mode == "reverse_kl_rollout_student_topk"
                    and self._opd_config.algorithm.opd.use_importance_sampling
                )
            )
        )

    @property
    def _uses_multi_sample_policy_gradient_kl(self):
        return (
            self._opd_config is not None
            and self._opd_config.algorithm.opd.kl_loss_mode == "multi_sample_policy_gradient_kl"
        )

    @property
    def _uses_multi_sample_forward_kl(self):
        return (
            self._opd_config is not None
            and self._opd_config.algorithm.opd.kl_loss_mode == "multi_sample_forward_kl"
        )

    @property
    def _uses_mof_opd(self):
        return (
            self._opd_config is not None
            and self._opd_config.algorithm.opd.kl_loss_mode == "mof_opd"
        )

    @property
    def _uses_mof_mc_candidates(self):
        return (
            self._uses_mof_opd
            and int(self._opd_config.algorithm.opd.pg_kl_n_total_samples) > 1
        )

    @property
    def _uses_mof_generated_only(self):
        return (
            self._uses_mof_opd
            and int(self._opd_config.algorithm.opd.pg_kl_n_total_samples) <= 1
        )

    @property
    def _uses_mof_eos_aware(self):
        return (
            self._uses_mof_opd
            and self._opd_config.algorithm.opd.mof_partition == "eos_candidate_rest"
        )

    @property
    def _uses_multi_sample_kl(self):
        return (
            self._uses_multi_sample_policy_gradient_kl
            or self._uses_multi_sample_forward_kl
            or self._uses_mof_mc_candidates
        )

    # ------------------------------------------------------------------ #
    #  Data                                                               #
    # ------------------------------------------------------------------ #

    def data_iterator(self):
        """Yield (epoch, batch_dict) pairs for training."""
        tokenizer = self._get_tokenizer()
        oc = self._opd_config

        ds_kwargs = dict(
            prompt_key=oc.data.prompt_key,
            prompt_template=oc.data.prompt_template,
            enable_thinking=oc.data.enable_thinking,
            prompt_source=oc.data.prompt_source,
            filter_key=oc.data.filter_key,
            filter_value=oc.data.filter_value,
        )
        max_prompt_length = oc.data.max_prompt_length
        train_files = oc.data.train_files
        batch_size = oc.trainer.batch_size
        total_epochs = oc.trainer.total_epochs
        deterministic = oc.deterministic
        seed = oc.seed

        dataset = PromptDataset(
            train_files, tokenizer, max_prompt_length, **ds_kwargs,
        )
        # Deterministic DataLoader shuffle when deterministic mode is active
        dl_kwargs = {}
        if deterministic:
            dl_kwargs["generator"] = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, collate_fn=collate_fn,
                            **dl_kwargs)
        for epoch in range(total_epochs):
            for batch in loader:
                yield epoch, batch

    # ------------------------------------------------------------------ #
    #  Generation                                                         #
    # ------------------------------------------------------------------ #

    def async_generate(self, batch_dict):
        """Submit batch to rollout for generation (non-blocking)."""
        self.rollout_proxy.submit_generate(batch_dict)

    def wait_generate(self):
        """Collect generation result. Returns gen_output dict."""
        return self.rollout_proxy.collect_generate()

    # ------------------------------------------------------------------ #
    #  Teacher scoring                                                    #
    # ------------------------------------------------------------------ #

    def async_teacher(self, gen_output, batch=None):
        """Submit gen_output for ZMQ teacher scoring.

        Returns a SimpleNamespace(get=resolve) future. The background thread
        eagerly resolves ZMQ futures so we capture the real completion time
        (not whenever the run loop calls get()).
        """
        full_lists = gen_output["full_token_lists"]
        prompt_lengths = gen_output.get("prompt_lengths")
        query_indices_response = (
            gen_output.get("mc_query_indices_response")
            if (self._uses_multi_sample_policy_gradient_kl or self._uses_mof_mc_candidates)
            else gen_output.get("query_indices_response")
        )
        if self._uses_mof_mc_candidates:
            query_indices_response = self._mof_teacher_query_indices(query_indices_response)
        n = self.teacher_client.n_workers
        chunk = max((len(full_lists) + n - 1) // n, 1)
        futures = []
        for i in range(0, len(full_lists), chunk):
            if self._uses_rollout_support_topk or self._uses_multi_sample_kl:
                batch_request_ids = [f"q-{uuid.uuid4().hex}" for _ in full_lists[i:i + chunk]]
                submit_kwargs = {
                    "prompt_lengths": prompt_lengths[i:i + chunk].tolist(),
                    "query_request_ids": batch_request_ids,
                }
                if self._uses_multi_sample_forward_kl:
                    submit_kwargs["teacher_mc_n_total_samples"] = (
                        self._opd_config.algorithm.opd.pg_kl_n_total_samples
                    )
                else:
                    submit_kwargs["query_indices_response"] = query_indices_response[i:i + chunk]
                futures.append(self.teacher_client.submit(
                    full_lists[i:i + chunk],
                    **submit_kwargs,
                ))
            else:
                futures.append(self.teacher_client.submit(full_lists[i:i + chunk]))

        t_submit = time.time()
        t_submit_mono = time.monotonic()

        holder = {}

        def _bg_resolve():
            try:
                all_logps, all_idx, all_token_logps = [], [], []
                teacher_mono_start = None
                teacher_mono_end = None
                for f in futures:
                    _, logps, indices, token_logps, ms, me = f.result()
                    all_logps.extend(logps)
                    all_idx.extend(indices)
                    all_token_logps.extend(token_logps)
                    if ms is not None:
                        teacher_mono_start = min(ms, teacher_mono_start or ms)
                    if me is not None:
                        teacher_mono_end = max(me, teacher_mono_end or me)
                holder["t_done"] = time.monotonic()
                holder["dt"] = time.time() - t_submit
                holder["teacher_mono_start"] = teacher_mono_start
                holder["teacher_mono_end"] = teacher_mono_end
                holder["data"] = (all_logps, all_idx, all_token_logps)
            except Exception as e:
                holder["t_done"] = time.monotonic()
                holder["error"] = e

        thread = threading.Thread(target=_bg_resolve, daemon=True)
        thread.start()

        tracer = self.tracer
        teacher_trace_info = self._teacher_trace_info
        ray_teacher_actor = self._ray_teacher_actor

        def resolve():
            with timer() as t_join:
                thread.join()
            if "error" in holder:
                raise RuntimeError(
                    f"Teacher scoring failed: {holder['error']}"
                ) from holder["error"]
            use_remote_ts = ray_teacher_actor is None
            t_ts = (holder.get("teacher_mono_start") if use_remote_ts else None) or t_submit_mono
            t_te = (holder.get("teacher_mono_end") if use_remote_ts else None) or holder["t_done"]
            t_done = holder["t_done"]
            total_tok = sum(len(tl) for tl in full_lists)
            ba = {"n_prompts": len(full_lists), "total_tok": total_tok}
            if batch is not None:
                ba["batch"] = batch
            ba.update(teacher_trace_info)
            tracer.emit("teacher_score", cat="teacher",
                        tid=TID_TEACHER, t_start=t_ts,
                        t_end=t_te, args=ba)
            tracer.emit("wait_teacher", cat="pipeline",
                        tid=TID_PIPELINE,
                        t_start=t_join["mono_start"],
                        t_end=t_join["mono_end"], args=ba)
            if t_te < t_done:
                tracer.emit("zmq_deser", cat="pipeline",
                            tid=TID_PIPELINE,
                            t_start=t_te,
                            t_end=t_done, args=ba)
            all_logps, all_idx, all_token_logps = holder["data"]
            with timer() as t_pad:
                if self._uses_multi_sample_forward_kl:
                    from opd.data.batch_utils import adapt_mc_response_samples
                    out = adapt_mc_response_samples(
                        gen_output,
                        all_idx,
                        all_logps,
                        None,
                    )
                elif self._uses_mof_mc_candidates:
                    from opd.data.batch_utils import adapt_mc_response_samples
                    out = adapt_mc_response_samples(
                        gen_output,
                        query_indices_response,
                        all_logps,
                        None,
                    )
                    if out is not None and self._uses_mof_eos_aware:
                        out["eos_token_id"] = self._mof_eos_token_id()
                elif self._uses_multi_sample_policy_gradient_kl:
                    from opd.data.batch_utils import adapt_mc_response_samples
                    out = adapt_mc_response_samples(
                        gen_output,
                        gen_output["mc_query_indices_response"],
                        all_logps,
                        gen_output["mc_query_old_logprobs_response"],
                    )
                elif self._uses_rollout_support_topk:
                    out = adapt_response_support(
                        gen_output,
                        gen_output["query_indices_response"],
                        all_logps,
                        gen_output.get("query_logprobs_response"),
                    )
                elif self._uses_mof_generated_only:
                    out = pad_teacher(gen_output, all_logps, all_idx,
                                      all_token_logps)
                else:
                    out = pad_teacher(gen_output, all_logps, all_idx,
                                      all_token_logps)
                    if out is not None and self._uses_mof_eos_aware:
                        out["eos_token_id"] = self._mof_eos_token_id()
            tracer.emit("pad_teacher", cat="pipeline",
                        tid=TID_PIPELINE,
                        t_start=t_pad["mono_start"],
                        t_end=t_pad["mono_end"], args=ba)
            if out:
                out["_teacher_seconds"] = holder["dt"]
                out["_pad_seconds"] = t_pad["elapsed"]
                out["_resolve_end"] = time.monotonic()
            return out

        return SimpleNamespace(get=resolve)

    def resolve_teacher(self, teacher_fut, timing, batch=None):
        """Resolve a teacher future and record timing."""
        out = teacher_fut.get() if hasattr(teacher_fut, "get") else teacher_fut
        if isinstance(out, dict) and "_teacher_seconds" in out:
            timing["teacher_seconds"] = out.pop("_teacher_seconds")
        if isinstance(out, dict) and "_pad_seconds" in out:
            timing["pad_seconds"] = out.pop("_pad_seconds")
            print(f"[Pipeline] teacher pad={timing['pad_seconds']:.2f}s", flush=True)
        if isinstance(out, dict) and "_resolve_end" in out and self.tracer is not None:
            t_now = time.monotonic()
            ba = {"batch": batch} if batch is not None else None
            self.tracer.emit("resolve_teacher", cat="pipeline",
                             tid=TID_PIPELINE,
                             t_start=out.pop("_resolve_end"),
                             t_end=t_now, args=ba)
        elif isinstance(out, dict) and "_resolve_end" in out:
            out.pop("_resolve_end", None)
        return out

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def async_train(self, gen_output, teacher_output):
        """Send training batch to trainer subprocess (non-blocking)."""
        self.trainer_proxy.submit_train(gen_output, teacher_output)

    def async_train_direct_teacher_artifacts(
        self,
        gen_output,
        *,
        teacher_buffer_id: int,
        logical_batch_id: int,
        gen_weight_version: int,
        expected_samples: int,
        timeout_s: float = 300.0,
    ):
        """Send generation batch plus trainer-side teacher-buffer reference."""
        self.trainer_proxy.submit_train_direct_teacher_artifacts(
            gen_output,
            teacher_buffer_id=teacher_buffer_id,
            logical_batch_id=logical_batch_id,
            gen_weight_version=gen_weight_version,
            expected_samples=expected_samples,
            timeout_s=timeout_s,
        )

    def async_train_direct_teacher_output(
        self,
        gen_output,
        teacher_output,
        *,
        logical_batch_id: int,
        gen_weight_version: int,
        timeout_s: float = 300.0,
    ):
        """Send canonical teacher output on the trainer artifact channel.

        This is used by the regular step-off scheduler, where teacher scoring
        resolves as a whole canonical batch rather than async per-sample raw
        artifacts.  It gives deterministic HF integration tests the same
        trainer-side buffer/materialization path as the efficient async mode.
        """
        if self._teacher_artifact_queue is None:
            raise RuntimeError("direct teacher artifact queue is not configured")
        expected = int(gen_output["input_ids"].size(0))
        sample_seq_ids = gen_output.get("sample_seq_ids")
        for idx in range(expected):
            payload = self._slice_teacher_payload(teacher_output, idx)
            n_bytes = estimate_tensor_bytes(payload)
            envelope = {
                "schema_version": 1,
                "logical_batch_id": int(logical_batch_id),
                "sample_in_batch_idx": int(idx),
                "sample_seq_id": (
                    sample_seq_ids[idx]
                    if sample_seq_ids is not None and idx < len(sample_seq_ids)
                    else None
                ),
                "train_step": int(logical_batch_id) + 1,
                "gen_weight_version": int(gen_weight_version),
                "n_expected": expected,
                "payload_kind": "canonical_teacher_output",
                "shape": self._shape_meta(payload),
                "dtype": self._dtype_meta(payload),
                "position_spec": {"alignment": "canonical_teacher_batch_slice"},
                "n_tokens": int(gen_output["attention_mask"][idx].sum().item())
                if "attention_mask" in gen_output else 0,
                "n_bytes": n_bytes,
                "payload": payload,
            }
            self._teacher_artifact_queue.put(envelope)
            if self.tracer is not None:
                self.tracer.instant(
                    "teacher_artifact_send",
                    cat="teacher",
                    tid=TID_TEACHER,
                    args={"logical_batch_id": int(logical_batch_id),
                          "sample_in_batch_idx": idx,
                          "n_bytes": n_bytes,
                          "payload_kind": "canonical_teacher_output"},
                )
        self.async_train_direct_teacher_artifacts(
            gen_output,
            teacher_buffer_id=int(logical_batch_id),
            logical_batch_id=int(logical_batch_id),
            gen_weight_version=int(gen_weight_version),
            expected_samples=expected,
            timeout_s=timeout_s,
        )

    @staticmethod
    def _slice_teacher_payload(teacher_output, idx: int) -> dict:
        payload = {}
        for key, value in teacher_output.items():
            if key.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                payload[key] = value[idx:idx + 1]
            elif isinstance(value, list):
                if len(value) > idx:
                    payload[key] = [value[idx]]
            else:
                payload[key] = value
        if not payload:
            raise RuntimeError("teacher output did not contain artifacts")
        return payload

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

    def wait_train(self):
        """Wait for trainer subprocess to finish and return result."""
        self._wait_checkpoint_save()
        return self.trainer_proxy.collect_train()

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #

    def log_train_step(self, step, timing, gen_out, result):
        """Log a completed training step."""
        if result is None:
            return
        if "metrics" not in result:
            print(f"[Pipeline] WARNING: step {step} train result missing 'metrics' "
                  f"(got keys: {list(result.keys())}). Skipping log.", flush=True)
            return
        m = result["metrics"]
        train_secs = m.get("train_seconds", 0)
        iter_seconds = (
            timing.get("generate_seconds", 0)
            + timing.get("teacher_seconds", 0)
            + train_secs
            + timing.get("sync_seconds", 0)
        )
        n_tokens = m.get("n_tokens", 0) * self._n_trainer_gpus

        lr_str = f" lr={m['lr']:.2e}" if "lr" in m else ""
        avg_resp_len = ""
        if "response_lengths" in gen_out:
            avg_resp_len = (f" avg_resp="
                            f"{gen_out['response_lengths'].float().mean():.0f}")
        pg_str = ""
        if "r_mean" in m:
            pg_str = (f" r={m['r_mean']:.3f}±{m['r_std']:.3f}"
                      f" clip={m['clip_frac_high']:.1%}↑{m['clip_frac_low']:.1%}↓"
                      f" adv={m['adv_mean']:.4f}")
        stale_str = ""
        if "staleness_mean" in timing:
            stale_str = f" stale={timing['staleness_mean']}"
        print(
            f"[Step {step}] kl={m.get('kl_loss',0):.4f}{lr_str} "
            f"gen={timing.get('generate_seconds',0):.1f}s "
            f"teach={timing.get('teacher_seconds',0):.1f}s "
            f"train={train_secs:.1f}s "
            f"sync={timing.get('sync_seconds',0):.1f}s"
            f"{avg_resp_len}{pg_str}{stale_str}",
            flush=True,
        )

        if self.logger:
            log_data = {
                "wall_time": time.time(),
                "kl_loss": m.get("kl_loss", 0),
                "grad_norm": m.get("grad_norm", 0),
                "n_tokens": n_tokens,
                "throughput_tok_per_s": (n_tokens / iter_seconds
                                         if iter_seconds > 0 else 0),
                "generate_seconds": timing.get("generate_seconds", 0),
                "teacher_seconds": timing.get("teacher_seconds", 0),
                "train_seconds": train_secs,
                "sync_seconds": timing.get("sync_seconds", 0),
                "iter_seconds": iter_seconds,
            }
            if "response_lengths" in gen_out:
                log_data["avg_response_length"] = (
                    gen_out["response_lengths"].float().mean().item())
            if "lr" in m:
                log_data["lr"] = m["lr"]
            for wk in ("sample_q_depth", "scored_q_depth",
                        "evicted_sample_q", "evicted_scored_q",
                        "staleness_min", "staleness_max",
                        "staleness_mean", "staleness_std"):
                if wk in timing:
                    log_data[wk] = timing[wk]
            for k, v in m.items():
                if k not in log_data and k not in ("timing", "train_seconds"):
                    log_data[k] = v
            self.logger.log_step(step, log_data)

    # ------------------------------------------------------------------ #
    #  Lifecycle queries                                                  #
    # ------------------------------------------------------------------ #

    def needs_teacher(self):
        """Whether this mode requires a teacher process."""
        return True

    def needs_rollout(self):
        """Whether this mode requires rollout worker(s)."""
        return True

    def get_trainer_fn(self):
        """Return (trainer_entry_point, extra_kwargs) for process spawning.

        Returns opd_trainer_main which wraps FSDPBackend with OPDTrainer
        for composition-based loss injection. The KL loss params
        (kl_loss_mode, pg_clip_eps, etc.) come from common_trainer_kwargs
        in process_lifecycle and are accepted by opd_trainer_main's signature.
        """
        return opd_trainer_main

    # ------------------------------------------------------------------ #
    #  Streaming support                                                  #
    # ------------------------------------------------------------------ #

    def make_stream_score_fn(self, teacher_client):
        """Return score_fn that wraps teacher ZMQ scoring for streaming."""
        def score_fn(batch_samples):
            full_token_lists = []
            prompt_lengths = []
            query_indices_response = []
            for s in batch_samples:
                full_token_lists.extend(s["full_token_lists"])
                if self._uses_rollout_support_topk:
                    prompt_lengths.extend(s["prompt_lengths"].tolist())
                    query_indices_response.extend(s["query_indices_response"])
                elif self._uses_multi_sample_policy_gradient_kl:
                    prompt_lengths.extend(s["prompt_lengths"].tolist())
                    query_indices_response.extend(s["mc_query_indices_response"])
                elif self._uses_mof_mc_candidates:
                    prompt_lengths.extend(s["prompt_lengths"].tolist())
                    query_indices_response.extend(
                        self._mof_teacher_query_indices(s["mc_query_indices_response"])
                    )
                elif self._uses_multi_sample_forward_kl:
                    prompt_lengths.extend(s["prompt_lengths"].tolist())
            if self._uses_rollout_support_topk or self._uses_multi_sample_kl:
                query_request_ids = [f"q-{uuid.uuid4().hex}" for _ in full_token_lists]
                submit_kwargs = {
                    "prompt_lengths": prompt_lengths,
                    "query_request_ids": query_request_ids,
                }
                if self._uses_multi_sample_forward_kl:
                    submit_kwargs["teacher_mc_n_total_samples"] = (
                        self._opd_config.algorithm.opd.pg_kl_n_total_samples
                    )
                else:
                    submit_kwargs["query_indices_response"] = query_indices_response
                teacher_fut = teacher_client.submit(full_token_lists, **submit_kwargs)
            else:
                submit_kwargs = {}
                if self.uses_hidden_teacher_recompute:
                    submit_kwargs.update({
                        "return_hidden_states": True,
                        "teacher_hidden_dtype": self._opd_config.algorithm.opd.teacher_hidden_dtype,
                        "teacher_hidden_semantics": self._opd_config.algorithm.opd.teacher_hidden_semantics,
                    })
                teacher_fut = teacher_client.submit(full_token_lists, **submit_kwargs)
            _, logps, indices, token_logps, t_start, t_end = teacher_fut.result()
            offset = 0
            for sample in batch_samples:
                n = len(sample["full_token_lists"])
                if self._uses_multi_sample_forward_kl:
                    sample["teacher_query_logprobs_response"] = logps[offset:offset + n]
                    sample["teacher_mc_indices_response"] = indices[offset:offset + n]
                elif (
                    self._uses_rollout_support_topk
                    or self._uses_multi_sample_policy_gradient_kl
                    or self._uses_mof_mc_candidates
                ):
                    sample["teacher_query_logprobs_response"] = logps[offset:offset + n]
                    if self._uses_mof_eos_aware:
                        sample["eos_token_id"] = self._mof_eos_token_id()
                elif self._uses_mof_generated_only:
                    sample["teacher_topk_logps"] = logps[offset:offset + n]
                    sample["teacher_topk_indices"] = indices[offset:offset + n]
                    sample["teacher_token_logps"] = token_logps[offset:offset + n]
                else:
                    if self.uses_hidden_teacher_recompute:
                        sample["teacher_hidden_states"] = logps[offset:offset + n]
                        sample["teacher_hidden_token_ids"] = indices[offset:offset + n]
                        sample["teacher_hidden_metadata"] = token_logps[offset:offset + n]
                    else:
                        sample["teacher_topk_logps"] = logps[offset:offset + n]
                        sample["teacher_topk_indices"] = indices[offset:offset + n]
                        sample["teacher_token_logps"] = token_logps[offset:offset + n]
                offset += n
            return t_start, t_end
        return score_fn

    def make_stream_assemble_fn(self, max_response_length):
        """Return assemble_batch_fn that splits gen/teacher for streaming."""
        from opd.data.batch_utils import split_gen_teacher, pad_teacher
        def assemble_fn(samples):
            return split_gen_teacher(samples, pad_teacher_fn=pad_teacher)
        return assemble_fn

    @property
    def stream_batch_multiplier(self):
        """Batch multiplier for streaming. OPD: 1 (one response per prompt)."""
        return 1

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _get_tokenizer(self):
        """Return the tokenizer passed from the coordinator."""
        if self._tokenizer is None:
            raise RuntimeError(
                "OPDMode requires a tokenizer — pass tokenizer= to __init__ "
                "or use OPDMode.from_coordinator()")
        return self._tokenizer

    def _mof_eos_token_id(self):
        model_eos = getattr(getattr(self._opd_config, "model", None), "eos_token_id", None)
        if model_eos is not None:
            return int(model_eos)
        tokenizer = self._get_tokenizer()
        tok_eos = getattr(tokenizer, "eos_token_id", None)
        if tok_eos is None:
            raise ValueError("mof_opd eos_candidate_rest requires an eos_token_id")
        return int(tok_eos)

    def _mof_teacher_query_indices(self, query_indices_response):
        """Return MOF teacher query support, appending EOS for EOS-aware MC MOF.

        Rollout MC support already includes the generated token in column 0.
        EOS-aware MOF additionally requests EOS as an extra teacher-scored
        candidate so the loss can form the EOS/candidate/rest partition.
        """
        if not self._uses_mof_eos_aware:
            return query_indices_response
        eos = self._mof_eos_token_id()
        out = []
        for q_idx in query_indices_response:
            if q_idx is None or q_idx.dim() != 2:
                out.append(q_idx)
                continue
            eos_col = torch.full(
                (q_idx.size(0), 1),
                eos,
                dtype=q_idx.dtype,
                device=q_idx.device,
            )
            out.append(torch.cat([q_idx, eos_col], dim=-1))
        return out

    def _wait_checkpoint_save(self):
        """Drain pending checkpoint save result from trainer queue (if any)."""
        if self.trainer_proxy is None:
            return
        result = self.trainer_proxy.collect_checkpoint_save()
        if result is not None:
            if self.tracer and isinstance(result, dict) and "mono_start" in result:
                self.tracer.emit("save_checkpoint", cat="checkpoint",
                                tid=TID_TRAIN,
                                t_start=result["mono_start"],
                                t_end=result["mono_end"])


# ===================================================================== #
#  OPSDMode — on-policy self-distillation                                #
# ===================================================================== #

class OPSDMode(OPDMode):
    """On-Policy Self-Distillation mode.

    Overrides OPDMode to:
    - Include solution_key in dataset loading
    - Stash/discard solutions during generation
    - Score via rollout self-scoring instead of ZMQ teacher
    - Re-inject buffered generates before collecting results
    - Report needs_teacher() as False
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._solution_queue = deque()
        self._opsd_gen_buffer = {}  # {worker_idx: deque(results)}

    @classmethod
    def from_coordinator(cls, coordinator):
        """Construct an OPSDMode from a coordinator's live state."""
        return cls(
            rollout_proxy=getattr(coordinator, 'rollout_proxy', None),
            teacher_client=getattr(coordinator, 'teacher_client', None),
            trainer_proxy=getattr(coordinator, 'trainer_proxy', None),
            tracer=getattr(coordinator, 'tracer', None),
            opd_config=getattr(coordinator, 'opd_config', None),
            logger=getattr(coordinator, 'logger', None),
            tokenizer=coordinator._init_tokenizer(),
            n_trainer_gpus=getattr(coordinator, '_n_trainer_gpus', 1),
            ray_teacher_actor=getattr(coordinator, '_ray_teacher_actor', None),
            teacher_trace_info=getattr(coordinator, '_teacher_trace_info', {}),
        )

    # ------------------------------------------------------------------ #
    #  Data — adds solution_key                                           #
    # ------------------------------------------------------------------ #

    def data_iterator(self):
        """Yield (epoch, batch_dict) pairs for training with solutions."""
        tokenizer = self._get_tokenizer()
        oc = self._opd_config

        ds_kwargs = dict(
            prompt_key=oc.data.prompt_key,
            prompt_template=oc.data.prompt_template,
            enable_thinking=oc.data.enable_thinking,
            prompt_source=oc.data.prompt_source,
            filter_key=oc.data.filter_key,
            filter_value=oc.data.filter_value,
            solution_key=(oc.data.solution_key or oc.data.answer_key or "answer"),
        )
        max_prompt_length = oc.data.max_prompt_length
        train_files = oc.data.train_files
        batch_size = oc.trainer.batch_size
        total_epochs = oc.trainer.total_epochs
        deterministic = oc.deterministic
        seed = oc.seed

        dataset = PromptDataset(
            train_files, tokenizer, max_prompt_length, **ds_kwargs,
        )
        dl_kwargs = {}
        if deterministic:
            dl_kwargs["generator"] = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, collate_fn=collate_fn,
                            **dl_kwargs)
        for epoch in range(total_epochs):
            for batch in loader:
                yield epoch, batch

    # ------------------------------------------------------------------ #
    #  Generation — stash/discard solutions                               #
    # ------------------------------------------------------------------ #

    def async_generate(self, batch_dict):
        """Submit batch, stashing solutions for later scoring."""
        if not batch_dict.get("eval", False):
            sols = batch_dict.pop("solutions", None)
            problems = batch_dict.pop("problem_texts", None)
            self._solution_queue.append((sols, problems))
        else:
            batch_dict.pop("solutions", None)
            batch_dict.pop("problem_texts", None)
        self.rollout_proxy.submit_generate(batch_dict)

    def wait_generate(self):
        """Collect generation result, re-injecting buffered generates first."""
        self._reinject_buffered_generates()
        return self.rollout_proxy.collect_generate()

    # ------------------------------------------------------------------ #
    #  Teacher scoring — self-score via rollout                           #
    # ------------------------------------------------------------------ #

    def async_teacher(self, gen_output, batch=None):
        """OPSD teacher path: score via rollout instead of ZMQ teacher."""
        solutions, problem_texts = self._solution_queue.popleft()
        return self._async_self_score(gen_output, solutions, problem_texts)

    # ------------------------------------------------------------------ #
    #  Lifecycle queries                                                  #
    # ------------------------------------------------------------------ #

    def needs_teacher(self):
        """OPSD does not need a teacher process."""
        return False

    # ------------------------------------------------------------------ #
    #  Internal: self-scoring via rollout                                 #
    # ------------------------------------------------------------------ #

    def _reinject_buffered_generates(self):
        """Re-inject buffered generate results into result queues.

        During _async_self_score.resolve(), generate results found in the
        shared result queue are buffered. Before wait_generate, we put
        them back so collect_generate picks them up normally.
        """
        if not self._opsd_gen_buffer:
            return
        for i in range(self.rollout_proxy.n_workers):
            buf = self._opsd_gen_buffer.get(i)
            if buf:
                r = buf.popleft()
                if not buf:
                    del self._opsd_gen_buffer[i]
                self.rollout_proxy._result_queues[i].put(r)

    def _build_teacher_prompts(self, gen_output, solutions, problem_texts=None):
        """Construct privileged teacher prompts for OPSD self-scoring.

        Each prompt = privileged prefix (problem + solution) + student-generated
        response tokens. Returns (teacher_prompts, prefix_lengths).
        """
        tokenizer = self._get_tokenizer()
        full_token_lists = gen_output["full_token_lists"]
        prompt_lengths = gen_output["prompt_lengths"]

        oc = self._opd_config
        max_model_len = oc.rollout.vllm.max_model_len if oc.rollout else None

        teacher_prompts = []
        prefix_lengths = []
        for i, (full_tokens, sol) in enumerate(zip(full_token_lists, solutions)):
            p_len = prompt_lengths[i] if isinstance(prompt_lengths, (list, tuple)) else prompt_lengths[i].item()
            response_tokens = full_tokens[p_len:]

            if sol and sol.strip():
                problem = problem_texts[i] if problem_texts else ""
                priv_text = (
                    f"Problem: {problem}\n\n"
                    f"Here is a reference solution to this problem:\n"
                    f"=== Reference Solution Begin ===\n{sol}\n=== Reference Solution End ===\n\n"
                    f"After reading the reference solution above, make sure you truly understand "
                    f"the reasoning behind each step — do not copy or paraphrase it. Now, using your "
                    f"own words and independent reasoning, derive the same final answer to the problem above. "
                    f"Think step by step, explore different approaches, and don't be afraid to backtrack "
                    f"or reconsider if something doesn't work out:\n"
                    f"Please reason step by step, and put your final answer within \\boxed{{}}."
                )
                msgs = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": priv_text},
                ]
                chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
                oc = self._opd_config
                teacher_thinking = oc.data.teacher_enable_thinking if oc.data.teacher_enable_thinking is not None else oc.data.enable_thinking
                if teacher_thinking is not None:
                    chat_kwargs["enable_thinking"] = teacher_thinking
                prefix_text = tokenizer.apply_chat_template(msgs, **chat_kwargs)
                prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                teacher_token_ids = prefix_ids + list(response_tokens)
                prefix_lengths.append(len(prefix_ids))
            else:
                raise ValueError(
                    f"[OPSD] Sample {i} has empty/missing solution. "
                    f"OPSD requires non-empty solutions for privileged teacher context. "
                    f"Check your data.solution_key config and dataset columns."
                )

            if max_model_len and len(teacher_token_ids) > max_model_len:
                orig_len = len(teacher_token_ids)
                teacher_token_ids = teacher_token_ids[:max_model_len]
                print(f"[OPSD] Truncating teacher prompt from {orig_len} to {max_model_len} tokens", flush=True)

            teacher_prompts.append(teacher_token_ids)

        return teacher_prompts, prefix_lengths

    def _async_self_score(self, gen_output, solutions, problem_texts=None):
        """Score student-generated tokens using privileged prompts.

        Sends 'score' command to rollout workers, drains result queue
        (buffering any interleaved generate results), trims teacher logprobs
        to response-only, and aligns them to student response positions.
        """
        teacher_prompts, prefix_lengths = self._build_teacher_prompts(
            gen_output, solutions, problem_texts)
        oc = self._opd_config
        n_logprobs = resolve_teacher_n_logprobs(oc) if oc is not None else 256

        self.rollout_proxy.submit_command("score", {
            "prompt_token_ids": teacher_prompts,
            "n_logprobs": n_logprobs,
        })

        t_submit = time.time()
        t_submit_mono = time.monotonic()

        tracer = self.tracer

        def resolve():
            # Collect score results, buffering any generate results encountered.
            results = []
            for i, q in enumerate(self.rollout_proxy._result_queues):
                while True:
                    raw = q.get(timeout=120)
                    r = pickle.loads(raw) if isinstance(raw, bytes) else raw
                    if isinstance(r, dict) and r.get("_cmd") == "score":
                        results.append(r)
                        break
                    else:
                        self._opsd_gen_buffer.setdefault(i, deque()).append(r)

            # Merge results from all workers
            all_logps, all_idx, all_token_logps = [], [], []
            for r in results:
                all_logps.extend(r["teacher_topk_logprobs"])
                all_idx.extend(r["teacher_topk_indices"])
                all_token_logps.extend(r["teacher_token_logps"])

            # Trim to response-only and align to student positions.
            ids = gen_output["input_ids"]
            mask = gen_output["attention_mask"]
            bs, seq_len = ids.shape
            topk = all_logps[0].size(-1) if all_logps and all_logps[0].numel() > 0 else n_logprobs
            student_prompt_lengths = gen_output["prompt_lengths"]

            p_logps = torch.zeros(bs, seq_len, topk, dtype=torch.float32)
            p_idx = torch.zeros(bs, seq_len, topk, dtype=torch.int32)
            valid_mask = torch.zeros(bs, seq_len, dtype=torch.bool)
            p_token_logps = torch.full((bs, seq_len), -1e10, dtype=torch.float32)

            for i in range(bs):
                teacher_prefix_len = prefix_lengths[i]
                resp_start_idx = teacher_prefix_len - 1
                resp_logps = all_logps[i][resp_start_idx:] if all_logps[i].size(0) > resp_start_idx else all_logps[i][:0]
                resp_idx = all_idx[i][resp_start_idx:] if all_idx[i].size(0) > resp_start_idx else all_idx[i][:0]
                resp_tok_logps = all_token_logps[i][resp_start_idx:] if all_token_logps[i].size(0) > resp_start_idx else all_token_logps[i][:0]

                s_plen = student_prompt_lengths[i] if isinstance(student_prompt_lengths, (list, tuple)) else student_prompt_lengths[i].item()
                valid_positions = mask[i].bool().nonzero(as_tuple=True)[0]
                if len(valid_positions) == 0:
                    continue
                prompt_start = valid_positions[0].item()
                place_start = max(prompt_start + s_plen - 1, 0)

                n_resp = min(resp_logps.size(0), seq_len - place_start)
                if n_resp > 0:
                    p_logps[i, place_start:place_start + n_resp] = resp_logps[:n_resp]
                    p_idx[i, place_start:place_start + n_resp] = resp_idx[:n_resp]
                    valid_mask[i, place_start:place_start + n_resp] = True
                    p_token_logps[i, place_start:place_start + n_resp] = resp_tok_logps[:n_resp]

            dt = time.time() - t_submit
            if tracer:
                tracer.emit("teacher_score", cat="teacher",
                            tid=TID_TEACHER, t_start=t_submit_mono,
                            t_end=time.monotonic())

            return {
                "teacher_topk_logps": p_logps,
                "teacher_topk_indices": p_idx,
                "teacher_valid_mask": valid_mask,
                "teacher_token_logps": p_token_logps,
                "_teacher_seconds": dt,
                "_pad_seconds": 0.0,
                "_resolve_end": time.monotonic(),
            }

        return SimpleNamespace(get=resolve)

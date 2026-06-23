"""Base class for pipeline coordinators.

Contains shared infrastructure: config parsing, process lifecycle,
weight sync, data loading, evaluation, and checkpointing.
Subclasses implement run() for specific scheduling modes.
"""

import os

from opd.data.batch_utils import pad_teacher
from opd.utils.gpu_trace import (
    GPUMetricsSampler,
    build_gpu_trace_tracks,
    gpu_trace_interval_from_env,
)
from opd.utils.trace import Tracer
from opd.worker.proxy import (
    QueueRolloutProxy, QueueTrainerProxy, NCCLWeightSyncEngine,
    CPUWeightSyncEngine,
)

from opd.coordinator.process_lifecycle import ProcessLifecycleMixin
from opd.coordinator.evaluation import EvaluationMixin
from opd.coordinator.config_mixin import ConfigMixin


class CoordinatorBase(ProcessLifecycleMixin, EvaluationMixin,
                      ConfigMixin):
    """Base class for pipeline coordinators.

    Contains shared infrastructure: config parsing, process lifecycle,
    weight sync, data loading, evaluation, and checkpointing.
    Subclasses implement run() for specific scheduling modes."""

    # Trace thread IDs — each stage gets its own lane in Perfetto (lower = higher on screen)
    TID_ROLLOUT_BASE = 0   # per-worker: TID_ROLLOUT_BASE + worker_id (0, 1, 2, ...)
    TID_ROLLOUT = 10       # overall generate span
    TID_TEACHER = 11
    TID_TRAIN = 12
    TID_EVAL = 13
    TID_PIPELINE = 14      # orchestration overhead + sync_weights

    def __init__(self, config: dict, logger=None, run_dir=None, mode_cls=None,
                 opd_config=None):
        self.opd_config = opd_config
        self.logger = logger
        self.run_dir = run_dir

        oc = opd_config
        self.step_off = oc.pipeline.n_step_off.step_off
        self.total_steps = oc.trainer.total_steps
        self.total_epochs = oc.trainer.total_epochs
        self.max_response_length = oc.data.max_response_length
        self.max_prompt_length = oc.data.max_prompt_length
        self.batch_size = oc.trainer.batch_size
        self.model_path = oc.model.path
        self.use_nccl = oc.weight_sync.backend == "nccl"
        self.verify_weight_sync = oc.weight_sync.verify_checksum
        self.rollout_quantization = oc.rollout.quantization if oc.rollout else None
        self.save_freq = oc.trainer.save_freq
        self.save_optimizer = oc.trainer.save_optimizer
        self.scheduling_mode = oc.pipeline.scheduling_mode
        self.staleness_threshold = oc.pipeline.fully_async.staleness_threshold
        self.partial_rollout = False
        self.trace_per_sample = False
        kl_mode = oc.algorithm.opd.kl_loss_mode
        resume = oc.trainer.resume_from is not None

        self.teacher_port = None  # assigned dynamically in start()
        self.teacher_proc = None
        self.rollout_procs = []
        self.trainer_proc = None
        self.rollout_cmd_queues = []
        self.rollout_result_queues = []
        self.trainer_cmd_queue = None
        self.trainer_result_queue = None

        # Proxy objects (constructed in start() after spawning workers)
        self.rollout_proxy: QueueRolloutProxy | None = None
        self.trainer_proxy: QueueTrainerProxy | None = None
        self.weight_engine: NCCLWeightSyncEngine | CPUWeightSyncEngine | None = None

        # Validate: partial_rollout + PG-KL variants are incompatible
        if self.partial_rollout and kl_mode in {
            "policy_gradient_kl",
            "thunlp_opd_default_loss",
            "multi_sample_policy_gradient_kl",
        }:
            raise ValueError(
                f"partial_rollout=true is incompatible with {kl_mode}: "
                "stale student logprobs break importance sampling ratio. "
                "Use partial_rollout=false (default) with PG-KL.")

        # Mode object (set by subclass run() — e.g. OPDMode, GRPOMode)
        self._mode = None
        self._mode_cls = mode_cls

        stream_path = os.path.join(run_dir, "trace_live.json") if run_dir else None
        self.tracer = Tracer(stream_path=stream_path, resume=resume)
        self._gpu_metrics_sampler = self._init_gpu_trace_sampler()

        # Config validation happens on the typed OPDConfig before coordinator dispatch.

    def _init_gpu_trace_sampler(self):
        """Start background GPU metrics sampling for Perfetto counters."""
        teacher_gpu_ids = self._gpu_ids("teacher") if self.opd_config.teacher else None
        rollout_gpu_ids = self._gpu_ids("rollout") if self.opd_config.rollout else None
        trainer_gpu_ids = self._gpu_ids("trainer")
        tracks = build_gpu_trace_tracks(
            teacher_gpu_ids=teacher_gpu_ids,
            rollout_gpu_ids=rollout_gpu_ids,
            trainer_gpu_ids=trainer_gpu_ids,
        )
        sampler = GPUMetricsSampler(
            self.tracer,
            tracks,
            interval_sec=gpu_trace_interval_from_env(),
        )
        sampler.start()
        return sampler

    def stop_trace_monitors(self):
        """Stop background trace samplers before saving trace.json."""
        if self._gpu_metrics_sampler is not None:
            self._gpu_metrics_sampler.stop()

    # ------------------------------------------------------------------ #
    #  Resume helper                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _skip_data_for_resume(data_iter, resume_step, label="Pipeline"):
        """Fast-forward a data iterator past already-trained steps."""
        if resume_step > 0:
            print(f"[{label}] Skipping {resume_step} batches for resume...",
                  flush=True)
            for _ in range(resume_step):
                try:
                    next(data_iter)
                except StopIteration:
                    break

    # ------------------------------------------------------------------ #
    #  Async primitives                                                   #
    # ------------------------------------------------------------------ #

    def _async_generate(self, batch_dict):
        """Submit batch to rollout for generation. Delegates to mode."""
        if getattr(self, '_mode', None) is not None:
            return self._mode.async_generate(batch_dict)
        raise NotImplementedError("No mode set — use a coordinator subclass")

    def _wait_generate(self):
        """Collect generation result. Delegates to mode."""
        if getattr(self, '_mode', None) is not None:
            return self._mode.wait_generate()
        raise NotImplementedError("No mode set — use a coordinator subclass")

    def _async_teacher(self, gen_output, batch=None):
        """Submit gen_output for scoring. Delegates to mode."""
        if getattr(self, '_mode', None) is not None:
            return self._mode.async_teacher(gen_output, batch=batch)
        raise NotImplementedError("No mode set — use a coordinator subclass")

    def _pad_teacher(self, gen_output, all_logps, all_indices,
                     all_token_logps=None):
        return pad_teacher(gen_output, all_logps, all_indices, all_token_logps)

    def _sync_weights(self):
        """Trigger weight sync between trainer and rollout workers."""
        self._wait_checkpoint_save()
        return self.weight_engine.sync(self.trainer_proxy, self.rollout_proxy)

    def _verify_weight_checksums(self):
        """Compare weight checksums between trainer and all rollout workers."""
        self.weight_engine.verify_checksums(self.trainer_proxy, self.rollout_proxy)

    def _sync_weights_paused(self, forward_target=None):
        """Weight sync while rollout workers are paused (streaming path).

        Delegates to weight_engine.sync_paused() which uses drain_until_status
        to handle interleaved data + status messages on result queues.
        """
        self._wait_checkpoint_save()
        return self.weight_engine.sync_paused(
            self.trainer_proxy, self.rollout_proxy,
            forward_target=forward_target,
        )

    def _resolve_teacher(self, teacher_fut, timing, batch=None):
        """Resolve a teacher future and record timing. Delegates to mode."""
        if getattr(self, '_mode', None) is not None:
            return self._mode.resolve_teacher(teacher_fut, timing, batch=batch)
        raise NotImplementedError("No mode set — use a coordinator subclass")

    def _train_step(self, gen_output, teacher_output):
        """Synchronous train step (send + wait)."""
        self._async_train(gen_output, teacher_output)
        return self._wait_train()

    def _async_train(self, gen_output, teacher_output):
        """Send training batch to trainer subprocess (non-blocking). Delegates to mode."""
        if getattr(self, '_mode', None) is not None:
            return self._mode.async_train(gen_output, teacher_output)
        raise NotImplementedError("No mode set — use a coordinator subclass")

    def _wait_train(self):
        """Wait for trainer subprocess to finish and return result. Delegates to mode."""
        if getattr(self, '_mode', None) is not None:
            return self._mode.wait_train()
        raise NotImplementedError("No mode set — use a coordinator subclass")

    # ------------------------------------------------------------------ #
    #  Data                                                               #
    # ------------------------------------------------------------------ #

    def _data_iterator(self):
        """Yield (epoch, batch) pairs for training. Delegates to mode."""
        if getattr(self, '_mode', None) is not None:
            yield from self._mode.data_iterator()
            return
        raise NotImplementedError("No mode set — use a coordinator subclass")

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #

    def _log_train_step(self, step, timing, gen_out, result):
        """Log a completed training step. Delegates to mode."""
        if getattr(self, '_mode', None) is not None:
            return self._mode.log_train_step(step, timing, gen_out, result)
        raise NotImplementedError("No mode set — use a coordinator subclass")

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                      #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, step):
        """Save model checkpoint at the given step (non-blocking)."""
        self._wait_checkpoint_save()  # drain any previous pending save
        checkpoint_dir = os.path.join(self.run_dir, "checkpoints", f"step_{step}")
        self.trainer_proxy.submit_save_checkpoint(step, checkpoint_dir, self.save_optimizer)
        print(f"[Pipeline] Checkpoint save dispatched: {checkpoint_dir}", flush=True)

    def _wait_checkpoint_save(self):
        """Drain pending checkpoint save result from trainer queue (if any)."""
        if self.trainer_proxy is None:
            return None
        result = self.trainer_proxy.collect_checkpoint_save()
        if result is not None:
            tr = getattr(self, "tracer", None)
            if tr and isinstance(result, dict) and "mono_start" in result:
                tr.emit("save_checkpoint", cat="checkpoint",
                        tid=self.TID_TRAIN,
                        t_start=result["mono_start"],
                        t_end=result["mono_end"])
        return result

    def _load_checkpoint(self, checkpoint_dir):
        """Load checkpoint into trainer, returns resumed step number."""
        self._wait_checkpoint_save()
        step = self.trainer_proxy.load_checkpoint(checkpoint_dir)
        # Sync restored weights to rollout workers
        self._sync_weights()
        print(f"[Pipeline] Resumed from checkpoint step {step}: {checkpoint_dir}",
              flush=True)
        return step

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint directory, or None."""
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            return None
        steps = []
        for d in os.listdir(ckpt_dir):
            if d.startswith("step_"):
                try:
                    steps.append(int(d.split("_")[1]))
                except (ValueError, IndexError):
                    pass
        if not steps:
            return None
        latest = max(steps)
        return os.path.join(ckpt_dir, f"step_{latest}")

    # ------------------------------------------------------------------ #
    #  Abstract                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        raise NotImplementedError

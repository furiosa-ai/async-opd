"""SFT (Supervised Fine-Tuning) coordinator and mode.

Minimal coordinator that trains on (prompt, completion) pairs with
causal LM cross-entropy loss. No teacher, rollout, or weight sync.

SFTMode implements the CoordinatorMode protocol for SFT: data iteration,
direct queue-based training dispatch, and step logging.

Usage:
    python run.py --config configs/examples/sft_qwen3_1.7b.yaml --overwrite
"""

import multiprocessing as mp
import time

from torch.utils.data import DataLoader

from opd.coordinator.base import CoordinatorBase
from opd.utils.config import resolve_trust_remote_code
from opd.utils.net import find_free_port, release_all_port_leases
from opd.data.sft import SFTDataset, make_sft_collate_fn
from opd.trainer.sft import sft_trainer_main
from opd.worker.proxy import QueueTrainerProxy


# Trace thread IDs — must match CoordinatorBase constants
TID_TRAIN = 12


class SFTMode:
    """Supervised fine-tuning mode.

    Encapsulates all SFT-specific pipeline operations: data loading,
    training dispatch via direct queue put, and logging.

    Constructor takes explicit dependencies so it can be composed with
    any scheduler without a back-reference to the coordinator.
    """

    def __init__(self, *, trainer_cmd_queue, trainer_result_queue,
                 tracer, config=None, opd_config=None, logger=None):
        """
        Args:
            trainer_cmd_queue: mp.Queue for sending train commands.
            trainer_result_queue: mp.Queue for receiving train results.
            tracer: Tracer for Perfetto spans.
            config: Full config dict with "training", "data" keys.
            opd_config: Optional OPDConfig dataclass for typed access.
            logger: Optional JSONL/ClearML logger.
        """
        self.trainer_cmd_queue = trainer_cmd_queue
        self.trainer_result_queue = trainer_result_queue
        self.tracer = tracer
        self.logger = logger
        self._opd_config = opd_config

        # Cached tokenizer (lazy init)
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    #  Data                                                               #
    # ------------------------------------------------------------------ #

    def data_iterator(self):
        """Yield (epoch, batch_dict) pairs for training."""
        tokenizer = self._get_tokenizer()
        oc = self._opd_config

        if oc is not None:
            max_prompt_length = oc.data.max_prompt_length
            max_response_length = oc.data.max_response_length
            dataset = SFTDataset(
                path=oc.data.train_files,
                tokenizer=tokenizer,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                prompt_key=oc.data.prompt_key,
                completion_key=oc.data.completion_key or "completion",
                prompt_template=oc.data.prompt_template,
                enable_thinking=oc.data.enable_thinking,
                allow_pickle_teacher_logits=oc.data.allow_pickle_teacher_logits,
            )
            batch_size = oc.trainer.batch_size
            total_epochs = oc.trainer.total_epochs
        else:
            raise RuntimeError("SFTMode.data_iterator requires opd_config")
        print(f"[SFT] Training dataset: {len(dataset)} samples", flush=True)
        loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, drop_last=True,
            collate_fn=make_sft_collate_fn(tokenizer.pad_token_id),
        )
        for epoch in range(total_epochs):
            for batch in loader:
                yield epoch, batch

    # ------------------------------------------------------------------ #
    #  Generation (not used in pure SFT)                                  #
    # ------------------------------------------------------------------ #

    def async_generate(self, batch_dict):
        """Not used in pure SFT mode."""
        raise NotImplementedError("SFT mode does not use rollout generation")

    def wait_generate(self):
        """Not used in pure SFT mode."""
        raise NotImplementedError("SFT mode does not use rollout generation")

    # ------------------------------------------------------------------ #
    #  Scoring (not used in pure SFT)                                     #
    # ------------------------------------------------------------------ #

    def async_teacher(self, gen_output, batch=None):
        """Not used in pure SFT mode."""
        raise NotImplementedError("SFT mode does not use teacher scoring")

    def resolve_teacher(self, future, timing, batch=None):
        """Not used in pure SFT mode."""
        raise NotImplementedError("SFT mode does not use teacher scoring")

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def async_train(self, gen_output, teacher_output):
        """Send batch to trainer via direct queue put.

        For SFT, gen_output IS the batch (no separate gen/teacher phases).
        teacher_output is ignored.
        """
        gen_output["_send_mono"] = time.monotonic()
        self.trainer_cmd_queue.put(("train", gen_output))

    def wait_train(self):
        """Collect training result from trainer subprocess."""
        return self.trainer_result_queue.get()

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #

    def log_train_step(self, step, timing, gen_out, result):
        """Log SFT training step metrics."""
        if result is None:
            return
        metrics = result.get("metrics", {})
        loss = metrics.get("kl_loss", 0)  # reused key for compat
        lr = metrics.get("lr", 0)
        n_tok = metrics.get("n_tokens", 0)
        grad_norm = metrics.get("grad_norm", 0)
        train_s = metrics.get("train_seconds", 0)

        print(f"[Step {step}] loss={loss:.4f} lr={lr:.2e} "
              f"train={train_s:.1f}s n_tok={n_tok} grad_norm={grad_norm:.2f}",
              flush=True)

        if self.logger:
            self.logger.log_step(step, {
                "type": "train",
                "step": step,
                "kl_loss": loss,
                "lr": lr,
                "n_tokens": n_tok,
                "grad_norm": grad_norm,
                "train_seconds": train_s,
            })

    # ------------------------------------------------------------------ #
    #  Lifecycle queries                                                  #
    # ------------------------------------------------------------------ #

    def needs_teacher(self):
        """Whether this mode requires a teacher process."""
        return False

    def needs_rollout(self):
        """Whether this mode requires rollout worker(s)."""
        return False

    def get_trainer_fn(self):
        """Return trainer_entry_point for process spawning.

        All config is in the trainer config dict — no extra kwargs needed.
        """
        return sft_trainer_main

    def make_stream_score_fn(self, teacher_client):
        raise NotImplementedError("SFT does not support streaming")

    def make_stream_assemble_fn(self, max_response_length):
        raise NotImplementedError("SFT does not support streaming")

    @property
    def stream_batch_multiplier(self):
        return 1

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _get_tokenizer(self):
        """Lazy-init and cache tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            oc = self._opd_config
            if oc is None:
                raise RuntimeError("SFTMode._get_tokenizer requires opd_config")
            model_path = oc.model.path
            tok_path = oc.data.tokenizer_path or model_path
            self._tokenizer = AutoTokenizer.from_pretrained(
                tok_path,
                trust_remote_code=resolve_trust_remote_code(
                    oc.model.trust_remote_code,
                    context="SFT tokenizer loading",
                ),
            )
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer


class SFTCoordinator(CoordinatorBase):
    """Coordinator for standalone SFT training.

    Inherits shared infra from CoordinatorBase (tokenizer, checkpoints,
    logger, tracer) but overrides lifecycle methods to skip teacher/rollout.
    """

    def __init__(self, config: dict, *, opd_config=None, **kwargs):
        # Guard: inject dummy teacher so CoordinatorBase.__init__ doesn't crash
        config.setdefault("teacher", {"model": "_sft_placeholder_"})
        super().__init__(config, opd_config=opd_config, **kwargs)
        self._trainer_fsdp_procs = []

    def _gpu_ids(self, role):
        """Override: SFT has no teacher/rollout — all GPUs go to trainer."""
        if role == "trainer":
            oc = self.opd_config
            if oc.trainer.gpu_ids:
                return str(oc.trainer.gpu_ids)
            n = oc.trainer.n_gpus or 1
            return ",".join(str(i) for i in range(n))
        return ""  # no teacher or rollout GPUs

    def start(self):
        """Launch trainer subprocess(es) only."""
        self._setup_env()

        ctx = mp.get_context("spawn")
        trainer_gpus = self._gpu_ids("trainer")
        n_trainer_gpus = len(trainer_gpus.split(","))

        self.trainer_cmd_queue = ctx.Queue()
        self.trainer_result_queue = ctx.Queue()

        if n_trainer_gpus > 1:
            fsdp_master_port = find_free_port("sft.fsdp_master")
            self._trainer_fsdp_procs = []
            for fsdp_rank in range(n_trainer_gpus):
                rank_info = self._build_fsdp_rank_info(
                    fsdp_rank, n_trainer_gpus, fsdp_master_port)
                trainer_spec = self._build_trainer_launch_spec(rank_info)
                cmd_q = self.trainer_cmd_queue if fsdp_rank == 0 else None
                res_q = self.trainer_result_queue if fsdp_rank == 0 else None
                p = ctx.Process(
                    target=sft_trainer_main,
                    args=(trainer_spec, cmd_q, res_q, None))
                p.start()
                self._trainer_fsdp_procs.append(p)
            self.trainer_proc = self._trainer_fsdp_procs[0]
        else:
            self._trainer_fsdp_procs = []
            rank_info = self._build_fsdp_rank_info(0, 1, None)
            trainer_spec = self._build_trainer_launch_spec(rank_info)
            self.trainer_proc = ctx.Process(
                target=sft_trainer_main,
                args=(trainer_spec, self.trainer_cmd_queue,
                      self.trainer_result_queue, None))
            self.trainer_proc.start()

        self.trainer_proxy = QueueTrainerProxy(
            cmd_queue=self.trainer_cmd_queue,
            result_queue=self.trainer_result_queue,
            proc=self.trainer_proc,
            fsdp_procs=self._trainer_fsdp_procs,
        )

        print(f"[SFT] Trainer started: {n_trainer_gpus} GPU(s) on {trainer_gpus}",
              flush=True)
        time.sleep(2)

    def _evaluate(self, global_step):
        """Override: run perplexity eval inside the trainer (no rollout workers).

        Sends an 'evaluate' command to the trainer which computes CE loss
        on the validation SFT dataset (forward pass only, fast).
        """
        oc = self.opd_config
        eval_config = {
            "val_files": oc.data.val_files,
            "tokenizer_path": self.model_path,
            "prompt_key": oc.data.prompt_key,
            "completion_key": oc.data.completion_key or "completion",
            "prompt_template": oc.data.prompt_template,
            "enable_thinking": oc.data.enable_thinking,
            "max_prompt_length": self.max_prompt_length,
            "max_response_length": self.max_response_length,
            "trust_remote_code": oc.model.trust_remote_code,
            "allow_pickle_teacher_logits": oc.data.allow_pickle_teacher_logits,
        }

        if not eval_config["val_files"]:
            print(f"[Eval @ step {global_step}] No val_files configured, skipping.",
                  flush=True)
            return

        print(f"[Eval @ step {global_step}] Running perplexity eval...", flush=True)

        with self.tracer.span("eval", cat="eval", tid=self.TID_EVAL) as sp:
            sp["step"] = global_step
            self.trainer_cmd_queue.put(("evaluate", eval_config))
            result = self.trainer_result_queue.get()

        metrics = result.get("eval_metrics", {})
        val_loss = metrics.get("val_loss", 0)
        ppl = metrics.get("perplexity", 0)

        print(f"[Eval @ step {global_step}] val_loss={val_loss:.4f} "
              f"perplexity={ppl:.2f}", flush=True)

        if self.logger:
            self.logger.log_eval(global_step, metrics)

    def _load_checkpoint(self, checkpoint_dir):
        """Override: load checkpoint without _sync_weights (no rollout workers)."""
        self._wait_checkpoint_save()
        step = self.trainer_proxy.load_checkpoint(checkpoint_dir)
        print(f"[Pipeline] Resumed from checkpoint step {step}: {checkpoint_dir}",
              flush=True)
        return step

    def run(self):
        """SFT training loop — simple epoch/batch iteration."""
        resume_step, test_freq, eval_modes, val_before_train = self._prepare_run()

        # Build SFTMode for data iteration and logging
        mode = SFTMode(
            trainer_cmd_queue=self.trainer_cmd_queue,
            trainer_result_queue=self.trainer_result_queue,
            tracer=self.tracer,
            opd_config=self.opd_config,
            logger=self.logger,
        )

        step = resume_step
        tr = self.tracer

        for epoch, batch in mode.data_iterator():
            if step >= self.total_steps:
                break
            step += 1

            # Drain any pending checkpoint save result before sending train
            self._wait_checkpoint_save()

            # Send batch to trainer via mode
            mode.async_train(batch, None)

            # Collect result
            result = mode.wait_train()
            metrics = result.get("metrics", {})
            timing = metrics.get("timing", {})

            # Trace
            if tr:
                tr.emit("train", cat="train", tid=self.TID_TRAIN,
                        t_start=timing.get("mono_start", time.monotonic()),
                        t_end=timing.get("mono_end", time.monotonic()),
                        args={"step": step, "n_tokens": metrics.get("n_tokens", 0)})

            # Log via mode
            mode.log_train_step(step, timing, batch, result)

            # Eval + checkpoint
            if test_freq > 0 and step % test_freq == 0:
                if "perplexity" in eval_modes:
                    self._wait_checkpoint_save()  # drain any pending save
                    self._evaluate(step)
                self._save_checkpoint(step)
            elif self.save_freq > 0 and step % self.save_freq == 0:
                self._save_checkpoint(step)

        # Final checkpoint if not already saved
        if step > 0 and (self.save_freq <= 0 or step % self.save_freq != 0):
            if test_freq <= 0 or step % test_freq != 0:
                self._save_checkpoint(step)

        self._wait_checkpoint_save()
        print(f"[SFT] Training complete: {step} steps", flush=True)

    def shutdown(self):
        """Shut down trainer process(es)."""
        self.stop_trace_monitors()
        self._wait_checkpoint_save()
        if self.trainer_proxy:
            self.trainer_proxy.shutdown()
        if self.trainer_proc:
            self.trainer_proc.join(timeout=60)
        for p in self._trainer_fsdp_procs[1:]:
            p.join(timeout=5)
        release_all_port_leases()

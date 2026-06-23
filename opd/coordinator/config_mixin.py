"""Configuration parsing and run preparation for coordinator."""

import os

from transformers import AutoTokenizer

from opd.data.prompt import PromptDataset
from opd.utils.config import resolve_trust_remote_code


class ConfigMixin:
    """Configuration parsing and run preparation for coordinator.

    Required attributes from host class:
        self.opd_config (OPDConfig), self.model_path, self.max_prompt_length,
        self.max_response_length, self.batch_size
    """

    @staticmethod
    def _n_gpus_from(cfg):
        """Derive GPU count from a config section with n_gpus/gpu_ids."""
        if cfg is None:
            return 1
        if cfg.n_gpus is not None:
            return cfg.n_gpus
        if cfg.gpu_ids is not None:
            return len([x for x in cfg.gpu_ids.split(",") if x.strip()])
        return 1

    def _gpu_ids(self, role):
        oc = getattr(self, 'opd_config', None)
        if oc is None:
            raise RuntimeError("_gpu_ids requires opd_config")
        if role == "teacher":
            if oc.teacher and oc.teacher.gpu_ids:
                return str(oc.teacher.gpu_ids)
        elif role == "rollout":
            if oc.rollout and oc.rollout.gpu_ids:
                return str(oc.rollout.gpu_ids)
        elif role == "trainer":
            if oc.trainer.gpu_ids:
                return str(oc.trainer.gpu_ids)

        # Default sequential assignment
        teacher_n = self._n_gpus_from(oc.teacher) if oc.teacher else 1
        rollout_n = self._n_gpus_from(oc.rollout) if oc.rollout else 1
        trainer_n = self._n_gpus_from(oc.trainer)
        if role == "teacher":
            return ",".join(str(i) for i in range(teacher_n))
        elif role == "rollout":
            s = teacher_n
            return ",".join(str(i) for i in range(s, s + rollout_n))
        elif role == "trainer":
            s = teacher_n + rollout_n
            return ",".join(str(i) for i in range(s, s + trainer_n))

    def _compute_total_steps(self):
        """Estimate total training steps from config or dataset size."""
        oc = getattr(self, 'opd_config', None)
        explicit = oc.trainer.total_steps
        if explicit and explicit < int(1e9):
            return explicit
        try:
            tokenizer = self._init_tokenizer()
            dataset = PromptDataset(
                oc.data.train_files, tokenizer, self.max_prompt_length,
                prompt_key=oc.data.prompt_key,
                enable_thinking=oc.data.enable_thinking,
                prompt_source=oc.data.prompt_source,
                filter_key=oc.data.filter_key,
                filter_value=oc.data.filter_value,
            )
            steps_per_epoch = len(dataset) // self.batch_size
            return steps_per_epoch * self.total_epochs
        except Exception as e:
            print(f"[Pipeline] Could not estimate total steps: {e}", flush=True)
            return 0

    def _get_backend(self):
        oc = getattr(self, 'opd_config', None)
        return oc.trainer.backend

    def _prepare_run(self):
        """Shared preamble for all run() implementations.

        Reads eval config, auto-sets save_freq for post-eval modes, initializes
        the tokenizer, handles resume-from-checkpoint, and runs the optional
        pre-train validation.

        Returns:
            tuple: (resume_step, test_freq, eval_mode, val_before_train)
                resume_step     -- 0 if fresh run, else the step we resumed from
                test_freq       -- eval frequency (-1 = never)
                eval_mode       -- "inline" | "post" | "post_allgpu"
                val_before_train -- whether to eval at step 0
        """
        oc = getattr(self, 'opd_config', None)
        test_freq = oc.eval.freq
        eval_modes = set(oc.eval.mode)
        val_before_train = oc.eval.before_train
        resume_from = oc.trainer.resume_from

        # Post-training eval: auto-save checkpoints at test_freq
        if eval_modes & {"post", "post_allgpu"}:
            if self.save_freq <= 0 and test_freq > 0:
                self.save_freq = test_freq
                print(f"[Pipeline] eval_mode={eval_modes}: auto-setting save_freq={test_freq}",
                      flush=True)

        self._init_tokenizer()

        # Resume from checkpoint
        resume_step = 0
        if resume_from == "latest":
            ckpt_dir = self._find_latest_checkpoint()
            if ckpt_dir:
                resume_step = self._load_checkpoint(ckpt_dir)
        elif resume_from and os.path.isdir(resume_from):
            resume_step = self._load_checkpoint(resume_from)

        if val_before_train and test_freq != -1 and eval_modes & {"inline", "perplexity"} and resume_step == 0:
            self._evaluate(0)

        return resume_step, test_freq, eval_modes, val_before_train

    def _init_tokenizer(self):
        """Initialize and cache the tokenizer.

        Uses `data.tokenizer_path` if set, otherwise falls back to the student
        model path. Useful when the student is a base model without a chat
        template — set tokenizer_path to the teacher (instruct) model.
        """
        if not hasattr(self, "_tokenizer") or self._tokenizer is None:
            oc = getattr(self, 'opd_config', None)
            tok_path = oc.data.tokenizer_path or self.model_path
            self._tokenizer = AutoTokenizer.from_pretrained(
                tok_path,
                trust_remote_code=resolve_trust_remote_code(
                    oc.model.trust_remote_code,
                    context="coordinator tokenizer loading",
                ),
            )
            # Left padding required: rollout code assumes padding is at the start
            # of the sequence (pad_len = max_prompt - prompt_len, real tokens
            # start at pad_len). Many decoder tokenizers default to right padding.
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

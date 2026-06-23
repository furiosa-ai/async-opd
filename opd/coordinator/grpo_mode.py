"""GRPOMode — GRPO training mode for coordinator composition.

Implements the CoordinatorMode protocol for GRPO (Group Relative Policy
Optimization): data iteration with ground truth threading, rollout generation
with group sampling, reward computation + optional reference model KL scoring,
GRPO batch assembly with advantages, and step logging.

Also contains GRPOPromptDataset and grpo_collate_fn data utilities.

Receives explicit dependencies (no coordinator back-ref).
"""

import threading
import time
from collections import deque
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from opd.data.prompt import format_prompt, load_dataframe
from opd.data.batch_utils import pad_teacher
from opd.reward import (
    get_reward_fn, compute_group_advantages, apply_overlong_penalty,
    filter_zero_variance_groups,
)
from opd.trainer.config import GRPOConfig
from opd.utils.trace import timer


# Trace thread IDs — must match CoordinatorBase constants
TID_ROLLOUT = 10
TID_TEACHER = 11
TID_TRAIN = 12
TID_PIPELINE = 14


# ---------------------------------------------------------------------------
# GRPO Prompt Dataset — PromptDataset + ground truths
# ---------------------------------------------------------------------------

class GRPOPromptDataset(Dataset):
    """PromptDataset extended with ground truth answers for reward computation."""

    def __init__(self, path, tokenizer, max_prompt_length,
                 prompt_key="prompt", answer_key="answer",
                 prompt_template=None, enable_thinking=None):
        self.df = load_dataframe(path)
        self.raw_prompts = self.df[prompt_key].tolist()
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_template = prompt_template
        self.enable_thinking = enable_thinking

        # Extract ground truths (reuse ValDataset pattern)
        if answer_key == "auto":
            if "answer" in self.df.columns:
                answer_key = "answer"
            elif "reward_model" in self.df.columns:
                import json
                reward_col = self.df["reward_model"].tolist()
                self.ground_truths = []
                for rm in reward_col:
                    if isinstance(rm, dict):
                        self.ground_truths.append(str(rm.get("ground_truth", "")))
                    elif isinstance(rm, str):
                        try:
                            d = json.loads(rm)
                            self.ground_truths.append(str(d.get("ground_truth", "")))
                        except (json.JSONDecodeError, AttributeError):
                            self.ground_truths.append("")
                    else:
                        self.ground_truths.append("")
                answer_key = None  # already extracted
            else:
                raise ValueError(
                    f"Cannot auto-detect answer column. "
                    f"Available: {list(self.df.columns)}. Set answer_key.")

        if answer_key is not None:
            self.ground_truths = [str(a).strip() for a in self.df[answer_key].tolist()]

    def __len__(self):
        return len(self.raw_prompts)

    def __getitem__(self, idx):
        text = format_prompt(self.raw_prompts[idx], self.tokenizer,
                              self.prompt_template,
                              enable_thinking=self.enable_thinking)
        encoded = self.tokenizer(
            text,
            max_length=self.max_prompt_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "ground_truth": self.ground_truths[idx],
        }


def grpo_collate_fn(batch):
    """Collate for GRPO: stack tensors, pass ground truths as list."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "ground_truths": [b["ground_truth"] for b in batch],
    }


# ---------------------------------------------------------------------------
# GRPO Mode
# ---------------------------------------------------------------------------

class GRPOMode:
    """GRPO training mode.

    Encapsulates all GRPO-specific pipeline operations: data loading with
    ground truth FIFO, rollout generation with group sampling, reward
    computation + optional reference model scoring, training dispatch
    with advantages, and logging.

    Constructor takes explicit dependencies so it can be composed with
    any scheduler (StepOffScheduler, streaming stages, etc.) without
    a back-reference to the coordinator.
    """

    @classmethod
    def from_coordinator(cls, coordinator):
        """Construct a GRPOMode from a coordinator's live state.

        Called in run() after start() — proxies are ready. Parses all GRPO
        algorithm config from the coordinator's config dict.
        """
        from opd.reward import get_reward_fn
        from opd.coordinator.streaming import StreamCoordinator

        oc = getattr(coordinator, 'opd_config', None)
        if oc is not None:
            g = oc.algorithm.grpo
            reward_fn_name = g.reward_fn
            clip_ratio_low = float(g.clip_ratio_low) if g.clip_ratio_low is not None else None
            clip_ratio_high = float(g.clip_ratio_high) if g.clip_ratio_high is not None else None
            clip_ratio_c = float(g.clip_ratio_c) if g.clip_ratio_c is not None else None
            answer_pattern = oc.algorithm.reward.answer_pattern
            grpo_group_size = g.group_size
            grpo_clip_eps = float(g.clip_eps)
            grpo_kl_beta = float(g.kl_beta)
            loss_agg_mode = g.loss_agg_mode
            kl_type = g.kl_type
            norm_adv_by_std = g.norm_adv_by_std
            filter_groups = g.filter_groups
            overlong_buffer_len = g.overlong_buffer_len
            overlong_penalty_factor = float(g.overlong_penalty_factor)
        else:
            raise RuntimeError("GRPOMode.from_coordinator requires coordinator.opd_config")

        reward_fn = get_reward_fn(reward_fn_name)
        streaming = isinstance(coordinator, StreamCoordinator)

        return cls(
            rollout_proxy=getattr(coordinator, 'rollout_proxy', None),
            teacher_client=getattr(coordinator, 'teacher_client', None),
            trainer_proxy=getattr(coordinator, 'trainer_proxy', None),
            trainer_cmd_queue=coordinator.trainer_cmd_queue,
            trainer_result_queue=coordinator.trainer_result_queue,
            tracer=getattr(coordinator, 'tracer', None),
            opd_config=oc,
            logger=getattr(coordinator, 'logger', None),
            grpo_group_size=grpo_group_size,
            grpo_clip_eps=grpo_clip_eps,
            grpo_kl_beta=grpo_kl_beta,
            reward_fn_name=reward_fn_name,
            reward_fn=reward_fn,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
            loss_agg_mode=loss_agg_mode,
            kl_type=kl_type,
            norm_adv_by_std=norm_adv_by_std,
            filter_groups=filter_groups,
            overlong_buffer_len=overlong_buffer_len,
            overlong_penalty_factor=overlong_penalty_factor,
            answer_pattern=answer_pattern,
            tokenizer=coordinator._init_tokenizer(),
            teacher_trace_info=getattr(coordinator, '_teacher_trace_info', {}),
            streaming=streaming,
        )

    def __init__(self, *, rollout_proxy, teacher_client, trainer_proxy,
                 trainer_cmd_queue=None, trainer_result_queue=None,
                 tracer, config=None, opd_config=None, logger=None,
                 grpo_group_size, grpo_clip_eps, grpo_kl_beta,
                 reward_fn_name, reward_fn,
                 clip_ratio_low=None, clip_ratio_high=None,
                 clip_ratio_c=3.0, loss_agg_mode="token-mean",
                 kl_type="k1", norm_adv_by_std=True,
                 filter_groups=False,
                 overlong_buffer_len=0, overlong_penalty_factor=1.0,
                 answer_pattern=None,
                 tokenizer=None,
                 teacher_trace_info=None,
                 streaming=False):
        """
        Args:
            rollout_proxy: QueueRolloutProxy for generation commands.
            teacher_client: TeacherClient for ZMQ scoring (None when kl_beta=0).
            trainer_proxy: QueueTrainerProxy (used for wait_checkpoint_save).
            trainer_cmd_queue: mp.Queue for sending train commands.
            trainer_result_queue: mp.Queue for receiving train results.
            tracer: Tracer for Perfetto spans.
            config: Full config dict with "teacher", "training", "data" keys.
            logger: Optional JSONL/ClearML logger.
            grpo_group_size: G — number of samples per prompt.
            grpo_clip_eps: PPO clip epsilon.
            grpo_kl_beta: KL penalty coefficient (0 = no reference model).
            reward_fn_name: Name of reward function.
            reward_fn: Callable reward function.
            clip_ratio_low: DAPO asymmetric clip low (None = use clip_eps).
            clip_ratio_high: DAPO asymmetric clip high (None = use clip_eps).
            clip_ratio_c: DAPO dual-clip threshold.
            loss_agg_mode: "token-mean" or "seq-mean-token-sum".
            kl_type: KL penalty type ("k1", "k3", etc.).
            norm_adv_by_std: Whether to normalize advantages by std.
            filter_groups: Whether to filter zero-variance groups.
            overlong_buffer_len: DAPO overlong buffer length.
            overlong_penalty_factor: DAPO overlong penalty factor.
            answer_pattern: Regex for strict answer extraction.
            teacher_trace_info: Extra trace args for teacher spans.
            streaming: When True, data_iterator() keeps ground_truths in batch
                and adds grpo_n_samples/return_logprobs for PromptFeeder.
                When False (default), pops ground_truths into _gt_queue (step-off path).
        """
        self.rollout_proxy = rollout_proxy
        self.teacher_client = teacher_client
        self.trainer_proxy = trainer_proxy
        self.trainer_cmd_queue = trainer_cmd_queue
        self.trainer_result_queue = trainer_result_queue
        self.tracer = tracer
        self.logger = logger

        self._opd_config = opd_config

        self.grpo_group_size = grpo_group_size
        self.config = GRPOConfig(
            clip_eps=grpo_clip_eps,
            kl_beta=grpo_kl_beta,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
            loss_agg_mode=loss_agg_mode,
            kl_type=kl_type,
        )
        self._reward_fn = reward_fn
        self.reward_fn_name = reward_fn_name

        self.norm_adv_by_std = norm_adv_by_std
        self.filter_groups = filter_groups
        self.overlong_buffer_len = overlong_buffer_len
        self.overlong_penalty_factor = overlong_penalty_factor
        self.answer_pattern = answer_pattern

        self._teacher_trace_info = teacher_trace_info or {}
        self.streaming = streaming

        # Ground truth FIFO: data_iterator pushes, wait_generate pops
        self._gt_queue = deque()

        # Cached tokenizer (lazy init)
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    #  Data                                                               #
    # ------------------------------------------------------------------ #

    def data_iterator(self):
        """Yield (epoch, batch_dict) pairs for training.

        When streaming=False (step-off path): ground truths are popped from
        each batch and stashed in the internal FIFO queue. They are re-attached
        to generation output in wait_generate().

        When streaming=True: ground truths stay in the batch dict for
        PromptFeeder to attach to individual prompts. Also adds
        grpo_n_samples and return_logprobs flags.
        """
        tokenizer = self._get_tokenizer()
        oc = self._opd_config

        if oc is not None:
            dataset = GRPOPromptDataset(
                path=oc.data.train_files,
                tokenizer=tokenizer,
                max_prompt_length=oc.data.max_prompt_length,
                prompt_key=oc.data.prompt_key,
                answer_key=oc.data.answer_key or "answer",
                prompt_template=oc.data.prompt_template,
                enable_thinking=oc.data.enable_thinking,
            )
            batch_size = oc.trainer.batch_size
            total_epochs = oc.trainer.total_epochs
        else:
            raise RuntimeError("GRPOMode.data_iterator requires opd_config")
        print(f"[GRPO] Training dataset: {len(dataset)} prompts, "
              f"G={self.grpo_group_size}", flush=True)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True,
                            collate_fn=grpo_collate_fn)
        for epoch in range(total_epochs):
            for batch in loader:
                if self.streaming:
                    # Streaming path: keep ground_truths for PromptFeeder,
                    # add GRPO flags for G-repetition and logprobs
                    batch["grpo_n_samples"] = self.grpo_group_size
                    batch["return_logprobs"] = True
                else:
                    # Step-off path: pop ground_truths into FIFO queue
                    gt = batch.pop("ground_truths", [])
                    self._gt_queue.append(gt)
                yield epoch, batch

    def on_resume_skip_complete(self):
        """Clear stale ground truths pushed during resume skip.

        data_iterator() pushes ground_truths into _gt_queue; skipped batches
        must not poison post-resume reward/advantage computation.
        """
        self._gt_queue.clear()

    # ------------------------------------------------------------------ #
    #  Generation                                                         #
    # ------------------------------------------------------------------ #

    def async_generate(self, batch_dict):
        """Submit prompts to rollout with grpo_n_samples flag."""
        batch_dict["grpo_n_samples"] = self.grpo_group_size
        batch_dict["return_logprobs"] = True
        self.rollout_proxy.submit_generate(batch_dict)

    def wait_generate(self):
        """Collect generation result. Pops ground truths from FIFO."""
        gen = self.rollout_proxy.collect_generate()
        gt = self._gt_queue.popleft() if self._gt_queue else []
        gen["_ground_truths"] = gt
        return gen

    # ------------------------------------------------------------------ #
    #  Scoring (rewards + optional reference model)                       #
    # ------------------------------------------------------------------ #

    def async_teacher(self, gen_output, batch=None):
        """Compute rewards + advantages, then optionally score with reference.

        Returns a future-like object with a .get() method.
        """
        G = self.grpo_group_size
        tokenizer = self._get_tokenizer()
        oc = self._opd_config
        if oc is None:
            raise RuntimeError("GRPOMode.async_teacher requires opd_config")
        max_response_length = oc.data.max_response_length

        # 1. Compute rewards on CPU
        responses_tensor = gen_output["responses"]
        B = responses_tensor.size(0)
        num_prompts = B // G

        response_lengths = gen_output["response_lengths"]
        responses_list = []
        for i in range(B):
            r_len = int(response_lengths[i].item())
            responses_list.append(responses_tensor[i, :r_len].tolist())

        ground_truths = gen_output.get("_ground_truths", [])
        if len(ground_truths) == num_prompts:
            gt_repeated = []
            for gt in ground_truths:
                gt_repeated.extend([gt] * G)
            ground_truths = gt_repeated

        rewards = self._reward_fn(responses_list, ground_truths, tokenizer, G,
                                   answer_pattern=self.answer_pattern)

        # DAPO: overlong reward shaping
        if self.overlong_buffer_len > 0:
            rewards = apply_overlong_penalty(
                rewards, response_lengths, max_response_length,
                self.overlong_buffer_len, self.overlong_penalty_factor)

        # DAPO: filter zero-variance groups
        if self.filter_groups:
            keep_mask, n_filtered = filter_zero_variance_groups(rewards, G)
            if n_filtered > 0:
                print(f"[GRPO] filter_groups: dropped {n_filtered}/{B // G} "
                      f"zero-variance groups", flush=True)

        advantages = compute_group_advantages(
            rewards, G, norm_by_std=self.norm_adv_by_std)

        result = {
            "advantages": advantages,
            "rewards": rewards,
            "ref_token_logps": None,
        }

        # 2. Optionally score with reference for KL penalty.
        if self.config.kl_beta > 0 and self.teacher_client is not None:
            teacher_fut = self._ref_model_score(gen_output)
            teacher_out = teacher_fut.get() if hasattr(teacher_fut, "get") else teacher_fut
            if teacher_out and "teacher_token_logps" in teacher_out:
                result["ref_token_logps"] = teacher_out["teacher_token_logps"]

        class _Resolved:
            def get(self):
                return result
        return _Resolved()

    def resolve_teacher(self, teacher_fut, timing, batch=None):
        """Resolve the teacher/reward future."""
        result = teacher_fut.get() if hasattr(teacher_fut, "get") else teacher_fut
        return result

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def async_train(self, gen_output, teacher_output):
        """Assemble GRPO batch and send to trainer via queue.

        Stores rewards in gen_output for logging (side effect).
        """
        # Store rewards for log_train_step
        gen_output["_rewards"] = teacher_output.get("rewards")

        input_ids = gen_output["input_ids"]
        attention_mask = gen_output["attention_mask"]
        prompt_lengths = gen_output["prompt_lengths"]

        max_prompt_len = int(prompt_lengths.max().item())
        response_mask = attention_mask.clone()
        response_mask[:, :max_prompt_len] = False

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "prompt_lengths": prompt_lengths,
            "student_old_logprobs": gen_output["student_logprobs"],
            "advantages": teacher_output["advantages"],
            "_send_mono": time.monotonic(),
        }
        if teacher_output.get("ref_token_logps") is not None:
            batch["ref_token_logps"] = teacher_output["ref_token_logps"]

        self.trainer_cmd_queue.put(("train", batch))

    def wait_train(self):
        """Wait for trainer result."""
        self._wait_checkpoint_save()
        return self.trainer_result_queue.get()

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #

    def log_train_step(self, step, timing, gen_out, result):
        """Log GRPO training step metrics."""
        if result is None:
            return
        m = result.get("metrics", {})
        train_secs = m.get("train_seconds", 0)
        loss = m.get("kl_loss", 0)
        lr = m.get("lr", 0)
        grad_norm = m.get("grad_norm", 0)
        clip_frac = m.get("clip_fraction", 0)
        mean_adv = m.get("mean_advantage", 0)
        mean_kl = m.get("mean_kl", 0)

        rewards = gen_out.get("_rewards")
        reward_str = ""
        if rewards is not None:
            reward_str = f" reward={rewards.mean():.2f}"

        print(
            f"[Step {step}] loss={loss:.4f} lr={lr:.2e} "
            f"clip={clip_frac:.1%} adv={mean_adv:.4f} kl={mean_kl:.4f}"
            f"{reward_str} "
            f"gen={timing.get('generate_seconds', 0):.1f}s "
            f"train={train_secs:.1f}s "
            f"grad_norm={grad_norm:.2f}",
            flush=True,
        )

        if self.logger:
            iter_seconds = (
                timing.get("generate_seconds", 0)
                + timing.get("teacher_seconds", 0)
                + train_secs
                + timing.get("sync_seconds", 0)
            )
            n_trainer_gpus = getattr(self, '_n_trainer_gpus', 1)
            n_tokens = m.get("n_tokens", 0) * n_trainer_gpus
            log_data = {
                "wall_time": time.time(),
                "kl_loss": loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "clip_fraction": clip_frac,
                "mean_advantage": mean_adv,
                "mean_kl": mean_kl,
                "n_tokens": n_tokens,
                "throughput_tok_per_s": (n_tokens / iter_seconds
                                         if iter_seconds > 0 else 0),
                "generate_seconds": timing.get("generate_seconds", 0),
                "teacher_seconds": timing.get("teacher_seconds", 0),
                "train_seconds": train_secs,
                "sync_seconds": timing.get("sync_seconds", 0),
                "iter_seconds": iter_seconds,
            }
            if rewards is not None:
                log_data["reward_mean"] = rewards.mean().item()
                log_data["reward_std"] = rewards.std().item()
                if rewards.numel() > 1:
                    log_data["reward_p10"] = rewards.quantile(0.1).item()
                    log_data["reward_p50"] = rewards.quantile(0.5).item()
                    log_data["reward_p90"] = rewards.quantile(0.9).item()
            if "response_lengths" in gen_out:
                log_data["avg_response_length"] = (
                    gen_out["response_lengths"].float().mean().item())
            for wk in ("sample_q_depth", "scored_q_depth",
                        "evicted_sample_q", "evicted_scored_q",
                        "staleness_min", "staleness_max",
                        "staleness_mean", "staleness_std"):
                if wk in timing:
                    log_data[wk] = timing[wk]
            # Pass through all remaining metrics (ratio stats, etc.)
            for k, v in m.items():
                if k not in log_data and k not in ("timing", "train_seconds"):
                    log_data[k] = v
            self.logger.log_step(step, log_data)

    # ------------------------------------------------------------------ #
    #  Lifecycle queries                                                  #
    # ------------------------------------------------------------------ #

    def needs_teacher(self):
        """Only need teacher (reference model) when KL penalty is enabled."""
        return self.config.kl_beta > 0

    def needs_rollout(self):
        """Whether this mode requires rollout worker(s)."""
        return True

    def get_trainer_fn(self):
        """Return trainer_entry_point for process spawning.

        All config is in the trainer config dict — no extra kwargs needed.
        """
        from opd.trainer.grpo import grpo_trainer_main
        return grpo_trainer_main

    # ------------------------------------------------------------------ #
    #  Streaming support                                                  #
    # ------------------------------------------------------------------ #

    def make_stream_score_fn(self, teacher_client):
        """Return score_fn that computes per-sample rewards on CPU."""
        import torch
        reward_fn = self._reward_fn
        tokenizer = self._get_tokenizer()
        answer_pattern = self.answer_pattern

        def score_fn(batch_samples):
            for sample in batch_samples:
                response_lengths = sample["response_lengths"]
                responses_tensor = sample["responses"]
                r_len = int(response_lengths[0].item()) if response_lengths.dim() > 0 else int(response_lengths.item())
                response_ids = responses_tensor[0, :r_len].tolist() if responses_tensor.dim() > 1 else responses_tensor[:r_len].tolist()
                gt = sample.get("ground_truth")
                if gt is None:
                    # Reject samples without ground_truth (e.g., seed prompts)
                    sample["reward"] = 0.0
                    continue
                gt_list = [gt] if gt else [""]
                reward = reward_fn([response_ids], gt_list, tokenizer, 1,
                                   answer_pattern=answer_pattern)
                sample["reward"] = float(reward[0].item()) if isinstance(reward, torch.Tensor) else float(reward[0])
            return None, None
        return score_fn

    def make_stream_assemble_fn(self, max_response_length):
        """Return assemble_batch_fn that groups by prompt_group_id and computes advantages."""
        import torch
        from opd.reward import (
            compute_group_advantages, apply_overlong_penalty,
            filter_zero_variance_groups,
        )
        group_size = self.grpo_group_size
        norm_adv_by_std = self.norm_adv_by_std
        filter_groups = self.filter_groups
        overlong_buffer_len = self.overlong_buffer_len
        overlong_penalty_factor = self.overlong_penalty_factor

        def assemble_fn(samples):
            # TrainDispatcher guarantees complete groups — samples are already
            # grouped by prompt_group_id with exactly group_size per group.
            # Reject any samples missing prompt_group_id (e.g., seed prompts).
            complete_samples = [s for s in samples if s.get("prompt_group_id") is not None]
            n_rejected = len(samples) - len(complete_samples)
            if n_rejected > 0:
                print(f"[GRPO assemble] Rejected {n_rejected} samples missing prompt_group_id",
                      flush=True)

            if not complete_samples:
                return None, None

            B = len(complete_samples)
            input_ids = torch.cat([s["input_ids"] for s in complete_samples], dim=0)
            attention_mask = torch.cat([s["attention_mask"] for s in complete_samples], dim=0)
            prompt_lengths = torch.cat([s["prompt_lengths"] for s in complete_samples], dim=0)
            response_lengths = torch.cat([s["response_lengths"] for s in complete_samples], dim=0)
            responses = torch.cat([s["responses"] for s in complete_samples], dim=0)

            student_logprobs = None
            if "student_logprobs" in complete_samples[0]:
                student_logprobs = torch.cat(
                    [s["student_logprobs"] for s in complete_samples], dim=0)

            rewards = torch.tensor([s["reward"] for s in complete_samples], dtype=torch.float32)

            if overlong_buffer_len > 0:
                rewards = apply_overlong_penalty(
                    rewards, response_lengths, max_response_length,
                    overlong_buffer_len, overlong_penalty_factor)

            if filter_groups:
                keep_mask, n_filtered = filter_zero_variance_groups(rewards, group_size)
                if n_filtered > 0:
                    print(f"[GRPO assemble] filter_groups: dropped {n_filtered}/{B // group_size} "
                          f"zero-variance groups", flush=True)

            advantages = compute_group_advantages(
                rewards, group_size, norm_by_std=norm_adv_by_std)

            max_prompt_len = int(prompt_lengths.max().item())
            response_mask = attention_mask.clone()
            response_mask[:, :max_prompt_len] = False

            gen_out = {
                "input_ids": input_ids, "attention_mask": attention_mask,
                "response_mask": response_mask, "prompt_lengths": prompt_lengths,
                "response_lengths": response_lengths, "responses": responses,
                "student_logprobs": student_logprobs,
                "weight_version": [s.get("weight_version", 0) for s in complete_samples],
                "worker_id": [s.get("worker_id", 0) for s in complete_samples],
                "_rewards": rewards,
            }
            teacher_out = {"advantages": advantages, "rewards": rewards}
            return gen_out, teacher_out
        return assemble_fn

    @property
    def stream_batch_multiplier(self):
        """Batch multiplier for streaming. GRPO: G (one response per sample)."""
        return self.grpo_group_size

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _get_tokenizer(self):
        """Return the tokenizer passed from the coordinator."""
        if self._tokenizer is None:
            raise RuntimeError(
                "GRPOMode requires a tokenizer — pass tokenizer= to __init__ "
                "or use GRPOMode.from_coordinator()")
        return self._tokenizer

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

    def _ref_model_score(self, gen_output):
        """Score with reference model via ZMQ (same logic as OPDMode.async_teacher).

        Returns a future-like SimpleNamespace(get=resolve).
        """
        full_lists = gen_output["full_token_lists"]
        n = self.teacher_client.n_workers
        chunk = max((len(full_lists) + n - 1) // n, 1)
        futures = []
        for i in range(0, len(full_lists), chunk):
            futures.append(self.teacher_client.submit(full_lists[i : i + chunk]))

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

        def resolve():
            with timer() as t_join:
                thread.join()
            if "error" in holder:
                raise RuntimeError(
                    f"Teacher scoring failed: {holder['error']}"
                ) from holder["error"]
            t_ts = holder.get("teacher_mono_start") or t_submit_mono
            t_te = holder.get("teacher_mono_end") or holder["t_done"]
            total_tok = sum(len(tl) for tl in full_lists)
            ba = {"n_prompts": len(full_lists), "total_tok": total_tok}
            ba.update(teacher_trace_info)
            tracer.emit("teacher_score", cat="teacher",
                        tid=TID_TEACHER, t_start=t_ts,
                        t_end=t_te, args=ba)
            tracer.emit("wait_teacher", cat="pipeline",
                        tid=TID_PIPELINE,
                        t_start=t_join["mono_start"],
                        t_end=t_join["mono_end"], args=ba)
            all_logps, all_idx, all_token_logps = holder["data"]
            with timer() as t_pad:
                out = pad_teacher(gen_output, all_logps, all_idx,
                                  all_token_logps)
            tracer.emit("pad_teacher", cat="pipeline",
                        tid=TID_PIPELINE,
                        t_start=t_pad["mono_start"],
                        t_end=t_pad["mono_end"], args=ba)
            return out

        return SimpleNamespace(get=resolve)

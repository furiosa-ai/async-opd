"""Actor-critic OPD trainer for critic-augmented policy_gradient_kl."""

import torch.nn.functional as F

from opd.launch_specs import TrainerLaunchSpec
from opd.loss.advantages import (
    compute_gae_advantages,
    compute_returns_from_advantages,
    compute_td0_advantages,
    masked_normalize,
)
from opd.loss.kl import chunked_log_softmax_gather
from opd.loss.ppo import ppo_clip_loss
from opd.trainer.base_trainer import BaseTrainer
from opd.trainer.config import build_actor_critic_config_from_algorithm_payload


class ActorCriticOPDTrainer(BaseTrainer):
    """Actor-critic OPD trainer using trainer-side value predictions."""

    def __init__(self, config: dict, rank_info: dict | None = None):
        algo = config.static.algorithm if isinstance(config, TrainerLaunchSpec) else config["algorithm"]
        self.config = build_actor_critic_config_from_algorithm_payload(algo)
        self.pg_clip_eps = self.config.clip_eps
        self.pg_online_advantage = self.config.online_advantage
        self.value_mode = self.config.value_mode
        self.gae_lambda = self.config.gae_lambda
        self.value_coef = self.config.value_coef
        self.normalize_advantages = self.config.normalize_advantages
        self._use_decoupled_ppo = self.config.use_decoupled_loss
        self.behave_imp_weight_cap = self.config.behave_imp_weight_cap
        self.m2po_budget = self.config.m2po_budget
        self.m2po_miniclip_low = self.config.m2po_miniclip_low
        self.m2po_miniclip_high = self.config.m2po_miniclip_high

        super().__init__(config, rank_info)
        self._backend._algorithm_launch = algo
        self._use_chunked = getattr(self._backend, '_chunked_kl_patched', False)

    def _compute_advantages(self, rewards, values_t, next_values_t, shifted_mask):
        if self.value_mode == "td0":
            advantages = compute_td0_advantages(
                rewards, values_t, next_values_t, shifted_mask, gamma=1.0)
        elif self.value_mode == "gae":
            advantages = compute_gae_advantages(
                rewards, values_t, shifted_mask, gamma=1.0, lam=self.gae_lambda)
        else:
            raise ValueError(f"Unsupported pg_value_mode: {self.value_mode}")

        if self.normalize_advantages:
            advantages = masked_normalize(advantages, shifted_mask)
        returns = compute_returns_from_advantages(values_t, advantages, shifted_mask)
        return advantages, returns

    @staticmethod
    def _align_teacher_token_logps(mb, shifted_len, device):
        """Align teacher token logprobs to the truncated shifted sequence length."""
        teacher_token_logps = mb["teacher_token_logps"]
        actual_max_len = mb.get("actual_max_len")
        seq_len = mb.get("seq_len", teacher_token_logps.size(1))
        if actual_max_len is not None and actual_max_len < seq_len:
            teacher_token_logps = teacher_token_logps[:, :actual_max_len]
        teacher_token_logps = teacher_token_logps[:, :-1].to(device)
        if teacher_token_logps.size(1) > shifted_len:
            teacher_token_logps = teacher_token_logps[:, :shifted_len]
        elif teacher_token_logps.size(1) < shifted_len:
            teacher_token_logps = F.pad(
                teacher_token_logps,
                (0, shifted_len - teacher_token_logps.size(1)),
            )
        return teacher_token_logps

    def _actor_critic_loss(self, *, student_new_logps, values, old_logps,
                           teacher_token_logps, shifted_mask, prox_logprobs=None):
        values_t = values[:, :-1]
        next_values_t = values[:, 1:]
        rewards = (teacher_token_logps - old_logps).detach()
        advantages, returns = self._compute_advantages(
            rewards, values_t, next_values_t, shifted_mask)
        td_errors = (returns - values_t).detach()

        actor_loss, raw_stats = ppo_clip_loss(
            student_new_logps, old_logps, advantages, shifted_mask,
            clip_eps=self.pg_clip_eps,
            use_decoupled_loss=self._is_decoupled_ppo_enabled(),
            prox_logprobs=prox_logprobs,
            behave_imp_weight_cap=self.behave_imp_weight_cap,
            m2po_budget=self.m2po_budget,
            m2po_miniclip_low=self.m2po_miniclip_low,
            m2po_miniclip_high=self.m2po_miniclip_high,
        )

        masked_value_err = (values_t - returns.detach()).square()[shifted_mask]
        if masked_value_err.numel() == 0:
            value_loss = values_t.new_zeros((), requires_grad=True)
        else:
            value_loss = masked_value_err.mean()

        total_loss = actor_loss + self.value_coef * value_loss
        extras = {
            "actor_loss": actor_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
            "return_mean": returns[shifted_mask].mean().item() if shifted_mask.any() else 0.0,
            "return_std": returns[shifted_mask].std().item() if shifted_mask.sum() > 1 else 0.0,
            "value_mean": values_t[shifted_mask].mean().item() if shifted_mask.any() else 0.0,
            "value_std": values_t[shifted_mask].std().item() if shifted_mask.sum() > 1 else 0.0,
            "td_error_mean": td_errors[shifted_mask].mean().item() if shifted_mask.any() else 0.0,
        }

        raw_tensors = {
            "ratios": raw_stats["_ratios"].detach().cpu(),
            "log_ratios": raw_stats["_log_ratios"].detach().cpu(),
            "advantages": raw_stats["_advantages"].detach().cpu(),
            "clip_high": raw_stats["_clip_high"].detach().cpu(),
            "clip_low": raw_stats["_clip_low"].detach().cpu(),
            "returns": returns[shifted_mask].detach().cpu(),
            "values": values_t[shifted_mask].detach().cpu(),
        }
        if "m2po_clip_low" in raw_stats:
            extras["m2po_clip_low"] = raw_stats["m2po_clip_low"]
            extras["m2po_clip_high"] = raw_stats["m2po_clip_high"]
            extras["m2po_m2_before"] = raw_stats["m2po_m2_before"]
            extras["m2po_m2_after"] = raw_stats["m2po_m2_after"]
        if "_behave_imp_weight" in raw_stats:
            raw_tensors["behave_imp_weight"] = raw_stats["_behave_imp_weight"].detach().cpu()
            raw_tensors["behave_mask"] = raw_stats["_behave_mask"].detach().cpu()
            if raw_stats["_behave_imp_weight"].numel() > 0:
                extras["behave_imp_weight"] = raw_stats["_behave_imp_weight"].mean().item()
                extras["behave_mask_ratio"] = raw_stats["_behave_mask"].float().mean().item()

        extras["r_mean"] = raw_stats["_ratios"].mean().item() if raw_stats["_ratios"].numel() > 0 else 0.0
        extras["clip_frac_high"] = raw_stats["_clip_high"].float().mean().item() if raw_stats["_clip_high"].numel() > 0 else 0.0
        extras["clip_frac_low"] = raw_stats["_clip_low"].float().mean().item() if raw_stats["_clip_low"].numel() > 0 else 0.0
        extras["_raw_tensors"] = raw_tensors
        return total_loss, extras

    def loss_fn(self, logits, mb):
        hidden_states = logits.hidden_states[-1] if hasattr(logits, "hidden_states") else None
        if hidden_states is None:
            raise NotImplementedError("ActorCriticOPDTrainer fallback path requires hidden_states")
        target_ids = mb["input_ids"][:, 1:]
        student_new_logps = chunked_log_softmax_gather(
            logits.logits[:, :-1], target_ids.unsqueeze(-1)).squeeze(-1)
        values = self._backend.model.value_head(hidden_states).squeeze(-1)
        shifted_mask = mb["response_mask"][:, 1:]
        teacher_token_logps = self._align_teacher_token_logps(
            mb, student_new_logps.shape[1], values.device)
        old_logps = self._align_old_logprobs(
            mb["student_logprobs"], student_new_logps.shape[1], shifted_mask, values.device)
        loss, extras = self._actor_critic_loss(
            student_new_logps=student_new_logps,
            values=values,
            old_logps=old_logps,
            teacher_token_logps=teacher_token_logps,
            shifted_mask=shifted_mask,
        )
        return loss, int(shifted_mask.sum().item()), extras

    def forward_and_loss_fn(self, model, mb, device):
        if getattr(self._backend, 'use_sequence_packing', False):
            raise NotImplementedError(
                "ActorCriticOPDTrainer does not support sequence packing yet.")

        kl_chunk_size = getattr(self._backend, 'kl_chunk_size', 1024)
        outputs = model(
            input_ids=mb["input_ids"],
            attention_mask=mb["attention_mask"],
            position_ids=mb.get("position_ids"),
            _kl_args={
                'mode': 'policy_gradient_kl',
                'chunk_size': kl_chunk_size,
                'return_values': True,
            },
        )
        student_new_logps = outputs["student_token_logps"].squeeze(-1)
        values = outputs["values"]
        shifted_mask = mb["response_mask"][:, 1:]
        shifted_len = student_new_logps.shape[1]

        teacher_token_logps = self._align_teacher_token_logps(
            mb, shifted_len, device)
        old_logps = self._align_old_logprobs(
            mb["student_logprobs"], shifted_len, shifted_mask, device)

        prox = self._consume_prox_logprobs(student_new_logps, device=device)

        loss, extras = self._actor_critic_loss(
            student_new_logps=student_new_logps,
            values=values,
            old_logps=old_logps,
            teacher_token_logps=teacher_token_logps,
            shifted_mask=shifted_mask,
            prox_logprobs=prox,
        )
        return loss, int(shifted_mask.sum().item()), extras

    def train_step(self, batch, backend):
        prepared = backend._prepare_train_batch(batch)
        flat = {
            "input_ids": prepared["input_ids"],
            "attention_mask": prepared["attention_mask"],
            "response_mask": prepared["response_mask"],
            "prompt_lengths": prepared["prompt_lengths"],
            "teacher_token_logps": prepared["batch"]["teacher_token_logps"],
            "student_logprobs": prepared["batch"]["student_logprobs"],
            "max_prompt": prepared["max_prompt"],
            "actual_max_len": prepared["actual_max_len"],
            "seq_len": prepared.get("orig_seq_len", prepared["input_ids"].size(1)),
            "orig_seq_len": prepared.get("orig_seq_len", prepared["input_ids"].size(1)),
        }
        self._copy_global_mini_plan_metadata(prepared, flat)

        self._maybe_precompute_prox(flat, backend)

        return backend._run_train_step(
            flat, self.loss_fn, forward_and_loss_fn=self.forward_and_loss_fn)


def ac_opd_trainer_main(config, cmd_queue, result_queue, rank_info):
    trainer = ActorCriticOPDTrainer(config, rank_info)
    trainer.run(cmd_queue, result_queue)

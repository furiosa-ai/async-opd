"""GRPO Trainer — Group Relative Policy Optimization.

Composition-based trainer: inherits shared infrastructure from BaseTrainer.
Uses chunked LM head to avoid materializing [B,S,V] logits — gathers only
K=1 next-token logprobs via the same _kl_args monkey-patched forward as
OPDTrainer's policy_gradient_kl mode.
"""

import torch

from opd.launch_specs import TrainerLaunchSpec
from opd.trainer.config import build_grpo_config_from_algorithm_payload
from opd.trainer.base_trainer import BaseTrainer


class GRPOTrainer(BaseTrainer):
    """Group Relative Policy Optimization trainer.

    Inherits from BaseTrainer for backend creation, run(), and prox
    precomputation. Provides GRPO-specific loss via ppo_clip_loss.

    Batch expected keys:
    - input_ids, attention_mask, response_mask, prompt_lengths
    - student_old_logprobs: [B, resp_len] log pi_old from rollout
    - advantages: [B] group-relative advantages
    - ref_token_logps: [B, S] log pi_ref (optional, when kl_beta > 0)
    """

    def __init__(self, config: dict, rank_info: dict | None = None):
        algo = config.static.algorithm if isinstance(config, TrainerLaunchSpec) else config["algorithm"]
        self.config = build_grpo_config_from_algorithm_payload(algo)

        # Don't set config["loss_mode"] = "sft" — let backend apply chunked LM head patch
        super().__init__(config, rank_info)
        # GRPO steps scheduler once per generation batch
        self._backend._scheduler_needs_rebuild = False

    def loss_fn(self, logits, mb):
        """GRPO PPO-clip loss from full [B, S, V] logits (fallback path).

        Used when chunked LM head is not available (e.g., unsupported model
        architecture). Delegates to grpo_clip_loss which gathers logprobs
        internally.

        Args:
            logits: [B, S, V] student model logits
            mb: dict with input_ids, response_mask, student_old_logprobs, advantages, etc.

        Returns:
            (loss, n_tokens, extras_dict)
        """
        from opd.loss.grpo import grpo_clip_loss

        cfg = self.config

        prox_logprobs = None
        if cfg.use_decoupled_loss:
            if "_prox_logprobs" in mb:
                prox_logprobs = mb["_prox_logprobs"].to(logits.device)
            else:
                from opd.loss.kl import chunked_log_softmax_gather
                target_ids = mb["input_ids"][:, 1:]
                with torch.no_grad():
                    prox_logprobs = chunked_log_softmax_gather(
                        logits[:, :-1], target_ids.unsqueeze(-1)
                    ).squeeze(-1).detach()

        loss, stats = grpo_clip_loss(
            logits,
            mb["input_ids"],
            mb["student_old_logprobs"],
            mb["advantages"],
            mb["response_mask"],
            clip_eps=cfg.clip_eps,
            clip_ratio_low=cfg.clip_ratio_low,
            clip_ratio_high=cfg.clip_ratio_high,
            clip_ratio_c=cfg.clip_ratio_c,
            ref_token_logps=mb.get("ref_token_logps"),
            kl_beta=cfg.kl_beta,
            kl_type=cfg.kl_type,
            loss_agg_mode=cfg.loss_agg_mode,
            use_decoupled_loss=cfg.use_decoupled_loss,
            prox_logprobs=prox_logprobs,
            behave_imp_weight_cap=cfg.behave_imp_weight_cap,
        )

        n_tok = int(mb["response_mask"][:, 1:].sum().item())
        return loss, n_tok, stats

    def forward_and_loss_fn(self, model, mb, device):
        """Chunked LM head path: gathers K=1 next-token logprobs without
        materializing [B,S,V] logits, then computes PPO-clip loss.

        Args:
            model: the (possibly FSDP-wrapped) model with _kl_args monkey-patch
            mb: dict of micro-batch tensors + metadata scalars
            device: torch device

        Returns:
            (loss, n_tokens, extras_dict)
        """
        from opd.loss.ppo import ppo_clip_loss

        cfg = self.config
        kl_chunk_size = getattr(self._backend, 'kl_chunk_size', 1024)

        mb_input_ids = mb["input_ids"]
        mb_attention_mask = mb["attention_mask"]
        mb_position_ids = mb.get("position_ids")
        mb_response_mask = mb["response_mask"]

        # ---- Sequence packing (optional) ----
        mb_packing_kwargs = {}
        use_packing = getattr(self._backend, 'use_sequence_packing', False)
        if use_packing:
            mb_prompt_lens = mb.get("prompt_lengths")
            if mb_prompt_lens is not None and isinstance(mb_prompt_lens, torch.Tensor):
                # Guard: packing not yet supported with seq-mean-token-sum
                # (packed [1,T] layout breaks per-sequence aggregation)
                if cfg.loss_agg_mode == "seq-mean-token-sum":
                    raise NotImplementedError(
                        "Sequence packing with loss_agg_mode='seq-mean-token-sum' "
                        "not yet supported. Packed layout collapses to 1 sequence.")
                # Guard: packing not yet supported with reference KL penalty
                # (ref_token_logps not packed/aligned with student logps)
                if cfg.kl_beta > 0 and "ref_token_logps" in mb:
                    raise NotImplementedError(
                        "Sequence packing with kl_beta > 0 (reference KL penalty) "
                        "not yet supported. ref_token_logps alignment not implemented.")

                from opd.data.packing import pack_micro_batch
                pack_kwargs = dict(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attention_mask,
                    response_mask=mb_response_mask,
                    prompt_lengths=mb_prompt_lens,
                )
                # Reuse student_logprobs slot for student_old_logprobs
                if "student_old_logprobs" in mb and isinstance(mb["student_old_logprobs"], torch.Tensor):
                    pack_kwargs["student_logprobs"] = mb["student_old_logprobs"]

                packed = pack_micro_batch(**pack_kwargs)
                mb_input_ids = packed.input_ids
                mb_position_ids = packed.position_ids
                mb_attention_mask = None  # Must be None for FA varlen
                mb_response_mask = packed.response_mask
                mb_packing_kwargs = {
                    "cu_seq_lens_q": packed.cu_seq_lens,
                    "cu_seq_lens_k": packed.cu_seq_lens,
                    "max_length_q": packed.max_seq_len,
                    "max_length_k": packed.max_seq_len,
                }
                if packed.student_logprobs is not None:
                    mb["_packed_student_old_logprobs"] = packed.student_logprobs

        # Model forward — gather K=1 at next tokens via chunked LM head
        student_new_logps = model(
            input_ids=mb_input_ids,
            attention_mask=mb_attention_mask,
            position_ids=mb_position_ids,
            _kl_args={'mode': 'policy_gradient_kl', 'chunk_size': kl_chunk_size},
            **mb_packing_kwargs,
        ).squeeze(-1)  # [B, S-1] or [1, T-1] if packed

        shifted_mask = mb_response_mask[:, 1:]
        shifted_len = student_new_logps.shape[1]

        # Align old logprobs to shifted positions
        if "_packed_student_old_logprobs" in mb:
            # Packed: student_logprobs[i] = log P_old(token_i).
            # student_new_logps[i] = log P(token_{i+1}).
            # old_logps[i] should be log P_old(token_{i+1}) = packed[i+1].
            old_logps = mb["_packed_student_old_logprobs"][:, 1:].to(device)
            if old_logps.shape[1] < shifted_len:
                old_logps = torch.nn.functional.pad(
                    old_logps, (0, shifted_len - old_logps.shape[1]))
            elif old_logps.shape[1] > shifted_len:
                old_logps = old_logps[:, :shifted_len]
        else:
            old_logps = self._align_old_logprobs(
                mb["student_old_logprobs"], shifted_len, shifted_mask, device)

        # Broadcast sequence-level advantages [B] → [B, S-1]
        adv = mb["advantages"].to(device)
        if adv.dim() == 1:
            if use_packing and mb_packing_kwargs:
                # Packed: expand per-sequence advantages to per-token
                # packed layout has all seqs concatenated, need per-token broadcast
                adv_expanded = torch.zeros(1, shifted_len, device=device, dtype=adv.dtype)
                cu = mb_packing_kwargs["cu_seq_lens_q"]
                for i in range(adv.size(0)):
                    s_start = cu[i].item()
                    s_end = cu[i + 1].item()
                    # Shifted: positions [s_start, s_end-1) in the packed [1, T-1]
                    t_start = max(0, s_start - 0)  # no shift needed for cu_seq_lens
                    t_end = s_end - 1  # -1 for the shift
                    if t_end > t_start and t_end <= shifted_len:
                        adv_expanded[0, t_start:t_end] = adv[i].detach()
                adv = adv_expanded
            else:
                adv = adv.detach().unsqueeze(1).expand_as(student_new_logps)

        # KL penalty vs reference (optional)
        per_token_kl = None
        if cfg.kl_beta > 0 and "ref_token_logps" in mb:
            ref_logps = mb["ref_token_logps"][:, :-1].detach().to(device)
            if ref_logps.shape[1] > shifted_len:
                ref_logps = ref_logps[:, :shifted_len]
            kl_type = cfg.kl_type
            if kl_type in ("low_var_kl", "k3", "low_var_kl+", "k3+"):
                kl_diff = (ref_logps - student_new_logps).clamp(-20.0, 20.0)
                kld = (kl_diff.exp() - kl_diff - 1).clamp(-10.0, 10.0)
                if kl_type.endswith("+"):
                    k2 = 0.5 * (student_new_logps - ref_logps).square()
                    kld = k2 - k2.detach() + kld.detach()
                per_token_kl = cfg.kl_beta * kld
            else:
                per_token_kl = cfg.kl_beta * (student_new_logps - ref_logps)

        prox = self._consume_prox_logprobs(student_new_logps, device=device)

        loss, raw_stats = ppo_clip_loss(
            student_new_logps, old_logps, adv, shifted_mask,
            clip_eps=cfg.clip_eps,
            clip_ratio_low=cfg.clip_ratio_low,
            clip_ratio_high=cfg.clip_ratio_high,
            clip_ratio_c=cfg.clip_ratio_c,
            per_token_penalty=per_token_kl,
            loss_agg_mode=cfg.loss_agg_mode,
            use_decoupled_loss=cfg.use_decoupled_loss,
            prox_logprobs=prox,
            behave_imp_weight_cap=cfg.behave_imp_weight_cap,
        )

        n_tok = int(shifted_mask.sum().item())

        from opd.loss.grpo import _build_grpo_stats
        stats = _build_grpo_stats(raw_stats, per_token_kl, shifted_mask, cfg.kl_beta)
        return loss, n_tok, stats

    def train_step(self, batch, backend):
        """Prepare batch and run training step via backend.

        Uses forward_and_loss_fn (chunked LM head) when available,
        falls back to loss_fn (full logits) otherwise.
        """
        prepared = backend._prepare_batch(batch)

        # Flatten: extract tensors, drop metadata
        flat = {k: v for k, v in prepared.items()
                if k not in ("n_mini", "mini_bs", "seq_len", "actual_max_len")}

        # Decoupled PPO: precompute pi_prox before any optimizer steps
        self._maybe_precompute_prox(flat, backend, allow_megatron=False)

        if self._use_chunked:
            return backend._run_train_step(flat, self.loss_fn,
                                           forward_and_loss_fn=self.forward_and_loss_fn)
        else:
            return backend._run_train_step(flat, self.loss_fn)


# ============================================================
# Entry point for subprocess spawning
# ============================================================


def grpo_trainer_main(config, cmd_queue, result_queue, rank_info):
    """Entry point for GRPO training subprocess."""
    trainer = GRPOTrainer(config, rank_info)
    trainer.run(cmd_queue, result_queue)

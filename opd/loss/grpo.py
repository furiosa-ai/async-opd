"""GRPO (Group Relative Policy Optimization) loss.

PPO-clip with sequence-level group-relative advantages + optional per-token
KL penalty vs a frozen reference policy. Supports DAPO extensions:
asymmetric clipping, dual-clip, and token-mean loss aggregation.

Thin wrapper around ppo_clip_loss — handles GRPO-specific input preparation
(logprob gathering, old-logprob alignment, advantage broadcasting, KL penalty)
and stats format conversion.
"""

import torch

from opd.loss.kl import chunked_log_softmax_gather
from opd.loss.ppo import ppo_clip_loss
from opd.utils.config import GRPOAlgorithmConfig


_GRPO_DEFAULTS = GRPOAlgorithmConfig()


def grpo_clip_loss(
    student_logits,           # [B, S, V] current policy logits
    input_ids,                # [B, S] token IDs
    student_old_logprobs,     # [B, R] log pi_old from rollout (R = resp_len)
    advantages,               # [B] sequence-level group-relative advantages
    mask,                     # [B, S] bool — response token positions
    clip_eps=None,
    clip_ratio_low=None,      # DAPO: asymmetric lower clip (default = clip_eps)
    clip_ratio_high=None,     # DAPO: asymmetric upper clip (default = clip_eps)
    clip_ratio_c=None,        # dual-clip lower bound for negative advantages
    ref_token_logps=None,     # [B, S] log pi_ref from reference policy (optional)
    kl_beta=None,
    kl_type=None,             # "k1" | "low_var_kl"/"k3" | "k3+" (see docstring)
    loss_agg_mode=None,       # "token-mean" (DAPO) or "seq-mean-token-sum" (standard GRPO)
    use_decoupled_loss=None,
    prox_logprobs=None,       # [B, S-1] log pi_prox (detached, from cache)
    behave_imp_weight_cap=None,
):
    """GRPO PPO-clip loss with sequence-level advantages.

    Args:
        student_logits: [B, S, V] — current student model logits
        input_ids: [B, S] — token IDs of the full sequence
        student_old_logprobs: [B, R] — log pi_old(y_t) from rollout generation
        advantages: [B] — pre-computed group-relative advantages (detached)
        mask: [B, S] bool — True at response token positions
        clip_eps: PPO clipping epsilon (default 0.2, used when low/high not set)
        clip_ratio_low: lower clip range (default = clip_eps). DAPO uses 0.2.
        clip_ratio_high: upper clip range (default = clip_eps). DAPO uses 0.28.
        clip_ratio_c: dual-clip lower bound for negative advantages (default=3.0,
                      matching verl). DAPO uses 10.0. None=disabled.
        ref_token_logps: [B, S] — log pi_ref(y_t) from reference policy (optional)
        kl_beta: KL penalty coefficient (default 0.0 = no KL penalty)
        kl_type: "k1" = simple log-ratio (can be negative),
                 "low_var_kl" or "k3" = Schulman approximation (always >= 0, lower variance),
                 "k3+" = k3 forward + k2 (MSE) backward for unbiased KL gradients
        loss_agg_mode: "token-mean" = total loss / total tokens (DAPO default),
                       "seq-mean-token-sum" = per-seq sum, then mean over seqs

    Returns:
        loss: scalar tensor (requires_grad through student_logits only)
        stats: dict with logging metrics
    """
    if clip_eps is None:
        clip_eps = _GRPO_DEFAULTS.clip_eps
    if clip_ratio_c is None:
        clip_ratio_c = _GRPO_DEFAULTS.clip_ratio_c
    if kl_beta is None:
        kl_beta = _GRPO_DEFAULTS.kl_beta
    if kl_type is None:
        kl_type = _GRPO_DEFAULTS.kl_type
    if loss_agg_mode is None:
        loss_agg_mode = _GRPO_DEFAULTS.loss_agg_mode
    if use_decoupled_loss is None:
        use_decoupled_loss = _GRPO_DEFAULTS.use_decoupled_loss
    if behave_imp_weight_cap is None:
        behave_imp_weight_cap = _GRPO_DEFAULTS.behave_imp_weight_cap

    # ---- GRPO-specific input preparation ----

    # Gather log pi_theta at next tokens from full logits
    target_ids = input_ids[:, 1:]           # [B, S-1]
    logits = student_logits[:, :-1]         # [B, S-1, V]
    student_new_logps = chunked_log_softmax_gather(
        logits, target_ids.unsqueeze(-1)
    ).squeeze(-1)                           # [B, S-1]

    # Align old logprobs to shifted positions (right-aligned)
    bs, shifted_len = student_new_logps.shape
    resp_len = student_old_logprobs.size(1)
    if resp_len > shifted_len:
        actual_resp = int(mask[:, 1:].sum(dim=1).max().item())
        student_old_logprobs = student_old_logprobs[:, :actual_resp]
        resp_len = actual_resp
    old_logps = torch.zeros(bs, shifted_len, device=student_new_logps.device,
                            dtype=student_old_logprobs.dtype)
    old_logps[:, -resp_len:] = student_old_logprobs

    # Shifted mask
    m = mask[:, 1:]  # [B, S-1]

    # Broadcast sequence-level advantages [B] → [B, S-1]
    adv = advantages.detach().unsqueeze(1).expand_as(student_new_logps)

    # ---- KL penalty vs reference (GRPO-specific) ----
    per_token_kl = None
    if kl_beta > 0 and ref_token_logps is not None:
        ref_logps = ref_token_logps[:, :-1].detach()
        if kl_type in ("low_var_kl", "k3", "low_var_kl+", "k3+"):
            kl_diff = (ref_logps - student_new_logps).clamp(-20.0, 20.0)
            kld = (kl_diff.exp() - kl_diff - 1).clamp(-10.0, 10.0)
            if kl_type.endswith("+"):
                k2 = 0.5 * (student_new_logps - ref_logps).square()
                kld = k2 - k2.detach() + kld.detach()
            per_token_kl = kl_beta * kld
        else:
            per_token_kl = kl_beta * (student_new_logps - ref_logps)

    # ---- Call shared PPO-clip core ----
    loss, raw_stats = ppo_clip_loss(
        student_new_logps, old_logps, adv, m,
        clip_eps=clip_eps,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        clip_ratio_c=clip_ratio_c,
        per_token_penalty=per_token_kl,
        loss_agg_mode=loss_agg_mode,
        use_decoupled_loss=use_decoupled_loss,
        prox_logprobs=prox_logprobs,
        behave_imp_weight_cap=behave_imp_weight_cap,
    )

    stats = _build_grpo_stats(raw_stats, per_token_kl, m, kl_beta)
    return loss, stats


def _build_grpo_stats(raw_stats, per_token_kl, mask, kl_beta):
    """Convert ppo_clip_loss raw_stats into GRPO's aggregated logging format."""
    with torch.no_grad():
        masked_kl = torch.zeros(0)
        if per_token_kl is not None:
            masked_kl = per_token_kl[mask] / max(kl_beta, 1e-10)

        n = raw_stats["_ratios"].numel()
        stats = {
            "mean_ratio": raw_stats["_ratios"].mean().item() if n > 0 else 0.0,
            "mean_log_ratio": raw_stats["_log_ratios"].mean().item() if n > 0 else 0.0,
            "mean_advantage": raw_stats["_advantages"].mean().item() if n > 0 else 0.0,
            "clip_fraction": (
                (raw_stats["_clip_high"] | raw_stats["_clip_low"]).float().mean().item()
                if n > 0 else 0.0
            ),
            "mean_kl": masked_kl.mean().item() if masked_kl.numel() > 0 else 0.0,
            "_raw_tensors": {
                "ratios": raw_stats["_ratios"],
                "log_ratios": raw_stats["_log_ratios"],
                "advantages": raw_stats["_advantages"],
                "clip_high": raw_stats["_clip_high"],
                "clip_low": raw_stats["_clip_low"],
            },
        }
        behave_w = raw_stats.get("_behave_imp_weight")
        behave_m = raw_stats.get("_behave_mask")
        if behave_w is not None and behave_m is not None:
            stats["behave_imp_weight"] = behave_w.mean().item() if behave_w is not None and behave_w.numel() > 0 else 0.0
            stats["behave_mask_ratio"] = behave_m.float().mean().item() if behave_m is not None and behave_m.numel() > 0 else 1.0

    return stats

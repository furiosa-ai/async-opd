"""Shared PPO-clip surrogate loss for GRPO and policy_gradient_kl.

Both GRPO (reward-based advantages) and PG-KL (KL-based advantages) use
identical PPO-clip math: ratio computation, asymmetric clipping, dual-clip,
decoupled PPO with behavioral importance weighting, and M2PO dynamic clip.

This module extracts the shared core so both callers use one implementation.

Formula:
    ratio_t = pi_theta(y_t) / pi_base(y_t)   # pi_base = pi_prox or pi_old
    surr1 = -adv * ratio
    surr2 = -adv * clip(ratio, 1-eps_low, 1+eps_high)
    L_clip = max(surr1, surr2)                # per-token
    L = L_clip + per_token_penalty            # optional (GRPO KL penalty)
    L = L * w_behav                           # optional (decoupled PPO)
    loss = aggregate(L, mask)
"""

import torch

from opd.loss.kl import _m2po_clip_bounds
from opd.utils.config import OPDAlgorithmConfig


_OPD_DEFAULTS = OPDAlgorithmConfig()


def ppo_clip_loss(
    student_new_logps,           # [B, S-1] log pi_theta (pre-gathered)
    old_logps,                   # [B, S-1] log pi_old (pre-aligned to shifted positions)
    advantages,                  # [B, S-1] advantages (pre-broadcast by caller)
    mask,                        # [B, S-1] bool — response token positions
    clip_eps=None,
    use_importance_sampling=True,
    clip_ratio_low=None,         # DAPO: asymmetric lower clip (default = clip_eps)
    clip_ratio_high=None,        # DAPO: asymmetric upper clip (default = clip_eps)
    clip_ratio_c=None,           # DAPO: dual-clip lower bound for neg advantages (None=disabled)
    per_token_penalty=None,      # [B, S-1] added after clip loss, before behave weighting
    loss_agg_mode="token-mean",  # "token-mean" | "seq-mean-token-sum"
    use_decoupled_loss=None,
    prox_logprobs=None,          # [B, S-1] log pi_prox (detached)
    behave_imp_weight_cap=None,
    m2po_budget=None,            # M2PO: second-moment budget (None=disabled, use fixed clip_eps)
    m2po_miniclip_low=None,      # M2PO: floor for dynamic clip low margin
    m2po_miniclip_high=None,     # M2PO: floor for dynamic clip high margin
    sample_axis=None,            # Optional sample axis to mean-reduce before token aggregation
):
    """Shared PPO-clip surrogate loss.

    All inputs are pre-gathered, pre-aligned, and pre-broadcast by the caller.
    This function contains only the PPO math — no logprob gathering, no old-logprob
    alignment, no advantage computation.

    Args:
        student_new_logps: [B, S-1] log pi_theta at next tokens (shifted)
        old_logps: [B, S-1] log pi_old, right-aligned to shifted positions
        advantages: [B, S-1] per-token advantages (detached)
        mask: [B, S-1] bool, True at response positions
        clip_eps: symmetric PPO clip epsilon (used when low/high not set)
        clip_ratio_low: asymmetric lower clip (DAPO). Default = clip_eps.
        clip_ratio_high: asymmetric upper clip (DAPO). Default = clip_eps.
        clip_ratio_c: dual-clip lower bound for negative advantages (DAPO).
            None=disabled (PG-KL default). GRPO uses 3.0, DAPO uses 10.0.
        per_token_penalty: [B, S-1] optional per-token penalty added to clip
            loss before behavioral importance weighting (e.g., GRPO KL penalty).
        loss_agg_mode: "token-mean" (default) or "seq-mean-token-sum".
        use_importance_sampling: if False, force ratio=1 and skip any
            decoupled/behavioral importance weighting corrections.
        use_decoupled_loss: if True, ratio = pi_theta / pi_prox (not pi_old).
        prox_logprobs: [B, S-1] log pi_prox for decoupled PPO.
        behave_imp_weight_cap: cap for behavioral importance weight.
        m2po_budget: M2PO second-moment budget (None=disabled).
        m2po_miniclip_low: M2PO floor for dynamic clip low margin.
        m2po_miniclip_high: M2PO floor for dynamic clip high margin.
        sample_axis: optional axis (e.g. -1 for [B, T, N]) to mean-reduce
            after per-sample PPO math and before token aggregation.

    Returns:
        (loss, raw_stats): loss is a scalar tensor; raw_stats is a dict with:
            _ratios, _log_ratios, _advantages, _clip_high, _clip_low: [N] masked
            Optional: _behave_imp_weight, _behave_mask (decoupled PPO)
            Optional: m2po_clip_low, m2po_clip_high, m2po_m2_before, m2po_m2_after
    """
    if clip_eps is None:
        clip_eps = _OPD_DEFAULTS.pg_clip_eps
    if use_decoupled_loss is None:
        use_decoupled_loss = _OPD_DEFAULTS.use_decoupled_loss
    if behave_imp_weight_cap is None:
        behave_imp_weight_cap = _OPD_DEFAULTS.behave_imp_weight_cap
    if m2po_miniclip_low is None:
        m2po_miniclip_low = _OPD_DEFAULTS.pg_m2po_miniclip_low
    if m2po_miniclip_high is None:
        m2po_miniclip_high = _OPD_DEFAULTS.pg_m2po_miniclip_high

    if clip_ratio_low is None:
        clip_ratio_low = clip_eps
    if clip_ratio_high is None:
        clip_ratio_high = clip_eps

    # Importance sampling ratio
    if not use_importance_sampling:
        ratio_base = student_new_logps.detach().float()
        use_decoupled_loss = False
    elif use_decoupled_loss:
        if prox_logprobs is None:
            raise ValueError(
                "use_decoupled_loss=True requires prox_logprobs to be provided.")
        ratio_base = prox_logprobs.detach().float()
    else:
        ratio_base = old_logps.detach().float()

    # Clamp log-ratio for numerical stability (matches verl-opd)
    log_ratio = (student_new_logps.float() - ratio_base).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()

    adv = advantages.detach()

    # PPO clipped surrogate with asymmetric clipping
    if m2po_budget is not None:
        # M2PO: dynamic asymmetric clip bounds from second-moment budget
        cl, ch, m2_before, m2_after = _m2po_clip_bounds(
            old_logps, student_new_logps, adv, mask, m2po_budget)
        cl = max(cl, m2po_miniclip_low)
        ch = max(ch, m2po_miniclip_high)
        surr1 = -adv * ratio
        surr2 = -adv * ratio.clamp(1.0 - cl, 1.0 + ch)
        _m2po_stats = {"m2po_clip_low": cl, "m2po_clip_high": ch,
                       "m2po_m2_before": m2_before, "m2po_m2_after": m2_after}
    else:
        surr1 = -adv * ratio
        surr2 = -adv * ratio.clamp(1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
        _m2po_stats = None
        cl = clip_ratio_low
        ch = clip_ratio_high

    per_token_clip_loss = torch.max(surr1, surr2)

    # Dual-clip: extra lower bound for negative advantages (DAPO)
    if clip_ratio_c is not None:
        surr3 = -adv * clip_ratio_c
        per_token_clip_loss = torch.where(
            adv < 0,
            torch.min(surr3, per_token_clip_loss),
            per_token_clip_loss,
        )

    # Add optional per-token penalty (e.g., GRPO KL penalty vs reference)
    if per_token_penalty is not None:
        per_token_loss = per_token_clip_loss + per_token_penalty
    else:
        per_token_loss = per_token_clip_loss

    # Behavioral importance weight correction (decoupled PPO only)
    if use_decoupled_loss:
        behave_log_ratio = prox_logprobs.detach().float() - old_logps.detach().float()
        w_behav = behave_log_ratio.exp()
        behave_mask = w_behav <= behave_imp_weight_cap
        w_behav = torch.where(behave_mask, w_behav, w_behav.new_zeros(()))
        per_token_loss = per_token_loss * w_behav

    if sample_axis is not None:
        per_token_loss = per_token_loss.mean(dim=sample_axis)

    # Loss aggregation
    if loss_agg_mode == "seq-mean-token-sum":
        per_seq_loss = (per_token_loss * mask.float()).sum(dim=1)
        n_seqs = mask.any(dim=1).sum()
        loss = per_seq_loss.sum() / max(n_seqs.item(), 1)
    else:
        masked_loss = per_token_loss[mask]
        if masked_loss.numel() == 0:
            loss = per_token_loss.new_zeros((), requires_grad=True)
        else:
            loss = masked_loss.mean()

    # Collect raw stats for logging
    with torch.no_grad():
        stats_mask = mask
        if sample_axis is not None and stats_mask.dim() == ratio.dim() - 1:
            sample_dim = sample_axis if sample_axis >= 0 else ratio.dim() + sample_axis
            stats_mask = stats_mask.unsqueeze(sample_dim).expand_as(ratio)

        masked_ratio = ratio[stats_mask]
        masked_log_ratio = log_ratio[stats_mask]
        masked_adv = adv[stats_mask]

        raw_stats = {
            "_ratios": masked_ratio.cpu(),
            "_log_ratios": masked_log_ratio.cpu(),
            "_advantages": masked_adv.cpu(),
            "_clip_high": (ratio > 1.0 + ch)[stats_mask].cpu(),
            "_clip_low": (ratio < 1.0 - cl)[stats_mask].cpu(),
        }
        if _m2po_stats is not None:
            raw_stats.update(_m2po_stats)
        if use_decoupled_loss:
            raw_stats["_behave_imp_weight"] = w_behav[stats_mask].detach().cpu()
            raw_stats["_behave_mask"] = behave_mask[stats_mask].cpu()

    return loss, raw_stats

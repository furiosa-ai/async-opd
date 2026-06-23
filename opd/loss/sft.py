"""Cross-entropy loss for supervised fine-tuning with response masking."""

import torch.nn.functional as F

from opd.loss.kl import KLConfig, resolve_kl_config
from opd.trainer.config import SFTConfig
from opd.utils.config import SFTAlgorithmConfig


_SFT_DEFAULTS = SFTAlgorithmConfig()


def sft_loss(logits, input_ids, response_mask):
    """Causal LM cross-entropy loss on completion tokens only.

    Args:
        logits: [B, T, V] model output logits
        input_ids: [B, T] token IDs (shifted internally for labels)
        response_mask: [B, T] binary mask (1 = completion token)

    Returns:
        loss: scalar tensor (mean CE over completion tokens)
        n_tokens: int, number of completion tokens
    """
    # Standard causal LM shift: predict token t+1 from logits at t
    # logits[:, :-1] predicts labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
    shift_labels = input_ids[:, 1:].contiguous()     # [B, T-1]
    shift_mask = response_mask[:, 1:].contiguous()   # [B, T-1]

    # Per-token cross-entropy, flat for efficiency
    B, T, V = shift_logits.shape
    per_token_loss = F.cross_entropy(
        shift_logits.view(B * T, V),
        shift_labels.view(B * T),
        reduction="none",
    ).view(B, T)  # [B, T-1]

    # Apply mask: only completion tokens contribute
    masked_loss = per_token_loss * shift_mask.float()

    n_tokens = int(shift_mask.sum().item())
    denom = shift_mask.float().sum().clamp(min=1.0)
    loss = masked_loss.sum() / denom

    return loss, n_tokens


def compute_sft_loss(logits, input_ids, response_mask,
                     teacher_topk_logps=None, teacher_topk_indices=None,
                     teacher_valid_mask=None,
                     sft_loss_mode=None, ce_alpha=None, n_kl_logprobs=None,
                     kl_config: KLConfig | None = None,
                     sft_config: SFTConfig | None = None):
    """Dispatch SFT loss computation based on sft_loss_mode.

    Supports "ce" (cross-entropy only), "kl" (KL divergence only),
    and "mixed" (ce_alpha * CE + (1-ce_alpha) * KL).

    Returns:
        (loss, n_tokens): scalar loss tensor and token count
        extras: dict with per-component losses for logging
    """
    if sft_config is not None:
        sft_loss_mode = sft_config.loss_mode
        ce_alpha = sft_config.ce_alpha
        n_kl_logprobs = sft_config.n_kl_logprobs
    else:
        if sft_loss_mode is None:
            sft_loss_mode = _SFT_DEFAULTS.loss_mode
        if ce_alpha is None:
            ce_alpha = _SFT_DEFAULTS.ce_alpha
        if n_kl_logprobs is None:
            raise ValueError("n_kl_logprobs must be provided when sft_config is not supplied")

    if sft_loss_mode == "ce":
        loss, n_tok = sft_loss(logits, input_ids, response_mask)
        return (loss, n_tok), {}

    from opd.loss.kl import compute_kl_loss

    K = teacher_topk_logps.size(-1)
    n = n_kl_logprobs
    if n > K:
        raise ValueError(
            f"n_kl_logprobs={n} but data only has {K} teacher logprobs. "
            f"Regenerate data with --n-logprobs >= {n} or reduce n_kl_logprobs.")
    if n < K:
        teacher_topk_logps = teacher_topk_logps[:, :, :n]
        teacher_topk_indices = teacher_topk_indices[:, :, :n]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_teacher_logps = teacher_topk_logps[:, :-1, :].contiguous()
    shift_teacher_idx = teacher_topk_indices[:, :-1, :].contiguous()
    shift_mask = (response_mask[:, 1:] & teacher_valid_mask[:, :-1]).contiguous()

    kl_loss = compute_kl_loss(
        shift_logits, shift_teacher_logps, shift_teacher_idx,
        mask=shift_mask, kl_config=resolve_kl_config(kl_config))

    if sft_loss_mode == "kl":
        n_tok = int(shift_mask.sum().item())
        return (kl_loss, n_tok), {"kl_component": kl_loss.detach().item()}

    ce_loss, n_tok = sft_loss(logits, input_ids, response_mask)
    mixed = ce_alpha * ce_loss + (1 - ce_alpha) * kl_loss
    return (mixed, n_tok), {
        "ce_component": ce_loss.detach().item(),
        "kl_component": kl_loss.detach().item(),
    }

"""Helpers for multi-sample Monte Carlo rollout sampling."""

from __future__ import annotations

import torch


def sample_multiset_from_log_probs(
    log_probs: torch.Tensor,
    actual_token_ids: torch.Tensor,
    n_total_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a literal multiset from per-row log-probabilities.

    Args:
        log_probs: [R, V] float tensor of log-probabilities.
        actual_token_ids: [R] long/int tensor of actually generated token ids.
        n_total_samples: total number of samples per row, including the actual token.

    Returns:
        (sample_ids, sample_logps) each shaped [R, N]. Column 0 is always the
        actual token / its log-probability. Remaining columns are sampled with
        replacement from the same categorical distribution.
    """
    if n_total_samples < 1:
        raise ValueError(f"n_total_samples must be >= 1, got {n_total_samples}")
    if log_probs.dim() != 2:
        raise ValueError(f"log_probs must be rank-2 [R, V], got {tuple(log_probs.shape)}")
    if actual_token_ids.dim() != 1 or actual_token_ids.size(0) != log_probs.size(0):
        raise ValueError(
            "actual_token_ids must be rank-1 with same row count as log_probs: "
            f"{tuple(actual_token_ids.shape)} vs {tuple(log_probs.shape)}"
        )

    actual_token_ids = actual_token_ids.long()
    rows = log_probs.size(0)
    sample_ids = torch.empty(
        rows, n_total_samples, dtype=torch.long, device=log_probs.device)
    sample_logps = torch.empty(
        rows, n_total_samples, dtype=log_probs.dtype, device=log_probs.device)

    sample_ids[:, 0] = actual_token_ids
    sample_logps[:, 0] = log_probs.gather(
        dim=-1, index=actual_token_ids.unsqueeze(-1)).squeeze(-1)

    if n_total_samples > 1:
        probs = log_probs.exp()
        extra_ids = torch.multinomial(
            probs, num_samples=n_total_samples - 1, replacement=True)
        extra_logps = log_probs.gather(dim=-1, index=extra_ids)
        sample_ids[:, 1:] = extra_ids
        sample_logps[:, 1:] = extra_logps

    return sample_ids, sample_logps


def sample_teacher_multiset_from_log_probs(
    log_probs: torch.Tensor,
    n_total_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a literal multiset from teacher log-probabilities.

    Unlike `sample_multiset_from_log_probs`, this does not force the actual
    generated token into column 0; all samples are drawn from the teacher
    categorical distribution with replacement.
    """
    if n_total_samples < 1:
        raise ValueError(f"n_total_samples must be >= 1, got {n_total_samples}")
    if log_probs.dim() != 2:
        raise ValueError(f"log_probs must be rank-2 [R, V], got {tuple(log_probs.shape)}")

    probs = log_probs.exp()
    sample_ids = torch.multinomial(probs, num_samples=n_total_samples, replacement=True)
    sample_logps = log_probs.gather(dim=-1, index=sample_ids)
    return sample_ids, sample_logps


def sample_multiset_from_logits(
    logits: torch.Tensor,
    actual_token_ids: torch.Tensor,
    n_total_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a literal multiset directly from logits.

    This avoids materializing a full log-softmax tensor when callers only need:
    - sampled token ids
    - log-probabilities for the sampled ids
    """
    if n_total_samples < 1:
        raise ValueError(f"n_total_samples must be >= 1, got {n_total_samples}")
    if logits.dim() != 2:
        raise ValueError(f"logits must be rank-2 [R, V], got {tuple(logits.shape)}")
    if actual_token_ids.dim() != 1 or actual_token_ids.size(0) != logits.size(0):
        raise ValueError(
            "actual_token_ids must be rank-1 with same row count as logits: "
            f"{tuple(actual_token_ids.shape)} vs {tuple(logits.shape)}"
        )

    actual_token_ids = actual_token_ids.long()
    rows = logits.size(0)
    sample_ids = torch.empty(
        rows, n_total_samples, dtype=torch.long, device=logits.device)
    sample_logps = torch.empty(
        rows, n_total_samples, dtype=logits.dtype, device=logits.device)

    log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
    sample_ids[:, 0] = actual_token_ids
    sample_logps[:, 0] = (
        logits.gather(dim=-1, index=actual_token_ids.unsqueeze(-1)) - log_norm
    ).squeeze(-1)

    if n_total_samples > 1:
        probs = torch.softmax(logits, dim=-1)
        extra_ids = torch.multinomial(
            probs, num_samples=n_total_samples - 1, replacement=True)
        extra_logps = logits.gather(dim=-1, index=extra_ids) - log_norm
        sample_ids[:, 1:] = extra_ids
        sample_logps[:, 1:] = extra_logps

    return sample_ids, sample_logps


def sample_teacher_multiset_from_logits(
    logits: torch.Tensor,
    n_total_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a literal multiset directly from teacher logits."""
    if n_total_samples < 1:
        raise ValueError(f"n_total_samples must be >= 1, got {n_total_samples}")
    if logits.dim() != 2:
        raise ValueError(f"logits must be rank-2 [R, V], got {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=-1)
    sample_ids = torch.multinomial(probs, num_samples=n_total_samples, replacement=True)
    log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
    sample_logps = logits.gather(dim=-1, index=sample_ids) - log_norm
    return sample_ids, sample_logps


def sample_multiset_from_probs(
    probs: torch.Tensor,
    actual_token_ids: torch.Tensor,
    n_total_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a literal multiset directly from probabilities."""
    if n_total_samples < 1:
        raise ValueError(f"n_total_samples must be >= 1, got {n_total_samples}")
    if probs.dim() != 2:
        raise ValueError(f"probs must be rank-2 [R, V], got {tuple(probs.shape)}")
    if actual_token_ids.dim() != 1 or actual_token_ids.size(0) != probs.size(0):
        raise ValueError(
            "actual_token_ids must be rank-1 with same row count as probs: "
            f"{tuple(actual_token_ids.shape)} vs {tuple(probs.shape)}"
        )

    actual_token_ids = actual_token_ids.long()
    rows = probs.size(0)
    sample_ids = torch.empty(rows, n_total_samples, dtype=torch.long, device=probs.device)
    sample_logps = torch.empty(rows, n_total_samples, dtype=probs.dtype, device=probs.device)

    sample_ids[:, 0] = actual_token_ids
    actual_probs = probs.gather(dim=-1, index=actual_token_ids.unsqueeze(-1)).squeeze(-1)
    sample_logps[:, 0] = actual_probs.log()

    if n_total_samples > 1:
        extra_ids = torch.multinomial(probs, num_samples=n_total_samples - 1, replacement=True)
        extra_probs = probs.gather(dim=-1, index=extra_ids)
        sample_ids[:, 1:] = extra_ids
        sample_logps[:, 1:] = extra_probs.log()

    return sample_ids, sample_logps


def sample_multiset_all_from_probs(
    probs: torch.Tensor,
    n_total_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample all columns directly from probabilities.

    Unlike sample_multiset_from_probs(), column 0 is also sampled here instead
    of being forced to a provided actual token. This is used on the sync rollout
    hot path so the actual token and extra MC samples come from one batched draw.
    """
    if n_total_samples < 1:
        raise ValueError(f"n_total_samples must be >= 1, got {n_total_samples}")
    if probs.dim() != 2:
        raise ValueError(f"probs must be rank-2 [R, V], got {tuple(probs.shape)}")

    rows = probs.size(0)
    cdf = probs.cumsum(dim=-1)
    cdf[:, -1] = 1.0
    uniform = torch.rand(rows, n_total_samples, device=probs.device, dtype=probs.dtype)
    sample_ids = torch.searchsorted(cdf, uniform, right=False).long()
    sample_logps = probs.gather(dim=-1, index=sample_ids).log()
    return sample_ids, sample_logps

"""Advantage helpers for actor-critic OPD losses."""

import torch


def _continuation_mask(mask):
    cont = torch.zeros_like(mask, dtype=torch.bool)
    if mask.size(1) > 1:
        cont[:, :-1] = mask[:, 1:]
    return cont


def masked_normalize(tensor, mask, eps=1e-8):
    """Normalize values over masked positions only."""
    out = tensor.clone()
    vals = out[mask]
    if vals.numel() <= 1:
        out[~mask] = 0
        return out
    mean = vals.mean()
    std = vals.std(unbiased=False)
    out[mask] = (vals - mean) / (std + eps)
    out[~mask] = 0
    return out


def compute_td0_advantages(rewards, values, next_values, mask, gamma=1.0):
    """Masked TD(0) advantages with terminal bootstrap suppression."""
    cont = _continuation_mask(mask).to(values.dtype)
    advantages = rewards + gamma * next_values * cont - values
    return advantages * mask.to(advantages.dtype)


def compute_gae_advantages(rewards, values, mask, gamma=1.0, lam=0.95):
    """Masked GAE(λ) with gamma/lam recursion over response-token positions."""
    next_values = torch.zeros_like(values)
    if values.size(1) > 1:
        next_values[:, :-1] = values[:, 1:]
    deltas = compute_td0_advantages(rewards, values, next_values, mask, gamma=gamma)
    cont = _continuation_mask(mask).to(values.dtype)
    advantages = torch.zeros_like(values)
    gae = torch.zeros(values.size(0), device=values.device, dtype=values.dtype)
    for t in range(values.size(1) - 1, -1, -1):
        gae = deltas[:, t] + gamma * lam * cont[:, t] * gae
        gae = gae * mask[:, t].to(values.dtype)
        advantages[:, t] = gae
    return advantages


def compute_returns_from_advantages(values, advantages, mask):
    """Masked bootstrap returns."""
    returns = (values + advantages) * mask.to(values.dtype)
    return returns

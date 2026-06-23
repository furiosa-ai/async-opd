"""Unit tests for sft_loss."""

import pytest
import torch
import torch.nn.functional as F

from opd.loss.sft import sft_loss


def make_logits(B, T, V, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, T, V, requires_grad=True)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_returns_scalar_and_n_tokens():
    B, T, V = 2, 8, 100
    logits = make_logits(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    response_mask = torch.zeros(B, T, dtype=torch.bool)
    response_mask[:, 4:] = True  # completion starts at position 4

    loss, n_tokens = sft_loss(logits, input_ids, response_mask)
    assert loss.shape == (), "loss should be a scalar"
    assert isinstance(n_tokens, int)


def test_n_tokens_counts_shifted_mask():
    """n_tokens reflects completion tokens in the shifted (T-1) sequence."""
    B, T, V = 2, 8, 100
    logits = make_logits(B, T, V)
    input_ids = torch.randint(0, V, (B, T))

    # response_mask[:,4:] = True  → after shift, mask[:,4:] maps to mask[:,1:][:,3:]
    # shifted mask = response_mask[:,1:], so positions 4..7 → indices 3..6 in shifted
    response_mask = torch.zeros(B, T, dtype=torch.bool)
    response_mask[:, 4:] = True  # 4 True per row

    _, n_tokens = sft_loss(logits, input_ids, response_mask)
    # shifted mask: response_mask[:,1:] has True at positions 3..6 (4 per row)
    expected = int((response_mask[:, 1:]).float().sum().item())
    assert n_tokens == expected


def test_loss_zero_on_prompt_positions():
    """Loss should come only from completion positions; prompt positions contribute 0."""
    B, T, V = 1, 6, 50
    # Make logits that predict token 0 perfectly (only token 0 has high logit)
    logits = torch.zeros(B, T, V)
    logits[:, :, 0] = 1e9  # model always predicts token 0

    input_ids = torch.zeros(B, T, dtype=torch.long)  # all labels = 0, so CE = 0 everywhere

    # Mask: only positions 3..5 are completion
    response_mask = torch.zeros(B, T, dtype=torch.bool)
    response_mask[:, 3:] = True

    loss, n_tokens = sft_loss(logits, input_ids, response_mask)
    assert loss.item() < 1e-4, f"Expected near-zero loss, got {loss.item()}"


def test_loss_nonzero_when_wrong():
    """Loss should be > 0 when model predicts wrong tokens in completion positions."""
    B, T, V = 1, 6, 50
    logits = torch.zeros(B, T, V)
    logits[:, :, 0] = 1e9  # model always predicts token 0

    # Labels are token 1 — model is wrong everywhere
    input_ids = torch.ones(B, T, dtype=torch.long)

    response_mask = torch.zeros(B, T, dtype=torch.bool)
    response_mask[:, 3:] = True

    loss, _ = sft_loss(logits, input_ids, response_mask)
    assert loss.item() > 1.0, f"Expected large loss for wrong predictions, got {loss.item()}"


def test_loss_only_completion_not_prompt():
    """Changing only prompt logits should not affect loss value."""
    B, T, V = 2, 10, 50
    torch.manual_seed(42)
    logits_base = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))

    response_mask = torch.zeros(B, T, dtype=torch.bool)
    response_mask[:, 5:] = True  # completion at positions 5..9

    loss_base, _ = sft_loss(logits_base, input_ids, response_mask)

    # Perturb ONLY prompt logits (positions 0..4 → affects shifted positions 0..3)
    logits_perturbed = logits_base.clone()
    logits_perturbed[:, :5, :] += 1000.0  # huge perturbation in prompt region

    loss_perturbed, _ = sft_loss(logits_perturbed, input_ids, response_mask)

    assert torch.isclose(loss_base, loss_perturbed, atol=1e-4), (
        f"Prompt perturbation changed loss: {loss_base.item()} vs {loss_perturbed.item()}"
    )


# ---------------------------------------------------------------------------
# Edge case: zero completion tokens
# ---------------------------------------------------------------------------

def test_zero_completion_tokens():
    """When response_mask is all False, loss should be 0 (clamped denominator)."""
    B, T, V = 2, 8, 50
    logits = make_logits(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    response_mask = torch.zeros(B, T, dtype=torch.bool)  # no completion tokens

    loss, n_tokens = sft_loss(logits, input_ids, response_mask)
    assert loss.item() == 0.0, f"Expected 0 loss with empty mask, got {loss.item()}"
    assert n_tokens == 0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flows():
    """Loss.backward() should produce gradients on logits."""
    B, T, V = 2, 8, 100
    logits = make_logits(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    response_mask = torch.zeros(B, T, dtype=torch.bool)
    response_mask[:, 4:] = True

    loss, _ = sft_loss(logits, input_ids, response_mask)
    loss.backward()

    assert logits.grad is not None, "Expected gradient on logits"
    assert not torch.isnan(logits.grad).any(), "Gradient contains NaN"
    assert not torch.isinf(logits.grad).any(), "Gradient contains Inf"


def test_gradient_zero_on_prompt_logits():
    """Gradients should be zero for prompt logit positions after backward."""
    B, T, V = 1, 8, 50
    torch.manual_seed(7)
    logits = torch.randn(B, T, V, requires_grad=True)
    input_ids = torch.randint(0, V, (B, T))

    response_mask = torch.zeros(B, T, dtype=torch.bool)
    response_mask[:, 5:] = True  # completion at positions 5..7

    loss, _ = sft_loss(logits, input_ids, response_mask)
    loss.backward()

    # Shifted mask: completion positions in logits are [:, 4:6] (shift means
    # logits[:,t] predicts labels[:,t+1]; response_mask[:,1:] is the shifted mask)
    # shifted mask True at positions 4..6 in logits dim
    shifted_mask = response_mask[:, 1:]  # [B, T-1], True at positions 4..6
    prompt_grad = logits.grad[:, :-1, :][~shifted_mask]  # prompt positions in shifted

    # All prompt position gradients should be zero (they don't affect the loss)
    assert (prompt_grad == 0).all(), "Expected zero gradients at prompt logit positions"


# ---------------------------------------------------------------------------
# Numerical equivalence with manual computation
# ---------------------------------------------------------------------------

def test_matches_manual_ce():
    """sft_loss should match a manual masked cross-entropy computation."""
    B, T, V = 2, 6, 20
    torch.manual_seed(99)
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    response_mask = torch.zeros(B, T, dtype=torch.bool)
    response_mask[:, 3:] = True

    loss, _ = sft_loss(logits, input_ids, response_mask)

    # Manual computation
    shift_logits = logits[:, :-1, :].reshape(-1, V)
    shift_labels = input_ids[:, 1:].reshape(-1)
    shift_mask = response_mask[:, 1:].reshape(-1).float()

    per_token = F.cross_entropy(shift_logits, shift_labels, reduction="none")
    expected = (per_token * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)

    assert torch.isclose(loss, expected, atol=1e-5), (
        f"Loss mismatch: {loss.item()} vs {expected.item()}"
    )

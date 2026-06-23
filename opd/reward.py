"""Reward functions for GRPO/DAPO training.

Pluggable reward function interface with a registry for extensibility.
Default: binary math correctness using existing answer extraction.
Supports DAPO extensions: overlong reward shaping, filter_groups.
"""

import torch

from opd.utils.eval import (
    extract_answer,
    answers_match,
    should_try_full_response_match,
)


# ---------------------------------------------------------------------------
# Reward function registry
# ---------------------------------------------------------------------------

REWARD_FNS = {}


def register_reward(name):
    """Decorator to register a reward function."""
    def decorator(fn):
        REWARD_FNS[name] = fn
        return fn
    return decorator


def get_reward_fn(name):
    """Look up a reward function by name."""
    if name not in REWARD_FNS:
        raise ValueError(
            f"Unknown reward_fn: {name!r}. Available: {list(REWARD_FNS.keys())}"
        )
    return REWARD_FNS[name]


# ---------------------------------------------------------------------------
# Built-in reward functions
# ---------------------------------------------------------------------------

@register_reward("correctness")
def correctness_reward(responses, ground_truths, tokenizer, group_size,
                       answer_pattern=None):
    """Binary correctness reward: 1 if answer matches, 0 otherwise.

    Args:
        responses: list of token ID lists, length B = num_prompts * G
        ground_truths: list of str, length B (ground truth repeated G times per prompt)
        tokenizer: tokenizer for decoding responses
        group_size: G (number of responses per prompt)
        answer_pattern: optional regex with capture group for strict answer extraction.
            E.g. ``"#### (\\-?[0-9\\.\\,]+)"`` for GSM8K strict format.

    Returns:
        rewards: [B] float tensor with 0.0 or 1.0
    """
    B = len(responses)
    rewards = torch.zeros(B)

    for i in range(B):
        text = tokenizer.decode(responses[i], skip_special_tokens=True)
        predicted = extract_answer(text, pattern=answer_pattern)
        # Ground truth may be a full solution (e.g., GSM8K "#### 72"),
        # so extract the answer from it too.
        gt = extract_answer(ground_truths[i], pattern=answer_pattern) or ground_truths[i]
        is_correct = answers_match(predicted, gt)
        if (
            not is_correct
            and answer_pattern is None
            and should_try_full_response_match(ground_truths[i])
        ):
            # Symbolic datasets may need full-response parsing if the lightweight
            # extractor misses nested LaTeX.
            is_correct = answers_match(text, ground_truths[i])
        if is_correct:
            rewards[i] = 1.0

    return rewards


@register_reward("token_hash")
def token_hash_reward(responses, ground_truths, tokenizer, group_size,
                      answer_pattern=None):
    """Deterministic pseudo-reward based on response token content.

    Maps token ID sum to [0, 1] via modular hash. Useful for integration
    tests where tiny models never produce correct answers but we still
    need non-zero group variance for GRPO/DAPO loss testing.
    """
    B = len(responses)
    rewards = torch.zeros(B)
    for i in range(B):
        token_sum = sum(responses[i]) if responses[i] else 0
        # Map to 0 or 1 based on parity — guarantees variance if responses differ
        rewards[i] = float(token_sum % 2)
    return rewards


# ---------------------------------------------------------------------------
# Overlong reward shaping (DAPO)
# ---------------------------------------------------------------------------

def apply_overlong_penalty(rewards, response_lengths, max_response_length,
                           overlong_buffer_len, penalty_factor=1.0):
    """Apply linear penalty for responses that exceed a length threshold.

    DAPO's Overlong Reward Shaping: responses within [expected_len, max_len]
    get a linearly increasing penalty from 0 to -penalty_factor.

    Args:
        rewards: [B] reward tensor (modified in-place)
        response_lengths: [B] actual response lengths
        max_response_length: maximum generation length
        overlong_buffer_len: buffer zone length (e.g., 4096)
        penalty_factor: max penalty magnitude (default 1.0)

    Returns:
        rewards: [B] modified reward tensor
    """
    expected_len = max_response_length - overlong_buffer_len
    exceed_len = response_lengths.float() - expected_len
    penalty = torch.clamp(-exceed_len / overlong_buffer_len * penalty_factor, max=0.0)
    return rewards + penalty


# ---------------------------------------------------------------------------
# Group-relative advantage normalization
# ---------------------------------------------------------------------------

def compute_group_advantages(rewards, group_size, eps=1e-8, norm_by_std=True):
    """Normalize rewards within each group of G responses.

    Args:
        rewards: [B] tensor where B = num_prompts * G, ordered
                 [prompt0_resp0, prompt0_resp1, ..., prompt0_respG-1,
                  prompt1_resp0, ...]
        group_size: G (number of responses per prompt)
        eps: epsilon for numerical stability in std
        norm_by_std: if True (default, standard GRPO), divide by std.
                     if False (Dr.GRPO), only subtract mean.

    Returns:
        advantages: [B] tensor, normalized per group
    """
    B = rewards.size(0)
    assert B % group_size == 0, (
        f"Batch size {B} not divisible by group_size {group_size}"
    )
    num_prompts = B // group_size

    # Reshape to [num_prompts, G]
    grouped = rewards.view(num_prompts, group_size).float()

    # Per-group mean
    mean = grouped.mean(dim=1, keepdim=True)   # [num_prompts, 1]

    if group_size == 1:
        # Single sample per group: advantage = 0 (no group contrast)
        return torch.zeros(B)

    advantages = grouped - mean

    if norm_by_std:
        std = grouped.std(dim=1, keepdim=True)  # [num_prompts, 1]
        advantages = advantages / (std + eps)
        # Zero out groups where all rewards are identical (std < eps)
        zero_std_mask = std.squeeze(1) < eps    # [num_prompts]
        if zero_std_mask.any():
            advantages[zero_std_mask] = 0.0

    return advantages.view(B)


# ---------------------------------------------------------------------------
# Filter groups (DAPO dynamic sampling)
# ---------------------------------------------------------------------------

def filter_zero_variance_groups(rewards, group_size):
    """Identify groups with non-zero reward variance (informative groups).

    DAPO's dynamic sampling: groups where all responses got the same reward
    carry zero learning signal and can be filtered out.

    Args:
        rewards: [B] tensor, B = num_prompts * G
        group_size: G

    Returns:
        keep_mask: [num_prompts] bool tensor — True for groups to keep
        n_filtered: number of groups filtered out
    """
    B = rewards.size(0)
    num_prompts = B // group_size
    grouped = rewards.view(num_prompts, group_size)
    std = grouped.std(dim=1)
    keep_mask = std > 0
    n_filtered = int((~keep_mask).sum().item())
    return keep_mask, n_filtered

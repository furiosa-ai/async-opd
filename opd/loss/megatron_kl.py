"""Vocab-parallel KL losses for Megatron tensor-parallel training.

Each TP rank holds a shard of the student logits: [B, S, V/TP].
Distributed softmax is computed via two allreduces across the TP group:
  1. MAX reduce for numerical stability (subtract before exp)
  2. SUM reduce to normalize (sum of exp across all shards)

The teacher provides full-vocab top-k logprobs and indices. Each rank masks
teacher indices to its local vocab partition and contributes to the per-token
KL sum; a final SUM allreduce aggregates contributions across all TP ranks.

For token-level and policy-gradient modes, only the logprob at the sampled
token is needed — each rank checks if the token falls in its partition and
a SUM allreduce combines the single-element contributions.

When TP=1, all functions produce identical results to their counterparts
in opd/loss/kl.py.

Supported modes:
  - forward_kl:        KL(teacher || student) — sparse top-k
  - reverse_kl:        KL(student || teacher) — sparse top-k
  - skewed_kl:         alpha * forward + (1-alpha) * reverse — sparse top-k
  - token_level_kl:    per-token forward KL at sampled token
  - policy_gradient_kl: PPO-clip with reverse KL advantage at sampled token
"""

import torch
import torch.distributed

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.utils import VocabUtility

from opd.loss.kl import KLConfig, resolve_kl_config


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _distributed_softmax(vocab_parallel_logits, tp_group):
    """Numerically stable distributed softmax over TP-sharded logits.

    Args:
        vocab_parallel_logits: [B, S, V/TP] — local logit shard
        tp_group: torch.distributed process group for TP communication

    Returns:
        student_probs: [B, S, V/TP] — softmax probabilities (local shard)
        log_sum_exp: [B, S] — log of global sum-exp (for computing log-probs)
        logits_max: [B, S] — global max of logits (for numerical stability)
    """
    # 1. Local max, then global MAX reduce
    logits_max = vocab_parallel_logits.max(dim=-1).values  # [B, S]
    torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

    # 2. Shift and exp
    shifted = vocab_parallel_logits - logits_max.unsqueeze(-1)  # [B, S, V/TP]
    exp_logits = shifted.exp()  # [B, S, V/TP]

    # 3. Local sum-exp, then global SUM reduce
    sum_exp = exp_logits.sum(dim=-1)  # [B, S]
    torch.distributed.all_reduce(sum_exp, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    # 4. Student softmax probabilities (local shard)
    student_probs = exp_logits / sum_exp.unsqueeze(-1)  # [B, S, V/TP]

    # log_sum_exp = logits_max + log(sum_exp)
    log_sum_exp = logits_max + sum_exp.log()

    return student_probs, log_sum_exp, logits_max


def _gather_at_teacher_topk(student_probs, teacher_topk_indices, teacher_topk_logps,
                             V_local, rank, world_size):
    """Gather student probs/logprobs at teacher top-k indices on local vocab partition.

    Args:
        student_probs: [B, S, V/TP] — local softmax probabilities
        teacher_topk_indices: [B, S, K] — teacher top-k token indices (full vocab)
        teacher_topk_logps: [B, S, K] — teacher top-k log-probs
        V_local: per-partition vocab size
        rank: TP rank
        world_size: TP world size

    Returns:
        student_topk_probs: [B, S, K] — student probs at teacher top-k (zeroed outside partition)
        student_topk_logps: [B, S, K] — student log-probs at teacher top-k (zeroed outside partition)
        local_mask: [B, S, K] — bool mask of indices in this rank's partition
        teacher_probs_local: [B, S, K] — teacher probs (zeroed outside partition)
        teacher_logps_local: [B, S, K] — teacher log-probs (zeroed outside partition)
    """
    vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
        V_local, rank, world_size
    )

    # Boolean mask: which teacher top-k indices fall in this rank's partition
    local_mask = (teacher_topk_indices >= vocab_start) & (teacher_topk_indices < vocab_end)

    # Local indices (clamp out-of-range to valid range to avoid index errors;
    # local_mask zeroes out contributions from out-of-range indices later)
    local_indices = (teacher_topk_indices - vocab_start).clamp(min=0, max=V_local - 1)  # [B, S, K]

    # Gather student probs at teacher top-k tokens (local partition)
    B, S, K = teacher_topk_indices.shape
    student_probs_2d = student_probs.view(B * S, V_local)
    local_indices_2d = local_indices.view(B * S, K)
    arange = torch.arange(B * S, device=student_probs_2d.device)

    student_topk_probs_2d = student_probs_2d[arange.unsqueeze(-1), local_indices_2d]
    student_topk_probs = student_topk_probs_2d.view(B, S, K)

    # Log student probs; zero out positions outside this rank's partition
    student_topk_logps = torch.log(student_topk_probs.clamp(min=1e-20))
    student_topk_logps = student_topk_logps * local_mask

    # Teacher probs and logps, zeroed outside local partition
    teacher_probs = teacher_topk_logps.exp()
    teacher_probs_local = teacher_probs * local_mask
    teacher_logps_local = teacher_topk_logps * local_mask

    return student_topk_probs, student_topk_logps, local_mask, teacher_probs_local, teacher_logps_local


def _masked_mean(per_token_kl, mask):
    """Apply mask and compute mean, returning (loss, n_valid, is_empty)."""
    if mask is not None:
        valid = per_token_kl[mask]
        if valid.numel() == 0:
            loss = per_token_kl.new_zeros((), requires_grad=True)
            return loss, 0, True
        loss = valid.mean()
        n_valid = valid.numel()
    else:
        loss = per_token_kl.mean()
        n_valid = per_token_kl.numel()
    return loss, n_valid, False


# ---------------------------------------------------------------------------
# Forward KL: KL(teacher || student)
# ---------------------------------------------------------------------------

class _VocabParallelForwardKL(torch.autograd.Function):
    """Custom autograd function for TP-sharded forward KL(teacher || student).

    Forward:
      - Distributed softmax over sharded logits (2 allreduces)
      - Gather student log-probs at teacher top-k indices (local shard only)
      - Sum over top-k: teacher_prob * (teacher_logp - student_logp)
      - SUM allreduce to combine per-rank contributions into per-token KL
      - Apply response mask and return mean scalar

    Backward:
      - Gradient of forward KL w.r.t. logits:  student_prob - teacher_prob
        (student_prob is already softmax, teacher_prob scattered onto local vocab)
      - Scaled by upstream grad_output (after mask/mean)
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, teacher_topk_logps, teacher_topk_indices, mask,
                token_clip):
        tp_group = get_tensor_model_parallel_group()
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        V_local = vocab_parallel_logits.size(-1)

        student_probs, log_sum_exp, logits_max = _distributed_softmax(
            vocab_parallel_logits, tp_group)

        (student_topk_probs, student_topk_logps, local_mask,
         teacher_probs_local, teacher_logps_local) = _gather_at_teacher_topk(
            student_probs, teacher_topk_indices, teacher_topk_logps,
            V_local, rank, world_size)

        B, S, K = teacher_topk_indices.shape

        # Per-vocab forward KL: teacher_prob * (teacher_logp - student_logp)
        per_vocab_kl = teacher_probs_local * (teacher_logps_local - student_topk_logps)
        if token_clip > 0:
            per_vocab_kl = per_vocab_kl.clamp(max=token_clip)
        per_token_kl = per_vocab_kl.sum(dim=-1)  # [B, S]

        torch.distributed.all_reduce(per_token_kl, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        loss, n_valid, is_empty = _masked_mean(per_token_kl, mask)

        # teacher_prob_sum: Σ_k teacher_prob[k] per token (needed for correct backward)
        # With full-vocab teacher this is 1.0; with sparse top-k it's < 1.0.
        teacher_prob_sum = (teacher_probs_local * local_mask).sum(dim=-1)  # [B, S]
        torch.distributed.all_reduce(teacher_prob_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(student_probs, teacher_probs_local,
                              (teacher_topk_indices - vocab_parallel_logits.new_tensor(
                                  VocabUtility.vocab_range_from_per_partition_vocab_size(
                                      V_local, rank, world_size)[0], dtype=torch.long
                              )).clamp(min=0, max=V_local - 1),
                              local_mask, mask, teacher_prob_sum)
        ctx.n_valid = n_valid
        ctx.empty_mask = is_empty
        ctx.B, ctx.S, ctx.K, ctx.V_local = B, S, K, V_local

        # Attach kl_stats
        loss.kl_stats = {"_vals": per_token_kl[mask].detach() if mask is not None else per_token_kl.detach().flatten()}
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.empty_mask:
            student_probs = ctx.saved_tensors[0]
            return torch.zeros_like(student_probs), None, None, None, None

        student_probs, teacher_probs_local, local_indices, local_mask, mask, teacher_prob_sum = ctx.saved_tensors
        B, S, K, V_local = ctx.B, ctx.S, ctx.K, ctx.V_local
        n_valid = ctx.n_valid

        local_idx_2d = local_indices.view(B * S, K)
        teacher_probs_2d = (teacher_probs_local * local_mask).view(B * S, K)

        # Correct gradient for sparse top-k forward KL:
        # d/d(logit_j) = student_prob_j * teacher_prob_sum - teacher_prob_j
        # (teacher_prob_sum = Σ_k teacher_prob[k], which is < 1 for sparse top-k)
        grad_logits = student_probs * teacher_prob_sum.unsqueeze(-1)  # [B, S, V/TP]
        grad_2d = grad_logits.view(B * S, V_local)
        grad_2d.scatter_add_(1, local_idx_2d, -teacher_probs_2d)

        if mask is not None:
            grad_logits = grad_logits * mask.unsqueeze(-1).float()

        scale = grad_output / n_valid
        grad_logits = grad_logits * scale

        return grad_logits, None, None, None, None


def vocab_parallel_forward_kl(
    student_logits: torch.Tensor,
    teacher_topk_logps: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    mask: torch.Tensor | None = None,
    token_clip: float = 0.0,
) -> torch.Tensor:
    """Forward KL(teacher || student) with TP-sharded student logits.

    Each TP rank holds student_logits with shape [B, S, V/TP]. The teacher
    top-k logprobs and indices are full-vocab (not sharded) and identical
    on all ranks.

    When TP=1, produces identical results to sparse_forward_kl from kl.py.

    Args:
        student_logits:        [B, S, V/TP] — TP-sharded student logits
        teacher_topk_logps:    [B, S, K]    — teacher log-probs (top-K, full vocab)
        teacher_topk_indices:  [B, S, K]    — teacher top-K token indices (full vocab)
        mask:                  [B, S]       — bool response mask (True = include)
        token_clip:            per-vocab-element KL clip threshold (0 = disabled)

    Returns:
        Scalar loss tensor (mean over masked positions).
    """
    return _VocabParallelForwardKL.apply(
        student_logits, teacher_topk_logps, teacher_topk_indices, mask, token_clip
    )


# ---------------------------------------------------------------------------
# Reverse KL: KL(student || teacher)
# ---------------------------------------------------------------------------

class _VocabParallelReverseKL(torch.autograd.Function):
    """Custom autograd function for TP-sharded reverse KL(student || teacher).

    Forward:
      - Distributed softmax over sharded logits
      - Gather student probs/logprobs at teacher top-k indices
      - Sum over top-k: student_prob * (student_logp - teacher_logp)
      - SUM allreduce to combine per-rank contributions

    Backward:
      - Gradient of reverse KL w.r.t. logits at position j:
        d/d logit_j = Σ_k [d/d logit_j (s_k * (log s_k - t_k))]
        For k in top-k: d/d logit_j s_k * (1 + log s_k - t_k)
        Using softmax Jacobian: d s_k / d z_j = s_k (delta_{jk} - s_j)
        Full gradient: Σ_k s_k(1 + log s_k - t_k)(delta_{jk} - s_j)
          = s_j(1 + log s_j - t_j) [if j in top-k] - s_j * Σ_k s_k(1 + log s_k - t_k)
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, teacher_topk_logps, teacher_topk_indices, mask,
                token_clip):
        tp_group = get_tensor_model_parallel_group()
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        V_local = vocab_parallel_logits.size(-1)

        student_probs, log_sum_exp, logits_max = _distributed_softmax(
            vocab_parallel_logits, tp_group)

        (student_topk_probs, student_topk_logps, local_mask,
         teacher_probs_local, teacher_logps_local) = _gather_at_teacher_topk(
            student_probs, teacher_topk_indices, teacher_topk_logps,
            V_local, rank, world_size)

        B, S, K = teacher_topk_indices.shape

        # Per-vocab reverse KL: student_prob * (student_logp - teacher_logp)
        # Only count terms where local_mask is True (index in this partition)
        student_topk_probs_local = student_topk_probs * local_mask
        per_vocab_kl = student_topk_probs_local * (student_topk_logps - teacher_logps_local)
        if token_clip > 0:
            per_vocab_kl = per_vocab_kl.clamp(max=token_clip)
        per_token_kl = per_vocab_kl.sum(dim=-1)  # [B, S]

        torch.distributed.all_reduce(per_token_kl, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        loss, n_valid, is_empty = _masked_mean(per_token_kl, mask)

        # Save for backward
        vocab_start = VocabUtility.vocab_range_from_per_partition_vocab_size(
            V_local, rank, world_size)[0]
        local_indices = (teacher_topk_indices - vocab_start).clamp(min=0, max=V_local - 1)

        ctx.save_for_backward(student_probs, student_topk_probs_local, student_topk_logps,
                              teacher_logps_local, local_indices, local_mask, mask)
        ctx.n_valid = n_valid
        ctx.empty_mask = is_empty
        ctx.B, ctx.S, ctx.K, ctx.V_local = B, S, K, V_local

        loss.kl_stats = {"_vals": per_token_kl[mask].detach() if mask is not None else per_token_kl.detach().flatten()}
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.empty_mask:
            student_probs = ctx.saved_tensors[0]
            return torch.zeros_like(student_probs), None, None, None, None

        (student_probs, student_topk_probs_local, student_topk_logps,
         teacher_logps_local, local_indices, local_mask, mask) = ctx.saved_tensors
        B, S, K, V_local = ctx.B, ctx.S, ctx.K, ctx.V_local
        n_valid = ctx.n_valid

        # Weight per top-k element: (1 + log s_k - t_k) * local_mask
        weights = (1.0 + student_topk_logps - teacher_logps_local) * local_mask  # [B, S, K]

        # Σ_k s_k * weight_k (for the softmax correction term)
        weighted_sum = (student_topk_probs_local * weights).sum(dim=-1, keepdim=True)  # [B, S, 1]

        # Scatter term: at top-k positions j, add s_j * weight_j
        local_idx_2d = local_indices.view(B * S, K)
        scatter_vals = (student_topk_probs_local * weights).view(B * S, K)

        grad_logits = -student_probs * weighted_sum  # [B, S, V/TP]
        grad_2d = grad_logits.view(B * S, V_local)
        grad_2d.scatter_add_(1, local_idx_2d, scatter_vals)

        if mask is not None:
            grad_logits = grad_logits * mask.unsqueeze(-1).float()

        scale = grad_output / n_valid
        grad_logits = grad_logits * scale

        return grad_logits, None, None, None, None


def vocab_parallel_reverse_kl(
    student_logits: torch.Tensor,
    teacher_topk_logps: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    mask: torch.Tensor | None = None,
    token_clip: float = 0.0,
) -> torch.Tensor:
    """Reverse KL(student || teacher) with TP-sharded student logits.

    Summed over teacher's top-k tokens only (same approximation as
    sparse_reverse_kl in kl.py).

    Args:
        student_logits:        [B, S, V/TP] — TP-sharded student logits
        teacher_topk_logps:    [B, S, K]    — teacher log-probs (top-K)
        teacher_topk_indices:  [B, S, K]    — teacher top-K token indices
        mask:                  [B, S]       — bool response mask
        token_clip:            per-vocab-element KL clip threshold (0 = disabled)

    Returns:
        Scalar loss tensor.
    """
    return _VocabParallelReverseKL.apply(
        student_logits, teacher_topk_logps, teacher_topk_indices, mask, token_clip
    )


# ---------------------------------------------------------------------------
# Skewed KL: alpha * forward + (1-alpha) * reverse
# ---------------------------------------------------------------------------

class _VocabParallelSkewedKL(torch.autograd.Function):
    """Custom autograd function for TP-sharded skewed KL.

    Combines per-vocab forward and reverse KL before summing over K.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, teacher_topk_logps, teacher_topk_indices, mask,
                alpha, token_clip):
        tp_group = get_tensor_model_parallel_group()
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        V_local = vocab_parallel_logits.size(-1)

        student_probs, log_sum_exp, logits_max = _distributed_softmax(
            vocab_parallel_logits, tp_group)

        (student_topk_probs, student_topk_logps, local_mask,
         teacher_probs_local, teacher_logps_local) = _gather_at_teacher_topk(
            student_probs, teacher_topk_indices, teacher_topk_logps,
            V_local, rank, world_size)

        B, S, K = teacher_topk_indices.shape

        student_topk_probs_local = student_topk_probs * local_mask

        # Per-vocab forward: teacher_prob * (teacher_logp - student_logp)
        fwd_per_vocab = teacher_probs_local * (teacher_logps_local - student_topk_logps)
        # Per-vocab reverse: student_prob * (student_logp - teacher_logp)
        rev_per_vocab = student_topk_probs_local * (student_topk_logps - teacher_logps_local)

        per_vocab_kl = alpha * fwd_per_vocab + (1 - alpha) * rev_per_vocab
        if token_clip > 0:
            per_vocab_kl = per_vocab_kl.clamp(max=token_clip)
        per_token_kl = per_vocab_kl.sum(dim=-1)

        torch.distributed.all_reduce(per_token_kl, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        loss, n_valid, is_empty = _masked_mean(per_token_kl, mask)

        # teacher_prob_sum for correct forward KL backward (sparse top-k)
        teacher_prob_sum = (teacher_probs_local * local_mask).sum(dim=-1)  # [B, S]
        torch.distributed.all_reduce(teacher_prob_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        vocab_start = VocabUtility.vocab_range_from_per_partition_vocab_size(
            V_local, rank, world_size)[0]
        local_indices = (teacher_topk_indices - vocab_start).clamp(min=0, max=V_local - 1)

        ctx.save_for_backward(student_probs, student_topk_probs_local, student_topk_logps,
                              teacher_probs_local, teacher_logps_local,
                              local_indices, local_mask, mask, teacher_prob_sum)
        ctx.n_valid = n_valid
        ctx.empty_mask = is_empty
        ctx.B, ctx.S, ctx.K, ctx.V_local = B, S, K, V_local
        ctx.alpha = alpha

        loss.kl_stats = {"_vals": per_token_kl[mask].detach() if mask is not None else per_token_kl.detach().flatten()}
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.empty_mask:
            student_probs = ctx.saved_tensors[0]
            return torch.zeros_like(student_probs), None, None, None, None, None

        (student_probs, student_topk_probs_local, student_topk_logps,
         teacher_probs_local, teacher_logps_local,
         local_indices, local_mask, mask, teacher_prob_sum) = ctx.saved_tensors
        B, S, K, V_local = ctx.B, ctx.S, ctx.K, ctx.V_local
        n_valid = ctx.n_valid
        alpha = ctx.alpha

        # Forward KL gradient: s_j * teacher_prob_sum - t_j (sparse top-k corrected)
        # Reverse KL gradient: Σ_k s_k(1+log s_k - t_k)(delta_{jk} - s_j)

        # Forward part: scatter -teacher_probs, scale student_probs by teacher_prob_sum
        local_idx_2d = local_indices.view(B * S, K)
        teacher_probs_2d = (teacher_probs_local * local_mask).view(B * S, K)

        grad_fwd = student_probs * teacher_prob_sum.unsqueeze(-1)
        grad_fwd_2d = grad_fwd.view(B * S, V_local)
        grad_fwd_2d.scatter_add_(1, local_idx_2d, -teacher_probs_2d)

        # Reverse part
        weights = (1.0 + student_topk_logps - teacher_logps_local) * local_mask
        weighted_sum = (student_topk_probs_local * weights).sum(dim=-1, keepdim=True)
        scatter_vals = (student_topk_probs_local * weights).view(B * S, K)

        grad_rev = -student_probs * weighted_sum
        grad_rev_2d = grad_rev.view(B * S, V_local)
        grad_rev_2d.scatter_add_(1, local_idx_2d, scatter_vals)

        grad_logits = alpha * grad_fwd + (1 - alpha) * grad_rev

        if mask is not None:
            grad_logits = grad_logits * mask.unsqueeze(-1).float()

        scale = grad_output / n_valid
        grad_logits = grad_logits * scale

        return grad_logits, None, None, None, None, None


def vocab_parallel_skewed_kl(
    student_logits: torch.Tensor,
    teacher_topk_logps: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    mask: torch.Tensor | None = None,
    alpha: float = 0.5,
    token_clip: float = 0.0,
) -> torch.Tensor:
    """Skewed KL: alpha * forward + (1-alpha) * reverse, with TP-sharded logits.

    Args:
        student_logits:        [B, S, V/TP] — TP-sharded student logits
        teacher_topk_logps:    [B, S, K]    — teacher log-probs (top-K)
        teacher_topk_indices:  [B, S, K]    — teacher top-K token indices
        mask:                  [B, S]       — bool response mask
        alpha:                 weight for forward KL (0=pure reverse, 1=pure forward)
        token_clip:            per-vocab-element KL clip threshold (0 = disabled)

    Returns:
        Scalar loss tensor.
    """
    return _VocabParallelSkewedKL.apply(
        student_logits, teacher_topk_logps, teacher_topk_indices, mask, alpha, token_clip
    )


# ---------------------------------------------------------------------------
# Token-level KL: per-token forward KL at sampled token
# ---------------------------------------------------------------------------

def vocab_parallel_token_level_kl(
    student_logits: torch.Tensor,
    teacher_token_logps: torch.Tensor,
    input_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token forward KL contribution using teacher's logprob at sampled token.

    At each position t, computes:
        p_t(y_t) * [log p_t(y_t) - log p_s(y_t)]

    where y_t = input_ids[t+1] is the actual next token.

    Each rank checks if the sampled token falls in its vocab partition and
    gathers the student logprob locally; a SUM allreduce combines contributions.

    Args:
        student_logits:     [B, S, V/TP] — TP-sharded student logits
        teacher_token_logps: [B, S] — log π_teacher(actual_token)
        input_ids:          [B, S] — token IDs of the sequence
        mask:               [B, S] — bool response mask

    Returns:
        Scalar loss tensor.
    """
    tp_group = get_tensor_model_parallel_group()
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    V_local = student_logits.size(-1)

    # Shift: predict next token
    logits = student_logits[:, :-1]         # [B, S-1, V/TP]
    target_ids = input_ids[:, 1:]           # [B, S-1]
    t_logps = teacher_token_logps[:, :-1]   # [B, S-1]

    # Distributed log-softmax at target token
    student_probs, log_sum_exp, logits_max = _distributed_softmax(logits, tp_group)

    vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
        V_local, rank, world_size
    )

    # Check which tokens fall in this rank's partition
    local_mask = (target_ids >= vocab_start) & (target_ids < vocab_end)  # [B, S-1]
    local_indices = (target_ids - vocab_start).clamp(min=0, max=V_local - 1)  # [B, S-1]

    # Gather student logprob at target token
    B, S_shifted = local_indices.shape
    probs_2d = student_probs.view(B * S_shifted, V_local)
    idx_2d = local_indices.view(B * S_shifted, 1)
    arange = torch.arange(B * S_shifted, device=probs_2d.device)

    student_token_probs = probs_2d[arange.unsqueeze(-1), idx_2d].view(B, S_shifted)
    student_token_logps = torch.log(student_token_probs.clamp(min=1e-20))
    student_token_logps = student_token_logps * local_mask  # zero non-local

    # SUM allreduce: each rank contributes its partition's logprob
    torch.distributed.all_reduce(student_token_logps, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    # Per-token KL: teacher_prob * (teacher_logp - student_logp)
    teacher_probs = torch.exp(t_logps).detach()
    per_token_kl = teacher_probs * (t_logps.detach() - student_token_logps)

    # Shift mask
    m = mask[:, 1:] if mask is not None else None

    if m is not None:
        masked = per_token_kl[m]
        loss = masked.mean() if masked.numel() > 0 else per_token_kl.new_zeros((), requires_grad=True)
    else:
        loss = per_token_kl.mean()
    loss.kl_stats = {"_vals": per_token_kl[m].detach() if m is not None else per_token_kl.detach().flatten()}
    return loss


# ---------------------------------------------------------------------------
# Policy gradient KL: PPO-clip with reverse KL advantage
# ---------------------------------------------------------------------------

def vocab_parallel_policy_gradient_kl(
    student_logits: torch.Tensor,
    teacher_token_logps: torch.Tensor,
    input_ids: torch.Tensor,
    student_old_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO-clip style policy gradient with per-token reverse KL as advantage.

    Same as policy_gradient_kl in kl.py but with TP-sharded student logits.
    Only the student logprob gather needs TP coordination; the ratio and
    advantage computations are per-token scalars.

    Args:
        student_logits:       [B, S, V/TP] — TP-sharded student logits
        teacher_token_logps:  [B, S] — log π_teacher(actual_token)
        input_ids:            [B, S] — token IDs
        student_old_logprobs: [B, resp_len] — log π_old from rollout
        mask:                 [B, S] — bool response mask
        clip_eps:             PPO clipping epsilon (default 0.2)

    Returns:
        Scalar loss tensor.
    """
    tp_group = get_tensor_model_parallel_group()
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    V_local = student_logits.size(-1)

    # Shift: predict next token
    logits = student_logits[:, :-1]         # [B, S-1, V/TP]
    target_ids = input_ids[:, 1:]           # [B, S-1]
    t_logps = teacher_token_logps[:, :-1]   # [B, S-1]

    # Distributed log-softmax at target token
    student_probs, log_sum_exp, logits_max = _distributed_softmax(logits, tp_group)

    vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
        V_local, rank, world_size
    )

    local_mask = (target_ids >= vocab_start) & (target_ids < vocab_end)
    local_indices = (target_ids - vocab_start).clamp(min=0, max=V_local - 1)

    B, S_shifted = local_indices.shape
    probs_2d = student_probs.view(B * S_shifted, V_local)
    idx_2d = local_indices.view(B * S_shifted, 1)
    arange = torch.arange(B * S_shifted, device=probs_2d.device)

    student_token_probs = probs_2d[arange.unsqueeze(-1), idx_2d].view(B, S_shifted)
    student_token_logps = torch.log(student_token_probs.clamp(min=1e-20))
    student_token_logps = student_token_logps * local_mask

    torch.distributed.all_reduce(student_token_logps, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    # Align old logprobs (same logic as policy_gradient_kl eval path)
    resp_len = student_old_logprobs.size(1)
    old_logps = torch.zeros(B, S_shifted, device=student_logits.device,
                            dtype=student_old_logprobs.dtype)
    usable_resp = min(resp_len, S_shifted)
    old_logps[:, -usable_resp:] = student_old_logprobs[:, :usable_resp]

    # Advantage: log π_teacher - log π_old (detached)
    advantage = (t_logps - old_logps).detach()

    # Importance ratio: π_θ / π_old
    log_ratio = student_token_logps - old_logps.detach()
    ratio = log_ratio.exp()

    # PPO clipped surrogate
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage
    per_token_loss = -torch.min(surr1, surr2)

    m = mask[:, 1:] if mask is not None else None

    if m is not None:
        masked = per_token_loss[m]
        loss = masked.mean() if masked.numel() > 0 else per_token_loss.new_zeros((), requires_grad=True)
        loss.pg_stats = {
            "_ratios": ratio[m].detach(),
            "_log_ratios": log_ratio[m].detach(),
            "_advantages": advantage[m].detach(),
            "_clip_high": (ratio > 1.0 + clip_eps)[m],
            "_clip_low": (ratio < 1.0 - clip_eps)[m],
        }
        return loss

    loss = per_token_loss.mean()
    loss.pg_stats = {
        "_ratios": ratio.detach(),
        "_log_ratios": log_ratio.detach(),
        "_advantages": advantage.detach(),
        "_clip_high": (ratio > 1.0 + clip_eps),
        "_clip_low": (ratio < 1.0 - clip_eps),
    }
    return loss


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def vocab_parallel_compute_kl_loss(
    student_logits: torch.Tensor,
    teacher_topk_logps: torch.Tensor | None = None,
    teacher_topk_indices: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    teacher_token_logps: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    student_old_logprobs: torch.Tensor | None = None,
    kl_config: KLConfig | None = None,
) -> torch.Tensor:
    """Dispatch to the appropriate vocab-parallel KL function.

    Args:
        student_logits: [B, S, V/TP] — TP-sharded student logits
        teacher_topk_logps: [B, S, K] — teacher top-k log-probs
        teacher_topk_indices: [B, S, K] — teacher top-k token indices
        mask: [B, S] — bool response mask
        teacher_token_logps: [B, S] — for token_level_kl / policy_gradient_kl
        input_ids: [B, S] — for token_level_kl / policy_gradient_kl
        student_old_logprobs: [B, resp_len] — for policy_gradient_kl
        kl_config: optional KLConfig overriding mode and hyperparams

    Returns:
        Scalar loss tensor.
    """
    kl_config = resolve_kl_config(kl_config)
    mode = kl_config.mode
    skew_alpha = kl_config.skew_alpha
    pg_clip_eps = kl_config.pg_clip_eps
    token_clip = kl_config.token_clip

    if mode == "forward_kl":
        return vocab_parallel_forward_kl(
            student_logits, teacher_topk_logps, teacher_topk_indices,
            mask, token_clip=token_clip)
    elif mode == "reverse_kl":
        return vocab_parallel_reverse_kl(
            student_logits, teacher_topk_logps, teacher_topk_indices,
            mask, token_clip=token_clip)
    elif mode == "skewed_kl":
        return vocab_parallel_skewed_kl(
            student_logits, teacher_topk_logps, teacher_topk_indices,
            mask, alpha=skew_alpha, token_clip=token_clip)
    elif mode == "token_level_kl":
        assert teacher_token_logps is not None, "token_level_kl requires teacher_token_logps"
        assert input_ids is not None, "token_level_kl requires input_ids"
        return vocab_parallel_token_level_kl(
            student_logits, teacher_token_logps, input_ids, mask)
    elif mode == "policy_gradient_kl":
        assert teacher_token_logps is not None, "policy_gradient_kl requires teacher_token_logps"
        assert input_ids is not None, "policy_gradient_kl requires input_ids"
        assert student_old_logprobs is not None, "policy_gradient_kl requires student_old_logprobs"
        return vocab_parallel_policy_gradient_kl(
            student_logits, teacher_token_logps, input_ids,
            student_old_logprobs, mask, clip_eps=pg_clip_eps)
    else:
        raise ValueError(
            f"Unknown kl_loss_mode: {mode!r}. Choose from: forward_kl, reverse_kl, "
            f"skewed_kl, token_level_kl, policy_gradient_kl"
        )

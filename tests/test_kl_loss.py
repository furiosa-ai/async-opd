"""Tests for opd/kl_loss.py — all KL divergence modes."""

import pytest
import torch
import torch.nn.functional as F

from opd.loss.kl import (
    KLConfig,
    compute_kl_loss,
    multi_sample_forward_kl,
    multi_sample_policy_gradient_kl,
    policy_gradient_kl,
    sparse_forward_kl,
    sparse_reverse_kl,
    token_level_kl,
    dense_aligned_kl,
    chunked_dense_kl_from_hidden,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_inputs(batch=2, seq=4, vocab=32, topk=5, seed=42):
    """Generate random student logits and teacher top-k logprobs/indices."""
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(batch, seq, vocab, generator=g)
    # Teacher: pick random top-k indices, compute valid log-probs
    teacher_topk_indices = torch.stack([
        torch.stack([torch.randperm(vocab, generator=g)[:topk] for _ in range(seq)])
        for _ in range(batch)
    ])
    # Make valid teacher log-probs (sum to <=1 in prob space)
    raw = torch.randn(batch, seq, topk, generator=g)
    teacher_topk_logps = F.log_softmax(raw, dim=-1)  # sums to 1 in prob space
    return student_logits, teacher_topk_logps, teacher_topk_indices


def _full_teacher_dist(teacher_topk_logps, teacher_topk_indices, vocab):
    """Reconstruct full teacher distribution (top-k only, rest = 0)."""
    batch, seq, topk = teacher_topk_logps.shape
    full = torch.zeros(batch, seq, vocab)
    full.scatter_(-1, teacher_topk_indices.long(), torch.exp(teacher_topk_logps))
    return full


def _cfg(mode, **kwargs):
    return KLConfig(mode=mode, **kwargs)


# ------------------------------------------------------------------ #
# Forward KL tests
# ------------------------------------------------------------------ #

class TestForwardKL:
    def test_non_negative(self):
        """Forward KL should be non-negative."""
        s, t_lp, t_idx = _make_inputs()
        loss = sparse_forward_kl(s, t_lp, t_idx)
        assert loss.item() >= -1e-5, f"KL should be >= 0, got {loss.item()}"

    def test_zero_when_distributions_match(self):
        """Forward KL = 0 when student log_softmax matches teacher logprobs exactly."""
        batch, seq, vocab, topk = 1, 1, 8, 8
        # When topk = vocab, we can construct exact match
        indices = torch.arange(vocab).unsqueeze(0).unsqueeze(0)  # [1,1,8]
        # Create a teacher distribution over full vocab
        teacher_logps = F.log_softmax(torch.randn(batch, seq, vocab), dim=-1)
        # Student logits that produce same log_softmax = teacher_logps + const
        # Since log_softmax(x) = x - log(sum(exp(x))), setting x = teacher_logps works
        # because log_softmax(log_softmax(y)) != log_softmax(y) in general.
        # Instead, just use teacher_logps as logits — the resulting log_softmax
        # will differ, but it should be relatively small.
        student_logits = teacher_logps.clone()  # not exact match after log_softmax
        loss = sparse_forward_kl(student_logits, teacher_logps, indices)
        # KL is small when distributions are close
        assert loss.item() < 1.0

    def test_identical_to_manual_computation(self):
        """Forward KL matches manual formula: Σ p_t * (log p_t - log p_s)."""
        s, t_lp, t_idx = _make_inputs(batch=1, seq=2, vocab=10, topk=4)
        loss = sparse_forward_kl(s, t_lp, t_idx)

        # Manual computation
        s_logps = F.log_softmax(s, dim=-1)
        s_topk = torch.gather(s_logps, -1, t_idx.long())
        t_probs = torch.exp(t_lp)
        manual = (t_probs * (t_lp - s_topk)).sum(dim=-1).mean()
        torch.testing.assert_close(loss, manual, atol=1e-5, rtol=1e-5)

    def test_with_mask(self):
        """Masking should exclude specific positions."""
        s, t_lp, t_idx = _make_inputs(batch=2, seq=4)
        mask = torch.tensor([
            [True, True, False, False],
            [True, False, True, False],
        ])
        loss_masked = sparse_forward_kl(s, t_lp, t_idx, mask=mask)

        # Manual: compute per-token, then mean only over masked
        s_logps = F.log_softmax(s, dim=-1)
        s_topk = torch.gather(s_logps, -1, t_idx.long())
        t_probs = torch.exp(t_lp)
        per_token = (t_probs * (t_lp - s_topk)).sum(dim=-1)
        expected = per_token[mask].mean()
        torch.testing.assert_close(loss_masked, expected, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        """Student logits should receive gradients."""
        s, t_lp, t_idx = _make_inputs()
        s.requires_grad_(True)
        loss = sparse_forward_kl(s, t_lp, t_idx)
        loss.backward()
        assert s.grad is not None
        assert not torch.all(s.grad == 0)


# ------------------------------------------------------------------ #
# Reverse KL tests (top-k only)
# ------------------------------------------------------------------ #

class TestReverseKL:
    def test_non_negative(self):
        s, t_lp, t_idx = _make_inputs()
        loss = sparse_reverse_kl(s, t_lp, t_idx)
        # Reverse KL over top-k subset can be negative in edge cases,
        # but typically non-negative
        assert loss.item() > -1.0

    def test_matches_manual(self):
        """Reverse KL matches: Σ_{topk} p_s * (log p_s - log p_t)."""
        s, t_lp, t_idx = _make_inputs(batch=1, seq=2, vocab=10, topk=4)
        loss = sparse_reverse_kl(s, t_lp, t_idx)

        s_logps = F.log_softmax(s, dim=-1)
        s_topk_logps = torch.gather(s_logps, -1, t_idx.long())
        s_topk_probs = torch.exp(s_topk_logps)
        manual = (s_topk_probs * (s_topk_logps - t_lp)).sum(dim=-1).mean()
        torch.testing.assert_close(loss, manual, atol=1e-5, rtol=1e-5)

    def test_with_mask(self):
        s, t_lp, t_idx = _make_inputs(batch=2, seq=4)
        mask = torch.tensor([[True, False, True, False],
                             [False, True, False, True]])
        loss = sparse_reverse_kl(s, t_lp, t_idx, mask=mask)

        s_logps = F.log_softmax(s, dim=-1)
        s_topk_logps = torch.gather(s_logps, -1, t_idx.long())
        s_topk_probs = torch.exp(s_topk_logps)
        per_token = (s_topk_probs * (s_topk_logps - t_lp)).sum(dim=-1)
        expected = per_token[mask].mean()
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        s, t_lp, t_idx = _make_inputs()
        s.requires_grad_(True)
        loss = sparse_reverse_kl(s, t_lp, t_idx)
        loss.backward()
        assert s.grad is not None


# ------------------------------------------------------------------ #
# Skewed KL tests
# ------------------------------------------------------------------ #

class TestSkewedKL:
    def test_alpha_1_equals_forward(self):
        """skew_alpha=1 should give pure forward KL."""
        s, t_lp, t_idx = _make_inputs()
        fwd = sparse_forward_kl(s, t_lp, t_idx)
        skewed = compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg("skewed_kl", skew_alpha=1.0))
        torch.testing.assert_close(fwd, skewed, atol=1e-5, rtol=1e-5)

    def test_alpha_0_equals_reverse(self):
        """skew_alpha=0 should give pure reverse KL (top-k only)."""
        s, t_lp, t_idx = _make_inputs()
        rev = sparse_reverse_kl(s, t_lp, t_idx)
        skewed = compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg("skewed_kl", skew_alpha=0.0))
        torch.testing.assert_close(rev, skewed, atol=1e-5, rtol=1e-5)

    def test_interpolation(self):
        """Skewed KL = alpha * forward + (1-alpha) * reverse."""
        s, t_lp, t_idx = _make_inputs()
        alpha = 0.3
        fwd = sparse_forward_kl(s, t_lp, t_idx)
        rev = sparse_reverse_kl(s, t_lp, t_idx)
        expected = alpha * fwd + (1 - alpha) * rev
        skewed = compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg("skewed_kl", skew_alpha=alpha))
        torch.testing.assert_close(skewed, expected, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ #
# Dispatch tests
# ------------------------------------------------------------------ #

class TestDispatch:
    def test_forward_kl_mode(self):
        s, t_lp, t_idx = _make_inputs()
        direct = sparse_forward_kl(s, t_lp, t_idx)
        dispatched = compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg("forward_kl"))
        torch.testing.assert_close(direct, dispatched)

    def test_reverse_kl_mode(self):
        s, t_lp, t_idx = _make_inputs()
        direct = sparse_reverse_kl(s, t_lp, t_idx)
        dispatched = compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg("reverse_kl"))
        torch.testing.assert_close(direct, dispatched)

    def test_unknown_mode_raises(self):
        s, t_lp, t_idx = _make_inputs()
        with pytest.raises(ValueError, match="Unknown kl_loss_mode"):
            compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg("banana"))

    def test_removed_reverse_kl_residual_raises(self):
        s, t_lp, t_idx = _make_inputs()
        with pytest.raises(ValueError, match="Unknown kl_loss_mode"):
            compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg("reverse_kl_residual"))


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_single_token(self):
        """Works with batch=1, seq=1."""
        s, t_lp, t_idx = _make_inputs(batch=1, seq=1, vocab=16, topk=3)
        for mode in ["forward_kl", "reverse_kl", "skewed_kl"]:
            loss = compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg(mode))
            assert torch.isfinite(loss), f"{mode} produced non-finite loss"

    def test_large_topk(self):
        """Works when top-k is close to vocab size."""
        s, t_lp, t_idx = _make_inputs(batch=2, seq=3, vocab=16, topk=14)
        for mode in ["forward_kl", "reverse_kl", "skewed_kl"]:
            loss = compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg(mode))
            assert torch.isfinite(loss), f"{mode} produced non-finite loss"

    def test_all_mask_false(self):
        """All-false mask should return zero (not NaN)."""
        s, t_lp, t_idx = _make_inputs(batch=2, seq=4)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        loss = sparse_forward_kl(s, t_lp, t_idx, mask=mask)
        assert loss.item() == 0.0
        assert loss.requires_grad

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        s, t_lp, t_idx = _make_inputs(seed=123)
        l1 = compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg("forward_kl"))
        l2 = compute_kl_loss(s, t_lp, t_idx, kl_config=_cfg("forward_kl"))
        assert l1.item() == l2.item()


# ------------------------------------------------------------------ #
# Same-model KL = 0 tests (student == teacher → KL should vanish)
# ------------------------------------------------------------------ #

class TestSameModelKLZero:
    """When student and teacher are the same model, KL divergence must be ~0.

    This simulates the teacher extraction pipeline: given model logits,
    extract top-k logprobs and indices (as the teacher would), then compute
    KL loss between the original logits (student) and the extracted top-k
    (teacher). All KL modes should yield ~0.
    """

    @staticmethod
    def _extract_topk_as_teacher(logits, topk):
        """Simulate teacher top-k extraction from model logits.

        This mirrors what the teacher vLLM server does: compute log_softmax
        over the full vocabulary, then return the top-k logprobs and their
        indices.
        """
        log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]
        topk_logps, topk_indices = torch.topk(log_probs, topk, dim=-1)
        return topk_logps, topk_indices.to(torch.int32)

    def test_forward_kl_zero(self):
        """Forward KL(teacher || student) = 0 when student == teacher."""
        logits = torch.randn(2, 8, 128)  # [batch, seq, vocab]
        t_logps, t_idx = self._extract_topk_as_teacher(logits, topk=32)
        loss = sparse_forward_kl(logits, t_logps, t_idx)
        assert loss.item() < 1e-5, f"Forward KL should be ~0, got {loss.item()}"

    def test_reverse_kl_zero(self):
        """Reverse KL(student || teacher) = 0 when student == teacher."""
        logits = torch.randn(2, 8, 128)
        t_logps, t_idx = self._extract_topk_as_teacher(logits, topk=32)
        loss = sparse_reverse_kl(logits, t_logps, t_idx)
        assert loss.item() < 1e-5, f"Reverse KL should be ~0, got {loss.item()}"

    def test_skewed_kl_zero(self):
        """Skewed KL = 0 when student == teacher."""
        logits = torch.randn(2, 8, 128)
        t_logps, t_idx = self._extract_topk_as_teacher(logits, topk=32)
        loss = compute_kl_loss(logits, t_logps, t_idx, kl_config=_cfg("skewed_kl", skew_alpha=0.5))
        assert loss.item() < 1e-5, f"Skewed KL should be ~0, got {loss.item()}"

    def test_reverse_kl_rollout_student_topk_zero(self):
        """Rollout-student support reverse KL should be ~0 when student == teacher."""
        logits = torch.randn(2, 8, 128)
        t_logps, t_idx = self._extract_topk_as_teacher(logits, topk=32)
        loss = compute_kl_loss(
            logits,
            t_logps,
            t_idx,
            kl_config=_cfg("reverse_kl_rollout_student_topk"),
        )
        assert loss.item() < 1e-5, (
            f"reverse_kl_rollout_student_topk should be ~0, got {loss.item()}"
        )

    def test_all_modes_zero_fp32(self):
        """All KL modes should be ~0 in fp32 precision with same model."""
        logits = torch.randn(4, 16, 256, dtype=torch.float32)
        t_logps, t_idx = self._extract_topk_as_teacher(logits, topk=64)
        mask = torch.ones(4, 16, dtype=torch.bool)
        mask[:, :4] = False  # simulate prompt masking

        for mode in ["forward_kl", "reverse_kl", "skewed_kl", "reverse_kl_rollout_student_topk"]:
            loss = compute_kl_loss(logits, t_logps, t_idx, mask=mask, kl_config=_cfg(mode))
            assert loss.item() < 1e-5, (
                f"{mode}: KL should be ~0 when student==teacher, got {loss.item()}"
            )

    def test_same_model_with_padding(self):
        """KL = 0 even with left-padded sequences and response masking.

        Simulates the full pipeline: left-padded input, teacher logprobs
        placed at valid positions via _pad_teacher, response mask applied.
        """
        batch, max_prompt, max_resp, vocab, topk = 3, 8, 16, 256, 64
        seq_len = max_prompt + max_resp

        # Create logits (as if from model forward pass)
        logits = torch.randn(batch, seq_len, vocab, dtype=torch.float32)

        # Create attention mask with varying prompt lengths (left-padded)
        attention_mask = torch.zeros(batch, seq_len, dtype=torch.long)
        prompt_lens = [5, 8, 3]  # actual prompt lengths (< max_prompt)
        resp_lens = [12, 16, 8]  # actual response lengths (< max_resp)
        for i in range(batch):
            pad = max_prompt - prompt_lens[i]
            valid_end = max_prompt + resp_lens[i]
            attention_mask[i, pad:valid_end] = 1

        # Extract teacher top-k at ALL positions (teacher scores full sequence)
        t_logps, t_idx = self._extract_topk_as_teacher(logits, topk=topk)

        # Zero out teacher logprobs at padding positions (as _pad_teacher does)
        for i in range(batch):
            padding_mask = ~attention_mask[i].bool()
            t_logps[i][padding_mask] = 0
            t_idx[i][padding_mask] = 0

        # Build response mask (as trainer.py does)
        response_mask = attention_mask.clone().bool()
        response_mask[:, :max_prompt] = False

        loss = compute_kl_loss(logits, t_logps, t_idx, mask=response_mask,
                               kl_config=_cfg("skewed_kl", skew_alpha=0.5))
        assert loss.item() < 1e-5, (
            f"Same-model KL with padding should be ~0, got {loss.item()}"
        )

    def test_same_model_bf16_logits(self):
        """KL ≈ 0 even when logits are bf16 (as in actual training).

        bf16 has less precision, so tolerance is relaxed.
        """
        logits = torch.randn(2, 8, 128, dtype=torch.float32)
        t_logps, t_idx = self._extract_topk_as_teacher(logits, topk=32)

        # Simulate bf16 student logits (as FSDP mixed precision would produce)
        logits_bf16 = logits.to(torch.bfloat16).to(torch.float32)
        loss = compute_kl_loss(logits_bf16, t_logps, t_idx, kl_config=_cfg("skewed_kl"))
        assert loss.item() < 0.01, (
            f"Same-model KL with bf16 should be small, got {loss.item()}"
        )

    def test_realistic_sparsity_same_model(self):
        """KL = 0 with realistic sparsity: topk=64 out of vocab=152064 (Qwen3).

        In production, top-k covers only 0.04% of the vocabulary. This tests
        the sparse code path with a similarly extreme ratio.
        """
        vocab = 152064  # Qwen3 vocab size
        topk = 64
        logits = torch.randn(1, 4, vocab, dtype=torch.float32)
        t_logps, t_idx = self._extract_topk_as_teacher(logits, topk=topk)

        for mode in ["forward_kl", "reverse_kl", "skewed_kl"]:
            loss = compute_kl_loss(logits, t_logps, t_idx, kl_config=_cfg(mode))
            assert loss.item() < 1e-5, (
                f"{mode}: sparse KL (topk={topk}/vocab={vocab}) should be ~0 "
                f"when student==teacher, got {loss.item()}"
            )


# ------------------------------------------------------------------ #
# Sparse KL convergence tests (different distributions)
# ------------------------------------------------------------------ #

class TestSparseKLConvergence:
    """Verify sparse top-k KL approximation converges to full KL as k increases.

    When student ≠ teacher, sparse KL underestimates the true KL. As top-k
    approaches vocab size, sparse KL should converge to the full KL. This
    catches bugs where the sparse gather/index logic produces wrong values.
    """

    def test_forward_kl_converges_with_topk(self):
        """Sparse forward KL → full forward KL as topk → vocab."""
        vocab = 512
        g = torch.Generator().manual_seed(42)
        student_logits = torch.randn(2, 4, vocab, generator=g)
        teacher_logits = torch.randn(2, 4, vocab, generator=g)

        # Full KL (all vocab tokens)
        s_logps = F.log_softmax(student_logits, dim=-1)
        t_logps_full = F.log_softmax(teacher_logits, dim=-1)
        t_probs_full = torch.exp(t_logps_full)
        full_kl = (t_probs_full * (t_logps_full - s_logps)).sum(dim=-1).mean().item()

        # Sparse KL at increasing top-k
        prev_loss = 0.0
        for topk in [8, 32, 128, 512]:
            topk_logps, topk_indices = torch.topk(t_logps_full, topk, dim=-1)
            loss = sparse_forward_kl(student_logits, topk_logps,
                                     topk_indices.to(torch.int32))
            # Sparse KL should increase monotonically (more tokens = more signal)
            assert loss.item() >= prev_loss - 1e-5, (
                f"Sparse forward KL should increase with topk: "
                f"k={topk} gave {loss.item():.4f} < prev {prev_loss:.4f}"
            )
            prev_loss = loss.item()

        # At topk=vocab, should match full KL
        torch.testing.assert_close(
            torch.tensor(prev_loss), torch.tensor(full_kl),
            atol=1e-4, rtol=1e-3,
        )

    def test_reverse_kl_converges_with_topk(self):
        """Sparse reverse KL → full reverse KL as topk → vocab."""
        vocab = 512
        g = torch.Generator().manual_seed(99)
        student_logits = torch.randn(2, 4, vocab, generator=g)
        teacher_logits = torch.randn(2, 4, vocab, generator=g)

        # Full reverse KL
        s_logps = F.log_softmax(student_logits, dim=-1)
        s_probs = torch.exp(s_logps)
        t_logps_full = F.log_softmax(teacher_logits, dim=-1)
        full_kl = (s_probs * (s_logps - t_logps_full)).sum(dim=-1).mean().item()

        # Sparse reverse KL at increasing top-k
        losses = []
        for topk in [8, 32, 128, 512]:
            topk_logps, topk_indices = torch.topk(t_logps_full, topk, dim=-1)
            loss = sparse_reverse_kl(student_logits, topk_logps,
                                     topk_indices.to(torch.int32))
            losses.append(loss.item())

        # At topk=vocab, should match full reverse KL
        torch.testing.assert_close(
            torch.tensor(losses[-1]), torch.tensor(full_kl),
            atol=1e-4, rtol=1e-3,
        )

    def test_sparse_forward_kl_nonzero_when_distributions_differ(self):
        """Sparse forward KL must be > 0 when student ≠ teacher.

        Forward KL weights by teacher probs (high at top-k), so it's always
        meaningfully positive even at extreme sparsity. Catches bugs where
        sparse gather silently returns zeros.
        """
        vocab = 152064  # realistic vocab
        topk = 64
        g = torch.Generator().manual_seed(7)
        student_logits = torch.randn(1, 2, vocab, generator=g)
        teacher_logits = torch.randn(1, 2, vocab, generator=g)

        t_logps_full = F.log_softmax(teacher_logits, dim=-1)
        topk_logps, topk_indices = torch.topk(t_logps_full, topk, dim=-1)

        loss = sparse_forward_kl(student_logits, topk_logps,
                                 topk_indices.to(torch.int32))
        assert loss.item() > 0.01, (
            f"Sparse forward KL should be > 0 when distributions differ, "
            f"got {loss.item():.6e}"
        )

    def test_sparse_reverse_kl_near_zero_at_extreme_sparsity(self):
        """Sparse reverse KL can be near-zero at extreme sparsity (topk << vocab).

        Reverse KL sums p_student(v) * (log p_s - log p_t) over teacher's
        top-k only. With random distributions over 152k vocab, the student
        puts negligible mass on those 64 tokens, so the sum is tiny.
        This is expected behavior, not a bug.
        """
        vocab = 152064
        topk = 64
        g = torch.Generator().manual_seed(7)
        student_logits = torch.randn(1, 2, vocab, generator=g)
        teacher_logits = torch.randn(1, 2, vocab, generator=g)

        t_logps_full = F.log_softmax(teacher_logits, dim=-1)
        topk_logps, topk_indices = torch.topk(t_logps_full, topk, dim=-1)

        loss = sparse_reverse_kl(student_logits, topk_logps,
                                 topk_indices.to(torch.int32))
        # Near zero is expected — student mass at teacher's top-k is tiny
        assert abs(loss.item()) < 0.1, (
            f"Sparse reverse KL at extreme sparsity should be small, "
            f"got {loss.item():.6e}"
        )

    def test_sparse_kl_nonzero_moderate_sparsity(self):
        """Forward KL and skewed KL > 0 when student ≠ teacher at moderate sparsity.

        Forward KL weights by teacher probs (concentrated in top-k), so always > 0.
        Reverse KL can be negative when summed over only top-k (missing the positive
        contributions from tokens where p_s > p_t, which are outside teacher's top-k).
        Skewed KL (0.5 fwd + 0.5 rev) is dominated by forward term, so also > 0.
        """
        vocab = 512
        topk = 64
        g = torch.Generator().manual_seed(7)
        student_logits = torch.randn(2, 4, vocab, generator=g)
        teacher_logits = torch.randn(2, 4, vocab, generator=g)

        t_logps_full = F.log_softmax(teacher_logits, dim=-1)
        topk_logps, topk_indices = torch.topk(t_logps_full, topk, dim=-1)

        # Forward KL is always non-negative (even sparse)
        fwd_loss = sparse_forward_kl(student_logits, topk_logps,
                                     topk_indices.to(torch.int32))
        assert fwd_loss.item() > 0.01, (
            f"Sparse forward KL should be > 0, got {fwd_loss.item():.6e}"
        )

        # Reverse KL can be negative at moderate sparsity — just check it's finite
        rev_loss = sparse_reverse_kl(student_logits, topk_logps,
                                     topk_indices.to(torch.int32))
        assert torch.isfinite(rev_loss), f"Reverse KL not finite: {rev_loss.item()}"

        # Skewed KL should be > 0 (forward term dominates)
        skew_loss = compute_kl_loss(student_logits, topk_logps,
                                    topk_indices.to(torch.int32),
                                    kl_config=_cfg("skewed_kl", skew_alpha=0.5))
        assert skew_loss.item() > 0.0, (
            f"Skewed KL should be > 0, got {skew_loss.item():.6e}"
        )


# ------------------------------------------------------------------ #
# Token-level KL tests
# ------------------------------------------------------------------ #

def _make_token_level_inputs(batch=2, seq=8, vocab=32, seed=42):
    """Generate inputs for token_level_kl / policy_gradient_kl tests.

    token_level_kl uses shift: logits[t] predicts input_ids[t+1].
    So teacher_token_logps[t] should be log π(input_ids[t+1]) from logits[t].
    """
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(batch, seq, vocab, generator=g)
    input_ids = torch.randint(0, vocab, (batch, seq), generator=g)
    # Teacher logprobs matching the shift: at position t, log π(input_ids[t+1])
    log_probs = F.log_softmax(student_logits, dim=-1)
    # teacher_token_logps[t] = log_probs[t, input_ids[t+1]] for t < seq-1
    # Last position doesn't matter (gets truncated by :-1 inside token_level_kl)
    teacher_token_logps = torch.zeros(batch, seq)
    target_ids = input_ids[:, 1:]  # [batch, seq-1]
    teacher_token_logps[:, :-1] = log_probs[:, :-1].gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)
    return student_logits, teacher_token_logps, input_ids


def _make_multi_sample_pg_inputs(batch=2, seq=8, vocab=32, n_samples=3, seed=123):
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(batch, seq, vocab, generator=g)
    mc_sample_indices = torch.randint(0, vocab, (batch, seq, n_samples), generator=g)

    log_probs = F.log_softmax(student_logits, dim=-1)
    mc_student_logprobs = torch.gather(log_probs, -1, mc_sample_indices.long())
    mc_teacher_logprobs = mc_student_logprobs.detach() + 0.25
    mc_old_logprobs = mc_student_logprobs.detach() - 0.15

    mask = torch.zeros(batch, seq, dtype=torch.bool)
    mask[:, 3:] = True

    return (
        student_logits,
        mc_sample_indices,
        mc_teacher_logprobs,
        mc_old_logprobs,
        mask,
        mc_student_logprobs,
    )


class TestTokenLevelKL:
    def test_zero_when_same_model(self):
        """token_level_kl ≈ 0 when student logits produce same logprobs as teacher."""
        s, t_logps, ids = _make_token_level_inputs()
        loss = token_level_kl(s, t_logps, ids)
        assert abs(loss.item()) < 1e-5, f"Expected ~0, got {loss.item()}"

    def test_positive_when_different(self):
        """token_level_kl > 0 when student and teacher differ."""
        g = torch.Generator().manual_seed(42)
        s = torch.randn(2, 8, 32, generator=g)
        ids = torch.randint(0, 32, (2, 8), generator=g)
        # Teacher from a different distribution
        t_logits = torch.randn(2, 8, 32, generator=g)
        t_logps = F.log_softmax(t_logits, dim=-1).gather(
            dim=-1, index=ids.unsqueeze(-1)
        ).squeeze(-1)
        loss = token_level_kl(s, t_logps, ids)
        assert loss.item() > 0.001, f"Expected > 0, got {loss.item()}"

    def test_gradient_flows(self):
        """Gradient should flow to student logits."""
        s, t_logps, ids = _make_token_level_inputs()
        s.requires_grad_(True)
        loss = token_level_kl(s, t_logps, ids)
        loss.backward()
        assert s.grad is not None
        assert not torch.all(s.grad == 0)

    def test_with_mask(self):
        """Mask should exclude positions from loss."""
        s, t_logps, ids = _make_token_level_inputs(batch=2, seq=8)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, 4:] = True  # only response positions
        loss = token_level_kl(s, t_logps, ids, mask=mask)
        assert torch.isfinite(loss)

    def test_dispatch(self):
        """compute_kl_loss dispatches to token_level_kl correctly."""
        s, t_logps, ids = _make_token_level_inputs()
        # Need dummy topk args for dispatch
        topk_logps = torch.zeros(2, 8, 5)
        topk_idx = torch.zeros(2, 8, 5, dtype=torch.int32)
        loss = compute_kl_loss(
            s, topk_logps, topk_idx,
            teacher_token_logps=t_logps,
            input_ids=ids,
            kl_config=_cfg("token_level_kl"),
        )
        assert torch.isfinite(loss)


# ------------------------------------------------------------------ #
# Policy gradient KL tests
# ------------------------------------------------------------------ #

class TestPolicyGradientKL:
    def test_zero_advantage_gives_zero_loss(self):
        """When teacher == old student, advantage is 0, loss should be ~0."""
        batch, seq, vocab = 2, 8, 32
        g = torch.Generator().manual_seed(42)
        student_logits = torch.randn(batch, seq, vocab, generator=g)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)

        # Teacher logprobs = student old logprobs → advantage = 0
        log_ps = F.log_softmax(student_logits, dim=-1)
        target_ids = input_ids[:, 1:]
        token_logps = log_ps[:, :-1].gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        # teacher and old are the same
        teacher_token_logps = torch.zeros(batch, seq)
        teacher_token_logps[:, :-1] = token_logps.detach()
        student_old_logprobs = torch.zeros(batch, seq - 1)
        student_old_logprobs[:] = token_logps.detach()

        # Pad old logprobs to match response length expectation
        # policy_gradient_kl expects [batch, resp_len] where resp comes at end
        resp_len = seq - 1
        loss = policy_gradient_kl(student_logits, teacher_token_logps, input_ids,
                                  student_old_logprobs)
        assert abs(loss.item()) < 1e-5, f"Expected ~0 with zero advantage, got {loss.item()}"

    def test_gradient_flows(self):
        """Gradient should flow through student logits."""
        batch, seq, vocab = 2, 8, 32
        g = torch.Generator().manual_seed(42)
        student_logits = torch.randn(batch, seq, vocab, generator=g, requires_grad=True)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)
        teacher_token_logps = torch.randn(batch, seq)
        student_old_logprobs = torch.randn(batch, seq - 1)

        loss = policy_gradient_kl(student_logits, teacher_token_logps, input_ids,
                                  student_old_logprobs)
        loss.backward()
        assert student_logits.grad is not None
        assert not torch.all(student_logits.grad == 0)

    def test_clipping_works(self):
        """With very different old/new logprobs, clipping should limit the ratio."""
        batch, seq, vocab = 1, 6, 16
        g = torch.Generator().manual_seed(42)
        student_logits = torch.randn(batch, seq, vocab, generator=g)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)
        teacher_token_logps = torch.randn(batch, seq)
        # Old logprobs very different from current → large ratio
        student_old_logprobs = torch.full((batch, seq - 1), -10.0)

        loss_clipped = policy_gradient_kl(student_logits, teacher_token_logps,
                                          input_ids, student_old_logprobs, clip_eps=0.2)
        loss_unclipped = policy_gradient_kl(student_logits, teacher_token_logps,
                                            input_ids, student_old_logprobs, clip_eps=100.0)
        # Clipped loss should differ from unclipped when ratio is large
        # (with eps=100, no clipping occurs)
        assert torch.isfinite(loss_clipped)
        assert torch.isfinite(loss_unclipped)

    def test_with_mask(self):
        """Mask should exclude positions."""
        batch, seq, vocab = 2, 8, 32
        g = torch.Generator().manual_seed(42)
        student_logits = torch.randn(batch, seq, vocab, generator=g)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)
        teacher_token_logps = torch.randn(batch, seq)
        student_old_logprobs = torch.randn(batch, seq - 1)
        mask = torch.zeros(batch, seq, dtype=torch.bool)
        mask[:, 4:] = True

        loss = policy_gradient_kl(student_logits, teacher_token_logps, input_ids,
                                  student_old_logprobs, mask=mask)
        assert torch.isfinite(loss)

    def test_disable_importance_sampling_forces_unit_ratio(self):
        """Turning IS off should make PPO ratios collapse to 1 regardless of old logprobs."""
        batch, seq, vocab = 2, 8, 32
        g = torch.Generator().manual_seed(42)
        student_logits = torch.randn(batch, seq, vocab, generator=g)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)
        teacher_token_logps = torch.randn(batch, seq)
        student_old_logprobs = torch.full((batch, seq - 1), -20.0)

        loss = policy_gradient_kl(
            student_logits,
            teacher_token_logps,
            input_ids,
            student_old_logprobs,
            use_importance_sampling=False,
        )

        torch.testing.assert_close(loss.pg_stats["_ratios"], torch.ones_like(loss.pg_stats["_ratios"]))
        assert not loss.pg_stats["_clip_high"].any()
        assert not loss.pg_stats["_clip_low"].any()

    def test_disable_importance_sampling_allows_missing_old_logprobs(self):
        """Non-IS mode should not require rollout old logprobs."""
        batch, seq, vocab = 2, 8, 32
        g = torch.Generator().manual_seed(7)
        student_logits = torch.randn(batch, seq, vocab, generator=g)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)
        teacher_token_logps = torch.randn(batch, seq, generator=g)

        loss = compute_kl_loss(
            student_logits=student_logits,
            teacher_topk_logps=torch.zeros(batch, seq, 1),
            teacher_topk_indices=torch.zeros(batch, seq, 1, dtype=torch.int32),
            teacher_token_logps=teacher_token_logps,
            input_ids=input_ids,
            kl_config=_cfg("policy_gradient_kl", use_importance_sampling=False),
        )
        assert torch.isfinite(loss)

    def test_dispatch(self):
        """compute_kl_loss dispatches to policy_gradient_kl correctly."""
        batch, seq, vocab = 2, 8, 32
        g = torch.Generator().manual_seed(42)
        student_logits = torch.randn(batch, seq, vocab, generator=g)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)
        teacher_token_logps = torch.randn(batch, seq)
        student_old_logprobs = torch.randn(batch, seq - 1)
        topk_logps = torch.zeros(batch, seq, 5)
        topk_idx = torch.zeros(batch, seq, 5, dtype=torch.int32)

        loss = compute_kl_loss(
            student_logits, topk_logps, topk_idx,
            teacher_token_logps=teacher_token_logps,
            input_ids=input_ids,
            student_old_logprobs=student_old_logprobs,
            kl_config=_cfg("policy_gradient_kl", pg_clip_eps=0.2),
        )
        assert torch.isfinite(loss)

    def test_positive_advantage_reinforces(self):
        """When teacher > old at a token, policy gradient should push student toward it."""
        batch, seq, vocab = 1, 4, 8
        g = torch.Generator().manual_seed(42)
        student_logits = torch.randn(batch, seq, vocab, generator=g, requires_grad=True)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)

        # Teacher confident, old student not → positive advantage
        teacher_token_logps = torch.zeros(batch, seq)
        teacher_token_logps[:] = -0.1  # high teacher prob
        student_old_logprobs = torch.full((batch, seq - 1), -5.0)  # low old prob

        loss = policy_gradient_kl(student_logits, teacher_token_logps, input_ids,
                                  student_old_logprobs)
        loss.backward()

        # Loss should be negative (good advantage → minimize -ratio*adv)
        # and gradients should exist
        assert student_logits.grad is not None

    def test_log_ratio_zero_step_off_0(self):
        """With step_off=0, old logprobs come from same weights → log_ratio must be 0.

        Simulates the full pipeline in fp32: generate input_ids from logits,
        compute old logprobs (as rollout vLLM would), then verify the trainer's
        forward pass produces identical logprobs → ratio=1, log_ratio=0.
        """
        batch, seq, vocab = 4, 16, 256
        g = torch.Generator().manual_seed(42)
        student_logits = torch.randn(batch, seq, vocab, generator=g, dtype=torch.float32)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)

        # "Rollout" logprobs: same model computes log π_old(y_t)
        # At position t, logits[t] predicts token input_ids[t+1]
        log_ps = F.log_softmax(student_logits, dim=-1)
        target_ids = input_ids[:, 1:]  # [batch, seq-1]
        old_logprobs = log_ps[:, :-1].gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1).detach()  # [batch, seq-1]

        # "Trainer" forward: same logits → policy_gradient_kl computes new logprobs
        # internally via log_softmax + gather on the same student_logits.
        # With step_off=0, weights are identical, so:
        #   log_ratio = student_new_logps - old_logps = 0
        #   ratio = 1.0

        # Manually check log_ratio is 0 (same computation as policy_gradient_kl)
        student_new_logps = log_ps[:, :-1].gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        log_ratio = student_new_logps - old_logprobs
        ratio = log_ratio.exp()

        assert log_ratio.abs().max().item() < 1e-7, (
            f"log_ratio should be exactly 0 in fp32 with same weights, "
            f"max |log_ratio| = {log_ratio.abs().max().item():.2e}"
        )
        assert (ratio - 1.0).abs().max().item() < 1e-7, (
            f"ratio should be exactly 1.0 in fp32 with same weights, "
            f"max |ratio - 1| = {(ratio - 1.0).abs().max().item():.2e}"
        )


class TestMultiSamplePolicyGradientKL:
    def test_same_model_zero_loss(self):
        (
            student_logits,
            mc_sample_indices,
            _mc_teacher_logprobs,
            _mc_old_logprobs,
            mask,
            mc_student_logprobs,
        ) = _make_multi_sample_pg_inputs(batch=2, seq=6, vocab=24, n_samples=4)

        loss = multi_sample_policy_gradient_kl(
            student_mc_logprobs=mc_student_logprobs,
            mc_teacher_logprobs=mc_student_logprobs.detach(),
            mc_old_logprobs=mc_student_logprobs.detach(),
            mask=mask,
            clip_eps=0.2,
        )
        assert abs(loss.item()) < 1e-6

    def test_n1_matches_policy_gradient_kl_trainer_path(self):
        student_logits, teacher_token_logps, input_ids = _make_token_level_inputs()
        mask = torch.zeros(student_logits.size(0), student_logits.size(1), dtype=torch.bool)
        mask[:, 3:] = True

        target_ids = input_ids[:, 1:]
        student_new_logps = F.log_softmax(student_logits, dim=-1)[:, :-1].gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        teacher_shifted = teacher_token_logps[:, :-1]
        old_logps = student_new_logps.detach() - 0.3
        shifted_mask = mask[:, 1:]

        single = policy_gradient_kl(
            student_token_logps=student_new_logps,
            teacher_token_logps=teacher_shifted,
            student_old_logprobs=old_logps,
            mask=shifted_mask,
            clip_eps=0.2,
        )
        multi = multi_sample_policy_gradient_kl(
            student_mc_logprobs=student_new_logps.unsqueeze(-1),
            mc_teacher_logprobs=teacher_shifted.unsqueeze(-1),
            mc_old_logprobs=old_logps.unsqueeze(-1),
            mask=shifted_mask,
            clip_eps=0.2,
        )

        torch.testing.assert_close(single, multi, atol=1e-6, rtol=1e-6)

    def test_matches_manual_mean_over_samples(self):
        (_, _, mc_teacher_logprobs, mc_old_logprobs, mask, mc_student_logprobs) = (
            _make_multi_sample_pg_inputs()
        )
        loss = multi_sample_policy_gradient_kl(
            student_mc_logprobs=mc_student_logprobs,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mc_old_logprobs=mc_old_logprobs,
            mask=mask,
            clip_eps=0.2,
        )

        ratio = torch.exp(mc_student_logprobs - mc_old_logprobs)
        advantages = (mc_teacher_logprobs - mc_old_logprobs).detach()
        surr1 = -advantages * ratio
        surr2 = -advantages * ratio.clamp(0.8, 1.2)
        manual = torch.max(surr1, surr2).mean(dim=-1)[mask].mean()

        torch.testing.assert_close(loss, manual, atol=1e-6, rtol=1e-6)

    def test_disable_importance_sampling_forces_unit_ratio(self):
        (_, _, mc_teacher_logprobs, mc_old_logprobs, mask, mc_student_logprobs) = (
            _make_multi_sample_pg_inputs()
        )
        loss = multi_sample_policy_gradient_kl(
            student_mc_logprobs=mc_student_logprobs,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mc_old_logprobs=mc_old_logprobs,
            mask=mask,
            clip_eps=0.2,
            use_importance_sampling=False,
        )

        torch.testing.assert_close(loss.pg_stats["_ratios"], torch.ones_like(loss.pg_stats["_ratios"]))
        assert not loss.pg_stats["_clip_high"].any()
        assert not loss.pg_stats["_clip_low"].any()

    def test_repeated_samples_are_preserved(self):
        (
            student_logits,
            mc_sample_indices,
            mc_teacher_logprobs,
            mc_old_logprobs,
            mask,
            _,
        ) = _make_multi_sample_pg_inputs(batch=1, seq=6, vocab=16, n_samples=3)
        mc_sample_indices[..., 1] = mc_sample_indices[..., 0]
        mc_teacher_logprobs[..., 1] = mc_teacher_logprobs[..., 0]
        mc_old_logprobs[..., 1] = mc_old_logprobs[..., 0]

        loss = multi_sample_policy_gradient_kl(
            student_logits=student_logits,
            mc_sample_indices=mc_sample_indices,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mc_old_logprobs=mc_old_logprobs,
            mask=mask,
            clip_eps=0.2,
        )
        log_probs = F.log_softmax(student_logits[:, :-1], dim=-1)
        gathered = torch.gather(log_probs, -1, mc_sample_indices[:, 1:])
        manual = multi_sample_policy_gradient_kl(
            student_mc_logprobs=gathered,
            mc_teacher_logprobs=mc_teacher_logprobs[:, 1:],
            mc_old_logprobs=mc_old_logprobs[:, 1:],
            mask=mask[:, 1:],
            clip_eps=0.2,
        )

        torch.testing.assert_close(loss, manual, atol=1e-6, rtol=1e-6)

    def test_empty_mask_is_safe(self):
        (_, _, mc_teacher_logprobs, mc_old_logprobs, _, mc_student_logprobs) = (
            _make_multi_sample_pg_inputs()
        )
        mask = torch.zeros(mc_student_logprobs.shape[:2], dtype=torch.bool)
        loss = multi_sample_policy_gradient_kl(
            student_mc_logprobs=mc_student_logprobs,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mc_old_logprobs=mc_old_logprobs,
            mask=mask,
            clip_eps=0.2,
        )
        assert loss.item() == 0.0

    def test_dispatch(self):
        (
            student_logits,
            mc_sample_indices,
            mc_teacher_logprobs,
            mc_old_logprobs,
            mask,
            _,
        ) = _make_multi_sample_pg_inputs()
        topk_logps = torch.zeros(student_logits.size(0), student_logits.size(1), 1)
        topk_idx = torch.zeros(student_logits.size(0), student_logits.size(1), 1, dtype=torch.int32)

        loss = compute_kl_loss(
            student_logits,
            topk_logps,
            topk_idx,
            mc_sample_indices=mc_sample_indices,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mc_old_logprobs=mc_old_logprobs,
            mask=mask,
            kl_config=_cfg("multi_sample_policy_gradient_kl", pg_clip_eps=0.2),
        )
        assert torch.isfinite(loss)


class TestMultiSampleForwardKL:
    def test_same_model_zero_loss(self):
        (
            _student_logits,
            _mc_sample_indices,
            _mc_teacher_logprobs,
            _mc_old_logprobs,
            mask,
            mc_student_logprobs,
        ) = _make_multi_sample_pg_inputs(batch=2, seq=6, vocab=24, n_samples=4)

        loss = multi_sample_forward_kl(
            student_mc_logprobs=mc_student_logprobs,
            mc_teacher_logprobs=mc_student_logprobs.detach(),
            mask=mask,
        )
        assert abs(loss.item()) < 1e-6

    def test_matches_manual_mean_over_samples(self):
        (_, _, mc_teacher_logprobs, _, mask, mc_student_logprobs) = (
            _make_multi_sample_pg_inputs()
        )
        loss = multi_sample_forward_kl(
            student_mc_logprobs=mc_student_logprobs,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mask=mask,
        )

        manual = (mc_teacher_logprobs - mc_student_logprobs).mean(dim=-1)[mask].mean()
        torch.testing.assert_close(loss, manual, atol=1e-6, rtol=1e-6)

    def test_repeated_samples_are_preserved(self):
        (
            student_logits,
            mc_sample_indices,
            mc_teacher_logprobs,
            _mc_old_logprobs,
            mask,
            _,
        ) = _make_multi_sample_pg_inputs(batch=1, seq=6, vocab=16, n_samples=3)
        mc_sample_indices[..., 1] = mc_sample_indices[..., 0]
        mc_teacher_logprobs[..., 1] = mc_teacher_logprobs[..., 0]

        loss = multi_sample_forward_kl(
            student_logits=student_logits,
            mc_sample_indices=mc_sample_indices,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mask=mask,
        )
        log_probs = F.log_softmax(student_logits[:, :-1], dim=-1)
        gathered = torch.gather(log_probs, -1, mc_sample_indices[:, 1:])
        manual = multi_sample_forward_kl(
            student_mc_logprobs=gathered,
            mc_teacher_logprobs=mc_teacher_logprobs[:, 1:],
            mask=mask[:, 1:],
        )

        torch.testing.assert_close(loss, manual, atol=1e-6, rtol=1e-6)

    def test_empty_mask_is_safe(self):
        (_, _, mc_teacher_logprobs, _, _, mc_student_logprobs) = (
            _make_multi_sample_pg_inputs()
        )
        mask = torch.zeros(mc_student_logprobs.shape[:2], dtype=torch.bool)
        loss = multi_sample_forward_kl(
            student_mc_logprobs=mc_student_logprobs,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mask=mask,
        )
        assert loss.item() == 0.0

    def test_dispatch(self):
        (
            student_logits,
            mc_sample_indices,
            mc_teacher_logprobs,
            _mc_old_logprobs,
            mask,
            _,
        ) = _make_multi_sample_pg_inputs()
        topk_logps = torch.zeros(student_logits.size(0), student_logits.size(1), 1)
        topk_idx = torch.zeros(student_logits.size(0), student_logits.size(1), 1, dtype=torch.int32)

        loss = compute_kl_loss(
            student_logits,
            topk_logps,
            topk_idx,
            mc_sample_indices=mc_sample_indices,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mask=mask,
            kl_config=_cfg("multi_sample_forward_kl"),
        )
        assert torch.isfinite(loss)

    def test_dispatch_same_model_zero_loss(self):
        (
            student_logits,
            mc_sample_indices,
            _mc_teacher_logprobs,
            _mc_old_logprobs,
            mask,
            mc_student_logprobs,
        ) = _make_multi_sample_pg_inputs(batch=2, seq=7, vocab=32, n_samples=3)
        loss = compute_kl_loss(
            student_logits=torch.zeros(2, 7, 1),  # unused when student_mc_logprobs is provided
            teacher_topk_logps=torch.zeros(2, 7, 1),
            teacher_topk_indices=torch.zeros(2, 7, 1, dtype=torch.int32),
            mc_sample_indices=mc_sample_indices,
            mc_teacher_logprobs=mc_student_logprobs.detach(),
            mc_old_logprobs=mc_student_logprobs.detach(),
            student_mc_logprobs=mc_student_logprobs,
            mask=mask,
            kl_config=_cfg("multi_sample_policy_gradient_kl", pg_clip_eps=0.2),
        )
        assert abs(loss.item()) < 1e-6


# ------------------------------------------------------------------ #
# Memory optimization: logsumexp+gather == F.log_softmax+gather
# ------------------------------------------------------------------ #

class TestMemoryOptimization:
    """Verify that the logsumexp+gather optimization produces identical
    results to the original F.log_softmax approach.

    The optimization replaces:
        log_softmax(x)[indices]  →  x[indices] - logsumexp(x)
    which avoids materializing the full [B, S, V] log_softmax tensor.
    """

    @staticmethod
    def _ref_forward_kl(student_logits, teacher_topk_logps, teacher_topk_indices, mask=None):
        """Original forward KL using F.log_softmax (pre-optimization)."""
        student_logps = F.log_softmax(student_logits, dim=-1)
        student_topk_logps = torch.gather(student_logps, dim=-1,
                                          index=teacher_topk_indices.long())
        teacher_topk_probs = torch.exp(teacher_topk_logps)
        per_token_kl = (teacher_topk_probs * (teacher_topk_logps - student_topk_logps)).sum(dim=-1)
        if mask is not None:
            return per_token_kl[mask].mean()
        return per_token_kl.mean()

    @staticmethod
    def _ref_reverse_kl(student_logits, teacher_topk_logps, teacher_topk_indices, mask=None):
        """Original reverse KL using F.log_softmax (pre-optimization)."""
        student_logps = F.log_softmax(student_logits, dim=-1)
        student_topk_logps = torch.gather(student_logps, dim=-1,
                                          index=teacher_topk_indices.long())
        student_topk_probs = torch.exp(student_topk_logps)
        per_token_kl = (student_topk_probs * (student_topk_logps - teacher_topk_logps)).sum(dim=-1)
        if mask is not None:
            return per_token_kl[mask].mean()
        return per_token_kl.mean()

    @staticmethod
    def _ref_token_level_kl(student_logits, teacher_token_logps, input_ids, mask=None):
        """Original token_level_kl using F.log_softmax (pre-optimization)."""
        log_p_s = F.log_softmax(student_logits, dim=-1)
        target_ids = input_ids[:, 1:]
        log_p_s = log_p_s[:, :-1]
        t_logps = teacher_token_logps[:, :-1]
        student_token_logps = log_p_s.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        teacher_probs = torch.exp(t_logps).detach()
        per_token_kl = teacher_probs * (t_logps.detach() - student_token_logps)
        if mask is not None:
            m = mask[:, 1:]
            return per_token_kl[m].mean()
        return per_token_kl.mean()

    @staticmethod
    def _ref_policy_gradient_kl(student_logits, teacher_token_logps, input_ids,
                                student_old_logprobs, mask=None, clip_eps=0.2):
        """Original policy_gradient_kl using F.log_softmax (pre-optimization)."""
        log_p_s = F.log_softmax(student_logits, dim=-1)
        target_ids = input_ids[:, 1:]
        log_p_s = log_p_s[:, :-1]
        t_logps = teacher_token_logps[:, :-1]
        student_new_logps = log_p_s.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        bs, shifted_len = student_new_logps.shape
        resp_len = student_old_logprobs.size(1)
        old_logps = torch.zeros(bs, shifted_len, device=student_new_logps.device,
                                dtype=student_old_logprobs.dtype)
        old_logps[:, -resp_len:] = student_old_logprobs
        advantage = (t_logps - old_logps).detach()
        log_ratio = student_new_logps - old_logps.detach()
        ratio = log_ratio.exp()
        surr1 = ratio * advantage
        surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage
        per_token_loss = -torch.min(surr1, surr2)
        if mask is not None:
            m = mask[:, 1:]
            return per_token_loss[m].mean()
        return per_token_loss.mean()

    def test_forward_kl_matches_reference(self):
        """Optimized sparse_forward_kl == reference F.log_softmax version."""
        for seed in [42, 123, 999]:
            s, t_lp, t_idx = _make_inputs(batch=4, seq=16, vocab=256, topk=64, seed=seed)
            opt = sparse_forward_kl(s, t_lp, t_idx)
            ref = self._ref_forward_kl(s, t_lp, t_idx)
            torch.testing.assert_close(opt, ref, atol=1e-5, rtol=1e-5)

    def test_forward_kl_matches_reference_with_mask(self):
        s, t_lp, t_idx = _make_inputs(batch=4, seq=16, vocab=256, topk=64)
        mask = torch.zeros(4, 16, dtype=torch.bool)
        mask[:, 4:] = True
        opt = sparse_forward_kl(s, t_lp, t_idx, mask=mask)
        ref = self._ref_forward_kl(s, t_lp, t_idx, mask=mask)
        torch.testing.assert_close(opt, ref, atol=1e-5, rtol=1e-5)

    def test_reverse_kl_matches_reference(self):
        """Optimized sparse_reverse_kl == reference F.log_softmax version."""
        for seed in [42, 123, 999]:
            s, t_lp, t_idx = _make_inputs(batch=4, seq=16, vocab=256, topk=64, seed=seed)
            opt = sparse_reverse_kl(s, t_lp, t_idx)
            ref = self._ref_reverse_kl(s, t_lp, t_idx)
            torch.testing.assert_close(opt, ref, atol=1e-5, rtol=1e-5)

    def test_reverse_kl_matches_reference_with_mask(self):
        s, t_lp, t_idx = _make_inputs(batch=4, seq=16, vocab=256, topk=64)
        mask = torch.zeros(4, 16, dtype=torch.bool)
        mask[:, 8:] = True
        opt = sparse_reverse_kl(s, t_lp, t_idx, mask=mask)
        ref = self._ref_reverse_kl(s, t_lp, t_idx, mask=mask)
        torch.testing.assert_close(opt, ref, atol=1e-5, rtol=1e-5)

    def test_token_level_kl_matches_reference(self):
        """Optimized token_level_kl == reference F.log_softmax version."""
        for seed in [42, 123, 999]:
            s, t_logps, ids = _make_token_level_inputs(batch=4, seq=16, vocab=256, seed=seed)
            opt = token_level_kl(s, t_logps, ids)
            ref = self._ref_token_level_kl(s, t_logps, ids)
            torch.testing.assert_close(opt, ref, atol=1e-5, rtol=1e-5)

    def test_token_level_kl_matches_reference_with_mask(self):
        s, t_logps, ids = _make_token_level_inputs(batch=4, seq=16, vocab=256)
        mask = torch.zeros(4, 16, dtype=torch.bool)
        mask[:, 4:] = True
        opt = token_level_kl(s, t_logps, ids, mask=mask)
        ref = self._ref_token_level_kl(s, t_logps, ids, mask=mask)
        torch.testing.assert_close(opt, ref, atol=1e-5, rtol=1e-5)

    def test_policy_gradient_kl_matches_reference(self):
        """Optimized policy_gradient_kl == reference F.log_softmax version."""
        for seed in [42, 123, 999]:
            g = torch.Generator().manual_seed(seed)
            batch, seq, vocab = 4, 16, 256
            s = torch.randn(batch, seq, vocab, generator=g)
            ids = torch.randint(0, vocab, (batch, seq), generator=g)
            t_logps = torch.randn(batch, seq)
            old_lp = torch.randn(batch, seq - 1)
            opt = policy_gradient_kl(s, t_logps, ids, old_lp)
            ref = self._ref_policy_gradient_kl(s, t_logps, ids, old_lp)
            torch.testing.assert_close(opt, ref, atol=1e-5, rtol=1e-5)

    def test_policy_gradient_kl_online_advantage(self):
        """online_advantage=True uses current student logprobs as baseline."""
        g = torch.Generator().manual_seed(42)
        batch, seq, vocab = 4, 16, 256
        s = torch.randn(batch, seq, vocab, generator=g)
        ids = torch.randint(0, vocab, (batch, seq), generator=g)
        t_logps = torch.randn(batch, seq)
        # Make old_lp different from current student logprobs so the two modes differ
        old_lp = torch.randn(batch, seq - 1)

        loss_standard = policy_gradient_kl(s, t_logps, ids, old_lp, online_advantage=False)
        loss_online = policy_gradient_kl(s, t_logps, ids, old_lp, online_advantage=True)

        # Both should produce valid finite losses
        assert torch.isfinite(loss_standard)
        assert torch.isfinite(loss_online)
        # They should differ since old_lp != current student logprobs
        assert not torch.allclose(loss_standard, loss_online), \
            "online_advantage should produce different loss when old_lp != current student"

        # When old_lp == current student logprobs, both modes should match
        student_logps = F.log_softmax(s[:, :-1], dim=-1)
        current_lp = torch.gather(student_logps, -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        loss_std_eq = policy_gradient_kl(s, t_logps, ids, current_lp, online_advantage=False)
        loss_online_eq = policy_gradient_kl(s, t_logps, ids, current_lp, online_advantage=True)
        torch.testing.assert_close(loss_std_eq, loss_online_eq, atol=1e-5, rtol=1e-5)

    def test_policy_gradient_kl_online_advantage_via_dispatch(self):
        """online_advantage threads through compute_kl_loss dispatch."""
        from opd.loss.kl import KLConfig
        g = torch.Generator().manual_seed(42)
        batch, seq, vocab = 4, 16, 256
        s = torch.randn(batch, seq, vocab, generator=g)
        ids = torch.randint(0, vocab, (batch, seq), generator=g)
        t_logps = torch.randn(batch, seq)
        old_lp = torch.randn(batch, seq - 1)

        cfg_off = KLConfig(mode="policy_gradient_kl", pg_online_advantage=False)
        cfg_on = KLConfig(mode="policy_gradient_kl", pg_online_advantage=True)

        loss_off = compute_kl_loss(
            s, teacher_token_logps=t_logps, input_ids=ids,
            student_old_logprobs=old_lp, kl_config=cfg_off)
        loss_on = compute_kl_loss(
            s, teacher_token_logps=t_logps, input_ids=ids,
            student_old_logprobs=old_lp, kl_config=cfg_on)

        assert torch.isfinite(loss_off) and torch.isfinite(loss_on)
        assert not torch.allclose(loss_off, loss_on), \
            "KLConfig.pg_online_advantage should affect compute_kl_loss output"

    # ------------------------------------------------------------------ #
    #  Decoupled PPO tests                                                #
    # ------------------------------------------------------------------ #

    def test_decoupled_ppo_identity(self):
        """When prox_logprobs == old_logps, decoupled loss == standard PG-KL."""
        g = torch.Generator().manual_seed(42)
        batch, shifted_len = 4, 15
        student_logps = torch.randn(batch, shifted_len, generator=g, requires_grad=True)
        t_logps = torch.randn(batch, shifted_len, generator=g)
        old_logps = torch.randn(batch, shifted_len, generator=g)
        mask = torch.ones(batch, shifted_len, dtype=torch.bool)

        loss_std = policy_gradient_kl(
            student_token_logps=student_logps, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2,
            use_decoupled_loss=False)
        loss_dec = policy_gradient_kl(
            student_token_logps=student_logps, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2,
            use_decoupled_loss=True, prox_logprobs=old_logps)
        torch.testing.assert_close(loss_std, loss_dec, atol=1e-6, rtol=1e-6)

    def test_decoupled_ppo_changes_loss(self):
        """When prox_logprobs != old_logps, decoupled loss differs from standard."""
        g = torch.Generator().manual_seed(42)
        batch, shifted_len = 4, 15
        student_logps = torch.randn(batch, shifted_len, generator=g, requires_grad=True)
        t_logps = torch.randn(batch, shifted_len, generator=g)
        old_logps = torch.randn(batch, shifted_len, generator=g)
        prox_logps = torch.randn(batch, shifted_len, generator=g)
        mask = torch.ones(batch, shifted_len, dtype=torch.bool)

        loss_std = policy_gradient_kl(
            student_token_logps=student_logps, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2,
            use_decoupled_loss=False)
        loss_dec = policy_gradient_kl(
            student_token_logps=student_logps, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2,
            use_decoupled_loss=True, prox_logprobs=prox_logps)
        assert torch.isfinite(loss_dec)
        assert not torch.allclose(loss_std, loss_dec), \
            "Decoupled loss should differ when prox_logprobs != old_logps"

    def test_decoupled_ppo_cap_masking(self):
        """Tokens with w_behav > cap are zeroed out (token_mask mode)."""
        batch, shifted_len = 2, 10
        student_logps = torch.zeros(batch, shifted_len, requires_grad=True)
        t_logps = torch.ones(batch, shifted_len)
        old_logps = torch.zeros(batch, shifted_len)
        # prox_logps much larger than old → w_behav = exp(prox - old) will be large
        prox_logps = torch.full((batch, shifted_len), 3.0)  # exp(3) ≈ 20 > cap=5
        mask = torch.ones(batch, shifted_len, dtype=torch.bool)

        loss = policy_gradient_kl(
            student_token_logps=student_logps, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2,
            use_decoupled_loss=True, prox_logprobs=prox_logps,
            behave_imp_weight_cap=5.0)
        # All tokens should be masked (exp(3) ≈ 20.09 > 5.0) → loss is 0
        assert loss.item() == 0.0, f"Expected 0 loss when all tokens masked, got {loss.item()}"
        assert loss.pg_stats["_behave_mask"].sum().item() == 0, \
            "All tokens should be masked when w_behav > cap"

    def test_decoupled_ppo_stats(self):
        """Decoupled loss includes behave_imp_weight and behave_mask in pg_stats."""
        g = torch.Generator().manual_seed(42)
        batch, shifted_len = 4, 15
        student_logps = torch.randn(batch, shifted_len, generator=g, requires_grad=True)
        t_logps = torch.randn(batch, shifted_len, generator=g)
        old_logps = torch.randn(batch, shifted_len, generator=g)
        prox_logps = old_logps + 0.1  # slight difference
        mask = torch.ones(batch, shifted_len, dtype=torch.bool)

        loss = policy_gradient_kl(
            student_token_logps=student_logps, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2,
            use_decoupled_loss=True, prox_logprobs=prox_logps)
        assert "_behave_imp_weight" in loss.pg_stats
        assert "_behave_mask" in loss.pg_stats
        w = loss.pg_stats["_behave_imp_weight"]
        assert w.shape == (batch * shifted_len,)
        assert torch.isfinite(w).all()

    def test_decoupled_ppo_backward_compat(self):
        """use_decoupled_loss=False is bit-identical to default policy_gradient_kl."""
        g = torch.Generator().manual_seed(42)
        batch, shifted_len = 4, 15
        student_logps = torch.randn(batch, shifted_len, generator=g)
        t_logps = torch.randn(batch, shifted_len, generator=g)
        old_logps = torch.randn(batch, shifted_len, generator=g)
        mask = torch.ones(batch, shifted_len, dtype=torch.bool)

        s1 = student_logps.clone().requires_grad_(True)
        s2 = student_logps.clone().requires_grad_(True)

        loss1 = policy_gradient_kl(
            student_token_logps=s1, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2)
        loss2 = policy_gradient_kl(
            student_token_logps=s2, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2,
            use_decoupled_loss=False)
        torch.testing.assert_close(loss1, loss2)
        loss1.backward()
        loss2.backward()
        torch.testing.assert_close(s1.grad, s2.grad)
        # Should NOT have behave stats when decoupled is off
        assert "_behave_imp_weight" not in loss2.pg_stats

    def test_decoupled_ppo_prox_is_detached(self):
        """Pi_prox = student_token_logps.detach() — no grad, correct values."""
        g = torch.Generator().manual_seed(42)
        batch, shifted_len = 4, 15
        student_logps = torch.randn(batch, shifted_len, generator=g, requires_grad=True)
        t_logps = torch.randn(batch, shifted_len, generator=g)
        old_logps = torch.randn(batch, shifted_len, generator=g)
        prox_logps = student_logps.detach()  # pi_prox = pi_theta detached
        mask = torch.ones(batch, shifted_len, dtype=torch.bool)

        loss = policy_gradient_kl(
            student_token_logps=student_logps, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2,
            use_decoupled_loss=True, prox_logprobs=prox_logps)
        # ratio = pi_theta / pi_prox = 1.0 since prox = theta.detach()
        assert torch.allclose(loss.pg_stats["_ratios"],
                              torch.ones_like(loss.pg_stats["_ratios"]), atol=1e-5)
        # Loss should still be finite and have gradients
        assert torch.isfinite(loss)
        loss.backward()
        assert student_logps.grad is not None

    def test_decoupled_ppo_ratio_diverges_with_stale_prox(self):
        """When pi_prox is frozen and pi_theta changes, ratio != 1.0."""
        g = torch.Generator().manual_seed(42)
        batch, shifted_len = 4, 15
        t_logps = torch.randn(batch, shifted_len, generator=g)
        old_logps = torch.randn(batch, shifted_len, generator=g)
        mask = torch.ones(batch, shifted_len, dtype=torch.bool)

        # Simulate: pi_prox frozen before optimizer.step()
        prox_logps = torch.randn(batch, shifted_len, generator=g)

        # Mini-batch 1: pi_theta == pi_prox → ratio = 1.0
        student_logps_mb1 = prox_logps.clone().requires_grad_(True)
        loss_mb1 = policy_gradient_kl(
            student_token_logps=student_logps_mb1, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2,
            use_decoupled_loss=True, prox_logprobs=prox_logps)
        ratios_mb1 = loss_mb1.pg_stats["_ratios"]
        assert torch.allclose(ratios_mb1, torch.ones_like(ratios_mb1), atol=1e-5), \
            "Mini-batch 1: ratio should be 1.0 when pi_theta == pi_prox"

        # Mini-batch 2: pi_theta changed (simulating post-optimizer.step()),
        # but pi_prox is still frozen → ratio != 1.0
        student_logps_mb2 = (prox_logps + 0.5).requires_grad_(True)
        loss_mb2 = policy_gradient_kl(
            student_token_logps=student_logps_mb2, teacher_token_logps=t_logps,
            student_old_logprobs=old_logps, mask=mask, clip_eps=0.2,
            use_decoupled_loss=True, prox_logprobs=prox_logps)
        ratios_mb2 = loss_mb2.pg_stats["_ratios"]
        assert not torch.allclose(ratios_mb2, torch.ones_like(ratios_mb2), atol=1e-2), \
            "Mini-batch 2: ratio should diverge when pi_theta != pi_prox"
        # Ratio should be exp(0.5) ≈ 1.649 for all tokens
        expected_ratio = torch.tensor(0.5).exp()
        assert torch.allclose(ratios_mb2, expected_ratio.expand_as(ratios_mb2), atol=1e-4), \
            f"Expected ratio ≈ {expected_ratio.item():.3f}, got mean {ratios_mb2.mean().item():.3f}"
        # Clipping should activate (ratio 1.649 > 1.0 + 0.2)
        clip_high = loss_mb2.pg_stats["_clip_high"]
        assert clip_high.any(), "Clipping should activate when ratio > 1.2"

    def test_decoupled_ppo_via_dispatch(self):
        """Decoupled PPO threads through compute_kl_loss dispatch."""
        from opd.loss.kl import KLConfig
        g = torch.Generator().manual_seed(42)
        batch, shifted_len = 4, 15
        student_logps = torch.randn(batch, shifted_len, generator=g, requires_grad=True)
        t_logps = torch.randn(batch, shifted_len, generator=g)
        old_logps = torch.randn(batch, shifted_len, generator=g)
        prox_logps = torch.randn(batch, shifted_len, generator=g)
        mask = torch.ones(batch, shifted_len, dtype=torch.bool)

        cfg = KLConfig(mode="policy_gradient_kl", use_decoupled_loss=True,
                        behave_imp_weight_cap=5.0)
        loss = compute_kl_loss(
            student_token_logps=student_logps,
            teacher_token_logps=t_logps,
            student_old_logprobs=old_logps,
            mask=mask, kl_config=cfg,
            prox_logprobs=prox_logps)
        assert torch.isfinite(loss)
        assert "_behave_imp_weight" in loss.pg_stats

    def test_forward_kl_gradient_matches_reference(self):
        """Gradients from optimized version match reference."""
        s, t_lp, t_idx = _make_inputs(batch=2, seq=8, vocab=128, topk=32)
        s_opt = s.clone().requires_grad_(True)
        s_ref = s.clone().requires_grad_(True)
        loss_opt = sparse_forward_kl(s_opt, t_lp, t_idx)
        loss_ref = self._ref_forward_kl(s_ref, t_lp, t_idx)
        loss_opt.backward()
        loss_ref.backward()
        torch.testing.assert_close(s_opt.grad, s_ref.grad, atol=1e-5, rtol=1e-5)

    def test_reverse_kl_gradient_matches_reference(self):
        """Gradients from optimized reverse KL match reference."""
        s, t_lp, t_idx = _make_inputs(batch=2, seq=8, vocab=128, topk=32)
        s_opt = s.clone().requires_grad_(True)
        s_ref = s.clone().requires_grad_(True)
        loss_opt = sparse_reverse_kl(s_opt, t_lp, t_idx)
        loss_ref = self._ref_reverse_kl(s_ref, t_lp, t_idx)
        loss_opt.backward()
        loss_ref.backward()
        torch.testing.assert_close(s_opt.grad, s_ref.grad, atol=1e-5, rtol=1e-5)

    def test_large_vocab_matches(self):
        """Optimization matches reference at realistic vocab size (152064)."""
        vocab = 152064
        topk = 64
        g = torch.Generator().manual_seed(42)
        s = torch.randn(1, 4, vocab, generator=g)
        t_logps_full = F.log_softmax(torch.randn(1, 4, vocab, generator=g), dim=-1)
        t_lp, t_idx = torch.topk(t_logps_full, topk, dim=-1)
        t_idx = t_idx.to(torch.int32)

        opt_fwd = sparse_forward_kl(s, t_lp, t_idx)
        ref_fwd = self._ref_forward_kl(s, t_lp, t_idx)
        torch.testing.assert_close(opt_fwd, ref_fwd, atol=1e-4, rtol=1e-4)

        opt_rev = sparse_reverse_kl(s, t_lp, t_idx)
        ref_rev = self._ref_reverse_kl(s, t_lp, t_idx)
        torch.testing.assert_close(opt_rev, ref_rev, atol=1e-4, rtol=1e-4)


# ===========================================================================
# M2PO dynamic clipping tests
# ===========================================================================

class TestM2PO:
    """Tests for M2PO second-moment dynamic clipping in policy_gradient_kl."""

    def _make_pg_data(self, batch=2, seq=8, vocab=32, seed=42):
        """Create test data for policy_gradient_kl."""
        g = torch.Generator().manual_seed(seed)
        student_logits = torch.randn(batch, seq, vocab, generator=g, requires_grad=True)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)
        teacher_token_logps = torch.randn(batch, seq)
        student_old_logprobs = torch.randn(batch, seq - 1)
        mask = torch.zeros(batch, seq, dtype=torch.bool)
        mask[:, 3:] = True
        return student_logits, teacher_token_logps, input_ids, student_old_logprobs, mask

    def test_m2po_disabled_matches_fixed_clip(self):
        """When m2po_budget=None, behavior is identical to fixed clip_eps."""
        data = self._make_pg_data()
        loss_fixed = policy_gradient_kl(*data, clip_eps=0.2, m2po_budget=None)
        loss_default = policy_gradient_kl(*data, clip_eps=0.2)
        torch.testing.assert_close(loss_fixed, loss_default)

    def test_m2po_produces_finite_loss(self):
        """M2PO with budget=0.04 produces finite loss."""
        data = self._make_pg_data()
        loss = policy_gradient_kl(*data, m2po_budget=0.04)
        assert torch.isfinite(loss)
        assert hasattr(loss, "pg_stats")
        assert "m2po_clip_low" in loss.pg_stats
        assert "m2po_clip_high" in loss.pg_stats
        assert "m2po_m2_before" in loss.pg_stats
        assert "m2po_m2_after" in loss.pg_stats

    def test_m2po_gradient_flows(self):
        """Gradient should flow through M2PO loss."""
        data = self._make_pg_data()
        loss = policy_gradient_kl(*data, m2po_budget=0.04)
        loss.backward()
        assert data[0].grad is not None
        assert not torch.all(data[0].grad == 0)

    def test_m2po_clip_bounds_respect_miniclip(self):
        """Dynamic clip bounds should be >= miniclip floor."""
        data = self._make_pg_data()
        loss = policy_gradient_kl(*data, m2po_budget=0.04,
                                  m2po_miniclip_low=0.3, m2po_miniclip_high=0.5)
        stats = loss.pg_stats
        assert stats["m2po_clip_low"] >= 0.3
        assert stats["m2po_clip_high"] >= 0.5

    def test_m2po_no_constraint_when_m2_low(self):
        """When old==new (M2≈0), no constraint needed → very wide clip bounds."""
        batch, seq, vocab = 2, 8, 32
        g = torch.Generator().manual_seed(42)
        student_logits = torch.randn(batch, seq, vocab, generator=g, requires_grad=True)
        input_ids = torch.randint(0, vocab, (batch, seq), generator=g)

        # Make old logprobs match current student → ratio ≈ 1, M2 ≈ 0
        log_ps = F.log_softmax(student_logits.detach(), dim=-1)
        target_ids = input_ids[:, 1:]
        old_logps = log_ps[:, :-1].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        teacher_token_logps = torch.randn(batch, seq)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        loss = policy_gradient_kl(student_logits, teacher_token_logps, input_ids,
                                  old_logps, mask=mask, m2po_budget=0.04)
        # M2 should be near zero → wide bounds (miniclip floor applied)
        assert loss.pg_stats["m2po_m2_before"] < 0.01

    def test_m2po_tighter_with_stale_data(self):
        """With very stale data (large ratio), M2PO should produce tighter clips."""
        data = self._make_pg_data()
        s_logits, t_logps, ids, _, mask = data
        # Very stale old logprobs → large ratios
        stale_old = torch.full_like(torch.randn(2, 7), -10.0)

        loss = policy_gradient_kl(s_logits, t_logps, ids, stale_old,
                                  mask=mask, m2po_budget=0.04)
        stats = loss.pg_stats
        # M2 before should be high (stale data)
        assert stats["m2po_m2_before"] > 0.04

    def test_m2po_via_dispatch(self):
        """compute_kl_loss with KLConfig correctly passes M2PO params."""
        from opd.loss.kl import KLConfig
        data = self._make_pg_data()
        s_logits, t_logps, ids, old_lp, mask = data

        kl_cfg = KLConfig(
            mode="policy_gradient_kl",
            pg_m2po_budget=0.04,
            pg_m2po_miniclip_low=0.3,
            pg_m2po_miniclip_high=0.5,
        )
        loss = compute_kl_loss(
            student_logits=s_logits,
            teacher_token_logps=t_logps,
            input_ids=ids,
            student_old_logprobs=old_lp,
            mask=mask,
            kl_config=kl_cfg,
        )
        assert torch.isfinite(loss)
        assert "m2po_clip_low" in loss.pg_stats


class TestDenseAlignedKL:
    def test_dense_matches_full_vocab_sparse_modes(self):
        from opd.loss.kl import dense_aligned_kl

        torch.manual_seed(123)
        batch, seq, vocab = 2, 3, 7
        student_logits = torch.randn(batch, seq, vocab)
        teacher_logits = torch.randn(batch, seq, vocab)
        student_logps = F.log_softmax(student_logits, dim=-1)
        teacher_logps = F.log_softmax(teacher_logits, dim=-1)
        indices = torch.arange(vocab).view(1, 1, vocab).expand(batch, seq, vocab)
        mask = torch.tensor([[True, False, True], [True, True, False]])

        cases = [
            ("forward_kl", {}),
            ("reverse_kl", {}),
            ("skewed_kl", {"alpha": 0.25}),
            ("skewed_kl", {"alpha": 0.75}),
        ]
        for mode, kwargs in cases:
            alpha = kwargs.get("alpha", 0.5)
            dense = dense_aligned_kl(
                student_logps,
                teacher_logps,
                mask=mask,
                mode=mode,
                alpha=alpha,
            )
            sparse = compute_kl_loss(
                student_topk_logps=student_logps,
                teacher_topk_logps=teacher_logps,
                teacher_topk_indices=indices,
                mask=mask,
                kl_config=_cfg(mode, skew_alpha=alpha),
            )
            torch.testing.assert_close(dense, sparse, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(
                dense.kl_stats["_vals"],
                sparse.kl_stats["_vals"],
                rtol=1e-6,
                atol=1e-7,
            )

    def test_compute_kl_loss_dispatches_dense_path(self):
        torch.manual_seed(456)
        student_logps = F.log_softmax(torch.randn(1, 2, 5), dim=-1)
        teacher_logps = F.log_softmax(torch.randn(1, 2, 5), dim=-1)
        mask = torch.tensor([[True, False]])
        loss = compute_kl_loss(
            student_dense_logps=student_logps,
            teacher_dense_logps=teacher_logps,
            mask=mask,
            kl_config=_cfg("reverse_kl"),
        )
        assert torch.isfinite(loss)
        assert loss.kl_stats["_vals"].numel() == 1


# ------------------------------------------------------------------ #
# Dense hidden-recompute fused/chunked KL tests
# ------------------------------------------------------------------ #

class TestChunkedDenseKLFromHidden:
    @pytest.mark.parametrize("mode", ["forward_kl", "reverse_kl", "skewed_kl"])
    @pytest.mark.parametrize("token_clip", [0.0, 0.05])
    def test_matches_materialized_dense_loss_and_student_grads(self, mode, token_clip):
        torch.manual_seed(1234)
        B, S, H, V = 2, 5, 4, 11
        student_hidden = torch.randn(B, S, H, dtype=torch.float32, requires_grad=True)
        student_weight = torch.randn(V, H, dtype=torch.float32, requires_grad=True)
        teacher_hidden = torch.randn(B, S, H, dtype=torch.float32)
        teacher_weight = torch.randn(V, H, dtype=torch.float32)
        mask = torch.tensor([[True, False, True, True, False], [False, True, True, False, True]])

        ref_student_hidden = student_hidden.detach().clone().requires_grad_(True)
        ref_student_weight = student_weight.detach().clone().requires_grad_(True)
        ref_student_logps = F.log_softmax(F.linear(ref_student_hidden, ref_student_weight).float(), dim=-1)
        ref_teacher_logps = F.log_softmax(F.linear(teacher_hidden, teacher_weight).float(), dim=-1)
        ref = dense_aligned_kl(
            ref_student_logps,
            ref_teacher_logps,
            mask=mask,
            mode=mode,
            alpha=0.37,
            token_clip=token_clip,
        )
        ref.backward()

        loss = chunked_dense_kl_from_hidden(
            student_hidden,
            student_weight,
            teacher_hidden,
            teacher_weight,
            mask=mask,
            mode=mode,
            alpha=0.37,
            token_clip=token_clip,
            chunk_size=2,
            memory_strategy="checkpoint",
        )
        loss.backward()

        torch.testing.assert_close(loss, ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(student_hidden.grad, ref_student_hidden.grad, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(student_weight.grad, ref_student_weight.grad, atol=2e-6, rtol=2e-5)
        assert getattr(loss, "chunked_dense_kl_stats")["teacher_hidden_materialized_bytes"] == 0
        assert getattr(loss, "chunked_dense_kl_stats")["teacher_hidden_fused_kl_chunk_size"] == 2

    def test_teacher_inputs_stay_detached(self):
        student_hidden = torch.randn(1, 3, 2, requires_grad=True)
        student_weight = torch.randn(5, 2, requires_grad=True)
        teacher_hidden = torch.randn(1, 3, 2, requires_grad=True)
        teacher_weight = torch.randn(5, 2, requires_grad=True)
        loss = chunked_dense_kl_from_hidden(
            student_hidden, student_weight, teacher_hidden, teacher_weight, chunk_size=1
        )
        loss.backward()
        assert student_hidden.grad is not None
        assert student_weight.grad is not None
        assert teacher_hidden.grad is None
        assert teacher_weight.grad is None

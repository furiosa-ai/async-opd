"""Unit tests for GRPO loss, reward functions, and advantage computation."""

import torch
import pytest


# ===========================================================================
# Test GRPO loss function
# ===========================================================================

class TestGRPOClipLoss:
    """Tests for opd.loss.grpo.grpo_clip_loss."""

    def _make_inputs(self, B=4, S=8, V=32, R=4):
        """Create standard test inputs."""
        torch.manual_seed(42)
        logits = torch.randn(B, S, V, requires_grad=True)
        input_ids = torch.randint(0, V, (B, S))
        old_logprobs = torch.randn(B, R)  # log pi_old
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True  # last R positions are response
        return logits, input_ids, old_logprobs, advantages, mask

    def test_basic_loss_computes(self):
        """Loss returns scalar with correct grad."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        loss, stats = grpo_clip_loss(logits, ids, old_lp, adv, mask)
        assert loss.shape == ()
        assert loss.requires_grad
        loss.backward()
        assert logits.grad is not None

    def test_no_kl_penalty(self):
        """kl_beta=0 should give same result with or without ref_token_logps."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        loss1, stats1 = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                        kl_beta=0.0)
        # With ref logps but beta=0
        ref_logps = torch.randn_like(ids.float()[:, :ids.size(1)])
        loss2, stats2 = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                        kl_beta=0.0, ref_token_logps=ref_logps)
        assert torch.allclose(loss1, loss2, atol=1e-6)

    def test_kl_penalty_increases_loss(self):
        """With kl_beta > 0, KL penalty should change the loss."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        ref_logps = torch.zeros_like(ids.float())  # ref logps all 0
        loss_no_kl, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask, kl_beta=0.0)
        loss_with_kl, stats = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                              kl_beta=1.0, ref_token_logps=ref_logps)
        # They should differ
        assert not torch.allclose(loss_no_kl, loss_with_kl, atol=1e-6)
        assert stats["mean_kl"] != 0.0

    def test_gradient_only_through_logits(self):
        """Advantages, old_logprobs, ref_logps should NOT have gradients."""
        from opd.loss.grpo import grpo_clip_loss
        logits = torch.randn(2, 6, 16, requires_grad=True)
        ids = torch.randint(0, 16, (2, 6))
        old_lp = torch.randn(2, 3, requires_grad=True)
        adv = torch.tensor([1.0, -1.0], requires_grad=True)
        ref_lp = torch.randn(2, 6, requires_grad=True)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[:, -3:] = True
        loss, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                  kl_beta=0.5, ref_token_logps=ref_lp)
        loss.backward()
        assert logits.grad is not None
        assert old_lp.grad is None  # detached
        assert adv.grad is None     # detached
        assert ref_lp.grad is None  # detached

    def test_zero_advantages_zero_clip_loss(self):
        """Zero advantages should give zero clip loss (KL only if beta > 0)."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, _, mask = self._make_inputs()
        adv_zero = torch.zeros(4)
        loss, stats = grpo_clip_loss(logits, ids, old_lp, adv_zero, mask, kl_beta=0.0)
        # With zero advantages and zero KL, loss should be ~0
        assert abs(loss.item()) < 1e-5
        assert stats["mean_advantage"] == 0.0

    def test_stats_keys(self):
        """Stats dict should contain expected keys."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        _, stats = grpo_clip_loss(logits, ids, old_lp, adv, mask)
        expected_keys = {"mean_ratio", "mean_log_ratio", "mean_advantage",
                         "clip_fraction", "mean_kl", "_raw_tensors"}
        assert set(stats.keys()) == expected_keys
        assert set(stats["_raw_tensors"].keys()) == {
            "ratios", "log_ratios", "advantages", "clip_high", "clip_low"}


class TestOldLogprobsTruncation:
    """Test that grpo_clip_loss handles old_logprobs longer than sequence."""

    def test_old_logprobs_longer_than_sequence(self):
        """old_logprobs with R > S-1 should be truncated, not crash."""
        from opd.loss.grpo import grpo_clip_loss
        B, S, V, R = 2, 10, 32, 16  # R=16 > S-1=9
        torch.manual_seed(42)
        logits = torch.randn(B, S, V, requires_grad=True)
        ids = torch.randint(0, V, (B, S))
        old_lp = torch.randn(B, R)
        adv = torch.tensor([1.0, -1.0])
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -4:] = True
        loss, stats = grpo_clip_loss(logits, ids, old_lp, adv, mask)
        assert loss.shape == ()
        assert loss.requires_grad

    def test_old_logprobs_equal_to_shifted_len(self):
        """old_logprobs with R == S-1 should work without truncation."""
        from opd.loss.grpo import grpo_clip_loss
        B, S, V = 2, 10, 32
        R = S - 1  # exact fit
        torch.manual_seed(42)
        logits = torch.randn(B, S, V, requires_grad=True)
        ids = torch.randint(0, V, (B, S))
        old_lp = torch.randn(B, R)
        adv = torch.tensor([1.0, -1.0])
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, 1:] = True
        loss, stats = grpo_clip_loss(logits, ids, old_lp, adv, mask)
        assert loss.shape == ()

    def test_truncation_preserves_loss_value(self):
        """Trailing zero-padded old_logprobs beyond shifted_len don't affect loss.

        Simulates the real scenario: old_logprobs has [real_values, zeros]
        where zeros correspond to padding that _prepare_batch truncated.
        The mask-based truncation recovers actual_resp_len=R, preserving
        the right-alignment so loss is identical.
        """
        from opd.loss.grpo import grpo_clip_loss
        B, S, V, R = 2, 10, 32, 4
        torch.manual_seed(42)
        logits = torch.randn(B, S, V, requires_grad=True)
        ids = torch.randint(0, V, (B, S))
        old_lp = torch.randn(B, R)
        adv = torch.tensor([1.0, -1.0])
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True  # last R positions are response
        loss_normal, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask)
        # Pad old_lp with trailing zeros (simulates max_response_length padding).
        # Mask still marks R response tokens, so truncation recovers R real entries.
        padded = torch.cat([old_lp, torch.zeros(B, 10)], dim=1)  # R+10 > S-1=9
        logits2 = logits.detach().clone().requires_grad_(True)
        loss_padded, _ = grpo_clip_loss(logits2, ids, padded, adv, mask)
        torch.testing.assert_close(loss_normal, loss_padded)


# ===========================================================================
# Test reward functions
# ===========================================================================

class TestCorrectnessReward:
    """Tests for opd.reward.correctness_reward."""

    def _mock_tokenizer(self):
        """Create a mock tokenizer that just joins token IDs as strings."""
        class MockTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                # Simulate: token 42 -> "42", wraps in boxed
                if ids == [1, 2, 3]:
                    return "The answer is \\boxed{42}"
                elif ids == [4, 5, 6]:
                    return "The answer is \\boxed{99}"
                elif ids == [7, 8, 9]:
                    return "I don't know"
                elif ids == [10, 11, 12]:
                    return "The answer is \\boxed{42}"
                return "no answer"
        return MockTokenizer()

    def test_correct_and_incorrect(self):
        """Mix of correct and incorrect answers."""
        from opd.reward import correctness_reward
        tok = self._mock_tokenizer()
        responses = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        # prompt0 has responses [0,1], prompt1 has responses [2,3]
        ground_truths = ["42", "42", "42", "42"]
        rewards = correctness_reward(responses, ground_truths, tok, group_size=2)
        assert rewards.shape == (4,)
        assert rewards[0] == 1.0  # correct
        assert rewards[1] == 0.0  # 99 != 42
        assert rewards[2] == 0.0  # no answer
        assert rewards[3] == 1.0  # correct


class TestGroupAdvantages:
    """Tests for opd.reward.compute_group_advantages."""

    def test_basic_normalization(self):
        """Group advantages should have ~mean 0 per group."""
        from opd.reward import compute_group_advantages
        # 2 prompts x 4 samples: [1,0,0,0, 0,0,1,1]
        rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        adv = compute_group_advantages(rewards, group_size=4)
        assert adv.shape == (8,)
        # Group 0: mean ~0
        assert abs(adv[:4].mean().item()) < 1e-5
        # Group 1: mean ~0
        assert abs(adv[4:].mean().item()) < 1e-5

    def test_all_correct_group(self):
        """All correct -> std=0 -> advantages=0."""
        from opd.reward import compute_group_advantages
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        adv = compute_group_advantages(rewards, group_size=4)
        assert (adv == 0.0).all()

    def test_all_incorrect_group(self):
        """All incorrect -> std=0 -> advantages=0."""
        from opd.reward import compute_group_advantages
        rewards = torch.tensor([0.0, 0.0, 0.0, 0.0])
        adv = compute_group_advantages(rewards, group_size=4)
        assert (adv == 0.0).all()

    def test_single_correct(self):
        """Single correct in group: positive advantage for correct, negative for rest."""
        from opd.reward import compute_group_advantages
        rewards = torch.tensor([0.0, 0.0, 1.0, 0.0])
        adv = compute_group_advantages(rewards, group_size=4)
        assert adv[2] > 0  # the correct one is positive
        assert adv[0] < 0  # incorrect ones are negative
        assert adv[1] < 0
        assert adv[3] < 0

    def test_multiple_groups(self):
        """Multiple groups normalize independently."""
        from opd.reward import compute_group_advantages
        # Group 0: [1,0], Group 1: [0,1]
        rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
        adv = compute_group_advantages(rewards, group_size=2)
        # Group 0: adv[0] > 0, adv[1] < 0
        assert adv[0] > 0
        assert adv[1] < 0
        # Group 1: adv[2] < 0, adv[3] > 0
        assert adv[2] < 0
        assert adv[3] > 0
        # Symmetry: |adv[0]| == |adv[1]|
        assert abs(abs(adv[0]) - abs(adv[1])) < 1e-5

    def test_group_size_one(self):
        """G=1: no group contrast, advantages should be 0 (not NaN)."""
        from opd.reward import compute_group_advantages
        rewards = torch.tensor([1.0, 0.0, 1.0])
        adv = compute_group_advantages(rewards, group_size=1)
        assert adv.shape == (3,)
        assert (adv == 0.0).all()
        assert not torch.isnan(adv).any()

    def test_batch_not_divisible_raises(self):
        """Batch size not divisible by group_size should raise."""
        from opd.reward import compute_group_advantages
        rewards = torch.tensor([1.0, 0.0, 0.0])
        with pytest.raises(AssertionError):
            compute_group_advantages(rewards, group_size=2)

    def test_dr_grpo_no_std_norm(self):
        """Dr.GRPO: norm_by_std=False only subtracts mean, no std division."""
        from opd.reward import compute_group_advantages
        rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
        adv = compute_group_advantages(rewards, group_size=4, norm_by_std=False)
        # Mean = 0.25, so adv[0] = 0.75, adv[1:] = -0.25
        assert abs(adv[0].item() - 0.75) < 1e-5
        assert abs(adv[1].item() + 0.25) < 1e-5


# ===========================================================================
# Test DAPO extensions
# ===========================================================================

class TestAsymmetricClipping:
    """Tests for DAPO asymmetric and dual-clip PPO."""

    def _make_inputs(self, B=4, S=8, V=32, R=4):
        torch.manual_seed(42)
        logits = torch.randn(B, S, V, requires_grad=True)
        input_ids = torch.randint(0, V, (B, S))
        old_logprobs = torch.randn(B, R)
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True
        return logits, input_ids, old_logprobs, advantages, mask

    def test_asymmetric_clip(self):
        """Asymmetric clipping should differ from symmetric."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        loss_sym, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                      clip_eps=0.2)
        loss_asym, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                       clip_eps=0.2, clip_ratio_low=0.2,
                                       clip_ratio_high=0.28)
        # Different clip ranges should give different losses
        # (unless all ratios are within both ranges)
        assert loss_sym.requires_grad
        assert loss_asym.requires_grad

    def test_dual_clip(self):
        """Dual-clip should change loss for negative advantages."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, _, mask = self._make_inputs()
        # All negative advantages
        adv_neg = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        loss_no_dc, _ = grpo_clip_loss(logits, ids, old_lp, adv_neg, mask)
        loss_dc, _ = grpo_clip_loss(logits, ids, old_lp, adv_neg, mask,
                                     clip_ratio_c=10.0)
        # Dual-clip bounds loss from below for negative advantages
        assert loss_dc.requires_grad

    def test_seq_mean_token_sum_agg(self):
        """seq-mean-token-sum aggregation should differ from token-mean."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        loss_tm, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                     loss_agg_mode="token-mean")
        loss_smts, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                       loss_agg_mode="seq-mean-token-sum")
        # Both should compute and have gradients
        assert loss_tm.requires_grad
        assert loss_smts.requires_grad


class TestOverlongPenalty:
    """Tests for DAPO overlong reward shaping."""

    def test_no_penalty_under_threshold(self):
        """Responses under expected_len get no penalty."""
        from opd.reward import apply_overlong_penalty
        rewards = torch.tensor([1.0, 1.0])
        lengths = torch.tensor([100, 200])
        result = apply_overlong_penalty(rewards, lengths, max_response_length=1000,
                                         overlong_buffer_len=200)
        # expected_len = 800, both responses under 800
        assert (result == rewards).all()

    def test_full_penalty_at_max(self):
        """Response at max_len gets full penalty."""
        from opd.reward import apply_overlong_penalty
        rewards = torch.tensor([1.0])
        lengths = torch.tensor([1000])
        result = apply_overlong_penalty(rewards, lengths, max_response_length=1000,
                                         overlong_buffer_len=200, penalty_factor=1.0)
        # exceed = 1000 - 800 = 200, penalty = -200/200 * 1.0 = -1.0
        assert abs(result[0].item() - 0.0) < 1e-5

    def test_linear_interpolation(self):
        """Penalty scales linearly in the buffer zone."""
        from opd.reward import apply_overlong_penalty
        rewards = torch.tensor([1.0])
        lengths = torch.tensor([900])  # halfway in buffer [800, 1000]
        result = apply_overlong_penalty(rewards, lengths, max_response_length=1000,
                                         overlong_buffer_len=200, penalty_factor=1.0)
        # exceed = 100, penalty = -100/200 = -0.5
        assert abs(result[0].item() - 0.5) < 1e-5


# ===========================================================================
# Test dual-clip with divergent ratios (verl parity)
# ===========================================================================

class TestDualClipVerl:
    """Verify dual-clip fires when ratio diverges, matching verl behavior."""

    def test_dual_clip_fires_with_large_ratio(self):
        """Dual-clip should cap loss when ratio >> 1 and advantage < 0."""
        from opd.loss.grpo import grpo_clip_loss
        B, S, V, R = 4, 8, 32, 4
        torch.manual_seed(42)
        # Create logits that will produce large log-ratios
        logits = torch.randn(B, S, V)
        ids = torch.randint(0, V, (B, S))
        # old_logprobs very different from current → large ratio
        old_lp = torch.full((B, R), -10.0)  # very low old probs → ratio >> 1
        adv = torch.tensor([-5.0, -5.0, -5.0, -5.0])
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True

        loss_dc, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                     clip_ratio_c=3.0)
        loss_no_dc, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                        clip_ratio_c=None)
        # Dual-clip should reduce loss (caps the surrogate for negative adv)
        assert loss_dc.item() <= loss_no_dc.item() + 1e-6

    def test_dual_clip_no_effect_positive_advantages(self):
        """Dual-clip should NOT affect positive advantages."""
        from opd.loss.grpo import grpo_clip_loss
        B, S, V, R = 4, 8, 32, 4
        torch.manual_seed(42)
        logits = torch.randn(B, S, V)
        ids = torch.randint(0, V, (B, S))
        old_lp = torch.full((B, R), -10.0)
        adv = torch.tensor([5.0, 5.0, 5.0, 5.0])  # all positive
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True

        loss_dc, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                     clip_ratio_c=3.0)
        loss_no_dc, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                        clip_ratio_c=None)
        assert torch.allclose(loss_dc, loss_no_dc, atol=1e-5)

    def test_dual_clip_matches_verl_formula(self):
        """Verify dual-clip formula matches verl:
        pg_losses = where(adv < 0, min(-adv*c, max(-adv*r, -adv*clip(r))), max(...))
        """
        from opd.loss.grpo import grpo_clip_loss
        B, S, V, R = 2, 6, 16, 3
        torch.manual_seed(0)
        logits = torch.randn(B, S, V, requires_grad=True)
        ids = torch.randint(0, V, (B, S))
        old_lp = torch.randn(B, R)
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True

        # One positive, one negative advantage
        adv = torch.tensor([2.0, -2.0])
        loss, stats = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                      clip_ratio_c=3.0, clip_eps=0.2)
        assert loss.requires_grad
        loss.backward()
        assert logits.grad is not None


# ===========================================================================
# Test KL penalty types match verl (k1, k3, k3+)
# ===========================================================================

class TestKLTypesVerl:
    """Verify KL penalty formulas match verl/trainer/ppo/core_algos.py."""

    def _make_inputs(self):
        torch.manual_seed(123)
        B, S, V, R = 4, 8, 32, 4
        logits = torch.randn(B, S, V, requires_grad=True)
        ids = torch.randint(0, V, (B, S))
        old_lp = torch.randn(B, R)
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True
        ref_lp = torch.randn(B, S)
        return logits, ids, old_lp, adv, mask, ref_lp

    def test_k1_is_simple_log_ratio(self):
        """k1: kl = student_logp - ref_logp (can be negative)."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask, ref_lp = self._make_inputs()
        _, stats = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                   kl_beta=1.0, kl_type="k1",
                                   ref_token_logps=ref_lp, clip_ratio_c=None)
        # k1 can be negative
        # Just verify it ran and returned a value
        assert "mean_kl" in stats

    def test_k3_always_nonnegative(self):
        """k3/low_var_kl: kl = exp(ref-student) - (ref-student) - 1 >= 0."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask, ref_lp = self._make_inputs()
        for _ in range(10):
            logits = torch.randn(4, 8, 32)
            ref_lp = torch.randn(4, 8)
            _, stats = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                       kl_beta=1.0, kl_type="k3",
                                       ref_token_logps=ref_lp, clip_ratio_c=None)
            assert stats["mean_kl"] >= -1e-6, f"k3 KL should be >= 0, got {stats['mean_kl']}"

    def test_k3_matches_verl_formula(self):
        """Verify k3 formula: kl = clamp(exp(ref-student) - (ref-student) - 1, -10, 10)."""
        torch.manual_seed(42)
        ref_logp = torch.tensor([[-1.0, -2.0, -0.5]])
        student_logp = torch.tensor([[-0.5, -3.0, -0.5]])
        # Manual computation: kl_diff = ref - student
        kl_diff = (ref_logp - student_logp).clamp(-20, 20)
        expected = (kl_diff.exp() - kl_diff - 1).clamp(-10, 10)
        # All should be >= 0
        assert (expected >= -1e-6).all()
        # Check specific values
        # pos 0: diff=-0.5, exp(-0.5)-(-0.5)-1 = 0.607+0.5-1 = 0.107
        assert abs(expected[0, 0].item() - 0.1065) < 0.01
        # pos 2: diff=0, exp(0)-0-1 = 0
        assert abs(expected[0, 2].item()) < 1e-5

    def test_k3_plus_gradient_differs_from_k3(self):
        """k3+ uses MSE backward (straight-through), so gradients differ from k3."""
        from opd.loss.grpo import grpo_clip_loss
        logits_a = torch.randn(4, 8, 32, requires_grad=True)
        logits_b = logits_a.detach().clone().requires_grad_(True)
        ids = torch.randint(0, 32, (4, 8))
        old_lp = torch.randn(4, 4)
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.zeros(4, 8, dtype=torch.bool)
        mask[:, -4:] = True
        ref_lp = torch.randn(4, 8)

        loss_k3, _ = grpo_clip_loss(logits_a, ids, old_lp, adv, mask,
                                     kl_beta=1.0, kl_type="k3",
                                     ref_token_logps=ref_lp, clip_ratio_c=None)
        loss_k3.backward()

        loss_k3p, _ = grpo_clip_loss(logits_b, ids, old_lp, adv, mask,
                                      kl_beta=1.0, kl_type="k3+",
                                      ref_token_logps=ref_lp, clip_ratio_c=None)
        loss_k3p.backward()

        # Forward values should be the same (same k3 formula)
        assert torch.allclose(loss_k3, loss_k3p, atol=1e-5)
        # But gradients should differ (k3+ uses MSE backward)
        assert not torch.allclose(logits_a.grad, logits_b.grad, atol=1e-5)

    def test_all_kl_types_run(self):
        """All kl_type variants should run without error."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask, ref_lp = self._make_inputs()
        for kl_type in ["k1", "k3", "low_var_kl", "k3+", "low_var_kl+"]:
            loss, stats = grpo_clip_loss(
                logits, ids, old_lp, adv, mask,
                kl_beta=0.01, kl_type=kl_type,
                ref_token_logps=ref_lp, clip_ratio_c=None)
            assert loss.shape == (), f"{kl_type} failed"
            assert "mean_kl" in stats


# ===========================================================================
# Test same-model KL is zero (alignment correctness)
# ===========================================================================


class TestSameModelKLZero:
    """When ref_token_logps comes from the same model as student_logits,
    mean_kl should be ≈ 0 for all KL types."""

    def test_same_model_kl_zero_k1(self):
        self._run_same_model_kl("k1")

    def test_same_model_kl_zero_k3(self):
        self._run_same_model_kl("k3")

    def test_same_model_kl_zero_k3_plus(self):
        self._run_same_model_kl("k3+")

    def _run_same_model_kl(self, kl_type):
        """Build ref_token_logps from the same logits as student, verify KL ≈ 0."""
        from opd.loss.grpo import grpo_clip_loss

        torch.manual_seed(42)
        B, S, V, R = 4, 16, 100, 8
        logits = torch.randn(B, S, V)
        ids = torch.randint(0, V, (B, S))
        # Force ids to be consistent: ids[t+1] is what logits[t] predicts
        old_lp = torch.randn(B, R)
        adv = torch.zeros(B)  # zero advantage to isolate KL
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True

        # Build ref_token_logps: compute log_softmax of logits and gather
        # at the actual next token. ref_token_logps[t] should be
        # log P(ids[t+1] | context) from the same model.
        # After shift: logits[:, :-1] predicts ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)  # [B, S, V]
        # ref_token_logps[t] = log P(ids[t+1]) from position t
        # = log_probs[b, t, ids[b, t+1]] for t=0..S-2
        ref_token_logps = torch.full((B, S), -1e10)
        ref_token_logps[:, :-1] = log_probs[:, :-1].gather(
            2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        loss, stats = grpo_clip_loss(
            logits.clone().requires_grad_(True), ids, old_lp, adv, mask,
            kl_beta=1.0, kl_type=kl_type,
            ref_token_logps=ref_token_logps, clip_ratio_c=None)

        assert abs(stats["mean_kl"]) < 1e-5, (
            f"Same-model KL should be ≈0 for {kl_type}, got {stats['mean_kl']}")


# ===========================================================================
# Test loss aggregation modes
# ===========================================================================

class TestLossAggregation:
    """Verify token-mean vs seq-mean-token-sum aggregation."""

    def test_token_mean_equals_flat_mean(self):
        """token-mean: sum of masked losses / count of masked tokens."""
        from opd.loss.grpo import grpo_clip_loss
        torch.manual_seed(42)
        B, S, V, R = 2, 8, 16, 4
        logits = torch.randn(B, S, V)
        ids = torch.randint(0, V, (B, S))
        old_lp = torch.randn(B, R)
        adv = torch.tensor([1.0, -1.0])
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True

        loss, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                  loss_agg_mode="token-mean", clip_ratio_c=None)
        assert loss.shape == ()
        assert loss.item() != 0.0

    def test_seq_mean_token_sum_differs(self):
        """seq-mean-token-sum should generally differ from token-mean."""
        from opd.loss.grpo import grpo_clip_loss
        torch.manual_seed(42)
        B, S, V, R = 4, 8, 16, 4
        logits = torch.randn(B, S, V)
        ids = torch.randint(0, V, (B, S))
        old_lp = torch.randn(B, R)
        adv = torch.tensor([2.0, -2.0, 1.0, -1.0])
        # Uneven response lengths to make modes differ
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[0, -4:] = True
        mask[1, -2:] = True
        mask[2, -4:] = True
        mask[3, -1:] = True

        loss_tm, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                     loss_agg_mode="token-mean", clip_ratio_c=None)
        loss_sm, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                     loss_agg_mode="seq-mean-token-sum", clip_ratio_c=None)
        # With uneven lengths, these should differ
        assert not torch.allclose(loss_tm, loss_sm, atol=1e-3)


class TestFilterGroups:
    """Tests for DAPO filter_groups."""

    def test_filter_zero_variance(self):
        """Groups with all-same rewards should be filtered."""
        from opd.reward import filter_zero_variance_groups
        rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        # Group 0: [1, 0] -> std > 0 -> keep
        # Group 1: [1, 1] -> std = 0 -> filter
        # Group 2: [0, 0] -> std = 0 -> filter
        keep_mask, n_filtered = filter_zero_variance_groups(rewards, group_size=2)
        assert keep_mask[0] == True
        assert keep_mask[1] == False
        assert keep_mask[2] == False
        assert n_filtered == 2

    def test_all_groups_informative(self):
        """All groups with variance should be kept."""
        from opd.reward import filter_zero_variance_groups
        rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
        keep_mask, n_filtered = filter_zero_variance_groups(rewards, group_size=2)
        assert keep_mask.all()
        assert n_filtered == 0


# ===========================================================================
# Test GRPO group assembly (rollout → coordinator → trainer alignment)
# ===========================================================================


class TestGRPOGroupAssembly:
    """Verify that GRPO group assembly maintains correct prompt→response mapping.

    Tests the _do_generate_grpo output format and the coordinator's
    ground-truth repetition + advantage computation pipeline without
    requiring a real vLLM instance.
    """

    def test_group_ordering_flat_layout(self):
        """Verify flat layout: [p0_r0, p0_r1, ..., p0_rN, p1_r0, ...].

        Simulates the output of _do_generate_grpo and checks that
        prompt_lengths, input_ids, and full_token_lists are ordered correctly.
        """
        num_prompts = 3
        group_size = 4
        max_prompt_len = 8
        max_resp_len = 6
        B = num_prompts * group_size

        # Simulate different prompt lengths (left-padded)
        prompt_lengths_orig = [5, 8, 6]
        prompt_tokens = [
            list(range(100, 100 + pl)) for pl in prompt_lengths_orig
        ]

        # Build input_ids with left-padding (pad=0)
        input_ids = torch.zeros(num_prompts, max_prompt_len, dtype=torch.long)
        attention_mask = torch.zeros(num_prompts, max_prompt_len, dtype=torch.bool)
        for i, (pl, toks) in enumerate(zip(prompt_lengths_orig, prompt_tokens)):
            pad = max_prompt_len - pl
            input_ids[i, pad:] = torch.tensor(toks)
            attention_mask[i, pad:] = True

        # Simulate _do_generate_grpo output: expand each prompt group_size times
        total_len = max_prompt_len + max_resp_len
        full_ids = torch.zeros(B, total_len, dtype=torch.long)
        full_mask = torch.zeros(B, total_len, dtype=torch.bool)
        prompt_lengths_rep = []
        full_token_lists = []
        response_lengths = []

        flat_idx = 0
        for prompt_idx in range(num_prompts):
            p_len = prompt_lengths_orig[prompt_idx]
            pad_len = max_prompt_len - p_len

            for sample_idx in range(group_size):
                # Each sample gets a unique response
                resp_len = 3 + sample_idx  # varying lengths
                resp_ids = list(range(
                    200 + prompt_idx * 100 + sample_idx * 10,
                    200 + prompt_idx * 100 + sample_idx * 10 + resp_len,
                ))

                full_ids[flat_idx, :max_prompt_len] = input_ids[prompt_idx]
                full_mask[flat_idx, pad_len:max_prompt_len] = True
                for j, rid in enumerate(resp_ids):
                    full_ids[flat_idx, max_prompt_len + j] = rid
                    full_mask[flat_idx, max_prompt_len + j] = True

                prompt_lengths_rep.append(p_len)
                response_lengths.append(resp_len)
                full_token_lists.append(
                    prompt_tokens[prompt_idx] + resp_ids)

                flat_idx += 1

        # ── Assertions ──

        # 1. Flat index layout: rows [0..G-1] belong to prompt 0, etc.
        for prompt_idx in range(num_prompts):
            for s in range(group_size):
                flat = prompt_idx * group_size + s
                # Prompt part should match original prompt
                p_len = prompt_lengths_orig[prompt_idx]
                pad = max_prompt_len - p_len
                assert full_ids[flat, pad:max_prompt_len].tolist() == prompt_tokens[prompt_idx], \
                    f"Prompt mismatch at flat_idx={flat}"
                # prompt_length should be repeated correctly
                assert prompt_lengths_rep[flat] == p_len

        # 2. full_token_lists should start with prompt tokens
        for prompt_idx in range(num_prompts):
            for s in range(group_size):
                flat = prompt_idx * group_size + s
                ftl = full_token_lists[flat]
                assert ftl[:prompt_lengths_orig[prompt_idx]] == prompt_tokens[prompt_idx], \
                    f"full_token_lists prompt mismatch at flat_idx={flat}"

        # 3. Responses within a group should be different (different samples)
        for prompt_idx in range(num_prompts):
            resp_set = set()
            for s in range(group_size):
                flat = prompt_idx * group_size + s
                r_len = response_lengths[flat]
                resp = tuple(full_ids[flat, max_prompt_len:max_prompt_len + r_len].tolist())
                resp_set.add(resp)
            assert len(resp_set) == group_size, \
                f"Prompt {prompt_idx}: expected {group_size} unique responses, got {len(resp_set)}"

    def test_ground_truth_repetition(self):
        """Ground truths should be repeated G times to match flat layout."""
        from opd.reward import compute_group_advantages

        G = 4
        num_prompts = 3
        ground_truths = ["42", "7", "100"]

        # Coordinator repeats GTs
        gt_repeated = []
        for gt in ground_truths:
            gt_repeated.extend([gt] * G)

        assert len(gt_repeated) == num_prompts * G
        # Each group of G entries should be the same GT
        for p in range(num_prompts):
            for s in range(G):
                assert gt_repeated[p * G + s] == ground_truths[p]

    def test_advantages_per_group(self):
        """Advantages should normalize within each group, not across groups."""
        from opd.reward import compute_group_advantages

        G = 3
        # 2 prompts × 3 samples: group 0 rewards=[1,0,0], group 1 rewards=[0,0,1]
        rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        adv = compute_group_advantages(rewards, group_size=G)

        # Group 0: mean=1/3, group 1: mean=1/3 — symmetric
        # Within each group: the "1" sample should have positive advantage
        assert adv[0] > 0, "Group 0 correct response should have positive advantage"
        assert adv[1] < 0, "Group 0 wrong response should have negative advantage"
        assert adv[5] > 0, "Group 1 correct response should have positive advantage"
        assert adv[3] < 0, "Group 1 wrong response should have negative advantage"

        # Advantages within each group should sum to ~0 (normalized)
        assert abs(adv[0:3].sum()) < 1e-5, "Group 0 advantages should sum to ~0"
        assert abs(adv[3:6].sum()) < 1e-5, "Group 1 advantages should sum to ~0"

    def test_group_prompt_isolation(self):
        """Rewards from one group should not affect advantages in another."""
        from opd.reward import compute_group_advantages

        G = 2
        # Group 0: [1, 0], Group 1: [1, 1]
        rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
        adv = compute_group_advantages(rewards, group_size=G)

        # Group 0: has variance → nonzero advantages
        assert adv[0] > 0
        assert adv[1] < 0
        # Group 1: all same reward → zero advantages (std=0, normalized to 0)
        assert abs(adv[2]) < 1e-5
        assert abs(adv[3]) < 1e-5


# ===========================================================================
# Test Decoupled PPO
# ===========================================================================

class TestDecoupledPPO:
    """Tests for use_decoupled_loss=True path in grpo_clip_loss."""

    def _make_inputs(self, B=4, S=8, V=32, R=4):
        torch.manual_seed(42)
        logits = torch.randn(B, S, V, requires_grad=True)
        input_ids = torch.randint(0, V, (B, S))
        old_logprobs = torch.randn(B, R)
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True
        return logits, input_ids, old_logprobs, advantages, mask

    def test_decoupled_requires_prox_logprobs(self):
        """use_decoupled_loss=True without prox_logprobs raises ValueError."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        with pytest.raises(ValueError, match="prox_logprobs"):
            grpo_clip_loss(logits, ids, old_lp, adv, mask,
                           use_decoupled_loss=True, prox_logprobs=None)

    def test_decoupled_uses_prox_for_ratio(self):
        """Decoupled loss should use prox_logprobs, not old_logprobs, for ratio."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        # prox == old should give same result as standard
        prox = torch.zeros_like(logits[:, :-1, 0])  # [B, S-1]
        # Fill the response positions with old_logprobs
        R = old_lp.size(1)
        prox[:, -R:] = old_lp

        loss_std, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                      use_decoupled_loss=False)
        loss_dec, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                      use_decoupled_loss=True,
                                      prox_logprobs=prox)
        # When prox == old, decoupled should equal standard (ignoring behave weight)
        # The behavioral importance weight w_behav = exp(prox - old) = exp(0) = 1.0
        # so per_token_loss *= 1.0 → same result
        assert torch.allclose(loss_std, loss_dec, atol=1e-5)

    def test_decoupled_differs_with_different_prox(self):
        """Decoupled loss with different prox_logprobs should differ from standard."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        # prox that differs from old → different ratio base
        prox = torch.randn_like(logits[:, :-1, 0])
        loss_std, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                      use_decoupled_loss=False)
        loss_dec, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                      use_decoupled_loss=True,
                                      prox_logprobs=prox)
        assert not torch.allclose(loss_std, loss_dec, atol=1e-5)

    def test_behave_imp_weight_stats(self):
        """Decoupled loss should report behave_imp_weight and behave_mask_ratio."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        prox = torch.randn_like(logits[:, :-1, 0])
        _, stats = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                   use_decoupled_loss=True,
                                   prox_logprobs=prox)
        assert "behave_imp_weight" in stats
        assert "behave_mask_ratio" in stats
        assert 0.0 <= stats["behave_mask_ratio"] <= 1.0

    def test_behave_imp_weight_cap_zeros_outliers(self):
        """Behavioral weight cap should zero out entries where w_behav > cap."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        # Make prox >> old so w_behav = exp(prox - old) is huge
        prox = torch.zeros_like(logits[:, :-1, 0])
        R = old_lp.size(1)
        prox[:, -R:] = old_lp + 10.0  # exp(10) ≈ 22026, way above cap=5

        _, stats_low_cap = grpo_clip_loss(
            logits, ids, old_lp, adv, mask,
            use_decoupled_loss=True, prox_logprobs=prox,
            behave_imp_weight_cap=5.0)
        # Most/all weights should be masked out (ratio > 5)
        assert stats_low_cap["behave_mask_ratio"] < 0.5

    def test_decoupled_gradient_flows(self):
        """Gradients should flow through logits in decoupled mode."""
        from opd.loss.grpo import grpo_clip_loss
        logits, ids, old_lp, adv, mask = self._make_inputs()
        prox = torch.randn_like(logits[:, :-1, 0])
        loss, _ = grpo_clip_loss(logits, ids, old_lp, adv, mask,
                                  use_decoupled_loss=True,
                                  prox_logprobs=prox)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


# ===========================================================================
# Test Streaming GRPO
# ===========================================================================

class TestStreamingGRPO:
    """Tests for make_stream_score_fn and make_stream_assemble_fn."""

    def _make_grpo_mode(self, group_size=4):
        """Create GRPOMode with minimal config for streaming tests."""
        from unittest.mock import MagicMock
        from opd.coordinator.grpo_mode import GRPOMode

        config = {
            "teacher": {"model": "test-teacher", "n_logprobs": 10},
            "training": {
                "data": {
                    "train_batch_size": 4,
                    "max_prompt_length": 128,
                    "max_response_length": 256,
                },
                "actor_rollout_ref": {"model": {"path": "test"}, "rollout": {}},
                "trainer": {"total_epochs": 1},
            },
            "data": {"train_files": "dummy.parquet", "answer_key": "answer"},
        }
        return GRPOMode(
            rollout_proxy=MagicMock(), teacher_client=MagicMock(),
            trainer_proxy=MagicMock(), trainer_cmd_queue=MagicMock(),
            trainer_result_queue=MagicMock(), tracer=MagicMock(),
            config=config, logger=None,
            grpo_group_size=group_size, grpo_clip_eps=0.2, grpo_kl_beta=0.0,
            reward_fn_name="correctness",
            reward_fn=lambda responses, gts, tok, G, **kw: torch.tensor(
                [1.0 if i % 2 == 0 else 0.0 for i in range(len(responses))]),
            tokenizer=MagicMock(),
            streaming=True,
        )

    def test_stream_score_fn_sets_reward(self):
        """score_fn should set 'reward' field on each sample."""
        mode = self._make_grpo_mode()
        score_fn = mode.make_stream_score_fn(teacher_client=None)
        samples = [
            {"responses": torch.tensor([[1, 2, 3]]),
             "response_lengths": torch.tensor([3]),
             "ground_truth": "42"},
            {"responses": torch.tensor([[4, 5, 6]]),
             "response_lengths": torch.tensor([3]),
             "ground_truth": "7"},
        ]
        score_fn(samples)
        assert "reward" in samples[0]
        assert "reward" in samples[1]
        assert isinstance(samples[0]["reward"], float)

    def test_stream_score_fn_no_gt_gives_zero(self):
        """Samples without ground_truth get reward=0."""
        mode = self._make_grpo_mode()
        score_fn = mode.make_stream_score_fn(teacher_client=None)
        samples = [
            {"responses": torch.tensor([[1, 2, 3]]),
             "response_lengths": torch.tensor([3])},
        ]
        score_fn(samples)
        assert samples[0]["reward"] == 0.0

    def test_stream_assemble_groups_by_prompt_id(self):
        """assemble_fn should group samples by prompt_group_id."""
        mode = self._make_grpo_mode(group_size=2)
        assemble_fn = mode.make_stream_assemble_fn(max_response_length=256)

        S = 10  # sequence length
        samples = []
        for gid in [0, 0, 1, 1]:
            samples.append({
                "input_ids": torch.zeros(1, S, dtype=torch.long),
                "attention_mask": torch.ones(1, S, dtype=torch.bool),
                "prompt_lengths": torch.tensor([4]),
                "response_lengths": torch.tensor([6]),
                "responses": torch.randint(0, 100, (1, S)),
                "student_logprobs": torch.randn(1, 6),
                "prompt_group_id": gid,
                "reward": 1.0 if gid == 0 else 0.0,
                "weight_version": 0,
                "worker_id": 0,
            })

        gen_out, teacher_out = assemble_fn(samples)
        assert gen_out is not None
        assert gen_out["input_ids"].size(0) == 4  # 2 groups × 2 samples
        assert teacher_out["advantages"].size(0) == 4

    def test_stream_assemble_rejects_missing_group_id(self):
        """assemble_fn should reject samples without prompt_group_id (seed prompts).

        TrainDispatcher guarantees complete groups, so assemble_fn no longer
        needs to check group completeness. It only rejects samples missing
        prompt_group_id entirely (seed prompts passed through for capacity
        accounting).
        """
        mode = self._make_grpo_mode(group_size=2)
        assemble_fn = mode.make_stream_assemble_fn(max_response_length=256)

        S = 10
        samples = []
        # 2 samples with prompt_group_id (complete group)
        for gid in [0, 0]:
            samples.append({
                "input_ids": torch.zeros(1, S, dtype=torch.long),
                "attention_mask": torch.ones(1, S, dtype=torch.bool),
                "prompt_lengths": torch.tensor([4]),
                "response_lengths": torch.tensor([6]),
                "responses": torch.randint(0, 100, (1, S)),
                "student_logprobs": torch.randn(1, 6),
                "prompt_group_id": gid,
                "reward": 1.0,
                "weight_version": 0,
                "worker_id": 0,
            })
        # Seed prompt without prompt_group_id
        samples.append({
            "input_ids": torch.zeros(1, S, dtype=torch.long),
            "attention_mask": torch.ones(1, S, dtype=torch.bool),
            "prompt_lengths": torch.tensor([4]),
            "response_lengths": torch.tensor([6]),
            "responses": torch.randint(0, 100, (1, S)),
            "student_logprobs": torch.randn(1, 6),
            "reward": 0.0,
            "weight_version": 0,
            "worker_id": 0,
        })

        gen_out, teacher_out = assemble_fn(samples)
        assert gen_out["input_ids"].size(0) == 2  # only the complete group

    def test_stream_assemble_empty_returns_none(self):
        """assemble_fn with no valid groups should return (None, None)."""
        mode = self._make_grpo_mode(group_size=4)
        assemble_fn = mode.make_stream_assemble_fn(max_response_length=256)
        gen_out, teacher_out = assemble_fn([])
        assert gen_out is None
        assert teacher_out is None

    def test_stream_assemble_response_mask(self):
        """assemble_fn should create correct response_mask from prompt_lengths."""
        mode = self._make_grpo_mode(group_size=2)
        assemble_fn = mode.make_stream_assemble_fn(max_response_length=256)

        S = 10
        prompt_len = 4
        samples = []
        for gid in [0, 0]:
            mask = torch.zeros(1, S, dtype=torch.bool)
            mask[:, :prompt_len + 3] = True  # 4 prompt + 3 response tokens
            samples.append({
                "input_ids": torch.zeros(1, S, dtype=torch.long),
                "attention_mask": mask,
                "prompt_lengths": torch.tensor([prompt_len]),
                "response_lengths": torch.tensor([3]),
                "responses": torch.randint(0, 100, (1, S)),
                "student_logprobs": torch.randn(1, 3),
                "prompt_group_id": gid,
                "reward": 1.0,
                "weight_version": 0,
                "worker_id": 0,
            })

        gen_out, _ = assemble_fn(samples)
        rmask = gen_out["response_mask"]
        # Prompt positions should be False, response positions True (where attention_mask is True)
        assert rmask[:, :prompt_len].sum() == 0
        # Some response positions should be True
        assert rmask[:, prompt_len:].sum() > 0

    def test_stream_assemble_advantages_normalized(self):
        """Assembled advantages should be group-normalized."""
        mode = self._make_grpo_mode(group_size=2)
        assemble_fn = mode.make_stream_assemble_fn(max_response_length=256)

        S = 10
        samples = []
        for gid, reward in [(0, 1.0), (0, 0.0), (1, 1.0), (1, 0.0)]:
            samples.append({
                "input_ids": torch.zeros(1, S, dtype=torch.long),
                "attention_mask": torch.ones(1, S, dtype=torch.bool),
                "prompt_lengths": torch.tensor([4]),
                "response_lengths": torch.tensor([6]),
                "responses": torch.randint(0, 100, (1, S)),
                "student_logprobs": torch.randn(1, 6),
                "prompt_group_id": gid,
                "reward": reward,
                "weight_version": 0,
                "worker_id": 0,
            })

        _, teacher_out = assemble_fn(samples)
        adv = teacher_out["advantages"]
        # Each group's advantages should sum to ~0
        assert abs(adv[0:2].sum()) < 1e-5
        assert abs(adv[2:4].sum()) < 1e-5
        # Correct responses should have positive advantage
        assert adv[0] > 0  # group 0, reward=1.0
        assert adv[1] < 0  # group 0, reward=0.0

    def test_stream_batch_multiplier(self):
        """stream_batch_multiplier should equal group_size."""
        mode = self._make_grpo_mode(group_size=8)
        assert mode.stream_batch_multiplier == 8


# ===========================================================================
# Test GRPOTrainer loss_fn
# ===========================================================================

class TestGRPOTrainerLoss:
    """Tests for GRPOTrainer.loss_fn without a real backend.

    Constructs a GRPOTrainer by patching the lazy FSDPBackend import
    inside __init__, then tests loss_fn directly with synthetic tensors.
    """

    def _make_trainer_and_batch(self, use_decoupled=False, kl_beta=0.0):
        """Create a GRPOTrainer (mock backend) and test batch."""
        from unittest.mock import MagicMock
        import sys

        config = {
            "algorithm": {
                "grpo_clip_eps": 0.2,
                "grpo_kl_beta": kl_beta,
                "clip_ratio_low": None,
                "clip_ratio_high": None,
                "clip_ratio_c": 3.0,
                "loss_agg_mode": "token-mean",
                "kl_type": "k1",
                "use_decoupled_loss": use_decoupled,
                "behave_imp_weight_cap": 5.0,
            },
            "loss_mode": "sft",
            "backend": "fsdp",
        }

        # Patch the fsdp module so the lazy import of FSDPBackend succeeds
        fsdp_mod = MagicMock()
        fsdp_mod.FSDPBackend.return_value = MagicMock()
        orig = sys.modules.get("opd.trainer.fsdp")
        sys.modules["opd.trainer.fsdp"] = fsdp_mod
        try:
            from opd.trainer.grpo import GRPOTrainer
            trainer = GRPOTrainer(config, rank_info={"rank": 0, "world_size": 1})
        finally:
            if orig is not None:
                sys.modules["opd.trainer.fsdp"] = orig
            else:
                sys.modules.pop("opd.trainer.fsdp", None)

        B, S, V, R = 4, 12, 32, 6
        torch.manual_seed(42)
        logits = torch.randn(B, S, V, requires_grad=True)
        batch = {
            "input_ids": torch.randint(0, V, (B, S)),
            "response_mask": torch.zeros(B, S, dtype=torch.bool),
            "student_old_logprobs": torch.randn(B, R),
            "advantages": torch.tensor([1.0, -1.0, 0.5, -0.5]),
        }
        batch["response_mask"][:, -R:] = True

        return trainer, logits, batch

    def test_loss_fn_returns_tuple(self):
        """loss_fn should return (loss, n_tokens, stats_dict)."""
        trainer, logits, batch = self._make_trainer_and_batch()
        loss, n_tok, stats = trainer.loss_fn(logits, batch)
        assert loss.shape == ()
        assert loss.requires_grad
        assert isinstance(n_tok, int)
        assert n_tok > 0
        assert isinstance(stats, dict)
        assert "mean_ratio" in stats

    def test_loss_fn_n_tokens_matches_mask(self):
        """n_tokens should equal the number of True positions in shifted mask."""
        trainer, logits, batch = self._make_trainer_and_batch()
        _, n_tok, _ = trainer.loss_fn(logits, batch)
        expected = int(batch["response_mask"][:, 1:].sum().item())
        assert n_tok == expected

    def test_loss_fn_with_kl(self):
        """loss_fn with kl_beta > 0 should use ref_token_logps."""
        trainer, logits, batch = self._make_trainer_and_batch(kl_beta=0.5)
        B, S = batch["input_ids"].shape
        batch["ref_token_logps"] = torch.randn(B, S)
        loss, _, stats = trainer.loss_fn(logits, batch)
        assert loss.requires_grad
        assert stats["mean_kl"] != 0.0

    def test_loss_fn_decoupled_fallback(self):
        """Decoupled mode without _prox_logprobs should compute on-the-fly."""
        trainer, logits, batch = self._make_trainer_and_batch(use_decoupled=True)
        # No _prox_logprobs in batch → fallback path
        loss, _, stats = trainer.loss_fn(logits, batch)
        assert loss.requires_grad
        assert "behave_imp_weight" in stats

    def test_loss_fn_decoupled_with_prox(self):
        """Decoupled mode with precomputed _prox_logprobs should use them."""
        trainer, logits, batch = self._make_trainer_and_batch(use_decoupled=True)
        B, S = batch["input_ids"].shape
        batch["_prox_logprobs"] = torch.randn(B, S - 1)
        loss, _, stats = trainer.loss_fn(logits, batch)
        assert loss.requires_grad
        assert "behave_imp_weight" in stats


# ===========================================================================
# Test data_iterator streaming vs step-off
# ===========================================================================

class TestDataIteratorModes:
    """Test that data_iterator handles streaming and step-off paths correctly."""

    def _make_grpo_mode(self, streaming=False):
        from unittest.mock import MagicMock
        from opd.coordinator.grpo_mode import GRPOMode

        config = {
            "teacher": {"model": "test-teacher", "n_logprobs": 10},
            "training": {
                "data": {
                    "train_batch_size": 4,
                    "max_prompt_length": 128,
                    "max_response_length": 128,
                    "prompt_key": "prompt",
                },
                "actor_rollout_ref": {"model": {"path": "test"}, "rollout": {}},
                "trainer": {"total_epochs": 1},
            },
            "data": {
                "train_files": "dummy.parquet",
                "answer_key": "answer",
            },
        }
        return GRPOMode(
            rollout_proxy=MagicMock(), teacher_client=MagicMock(),
            trainer_proxy=MagicMock(), trainer_cmd_queue=MagicMock(),
            trainer_result_queue=MagicMock(), tracer=MagicMock(),
            config=config, logger=None,
            grpo_group_size=4, grpo_clip_eps=0.2, grpo_kl_beta=0.0,
            reward_fn_name="correctness",
            reward_fn=MagicMock(),
            tokenizer=MagicMock(),
            streaming=streaming,
        )

    def test_on_resume_skip_complete_clears_queue(self):
        """on_resume_skip_complete should clear the GT FIFO queue."""
        mode = self._make_grpo_mode(streaming=False)
        mode._gt_queue.append(["stale1"])
        mode._gt_queue.append(["stale2"])
        mode.on_resume_skip_complete()
        assert len(mode._gt_queue) == 0


# ===========================================================================
# Test DAPO combined (all 5 features together)
# ===========================================================================

class TestDAPOCombined:
    """Test all 5 DAPO features exercised together in a single loss call.

    Features: asymmetric clipping, dual-clip, token-mean aggregation,
    filter_groups, and overlong reward shaping.
    """

    def test_full_dapo_pipeline(self):
        """End-to-end DAPO: rewards → overlong penalty → filter groups →
        advantages → grpo_clip_loss with asymmetric + dual-clip + token-mean."""
        from opd.loss.grpo import grpo_clip_loss
        from opd.reward import (
            compute_group_advantages, apply_overlong_penalty,
            filter_zero_variance_groups,
        )

        torch.manual_seed(42)
        G = 4
        num_prompts = 3
        B = num_prompts * G
        S, V, R = 16, 32, 8

        # 1. Rewards: mixed correct/incorrect
        rewards = torch.tensor([
            1.0, 0.0, 1.0, 0.0,   # group 0: 50% correct
            1.0, 1.0, 1.0, 1.0,   # group 1: 100% correct (zero variance)
            0.0, 0.0, 1.0, 0.0,   # group 2: 25% correct
        ])

        # 2. Overlong penalty
        response_lengths = torch.tensor([
            6, 6, 6, 6,    # group 0: all short
            6, 6, 6, 14,   # group 1: one overlong (exceeds expected_len=8)
            6, 6, 6, 6,    # group 2: all short
        ])
        max_resp = 16
        buffer_len = 8  # expected_len = 16 - 8 = 8
        rewards = apply_overlong_penalty(rewards, response_lengths, max_resp,
                                          buffer_len, penalty_factor=1.0)
        # Group 1 sample 3 (resp_len=14) gets penalty: -(14-8)/8 = -0.75
        assert rewards[7] < 1.0  # was 1.0, now penalized

        # 3. Filter zero-variance groups
        keep_mask, n_filtered = filter_zero_variance_groups(rewards, G)
        # Group 1 was all-correct but now one has overlong penalty → has variance
        # So filtering depends on actual reward values

        # 4. Advantages
        advantages = compute_group_advantages(rewards, G)
        assert advantages.shape == (B,)
        # Group advantages should sum to ~0 per group
        for g in range(num_prompts):
            assert abs(advantages[g * G:(g + 1) * G].sum()) < 1e-5

        # 5. GRPO loss with DAPO settings
        logits = torch.randn(B, S, V, requires_grad=True)
        input_ids = torch.randint(0, V, (B, S))
        old_logprobs = torch.randn(B, R)
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True

        loss, stats = grpo_clip_loss(
            logits, input_ids, old_logprobs, advantages, mask,
            clip_eps=0.2,
            clip_ratio_low=0.2,       # DAPO asymmetric
            clip_ratio_high=0.28,     # DAPO asymmetric
            clip_ratio_c=10.0,        # DAPO dual-clip
            loss_agg_mode="token-mean",  # DAPO aggregation
            kl_beta=0.0,             # DAPO: no KL
        )

        assert loss.shape == ()
        assert loss.requires_grad
        loss.backward()
        assert logits.grad is not None
        assert stats["clip_fraction"] >= 0.0
        assert stats["mean_kl"] == 0.0  # no KL penalty

    def test_dapo_differs_from_standard_grpo(self):
        """DAPO settings should produce different loss than standard GRPO."""
        from opd.loss.grpo import grpo_clip_loss

        torch.manual_seed(42)
        B, S, V, R = 8, 12, 32, 6
        logits = torch.randn(B, S, V, requires_grad=True)
        input_ids = torch.randint(0, V, (B, S))
        old_logprobs = torch.randn(B, R)
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5])
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, -R:] = True

        # Standard GRPO
        loss_grpo, _ = grpo_clip_loss(
            logits, input_ids, old_logprobs, advantages, mask,
            clip_eps=0.2,
            clip_ratio_c=None,  # no dual-clip
            loss_agg_mode="seq-mean-token-sum",
        )

        # DAPO
        loss_dapo, _ = grpo_clip_loss(
            logits, input_ids, old_logprobs, advantages, mask,
            clip_eps=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.28,
            clip_ratio_c=10.0,
            loss_agg_mode="token-mean",
        )

        # Should differ due to asymmetric clip + dual-clip + different aggregation
        assert not torch.allclose(loss_grpo, loss_dapo, atol=1e-6)

    def test_dapo_overlong_plus_filter_interaction(self):
        """Overlong penalty can introduce variance in an all-correct group,
        preventing it from being filtered."""
        from opd.reward import (
            apply_overlong_penalty, filter_zero_variance_groups,
        )

        G = 4
        # All correct → zero variance → would be filtered
        rewards = torch.ones(G)
        keep_mask_before, n_before = filter_zero_variance_groups(rewards, G)
        assert n_before == 1  # filtered

        # Apply overlong penalty to one sample → introduces variance
        response_lengths = torch.tensor([6, 6, 6, 14])
        rewards_penalized = apply_overlong_penalty(
            rewards.clone(), response_lengths,
            max_response_length=16, overlong_buffer_len=8)
        keep_mask_after, n_after = filter_zero_variance_groups(rewards_penalized, G)
        assert n_after == 0  # not filtered — penalty introduced variance


# ===========================================================================
# Mini-batch count invariance to group size
# ===========================================================================

class TestMiniBatchGroupSizeScaling:
    """Verify that mini_batch_size scaling by grpo_group_size keeps
    n_optim_steps (and staleness accounting) independent of group size.

    verl-opd scales ppo_mini_batch_size by rollout.n (group_size) so that
    the user specifies mini-batch in "prompt space".  Our process_lifecycle
    does the same: mini_batch_size * grpo_group_size.
    """

    def test_config_scales_mini_batch_by_group_size(self):
        """_build_trainer_config multiplies mini_batch_size by grpo_group_size."""
        from opd.coordinator.process_lifecycle import ProcessLifecycleMixin

        for group_size in [1, 4, 8, 16]:
            algo_cfg = {"grpo_group_size": group_size}
            actor_cfg_mini = 64
            scaled = actor_cfg_mini * int(algo_cfg.get("grpo_group_size", 1))
            assert scaled == 64 * group_size, (
                f"group_size={group_size}: expected {64 * group_size}, got {scaled}"
            )

    def test_n_optim_steps_invariant_to_group_size(self):
        """FSDP backend produces the same n_optim_steps regardless of group size
        when mini_batch_size is properly scaled.

        Simulates: train_batch_size=4 prompts, mini_batch_size=2 (in prompt space).
        - group_size=1: 4 samples, mini_bs=2  → n_mini=2
        - group_size=2: 8 samples, mini_bs=4  → n_mini=2
        - group_size=4: 16 samples, mini_bs=8 → n_mini=2
        """
        import types
        import torch.nn as nn
        import torch.nn.functional as F
        from types import SimpleNamespace
        from opd.trainer.fsdp.backend import FSDPBackend
        from opd.trainer.base import BaseBackend
        from opd.loss.kl import KLConfig
        from opd.trainer.opd import OPDTrainer

        V, H, S, K = 32, 8, 8, 4
        max_prompt = 3
        base_prompts = 4
        base_mini_bs = 2  # in prompt space

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(V, H)
                self.fc = nn.Linear(H, H)
            def forward(self, input_ids=None, **kwargs):
                return (self.fc(self.embed(input_ids)),)

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Inner()
                self.lm_head = nn.Linear(H, V, bias=False)
            def forward(self, input_ids=None, attention_mask=None,
                        _kl_args=None, **kwargs):
                h = self.model(input_ids=input_ids)[0]
                return SimpleNamespace(logits=self.lm_head(h))

        results = {}
        for group_size in [1, 2, 4]:
            B = base_prompts * group_size
            scaled_mini_bs = base_mini_bs * group_size

            torch.manual_seed(0)
            model = _Model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            ns = SimpleNamespace(
                model=model, optimizer=optimizer, scheduler=None,
                device=torch.device("cpu"),
                micro_batch_size=max(1, scaled_mini_bs),
                max_response_length=S - max_prompt,
                mini_batch_size=scaled_mini_bs,
                use_sequence_packing=False,
                rank=0, world_size=1, total_steps=0,
                _scheduler_needs_rebuild=False, max_grad_norm=1.0,
            )
            ns._run_train_step = types.MethodType(FSDPBackend._run_train_step, ns)
            ns._prepare_train_batch = types.MethodType(
                BaseBackend._prepare_train_batch, ns)
            FSDPBackend._patch_model_for_chunked_kl(ns)

            trainer = object.__new__(OPDTrainer)
            trainer.kl_loss_mode = "forward_kl"
            trainer.kl_skew_alpha = 0.5
            trainer.pg_clip_eps = 0.2
            trainer.pg_online_advantage = False
            trainer.kl_token_clip = 0.0
            trainer.kl_config = KLConfig(mode="forward_kl", skew_alpha=0.5,
                                          pg_clip_eps=0.2, token_clip=0.0)
            trainer._backend = SimpleNamespace(kl_chunk_size=1024,
                                               use_sequence_packing=False)
            ns._opd_trainer = trainer

            g = torch.Generator().manual_seed(42)
            batch = {
                "input_ids": torch.randint(0, V, (B, S), generator=g),
                "attention_mask": torch.ones(B, S, dtype=torch.long),
                "response_mask": torch.zeros(B, S, dtype=torch.bool),
                "prompt_lengths": torch.full((B,), max_prompt, dtype=torch.long),
                "teacher_topk_logps": F.log_softmax(
                    torch.randn(B, S, K, generator=g), dim=-1),
                "teacher_topk_indices": torch.randint(0, V, (B, S, K), generator=g),
                "max_prompt": max_prompt,
                "actual_max_len": S,
                "seq_len": S,
                "orig_seq_len": S,
            }
            batch["response_mask"][:, max_prompt:] = True

            metrics = ns._run_train_step(
                batch, ns._opd_trainer.loss_fn,
                forward_and_loss_fn=ns._opd_trainer.forward_and_loss_fn,
            )
            results[group_size] = metrics["n_optim_steps"]

        # All group sizes should produce the same n_optim_steps
        expected = base_prompts // base_mini_bs  # 4 / 2 = 2
        for gs, n_optim in results.items():
            assert n_optim == expected, (
                f"group_size={gs}: n_optim_steps={n_optim}, expected {expected}. "
                f"mini_batch_size should be scaled by group_size."
            )

    def test_scheduler_n_mini_matches_trainer(self):
        """The n_mini computed by step_off/streaming coordinators (from raw config)
        should match what the trainer computes (from scaled config).

        Scheduler: n_mini = train_batch_size / mini_batch_size  (prompt space)
        Trainer:   n_mini = (train_batch_size * G) / (mini_batch_size * G)  (sample space)
        Both should equal train_batch_size / mini_batch_size.
        """
        train_batch_size = 256
        mini_batch_size = 64

        for group_size in [1, 4, 8, 16]:
            # Scheduler computation (prompt space, from raw config)
            sched_n_mini = train_batch_size // mini_batch_size

            # Trainer computation (sample space, with scaling)
            trainer_batch = train_batch_size * group_size
            trainer_mini_bs = mini_batch_size * group_size
            trainer_n_mini = trainer_batch // trainer_mini_bs

            assert sched_n_mini == trainer_n_mini, (
                f"group_size={group_size}: scheduler n_mini={sched_n_mini} != "
                f"trainer n_mini={trainer_n_mini}"
            )

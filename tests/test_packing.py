"""Tests for sequence packing utilities and numerical equivalence.

Tests cover:
1. Pack/unpack correctness (cu_seqlens, position_ids, data alignment)
2. Numerical equivalence: packed vs padded for all KL modes (requires GPU + FA2)
3. Gradient equivalence
4. Edge cases (single seq, uniform lengths, mixed short/long)
5. Boundary token safety
"""

import pytest
import torch

from opd.data.packing import pack_micro_batch, PackedBatch


# ============================================================
# Test 1: Pack/unpack round-trip
# ============================================================

class TestPackMicroBatch:
    """Unit tests for pack_micro_batch — no GPU needed."""

    def _make_padded_batch(self, seq_lens, prompt_lens, pad_to=None, K=4):
        """Create a synthetic padded micro-batch."""
        B = len(seq_lens)
        max_len = pad_to or max(seq_lens)

        input_ids = torch.zeros(B, max_len, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        teacher_topk_logps = torch.zeros(B, max_len, K)
        teacher_topk_indices = torch.zeros(B, max_len, K, dtype=torch.long)
        support_student_old_logps = torch.zeros(B, max_len, K)
        response_mask = torch.zeros(B, max_len, dtype=torch.bool)
        prompt_lengths = torch.tensor(prompt_lens, dtype=torch.long)

        for i, (n, p) in enumerate(zip(seq_lens, prompt_lens)):
            # Fill with distinct values so we can verify alignment
            input_ids[i, :n] = torch.arange(1, n + 1) + i * 1000
            attention_mask[i, :n] = 1
            teacher_topk_logps[i, :n] = torch.randn(n, K) + i
            teacher_topk_indices[i, :n] = torch.randint(0, 100, (n, K))
            support_student_old_logps[i, :n] = torch.randn(n, K) - i
            response_mask[i, p:n] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_topk_logps": teacher_topk_logps,
            "teacher_topk_indices": teacher_topk_indices,
            "support_student_old_logps": support_student_old_logps,
            "response_mask": response_mask,
            "prompt_lengths": prompt_lengths,
        }

    def test_basic_packing(self):
        """Pack 3 sequences of lengths [100, 200, 150] padded to 200."""
        data = self._make_padded_batch(
            seq_lens=[100, 200, 150],
            prompt_lens=[30, 50, 40],
            pad_to=200,
        )
        packed = pack_micro_batch(**data)

        assert isinstance(packed, PackedBatch)
        assert packed.input_ids.shape == (1, 450)
        assert packed.position_ids.shape == (1, 450)
        assert packed.teacher_topk_logps.shape == (1, 450, 4)
        assert packed.teacher_topk_indices.shape == (1, 450, 4)
        assert packed.support_student_old_logps.shape == (1, 450, 4)
        assert packed.response_mask.shape == (1, 450)
        assert packed.max_seq_len == 200
        assert packed.cu_seq_lens.tolist() == [0, 100, 300, 450]
        assert packed.seq_lens.tolist() == [100, 200, 150]
        assert packed.prompt_lens.tolist() == [30, 50, 40]

    def test_position_ids_reset(self):
        """Position IDs should reset at each sequence boundary."""
        data = self._make_padded_batch(
            seq_lens=[5, 3, 4],
            prompt_lens=[2, 1, 2],
            pad_to=5,
        )
        packed = pack_micro_batch(**data)

        expected_pos = [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3]
        assert packed.position_ids[0].tolist() == expected_pos

    def test_data_alignment(self):
        """Verify packed data maps back to correct original positions."""
        data = self._make_padded_batch(
            seq_lens=[5, 3, 4],
            prompt_lens=[2, 1, 2],
            pad_to=5,
        )
        packed = pack_micro_batch(**data)

        # Sequence 0: positions 0-4 in packed = original[0, 0:5]
        torch.testing.assert_close(
            packed.input_ids[0, :5],
            data["input_ids"][0, :5],
        )
        # Sequence 1: positions 5-7 in packed = original[1, 0:3]
        torch.testing.assert_close(
            packed.input_ids[0, 5:8],
            data["input_ids"][1, :3],
        )
        # Sequence 2: positions 8-11 in packed = original[2, 0:4]
        torch.testing.assert_close(
            packed.input_ids[0, 8:12],
            data["input_ids"][2, :4],
        )

    def test_teacher_data_alignment(self):
        """Teacher logps/indices aligned correctly in packed layout."""
        data = self._make_padded_batch(
            seq_lens=[5, 3],
            prompt_lens=[2, 1],
            pad_to=5,
        )
        packed = pack_micro_batch(**data)

        # Seq 0 teacher data
        torch.testing.assert_close(
            packed.teacher_topk_logps[0, :5],
            data["teacher_topk_logps"][0, :5],
        )
        # Seq 1 teacher data
        torch.testing.assert_close(
            packed.teacher_topk_logps[0, 5:8],
            data["teacher_topk_logps"][1, :3],
        )

    def test_response_mask_per_sequence(self):
        """Response mask uses per-sequence prompt lengths."""
        data = self._make_padded_batch(
            seq_lens=[6, 4],
            prompt_lens=[2, 3],
            pad_to=6,
        )
        packed = pack_micro_batch(**data)

        # Seq 0: prompt [0,1], response [2,3,4,5]
        # Seq 1: prompt [0,1,2], response [3]
        expected = [False, False, True, True, True, True,  # seq 0
                    False, False, False, True]              # seq 1
        assert packed.response_mask[0].tolist() == expected

    def test_cu_seq_lens_type_and_values(self):
        """cu_seq_lens should be int32 with correct cumulative sums."""
        data = self._make_padded_batch(
            seq_lens=[10, 20, 15],
            prompt_lens=[3, 5, 4],
            pad_to=20,
        )
        packed = pack_micro_batch(**data)

        assert packed.cu_seq_lens.dtype == torch.int32
        assert packed.cu_seq_lens.tolist() == [0, 10, 30, 45]

    def test_teacher_token_logps_packing(self):
        """Optional teacher_token_logps are packed correctly."""
        B, S = 3, 10
        seq_lens = [8, 10, 6]
        prompt_lens = [3, 4, 2]
        data = self._make_padded_batch(seq_lens, prompt_lens, pad_to=S)
        teacher_token_logps = torch.randn(B, S)
        for i, n in enumerate(seq_lens):
            teacher_token_logps[i, n:] = 0  # zero padding

        packed = pack_micro_batch(
            **data,
            teacher_token_logps=teacher_token_logps,
        )

        assert packed.teacher_token_logps is not None
        assert packed.teacher_token_logps.shape == (1, 24)  # 8+10+6
        # Verify alignment
        torch.testing.assert_close(
            packed.teacher_token_logps[0, :8],
            teacher_token_logps[0, :8],
        )

    def test_student_logprobs_packing(self):
        """Student logprobs placed at correct response positions."""
        seq_lens = [8, 6]
        prompt_lens = [3, 2]
        data = self._make_padded_batch(seq_lens, prompt_lens, pad_to=8)

        # student_logprobs: [B, resp_len] where resp_len = max(n - p)
        max_resp = max(n - p for n, p in zip(seq_lens, prompt_lens))
        student_logprobs = torch.randn(2, max_resp)

        packed = pack_micro_batch(
            **data,
            student_logprobs=student_logprobs,
        )

        assert packed.student_logprobs is not None
        assert packed.student_logprobs.shape == (1, 14)  # 8+6

        # Seq 0: prompt=3, resp=5. student_logprobs at positions [3:8]
        torch.testing.assert_close(
            packed.student_logprobs[0, 3:8],
            student_logprobs[0, :5],
        )
        # Seq 0: prompt positions should be zero
        assert (packed.student_logprobs[0, :3] == 0).all()

        # Seq 1: prompt=2, resp=4. student_logprobs at positions [8+2:8+6] = [10:14]
        torch.testing.assert_close(
            packed.student_logprobs[0, 10:14],
            student_logprobs[1, :4],
        )
        # Seq 1: prompt positions should be zero
        assert (packed.student_logprobs[0, 8:10] == 0).all()

    def test_support_student_old_logps_packing(self):
        """Support-aligned old top-k logprobs survive packing unchanged."""
        data = self._make_padded_batch(
            seq_lens=[6, 4],
            prompt_lens=[2, 1],
            pad_to=6,
            K=3,
        )
        support_student_old_logps = data["support_student_old_logps"].clone()

        packed = pack_micro_batch(**data)

        assert packed.support_student_old_logps is not None
        assert packed.support_student_old_logps.shape == (1, 10, 3)
        torch.testing.assert_close(
            packed.support_student_old_logps[0, :6],
            support_student_old_logps[0, :6],
        )
        torch.testing.assert_close(
            packed.support_student_old_logps[0, 6:10],
            support_student_old_logps[1, :4],
        )

    def test_multi_sample_tensors_pack_with_token_axis(self):
        """Canonical mc_* tensors preserve per-token/per-sample ordering."""
        seq_lens = [6, 4]
        prompt_lens = [2, 1]
        data = self._make_padded_batch(seq_lens, prompt_lens, pad_to=6)

        mc_sample_indices = torch.tensor(
            [
                [[10, 11], [20, 21], [30, 31], [40, 41], [50, 51], [60, 61]],
                [[70, 71], [80, 81], [90, 91], [100, 101], [0, 0], [0, 0]],
            ],
            dtype=torch.long,
        )
        mc_teacher_logprobs = torch.arange(24, dtype=torch.float32).view(2, 6, 2) * -0.1
        mc_old_logprobs = mc_teacher_logprobs - 0.5

        packed = pack_micro_batch(
            **data,
            mc_sample_indices=mc_sample_indices,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mc_old_logprobs=mc_old_logprobs,
        )

        assert packed.mc_sample_indices is not None
        assert packed.mc_teacher_logprobs is not None
        assert packed.mc_old_logprobs is not None
        assert packed.mc_sample_indices.shape == (1, 10, 2)
        torch.testing.assert_close(
            packed.mc_sample_indices[0, :6],
            mc_sample_indices[0, :6],
        )
        torch.testing.assert_close(
            packed.mc_sample_indices[0, 6:10],
            mc_sample_indices[1, :4],
        )
        torch.testing.assert_close(
            packed.mc_teacher_logprobs[0, :6],
            mc_teacher_logprobs[0, :6],
        )
        torch.testing.assert_close(
            packed.mc_old_logprobs[0, 6:10],
            mc_old_logprobs[1, :4],
        )


# ============================================================
# Test: Left padding support
# ============================================================

class TestLeftPadding:
    """Verify packing works with left-padded sequences (pipeline default)."""

    def _make_left_padded_batch(self, seq_lens, prompt_lens, pad_to=None, K=4):
        """Create a left-padded micro-batch (padding on the left)."""
        B = len(seq_lens)
        max_len = pad_to or max(seq_lens)

        input_ids = torch.zeros(B, max_len, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        teacher_topk_logps = torch.zeros(B, max_len, K)
        teacher_topk_indices = torch.zeros(B, max_len, K, dtype=torch.long)
        response_mask = torch.zeros(B, max_len, dtype=torch.bool)
        prompt_lengths = torch.tensor(prompt_lens, dtype=torch.long)

        for i, (n, p) in enumerate(zip(seq_lens, prompt_lens)):
            start = max_len - n  # left padding: real tokens at end
            input_ids[i, start:] = torch.arange(1, n + 1) + i * 1000
            attention_mask[i, start:] = 1
            teacher_topk_logps[i, start:] = torch.randn(n, K) + i
            teacher_topk_indices[i, start:] = torch.randint(0, 100, (n, K))
            response_mask[i, start + p:] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_topk_logps": teacher_topk_logps,
            "teacher_topk_indices": teacher_topk_indices,
            "response_mask": response_mask,
            "prompt_lengths": prompt_lengths,
        }

    def test_left_padded_token_extraction(self):
        """Packed tokens must match real (non-pad) tokens, not leading pads."""
        data = self._make_left_padded_batch(
            seq_lens=[5, 8],
            prompt_lens=[2, 3],
            pad_to=10,
        )
        packed = pack_micro_batch(**data)

        # Total packed tokens = 5 + 8 = 13
        assert packed.input_ids.shape == (1, 13)

        # Seq 0: real tokens at positions 5:10 in original → ids [1,2,3,4,5]
        assert packed.input_ids[0, :5].tolist() == [1, 2, 3, 4, 5]
        # Seq 1: real tokens at positions 2:10 in original → ids [1001..1008]
        assert packed.input_ids[0, 5:13].tolist() == list(range(1001, 1009))

    def test_left_padded_response_mask(self):
        """Response mask must reflect actual response positions, not pads."""
        data = self._make_left_padded_batch(
            seq_lens=[5, 8],
            prompt_lens=[2, 3],
            pad_to=10,
        )
        packed = pack_micro_batch(**data)

        # Seq 0: 2 prompt + 3 response → [F, F, T, T, T]
        # Seq 1: 3 prompt + 5 response → [F, F, F, T, T, T, T, T]
        expected = [False, False, True, True, True,
                    False, False, False, True, True, True, True, True]
        assert packed.response_mask[0].tolist() == expected

    def test_left_padded_n_tokens_matches_right_padded(self):
        """Left and right padded batches must produce same token count."""
        seq_lens = [5, 8, 6]
        prompt_lens = [2, 3, 2]
        K = 4

        # Right-padded
        B = len(seq_lens)
        max_len = 10
        rp = {"input_ids": torch.zeros(B, max_len, dtype=torch.long),
              "attention_mask": torch.zeros(B, max_len, dtype=torch.long),
              "teacher_topk_logps": torch.zeros(B, max_len, K),
              "teacher_topk_indices": torch.zeros(B, max_len, K, dtype=torch.long),
              "response_mask": torch.zeros(B, max_len, dtype=torch.bool),
              "prompt_lengths": torch.tensor(prompt_lens, dtype=torch.long)}
        for i, (n, p) in enumerate(zip(seq_lens, prompt_lens)):
            rp["input_ids"][i, :n] = torch.arange(1, n + 1)
            rp["attention_mask"][i, :n] = 1
            rp["response_mask"][i, p:n] = True

        # Left-padded
        lp = self._make_left_padded_batch(seq_lens, prompt_lens, pad_to=max_len, K=K)

        rp_packed = pack_micro_batch(**rp)
        lp_packed = pack_micro_batch(**lp)

        assert rp_packed.response_mask.sum() == lp_packed.response_mask.sum()
        assert rp_packed.input_ids.shape == lp_packed.input_ids.shape
        assert rp_packed.cu_seq_lens.tolist() == lp_packed.cu_seq_lens.tolist()

    def test_left_padded_teacher_token_logps(self):
        """Teacher token logps must be extracted from real positions."""
        data = self._make_left_padded_batch(
            seq_lens=[5, 3],
            prompt_lens=[2, 1],
            pad_to=8,
        )
        teacher_token_logps = torch.zeros(2, 8)
        # Fill real positions with distinct values
        teacher_token_logps[0, 3:8] = torch.arange(1, 6, dtype=torch.float)  # seq 0
        teacher_token_logps[1, 5:8] = torch.arange(11, 14, dtype=torch.float)  # seq 1

        packed = pack_micro_batch(**data, teacher_token_logps=teacher_token_logps)
        assert packed.teacher_token_logps.shape == (1, 8)  # 5 + 3
        assert packed.teacher_token_logps[0, :5].tolist() == [1, 2, 3, 4, 5]
        assert packed.teacher_token_logps[0, 5:8].tolist() == [11, 12, 13]

    def test_mc_valid_mask_alignment(self):
        """MC validity mask must be packed with the same real-token alignment."""
        data = self._make_left_padded_batch(
            seq_lens=[5, 3],
            prompt_lens=[2, 1],
            pad_to=8,
        )
        mc_valid_mask = torch.zeros(2, 8, dtype=torch.bool)
        # Fill real-token spans with distinct valid/invalid patterns.
        mc_valid_mask[0, 3:8] = torch.tensor([False, False, True, False, True])
        mc_valid_mask[1, 5:8] = torch.tensor([False, True, False])

        packed = pack_micro_batch(**data, mc_valid_mask=mc_valid_mask)

        assert packed.mc_valid_mask is not None
        assert packed.mc_valid_mask.shape == (1, 8)
        assert packed.mc_valid_mask[0].tolist() == [
            False, False, True, False, True,
            False, True, False,
        ]


# ============================================================
# Test 2-5: Numerical equivalence (requires GPU + FA2)
# ============================================================

def _has_flash_attn():
    """Check if flash_attn is available."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
requires_fa2 = pytest.mark.skipif(
    not _has_flash_attn(), reason="flash_attn required"
)


def _make_small_model(device, attn_impl="flash_attention_2"):
    """Create a small Qwen3-like model for testing."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    # Make it tiny for fast tests
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 32
    config.vocab_size = 1000
    # Disable sliding window for basic tests
    if hasattr(config, "layer_types"):
        config.layer_types = ["full_attention"] * config.num_hidden_layers

    model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    return model


def _make_test_batch(B=4, max_len=128, K=4, vocab_size=1000, device="cuda"):
    """Create a test batch with variable-length sequences."""
    torch.manual_seed(42)
    # Variable lengths: some short, some long
    seq_lens = [max_len - 30, max_len, max_len - 60, max_len - 10][:B]
    prompt_lens = [20, 30, 15, 25][:B]

    input_ids = torch.randint(1, vocab_size, (B, max_len), device=device)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    teacher_topk_logps = torch.randn(B, max_len, K, device=device, dtype=torch.float32) * 0.1
    teacher_topk_indices = torch.randint(0, vocab_size, (B, max_len, K), device=device)
    response_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    prompt_lengths = torch.tensor(prompt_lens[:B], dtype=torch.long, device=device)

    for i in range(B):
        n = seq_lens[i]
        p = prompt_lens[i]
        attention_mask[i, :n] = 1
        response_mask[i, p:n] = True
        # Zero out padding in teacher data
        teacher_topk_logps[i, n:] = 0
        teacher_topk_indices[i, n:] = 0

    # Normalize teacher logps to be valid log-probs
    teacher_topk_logps = torch.nn.functional.log_softmax(teacher_topk_logps, dim=-1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_topk_logps": teacher_topk_logps,
        "teacher_topk_indices": teacher_topk_indices,
        "response_mask": response_mask,
        "prompt_lengths": prompt_lengths,
        "seq_lens": seq_lens,
        "prompt_lens": prompt_lens,
    }


@requires_cuda
@requires_fa2
class TestNumericalEquivalence:
    """Test that packed path produces same results as padded path."""

    def _forward_padded(self, model, input_ids, attention_mask, indices, chunk_size=1024):
        """Standard padded forward with chunked LM head."""
        from opd.loss.kl import chunked_lm_head_gather
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = outputs[0]
        return chunked_lm_head_gather(hidden, model.lm_head.weight, indices, chunk_size=chunk_size)

    def _forward_packed(self, model, packed, indices_packed, chunk_size=1024):
        """Packed forward with cu_seqlens + position_ids."""
        from opd.loss.kl import chunked_lm_head_gather
        outputs = model.model(
            input_ids=packed.input_ids,
            attention_mask=None,
            position_ids=packed.position_ids,
            use_cache=False,
            cu_seq_lens_q=packed.cu_seq_lens,
            cu_seq_lens_k=packed.cu_seq_lens,
            max_length_q=packed.max_seq_len,
            max_length_k=packed.max_seq_len,
        )
        hidden = outputs[0]
        return chunked_lm_head_gather(hidden, model.lm_head.weight, indices_packed, chunk_size=chunk_size)

    def test_sparse_kl_equivalence(self):
        """forward_kl, reverse_kl, skewed_kl produce same loss packed vs padded."""
        device = torch.device("cuda")
        model = _make_small_model(device)
        batch = _make_test_batch(B=4, max_len=64, K=4, vocab_size=1000, device=device)

        packed = pack_micro_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            teacher_topk_logps=batch["teacher_topk_logps"],
            teacher_topk_indices=batch["teacher_topk_indices"],
            response_mask=batch["response_mask"],
            prompt_lengths=batch["prompt_lengths"],
        )

        with torch.no_grad():
            # Padded path
            student_logps_padded = self._forward_padded(
                model, batch["input_ids"], batch["attention_mask"],
                batch["teacher_topk_indices"],
            )
            # Packed path
            student_logps_packed = self._forward_packed(
                model, packed, packed.teacher_topk_indices,
            )

        # Extract and compare at response positions only
        for mode_name, kl_fn in [
            ("forward_kl", lambda t, s: (torch.exp(t) * (t - s)).sum(-1)),
            ("reverse_kl", lambda t, s: (torch.exp(s) * (s - t)).sum(-1)),
        ]:
            # Padded loss
            per_token_padded = kl_fn(batch["teacher_topk_logps"], student_logps_padded)
            loss_padded = per_token_padded[batch["response_mask"]].mean()

            # Packed loss
            per_token_packed = kl_fn(packed.teacher_topk_logps, student_logps_packed)
            loss_packed = per_token_packed[packed.response_mask].mean()

            torch.testing.assert_close(
                loss_padded, loss_packed, atol=1e-2, rtol=1e-2,
                msg=f"{mode_name}: padded={loss_padded.item():.6f} vs packed={loss_packed.item():.6f}",
            )

    def test_token_level_kl_equivalence(self):
        """token_level_kl produces same loss packed vs padded."""
        device = torch.device("cuda")
        model = _make_small_model(device)
        batch = _make_test_batch(B=4, max_len=64, K=4, vocab_size=1000, device=device)

        # Create teacher_token_logps
        teacher_token_logps = torch.randn(4, 64, device=device, dtype=torch.float32) * 0.1
        for i, n in enumerate(batch["seq_lens"]):
            teacher_token_logps[i, n:] = 0

        packed = pack_micro_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            teacher_topk_logps=batch["teacher_topk_logps"],
            teacher_topk_indices=batch["teacher_topk_indices"],
            response_mask=batch["response_mask"],
            prompt_lengths=batch["prompt_lengths"],
            teacher_token_logps=teacher_token_logps,
        )

        from opd.loss.kl import chunked_lm_head_gather

        with torch.no_grad():
            # Padded: get per-token student logps
            outputs_pad = model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            target_ids_pad = batch["input_ids"][:, 1:].unsqueeze(-1)
            student_pad = chunked_lm_head_gather(
                outputs_pad[0][:, :-1], model.lm_head.weight, target_ids_pad
            ).squeeze(-1)

            # Packed: same
            outputs_pack = model.model(
                input_ids=packed.input_ids,
                attention_mask=None,
                position_ids=packed.position_ids,
                use_cache=False,
                cu_seq_lens_q=packed.cu_seq_lens,
                cu_seq_lens_k=packed.cu_seq_lens,
                max_length_q=packed.max_seq_len,
                max_length_k=packed.max_seq_len,
            )
            target_ids_pack = packed.input_ids[:, 1:].unsqueeze(-1)
            student_pack = chunked_lm_head_gather(
                outputs_pack[0][:, :-1], model.lm_head.weight, target_ids_pack
            ).squeeze(-1)

        # Compute token_level_kl loss
        t_pad = teacher_token_logps[:, :-1]
        tp_pad = torch.exp(t_pad).detach()
        kl_pad = tp_pad * (t_pad.detach() - student_pad)
        shifted_mask_pad = batch["response_mask"][:, 1:]
        loss_pad = kl_pad[shifted_mask_pad].mean()

        t_pack = packed.teacher_token_logps[:, :-1]
        tp_pack = torch.exp(t_pack).detach()
        kl_pack = tp_pack * (t_pack.detach() - student_pack)
        shifted_mask_pack = packed.response_mask[:, 1:]
        loss_pack = kl_pack[shifted_mask_pack].mean()

        torch.testing.assert_close(
            loss_pad, loss_pack, atol=1e-2, rtol=1e-2,
            msg=f"token_level_kl: padded={loss_pad.item():.6f} vs packed={loss_pack.item():.6f}",
        )

    def test_multi_sample_policy_gradient_kl_equivalence(self):
        """multi-sample PG-KL produces same loss packed vs padded."""
        device = torch.device("cuda")
        model = _make_small_model(device)
        batch = _make_test_batch(B=4, max_len=64, K=4, vocab_size=1000, device=device)
        n_samples = 3

        mc_sample_indices = torch.zeros(4, 64, n_samples, device=device, dtype=torch.long)
        mc_teacher_logprobs = torch.zeros(4, 64, n_samples, device=device, dtype=torch.float32)
        mc_old_logprobs = torch.zeros(4, 64, n_samples, device=device, dtype=torch.float32)
        for i, n in enumerate(batch["seq_lens"]):
            for pos in range(n):
                base = (i * 100 + pos * 7) % 1000
                mc_sample_indices[i, pos] = torch.tensor(
                    [(base + j) % 1000 for j in range(n_samples)], device=device
                )
                mc_teacher_logprobs[i, pos] = torch.tensor(
                    [-0.2 * (1 + i + pos + j) for j in range(n_samples)],
                    device=device,
                    dtype=torch.float32,
                )
                mc_old_logprobs[i, pos] = torch.tensor(
                    [-0.25 * (1 + i + pos + j) for j in range(n_samples)],
                    device=device,
                    dtype=torch.float32,
                )

        packed = pack_micro_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            teacher_topk_logps=batch["teacher_topk_logps"],
            teacher_topk_indices=batch["teacher_topk_indices"],
            response_mask=batch["response_mask"],
            prompt_lengths=batch["prompt_lengths"],
            mc_sample_indices=mc_sample_indices,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mc_old_logprobs=mc_old_logprobs,
        )

        from opd.loss.kl import KLConfig, compute_kl_loss

        with torch.no_grad():
            student_pad = self._forward_padded(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                mc_sample_indices,
            )
            student_pack = self._forward_packed(
                model,
                packed,
                packed.mc_sample_indices,
            )

        loss_pad = compute_kl_loss(
            student_mc_logprobs=student_pad,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mc_old_logprobs=mc_old_logprobs,
            mask=batch["response_mask"],
            kl_config=KLConfig(mode="multi_sample_policy_gradient_kl", pg_clip_eps=0.2),
        )
        loss_pack = compute_kl_loss(
            student_mc_logprobs=student_pack,
            mc_teacher_logprobs=packed.mc_teacher_logprobs,
            mc_old_logprobs=packed.mc_old_logprobs,
            mask=packed.response_mask,
            kl_config=KLConfig(mode="multi_sample_policy_gradient_kl", pg_clip_eps=0.2),
        )

        torch.testing.assert_close(
            loss_pad, loss_pack, atol=1e-2, rtol=1e-2,
            msg=(
                "multi_sample_policy_gradient_kl: "
                f"padded={loss_pad.item():.6f} vs packed={loss_pack.item():.6f}"
            ),
        )

    def test_gradient_equivalence(self):
        """Gradients match between packed and padded paths."""
        device = torch.device("cuda")

        # Need two identical models
        torch.manual_seed(123)
        model_pad = _make_small_model(device)
        torch.manual_seed(123)
        model_pack = _make_small_model(device)

        batch = _make_test_batch(B=3, max_len=48, K=4, vocab_size=1000, device=device)

        packed = pack_micro_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            teacher_topk_logps=batch["teacher_topk_logps"],
            teacher_topk_indices=batch["teacher_topk_indices"],
            response_mask=batch["response_mask"],
            prompt_lengths=batch["prompt_lengths"],
        )

        from opd.loss.kl import chunked_lm_head_gather

        # Padded forward + backward
        out_pad = model_pad.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logps_pad = chunked_lm_head_gather(
            out_pad[0], model_pad.lm_head.weight, batch["teacher_topk_indices"]
        )
        teacher_probs = torch.exp(batch["teacher_topk_logps"])
        kl_pad = (teacher_probs * (batch["teacher_topk_logps"] - logps_pad)).sum(-1)
        loss_pad = kl_pad[batch["response_mask"]].mean()
        loss_pad.backward()
        grad_pad = model_pad.lm_head.weight.grad.clone()

        # Packed forward + backward
        out_pack = model_pack.model(
            input_ids=packed.input_ids,
            attention_mask=None,
            position_ids=packed.position_ids,
            use_cache=False,
            cu_seq_lens_q=packed.cu_seq_lens,
            cu_seq_lens_k=packed.cu_seq_lens,
            max_length_q=packed.max_seq_len,
            max_length_k=packed.max_seq_len,
        )
        logps_pack = chunked_lm_head_gather(
            out_pack[0], model_pack.lm_head.weight, packed.teacher_topk_indices
        )
        teacher_probs_p = torch.exp(packed.teacher_topk_logps)
        kl_pack = (teacher_probs_p * (packed.teacher_topk_logps - logps_pack)).sum(-1)
        loss_pack = kl_pack[packed.response_mask].mean()
        loss_pack.backward()
        grad_pack = model_pack.lm_head.weight.grad.clone()

        # Compare
        torch.testing.assert_close(
            loss_pad, loss_pack, atol=1e-2, rtol=1e-2,
            msg=f"Loss: padded={loss_pad.item():.6f} vs packed={loss_pack.item():.6f}",
        )
        torch.testing.assert_close(
            grad_pad, grad_pack, atol=5e-2, rtol=5e-2,
            msg="Gradients differ between padded and packed paths",
        )


# ============================================================
# Test 6: Edge cases
# ============================================================

class TestEdgeCases:
    """Edge cases for packing — no GPU needed."""

    def _make_padded_batch(self, seq_lens, prompt_lens, pad_to=None, K=4):
        B = len(seq_lens)
        max_len = pad_to or max(seq_lens)
        input_ids = torch.zeros(B, max_len, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        teacher_topk_logps = torch.zeros(B, max_len, K)
        teacher_topk_indices = torch.zeros(B, max_len, K, dtype=torch.long)
        response_mask = torch.zeros(B, max_len, dtype=torch.bool)
        prompt_lengths = torch.tensor(prompt_lens, dtype=torch.long)
        for i, (n, p) in enumerate(zip(seq_lens, prompt_lens)):
            input_ids[i, :n] = torch.arange(1, n + 1) + i * 1000
            attention_mask[i, :n] = 1
            teacher_topk_logps[i, :n] = torch.randn(n, K)
            teacher_topk_indices[i, :n] = torch.randint(0, 100, (n, K))
            response_mask[i, p:n] = True
        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "teacher_topk_logps": teacher_topk_logps,
            "teacher_topk_indices": teacher_topk_indices,
            "response_mask": response_mask, "prompt_lengths": prompt_lengths,
        }

    def test_single_sequence(self):
        """B=1 — packing is trivial but must not break."""
        data = self._make_padded_batch([50], [10], pad_to=50)
        packed = pack_micro_batch(**data)
        assert packed.input_ids.shape == (1, 50)
        assert packed.cu_seq_lens.tolist() == [0, 50]
        assert packed.max_seq_len == 50

    def test_all_same_length(self):
        """All sequences same length — no padding waste but must be correct."""
        data = self._make_padded_batch([30, 30, 30], [10, 10, 10], pad_to=30)
        packed = pack_micro_batch(**data)
        assert packed.input_ids.shape == (1, 90)
        assert packed.cu_seq_lens.tolist() == [0, 30, 60, 90]

    def test_mixed_short_long(self):
        """Very short + very long sequences."""
        data = self._make_padded_batch([10, 500], [3, 50], pad_to=500)
        packed = pack_micro_batch(**data)
        assert packed.input_ids.shape == (1, 510)
        assert packed.max_seq_len == 500
        # Verify no padding waste: total packed = 510, not 1000
        assert packed.cu_seq_lens.tolist() == [0, 10, 510]

    def test_micro_batch_size_one(self):
        """Trivial micro-batch of size 1."""
        data = self._make_padded_batch([100], [20], pad_to=200)
        packed = pack_micro_batch(**data)
        # Should strip padding: 200 -> 100
        assert packed.input_ids.shape == (1, 100)


# ============================================================
# Test 8: Boundary token safety
# ============================================================

class TestBoundarySafety:
    """Verify boundary tokens between sequences don't leak into loss."""

    def test_boundary_excluded_by_shifted_mask(self):
        """At seq boundary, shifted response_mask is False (next seq's prompt)."""
        B = 2
        seq_lens = [5, 4]
        prompt_lens = [2, 2]
        max_len = 5

        input_ids = torch.zeros(B, max_len, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        teacher_topk_logps = torch.zeros(B, max_len, 4)
        teacher_topk_indices = torch.zeros(B, max_len, 4, dtype=torch.long)
        response_mask = torch.zeros(B, max_len, dtype=torch.bool)
        prompt_lengths = torch.tensor(prompt_lens, dtype=torch.long)

        for i, (n, p) in enumerate(zip(seq_lens, prompt_lens)):
            input_ids[i, :n] = torch.arange(1, n + 1)
            attention_mask[i, :n] = 1
            response_mask[i, p:n] = True

        packed = pack_micro_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            teacher_topk_logps=teacher_topk_logps,
            teacher_topk_indices=teacher_topk_indices,
            response_mask=response_mask,
            prompt_lengths=prompt_lengths,
        )

        # packed response_mask: [F,F,T,T,T, F,F,T,T]  (len 9)
        # shifted_mask = response_mask[:, 1:]: len 8
        shifted_mask = packed.response_mask[:, 1:]

        # Position 4 in packed is last token of seq 0 (response=True)
        # shifted_mask at position 4 checks packed.response_mask at position 5
        # Position 5 = first token of seq 1 = prompt = False
        # So shifted_mask[0, 4] = False — boundary excluded!
        assert shifted_mask[0, 4].item() == False  # noqa: E712

        # shifted_mask = original response_mask shifted left by 1:
        # original: [F,F,T,T,T,F,F,T,T] -> shifted: [F,T,T,T,F,F,T,T]
        # Position 4 (boundary) is correctly False because next token is seq1's prompt
        expected_shifted = [False, True, True, True, False, False, True, True]
        assert shifted_mask[0].tolist() == expected_shifted


def test_pack_micro_batch_carries_teacher_hidden_tensors_with_left_padding():
    input_ids = torch.tensor([[0, 0, 10, 11, 12], [0, 0, 0, 20, 21]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])
    response_mask = attention_mask.bool()
    prompt_lengths = torch.tensor([1, 1])
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    valid = torch.zeros(2, 5, dtype=torch.bool)
    valid[0, 2:4] = True
    valid[1, 3:4] = True

    packed = pack_micro_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        prompt_lengths=prompt_lengths,
        teacher_hidden_states=hidden,
        teacher_hidden_valid_mask=valid,
    )

    manual_hidden = torch.cat([hidden[0, 2:5], hidden[1, 3:5]], dim=0).unsqueeze(0)
    manual_valid = torch.tensor([[True, True, False, True, False]])
    torch.testing.assert_close(packed.teacher_hidden_states, manual_hidden)
    torch.testing.assert_close(packed.teacher_hidden_valid_mask, manual_valid)

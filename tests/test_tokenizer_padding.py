"""Regression tests for tokenizer padding_side="left" and pad_token setup.

Bug: OPDMode._get_tokenizer() and GRPOMode._get_tokenizer() previously
constructed their own tokenizers without setting padding_side="left" or
pad_token, causing right-padded prompts and broken teacher logprob alignment.

Fix: coordinator._init_tokenizer() is now passed via from_coordinator() so
the correctly-configured tokenizer flows through to all mode classes.

These tests prevent that regression.
"""

import pytest
import torch
import torch.nn.functional as F

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_tokenizer():
    """Load the real tokenizer from HuggingFace (requires network on first run)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return tok


def _make_gen_output(input_ids, attention_mask):
    """Build a minimal gen_output dict for pad_teacher."""
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _left_padded_batch(pad_id, seq_len, pad_lens):
    """Build a left-padded batch.

    pad_lens: list of ints — number of pad tokens at the start of each row.
    Returns (input_ids [bs, seq_len], attention_mask [bs, seq_len]).
    """
    bs = len(pad_lens)
    ids = torch.zeros(bs, seq_len, dtype=torch.long)
    mask = torch.zeros(bs, seq_len, dtype=torch.long)
    for i, pl in enumerate(pad_lens):
        # real tokens start at pl, values are 1...(seq_len-pl)
        ids[i, :pl] = pad_id
        ids[i, pl:] = torch.arange(1, seq_len - pl + 1, dtype=torch.long)
        mask[i, pl:] = 1
    return ids, mask


def _right_padded_batch(pad_id, seq_len, pad_lens):
    """Build a right-padded batch (pad tokens at the end of each row)."""
    bs = len(pad_lens)
    ids = torch.zeros(bs, seq_len, dtype=torch.long)
    mask = torch.zeros(bs, seq_len, dtype=torch.long)
    for i, pl in enumerate(pad_lens):
        real_len = seq_len - pl
        ids[i, :real_len] = torch.arange(1, real_len + 1, dtype=torch.long)
        ids[i, real_len:] = pad_id
        mask[i, :real_len] = 1
    return ids, mask


# ---------------------------------------------------------------------------
# 1. ConfigMixin._init_tokenizer()
# ---------------------------------------------------------------------------

class TestConfigMixinInitTokenizer:
    """_init_tokenizer() must set padding_side='left' and pad_token."""

    @pytest.mark.slow
    def test_padding_side_is_left(self):
        """_init_tokenizer() sets padding_side='left'."""
        from opd.coordinator.config_mixin import ConfigMixin
        from tests.conftest_opd import make_test_opd_config

        class _FakeMixin(ConfigMixin):
            def __init__(self):
                self._tokenizer = None
                self.opd_config = make_test_opd_config(model_path=MODEL_NAME)
                self.model_path = MODEL_NAME

        m = _FakeMixin()
        tok = m._init_tokenizer()
        assert tok.padding_side == "left", (
            f"Expected padding_side='left', got '{tok.padding_side}'. "
            "_init_tokenizer() must set padding_side='left'."
        )

    @pytest.mark.slow
    def test_pad_token_is_not_none(self):
        """_init_tokenizer() ensures pad_token is set (falls back to eos_token)."""
        from opd.coordinator.config_mixin import ConfigMixin
        from tests.conftest_opd import make_test_opd_config

        class _FakeMixin(ConfigMixin):
            def __init__(self):
                self._tokenizer = None
                self.opd_config = make_test_opd_config(model_path=MODEL_NAME)
                self.model_path = MODEL_NAME

        m = _FakeMixin()
        tok = m._init_tokenizer()
        assert tok.pad_token is not None, (
            "pad_token must not be None after _init_tokenizer(). "
            "Without a pad_token, tokenizer(padding='max_length') will fail."
        )

    @pytest.mark.slow
    def test_caches_tokenizer_on_second_call(self):
        """Calling _init_tokenizer() twice returns the same object."""
        from opd.coordinator.config_mixin import ConfigMixin
        from tests.conftest_opd import make_test_opd_config

        class _FakeMixin(ConfigMixin):
            def __init__(self):
                self._tokenizer = None
                self.opd_config = make_test_opd_config(model_path=MODEL_NAME)
                self.model_path = MODEL_NAME

        m = _FakeMixin()
        tok1 = m._init_tokenizer()
        tok2 = m._init_tokenizer()
        assert tok1 is tok2


# ---------------------------------------------------------------------------
# 2. OPDMode._get_tokenizer()
# ---------------------------------------------------------------------------

class TestOPDModeGetTokenizer:
    """OPDMode._get_tokenizer() must return a left-padded tokenizer."""

    @pytest.mark.slow
    def test_returns_tokenizer_with_left_padding(self):
        """When tokenizer is passed with padding_side='left', _get_tokenizer() returns it."""
        from opd.coordinator.opd_mode import OPDMode

        tok = _real_tokenizer()
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        mode = OPDMode(
            rollout_proxy=None, teacher_client=None, trainer_proxy=None,
            tracer=None,
            config={"teacher": {}, "training": {}, "data": {}},
            tokenizer=tok,
        )
        result = mode._get_tokenizer()
        assert result.padding_side == "left"

    def test_raises_when_tokenizer_is_none(self):
        """_get_tokenizer() raises RuntimeError when no tokenizer was injected."""
        from opd.coordinator.opd_mode import OPDMode

        mode = OPDMode(
            rollout_proxy=None, teacher_client=None, trainer_proxy=None,
            tracer=None,
            config={"teacher": {}, "training": {}, "data": {}},
            tokenizer=None,
        )
        with pytest.raises(RuntimeError, match="OPDMode requires a tokenizer"):
            mode._get_tokenizer()

    @pytest.mark.slow
    def test_tokenizer_padding_side_not_mutated_by_get_tokenizer(self):
        """_get_tokenizer() is a passthrough — it must not change padding_side."""
        from opd.coordinator.opd_mode import OPDMode

        tok = _real_tokenizer()
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        mode = OPDMode(
            rollout_proxy=None, teacher_client=None, trainer_proxy=None,
            tracer=None,
            config={"teacher": {}, "training": {}, "data": {}},
            tokenizer=tok,
        )
        _ = mode._get_tokenizer()
        assert tok.padding_side == "left"


# ---------------------------------------------------------------------------
# 3. GRPOMode._get_tokenizer()
# ---------------------------------------------------------------------------

class TestGRPOModeGetTokenizer:
    """GRPOMode._get_tokenizer() must return a left-padded tokenizer."""

    @pytest.mark.slow
    def test_returns_tokenizer_with_left_padding(self):
        """When tokenizer is passed with padding_side='left', _get_tokenizer() returns it."""
        from opd.coordinator.grpo_mode import GRPOMode

        tok = _real_tokenizer()
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        mode = GRPOMode(
            rollout_proxy=None, teacher_client=None, trainer_proxy=None,
            tracer=None,
            config={"teacher": {}, "training": {}, "data": {}},
            grpo_group_size=4,
            grpo_clip_eps=0.2,
            grpo_kl_beta=0.0,
            reward_fn_name="correctness",
            reward_fn=lambda *a, **kw: torch.zeros(1),
            tokenizer=tok,
        )
        result = mode._get_tokenizer()
        assert result.padding_side == "left"

    def test_raises_when_tokenizer_is_none(self):
        """_get_tokenizer() raises RuntimeError when no tokenizer was injected."""
        from opd.coordinator.grpo_mode import GRPOMode

        mode = GRPOMode(
            rollout_proxy=None, teacher_client=None, trainer_proxy=None,
            tracer=None,
            config={"teacher": {}, "training": {}, "data": {}},
            grpo_group_size=4,
            grpo_clip_eps=0.2,
            grpo_kl_beta=0.0,
            reward_fn_name="correctness",
            reward_fn=lambda *a, **kw: torch.zeros(1),
            tokenizer=None,
        )
        with pytest.raises(RuntimeError, match="GRPOMode requires a tokenizer"):
            mode._get_tokenizer()

    @pytest.mark.slow
    def test_tokenizer_padding_side_not_mutated_by_get_tokenizer(self):
        """_get_tokenizer() is a passthrough — it must not change padding_side."""
        from opd.coordinator.grpo_mode import GRPOMode

        tok = _real_tokenizer()
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        mode = GRPOMode(
            rollout_proxy=None, teacher_client=None, trainer_proxy=None,
            tracer=None,
            config={"teacher": {}, "training": {}, "data": {}},
            grpo_group_size=4,
            grpo_clip_eps=0.2,
            grpo_kl_beta=0.0,
            reward_fn_name="correctness",
            reward_fn=lambda *a, **kw: torch.zeros(1),
            tokenizer=tok,
        )
        _ = mode._get_tokenizer()
        assert tok.padding_side == "left"


# ---------------------------------------------------------------------------
# 4. collate_fn produces LEFT-padded batches
# ---------------------------------------------------------------------------

class TestCollateFnLeftPadding:
    """collate_fn from opd/data/prompt.py preserves left-padding from tokenizer.

    The tokenizer pads on the LEFT (pad tokens at the start), so real tokens
    appear at the END of each row. collate_fn must not reorder tokens.
    """

    @pytest.mark.slow
    def test_real_tokenizer_pads_on_left(self):
        """The tokenizer (configured by _init_tokenizer) pads on the left."""
        tok = _real_tokenizer()
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        short_text = "Hi"
        long_text = "This is a much longer sentence to pad against."
        encoded_short = tok(
            short_text,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # For left-padding: padding tokens appear at the START
        # The first token of the shorter string should be a pad token
        pad_id = tok.pad_token_id
        # At least one leading pad token for the short sequence
        assert encoded_short["input_ids"][0, 0].item() == pad_id, (
            "Left-padding: first token of short sequence should be pad_token. "
            "Got a non-pad token — tokenizer may be right-padding."
        )
        # Attention mask should be 0 at start (pad positions)
        assert encoded_short["attention_mask"][0, 0].item() == 0

    @pytest.mark.slow
    def test_collate_fn_preserves_left_pad_pattern(self):
        """collate_fn stacks tensors without reordering — left-pad layout survives."""
        from opd.data.prompt import collate_fn

        tok = _real_tokenizer()
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        texts = ["Hi", "This is a much longer sentence to pad against."]
        max_len = 32
        items = []
        for t in texts:
            enc = tok(t, max_length=max_len, padding="max_length",
                      truncation=True, return_tensors="pt")
            items.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

        batch = collate_fn(items)

        # First item (shorter): leading positions should be pad
        pad_id = tok.pad_token_id
        assert batch["input_ids"][0, 0].item() == pad_id, (
            "collate_fn must preserve left-padding. "
            "First position of short sequence should be pad_token."
        )
        assert batch["attention_mask"][0, 0].item() == 0

    def test_collate_fn_left_pad_real_tokens_at_end(self):
        """Construct left-padded items manually; verify real tokens are at the end."""
        from opd.data.prompt import collate_fn

        # Simulate left-padded items: pad=0 at start, real tokens at end
        pad_id = 0
        items = [
            {
                "input_ids": torch.tensor([pad_id, pad_id, 10, 20, 30]),
                "attention_mask": torch.tensor([0, 0, 1, 1, 1]),
            },
            {
                "input_ids": torch.tensor([pad_id, 40, 50, 60, 70]),
                "attention_mask": torch.tensor([0, 1, 1, 1, 1]),
            },
        ]
        batch = collate_fn(items)

        # First row: pad at positions 0,1; real tokens at 2,3,4
        assert batch["input_ids"][0, 0].item() == pad_id
        assert batch["input_ids"][0, 1].item() == pad_id
        assert batch["input_ids"][0, 2].item() == 10
        assert batch["input_ids"][0, 4].item() == 30

        # Second row: pad at position 0; real tokens at 1-4
        assert batch["input_ids"][1, 0].item() == pad_id
        assert batch["input_ids"][1, 1].item() == 40

    def test_collate_fn_does_not_right_pad(self):
        """collate_fn must not move padding to the right side.

        Right-padding would mean real tokens are at the start and pad tokens
        at the end. For left-padded inputs, real tokens must remain at the end.
        """
        from opd.data.prompt import collate_fn

        pad_id = 0
        real_token = 99
        # A clearly left-padded item: [0, 0, 0, real_token]
        items = [
            {
                "input_ids": torch.tensor([pad_id, pad_id, pad_id, real_token]),
                "attention_mask": torch.tensor([0, 0, 0, 1]),
            }
        ]
        batch = collate_fn(items)

        # Real token must still be at the end (position 3), not the start
        assert batch["input_ids"][0, -1].item() == real_token, (
            "Real token must remain at the END (left-padding preserved). "
            "collate_fn appears to have reordered tokens."
        )
        assert batch["input_ids"][0, 0].item() == pad_id


# ---------------------------------------------------------------------------
# 5. pad_teacher: left-padding vs right-padding correctness
# ---------------------------------------------------------------------------

class TestPadTeacherPaddingSide:
    """pad_teacher maps teacher logprobs to non-padding positions using attention_mask.

    LEFT-padding: pad tokens at the start, real tokens at the end.
    RIGHT-padding: real tokens at the start, pad tokens at the end.

    pad_teacher uses cumsum(attention_mask) to place teacher logprob j at the
    j-th valid (non-padding) position. This is correct for EITHER layout as
    long as the mask correctly marks real tokens. However, when the pipeline
    sends full_token_lists (prompt+response) to the teacher, left-padding means
    the teacher processes tokens in the right order (prompt then response).

    These tests verify:
    - With left-padding, teacher logprob[0] lands at the first non-pad position
      (i.e., the first real token — prompt start).
    - With right-padding, teacher logprob[0] would also land at the first
      non-pad position, but that would be the first real token at position 0,
      creating an inconsistency when the student has a different pad layout.
    - Switching from left-padding to right-padding (same token sequence)
      produces DIFFERENT placement of teacher logprobs in the padded tensor,
      demonstrating that wrong padding_side breaks alignment.
    """

    @staticmethod
    def _make_teacher_logps(n_tokens, topk, seed=0):
        g = torch.Generator()
        g.manual_seed(seed)
        logps = F.log_softmax(torch.randn(n_tokens, topk, generator=g), dim=-1)
        indices = torch.randint(0, 1000, (n_tokens, topk),
                                dtype=torch.int32, generator=g)
        return logps, indices

    def test_left_padded_teacher_logps_at_first_real_position(self):
        """For left-padded input, teacher logprob[0] goes to the first real token."""
        from opd.data.batch_utils import pad_teacher

        seq_len, pad_len, topk = 8, 3, 4
        n_real = seq_len - pad_len

        # Left-padded: mask is [0,0,0,1,1,1,1,1]
        ids, mask = _left_padded_batch(pad_id=0, seq_len=seq_len, pad_lens=[pad_len])
        gen_out = _make_gen_output(ids, mask)

        logps, indices = self._make_teacher_logps(n_real, topk, seed=10)
        result = pad_teacher(gen_out, [logps], [indices])

        # Teacher position 0 should land at index pad_len (first real token)
        assert result["teacher_valid_mask"][0, pad_len].item() is True
        # Positions before pad_len should NOT be valid
        for pos in range(pad_len):
            assert result["teacher_valid_mask"][0, pos].item() is False

    def test_right_padded_teacher_logps_at_first_real_position(self):
        """For right-padded input, teacher logprob[0] goes to position 0 (first real token)."""
        from opd.data.batch_utils import pad_teacher

        seq_len, pad_len, topk = 8, 3, 4
        n_real = seq_len - pad_len

        # Right-padded: mask is [1,1,1,1,1,0,0,0]
        ids, mask = _right_padded_batch(pad_id=0, seq_len=seq_len, pad_lens=[pad_len])
        gen_out = _make_gen_output(ids, mask)

        logps, indices = self._make_teacher_logps(n_real, topk, seed=10)
        result = pad_teacher(gen_out, [logps], [indices])

        # Teacher position 0 lands at position 0 (right-padded: real tokens first)
        assert result["teacher_valid_mask"][0, 0].item() is True
        # Padding positions at the end should NOT be valid
        for pos in range(seq_len - pad_len, seq_len):
            assert result["teacher_valid_mask"][0, pos].item() is False

    def test_left_vs_right_padding_produces_different_valid_mask(self):
        """Left-padded and right-padded inputs yield different valid_mask placements.

        This demonstrates that using the wrong padding_side breaks alignment:
        the same teacher logprobs end up at different positions in the batch tensor.
        """
        from opd.data.batch_utils import pad_teacher

        seq_len, pad_len, topk = 10, 4, 3
        n_real = seq_len - pad_len

        logps, indices = self._make_teacher_logps(n_real, topk, seed=20)

        # Left-padded
        ids_l, mask_l = _left_padded_batch(pad_id=0, seq_len=seq_len, pad_lens=[pad_len])
        result_left = pad_teacher(_make_gen_output(ids_l, mask_l), [logps], [indices])

        # Right-padded (same real tokens, just different padding side)
        ids_r, mask_r = _right_padded_batch(pad_id=0, seq_len=seq_len, pad_lens=[pad_len])
        result_right = pad_teacher(_make_gen_output(ids_r, mask_r), [logps], [indices])

        left_mask = result_left["teacher_valid_mask"][0]
        right_mask = result_right["teacher_valid_mask"][0]

        # The valid masks must differ — same logprobs placed at different positions
        assert not torch.equal(left_mask, right_mask), (
            "Left-padded and right-padded inputs should produce different valid_mask "
            "layouts. If they are equal, pad_teacher is ignoring the mask layout, "
            "which would mean padding_side has no effect on alignment."
        )

    def test_left_padded_logps_values_match_at_real_positions(self):
        """Teacher logprob[i] must exactly match the value at the i-th real position."""
        from opd.data.batch_utils import pad_teacher

        seq_len, pad_len, topk = 8, 2, 3
        n_real = seq_len - pad_len

        ids, mask = _left_padded_batch(pad_id=0, seq_len=seq_len, pad_lens=[pad_len])
        gen_out = _make_gen_output(ids, mask)

        logps, indices = self._make_teacher_logps(n_real, topk, seed=30)
        result = pad_teacher(gen_out, [logps], [indices])

        # For each real position (pad_len .. seq_len-1), teacher logp[j] = logps[j]
        for j in range(n_real):
            pos = pad_len + j
            torch.testing.assert_close(
                result["teacher_topk_logps"][0, pos],
                logps[j],
                msg=f"Teacher logps mismatch at real position {pos} (teacher_idx={j})",
            )

    def test_right_padding_misaligns_logps_relative_to_left_padding(self):
        """Right-padded input places logprob[0] at position 0, not at pad_len.

        This shows the concrete alignment error: if the student prompt was
        left-padded but teacher logprobs are mapped as if right-padded, the
        logprobs shift left by pad_len positions, corrupting the KL loss.
        """
        from opd.data.batch_utils import pad_teacher

        seq_len, pad_len, topk = 8, 3, 2
        n_real = seq_len - pad_len

        logps, indices = self._make_teacher_logps(n_real, topk, seed=40)

        # Correct (left-padded): logp[0] at position pad_len
        ids_l, mask_l = _left_padded_batch(pad_id=0, seq_len=seq_len, pad_lens=[pad_len])
        result_left = pad_teacher(_make_gen_output(ids_l, mask_l), [logps], [indices])
        left_first_valid_pos = result_left["teacher_valid_mask"][0].nonzero()[0].item()
        assert left_first_valid_pos == pad_len

        # Wrong (right-padded): logp[0] at position 0 — shifted left by pad_len
        ids_r, mask_r = _right_padded_batch(pad_id=0, seq_len=seq_len, pad_lens=[pad_len])
        result_right = pad_teacher(_make_gen_output(ids_r, mask_r), [logps], [indices])
        right_first_valid_pos = result_right["teacher_valid_mask"][0].nonzero()[0].item()
        assert right_first_valid_pos == 0

        # The shift is exactly pad_len — proving right-padding is wrong
        assert left_first_valid_pos - right_first_valid_pos == pad_len, (
            f"Expected a shift of {pad_len} between left and right padding. "
            f"Got left={left_first_valid_pos}, right={right_first_valid_pos}."
        )

    def test_batch_with_mixed_pad_lengths_left_padded(self):
        """Batch with unequal pad lengths: each row's valid mask starts at its own pad_len."""
        from opd.data.batch_utils import pad_teacher

        seq_len, topk = 10, 3
        pad_lens = [0, 2, 5]  # different padding for each sample

        ids, mask = _left_padded_batch(pad_id=0, seq_len=seq_len, pad_lens=pad_lens)
        gen_out = _make_gen_output(ids, mask)

        all_logps = []
        all_indices = []
        for pl in pad_lens:
            n_real = seq_len - pl
            lp, idx = self._make_teacher_logps(n_real, topk, seed=pl)
            all_logps.append(lp)
            all_indices.append(idx)

        result = pad_teacher(gen_out, all_logps, all_indices)

        for i, pl in enumerate(pad_lens):
            # First valid position must be at pl.
            first_valid = result["teacher_valid_mask"][i].nonzero()[0].item()
            assert first_valid == pl, (
                f"Row {i}: expected first valid position at {pl}, got {first_valid}. "
                "Left-padding must place real tokens starting at pad_len."
            )

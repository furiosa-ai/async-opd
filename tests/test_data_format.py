"""Tests for data.py: prompt formatting, datasets, and collation."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock


# --------------- Dataset reference aliases ---------------

class TestDatasetReferenceAliases:
    """Test eval dataset alias and HF split resolution without network access."""

    def test_eval_aliases_resolve_to_hf_refs(self):
        from opd.data.prompt import resolve_dataset_ref

        assert resolve_dataset_ref("AIME25") == "hf:yentinglin/aime_2025"
        assert resolve_dataset_ref("AIME_25") == "hf:yentinglin/aime_2025"
        assert resolve_dataset_ref("AIME_2025") == "hf:yentinglin/aime_2025"
        assert resolve_dataset_ref("AMC") == "hf:math-ai/amc23::test"
        assert resolve_dataset_ref("AMC23") == "hf:math-ai/amc23::test"
        assert resolve_dataset_ref("AMC_23") == "hf:math-ai/amc23::test"
        assert resolve_dataset_ref("MATH-500") == "hf:HuggingFaceH4/MATH-500::test"
        assert resolve_dataset_ref("HMMT Feb25") == "hf:MathArena/hmmt_feb_2025::train"
        assert resolve_dataset_ref("HMMT25 February") == "hf:MathArena/hmmt_feb_2025::train"
        assert resolve_dataset_ref("HMMT Nov25") == "hf:MathArena/hmmt_nov_2025::train"
        assert resolve_dataset_ref("HMMT25 November") == "hf:MathArena/hmmt_nov_2025::train"

    def test_known_hf_refs_default_to_available_split(self):
        from opd.data.prompt import parse_hf_dataset_ref

        assert parse_hf_dataset_ref("hf:yentinglin/aime_2025") == (
            "yentinglin/aime_2025", None, "train")
        assert parse_hf_dataset_ref("hf:math-ai/aime25") == ("math-ai/aime25", None, "test")
        assert parse_hf_dataset_ref("hf:math-ai/amc23") == ("math-ai/amc23", None, "test")
        assert parse_hf_dataset_ref("hf:HuggingFaceH4/MATH-500") == (
            "HuggingFaceH4/MATH-500", None, "test")
        assert parse_hf_dataset_ref("hf:MathArena/hmmt_feb_2025") == (
            "MathArena/hmmt_feb_2025", None, "train")
        assert parse_hf_dataset_ref("hf:MathArena/hmmt_nov_2025") == (
            "MathArena/hmmt_nov_2025", None, "train")

    def test_hf_ref_supports_no_config_split_syntax(self):
        from opd.data.prompt import parse_hf_dataset_ref

        assert parse_hf_dataset_ref("hf:HuggingFaceH4/MATH-500::test") == (
            "HuggingFaceH4/MATH-500", None, "test")
        assert parse_hf_dataset_ref("hf:HuggingFaceH4/MATH-500:test") == (
            "HuggingFaceH4/MATH-500", None, "test")


# --------------- format_prompt ---------------

class TestFormatPrompt:
    """Test format_prompt with various input types and options."""

    @pytest.fixture
    def mock_tokenizer(self):
        tok = MagicMock()
        tok.apply_chat_template = MagicMock(
            side_effect=lambda msgs, **kw: f"<chat>{msgs[0]['content']}</chat>"
        )
        return tok

    def test_string_with_template(self, mock_tokenizer):
        from opd.data.prompt import format_prompt
        result = format_prompt("What is 2+2?", mock_tokenizer,
                                prompt_template="{problem}\nShow your work.")
        assert "What is 2+2?" in result
        assert "Show your work." in result
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_string_without_template(self, mock_tokenizer):
        from opd.data.prompt import format_prompt
        result = format_prompt("raw text", mock_tokenizer, prompt_template=None)
        assert result == "raw text"
        mock_tokenizer.apply_chat_template.assert_not_called()

    def test_chat_list_input(self, mock_tokenizer):
        from opd.data.prompt import format_prompt
        msgs = [{"role": "user", "content": "Hello"}]
        result = format_prompt(msgs, mock_tokenizer)
        mock_tokenizer.apply_chat_template.assert_called_once()
        call_args = mock_tokenizer.apply_chat_template.call_args
        assert call_args[0][0] == msgs

    def test_numpy_array_input(self, mock_tokenizer):
        from opd.data.prompt import format_prompt
        msgs = np.array([{"role": "user", "content": "Hello"}])
        result = format_prompt(msgs, mock_tokenizer)
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_enable_thinking_passed(self, mock_tokenizer):
        from opd.data.prompt import format_prompt
        msgs = [{"role": "user", "content": "test"}]
        format_prompt(msgs, mock_tokenizer, enable_thinking=False)
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is False

    def test_enable_thinking_not_passed_when_none(self, mock_tokenizer):
        from opd.data.prompt import format_prompt
        msgs = [{"role": "user", "content": "test"}]
        format_prompt(msgs, mock_tokenizer, enable_thinking=None)
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert "enable_thinking" not in call_kwargs

    def test_template_with_boxed(self, mock_tokenizer):
        """Verify the G-OPD prompt template works with format()."""
        from opd.data.prompt import format_prompt
        template = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        result = format_prompt("Solve x+1=3", mock_tokenizer, prompt_template=template)
        assert "Solve x+1=3" in result
        assert "\\boxed{}" in result


# --------------- PromptDataset ---------------

class TestPromptDataset:
    """Test PromptDataset loading and tokenization."""

    @pytest.fixture
    def mock_tokenizer(self):
        tok = MagicMock()
        tok.apply_chat_template = MagicMock(
            side_effect=lambda msgs, **kw: f"formatted: {msgs[0]['content']}"
        )
        tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        }
        tok.padding_side = "left"
        tok.pad_token = "<pad>"
        return tok

    def test_getitem_returns_tensors(self, mock_tokenizer, tmp_path):
        import pandas as pd
        from opd.data.prompt import PromptDataset

        df = pd.DataFrame({"problem": ["What is 1+1?", "What is 2+2?"]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        ds = PromptDataset(str(path), mock_tokenizer, max_prompt_length=128,
                           prompt_key="problem", prompt_template="{problem}")
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["input_ids"].dim() == 1

    def test_len(self, mock_tokenizer, tmp_path):
        import pandas as pd
        from opd.data.prompt import PromptDataset

        df = pd.DataFrame({"problem": ["a", "b", "c"]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        ds = PromptDataset(str(path), mock_tokenizer, max_prompt_length=128,
                           prompt_key="problem")
        assert len(ds) == 3


# --------------- ValDataset ---------------

class TestValDataset:
    """Test ValDataset answer extraction logic."""

    @pytest.fixture
    def mock_tokenizer(self):
        tok = MagicMock()
        tok.apply_chat_template = MagicMock(
            side_effect=lambda msgs, **kw: f"formatted: {msgs[0]['content']}"
        )
        tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        return tok

    def test_answer_key_auto_answer_column(self, mock_tokenizer, tmp_path):
        import pandas as pd
        from opd.data.prompt import ValDataset

        df = pd.DataFrame({"problem": ["p1", "p2"], "answer": ["42", "7"]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        ds = ValDataset(str(path), mock_tokenizer, max_prompt_length=128,
                        prompt_key="problem", answer_key="auto")
        assert ds.ground_truths == ["42", "7"]

    def test_answer_key_auto_reward_model(self, mock_tokenizer, tmp_path):
        import json
        import pandas as pd
        from opd.data.prompt import ValDataset

        df = pd.DataFrame({
            "problem": ["p1"],
            "reward_model": [json.dumps({"ground_truth": "18"})],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        ds = ValDataset(str(path), mock_tokenizer, max_prompt_length=128,
                        prompt_key="problem", answer_key="auto")
        assert ds.ground_truths == ["18"]

    def test_answer_key_explicit(self, mock_tokenizer, tmp_path):
        import pandas as pd
        from opd.data.prompt import ValDataset

        df = pd.DataFrame({"problem": ["p1"], "solution": ["abc"]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        ds = ValDataset(str(path), mock_tokenizer, max_prompt_length=128,
                        prompt_key="problem", answer_key="solution")
        assert ds.ground_truths == ["abc"]

    def test_answer_key_auto_no_column_raises(self, mock_tokenizer, tmp_path):
        import pandas as pd
        from opd.data.prompt import ValDataset

        df = pd.DataFrame({"problem": ["p1"], "other": ["x"]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        with pytest.raises(ValueError, match="Cannot auto-detect"):
            ValDataset(str(path), mock_tokenizer, max_prompt_length=128,
                       prompt_key="problem", answer_key="auto")

    def test_reward_model_dict_input(self, mock_tokenizer, tmp_path):
        """reward_model column as actual dict (not JSON string)."""
        import pandas as pd
        from opd.data.prompt import ValDataset

        df = pd.DataFrame({
            "problem": ["p1"],
            "reward_model": [{"ground_truth": "99"}],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        ds = ValDataset(str(path), mock_tokenizer, max_prompt_length=128,
                        prompt_key="problem", answer_key="reward_model")
        assert ds.ground_truths == ["99"]

    def test_getitem_includes_ground_truth(self, mock_tokenizer, tmp_path):
        import pandas as pd
        from opd.data.prompt import ValDataset

        df = pd.DataFrame({"problem": ["p1"], "answer": ["42"]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        ds = ValDataset(str(path), mock_tokenizer, max_prompt_length=128,
                        prompt_key="problem")
        item = ds[0]
        assert item["ground_truth"] == "42"


# --------------- collate functions ---------------

class TestCollate:
    def test_collate_fn(self):
        from opd.data.prompt import collate_fn
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])},
            {"input_ids": torch.tensor([4, 5, 6]), "attention_mask": torch.tensor([1, 1, 0])},
        ]
        out = collate_fn(batch)
        assert out["input_ids"].shape == (2, 3)
        assert out["attention_mask"].shape == (2, 3)

    def test_val_collate_fn(self):
        from opd.data.prompt import val_collate_fn
        batch = [
            {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]),
             "ground_truth": "42"},
            {"input_ids": torch.tensor([3, 4]), "attention_mask": torch.tensor([1, 0]),
             "ground_truth": "7"},
        ]
        out = val_collate_fn(batch)
        assert out["input_ids"].shape == (2, 2)
        assert out["ground_truth"] == ["42", "7"]


# --------------- load_dataframe ---------------

class TestLoadDataframe:
    def test_local_parquet(self, tmp_path):
        import pandas as pd
        from opd.data.prompt import load_dataframe

        df = pd.DataFrame({"Problem": ["test"], "Answer": ["42"]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        loaded = load_dataframe(str(path))
        # Columns should be lowercased
        assert "problem" in loaded.columns
        assert "answer" in loaded.columns

    def test_columns_lowercased(self, tmp_path):
        import pandas as pd
        from opd.data.prompt import load_dataframe

        df = pd.DataFrame({"MyCol": [1], "UPPER": [2]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        loaded = load_dataframe(str(path))
        assert list(loaded.columns) == ["mycol", "upper"]

"""Dataset loading for on-policy distillation."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


HF_DATASET_ALIASES = {
    # Fast math eval suite aliases.
    "amc23": "hf:math-ai/amc23::test",
    "amc": "hf:math-ai/amc23::test",
    "amc_23": "hf:math-ai/amc23::test",
    "aime25": "hf:yentinglin/aime_2025",
    "aime_25": "hf:yentinglin/aime_2025",
    "aime_2025": "hf:yentinglin/aime_2025",
    "math500": "hf:HuggingFaceH4/MATH-500::test",
    "math_500": "hf:HuggingFaceH4/MATH-500::test",
    "math-500": "hf:HuggingFaceH4/MATH-500::test",
    "hmmt_feb25": "hf:MathArena/hmmt_feb_2025::train",
    "hmmt_feb_25": "hf:MathArena/hmmt_feb_2025::train",
    "hmmt_feb_2025": "hf:MathArena/hmmt_feb_2025::train",
    "hmmt_february_2025": "hf:MathArena/hmmt_feb_2025::train",
    "hmmt25_feb": "hf:MathArena/hmmt_feb_2025::train",
    "hmmt25_february": "hf:MathArena/hmmt_feb_2025::train",
    "hmmt_nov25": "hf:MathArena/hmmt_nov_2025::train",
    "hmmt_nov_25": "hf:MathArena/hmmt_nov_2025::train",
    "hmmt_nov_2025": "hf:MathArena/hmmt_nov_2025::train",
    "hmmt_november_2025": "hf:MathArena/hmmt_nov_2025::train",
    "hmmt25_nov": "hf:MathArena/hmmt_nov_2025::train",
    "hmmt25_november": "hf:MathArena/hmmt_nov_2025::train",
}

HF_DATASET_DEFAULT_SPLITS = {
    "HuggingFaceH4/MATH-500": "test",
    "yentinglin/aime_2025": "train",
    "math-ai/amc23": "test",
    "math-ai/aime25": "test",
    "MathArena/hmmt_feb_2025": "train",
    "MathArena/hmmt_nov_2025": "train",
}

_COMMON_SPLIT_NAMES = {"train", "test", "validation", "val", "dev"}


def _dataset_alias_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def resolve_dataset_ref(path_or_hf_name: str) -> str:
    """Resolve short eval dataset aliases to concrete dataset references."""
    return HF_DATASET_ALIASES.get(_dataset_alias_key(path_or_hf_name), path_or_hf_name)


def parse_hf_dataset_ref(path_or_hf_name: str, split="train"):
    """Parse ``hf:<dataset>[:<config>[:<split>]]`` into load_dataset args.

    Also supports ``hf:<dataset>::<split>`` for datasets without a config and
    defaults common benchmark-only datasets to their available split.
    """
    parts = path_or_hf_name[3:].split(":")
    dataset_name = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    if len(parts) > 2:
        split = parts[2]
    elif config in _COMMON_SPLIT_NAMES and dataset_name in HF_DATASET_DEFAULT_SPLITS:
        split = config
        config = None
    else:
        split = HF_DATASET_DEFAULT_SPLITS.get(dataset_name, split)
    return dataset_name, config, split


def load_dataframe(path_or_hf_name, split="train"):
    """Load data from a local parquet file or a HuggingFace dataset name.

    Supports:
      - Local parquet file:  "data/gsm8k/train.parquet"
      - Short eval alias:    "AMC23", "HMMT Feb25", "HMMT Nov25", "MATH-500"
      - HuggingFace dataset: "hf:agentica-org/DeepScaleR-Preview-Dataset"
                             "hf:yentinglin/aime_2025"
                             "hf:yentinglin/aime_2025:part1"  (specific config)
                             "hf:openai/gsm8k:main:test"      (config + split override)
                             "hf:HuggingFaceH4/MATH-500"      (split defaults to test)
    """
    path_or_hf_name = resolve_dataset_ref(path_or_hf_name)
    if path_or_hf_name.startswith("hf:"):
        from datasets import load_dataset
        dataset_name, config, split = parse_hf_dataset_ref(path_or_hf_name, split=split)
        ds = load_dataset(dataset_name, config, split=split)
        df = ds.to_pandas()
    else:
        df = pd.read_parquet(path_or_hf_name)
    # Normalize column names to lowercase for consistent access
    df.columns = [c.lower() for c in df.columns]
    return df


def _normalize_chat_messages(raw):
    """Return chat messages as plain dicts, or None when raw is not chat-like."""
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, list):
        return None
    messages = []
    for msg in raw:
        if not isinstance(msg, dict):
            return None
        messages.append(dict(msg))
    return messages


def extract_last_user_content(raw):
    """Extract the final user message content from a chat-format prompt."""
    messages = _normalize_chat_messages(raw)
    if messages is None:
        raise ValueError(
            "prompt_source='last_user_content' requires chat/list prompt rows"
        )
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content is None:
                return ""
            return str(content)
    raise ValueError(
        "prompt_source='last_user_content' could not find a user message"
    )


def _apply_dataframe_filter(df, filter_key=None, filter_value=None):
    """Apply an opt-in equality filter to a normalized dataframe."""
    if filter_key is None and filter_value is None:
        return df
    if filter_key is None or filter_value is None:
        raise ValueError("filter_key and filter_value must be set together")
    key = str(filter_key).lower()
    if key not in df.columns:
        raise ValueError(
            f"filter_key {filter_key!r} (normalized to {key!r}) not found; "
            f"available columns: {list(df.columns)}"
        )
    filtered = df[df[key] == filter_value].reset_index(drop=True)
    if len(filtered) == 0:
        raise ValueError(
            f"filter {key} == {filter_value!r} matched 0 rows"
        )
    return filtered


def _select_prompt_source(raw_prompts, prompt_source):
    if prompt_source == "raw":
        return list(raw_prompts)
    if prompt_source == "last_user_content":
        return [extract_last_user_content(raw) for raw in raw_prompts]
    raise ValueError(
        "prompt_source must be one of {'raw', 'last_user_content'}, "
        f"got {prompt_source!r}"
    )


def format_prompt(raw, tokenizer, prompt_template=None, enable_thinking=None):
    """Convert a raw prompt to text, applying chat template or prompt_template.

    Args:
        raw: Either a list of message dicts (chat format) or a plain string.
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        prompt_template: Optional template string with {problem} placeholder.
        enable_thinking: If set, passed to apply_chat_template (Qwen3 thinking mode).
    """
    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = enable_thinking

    msgs = _normalize_chat_messages(raw)
    if msgs is not None:
        return tokenizer.apply_chat_template(msgs, **chat_kwargs)
    if prompt_template:
        assert "{problem}" in prompt_template, \
            f"prompt_template must contain '{{problem}}' placeholder, got: {prompt_template!r}"
        text = prompt_template.replace("{problem}", str(raw))
        # Unescape doubled braces (legacy from .format()-style templates)
        text = text.replace("{{", "{").replace("}}", "}")
        msgs = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(msgs, **chat_kwargs)
    return raw


class PromptDataset(Dataset):
    """Loads prompts from parquet files or HuggingFace datasets."""

    def __init__(self, path, tokenizer, max_prompt_length,
                 prompt_key="prompt", prompt_template=None, enable_thinking=None,
                 solution_key=None, prompt_source="raw", filter_key=None,
                 filter_value=None):
        self.df = _apply_dataframe_filter(
            load_dataframe(path), filter_key=filter_key, filter_value=filter_value,
        )
        self.raw_prompts = _select_prompt_source(
            self.df[prompt_key].tolist(), prompt_source=prompt_source,
        )
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_template = prompt_template
        self.enable_thinking = enable_thinking
        if solution_key:
            if solution_key in self.df.columns:
                self.solutions = [str(s) for s in self.df[solution_key].tolist()]
            elif "answer" in self.df.columns:
                self.solutions = [str(s) for s in self.df["answer"].tolist()]
            else:
                self.solutions = None
        else:
            self.solutions = None

    def __len__(self):
        return len(self.raw_prompts)

    def __getitem__(self, idx):
        text = format_prompt(self.raw_prompts[idx], self.tokenizer,
                              self.prompt_template,
                              enable_thinking=self.enable_thinking)
        encoded = self.tokenizer(
            text,
            max_length=self.max_prompt_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.solutions is not None:
            item["solution"] = self.solutions[idx]
            # Pass raw problem text for OPSD teacher prompt construction
            raw = self.raw_prompts[idx]
            item["problem_text"] = str(raw) if not isinstance(raw, (list,)) else str(raw)
        return item


def collate_fn(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }
    if "solution" in batch[0]:
        out["solutions"] = [b["solution"] for b in batch]
        out["problem_texts"] = [b["problem_text"] for b in batch]
    return out


class ValDataset(Dataset):
    """Loads prompts and ground-truth answers for evaluation.

    Supports multiple ground-truth formats:
      - reward_model column with {"ground_truth": "..."} (GSM8K format)
      - answer column (DeepScaler, AIME format)
    """

    def __init__(self, path, tokenizer, max_prompt_length,
                 prompt_key="prompt", answer_key="auto", prompt_template=None,
                 enable_thinking=None):
        self.df = load_dataframe(path)
        # Auto-detect prompt column if specified key doesn't exist
        if prompt_key not in self.df.columns:
            for fallback in ("problem", "prompt", "question", "input"):
                if fallback in self.df.columns:
                    prompt_key = fallback
                    break
        self.raw_prompts = self.df[prompt_key].tolist()
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_template = prompt_template
        self.enable_thinking = enable_thinking

        self.ground_truths = self._extract_ground_truths(answer_key)

    def _extract_ground_truths(self, answer_key):
        """Extract ground truth answers from the dataframe."""
        if answer_key == "auto":
            # Auto-detect: try answer column first, then reward_model
            if "answer" in self.df.columns:
                return [str(a).strip() for a in self.df["answer"].tolist()]
            elif "reward_model" in self.df.columns:
                return self._extract_from_reward_model()
            else:
                raise ValueError(
                    f"Cannot auto-detect answer column. "
                    f"Available columns: {list(self.df.columns)}. "
                    f"Set answer_key explicitly."
                )
        elif answer_key == "reward_model":
            return self._extract_from_reward_model()
        else:
            return [str(a).strip() for a in self.df[answer_key].tolist()]

    def _extract_from_reward_model(self):
        """Extract ground truth from reward_model column (GSM8K format)."""
        import json
        reward_col = self.df["reward_model"].tolist()
        results = []
        for rm in reward_col:
            if isinstance(rm, dict):
                results.append(str(rm.get("ground_truth", "")))
            elif isinstance(rm, str):
                try:
                    d = json.loads(rm)
                    results.append(str(d.get("ground_truth", "")))
                except (json.JSONDecodeError, AttributeError):
                    results.append("")
            else:
                results.append("")
        return results

    def __len__(self):
        return len(self.raw_prompts)

    def __getitem__(self, idx):
        text = format_prompt(self.raw_prompts[idx], self.tokenizer,
                              self.prompt_template,
                              enable_thinking=self.enable_thinking)
        encoded = self.tokenizer(
            text,
            max_length=self.max_prompt_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "ground_truth": self.ground_truths[idx],
        }


def val_collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "ground_truth": [b["ground_truth"] for b in batch],
    }

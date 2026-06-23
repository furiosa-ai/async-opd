"""SFT dataset for supervised fine-tuning with (prompt, completion) pairs."""

import pickle

import torch
from torch.utils.data import Dataset

from opd.data.prompt import format_prompt, load_dataframe
from opd.utils.config import warn_unsafe_opt_in


class SFTDataset(Dataset):
    """Loads (prompt, completion) pairs for supervised fine-tuning.

    Tokenizes prompt and completion separately to determine the boundary,
    then concatenates into a single sequence with a response_mask that is
    0 for prompt tokens and 1 for completion tokens.

    Does NOT pad — padding is handled by sft_collate_fn or sequence packing.

    Optionally loads teacher top-k logprobs from parquet columns
    ``teacher_topk_logps`` and ``teacher_topk_indices``. The historical format
    stores pickle-bytes tensors of shape [completion_len, K], which is unsafe
    for untrusted datasets; loading those columns now requires the explicit
    ``allow_pickle_teacher_logits`` compatibility opt-in. When present and
    allowed, __getitem__ returns extra keys ``teacher_topk_logps`` and
    ``teacher_topk_indices``.
    """

    def __init__(
        self,
        path,
        tokenizer,
        max_prompt_length,
        max_response_length,
        prompt_key="prompt",
        completion_key="completion",
        prompt_template=None,
        enable_thinking=None,
        allow_pickle_teacher_logits=False,
    ):
        self.df = load_dataframe(path)
        self.raw_prompts = self.df[prompt_key].tolist()
        self.completions = self.df[completion_key].tolist()
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.prompt_template = prompt_template
        self.enable_thinking = enable_thinking
        self.allow_pickle_teacher_logits = bool(allow_pickle_teacher_logits)

        self.has_teacher_logits = (
            "teacher_topk_logps" in self.df.columns
            and "teacher_topk_indices" in self.df.columns
        )
        if self.has_teacher_logits:
            if not self.allow_pickle_teacher_logits:
                raise ValueError(
                    "SFT teacher logit columns use pickle serialization and are disabled "
                    "by default for public-safe dataset loading. Set "
                    "data.allow_pickle_teacher_logits=true (or pass "
                    "allow_pickle_teacher_logits=True) only for trusted internal data."
                )
            warn_unsafe_opt_in(
                True,
                context="SFTDataset",
                detail=(
                    "loading pickled teacher logit tensors from the dataset. "
                    "Only use trusted datasets; pickle can execute arbitrary code."
                ),
            )

    def __len__(self):
        return len(self.raw_prompts)

    def __getitem__(self, idx):
        prompt_text = format_prompt(
            self.raw_prompts[idx],
            self.tokenizer,
            self.prompt_template,
            enable_thinking=self.enable_thinking,
        )
        completion_text = str(self.completions[idx])

        # Tokenize separately — no padding, no special tokens on completion
        prompt_enc = self.tokenizer(
            prompt_text,
            max_length=self.max_prompt_length,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        completion_enc = self.tokenizer(
            completion_text,
            max_length=self.max_response_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            return_tensors="pt",
        )

        prompt_ids = prompt_enc["input_ids"].squeeze(0)       # [P]
        completion_ids = completion_enc["input_ids"].squeeze(0)  # [C]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=0)  # [P+C]
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        # response_mask: 0 for prompt tokens, 1 for completion tokens
        response_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        response_mask[len(prompt_ids):] = True

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "prompt_lengths": torch.tensor(len(prompt_ids), dtype=torch.long),
        }

        if self.has_teacher_logits:
            row = self.df.iloc[idx]
            logps: torch.Tensor = pickle.loads(row["teacher_topk_logps"])    # [C_full, K]
            indices: torch.Tensor = pickle.loads(row["teacher_topk_indices"])  # [C_full, K]

            # Truncate to the same length as the tokenized completion (may be shorter
            # if completion was truncated to max_response_length).
            c = len(completion_ids)
            logps = logps[:c]      # [C, K]
            indices = indices[:c]  # [C, K]

            item["teacher_topk_logps"] = logps.to(torch.float32)
            item["teacher_topk_indices"] = indices.to(torch.int32)

        return item


def make_sft_collate_fn(pad_token_id: int = 0):
    """Create an SFT collate function with the correct pad token ID."""
    def _collate(batch):
        """Pad a batch of SFT samples to the same length using LEFT padding.

        Matches the OPD pipeline convention of left-padding input_ids.
        response_mask is right-extended with False for pad positions.

        When items contain ``teacher_topk_logps`` / ``teacher_topk_indices``
        (present for every item or absent for every item), these are placed
        at the completion positions of a ``[B, max_len, K]`` zero-padded
        tensor, and a ``teacher_valid_mask`` of shape ``[B, max_len]`` is
        added to the batch indicating which positions have valid teacher data.
        """
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = pad_token_id
        has_teacher = "teacher_topk_logps" in batch[0]

        input_ids_list = []
        attention_mask_list = []
        response_mask_list = []
        prompt_lengths_list = []

        if has_teacher:
            K = batch[0]["teacher_topk_logps"].shape[1]
            teacher_logps_list = []
            teacher_indices_list = []
            teacher_valid_mask_list = []

        for b in batch:
            seq_len = len(b["input_ids"])
            pad_len = max_len - seq_len

            # Left-pad input_ids and attention_mask
            input_ids_list.append(
                torch.cat([
                    torch.full((pad_len,), pad_id, dtype=torch.long),
                    b["input_ids"],
                ])
            )
            attention_mask_list.append(
                torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    b["attention_mask"],
                ])
            )
            # Left-extend response_mask with False (pad positions are not completion)
            response_mask_list.append(
                torch.cat([
                    torch.zeros(pad_len, dtype=torch.bool),
                    b["response_mask"],
                ])
            )
            prompt_lengths_list.append(b["prompt_lengths"])

            if has_teacher:
                # Teacher tensors cover completion tokens only ([C, K]).
                # Place them in a [max_len, K] zero tensor at the positions
                # where response_mask is True (after left-padding).
                c = b["teacher_topk_logps"].shape[0]
                prompt_len = b["prompt_lengths"].item()

                logps_padded = torch.zeros(max_len, K, dtype=torch.float32)
                indices_padded = torch.zeros(max_len, K, dtype=torch.int32)
                valid_mask = torch.zeros(max_len, dtype=torch.bool)

                # Completion positions in the padded sequence start at:
                #   pad_len + prompt_len
                start = pad_len + prompt_len
                logps_padded[start:start + c] = b["teacher_topk_logps"]
                indices_padded[start:start + c] = b["teacher_topk_indices"]
                valid_mask[start:start + c] = True

                teacher_logps_list.append(logps_padded)
                teacher_indices_list.append(indices_padded)
                teacher_valid_mask_list.append(valid_mask)

        out = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "response_mask": torch.stack(response_mask_list),
            "prompt_lengths": torch.stack(prompt_lengths_list),
        }

        if has_teacher:
            out["teacher_topk_logps"] = torch.stack(teacher_logps_list)      # [B, max_len, K]
            out["teacher_topk_indices"] = torch.stack(teacher_indices_list)  # [B, max_len, K]
            out["teacher_valid_mask"] = torch.stack(teacher_valid_mask_list)  # [B, max_len]

        return out
    return _collate

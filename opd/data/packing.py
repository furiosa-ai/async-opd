"""Sequence packing utilities for eliminating padding waste in training.

Packs multiple variable-length sequences into a single dense tensor [1, T]
with cu_seqlens for FlashAttention varlen dispatch and position_ids with
per-sequence resets for correct RoPE computation.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PackedBatch:
    """Packed micro-batch with metadata for FA varlen dispatch."""
    input_ids: torch.Tensor           # [1, T]
    position_ids: torch.Tensor        # [1, T]
    cu_seq_lens: torch.Tensor         # [B+1] int32
    max_seq_len: int                  # max individual sequence length
    response_mask: torch.Tensor       # [1, T] bool
    seq_lens: torch.Tensor            # [B] int
    prompt_lens: torch.Tensor         # [B] int
    teacher_topk_logps: Optional[torch.Tensor] = None  # [1, T, K]
    teacher_topk_indices: Optional[torch.Tensor] = None  # [1, T, K]
    support_student_old_logps: Optional[torch.Tensor] = None  # [1, T, K]
    teacher_token_logps: Optional[torch.Tensor] = None  # [1, T]
    student_logprobs: Optional[torch.Tensor] = None     # [1, T]
    mc_sample_indices: Optional[torch.Tensor] = None    # [1, T, N]
    mc_teacher_logprobs: Optional[torch.Tensor] = None  # [1, T, N]
    mc_old_logprobs: Optional[torch.Tensor] = None      # [1, T, N]
    mc_valid_mask: Optional[torch.Tensor] = None        # [1, T] bool
    teacher_hidden_states: Optional[torch.Tensor] = None  # [1, T, H]
    teacher_hidden_valid_mask: Optional[torch.Tensor] = None  # [1, T] bool


def pack_micro_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    teacher_topk_logps: Optional[torch.Tensor] = None,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    support_student_old_logps: Optional[torch.Tensor] = None,
    teacher_token_logps: Optional[torch.Tensor] = None,
    student_logprobs: Optional[torch.Tensor] = None,
    mc_sample_indices: Optional[torch.Tensor] = None,
    mc_teacher_logprobs: Optional[torch.Tensor] = None,
    mc_old_logprobs: Optional[torch.Tensor] = None,
    mc_valid_mask: Optional[torch.Tensor] = None,
    teacher_hidden_states: Optional[torch.Tensor] = None,
    teacher_hidden_valid_mask: Optional[torch.Tensor] = None,
    max_prompt: Optional[int] = None,
) -> PackedBatch:
    """Pack a padded micro-batch into a dense packed representation.

    Args:
        input_ids: [B, S] padded input token IDs
        attention_mask: [B, S] binary mask (1 = real token, 0 = padding)
        teacher_topk_logps: [B, S, K] teacher top-k log-probs
        teacher_topk_indices: [B, S, K] teacher top-k token indices
        response_mask: [B, S] bool mask (True at response positions)
        prompt_lengths: [B] per-sequence prompt lengths
        teacher_token_logps: [B, S] optional teacher token-level log-probs
        student_logprobs: [B, resp_len] optional student old log-probs (for PG-KL)
        mc_sample_indices: [B, S, N] optional multi-sample token indices
        mc_teacher_logprobs: [B, S, N] optional multi-sample teacher log-probs
        mc_old_logprobs: [B, S, N] optional multi-sample old-policy log-probs
        mc_valid_mask: [B, S] optional valid-position mask for multi-sample rows
        teacher_hidden_states: [B, S, H] optional lazy hidden-recompute payload
        teacher_hidden_valid_mask: [B, S] optional validity mask for hidden payload
        max_prompt: int, fixed prompt offset used in padded layout (for PG-KL alignment)

    Returns:
        PackedBatch with all tensors concatenated into dense [1, T] layout
    """
    B = input_ids.size(0)
    device = input_ids.device

    # Compute actual lengths per sequence
    seq_lens = attention_mask.sum(dim=1).int()  # [B]

    # Extract non-padding tokens for each sequence
    packed_ids = []
    packed_pos = []
    packed_teacher_logps = [] if teacher_topk_logps is not None else None
    packed_teacher_idx = [] if teacher_topk_indices is not None else None
    packed_support_student_old = [] if support_student_old_logps is not None else None
    packed_resp_mask = []
    packed_teacher_token = [] if teacher_token_logps is not None else None
    packed_student = [] if student_logprobs is not None else None
    packed_mc_indices = [] if mc_sample_indices is not None else None
    packed_mc_teacher = [] if mc_teacher_logprobs is not None else None
    packed_mc_old = [] if mc_old_logprobs is not None else None
    packed_mc_valid = [] if mc_valid_mask is not None else None
    packed_teacher_hidden = [] if teacher_hidden_states is not None else None
    packed_teacher_hidden_valid = [] if teacher_hidden_valid_mask is not None else None

    for i in range(B):
        n = seq_lens[i].item()
        # Extract non-padding tokens using boolean mask (handles left padding)
        mask = attention_mask[i].bool()
        packed_ids.append(input_ids[i][mask])
        packed_pos.append(torch.arange(n, device=device, dtype=torch.long))
        if packed_teacher_logps is not None:
            packed_teacher_logps.append(teacher_topk_logps[i][mask])
        if packed_teacher_idx is not None:
            packed_teacher_idx.append(teacher_topk_indices[i][mask])
        if packed_support_student_old is not None:
            packed_support_student_old.append(support_student_old_logps[i][mask])
        packed_resp_mask.append(response_mask[i][mask])

        if teacher_token_logps is not None:
            packed_teacher_token.append(teacher_token_logps[i][mask])
        if packed_mc_indices is not None:
            packed_mc_indices.append(mc_sample_indices[i][mask])
        if packed_mc_teacher is not None:
            packed_mc_teacher.append(mc_teacher_logprobs[i][mask])
        if packed_mc_old is not None:
            packed_mc_old.append(mc_old_logprobs[i][mask])
        if packed_mc_valid is not None:
            if mc_valid_mask.shape != input_ids.shape:
                raise ValueError(
                    f"mc_valid_mask must have shape {tuple(input_ids.shape)}, "
                    f"got {tuple(mc_valid_mask.shape)}"
                )
            packed_mc_valid.append(mc_valid_mask[i][mask])
        if packed_teacher_hidden is not None:
            if teacher_hidden_states.size(0) != B or teacher_hidden_states.size(1) != input_ids.size(1):
                raise ValueError(
                    f"teacher_hidden_states must have shape [B,S,H] aligned to input_ids; "
                    f"got {tuple(teacher_hidden_states.shape)} for input_ids {tuple(input_ids.shape)}"
                )
            packed_teacher_hidden.append(teacher_hidden_states[i][mask])
        if packed_teacher_hidden_valid is not None:
            if teacher_hidden_valid_mask.shape != input_ids.shape:
                raise ValueError(
                    f"teacher_hidden_valid_mask must have shape {tuple(input_ids.shape)}, "
                    f"got {tuple(teacher_hidden_valid_mask.shape)}"
                )
            packed_teacher_hidden_valid.append(teacher_hidden_valid_mask[i][mask])

        if student_logprobs is not None:
            # student_logprobs is [B, resp_len] — response tokens only.
            # Place at correct response positions in a per-sequence [n] tensor.
            p_len = prompt_lengths[i].item()
            resp_len = n - p_len
            per_seq = torch.zeros(n, device=device, dtype=student_logprobs.dtype)
            if resp_len > 0 and student_logprobs.size(1) > 0:
                avail = min(resp_len, student_logprobs.size(1))
                per_seq[p_len:p_len + avail] = student_logprobs[i, :avail]
            packed_student.append(per_seq)

    # Concatenate into dense tensors
    total_tokens = seq_lens.sum().item()
    packed_input_ids = torch.cat(packed_ids).unsqueeze(0)          # [1, T]
    packed_position_ids = torch.cat(packed_pos).unsqueeze(0)       # [1, T]
    packed_teacher_topk_logps = (
        torch.cat(packed_teacher_logps).unsqueeze(0) if packed_teacher_logps else None)
    packed_teacher_topk_indices = (
        torch.cat(packed_teacher_idx).unsqueeze(0) if packed_teacher_idx else None)
    packed_support_student_old_logps = (
        torch.cat(packed_support_student_old).unsqueeze(0) if packed_support_student_old else None)
    packed_response_mask = torch.cat(packed_resp_mask).unsqueeze(0)  # [1, T]

    # Build cu_seq_lens: [0, len_0, len_0+len_1, ...]
    cu_seq_lens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_seq_lens[1:] = seq_lens.cumsum(0)

    max_seq_len = seq_lens.max().item()

    result = PackedBatch(
        input_ids=packed_input_ids,
        position_ids=packed_position_ids,
        cu_seq_lens=cu_seq_lens,
        max_seq_len=max_seq_len,
        teacher_topk_logps=packed_teacher_topk_logps,
        teacher_topk_indices=packed_teacher_topk_indices,
        support_student_old_logps=packed_support_student_old_logps,
        response_mask=packed_response_mask,
        seq_lens=seq_lens,
        prompt_lens=prompt_lengths.int(),
    )

    if packed_teacher_token is not None:
        result.teacher_token_logps = torch.cat(packed_teacher_token).unsqueeze(0)

    if packed_student is not None:
        result.student_logprobs = torch.cat(packed_student).unsqueeze(0)

    if packed_mc_indices is not None:
        result.mc_sample_indices = torch.cat(packed_mc_indices).unsqueeze(0)

    if packed_mc_teacher is not None:
        result.mc_teacher_logprobs = torch.cat(packed_mc_teacher).unsqueeze(0)

    if packed_mc_old is not None:
        result.mc_old_logprobs = torch.cat(packed_mc_old).unsqueeze(0)

    if packed_mc_valid is not None:
        result.mc_valid_mask = torch.cat(packed_mc_valid).bool().unsqueeze(0)

    if packed_teacher_hidden is not None:
        result.teacher_hidden_states = torch.cat(packed_teacher_hidden, dim=0).unsqueeze(0)

    if packed_teacher_hidden_valid is not None:
        result.teacher_hidden_valid_mask = torch.cat(packed_teacher_hidden_valid).bool().unsqueeze(0)

    return result


def pack_sft_micro_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> PackedBatch:
    """Pack an SFT micro-batch into dense representation (no teacher tensors).

    Args:
        input_ids: [B, S] padded input token IDs
        attention_mask: [B, S] binary mask (1 = real token, 0 = padding)
        response_mask: [B, S] bool mask (True at completion positions)
        prompt_lengths: [B] per-sequence prompt lengths

    Returns:
        PackedBatch with teacher fields = None
    """
    B = input_ids.size(0)
    device = input_ids.device

    seq_lens = attention_mask.sum(dim=1).int()  # [B]

    packed_ids = []
    packed_pos = []
    packed_resp_mask = []

    for i in range(B):
        n = seq_lens[i].item()
        mask = attention_mask[i].bool()
        packed_ids.append(input_ids[i][mask])
        packed_pos.append(torch.arange(n, device=device, dtype=torch.long))
        packed_resp_mask.append(response_mask[i][mask])

    packed_input_ids = torch.cat(packed_ids).unsqueeze(0)
    packed_position_ids = torch.cat(packed_pos).unsqueeze(0)
    packed_response_mask = torch.cat(packed_resp_mask).unsqueeze(0)

    cu_seq_lens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_seq_lens[1:] = seq_lens.cumsum(0)

    return PackedBatch(
        input_ids=packed_input_ids,
        position_ids=packed_position_ids,
        cu_seq_lens=cu_seq_lens,
        max_seq_len=seq_lens.max().item(),
        response_mask=packed_response_mask,
        seq_lens=seq_lens,
        prompt_lens=prompt_lengths.int(),
    )

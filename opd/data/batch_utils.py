"""Batch assembly and padding utilities for the distillation pipeline."""

import torch


def pad_teacher(gen_output, all_logps, all_indices, all_token_logps=None):
    """Pad variable-length teacher logprobs into fixed [bs, seq_len, topk] tensors.

    Maps teacher position j to the j-th valid (non-padding) position in each
    sample's attention mask using cumsum-based vectorized indexing.

    Args:
        gen_output: dict with 'input_ids' [bs, seq_len] and 'attention_mask' [bs, seq_len]
        all_logps: list of [t_len, topk] tensors (per-sample teacher log-probs)
        all_indices: list of [t_len, topk] tensors (per-sample teacher token indices)
        all_token_logps: optional list of [t_len] tensors (per-sample token-level logprobs)

    Returns:
        dict with padded teacher_topk_logps, teacher_topk_indices, teacher_valid_mask
        (and optionally teacher_token_logps), or None if all_logps is empty.
    """
    if not all_logps:
        return None
    ids = gen_output["input_ids"]
    mask = gen_output["attention_mask"]
    bs, seq_len = ids.shape
    topk = all_logps[0].size(-1)
    has_token_logps = all_token_logps and len(all_token_logps) > 0

    # Vectorized padding: use cumsum on mask to map teacher position j
    # to the j-th valid position in each sample's mask.
    mask_bool = mask.bool()  # [bs, seq_len]
    # cumsum gives 1-based index of each valid position within its row
    cumsum = mask_bool.long().cumsum(dim=1)  # [bs, seq_len]

    t_lens = torch.tensor([lp.size(0) for lp in all_logps])
    n_valids = mask.sum(dim=1)
    c_lens = torch.minimum(t_lens, n_valids)  # [bs]

    # A position (i, j) should receive teacher data if:
    #   mask[i,j] == 1  AND  cumsum[i,j] <= c_lens[i]
    # cumsum[i,j] is the 1-based index, so teacher_pos = cumsum[i,j] - 1
    eligible = mask_bool & (cumsum <= c_lens.unsqueeze(1))  # [bs, seq_len]

    # Stack teacher outputs into dense [bs, max_c, ...] tensors
    max_c = int(c_lens.max().item()) if bs > 0 else 0
    stacked_logps = torch.zeros(bs, max_c, topk, dtype=torch.float32)
    stacked_idx = torch.zeros(bs, max_c, topk, dtype=torch.int32)
    for i in range(bs):
        cl = c_lens[i].item()
        stacked_logps[i, :cl] = all_logps[i][:cl]
        stacked_idx[i, :cl] = all_indices[i][:cl]
    if has_token_logps:
        stacked_tlps = torch.full((bs, max_c), -1e10, dtype=torch.float32)
        for i in range(bs):
            cl = min(c_lens[i].item(), all_token_logps[i].size(0))
            stacked_tlps[i, :cl] = all_token_logps[i][:cl]

    # Scatter using eligible positions
    b_all, s_all = eligible.nonzero(as_tuple=True)
    teacher_pos = cumsum[b_all, s_all] - 1  # 0-based teacher index

    p_logps = torch.zeros(bs, seq_len, topk, dtype=torch.float32)
    p_idx = torch.zeros(bs, seq_len, topk, dtype=torch.int32)
    valid_mask = torch.zeros(bs, seq_len, dtype=torch.bool)

    p_logps[b_all, s_all] = stacked_logps[b_all, teacher_pos]
    p_idx[b_all, s_all] = stacked_idx[b_all, teacher_pos]
    valid_mask[b_all, s_all] = True

    out = {"teacher_topk_logps": p_logps, "teacher_topk_indices": p_idx,
           "teacher_valid_mask": valid_mask}
    if has_token_logps:
        p_token_logps = torch.full((bs, seq_len), -1e10, dtype=torch.float32)
        p_token_logps[b_all, s_all] = stacked_tlps[b_all, teacher_pos]
        out["teacher_token_logps"] = p_token_logps
    return out


def adapt_response_support(
    gen_output,
    query_indices_response,
    teacher_query_logprobs_response,
    student_query_logprobs_response=None,
):
    """Adapt response-local support tensors into trainer-facing full-sequence tensors."""
    if not query_indices_response or not teacher_query_logprobs_response:
        return None

    ids = gen_output["input_ids"]
    responses = gen_output["responses"]
    response_lengths = gen_output["response_lengths"]
    bs, seq_len = ids.shape
    max_response_length = responses.size(1)
    max_prompt = seq_len - max_response_length
    topk = 0
    for q_idx in query_indices_response:
        if q_idx.dim() == 2:
            topk = q_idx.size(-1)
            break

    support_idx = torch.zeros(bs, seq_len, topk, dtype=torch.int32)
    support_logps = torch.zeros(bs, seq_len, topk, dtype=torch.float32)
    support_student_old_logps = (
        torch.zeros(bs, seq_len, topk, dtype=torch.float32)
        if student_query_logprobs_response is not None
        else None
    )
    support_valid = torch.zeros(bs, seq_len, dtype=torch.bool)

    for i in range(bs):
        q_idx = query_indices_response[i]
        q_logps = teacher_query_logprobs_response[i]
        usable = min(int(response_lengths[i].item()), q_idx.size(0), q_logps.size(0))
        if support_student_old_logps is not None:
            q_old = student_query_logprobs_response[i]
            usable = min(usable, q_old.size(0))
        if usable <= 0:
            continue
        start = max_prompt
        end = start + usable
        support_idx[i, start:end] = q_idx[:usable].to(torch.int32)
        support_logps[i, start:end] = q_logps[:usable].to(torch.float32)
        if support_student_old_logps is not None:
            support_student_old_logps[i, start:end] = q_old[:usable].to(torch.float32)
        support_valid[i, start:end] = True

    out = {
        "support_topk_logps": support_logps,
        "support_topk_indices": support_idx,
        "support_valid_mask": support_valid,
    }
    if support_student_old_logps is not None:
        out["support_student_old_logps"] = support_student_old_logps
    return out


def adapt_mc_response_samples(
    gen_output,
    mc_query_indices_response,
    teacher_query_logprobs_response,
    mc_query_old_logprobs_response=None,
):
    """Adapt response-local MC tensors into trainer-facing full-sequence tensors."""
    if (not mc_query_indices_response
            or not teacher_query_logprobs_response):
        return None

    ids = gen_output["input_ids"]
    responses = gen_output["responses"]
    response_lengths = gen_output["response_lengths"]
    bs, seq_len = ids.shape
    max_response_length = responses.size(1)
    max_prompt = seq_len - max_response_length
    n_total_samples = 0
    for q_idx in mc_query_indices_response:
        if q_idx.dim() == 2:
            n_total_samples = q_idx.size(-1)
            break

    mc_idx = torch.zeros(bs, seq_len, n_total_samples, dtype=torch.int32)
    mc_teacher = torch.zeros(bs, seq_len, n_total_samples, dtype=torch.float32)
    mc_old = (
        torch.zeros(bs, seq_len, n_total_samples, dtype=torch.float32)
        if mc_query_old_logprobs_response is not None
        else None
    )
    mc_valid = torch.zeros(bs, seq_len, dtype=torch.bool)

    for i in range(bs):
        q_idx = mc_query_indices_response[i]
        q_teacher = teacher_query_logprobs_response[i]
        usable = min(
            int(response_lengths[i].item()),
            q_idx.size(0),
            q_teacher.size(0),
        )
        if mc_old is not None:
            q_old = mc_query_old_logprobs_response[i]
            usable = min(usable, q_old.size(0))
        if usable <= 0:
            continue
        start = max_prompt
        end = start + usable
        mc_idx[i, start:end] = q_idx[:usable].to(torch.int32)
        mc_teacher[i, start:end] = q_teacher[:usable].to(torch.float32)
        if mc_old is not None:
            mc_old[i, start:end] = q_old[:usable].to(torch.float32)
        mc_valid[i, start:end] = True

    out = {
        "mc_sample_indices": mc_idx,
        "mc_teacher_logprobs": mc_teacher,
        "mc_valid_mask": mc_valid,
    }
    if mc_old is not None:
        out["mc_old_logprobs"] = mc_old
    return out


def stack_gen_output(samples):
    """Stack rollout sample dicts into the generation half of a train batch."""
    gen_keys = ("input_ids", "attention_mask", "responses", "prompt_lengths",
                "response_lengths")
    gen_output = {}
    for key in gen_keys:
        vals = [s[key] for s in samples if key in s]
        if not vals:
            continue
        if isinstance(vals[0], torch.Tensor):
            gen_output[key] = torch.cat(vals, dim=0)
        elif isinstance(vals[0], list):
            gen_output[key] = sum(vals, [])
        else:
            gen_output[key] = vals

    for key in ("full_token_lists",):
        vals = [s[key] for s in samples if key in s]
        if vals:
            gen_output[key] = sum(vals, [])

    if samples and "student_logprobs" in samples[0]:
        gen_output["student_logprobs"] = torch.cat(
            [s["student_logprobs"] for s in samples], dim=0)

    for key in (
        "query_indices_response", "query_logprobs_response",
        "mc_query_indices_response", "mc_query_old_logprobs_response",
    ):
        if samples and key in samples[0]:
            vals = [s[key] for s in samples]
            gen_output[key] = sum(vals, []) if isinstance(vals[0], list) else vals

    gen_output["weight_version"] = [s.get("weight_version", 0) for s in samples]
    gen_output["worker_id"] = [s.get("worker_id", 0) for s in samples]
    gen_output["sample_seq_ids"] = [s.get("sample_seq_id") for s in samples]
    return gen_output


def split_gen_teacher(samples, pad_teacher_fn=None):
    """Stack individual sample dicts into gen_output + teacher_output batches.

    Each sample has raw (variable-length) teacher logprobs attached by the
    teacher thread. This function:
      1. Stacks gen fields (input_ids, attention_mask, etc.) into batch tensors
      2. Calls pad_teacher() to produce padded [bs, seq_len, topk] tensors
    Returns (gen_output, teacher_output) matching _async_train() format.

    Args:
        samples: list of per-sample dicts from rollout + teacher scoring
        pad_teacher_fn: padding function (defaults to pad_teacher from this module)
    """
    if pad_teacher_fn is None:
        pad_teacher_fn = pad_teacher

    gen_output = stack_gen_output(samples)

    # Response-local support/MC paths
    if "teacher_query_logprobs_response" in samples[0]:
        all_query_logps = []
        if "teacher_mc_indices_response" in samples[0]:
            all_query_indices = []
            for s in samples:
                all_query_logps.extend(s["teacher_query_logprobs_response"])
                all_query_indices.extend(s["teacher_mc_indices_response"])
            teacher_output = adapt_mc_response_samples(
                gen_output, all_query_indices, all_query_logps, None)
        elif "mc_query_indices_response" in samples[0]:
            all_query_indices = []
            all_query_old = [] if "mc_query_old_logprobs_response" in samples[0] else None
            eos_token_id = samples[0].get("eos_token_id")
            for s in samples:
                all_query_logps.extend(s["teacher_query_logprobs_response"])
                query_indices = s["mc_query_indices_response"]
                if eos_token_id is not None:
                    query_indices = [
                        torch.cat(
                            [
                                q,
                                torch.full(
                                    (q.size(0), 1),
                                    int(eos_token_id),
                                    dtype=q.dtype,
                                    device=q.device,
                                ),
                            ],
                            dim=-1,
                        )
                        if q is not None and q.dim() == 2 else q
                        for q in query_indices
                    ]
                all_query_indices.extend(query_indices)
                if all_query_old is not None:
                    all_query_old.extend(s["mc_query_old_logprobs_response"])
            teacher_output = adapt_mc_response_samples(
                gen_output, all_query_indices, all_query_logps, all_query_old)
            if teacher_output is not None and eos_token_id is not None:
                teacher_output["eos_token_id"] = int(eos_token_id)
        else:
            all_query_indices = []
            all_student_query_logps = [] if "query_logprobs_response" in samples[0] else None
            for s in samples:
                all_query_logps.extend(s["teacher_query_logprobs_response"])
                all_query_indices.extend(s["query_indices_response"])
                if all_student_query_logps is not None:
                    all_student_query_logps.extend(s["query_logprobs_response"])
            teacher_output = adapt_response_support(
                gen_output, all_query_indices, all_query_logps, all_student_query_logps)
        return gen_output, teacher_output

    # Extract raw teacher logprobs and pad
    all_logps = []
    all_indices = []
    all_token_logps = []
    for s in samples:
        all_logps.extend(s["teacher_topk_logps"])
        all_indices.extend(s["teacher_topk_indices"])
        if "teacher_token_logps" in s:
            all_token_logps.extend(s["teacher_token_logps"])

    teacher_output = pad_teacher_fn(
        gen_output, all_logps, all_indices,
        all_token_logps if all_token_logps else None)

    return gen_output, teacher_output


def broadcast_batch(batch, rank, world_size, device):
    """Broadcast training batch from rank 0 to all ranks efficiently.

    Uses torch.distributed.broadcast for tensors (GPU-direct NCCL) instead of
    broadcast_object_list which pickles everything through CPU.
    Non-tensor fields (like prompt_lengths list) use broadcast_object_list.

    Truncates padded sequences to actual max length before broadcasting to
    avoid wasting GPU memory on padding (e.g. 18432 padded → ~4000 actual).
    """
    # Step 0: rank 0 truncates batch to actual max length before broadcast.
    # Saves ~5-6 GB GPU reservation for typical batches (18432 → ~4000 tokens).
    if rank == 0 and "attention_mask" in batch:
        nonzero_cols = batch["attention_mask"].nonzero(as_tuple=True)[1]
        actual_max = int(nonzero_cols.max().item()) + 2 if nonzero_cols.numel() > 0 else 1
        orig_seq_len = batch["input_ids"].size(1)
        if actual_max < orig_seq_len:
            truncated = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.size(1) == orig_seq_len:
                    truncated[k] = v[:, :actual_max] if v.dim() == 2 else v[:, :actual_max, :]
                else:
                    truncated[k] = v
            # Preserve original seq_len so _fsdp_train_step can compute max_prompt correctly.
            # After truncation, input_ids.size(1) < orig_seq_len, but max_prompt depends on
            # the original padded layout: max_prompt = orig_seq_len - max_response_length.
            truncated["_orig_seq_len"] = orig_seq_len
            batch = truncated

    # Step 1: broadcast batch structure (keys, shapes, dtypes) — small metadata
    if rank == 0:
        meta = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                meta[k] = ("tensor", list(v.shape), str(v.dtype))
            else:
                meta[k] = ("other", v)
        meta_list = [meta]
    else:
        meta_list = [None]
    torch.distributed.broadcast_object_list(meta_list, src=0)
    meta = meta_list[0]

    # Step 2: broadcast tensors via NCCL, reconstruct batch on all ranks
    result = {}
    for k, info in meta.items():
        if info[0] == "tensor":
            _, shape, dtype_str = info
            dtype = getattr(torch, dtype_str.replace("torch.", ""))
            if rank == 0:
                t = batch[k].to(device)
            else:
                t = torch.empty(shape, dtype=dtype, device=device)
            torch.distributed.broadcast(t, src=0)
            result[k] = t.cpu()  # train step moves to device itself
            del t
        else:
            result[k] = info[1]

    # Free GPU memory reserved by the broadcast tensors
    torch.cuda.empty_cache()
    return result

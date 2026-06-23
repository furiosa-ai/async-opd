"""Canonical OPD trainer-payload assembly helpers."""

from __future__ import annotations


def build_opd_train_batch(gen_output: dict, teacher_output: dict) -> dict:
    """Build the trainer batch consumed by OPDTrainer/BaseBackend.

    This pure helper is shared by the legacy queue trainer path and fused hybrid
    mode so rollout/teacher alignment stays identical across schedulers.
    """
    required_gen = ("input_ids", "attention_mask", "responses", "prompt_lengths")
    for key in required_gen:
        if key not in gen_output:
            raise KeyError(f"gen_output missing required OPD field: {key}")

    batch = {
        "input_ids": gen_output["input_ids"],
        "attention_mask": gen_output["attention_mask"],
        "responses": gen_output["responses"],
        "prompt_lengths": gen_output["prompt_lengths"],
    }
    if "support_topk_logps" in teacher_output:
        batch["support_topk_logps"] = teacher_output["support_topk_logps"]
        batch["support_topk_indices"] = teacher_output["support_topk_indices"]
        batch["support_valid_mask"] = teacher_output["support_valid_mask"]
        if "support_student_old_logps" in teacher_output:
            batch["support_student_old_logps"] = teacher_output["support_student_old_logps"]
    elif "mc_teacher_logprobs" in teacher_output:
        batch["mc_teacher_logprobs"] = teacher_output["mc_teacher_logprobs"]
        batch["mc_sample_indices"] = teacher_output["mc_sample_indices"]
        if "mc_old_logprobs" in teacher_output:
            batch["mc_old_logprobs"] = teacher_output["mc_old_logprobs"]
        batch["mc_valid_mask"] = teacher_output["mc_valid_mask"]
    elif "teacher_hidden_states" in teacher_output:
        batch["teacher_hidden_states"] = teacher_output["teacher_hidden_states"]
        batch["teacher_hidden_valid_mask"] = teacher_output["teacher_hidden_valid_mask"]
    else:
        for key in ("teacher_topk_logps", "teacher_topk_indices", "teacher_valid_mask"):
            if key not in teacher_output:
                raise KeyError(f"teacher_output missing required OPD field: {key}")
        batch["teacher_topk_logps"] = teacher_output["teacher_topk_logps"]
        batch["teacher_topk_indices"] = teacher_output["teacher_topk_indices"]
        batch["teacher_valid_mask"] = teacher_output["teacher_valid_mask"]
    if "teacher_token_logps" in teacher_output:
        batch["teacher_token_logps"] = teacher_output["teacher_token_logps"]
    if "eos_token_id" in teacher_output:
        batch["eos_token_id"] = teacher_output["eos_token_id"]
    if "student_logprobs" in gen_output:
        batch["student_logprobs"] = gen_output["student_logprobs"]

    # Preserve global-mini metadata if upstream tests/probes attach it. The
    # normal FSDP path computes this later in _prepare_train_batch, but keeping
    # these keys stable makes helper use safe for direct/fake hybrid tests.
    for key in (
        "_use_global_mini_plan",
        "_mini_slices",
        "_global_mini_slices",
        "_rank_source_ranges",
        "_global_batch_size",
        "_configured_global_mini_batch_size",
        "_common_micro_counts",
    ):
        if key in gen_output:
            batch[key] = gen_output[key]
    return batch

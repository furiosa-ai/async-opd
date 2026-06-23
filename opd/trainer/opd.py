"""OPD Trainer — On-Policy Distillation via KL divergence loss.

Composition-based trainer: owns a backend (FSDP or Megatron) and provides
the KL distillation loss_fn. The backend handles forward/backward/optimizer.
"""

import time

import torch

from opd.launch_specs import TrainerLaunchSpec, algorithm_is_actor_critic
from opd.loss.kl import compute_kl_loss, chunked_dense_kl_from_hidden
from opd.trainer.base_trainer import BaseTrainer
from opd.trainer.config import build_kl_config_from_algorithm_payload


class OPDTrainer(BaseTrainer):
    """On-Policy Distillation trainer using KL divergence loss.

    Inherits from BaseTrainer for backend creation, run(), and prox
    precomputation. Provides OPD-specific loss methods:
      - _compute_loss: shared core — single place calling compute_kl_loss
      - loss_fn(logits, mb): non-chunked fallback (full [B,S,V] logits)
      - forward_and_loss_fn(model, mb, device): chunked path — controls
        model forward with _kl_args, handles OPD-specific packing
    """

    def __init__(self, config: dict, rank_info: dict | None = None):
        algo = config.static.algorithm if isinstance(config, TrainerLaunchSpec) else config["algorithm"]
        self.kl_config = build_kl_config_from_algorithm_payload(algo)

        super().__init__(config, rank_info)

    # ------------------------------------------------------------------ #
    #  Shared loss core — single source of truth for compute_kl_loss     #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, *, student_logits=None, student_topk_logps=None,
                      student_token_logps=None, student_mc_logprobs=None,
                      student_dense_logps=None, teacher_dense_logps=None,
                      mb, response_mask=None):
        """Core KL loss computation. The ONLY place calling compute_kl_loss.

        Accepts either full logits or pre-gathered student logprobs and
        returns (loss, n_tok, extras).

        Args:
            student_logits: [B, S, V] — full logits (non-chunked fallback)
            student_topk_logps: [B, S, K] — from top-k modes (forward/reverse/skewed KL)
            student_token_logps: [B, shifted_len] — from token-level/PG modes
            mb: dict with teacher tensors, masks, and optional PG-KL extras
            response_mask: override mask (used by chunked path with shifted mask)

        Returns:
            (loss, n_tokens, extras_dict) where extras may include _raw_tensors
        """
        mask = response_mask if response_mask is not None else mb["response_mask"]

        kl_kwargs = dict(kl_config=self.kl_config, mask=mask)

        if self.kl_config.mode == "mof_opd":
            if student_mc_logprobs is not None:
                kl_kwargs["student_candidate_logprobs"] = student_mc_logprobs
                kl_kwargs["teacher_candidate_logprobs"] = mb["mc_teacher_logprobs"]
                if "mc_sample_indices" in mb:
                    kl_kwargs["candidate_token_ids"] = mb["mc_sample_indices"]
                if "mc_valid_mask" in mb:
                    cm = mb["mc_valid_mask"]
                    if cm.dim() == 2:
                        cm = cm.unsqueeze(-1).expand_as(student_mc_logprobs)
                    kl_kwargs["candidate_mask"] = cm
            elif student_token_logps is not None:
                kl_kwargs["student_token_logps"] = student_token_logps
                kl_kwargs["teacher_token_logps"] = mb.get("_shifted_teacher_token_logps",
                                                           mb.get("teacher_token_logps"))
                if "candidate_token_ids" in mb:
                    kl_kwargs["candidate_token_ids"] = mb["candidate_token_ids"]
            elif student_logits is not None:
                kl_kwargs["student_logits"] = student_logits
                if "mc_sample_indices" in mb:
                    kl_kwargs["mc_sample_indices"] = mb["mc_sample_indices"]
                    kl_kwargs["mc_teacher_logprobs"] = mb["mc_teacher_logprobs"]
                else:
                    kl_kwargs["teacher_token_logps"] = mb["teacher_token_logps"]
                    kl_kwargs["input_ids"] = mb["input_ids"]
            else:
                raise ValueError("mof_opd requires token, MC, or full-logits student inputs")
            if "student_eos_logprobs" in mb:
                kl_kwargs["student_eos_logprobs"] = mb["student_eos_logprobs"]
            if "teacher_eos_logprobs" in mb:
                kl_kwargs["teacher_eos_logprobs"] = mb["teacher_eos_logprobs"]
            if "eos_token_id" in mb:
                kl_kwargs["eos_token_id"] = mb["eos_token_id"]
        elif self.kl_config.mode in {"multi_sample_policy_gradient_kl", "multi_sample_forward_kl"}:
            kl_kwargs["mc_teacher_logprobs"] = mb["mc_teacher_logprobs"]
            if self.kl_config.mode == "multi_sample_policy_gradient_kl":
                kl_kwargs["mc_old_logprobs"] = mb["mc_old_logprobs"]
            if student_logits is not None:
                kl_kwargs["student_logits"] = student_logits
                kl_kwargs["mc_sample_indices"] = mb["mc_sample_indices"]
            elif student_mc_logprobs is not None:
                kl_kwargs["student_mc_logprobs"] = student_mc_logprobs
            else:
                raise ValueError(
                    f"{self.kl_config.mode} requires student_logits or student_mc_logprobs"
                )
            if self.kl_config.mode == "multi_sample_policy_gradient_kl" and "prox_logprobs" in mb:
                kl_kwargs["prox_logprobs"] = mb["prox_logprobs"]
        elif student_dense_logps is not None or teacher_dense_logps is not None:
            kl_kwargs["student_dense_logps"] = student_dense_logps
            kl_kwargs["teacher_dense_logps"] = teacher_dense_logps
        elif student_logits is not None:
            # Full logits path — compute_kl_loss handles gather internally
            kl_kwargs["student_logits"] = student_logits
            kl_kwargs["teacher_topk_logps"] = mb["teacher_topk_logps"]
            kl_kwargs["teacher_topk_indices"] = mb["teacher_topk_indices"]
            if "student_old_topk_logps" in mb:
                kl_kwargs["student_old_topk_logps"] = mb["student_old_topk_logps"]
            # PG-KL extras for full-logits path
            if "teacher_token_logps" in mb:
                kl_kwargs["teacher_token_logps"] = mb["teacher_token_logps"]
                kl_kwargs["input_ids"] = mb["input_ids"]
            if "student_logprobs" in mb:
                kl_kwargs["student_old_logprobs"] = mb["student_logprobs"]
        elif student_topk_logps is not None:
            # Group A: forward_kl, reverse_kl, skewed_kl (pre-gathered)
            kl_kwargs["student_topk_logps"] = student_topk_logps
            kl_kwargs["teacher_topk_logps"] = mb["teacher_topk_logps"]
            if "student_old_topk_logps" in mb:
                kl_kwargs["student_old_topk_logps"] = mb["student_old_topk_logps"]
        elif student_token_logps is not None:
            # Group B: token_level_kl, policy_gradient_kl (pre-gathered)
            kl_kwargs["student_token_logps"] = student_token_logps
            kl_kwargs["teacher_token_logps"] = mb.get("_shifted_teacher_token_logps",
                                                       mb.get("teacher_token_logps"))
            # PG-KL extras
            if "student_old_logprobs" in mb:
                kl_kwargs["student_old_logprobs"] = mb["student_old_logprobs"]
            if "prox_logprobs" in mb:
                kl_kwargs["prox_logprobs"] = mb["prox_logprobs"]
        else:
            raise ValueError(
                "Must provide student_logits, student_topk_logps, student_token_logps, "
                "or student_mc_logprobs"
            )

        loss = compute_kl_loss(**kl_kwargs)

        n_tok = int(mask.sum().item())
        extras = {}
        raw_tensors = {}

        if hasattr(loss, "kl_stats"):
            kl_vals = loss.kl_stats["_vals"]
            if kl_vals.numel() > 0:
                extras["kl_std"] = kl_vals.std().item() if kl_vals.numel() > 1 else 0.0
                extras["kl_min"] = kl_vals.min().item()
                extras["kl_max"] = kl_vals.max().item()
                raw_tensors["kl_vals"] = kl_vals.detach().cpu()

        if hasattr(loss, "pg_stats"):
            r = loss.pg_stats["_ratios"]
            if r.numel() > 0:
                extras["r_mean"] = r.mean().item()
                extras["clip_frac_high"] = loss.pg_stats["_clip_high"].float().mean().item()
                extras["clip_frac_low"] = loss.pg_stats["_clip_low"].float().mean().item()
                raw_tensors["ratios"] = r.detach().cpu()
                raw_tensors["log_ratios"] = loss.pg_stats["_log_ratios"].detach().cpu()
                raw_tensors["advantages"] = loss.pg_stats["_advantages"].detach().cpu()
                raw_tensors["clip_high"] = loss.pg_stats["_clip_high"].detach().cpu()
                raw_tensors["clip_low"] = loss.pg_stats["_clip_low"].detach().cpu()
            if "m2po_clip_low" in loss.pg_stats:
                extras["m2po_clip_low"] = loss.pg_stats["m2po_clip_low"]
                extras["m2po_clip_high"] = loss.pg_stats["m2po_clip_high"]
                extras["m2po_m2_before"] = loss.pg_stats["m2po_m2_before"]
                extras["m2po_m2_after"] = loss.pg_stats["m2po_m2_after"]
            if "_behave_imp_weight" in loss.pg_stats:
                w = loss.pg_stats["_behave_imp_weight"]
                if w.numel() > 0:
                    extras["behave_imp_weight"] = w.mean().item()
                    bmask = loss.pg_stats["_behave_mask"]
                    extras["behave_mask_ratio"] = bmask.float().mean().item()

        if hasattr(loss, "mof_stats"):
            for key, value in loss.mof_stats.items():
                if isinstance(value, torch.Tensor):
                    extras[key] = value.item()
                else:
                    extras[key] = value

        if raw_tensors:
            extras["_raw_tensors"] = raw_tensors

        return loss, n_tok, extras

    # ------------------------------------------------------------------ #
    #  Non-chunked loss_fn (fallback — used when chunked not available)  #
    # ------------------------------------------------------------------ #

    def loss_fn(self, logits, mb):
        """KL distillation loss from full [B, S, V] logits.

        This is the fallback path used when chunked LM head is not available
        (e.g., unsupported model architecture). Delegates to _compute_loss.

        Args:
            logits: [B, S, V] student model logits
            mb: dict with teacher_topk_logps, teacher_topk_indices, response_mask, etc.

        Returns:
            (loss, n_tokens, extras_dict)
        """
        return self._compute_loss(student_logits=logits, mb=mb)

    # ------------------------------------------------------------------ #
    #  Chunked forward_and_loss_fn (preferred — controls model forward)  #
    # ------------------------------------------------------------------ #

    def forward_and_loss_fn(self, model, mb, device):
        """Chunked LM head path: controls model forward and computes KL loss.

        Handles OPD-specific sequence packing internally. The backend calls
        this instead of model(**fwd_kwargs) + loss_fn(logits, mb) when
        chunked LM head is active.

        Args:
            model: the (possibly FSDP-wrapped) model with _kl_args monkey-patch
            mb: dict of micro-batch tensors (raw, not yet packed) + metadata scalars
                Required: input_ids, attention_mask, response_mask,
                          teacher_topk_logps, teacher_topk_indices
                Optional: teacher_token_logps, student_logprobs, prompt_lengths,
                          position_ids
                Metadata: max_prompt, actual_max_len, seq_len, orig_seq_len
            device: torch device

        Returns:
            (loss, n_tokens, extras_dict) — same contract as loss_fn
        """
        kl_loss_mode = self.kl_config.mode
        kl_chunk_size = getattr(self._backend, 'kl_chunk_size', 1024)

        mb_input_ids = mb["input_ids"]
        mb_attention_mask = mb["attention_mask"]
        mb_teacher_logps = mb.get("teacher_topk_logps")
        mb_teacher_idx = mb.get("teacher_topk_indices")
        mb_teacher_hidden = mb.get("teacher_hidden_states")
        mb_teacher_hidden_valid = mb.get("teacher_hidden_valid_mask")
        mb_response_mask = mb["response_mask"]
        mb_position_ids = mb.get("position_ids")

        # Metadata scalars (non-tensor, passed through by _run_train_step)
        max_prompt = mb.get("max_prompt", 0)
        actual_max_len = mb.get("actual_max_len", mb_input_ids.size(1))
        seq_len = mb.get("seq_len", mb_input_ids.size(1))

        # ---- OPD-specific sequence packing ----
        mb_packing_kwargs = {}
        use_packing = getattr(self._backend, 'use_sequence_packing', False)
        if use_packing:
            from opd.data.packing import pack_micro_batch
            mb_prompt_lens = mb.get("prompt_lengths")
            if mb_prompt_lens is not None:
                pack_kwargs = dict(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attention_mask,
                    teacher_topk_logps=mb_teacher_logps,
                    teacher_topk_indices=mb_teacher_idx,
                    response_mask=mb_response_mask,
                    prompt_lengths=mb_prompt_lens,
                )
                if "teacher_token_logps" in mb and isinstance(mb["teacher_token_logps"], torch.Tensor):
                    mb_ttl = mb["teacher_token_logps"]
                    if actual_max_len < seq_len:
                        mb_ttl = mb_ttl[:, :actual_max_len]
                    pack_kwargs["teacher_token_logps"] = mb_ttl
                if "student_logprobs" in mb and isinstance(mb["student_logprobs"], torch.Tensor):
                    mb_slp = mb["student_logprobs"]
                    if actual_max_len < seq_len:
                        actual_resp_len = actual_max_len - max_prompt
                        mb_slp = mb_slp[:, :actual_resp_len]
                    pack_kwargs["student_logprobs"] = mb_slp
                    pack_kwargs["max_prompt"] = max_prompt
                if "support_student_old_logps" in mb and isinstance(mb["support_student_old_logps"], torch.Tensor):
                    mb_sto = mb["support_student_old_logps"]
                    if actual_max_len < seq_len:
                        mb_sto = mb_sto[:, :actual_max_len]
                    pack_kwargs["support_student_old_logps"] = mb_sto
                if "mc_sample_indices" in mb and isinstance(mb["mc_sample_indices"], torch.Tensor):
                    mb_mc_idx = mb["mc_sample_indices"]
                    if actual_max_len < seq_len:
                        mb_mc_idx = mb_mc_idx[:, :actual_max_len]
                    pack_kwargs["mc_sample_indices"] = mb_mc_idx
                if "mc_teacher_logprobs" in mb and isinstance(mb["mc_teacher_logprobs"], torch.Tensor):
                    mb_mc_teacher = mb["mc_teacher_logprobs"]
                    if actual_max_len < seq_len:
                        mb_mc_teacher = mb_mc_teacher[:, :actual_max_len]
                    pack_kwargs["mc_teacher_logprobs"] = mb_mc_teacher
                if "mc_old_logprobs" in mb and isinstance(mb["mc_old_logprobs"], torch.Tensor):
                    mb_mc_old = mb["mc_old_logprobs"]
                    if actual_max_len < seq_len:
                        mb_mc_old = mb_mc_old[:, :actual_max_len]
                    pack_kwargs["mc_old_logprobs"] = mb_mc_old
                if "mc_valid_mask" in mb and isinstance(mb["mc_valid_mask"], torch.Tensor):
                    mb_mc_valid = mb["mc_valid_mask"]
                    if actual_max_len < seq_len:
                        mb_mc_valid = mb_mc_valid[:, :actual_max_len]
                    pack_kwargs["mc_valid_mask"] = mb_mc_valid
                if mb_teacher_hidden is not None:
                    pack_kwargs["teacher_hidden_states"] = mb_teacher_hidden
                    if mb_teacher_hidden_valid is not None:
                        pack_kwargs["teacher_hidden_valid_mask"] = mb_teacher_hidden_valid

                packed = pack_micro_batch(**pack_kwargs)
                mb_input_ids = packed.input_ids
                mb_position_ids = packed.position_ids
                mb_attention_mask = None  # Must be None for FA varlen
                mb_teacher_logps = packed.teacher_topk_logps
                mb_teacher_idx = packed.teacher_topk_indices
                mb_teacher_hidden = packed.teacher_hidden_states
                mb_teacher_hidden_valid = packed.teacher_hidden_valid_mask
                mb_response_mask = packed.response_mask
                mb_packing_kwargs = {
                    "cu_seq_lens_q": packed.cu_seq_lens,
                    "cu_seq_lens_k": packed.cu_seq_lens,
                    "max_length_q": packed.max_seq_len,
                    "max_length_k": packed.max_seq_len,
                }
                # Store packed data for policy_gradient_kl
                if packed.teacher_token_logps is not None:
                    mb["_packed_teacher_token_logps"] = packed.teacher_token_logps
                if packed.student_logprobs is not None:
                    mb["_packed_student_logprobs"] = packed.student_logprobs
                    mb["_packed_prompt_lens"] = packed.prompt_lens
                    mb["_packed_seq_lens"] = packed.seq_lens
                if packed.support_student_old_logps is not None:
                    mb["_packed_support_student_old_logps"] = packed.support_student_old_logps
                if packed.mc_sample_indices is not None:
                    mb["_packed_mc_sample_indices"] = packed.mc_sample_indices
                if packed.mc_teacher_logprobs is not None:
                    mb["_packed_mc_teacher_logprobs"] = packed.mc_teacher_logprobs
                if packed.mc_old_logprobs is not None:
                    mb["_packed_mc_old_logprobs"] = packed.mc_old_logprobs
                if packed.mc_valid_mask is not None:
                    mb["_packed_mc_valid_mask"] = packed.mc_valid_mask

        # ---- Model forward with _kl_args ----
        if kl_loss_mode in ("forward_kl", "reverse_kl", "reverse_kl_rollout_student_topk", "thunlp_opd_default_loss", "skewed_kl"):
            if mb_teacher_hidden is not None:
                if kl_loss_mode not in {"forward_kl", "reverse_kl", "skewed_kl"}:
                    raise ValueError(
                        f"hidden_recompute lazy path does not support {kl_loss_mode}"
                    )
                if mb_teacher_hidden_valid is not None:
                    mb_response_mask = (
                        mb_response_mask
                        & mb_teacher_hidden_valid.to(mb_response_mask.device).bool()
                    )
                t0 = time.monotonic()
                hidden_out = model(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attention_mask,
                    position_ids=mb_position_ids,
                    _kl_args={"mode": "return_hidden", "chunk_size": kl_chunk_size},
                    **mb_packing_kwargs,
                )
                student_hidden = hidden_out["hidden_states"]
                teacher_head = self._backend._get_teacher_recompute_head()
                loss = chunked_dense_kl_from_hidden(
                    student_hidden=student_hidden,
                    student_lm_head_weight=hidden_out["lm_head_weight"],
                    teacher_hidden=mb_teacher_hidden,
                    teacher_lm_head_weight=teacher_head.lm_head_weight,
                    mask=mb_response_mask,
                    mode=kl_loss_mode,
                    alpha=self.kl_config.skew_alpha,
                    token_clip=self.kl_config.token_clip,
                    chunk_size=teacher_head.chunk_size or kl_chunk_size,
                    memory_strategy="checkpoint",
                )
                n_tok = int(mb_response_mask.sum().item())
                extras = {}
                vals = getattr(loss, "kl_stats", {}).get("_vals")
                if vals is not None and vals.numel() > 0:
                    extras["kl_mean"] = vals.float().mean().item()
                extras.update(getattr(loss, "chunked_dense_kl_stats", {}))
                extras["teacher_hidden_fused_kl_seconds"] = time.monotonic() - t0
                extras["teacher_hidden_vocab_size"] = teacher_head.vocab_size
                extras["teacher_hidden_size"] = teacher_head.hidden_size
                del hidden_out, student_hidden
                return loss, n_tok, extras

            # Group A: top-k gathered logprobs [B, S, K]
            student_topk_logps = model(
                input_ids=mb_input_ids,
                attention_mask=mb_attention_mask,
                position_ids=mb_position_ids,
                _kl_args={'mode': kl_loss_mode, 'indices': mb_teacher_idx,
                          'chunk_size': kl_chunk_size},
                **mb_packing_kwargs,
            )
            # Build mb for _compute_loss
            if kl_loss_mode == "thunlp_opd_default_loss":
                loss_mb = {
                    "teacher_topk_logps": mb_teacher_logps[:, 1:],
                    "response_mask": mb_response_mask[:, 1:],
                }
                loss_mb["student_old_topk_logps"] = (
                    mb["_packed_support_student_old_logps"]
                    if "_packed_support_student_old_logps" in mb
                    else mb["support_student_old_logps"]
                )[:, 1:]
            else:
                loss_mb = {
                    "teacher_topk_logps": mb_teacher_logps,
                    "response_mask": mb_response_mask,
                }
            return self._compute_loss(student_topk_logps=student_topk_logps, mb=loss_mb)

        elif kl_loss_mode == "mof_opd" and (
            mb.get("_packed_mc_sample_indices", mb.get("mc_sample_indices")) is None
        ):
            # Generated-only MOF path: the candidate set is the sampled token.
            # Reuse the token-level gather path to avoid adding another backend
            # patch mode.
            student_token_logps = model(
                input_ids=mb_input_ids,
                attention_mask=mb_attention_mask,
                position_ids=mb_position_ids,
                _kl_args={'mode': 'token_level_kl', 'chunk_size': kl_chunk_size},
                **mb_packing_kwargs,
            ).squeeze(-1)

            if "_packed_teacher_token_logps" in mb:
                mb_teacher_token_logps = mb["_packed_teacher_token_logps"]
            else:
                mb_teacher_token_logps = mb.get("teacher_token_logps")
                if mb_teacher_token_logps is not None:
                    if actual_max_len < seq_len:
                        mb_teacher_token_logps = mb_teacher_token_logps[:, :actual_max_len]
                    mb_teacher_token_logps = mb_teacher_token_logps.to(device)
            if mb_teacher_token_logps is None:
                raise ValueError("mof_opd generated-only path requires teacher_token_logps")

            shifted_mask = mb_response_mask[:, 1:]
            loss_mb = {
                "response_mask": shifted_mask,
                "_shifted_teacher_token_logps": mb_teacher_token_logps[:, :-1],
                "candidate_token_ids": mb_input_ids[:, 1:].unsqueeze(-1),
            }
            if "eos_token_id" in mb:
                loss_mb["eos_token_id"] = mb["eos_token_id"]
            return self._compute_loss(
                student_token_logps=student_token_logps,
                mb=loss_mb,
                response_mask=shifted_mask,
            )

        elif kl_loss_mode in ("token_level_kl", "policy_gradient_kl"):
            # Group B: single-token logprobs [B, S-1, 1] -> [B, S-1]
            student_token_logps = model(
                input_ids=mb_input_ids,
                attention_mask=mb_attention_mask,
                position_ids=mb_position_ids,
                _kl_args={'mode': kl_loss_mode, 'chunk_size': kl_chunk_size},
                **mb_packing_kwargs,
            ).squeeze(-1)

            # Extract teacher token logprobs
            if "_packed_teacher_token_logps" in mb:
                mb_teacher_token_logps = mb["_packed_teacher_token_logps"]
            else:
                mb_teacher_token_logps = mb.get("teacher_token_logps")
                if mb_teacher_token_logps is not None:
                    if actual_max_len < seq_len:
                        mb_teacher_token_logps = mb_teacher_token_logps[:, :actual_max_len]
                    mb_teacher_token_logps = mb_teacher_token_logps.to(device)

            # Shift teacher logps to match student's shifted output
            t_logps = mb_teacher_token_logps[:, :-1]
            shifted_mask = mb_response_mask[:, 1:]

            # Build loss_mb
            loss_mb = {"response_mask": shifted_mask,
                       "_shifted_teacher_token_logps": t_logps}

            if kl_loss_mode == "policy_gradient_kl":
                # Prepare old_logps
                shifted_len = student_token_logps.shape[1]

                if "_packed_student_logprobs" in mb:
                    # packed_student_logprobs[i] = log P_old(token_i).
                    # student_token_logps[i] = log P(token_{i+1} | token_{0:i}).
                    # So old_logps[i] should be log P_old(token_{i+1}) = packed[i+1].
                    # Use [:, 1:] (drop first), NOT [:, :-1] (drop last).
                    # packed_student_logprobs[i] = log P_old(token_i).
                    # student_token_logps[i] = log P(token_{i+1} | token_{0:i}).
                    # So old_logps[i] should be log P_old(token_{i+1}) = packed[i+1].
                    # Use [:, 1:] (drop first), NOT [:, :-1] (drop last).
                    old_logps = mb["_packed_student_logprobs"][:, 1:].to(device)
                    if old_logps.shape[1] < shifted_len:
                        old_logps = torch.nn.functional.pad(
                            old_logps, (0, shifted_len - old_logps.shape[1]))
                    elif old_logps.shape[1] > shifted_len:
                        old_logps = old_logps[:, :shifted_len]
                else:
                    mb_student_logprobs = mb.get("student_logprobs")
                    if mb_student_logprobs is not None:
                        if actual_max_len < seq_len:
                            actual_resp_len = actual_max_len - max_prompt
                            mb_student_logprobs = mb_student_logprobs[:, :actual_resp_len]
                        mb_student_logprobs = mb_student_logprobs.to(device)
                        resp_len = mb_student_logprobs.size(1)
                        resp_start = max(0, max_prompt - 1)
                        resp_span = min(resp_len, shifted_len - resp_start)
                        old_logps = torch.zeros(student_token_logps.shape[0], shifted_len,
                                                device=device, dtype=mb_student_logprobs.dtype)
                        if resp_span > 0:
                            old_logps[:, resp_start:resp_start + resp_span] = \
                                mb_student_logprobs[:, :resp_span]
                    else:
                        old_logps = torch.zeros_like(student_token_logps)

                loss_mb["student_old_logprobs"] = old_logps

            prox_logprobs = self._consume_prox_logprobs(student_token_logps)
            if prox_logprobs is not None:
                loss_mb["prox_logprobs"] = prox_logprobs

            return self._compute_loss(
                student_token_logps=student_token_logps, mb=loss_mb,
                response_mask=shifted_mask)

        elif kl_loss_mode in {"multi_sample_policy_gradient_kl", "multi_sample_forward_kl", "mof_opd"}:
            if (
                kl_loss_mode == "multi_sample_policy_gradient_kl"
                and self._is_decoupled_ppo_enabled()
            ):
                raise NotImplementedError(
                    "multi_sample_policy_gradient_kl does not support "
                    "use_decoupled_loss=True in v1"
                )
            mc_sample_indices = mb.get("_packed_mc_sample_indices", mb.get("mc_sample_indices"))
            if mc_sample_indices is None:
                raise ValueError(
                    f"{kl_loss_mode} requires mc_sample_indices in the micro-batch"
                )
            if "_packed_mc_sample_indices" not in mb and actual_max_len < seq_len:
                mc_sample_indices = mc_sample_indices[:, :actual_max_len]

            if "_packed_mc_teacher_logprobs" in mb:
                mc_teacher_logprobs = mb["_packed_mc_teacher_logprobs"]
            else:
                mc_teacher_logprobs = mb["mc_teacher_logprobs"]
                if actual_max_len < seq_len:
                    mc_teacher_logprobs = mc_teacher_logprobs[:, :actual_max_len]
                mc_teacher_logprobs = mc_teacher_logprobs.to(device)

            mc_teacher_logprobs = mc_teacher_logprobs[:, 1:]
            shifted_mc_sample_indices = mc_sample_indices[:, 1:]
            shifted_mask = mb_response_mask[:, 1:]

            mc_valid_mask = None
            if "_packed_mc_valid_mask" in mb:
                mc_valid_mask = mb["_packed_mc_valid_mask"]
            elif "mc_valid_mask" in mb:
                mc_valid_mask = mb["mc_valid_mask"]
                if actual_max_len < seq_len:
                    mc_valid_mask = mc_valid_mask[:, :actual_max_len]
                mc_valid_mask = mc_valid_mask.to(device)
            if mc_valid_mask is not None:
                mc_valid_mask = mc_valid_mask[:, 1:]

            mc_old_logprobs = None
            if kl_loss_mode == "multi_sample_policy_gradient_kl":
                if "_packed_mc_old_logprobs" in mb:
                    mc_old_logprobs = mb["_packed_mc_old_logprobs"]
                else:
                    mc_old_logprobs = mb["mc_old_logprobs"]
                    if actual_max_len < seq_len:
                        mc_old_logprobs = mc_old_logprobs[:, :actual_max_len]
                    mc_old_logprobs = mc_old_logprobs.to(device)
                mc_old_logprobs = mc_old_logprobs[:, 1:]

            student_mc_logprobs = model(
                input_ids=mb_input_ids,
                attention_mask=mb_attention_mask,
                position_ids=mb_position_ids,
                _kl_args={
                    'mode': 'multi_sample_forward_kl' if kl_loss_mode == 'mof_opd' else kl_loss_mode,
                    'indices': mc_sample_indices,
                    'chunk_size': kl_chunk_size,
                },
                **mb_packing_kwargs,
            )

            loss_mb = {
                "mc_teacher_logprobs": mc_teacher_logprobs,
                "mc_sample_indices": shifted_mc_sample_indices,
            }
            if mc_valid_mask is not None:
                loss_mb["mc_valid_mask"] = mc_valid_mask
            if mc_old_logprobs is not None:
                loss_mb["mc_old_logprobs"] = mc_old_logprobs
            if kl_loss_mode == "multi_sample_policy_gradient_kl":
                prox_logprobs = self._consume_prox_logprobs(student_mc_logprobs)
                if prox_logprobs is not None:
                    loss_mb["prox_logprobs"] = prox_logprobs
            if kl_loss_mode == "mof_opd" and "eos_token_id" in mb:
                loss_mb["eos_token_id"] = mb["eos_token_id"]

            return self._compute_loss(
                student_mc_logprobs=student_mc_logprobs,
                mb=loss_mb,
                response_mask=shifted_mask,
            )

        else:
            raise ValueError(f"Unsupported KL mode for chunked forward: {kl_loss_mode}")

    # ------------------------------------------------------------------ #
    #  Backend integration                                                #
    # ------------------------------------------------------------------ #

    def train_step(self, batch, backend):
        """Prepare batch and run training step via backend.

        Called by backend._handle_train when self is set as the trainer.
        """
        prepared = backend._prepare_train_batch(batch)

        # Flatten prepared dict for _run_train_step
        flat = {
            "input_ids": prepared["input_ids"],
            "attention_mask": prepared["attention_mask"],
            "response_mask": prepared["response_mask"],
            "prompt_lengths": prepared["prompt_lengths"],
            # Metadata scalars — pass through as non-tensor values
            "max_prompt": prepared["max_prompt"],
            "actual_max_len": prepared["actual_max_len"],
            "seq_len": prepared.get("orig_seq_len", prepared["input_ids"].size(1)),
            "orig_seq_len": prepared.get("orig_seq_len", prepared["input_ids"].size(1)),
        }
        self._copy_global_mini_plan_metadata(prepared, flat)
        if prepared["teacher_topk_logps"] is not None:
            flat["teacher_topk_logps"] = prepared["teacher_topk_logps"]
        if prepared["teacher_topk_indices"] is not None:
            flat["teacher_topk_indices"] = prepared["teacher_topk_indices"]
        if prepared.get("teacher_hidden_states") is not None:
            flat["teacher_hidden_states"] = prepared["teacher_hidden_states"]
        if prepared.get("teacher_hidden_valid_mask") is not None:
            flat["teacher_hidden_valid_mask"] = prepared["teacher_hidden_valid_mask"]

        # Pass through extra KL-specific tensors
        raw_batch = prepared["batch"]
        if "teacher_token_logps" in raw_batch:
            flat["teacher_token_logps"] = raw_batch["teacher_token_logps"]
        if "student_logprobs" in raw_batch:
            flat["student_logprobs"] = raw_batch["student_logprobs"]
        if "support_student_old_logps" in raw_batch:
            flat["support_student_old_logps"] = raw_batch["support_student_old_logps"]
        if "eos_token_id" in raw_batch:
            flat["eos_token_id"] = raw_batch["eos_token_id"]
        for key in ("mc_sample_indices", "mc_teacher_logprobs", "mc_old_logprobs", "mc_valid_mask"):
            if key in raw_batch:
                flat[key] = raw_batch[key]

        # Decoupled PPO: precompute pi_prox for full batch before any
        # optimizer steps, so mini-batches 2+ have pi_prox != pi_theta.
        self._maybe_precompute_prox(
            flat,
            backend,
            require_kl_mode="policy_gradient_kl",
        )

        return backend._run_train_step(flat, self.loss_fn,
                                       forward_and_loss_fn=self.forward_and_loss_fn)


def opd_trainer_main(config, cmd_queue, result_queue, rank_info):
    """Entry point for OPD training subprocess."""
    if isinstance(config, TrainerLaunchSpec):
        algo = config.static.algorithm
    else:
        algo = config.get("algorithm", {})
    if algorithm_is_actor_critic(algo):
        from opd.trainer.ac_opd import ActorCriticOPDTrainer
        trainer = ActorCriticOPDTrainer(config, rank_info)
    else:
        trainer = OPDTrainer(config, rank_info)
    trainer.run(cmd_queue, result_queue)

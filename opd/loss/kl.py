"""KL divergence losses for knowledge distillation.

Supports KL / PPO modes:

Sparse (using teacher top-k logprobs):
  - forward_kl:  KL(teacher || student) — mode-covering (default, used in GKD)
  - reverse_kl:  KL(student || teacher) — mode-seeking, summed over top-k only
  - skewed_kl:   interpolated forward + reverse KL (as in DistiLLM)

Per-token (using teacher logprob at sampled token only):
  - token_level_kl:      forward KL contribution via direct backprop
  - policy_gradient_kl:  PPO-clip with reverse KL as advantage (G-OPD / Thinking Machines style)
  - multi_sample_forward_kl: MC forward KL using teacher-sampled tokens
  - thunlp_opd_default_loss: faithful THUNLP-default detached 3D top-k PPO path

"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from opd.utils.config import OPDAlgorithmConfig


_OPD_DEFAULTS = OPDAlgorithmConfig()


@dataclass
class KLConfig:
    """Configuration for KL divergence loss computation.

    Groups the "config" parameters of compute_kl_loss (mode selection and
    clipping thresholds) into a single object, separate from the per-batch
    data tensors.
    """
    mode: str | None = None
    skew_alpha: float | None = None
    pg_clip_eps: float | None = None
    use_importance_sampling: bool | None = None
    pg_online_advantage: bool | None = None
    token_clip: float | None = None
    use_decoupled_loss: bool | None = None
    behave_imp_weight_cap: float | None = None
    pg_m2po_budget: float = None        # M2PO: second-moment budget (None = disabled, use fixed clip_eps)
    pg_m2po_miniclip_low: float | None = None   # M2PO: floor for dynamic clip low margin
    pg_m2po_miniclip_high: float | None = None  # M2PO: floor for dynamic clip high margin
    mof_variant: str | None = None
    mof_partition: str | None = None
    mof_eta_mass: float | None = None
    mof_eta_odds: float | None = None
    mof_lambda_odds: float | None = None
    mof_eps: float | None = None
    mof_deduplicate_candidates: bool | None = None


def resolve_kl_config(kl_config: "KLConfig | None") -> "KLConfig":
    """Fill unset KLConfig fields from the canonical OPD config owner."""

    config = kl_config or KLConfig()
    if config.mode is None:
        config.mode = _OPD_DEFAULTS.kl_loss_mode
    if config.skew_alpha is None:
        config.skew_alpha = _OPD_DEFAULTS.skewed_alpha
    if config.pg_clip_eps is None:
        config.pg_clip_eps = _OPD_DEFAULTS.pg_clip_eps
    if config.use_importance_sampling is None:
        config.use_importance_sampling = _OPD_DEFAULTS.use_importance_sampling
    if config.pg_online_advantage is None:
        config.pg_online_advantage = _OPD_DEFAULTS.pg_online_advantage
    if config.token_clip is None:
        config.token_clip = _OPD_DEFAULTS.kl_token_clip
    if config.use_decoupled_loss is None:
        config.use_decoupled_loss = _OPD_DEFAULTS.use_decoupled_loss
    if config.behave_imp_weight_cap is None:
        config.behave_imp_weight_cap = _OPD_DEFAULTS.behave_imp_weight_cap
    if config.pg_m2po_miniclip_low is None:
        config.pg_m2po_miniclip_low = _OPD_DEFAULTS.pg_m2po_miniclip_low
    if config.pg_m2po_miniclip_high is None:
        config.pg_m2po_miniclip_high = _OPD_DEFAULTS.pg_m2po_miniclip_high
    if config.mof_variant is None:
        config.mof_variant = _OPD_DEFAULTS.mof_variant
    if config.mof_partition is None:
        config.mof_partition = _OPD_DEFAULTS.mof_partition
    if config.mof_eta_mass is None:
        config.mof_eta_mass = _OPD_DEFAULTS.mof_eta_mass
    if config.mof_eta_odds is None:
        config.mof_eta_odds = _OPD_DEFAULTS.mof_eta_odds
    if config.mof_lambda_odds is None:
        config.mof_lambda_odds = _OPD_DEFAULTS.mof_lambda_odds
    if config.mof_eps is None:
        config.mof_eps = _OPD_DEFAULTS.mof_eps
    if config.mof_deduplicate_candidates is None:
        config.mof_deduplicate_candidates = _OPD_DEFAULTS.mof_deduplicate_candidates
    return config


# ---------------------------------------------------------------------------
# Chunked log-softmax gather (memory-efficient)
# ---------------------------------------------------------------------------

# Default chunk size for sequence-dimension chunking.
# 1024 tokens × 152K vocab × 4 bytes = 0.6 GB per chunk (vs 9.3 GB for 16K seq).
KL_CHUNK_SIZE = 1024


class _ChunkedLogSoftmaxGather(torch.autograd.Function):
    """Memory-efficient log-softmax + gather via sequence chunking.

    Instead of materializing the full [B, S, V] softmax during backward,
    recomputes softmax in chunks of `chunk_size` tokens along the sequence
    dimension. Peak activation memory: O(B * chunk_size * V) instead of
    O(B * S * V).

    Inspired by NeMo-RL's ChunkedDistributedGatherLogprob.
    """

    @staticmethod
    def forward(ctx, logits, indices, chunk_size):
        """
        Args:
            logits: [B, S, V] student logits (requires_grad)
            indices: [B, S, K] token indices to gather (int/long)
            chunk_size: number of sequence positions per chunk
        Returns:
            gathered_logps: [B, S, K] log-softmax values at indices
        """
        B, S, V = logits.shape
        K = indices.shape[-1]
        gathered_logps = torch.empty(B, S, K, device=logits.device,
                                     dtype=logits.dtype)

        # Use at least float32 for numerical stability (bf16 logits → fp32 logsumexp)
        compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32

        for s0 in range(0, S, chunk_size):
            s1 = min(s0 + chunk_size, S)
            chunk = logits[:, s0:s1, :]  # view, no copy
            idx = indices[:, s0:s1, :].long()
            lse = torch.logsumexp(chunk.to(compute_dtype), dim=-1, keepdim=True)
            topk_logits = torch.gather(chunk, dim=-1, index=idx)
            gathered_logps[:, s0:s1, :] = (topk_logits.to(compute_dtype) - lse).to(logits.dtype)

        ctx.save_for_backward(logits, indices)
        ctx.chunk_size = chunk_size
        return gathered_logps

    @staticmethod
    def backward(ctx, grad_output):
        """
        Gradient of log_softmax(z)[k] w.r.t. z[j]:
            d log_softmax(z)[k] / d z[j] = delta_{jk} - softmax(z)[j]

        So for loss L with dL/d(log_softmax[k]) = go[k]:
            dL/dz[j] = sum_k go[k] * (delta_{jk} - softmax_j)
                      = go[j] (if j in indices) - softmax_j * sum_k go[k]
                      = scatter(go at indices) - softmax * sum(go)
        """
        logits, indices = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        B, S, V = logits.shape

        compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        grad_input = torch.empty_like(logits, dtype=compute_dtype)

        for s0 in range(0, S, chunk_size):
            s1 = min(s0 + chunk_size, S)
            chunk = logits[:, s0:s1, :].to(compute_dtype)  # [B, C, V]
            softmax_chunk = torch.softmax(chunk, dim=-1)    # [B, C, V]

            go = grad_output[:, s0:s1, :].to(compute_dtype) # [B, C, K]
            go_sum = go.sum(dim=-1, keepdim=True)            # [B, C, 1]
            idx = indices[:, s0:s1, :].long()

            # scatter(go at indices) - softmax * sum(go)
            grad_chunk = -softmax_chunk * go_sum             # [B, C, V]
            grad_chunk.scatter_add_(-1, idx, go)

            grad_input[:, s0:s1, :] = grad_chunk

            del softmax_chunk, grad_chunk

        return grad_input.to(logits.dtype), None, None


def chunked_log_softmax_gather(logits, indices, chunk_size=KL_CHUNK_SIZE):
    """Memory-efficient log-softmax values at given indices.

    Args:
        logits: [B, S, V] — student logits
        indices: [B, S, K] — token indices to gather (int32 or int64)
        chunk_size: sequence chunk size (default 1024). 0 = no chunking.

    Returns:
        [B, S, K] — log_softmax(logits) gathered at indices
    """
    if chunk_size <= 0:
        chunk_size = logits.shape[1]
    return _ChunkedLogSoftmaxGather.apply(logits, indices, chunk_size)


# ---------------------------------------------------------------------------
# Chunked LM head + log-softmax gather (avoids materializing [B, S, V] logits)
# ---------------------------------------------------------------------------

class _ChunkedLMHeadGather(torch.autograd.Function):
    """Fused chunked lm_head + log-softmax + gather.

    Computes log_softmax(hidden @ weight.T)[indices] without ever
    materializing full [B, S, V] logits. The lm_head projection is computed
    per sequence chunk and immediately reduced to [B, C, K] gathered log-probs.

    During backward, logits are recomputed per chunk (activation recomputation).

    Saves: hidden_states [B,S,H], weight [V,H], indices [B,S,K]
    Never saves: logits [B,S,V] or softmax [B,S,V]

    Memory: O(B * chunk_size * V) per chunk instead of O(B * S * V) total.
    For B=1, S=16K, V=152K, chunk=1024: ~0.3GB per chunk vs ~4.6GB total.

    Note: Assumes lm_head has no bias (true for Qwen3, Llama, Mistral, etc.).
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, indices, chunk_size):
        """
        Args:
            hidden_states: [B, S, H] — last hidden states from transformer
            weight: [V, H] — lm_head weight matrix
            indices: [B, S, K] — token indices to gather
            chunk_size: number of sequence positions per chunk
        Returns:
            gathered_logps: [B, S, K] — log-softmax values at indices
        """
        B, S, H = hidden_states.shape
        K = indices.shape[-1]

        gathered_logps = torch.empty(B, S, K, device=hidden_states.device,
                                     dtype=hidden_states.dtype)
        compute_dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32

        for s0 in range(0, S, chunk_size):
            s1 = min(s0 + chunk_size, S)
            chunk_logits = F.linear(hidden_states[:, s0:s1], weight)  # [B, C, V]
            idx = indices[:, s0:s1].long()
            lse = torch.logsumexp(chunk_logits.to(compute_dtype), dim=-1, keepdim=True)
            topk_logits = torch.gather(chunk_logits, dim=-1, index=idx)
            gathered_logps[:, s0:s1] = (topk_logits.to(compute_dtype) - lse).to(
                hidden_states.dtype)
            del chunk_logits

        ctx.save_for_backward(hidden_states, weight, indices)
        ctx.chunk_size = chunk_size
        return gathered_logps

    @staticmethod
    def backward(ctx, grad_output):
        """
        Chain rule through log_softmax(z)[k] and z = hidden @ weight.T:

        d/d(hidden) = d/dz @ weight
        d/d(weight) = d/dz.T @ hidden

        where d/dz[j] = go[j] (if j in indices) - softmax(z)[j] * sum(go)
        (same as _ChunkedLogSoftmaxGather backward).
        """
        hidden_states, weight, indices = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        B, S, H = hidden_states.shape
        V = weight.shape[0]

        compute_dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32
        out_dtype = hidden_states.dtype

        grad_hidden = torch.empty(B, S, H, device=hidden_states.device, dtype=out_dtype)
        grad_weight = torch.zeros(V, H, device=weight.device, dtype=compute_dtype)

        for s0 in range(0, S, chunk_size):
            s1 = min(s0 + chunk_size, S)
            h_chunk = hidden_states[:, s0:s1]  # [B, C, H] view

            # Recompute logits for this chunk
            chunk_logits = F.linear(h_chunk, weight)  # [B, C, V]

            # Softmax gradient (in-place on softmax tensor to save memory)
            softmax_chunk = torch.softmax(chunk_logits.to(compute_dtype), dim=-1)
            del chunk_logits

            go = grad_output[:, s0:s1].to(compute_dtype)  # [B, C, K]
            go_sum = go.sum(dim=-1, keepdim=True)           # [B, C, 1]
            idx = indices[:, s0:s1].long()

            # grad_logits = scatter(go at indices) - softmax * sum(go)
            softmax_chunk.mul_(-go_sum)        # in-place: -softmax * sum(go)
            softmax_chunk.scatter_add_(-1, idx, go)  # add go at index positions
            # softmax_chunk is now grad_logits [B, C, V] in compute_dtype

            # Gradient through F.linear: grad_hidden = grad_logits @ weight
            grad_hidden[:, s0:s1] = (softmax_chunk @ weight.to(compute_dtype)).to(out_dtype)

            # grad_weight += hidden.T @ grad_logits (accumulated over chunks)
            h_flat = h_chunk.reshape(-1, H).to(compute_dtype)       # [B*C, H]
            gl_flat = softmax_chunk.reshape(-1, V)                   # [B*C, V]
            grad_weight += gl_flat.T @ h_flat                        # [V, H]

            del softmax_chunk, gl_flat, h_flat

        return grad_hidden, grad_weight.to(weight.dtype), None, None


def chunked_lm_head_gather(hidden_states, weight, indices, chunk_size=KL_CHUNK_SIZE):
    """Memory-efficient fused lm_head + log-softmax + gather.

    Replaces the pattern: log_softmax(lm_head(hidden_states)) gathered at indices.
    Never materializes full [B, S, V] logits tensor.

    Args:
        hidden_states: [B, S, H] — last hidden states from transformer backbone
        weight: [V, H] — lm_head weight matrix (nn.Linear.weight)
        indices: [B, S, K] — token indices to gather (int32 or int64)
        chunk_size: sequence chunk size (default 1024). 0 = no chunking (full sequence).

    Returns:
        [B, S, K] — log_softmax(hidden @ weight.T) gathered at indices
    """
    if chunk_size <= 0:
        chunk_size = hidden_states.shape[1]  # full sequence = no chunking
    return _ChunkedLMHeadGather.apply(hidden_states, weight, indices, chunk_size)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_kl_loss(student_logits=None, teacher_topk_logps=None,
                    teacher_topk_indices=None,
                    mask=None,
                    teacher_token_logps=None, input_ids=None,
                    student_old_logprobs=None,
                    student_old_topk_logps=None,
                    student_topk_logps=None, student_token_logps=None,
                    student_dense_logps=None, teacher_dense_logps=None,
                    mc_sample_indices=None,
                    mc_teacher_logprobs=None,
                    mc_old_logprobs=None,
                    student_mc_logprobs=None,
                    candidate_token_ids=None,
                    student_candidate_logprobs=None,
                    teacher_candidate_logprobs=None,
                    candidate_mask=None,
                    student_eos_logprobs=None,
                    teacher_eos_logprobs=None,
                    eos_token_id=None,
                    kl_config: "KLConfig | None" = None,
                    prox_logprobs=None) -> "torch.Tensor":
    """Compute KL loss with the specified mode.

    Supports two input paths:
    - Eval/test path: pass student_logits (full [B, S, V] logits)
    - Trainer path: pass student_topk_logps [B, S, K] (top-k modes) or
      student_token_logps [B, S-1] (token-level/PG modes) to skip gather

    Configuration is passed via ``kl_config``. When omitted, the default
    ``KLConfig()`` is used.

    Args:
        student_logits: [batch, seq_len, vocab_size] — full logits (eval/test)
        student_topk_logps: [batch, seq_len, K] — pre-gathered logprobs (trainer, top-k modes)
        student_token_logps: [batch, shifted_len] — pre-gathered logprobs (trainer, token modes)
        student_mc_logprobs: [batch, seq_len, N] — pre-gathered logprobs (trainer, MC mode)
        student_candidate_logprobs: [batch, shifted_len, K] — explicit MOF candidate logprobs
        teacher_candidate_logprobs: [batch, shifted_len, K] — teacher logprobs on the same candidates
        candidate_token_ids: [batch, shifted_len, K] — token IDs for candidate logprobs
        candidate_mask: [batch, shifted_len, K] bool — valid candidate slots
        teacher_topk_logps: [batch, seq_len, topk] log-probs
        teacher_topk_indices: [batch, seq_len, topk] token indices
        mask: [batch, seq_len] bool — positions to include
        teacher_token_logps: [batch, seq_len] log π_teacher — for token_level_kl/PG
        input_ids: [batch, seq_len] token IDs — for token_level_kl/PG
        student_old_logprobs: [batch, resp_len] log π_old — for policy_gradient_kl
        student_old_topk_logps: [batch, seq_len, K] rollout student support logprobs
        mc_sample_indices: [batch, seq_len, N] sampled MC token IDs
        mc_teacher_logprobs: [batch, seq_len, N] teacher logprobs at sampled MC tokens
        mc_old_logprobs: [batch, seq_len, N] old-policy logprobs at sampled MC tokens
        kl_config: optional KLConfig — defaults to ``KLConfig()``
    """
    kl_config = resolve_kl_config(kl_config)
    mode = kl_config.mode
    skew_alpha = kl_config.skew_alpha
    pg_clip_eps = kl_config.pg_clip_eps
    use_importance_sampling = kl_config.use_importance_sampling
    pg_online_advantage = kl_config.pg_online_advantage
    token_clip = kl_config.token_clip
    use_decoupled_loss = kl_config.use_decoupled_loss
    behave_imp_weight_cap = kl_config.behave_imp_weight_cap
    m2po_budget = kl_config.pg_m2po_budget
    m2po_miniclip_low = kl_config.pg_m2po_miniclip_low
    m2po_miniclip_high = kl_config.pg_m2po_miniclip_high
    if student_dense_logps is not None or teacher_dense_logps is not None:
        if student_dense_logps is None or teacher_dense_logps is None:
            raise ValueError("dense KL path requires both student_dense_logps and teacher_dense_logps")
        return dense_aligned_kl(
            student_logps=student_dense_logps,
            teacher_logps=teacher_dense_logps,
            mask=mask,
            mode=mode,
            alpha=skew_alpha,
            token_clip=token_clip,
        )

    if mode == "forward_kl":
        return sparse_forward_kl(student_logits=student_logits,
                                 teacher_topk_logps=teacher_topk_logps,
                                 teacher_topk_indices=teacher_topk_indices,
                                 mask=mask, token_clip=token_clip,
                                 student_topk_logps=student_topk_logps)
    elif mode in ("reverse_kl", "reverse_kl_rollout_student_topk"):
        return sparse_reverse_kl(student_logits=student_logits,
                                 teacher_topk_logps=teacher_topk_logps,
                                 teacher_topk_indices=teacher_topk_indices,
                                 mask=mask, token_clip=token_clip,
                                 student_topk_logps=student_topk_logps)
    elif mode == "thunlp_opd_default_loss":
        assert student_topk_logps is not None, (
            "thunlp_opd_default_loss requires student_topk_logps"
        )
        assert teacher_topk_logps is not None, (
            "thunlp_opd_default_loss requires teacher_topk_logps"
        )
        if use_importance_sampling:
            assert student_old_topk_logps is not None, (
                "thunlp_opd_default_loss requires student_old_topk_logps"
            )
        return thunlp_opd_default_loss(
            student_topk_logps=student_topk_logps,
            teacher_topk_logps=teacher_topk_logps,
            student_old_topk_logps=student_old_topk_logps,
            mask=mask,
            clip_eps=pg_clip_eps,
            use_importance_sampling=use_importance_sampling,
            online_advantage=pg_online_advantage,
        )
    elif mode == "skewed_kl":
        return sparse_skewed_kl(student_logits=student_logits,
                                teacher_topk_logps=teacher_topk_logps,
                                teacher_topk_indices=teacher_topk_indices,
                                mask=mask, alpha=skew_alpha,
                                token_clip=token_clip,
                                student_topk_logps=student_topk_logps)
    elif mode == "token_level_kl":
        assert teacher_token_logps is not None, "token_level_kl requires teacher_token_logps"
        return token_level_kl(student_logits=student_logits,
                              teacher_token_logps=teacher_token_logps,
                              input_ids=input_ids, mask=mask,
                              student_token_logps=student_token_logps)
    elif mode == "policy_gradient_kl":
        assert teacher_token_logps is not None, "policy_gradient_kl requires teacher_token_logps"
        if use_importance_sampling:
            assert student_old_logprobs is not None, "policy_gradient_kl requires student_old_logprobs"
        return policy_gradient_kl(student_logits=student_logits,
                                  teacher_token_logps=teacher_token_logps,
                                  input_ids=input_ids,
                                  student_old_logprobs=student_old_logprobs,
                                  mask=mask, clip_eps=pg_clip_eps,
                                  student_token_logps=student_token_logps,
                                  use_importance_sampling=use_importance_sampling,
                                  online_advantage=pg_online_advantage,
                                  use_decoupled_loss=use_decoupled_loss,
                                  prox_logprobs=prox_logprobs,
                                  behave_imp_weight_cap=behave_imp_weight_cap,
                                  m2po_budget=m2po_budget,
                                  m2po_miniclip_low=m2po_miniclip_low,
                                  m2po_miniclip_high=m2po_miniclip_high)
    elif mode == "multi_sample_policy_gradient_kl":
        assert mc_teacher_logprobs is not None, \
            "multi_sample_policy_gradient_kl requires mc_teacher_logprobs"
        if use_importance_sampling:
            assert mc_old_logprobs is not None, \
                "multi_sample_policy_gradient_kl requires mc_old_logprobs"
        if student_logits is None:
            assert student_mc_logprobs is not None, \
                "multi_sample_policy_gradient_kl requires student_logits or student_mc_logprobs"
        else:
            assert mc_sample_indices is not None, \
                "multi_sample_policy_gradient_kl requires mc_sample_indices with student_logits"
        return multi_sample_policy_gradient_kl(
            student_logits=student_logits,
            mc_sample_indices=mc_sample_indices,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mc_old_logprobs=mc_old_logprobs,
            mask=mask,
            clip_eps=pg_clip_eps,
            student_mc_logprobs=student_mc_logprobs,
            use_importance_sampling=use_importance_sampling,
            online_advantage=pg_online_advantage,
            use_decoupled_loss=use_decoupled_loss,
            prox_logprobs=prox_logprobs,
            behave_imp_weight_cap=behave_imp_weight_cap,
            m2po_budget=m2po_budget,
            m2po_miniclip_low=m2po_miniclip_low,
            m2po_miniclip_high=m2po_miniclip_high,
        )
    elif mode == "multi_sample_forward_kl":
        assert mc_teacher_logprobs is not None, \
            "multi_sample_forward_kl requires mc_teacher_logprobs"
        if student_logits is None:
            assert student_mc_logprobs is not None, \
                "multi_sample_forward_kl requires student_logits or student_mc_logprobs"
        else:
            assert mc_sample_indices is not None, \
                "multi_sample_forward_kl requires mc_sample_indices with student_logits"
        return multi_sample_forward_kl(
            student_logits=student_logits,
            mc_sample_indices=mc_sample_indices,
            mc_teacher_logprobs=mc_teacher_logprobs,
            mask=mask,
            student_mc_logprobs=student_mc_logprobs,
        )
    elif mode == "mof_opd":
        # Canonical explicit candidate interface.  Legacy MC aliases are
        # accepted as a bridge for rollout/teacher handoff paths.
        if student_candidate_logprobs is None and student_mc_logprobs is not None:
            student_candidate_logprobs = student_mc_logprobs
            teacher_candidate_logprobs = mc_teacher_logprobs
            if mc_sample_indices is not None:
                candidate_token_ids = (
                    mc_sample_indices
                    if mc_sample_indices.shape[:2] == student_mc_logprobs.shape[:2]
                    else mc_sample_indices[:, 1:]
                )
        elif student_candidate_logprobs is None and student_token_logps is not None:
            student_candidate_logprobs = student_token_logps.unsqueeze(-1)
            teacher_candidate_logprobs = teacher_token_logps.unsqueeze(-1)
            if candidate_token_ids is None and input_ids is not None:
                candidate_token_ids = input_ids[:, 1:].unsqueeze(-1)
        elif student_candidate_logprobs is None:
            if student_logits is None:
                raise ValueError(
                    "mof_opd requires student_logits, student_candidate_logprobs, "
                    "student_mc_logprobs, or student_token_logps"
                )
            if mc_sample_indices is not None and mc_teacher_logprobs is not None:
                candidate_token_ids = mc_sample_indices[:, 1:]
                student_candidate_logprobs = chunked_log_softmax_gather(
                    student_logits[:, :-1], candidate_token_ids
                )
                teacher_candidate_logprobs = mc_teacher_logprobs[:, 1:]
                candidate_mask = (
                    candidate_mask[:, 1:]
                    if candidate_mask is not None and candidate_mask.shape[:2] == mc_sample_indices.shape[:2]
                    else candidate_mask
                )
                mask = mask[:, 1:] if mask is not None and mask.shape[1] == student_logits.shape[1] else mask
            elif input_ids is not None and teacher_token_logps is not None:
                target_ids = input_ids[:, 1:].unsqueeze(-1)
                student_candidate_logprobs = chunked_log_softmax_gather(
                    student_logits[:, :-1], target_ids
                )
                teacher_candidate_logprobs = teacher_token_logps[:, :-1].unsqueeze(-1)
                candidate_token_ids = target_ids
                mask = mask[:, 1:] if mask is not None and mask.shape[1] == student_logits.shape[1] else mask
            else:
                raise ValueError(
                    "mof_opd full-logits path requires either MC samples "
                    "(mc_sample_indices/mc_teacher_logprobs) or generated-token "
                    "(input_ids/teacher_token_logps) inputs"
                )
        if teacher_candidate_logprobs is None:
            raise ValueError("mof_opd requires teacher_candidate_logprobs")
        return mof_opd_loss(
            student_candidate_logprobs=student_candidate_logprobs,
            teacher_candidate_logprobs=teacher_candidate_logprobs,
            candidate_token_ids=candidate_token_ids,
            candidate_mask=candidate_mask,
            mask=mask,
            student_eos_logprobs=student_eos_logprobs,
            teacher_eos_logprobs=teacher_eos_logprobs,
            eos_token_id=eos_token_id,
            variant=kl_config.mof_variant,
            partition=kl_config.mof_partition,
            eta_mass=kl_config.mof_eta_mass,
            eta_odds=kl_config.mof_eta_odds,
            lambda_odds=kl_config.mof_lambda_odds,
            eps=kl_config.mof_eps,
            deduplicate_candidates=kl_config.mof_deduplicate_candidates,
        )
    else:
        raise ValueError(f"Unknown kl_loss_mode: {mode!r}. "
                         f"Choose from: forward_kl, reverse_kl, reverse_kl_rollout_student_topk, "
                         f"thunlp_opd_default_loss, "
                         f"skewed_kl, token_level_kl, "
                         f"policy_gradient_kl, multi_sample_policy_gradient_kl, "
                         f"multi_sample_forward_kl, mof_opd")


# ---------------------------------------------------------------------------
# Token-level KL: uses only teacher's logprob at the sampled token
# ---------------------------------------------------------------------------

def token_level_kl(student_logits=None, teacher_token_logps=None, input_ids=None,
                   mask=None, student_token_logps=None):
    """Per-token forward KL contribution using only the teacher's logprob at
    the actual (sampled) token.

    At each position t, computes:
        p_t(y_t) * [log p_t(y_t) - log p_s(y_t)]

    where y_t = input_ids[t+1] is the actual next token.

    Args:
        student_logits: [batch, seq_len, vocab_size] — full logits (eval/test path)
        student_token_logps: [batch, seq_len-1] — pre-gathered shifted logprobs (trainer path)
        teacher_token_logps: [batch, seq_len] — log π_teacher(actual_token)
        input_ids: [batch, seq_len] — token IDs of the sequence
        mask: [batch, seq_len] bool — response positions to include
    """
    if student_token_logps is not None:
        # Trainer path: logprobs already gathered and shifted
        t_logps = teacher_token_logps  # already [batch, seq_len-1] from caller
        teacher_probs = torch.exp(t_logps).detach()
        per_token_kl = teacher_probs * (t_logps.detach() - student_token_logps)
        # mask already shifted by caller
        m = mask
    else:
        # Eval/test path: full logits, need shifting + gather
        target_ids = input_ids[:, 1:]  # [batch, seq_len-1]
        logits = student_logits[:, :-1]  # [batch, seq_len-1, vocab]
        t_logps = teacher_token_logps[:, :-1]  # [batch, seq_len-1]

        student_token_logps = chunked_log_softmax_gather(
            logits, target_ids.unsqueeze(-1)).squeeze(-1)

        teacher_probs = torch.exp(t_logps).detach()
        per_token_kl = teacher_probs * (t_logps.detach() - student_token_logps)
        # Shift mask for eval path
        m = mask[:, 1:] if mask is not None else None

    if m is not None:
        masked = per_token_kl[m]
        loss = masked.mean() if masked.numel() > 0 else per_token_kl.new_zeros((), requires_grad=True)
    else:
        loss = per_token_kl.mean()
    loss.kl_stats = {"_vals": per_token_kl[m].detach() if m is not None else per_token_kl.detach().flatten()}
    return loss


# ---------------------------------------------------------------------------
# M2PO: Second-moment trust region for dynamic clip bounds
# Reference: arxiv.org/abs/2510.01161
# Ported from: M2PO/verl/trainer/ppo/core_algos.py (lines 683-809)
# ---------------------------------------------------------------------------

def _m2po_trust_region_delta_sq(old_logps, new_logps, advantages, mask):
    """Identify harmful trust-region tokens and return their δ² values.

    Harmful tokens are those where PPO clipping would activate:
      (advantage > 0 AND ratio > 1) OR (advantage < 0 AND ratio < 1)

    Args:
        old_logps: [B, T] log π_old (behavior policy)
        new_logps: [B, T] log π_new (current policy)
        advantages: [B, T] per-token advantages
        mask: [B, T] response mask (bool)

    Returns:
        1-D tensor of δ² values for harmful tokens only
    """
    delta = old_logps - new_logps          # δ = log π_old - log π_new
    ratio = torch.exp(-delta)              # r = π_new / π_old
    delta_sq = delta.pow(2)

    m = mask.bool()
    harmful = ((advantages > 0) & (ratio > 1.0)) | ((advantages < 0) & (ratio < 1.0))
    harmful = harmful & m

    return delta_sq[harmful]


def _m2po_solve_tau(sorted_delta2, target_sum):
    """Find threshold τ such that Σ min(δᵢ², τ²) ≈ target_sum.

    Single-pass over sorted ascending δ² values (no binary search).

    Args:
        sorted_delta2: 1-D tensor of δ² values sorted ascending
        target_sum: M2_budget × num_harmful_tokens

    Returns:
        (tau, m2_after): threshold τ and achieved M2 after constraint
    """
    if sorted_delta2.numel() == 0:
        return 1e5, 0.0

    total = sorted_delta2.sum().item()
    if target_sum >= total - 1e-12:
        return 1e5, total / sorted_delta2.numel()
    if target_sum <= 1e-12:
        return 0.0, 0.0

    csum = torch.cumsum(sorted_delta2, dim=0)
    n = sorted_delta2.numel()

    for k in range(n):
        left_sum = csum[k].item()
        rest = n - k - 1
        thresh = sorted_delta2[k].item() - 1e-12
        if thresh * rest + left_sum >= target_sum - 1e-12:
            if k == 0:
                return 0.0, total / n
            m2_after = (sorted_delta2[k - 1].item() * (rest + 1) + csum[k - 1].item()) / n
            return (sorted_delta2[k - 1].item() - 1e-12) ** 0.5, m2_after

    return 1e5, total / n


def _m2po_clip_bounds(old_logps, new_logps, advantages, mask, m2_budget):
    """Compute dynamic (clip_low, clip_high) from M2 budget constraint.

    Args:
        old_logps, new_logps, advantages, mask: see _m2po_trust_region_delta_sq
        m2_budget: second-moment budget per token (e.g., 0.04)

    Returns:
        (clip_low, clip_high, m2_before, m2_after)
        clip_low/high are margins: ratio is clamped to [1 - clip_low, 1 + clip_high]
    """
    import math
    tr_delta_sq = _m2po_trust_region_delta_sq(old_logps, new_logps, advantages, mask)
    n = tr_delta_sq.numel()

    if n == 0:
        return 0.0, 1e5, 0.0, 0.0

    m2_before = tr_delta_sq.sum().item() / n
    if m2_before <= m2_budget + 1e-12:
        return 0.0, 1e5, m2_before, m2_before

    sorted_d2, _ = torch.sort(tr_delta_sq)
    tau, m2_after = _m2po_solve_tau(sorted_d2, m2_budget * n)

    # Map τ to ratio clip margins
    clip_low = 1.0 - math.exp(-tau)    # for (adv < 0, r < 1) quadrant
    clip_high = math.exp(tau) - 1.0    # for (adv > 0, r > 1) quadrant
    return clip_low, clip_high, m2_before, m2_after


# ---------------------------------------------------------------------------
# Policy gradient KL: PPO-clip with reverse KL as advantage
# ---------------------------------------------------------------------------

def policy_gradient_kl(student_logits=None, teacher_token_logps=None,
                       input_ids=None, student_old_logprobs=None,
                       mask=None, clip_eps=None,
                       student_token_logps=None,
                       use_importance_sampling=None,
                       online_advantage=None,
                       use_decoupled_loss=None,
                       prox_logprobs=None,
                       behave_imp_weight_cap=None,
                       m2po_budget=None,
                       m2po_miniclip_low=None,
                       m2po_miniclip_high=None):
    """PPO-clip style policy gradient with per-token reverse KL as advantage.

    Thin wrapper around ppo_clip_loss — handles PG-KL-specific input preparation
    (logprob gathering, advantage computation, old-logprob alignment) and attaches
    stats as loss.pg_stats.

    Matches the approach from G-OPD (Eq. 6) and Thinking Machines:
      advantage_t = log π_teacher(y_t) - log π_old(y_t)
      ratio_t = π_θ(y_t) / π_old(y_t)
      loss = -min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)

    Args:
        student_logits: [batch, seq_len, vocab_size] — full logits (eval/test path)
        student_token_logps: [batch, shifted_len] — pre-gathered shifted logprobs (trainer path)
        teacher_token_logps: [batch, seq_len] or [batch, shifted_len] — log π_teacher
        input_ids: [batch, seq_len] — token IDs
        student_old_logprobs: [batch, shifted_len] — log π_old (π_behav), pre-aligned
        mask: [batch, seq_len] or [batch, shifted_len] — response positions
        clip_eps: PPO clipping epsilon (default 0.2)
        online_advantage: if True, use detached current student logprobs as
            advantage baseline instead of rollout logprobs
        use_decoupled_loss: if True, use decoupled PPO with prox_logprobs
        prox_logprobs: [batch, shifted_len] — log π_prox (recomputed at training time)
        behave_imp_weight_cap: cap for behavioral importance weight (default 5.0)
        m2po_budget: M2PO second-moment budget (None = disabled)
        m2po_miniclip_low: M2PO floor for dynamic clip low margin
        m2po_miniclip_high: M2PO floor for dynamic clip high margin
    """
    from opd.loss.ppo import ppo_clip_loss

    if clip_eps is None:
        clip_eps = _OPD_DEFAULTS.pg_clip_eps
    if use_importance_sampling is None:
        use_importance_sampling = _OPD_DEFAULTS.use_importance_sampling
    if online_advantage is None:
        online_advantage = _OPD_DEFAULTS.pg_online_advantage
    if use_decoupled_loss is None:
        use_decoupled_loss = _OPD_DEFAULTS.use_decoupled_loss
    if behave_imp_weight_cap is None:
        behave_imp_weight_cap = _OPD_DEFAULTS.behave_imp_weight_cap
    if m2po_miniclip_low is None:
        m2po_miniclip_low = _OPD_DEFAULTS.pg_m2po_miniclip_low
    if m2po_miniclip_high is None:
        m2po_miniclip_high = _OPD_DEFAULTS.pg_m2po_miniclip_high

    # ---- PG-KL-specific input preparation ----
    if student_token_logps is not None:
        # Trainer path: logprobs already gathered, shifted, and aligned
        student_new_logps = student_token_logps
        t_logps = teacher_token_logps  # already shifted by caller
        old_logps = student_old_logprobs  # already aligned by caller
        m = mask  # already shifted by caller
    else:
        # Eval/test path: full logits, need shifting + gather + alignment
        target_ids = input_ids[:, 1:]       # [batch, seq_len-1]
        logits = student_logits[:, :-1]     # [batch, seq_len-1, vocab]
        t_logps = teacher_token_logps[:, :-1]  # [batch, seq_len-1]

        student_new_logps = chunked_log_softmax_gather(
            logits, target_ids.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len-1]

        bs, shifted_len = student_new_logps.shape
        old_logps = None
        if student_old_logprobs is not None:
            resp_len = student_old_logprobs.size(1)
            old_logps = torch.zeros(bs, shifted_len, device=student_new_logps.device,
                                    dtype=student_old_logprobs.dtype)
            usable_resp = min(resp_len, shifted_len)
            old_logps[:, -usable_resp:] = student_old_logprobs[:, :usable_resp]
        m = mask[:, 1:] if mask is not None else None

    if old_logps is None:
        old_logps = student_new_logps.detach()

    # Advantage: per-token KL (detached, no grad)
    if online_advantage or not use_importance_sampling:
        advantage = (t_logps - student_new_logps).detach()
    else:
        advantage = (t_logps - old_logps).detach()

    # ---- Call shared PPO-clip core ----
    loss, raw_stats = ppo_clip_loss(
        student_new_logps, old_logps, advantage,
        m if m is not None else torch.ones_like(student_new_logps, dtype=torch.bool),
        clip_eps=clip_eps,
        use_importance_sampling=use_importance_sampling,
        use_decoupled_loss=use_decoupled_loss,
        prox_logprobs=prox_logprobs,
        behave_imp_weight_cap=behave_imp_weight_cap,
        m2po_budget=m2po_budget,
        m2po_miniclip_low=m2po_miniclip_low,
        m2po_miniclip_high=m2po_miniclip_high,
    )

    # ---- Attach stats as loss.pg_stats (PG-KL convention) ----
    loss.pg_stats = raw_stats
    return loss


def thunlp_opd_default_loss(
    *,
    student_topk_logps,
    teacher_topk_logps,
    student_old_topk_logps,
    mask,
    clip_eps=None,
    use_importance_sampling=None,
    online_advantage=None,
):
    """Faithful THUNLP-default detached 3D top-k PPO loss.

    Thin wrapper around the THUNLP-default top-k OPD update shape — handles
    support-axis reward construction, offline/online detached advantage
    selection, and attaches PPO-style stats as ``loss.pg_stats``.

    Matches the intended THUNLP-default behavior:
      p_old_t(k) = softmax(log π_base_t(k)) over the top-k support
      reward_t(k) = -[log π_base_t(k) - log π_teacher_t(k)] * p_old_t(k)
      ratio_t(k) = π_θ_t(k) / π_old_t(k)
      loss = PPO-clip(ratio_t(k), reward_t(k)) reduced with token-mean

    where ``π_base`` is:
      - rollout-old support logprobs when ``online_advantage=False``
      - detached current support logprobs when ``online_advantage=True``

    Args:
        student_topk_logps: [batch, seq_len, K] — current student logprobs on
            the rollout support ids (trainer path)
        teacher_topk_logps: [batch, seq_len, K] — teacher logprobs on the same
            rollout support ids
        student_old_topk_logps: [batch, seq_len, K] — rollout-time student
            support logprobs used as the old policy for PPO ratios
        mask: [batch, seq_len] bool — response-token positions to include
        clip_eps: symmetric PPO clipping epsilon (default 0.2)
        online_advantage: if True, build detached THUNLP rewards from the
            detached current student support logprobs rather than the rollout
            old support logprobs; PPO ratios remain anchored to
            ``student_old_topk_logps`` either way
    """
    if clip_eps is None:
        clip_eps = _OPD_DEFAULTS.pg_clip_eps
    if use_importance_sampling is None:
        use_importance_sampling = _OPD_DEFAULTS.use_importance_sampling
    if online_advantage is None:
        online_advantage = _OPD_DEFAULTS.pg_online_advantage

    # THUNLP default PPO branch uses clip_ratio_c=3.0.
    clip_ratio_c = 3.0

    old_topk_logps = (
        student_old_topk_logps.float()
        if student_old_topk_logps is not None
        else student_topk_logps.detach().float()
    )
    new_topk_logps = student_topk_logps.float()
    teacher_topk_logps = teacher_topk_logps.float()
    mask_3d = mask.unsqueeze(-1).expand_as(new_topk_logps)

    advantage_logps = new_topk_logps.detach() if online_advantage else old_topk_logps
    support_probs = torch.softmax(advantage_logps, dim=-1)
    advantages = (-(advantage_logps - teacher_topk_logps) * support_probs).detach()

    ratio_base = old_topk_logps if use_importance_sampling else new_topk_logps.detach()
    negative_approx_kl = (new_topk_logps - ratio_base).clamp(-20.0, 20.0)
    ratio = negative_approx_kl.exp()

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    # Faithful token-mean reduction: sum over support first, then mean over valid positions.
    pg_losses = pg_losses.sum(dim=-1)

    masked_loss = pg_losses[mask]
    if masked_loss.numel() == 0:
        loss = pg_losses.new_zeros((), requires_grad=True)
    else:
        loss = masked_loss.mean()

    with torch.no_grad():
        loss.pg_stats = {
            "_ratios": ratio[mask_3d].detach().cpu(),
            "_log_ratios": negative_approx_kl[mask_3d].detach().cpu(),
            "_advantages": advantages[mask_3d].detach().cpu(),
            "_clip_high": (ratio > 1.0 + clip_eps)[mask_3d].detach().cpu(),
            "_clip_low": (ratio < 1.0 - clip_eps)[mask_3d].detach().cpu(),
        }
    return loss


def _mof_first_occurrence_mask(candidate_token_ids, candidate_mask):
    """Keep only the first valid occurrence of each candidate token per row.

    Duplicate MC candidates can appear after teacher/rollout sampling. MOF
    treats the candidate support as a set, so only the first valid occurrence is
    retained; if duplicate slots carry inconsistent logprobs, the earlier slot
    wins deterministically.
    """
    if candidate_token_ids is None:
        return candidate_mask
    candidate_token_ids = candidate_token_ids.long()
    keep = candidate_mask.clone().bool()
    for j in range(1, candidate_token_ids.size(-1)):
        prev_same = (
            (candidate_token_ids[..., :j] == candidate_token_ids[..., j:j + 1])
            & candidate_mask[..., :j]
        ).any(dim=-1)
        keep[..., j] &= ~prev_same
    return keep


def _mof_normalize_masses(masses, eps):
    masses = masses.clamp_min(eps)
    return masses / masses.sum(dim=-1, keepdim=True).clamp_min(eps)


def _mof_softmax_over_candidates(logps, valid_mask):
    masked_logps = logps.masked_fill(~valid_mask, -torch.inf)
    weights = torch.softmax(masked_logps.float(), dim=-1).to(logps.dtype)
    return torch.where(valid_mask, weights, torch.zeros_like(weights))


def mof_opd_loss(
    *,
    student_candidate_logprobs,
    teacher_candidate_logprobs,
    candidate_token_ids=None,
    candidate_mask=None,
    mask=None,
    student_eos_logprobs=None,
    teacher_eos_logprobs=None,
    eos_token_id=None,
    variant=None,
    partition=None,
    eta_mass=None,
    eta_odds=None,
    lambda_odds=None,
    eps=None,
    deduplicate_candidates=None,
):
    """Mass-and-odds factored OPD loss over an explicit candidate interface."""
    if variant is None:
        variant = _OPD_DEFAULTS.mof_variant
    if partition is None:
        partition = _OPD_DEFAULTS.mof_partition
    if eta_mass is None:
        eta_mass = _OPD_DEFAULTS.mof_eta_mass
    if eta_odds is None:
        eta_odds = _OPD_DEFAULTS.mof_eta_odds
    if lambda_odds is None:
        lambda_odds = _OPD_DEFAULTS.mof_lambda_odds
    if eps is None:
        eps = _OPD_DEFAULTS.mof_eps
    if deduplicate_candidates is None:
        deduplicate_candidates = _OPD_DEFAULTS.mof_deduplicate_candidates

    if variant not in {"lite", "full"}:
        raise ValueError(f"mof_opd variant must be 'lite' or 'full', got {variant!r}")
    if partition not in {"two_group", "eos_candidate_rest"}:
        raise ValueError(
            "mof_opd partition must be 'two_group' or 'eos_candidate_rest', "
            f"got {partition!r}"
        )
    if student_candidate_logprobs.shape != teacher_candidate_logprobs.shape:
        raise ValueError(
            "student_candidate_logprobs and teacher_candidate_logprobs must have "
            f"the same shape, got {tuple(student_candidate_logprobs.shape)} and "
            f"{tuple(teacher_candidate_logprobs.shape)}"
        )
    if student_candidate_logprobs.dim() != 3:
        raise ValueError(
            "MOF candidate logprobs must be [batch, shifted_len, candidates], "
            f"got {tuple(student_candidate_logprobs.shape)}"
        )

    device = student_candidate_logprobs.device
    student_logps = student_candidate_logprobs.float()
    teacher_logps = teacher_candidate_logprobs.to(device=device).float()

    if candidate_mask is None:
        candidate_mask = torch.ones_like(student_logps, dtype=torch.bool)
    else:
        candidate_mask = candidate_mask.to(device=device).bool()
        if candidate_mask.dim() == 2:
            candidate_mask = candidate_mask.unsqueeze(-1).expand_as(student_logps)
        elif candidate_mask.shape != student_logps.shape:
            raise ValueError(
                "MOF candidate_mask must be [batch, shifted_len] or "
                "[batch, shifted_len, candidates], got "
                f"{tuple(candidate_mask.shape)} for candidates {tuple(student_logps.shape)}"
            )
    if mask is None:
        mask = torch.ones(student_logps.shape[:2], device=device, dtype=torch.bool)
    else:
        mask = mask.to(device=device).bool()
        if mask.shape != student_logps.shape[:2]:
            raise ValueError(
                "MOF mask must be [batch, shifted_len], got "
                f"{tuple(mask.shape)} for candidates {tuple(student_logps.shape)}"
            )
    if candidate_token_ids is not None and candidate_token_ids.shape != student_logps.shape:
        raise ValueError(
            "MOF candidate_token_ids must match candidate logprob shape, got "
            f"{tuple(candidate_token_ids.shape)} for candidates {tuple(student_logps.shape)}"
        )

    candidate_mask = candidate_mask & torch.isfinite(student_logps) & torch.isfinite(teacher_logps)

    eos_mask = torch.zeros_like(candidate_mask)
    if partition == "eos_candidate_rest":
        if eos_token_id is None:
            raise ValueError("mof_opd eos_candidate_rest requires eos_token_id")
        if candidate_token_ids is not None:
            eos_mask = candidate_token_ids.to(device=device).long() == int(eos_token_id)
        if student_eos_logprobs is None or teacher_eos_logprobs is None:
            if not (eos_mask.any(dim=-1) | ~mask).all():
                raise ValueError(
                    "mof_opd eos_candidate_rest requires EOS logprobs or an EOS "
                    "candidate in every valid row"
                )
            eos_first = eos_mask.float().argmax(dim=-1, keepdim=True)
            student_eos_logprobs = torch.gather(student_logps, -1, eos_first).squeeze(-1)
            teacher_eos_logprobs = torch.gather(teacher_logps, -1, eos_first).squeeze(-1)
        else:
            student_eos_logprobs = student_eos_logprobs.to(device=device).float()
            teacher_eos_logprobs = teacher_eos_logprobs.to(device=device).float()
        candidate_mask = candidate_mask & ~eos_mask

    if deduplicate_candidates:
        candidate_mask = _mof_first_occurrence_mask(candidate_token_ids, candidate_mask)

    valid_any = candidate_mask.any(dim=-1)
    effective_mask = mask & (valid_any if partition == "two_group" else torch.ones_like(mask))

    student_candidate_mass_raw = (
        torch.exp(student_logps).masked_fill(~candidate_mask, 0.0).sum(dim=-1)
    )
    teacher_candidate_mass_raw = (
        torch.exp(teacher_logps).masked_fill(~candidate_mask, 0.0).sum(dim=-1)
    )

    if partition == "two_group":
        student_masses = _mof_normalize_masses(
            torch.stack([student_candidate_mass_raw, 1.0 - student_candidate_mass_raw], dim=-1),
            eps,
        )
        teacher_masses = _mof_normalize_masses(
            torch.stack([teacher_candidate_mass_raw, 1.0 - teacher_candidate_mass_raw], dim=-1),
            eps,
        )
        candidate_group_index = 0
    else:
        student_eos_mass = torch.exp(student_eos_logprobs).clamp_min(0.0)
        teacher_eos_mass = torch.exp(teacher_eos_logprobs).clamp_min(0.0)
        student_masses = _mof_normalize_masses(
            torch.stack(
                [
                    student_eos_mass,
                    student_candidate_mass_raw,
                    1.0 - student_eos_mass - student_candidate_mass_raw,
                ],
                dim=-1,
            ),
            eps,
        )
        teacher_masses = _mof_normalize_masses(
            torch.stack(
                [
                    teacher_eos_mass,
                    teacher_candidate_mass_raw,
                    1.0 - teacher_eos_mass - teacher_candidate_mass_raw,
                ],
                dim=-1,
            ),
            eps,
        )
        candidate_group_index = 1

    target_group_logits = (
        (1.0 - float(eta_mass)) * student_masses.detach().clamp_min(eps).log()
        + float(eta_mass) * teacher_masses.detach().clamp_min(eps).log()
    )
    target_masses = torch.softmax(target_group_logits, dim=-1).to(student_masses.dtype)
    group_loss = -(target_masses.detach() * student_masses.clamp_min(eps).log()).sum(dim=-1)

    student_cond_logps = student_logps - student_candidate_mass_raw.clamp_min(eps).log().unsqueeze(-1)
    teacher_cond_logps = teacher_logps - teacher_candidate_mass_raw.clamp_min(eps).log().unsqueeze(-1)
    odds_target_logits = (
        (1.0 - float(eta_odds)) * student_cond_logps.detach()
        + float(eta_odds) * teacher_cond_logps.detach()
    )
    odds_target = _mof_softmax_over_candidates(odds_target_logits, candidate_mask)
    odds_loss = -(odds_target.detach() * student_cond_logps).masked_fill(~candidate_mask, 0.0).sum(dim=-1)
    odds_loss = torch.where(valid_any, odds_loss, torch.zeros_like(odds_loss))

    candidate_weight = student_masses[..., candidate_group_index].detach()
    if variant == "lite":
        group_component = torch.zeros_like(group_loss)
        per_token_loss = float(lambda_odds) * candidate_weight * odds_loss
    else:
        group_component = group_loss
        per_token_loss = group_component + float(lambda_odds) * candidate_weight * odds_loss

    selected = per_token_loss[effective_mask]
    loss = selected.mean() if selected.numel() > 0 else per_token_loss.sum() * 0.0
    loss.kl_stats = {"_vals": selected.detach().cpu()}

    with torch.no_grad():
        stats_mask = effective_mask

        def masked_mean(x):
            vals = x[stats_mask]
            return vals.mean().detach().cpu() if vals.numel() > 0 else x.new_tensor(0.0).detach().cpu()

        loss.mof_stats = {
            "mof_student_candidate_mass": masked_mean(student_masses[..., candidate_group_index]),
            "mof_teacher_candidate_mass": masked_mean(teacher_masses[..., candidate_group_index]),
            "mof_target_candidate_mass": masked_mean(target_masses[..., candidate_group_index]),
            "mof_candidate_mass_gap": masked_mean(
                (student_masses[..., candidate_group_index] - teacher_masses[..., candidate_group_index]).abs()
            ),
            "mof_group_loss": masked_mean(group_component),
            "mof_odds_loss": masked_mean(odds_loss),
            "mof_total_loss": masked_mean(per_token_loss),
            "mof_num_candidates_mean": masked_mean(candidate_mask.float().sum(dim=-1)),
        }
        if partition == "two_group":
            loss.mof_stats.update({
                "mof_student_rest_mass": masked_mean(student_masses[..., 1]),
                "mof_teacher_rest_mass": masked_mean(teacher_masses[..., 1]),
                "mof_target_rest_mass": masked_mean(target_masses[..., 1]),
            })
        else:
            loss.mof_stats.update({
                "mof_student_eos_mass": masked_mean(student_masses[..., 0]),
                "mof_teacher_eos_mass": masked_mean(teacher_masses[..., 0]),
                "mof_target_eos_mass": masked_mean(target_masses[..., 0]),
                "mof_student_rest_mass": masked_mean(student_masses[..., 2]),
                "mof_teacher_rest_mass": masked_mean(teacher_masses[..., 2]),
                "mof_target_rest_mass": masked_mean(target_masses[..., 2]),
            })

    return loss


def multi_sample_policy_gradient_kl(student_logits=None,
                                    mc_sample_indices=None,
                                    mc_teacher_logprobs=None,
                                    mc_old_logprobs=None,
                                    mask=None,
                                    clip_eps=None,
                                    student_mc_logprobs=None,
                                    use_importance_sampling=None,
                                    online_advantage=None,
                                    use_decoupled_loss=None,
                                    prox_logprobs=None,
                                    behave_imp_weight_cap=None,
                                    m2po_budget=None,
                                    m2po_miniclip_low=None,
                                    m2po_miniclip_high=None):
    """PPO-style PG-KL surrogate over multiple sampled tokens per position.

    This is the multi-sample extension of ``policy_gradient_kl``. Inputs are
    already aligned to the full-sequence trainer contract: each valid response
    position contains ``N`` sampled tokens/logprobs, and prompt/invalid rows are
    masked out by ``mask``.
    """
    from opd.loss.ppo import ppo_clip_loss

    if clip_eps is None:
        clip_eps = _OPD_DEFAULTS.pg_clip_eps
    if use_importance_sampling is None:
        use_importance_sampling = _OPD_DEFAULTS.use_importance_sampling
    if online_advantage is None:
        online_advantage = _OPD_DEFAULTS.pg_online_advantage
    if use_decoupled_loss is None:
        use_decoupled_loss = _OPD_DEFAULTS.use_decoupled_loss
    if behave_imp_weight_cap is None:
        behave_imp_weight_cap = _OPD_DEFAULTS.behave_imp_weight_cap
    if m2po_miniclip_low is None:
        m2po_miniclip_low = _OPD_DEFAULTS.pg_m2po_miniclip_low
    if m2po_miniclip_high is None:
        m2po_miniclip_high = _OPD_DEFAULTS.pg_m2po_miniclip_high

    if student_mc_logprobs is not None:
        student_new_logps = student_mc_logprobs
        t_logps = mc_teacher_logprobs
        old_logps = mc_old_logprobs
        m = mask
    else:
        student_new_logps = chunked_log_softmax_gather(
            student_logits[:, :-1], mc_sample_indices[:, 1:]
        )
        t_logps = mc_teacher_logprobs[:, 1:]
        old_logps = mc_old_logprobs[:, 1:] if mc_old_logprobs is not None else None
        m = mask[:, 1:] if mask is not None else None

    if old_logps is None:
        old_logps = student_new_logps.detach()

    if online_advantage or not use_importance_sampling:
        advantages = (t_logps - student_new_logps).detach()
    else:
        advantages = (t_logps - old_logps).detach()

    loss, raw_stats = ppo_clip_loss(
        student_new_logps,
        old_logps,
        advantages,
        m if m is not None else torch.ones_like(student_new_logps[..., 0], dtype=torch.bool),
        clip_eps=clip_eps,
        use_importance_sampling=use_importance_sampling,
        use_decoupled_loss=use_decoupled_loss,
        prox_logprobs=prox_logprobs,
        behave_imp_weight_cap=behave_imp_weight_cap,
        m2po_budget=m2po_budget,
        m2po_miniclip_low=m2po_miniclip_low,
        m2po_miniclip_high=m2po_miniclip_high,
        sample_axis=-1,
    )
    loss.pg_stats = raw_stats
    return loss


def multi_sample_forward_kl(student_logits=None,
                            mc_sample_indices=None,
                            mc_teacher_logprobs=None,
                            mask=None,
                            student_mc_logprobs=None):
    """MC estimator of forward KL using teacher-sampled tokens per position.

    Estimates E_{y~p_teacher}[log p_teacher(y) - log p_student(y)] by averaging
    over sampled teacher tokens at each valid response position.
    """
    if student_mc_logprobs is not None:
        student_logps = student_mc_logprobs
        t_logps = mc_teacher_logprobs
        m = mask
    else:
        student_logps = chunked_log_softmax_gather(
            student_logits[:, :-1], mc_sample_indices[:, 1:]
        )
        t_logps = mc_teacher_logprobs[:, 1:]
        m = mask[:, 1:] if mask is not None else None

    per_token_kl = (t_logps.detach() - student_logps).mean(dim=-1)

    if m is not None:
        masked = per_token_kl[m]
        loss = masked.mean() if masked.numel() > 0 else per_token_kl.new_zeros((), requires_grad=True)
    else:
        loss = per_token_kl.mean()
    loss.kl_stats = {"_vals": per_token_kl[m].detach() if m is not None else per_token_kl.detach().flatten()}
    return loss


# ---------------------------------------------------------------------------
# Chunked dense KL from hidden states (lazy teacher recompute path)
# ---------------------------------------------------------------------------

def _dense_kl_per_token_from_logps(student_logps, teacher_logps, *, mode, alpha, token_clip):
    if mode == "forward_kl":
        per_vocab_kl = torch.exp(teacher_logps) * (teacher_logps - student_logps)
    elif mode in ("reverse_kl", "reverse_kl_rollout_student_topk"):
        per_vocab_kl = torch.exp(student_logps) * (student_logps - teacher_logps)
    elif mode == "skewed_kl":
        fwd = torch.exp(teacher_logps) * (teacher_logps - student_logps)
        rev = torch.exp(student_logps) * (student_logps - teacher_logps)
        per_vocab_kl = alpha * fwd + (1 - alpha) * rev
    else:
        raise ValueError(f"dense hidden KL does not support mode={mode!r}")
    if token_clip > 0:
        per_vocab_kl = per_vocab_kl.clamp(max=token_clip)
    return per_vocab_kl.sum(dim=-1)


def chunked_dense_kl_from_hidden(
    student_hidden,
    student_lm_head_weight,
    teacher_hidden,
    teacher_lm_head_weight,
    mask=None,
    *,
    mode="forward_kl",
    alpha=0.5,
    token_clip=0.0,
    chunk_size=KL_CHUNK_SIZE,
    memory_strategy="checkpoint",
):
    """Reduce dense KL from student/teacher hidden states in sequence chunks.

    This is the lazy hidden-recompute path: teacher artifacts carry hidden
    states, and the trainer applies both LM heads locally.  The function returns
    only the reduced KL loss (plus small per-token stats); it never returns or
    concatenates full ``[B,S,V]`` log-prob tensors.  With the default
    checkpoint strategy, per-chunk dense logits/log-probs are recomputed during
    backward so retained activation memory is bounded by ``chunk_size``.
    """
    if student_hidden.dim() != 3 or teacher_hidden.dim() != 3:
        raise ValueError(
            f"student_hidden and teacher_hidden must be [B,S,H], got "
            f"{tuple(student_hidden.shape)} and {tuple(teacher_hidden.shape)}"
        )
    if student_hidden.shape[:2] != teacher_hidden.shape[:2]:
        raise ValueError(
            f"student/teacher hidden sequence shape mismatch: "
            f"{tuple(student_hidden.shape[:2])} != {tuple(teacher_hidden.shape[:2])}"
        )
    if chunk_size <= 0:
        chunk_size = int(student_hidden.shape[1])
    chunk_size = max(int(chunk_size), 1)

    teacher_hidden = teacher_hidden.detach().to(
        device=student_hidden.device,
        dtype=teacher_lm_head_weight.dtype,
    )
    teacher_lm_head_weight = teacher_lm_head_weight.detach().to(student_hidden.device)
    student_lm_head_weight = student_lm_head_weight.to(student_hidden.device)

    B, S, _ = student_hidden.shape
    V = int(student_lm_head_weight.size(0))
    per_token_chunks = []
    max_logits_bytes = 0

    def _chunk_reduce(st_h, te_h):
        st_logits = F.linear(st_h, student_lm_head_weight)
        te_logits = F.linear(te_h, teacher_lm_head_weight)
        compute_dtype = torch.float64 if st_logits.dtype == torch.float64 else torch.float32
        student_logps = F.log_softmax(st_logits.to(compute_dtype), dim=-1)
        teacher_logps = F.log_softmax(te_logits.to(compute_dtype), dim=-1).detach()
        return _dense_kl_per_token_from_logps(
            student_logps,
            teacher_logps,
            mode=mode,
            alpha=alpha,
            token_clip=token_clip,
        )

    use_checkpoint = memory_strategy == "checkpoint" and torch.is_grad_enabled()
    if memory_strategy not in {"checkpoint", "none"}:
        raise ValueError(f"unsupported dense hidden KL memory_strategy={memory_strategy!r}")

    for s0 in range(0, S, chunk_size):
        s1 = min(s0 + chunk_size, S)
        st_h = student_hidden[:, s0:s1, :]
        te_h = teacher_hidden[:, s0:s1, :]
        # Two dense projections may exist transiently (student + teacher).  The
        # public API still retains only the reduced [B,C] result for the chunk.
        max_logits_bytes = max(
            max_logits_bytes,
            int(2 * B * (s1 - s0) * V * torch.tensor([], dtype=torch.float32).element_size()),
        )
        if use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            per_token = checkpoint(_chunk_reduce, st_h, te_h, use_reentrant=False)
        else:
            per_token = _chunk_reduce(st_h, te_h)
        per_token_chunks.append(per_token)

    per_token_kl = torch.cat(per_token_chunks, dim=1) if len(per_token_chunks) > 1 else per_token_chunks[0]
    if mask is not None:
        mask = mask.to(device=per_token_kl.device).bool()
        masked = per_token_kl[mask]
        loss = masked.mean() if masked.numel() > 0 else per_token_kl.new_zeros((), requires_grad=True)
    else:
        loss = per_token_kl.mean()
    loss.kl_stats = {
        "_vals": per_token_kl[mask].detach() if mask is not None else per_token_kl.detach().flatten()
    }
    loss.chunked_dense_kl_stats = {
        "teacher_hidden_fused_kl_chunk_size": chunk_size,
        "teacher_hidden_fused_kl_max_logits_bytes": max_logits_bytes,
        "teacher_hidden_materialized_bytes": 0,
        "teacher_hidden_fused_kl_tokens": int(mask.sum().item()) if mask is not None else int(per_token_kl.numel()),
    }
    return loss


# ---------------------------------------------------------------------------
# Dense aligned KL: full-vocab student/teacher logprobs already computed
# ---------------------------------------------------------------------------

def dense_aligned_kl(student_logps, teacher_logps, mask=None, mode="forward_kl",
                     alpha=0.5, token_clip=0.0):
    """KL over aligned full-vocab logprobs without top-k index tensors.

    Args:
        student_logps: [B, S, V] student log-probs.
        teacher_logps: [B, S, V] teacher log-probs on the same vocabulary.
        mask: [B, S] bool positions to include.
        mode: forward_kl, reverse_kl, or skewed_kl.
    """
    if student_logps.shape != teacher_logps.shape:
        raise ValueError(
            f"student/teacher dense logps shape mismatch: "
            f"{tuple(student_logps.shape)} != {tuple(teacher_logps.shape)}"
        )
    student_logps = student_logps.float()
    teacher_logps = teacher_logps.float()
    if mode == "forward_kl":
        per_vocab_kl = torch.exp(teacher_logps) * (teacher_logps - student_logps)
    elif mode in ("reverse_kl", "reverse_kl_rollout_student_topk"):
        per_vocab_kl = torch.exp(student_logps) * (student_logps - teacher_logps)
    elif mode == "skewed_kl":
        fwd = torch.exp(teacher_logps) * (teacher_logps - student_logps)
        rev = torch.exp(student_logps) * (student_logps - teacher_logps)
        per_vocab_kl = alpha * fwd + (1 - alpha) * rev
    else:
        raise ValueError(f"dense_aligned_kl does not support mode={mode!r}")
    if token_clip > 0:
        per_vocab_kl = per_vocab_kl.clamp(max=token_clip)
    per_token_kl = per_vocab_kl.sum(dim=-1)
    if mask is not None:
        masked = per_token_kl[mask]
        loss = masked.mean() if masked.numel() > 0 else per_token_kl.new_zeros((), requires_grad=True)
    else:
        loss = per_token_kl.mean()
    loss.kl_stats = {
        "_vals": per_token_kl[mask].detach() if mask is not None else per_token_kl.detach().flatten()
    }
    return loss


# ---------------------------------------------------------------------------
# Forward KL: KL(teacher || student)
# ---------------------------------------------------------------------------

def sparse_forward_kl(student_logits=None, teacher_topk_logps=None,
                      teacher_topk_indices=None, mask=None, token_clip=0.0,
                      student_topk_logps=None):
    """KL(teacher || student) using sparse teacher top-k logprobs.

    Uses chunked log-softmax gather to avoid materializing full [B, S, V]
    softmax during backward. Peak memory: O(B * 1024 * V) instead of O(B * S * V).

    Args:
        student_logits: [B, S, V] — full logits (eval/test path)
        student_topk_logps: [B, S, K] — pre-gathered logprobs (trainer path, skip gather)
        token_clip: if > 0, clamp per-vocab-element KL before summing over K.
            Matches OPSD's per-element [B, S, V] clipping (not per-token sum).
            Prevents style tokens from dominating gradients (OPSD paper uses 0.05).
    """
    if student_topk_logps is None:
        student_topk_logps = chunked_log_softmax_gather(
            student_logits, teacher_topk_indices)

    teacher_topk_probs = torch.exp(teacher_topk_logps)
    per_vocab_kl = teacher_topk_probs * (teacher_topk_logps - student_topk_logps)  # [B, S, K]

    if token_clip > 0:
        per_vocab_kl = per_vocab_kl.clamp(max=token_clip)

    per_token_kl = per_vocab_kl.sum(dim=-1)

    if mask is not None:
        masked = per_token_kl[mask]
        loss = masked.mean() if masked.numel() > 0 else per_token_kl.new_zeros((), requires_grad=True)
    else:
        loss = per_token_kl.mean()
    loss.kl_stats = {"_vals": per_token_kl[mask].detach() if mask is not None else per_token_kl.detach().flatten()}
    return loss


# ---------------------------------------------------------------------------
# Reverse KL (top-k only): KL(student || teacher)
# ---------------------------------------------------------------------------

def sparse_reverse_kl(student_logits=None, teacher_topk_logps=None,
                      teacher_topk_indices=None, mask=None,
                      student_topk_logps=None, token_clip=0.0):
    """KL(student || teacher) summed over teacher's top-k tokens only.

    Underestimates true reverse KL since it ignores student mass outside top-k,
    but avoids assumptions about the teacher's tail distribution.
    Uses chunked log-softmax gather for memory efficiency.

    Args:
        student_logits: [B, S, V] — full logits (eval/test path)
        student_topk_logps: [B, S, K] — pre-gathered logprobs (trainer path, skip gather)
        token_clip: if > 0, clamp per-vocab-element KL before summing over K.
    """
    if student_topk_logps is None:
        student_topk_logps = chunked_log_softmax_gather(
            student_logits, teacher_topk_indices)
    student_topk_probs = torch.exp(student_topk_logps)

    per_vocab_kl = student_topk_probs * (student_topk_logps - teacher_topk_logps)
    if token_clip > 0:
        per_vocab_kl = per_vocab_kl.clamp(max=token_clip)
    per_token_kl = per_vocab_kl.sum(dim=-1)

    if mask is not None:
        masked = per_token_kl[mask]
        loss = masked.mean() if masked.numel() > 0 else per_token_kl.new_zeros((), requires_grad=True)
    else:
        loss = per_token_kl.mean()
    loss.kl_stats = {"_vals": per_token_kl[mask].detach() if mask is not None else per_token_kl.detach().flatten()}
    return loss


# ---------------------------------------------------------------------------
# Skewed KL: alpha * forward + (1-alpha) * reverse, per-vocab combined clip
# ---------------------------------------------------------------------------

def sparse_skewed_kl(student_logits=None, teacher_topk_logps=None,
                     teacher_topk_indices=None, mask=None,
                     student_topk_logps=None, alpha=0.5, token_clip=0.0):
    """Skewed KL: alpha * KL(teacher||student) + (1-alpha) * KL(student||teacher).

    Computes per-vocab-element forward and reverse KL, combines them, then clips
    the combined value. This matches the trainer's inline behavior where
    clip(alpha*fwd + (1-alpha)*rev) is applied per-vocab-element before summing.

    Args:
        student_logits: [B, S, V] — full logits (eval/test path)
        student_topk_logps: [B, S, K] — pre-gathered logprobs (trainer path)
        alpha: weight for forward KL (0=pure reverse, 1=pure forward)
        token_clip: if > 0, clamp combined per-vocab KL before summing over K.
    """
    if student_topk_logps is None:
        student_topk_logps = chunked_log_softmax_gather(
            student_logits, teacher_topk_indices)

    teacher_topk_probs = torch.exp(teacher_topk_logps)
    student_topk_probs = torch.exp(student_topk_logps)

    fwd_per_vocab = teacher_topk_probs * (teacher_topk_logps - student_topk_logps)
    rev_per_vocab = student_topk_probs * (student_topk_logps - teacher_topk_logps)
    per_vocab_kl = alpha * fwd_per_vocab + (1 - alpha) * rev_per_vocab

    if token_clip > 0:
        per_vocab_kl = per_vocab_kl.clamp(max=token_clip)

    per_token_kl = per_vocab_kl.sum(dim=-1)

    if mask is not None:
        masked = per_token_kl[mask]
        loss = masked.mean() if masked.numel() > 0 else per_token_kl.new_zeros((), requires_grad=True)
    else:
        loss = per_token_kl.mean()
    loss.kl_stats = {"_vals": per_token_kl[mask].detach() if mask is not None else per_token_kl.detach().flatten()}
    return loss


# Megatron vocab-parallel KL removed. Re-add via strategy pattern if needed.

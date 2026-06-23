"""Shared vLLM utilities for rollout workers.

Contains code extracted from opd/rollout/base.py that is vLLM-specific:
- TraceStatLogger (vLLM StatLoggerBase implementation)
- extract_student_logprobs / extract_student_topk_support (3 vLLM output formats)
- Model manipulation functions for vLLM apply_model/collective_rpc
- Shared vLLM init helpers (env setup, LLM kwargs builder, pre-load patches)
"""

import heapq
import os
import sys
import time

import torch

from opd.utils.config import resolve_trust_remote_code


# ------------------------------------------------------------------ #
#  Model-manipulation functions (passed to vLLM collective_rpc / apply_model)
# ------------------------------------------------------------------ #

def disable_grad_fn(model):
    """Disable gradients for all model parameters (used after weight transfer init)."""
    for p in model.parameters():
        p.requires_grad_(False)


def get_params_info_fn(model):
    """Return list of (name, shape, dtype) for all model parameters."""
    return [(n, tuple(p.shape), p.dtype) for n, p in model.named_parameters()]


def compute_checksum_fn(model):
    """Compute an order-sensitive checksum over all model parameters.

    Uses position-dependent weights (golden ratio multiplier) so that
    swapping two parameters produces a different checksum — unlike a
    plain abs().sum() which is commutative.
    """
    checksum = 0.0
    phi = 1.6180339887  # golden ratio — irrational, avoids periodic collisions
    for i, (_, p) in enumerate(model.named_parameters()):
        weight = phi ** (i % 32)  # cycle to avoid overflow
        checksum += p.float().abs().sum().item() * weight
    return checksum


# ------------------------------------------------------------------ #
#  vLLM stat logger for trace throughput counters
# ------------------------------------------------------------------ #

class TraceStatLogger:
    """Lightweight vLLM stat logger that accumulates throughput samples.

    Registered on the LLM engine when log_stats is enabled. Records
    generation throughput and scheduler state at a fixed interval
    (default 5s). Samples are drained after each generate() call and
    sent back to the coordinator for trace counter logging.

    Implements the vllm.v1.metrics.loggers.StatLoggerBase interface.
    """

    INTERVAL = 5.0  # seconds between samples

    def __init__(self, vllm_config=None, engine_index=0):
        self._gen_tokens = 0
        self._prompt_tokens = 0
        self._last_time = None  # set on first record with data
        self._running_reqs = 0
        self._kv_cache_pct = 0.0
        self._idle_emitted = False
        self._last_activity = None  # time of last real token data
        self.samples = []  # [(mono_time, gen_tps, prompt_tps, running_reqs, kv_pct)]

    def record(self, scheduler_stats=None, iteration_stats=None,
               mm_cache_stats=None, engine_idx=0):
        if iteration_stats is not None:
            self._gen_tokens += iteration_stats.num_generation_tokens
            self._prompt_tokens += iteration_stats.prompt_token_stats.computed
            self._idle_emitted = False  # real activity — allow idle flush again
            self._last_activity = time.monotonic()
        if scheduler_stats is not None:
            self._running_reqs = scheduler_stats.num_running_reqs
            self._kv_cache_pct = scheduler_stats.kv_cache_usage * 100
        now = time.monotonic()
        if self._last_time is None:
            self._last_time = now
            return
        dt = now - self._last_time
        if dt >= self.INTERVAL:
            self.samples.append((
                now,
                self._gen_tokens / dt,
                self._prompt_tokens / dt,
                self._running_reqs,
                self._kv_cache_pct,
            ))
            self._gen_tokens = 0
            self._prompt_tokens = 0
            self._last_time = now

    def log_engine_initialized(self):
        pass

    def log(self):
        pass

    def record_sleep_state(self, is_awake, level):
        pass

    def flush_idle(self):
        """Flush any buffered tokens, then emit one zero sample.

        Call this from the worker's idle loop (no active requests) to ensure
        idle periods show as zero in the trace rather than as gaps.
        Only emits once per idle period — repeated calls are no-ops until
        new real data arrives via record().
        """
        if self._idle_emitted:
            return
        now = time.monotonic()
        if self._last_time is None:
            self._last_time = now
            return
        dt = now - self._last_time
        if dt < self.INTERVAL:
            return
        # Flush any buffered token counts as a real sample using full
        # elapsed dt (same as record()) for consistent throughput numbers.
        if self._gen_tokens > 0 or self._prompt_tokens > 0:
            self.samples.append((
                now, self._gen_tokens / dt, self._prompt_tokens / dt,
                self._running_reqs, self._kv_cache_pct,
            ))
            self._gen_tokens = 0
            self._prompt_tokens = 0
            # Emit zero with small offset so Perfetto shows both points
            self.samples.append((now + 0.001, 0.0, 0.0, 0, 0.0))
            self._last_time = now + 0.001
        else:
            self.samples.append((now, 0.0, 0.0, 0, 0.0))
            self._last_time = now
        self._idle_emitted = True

    def drain(self):
        """Return and clear accumulated samples."""
        samples = self.samples
        self.samples = []
        return samples


def extract_student_logprobs(output, student_logprobs, batch_idx, resp_len):
    """Extract log π_old(y_t) for each generated token from vLLM output.

    Handles three vLLM logprob formats:
      1. Tensor format (vLLM v1): SampleLogprobs with .logprobs attribute
      2. Raw tuple format: (token_id, logprobs_tensor, ...)
      3. Dict format: {token_id: Logprob(logprob=...)}
    """
    sl = output.outputs[0].logprobs
    if sl is None or len(sl) == 0:
        return

    if (
        hasattr(sl, "start_indices")
        and hasattr(sl, "end_indices")
        and hasattr(sl, "logprobs")
    ):
        rows = min(len(sl), resp_len)
        sampled = [
            sl.logprobs[sl.start_indices[t]]
            for t in range(rows)
            if sl.end_indices[t] > sl.start_indices[t]
        ]
        if sampled:
            student_logprobs[batch_idx, :len(sampled)] = torch.tensor(
                sampled, dtype=torch.float32)
        return

    # Tensor format (vLLM v1): index 0 is the sampled token's logprob
    if hasattr(sl[0], "logprobs"):
        lps = torch.stack([s.logprobs for s in sl])[:resp_len, 0]
        student_logprobs[batch_idx, :resp_len] = lps.to(torch.float32)
        return

    # vLLM v1 raw tuple format
    if isinstance(sl[0], tuple) and len(sl[0]) == 3:
        for t in range(min(len(sl), resp_len)):
            _, logprobs_t, _ = sl[t]
            student_logprobs[batch_idx, t] = logprobs_t[0][0].item()
        return

    # Dict format — sampled token is always included when logprobs >= 1
    gen_ids = list(output.outputs[0].token_ids)
    for t in range(min(len(sl), resp_len)):
        pos_dict = sl[t]
        actual_id = gen_ids[t]
        if actual_id in pos_dict:
            student_logprobs[batch_idx, t] = pos_dict[actual_id].logprob
        else:
            print(f"  [WARNING] sampled token {actual_id} not in logprobs dict "
                  f"at position {t}, keys={list(pos_dict.keys())}", flush=True)
            student_logprobs[batch_idx, t] = -1e10


def _empty_topk_result(rows, topk_k, *, include_logprobs):
    """Allocate zero/filled outputs for top-k extraction."""
    indices = torch.zeros(rows, topk_k, dtype=torch.int32)
    if not include_logprobs:
        return indices, None
    logprobs = torch.full((rows, topk_k), -1e10, dtype=torch.float32)
    return indices, logprobs


def _finalize_topk_result(indices, logprobs, *, rows, topk_k, include_logprobs):
    """Pad a possibly-short top-k result up to the requested width."""
    if indices.size(-1) == topk_k:
        return indices, logprobs if include_logprobs else None

    out_idx, out_logprobs = _empty_topk_result(
        rows, topk_k, include_logprobs=include_logprobs)
    keep = indices.size(-1)
    out_idx[:, :keep] = indices
    if include_logprobs and logprobs is not None:
        out_logprobs[:, :keep] = logprobs
    return out_idx, out_logprobs


def _stack_tensor_topk_rows(sl, rows):
    """Stack tensor-format per-token logprob rows into dense tensors."""
    logprobs = torch.stack([sl[t].logprobs for t in range(rows)], dim=0).to(torch.float32)
    token_ids = torch.stack([sl[t].logprob_token_ids for t in range(rows)], dim=0).to(torch.int32)
    return token_ids, logprobs


def _stack_tuple_topk_rows(sl, rows):
    """Stack raw tuple-format per-token logprob rows into dense tensors."""
    logprobs_rows = []
    token_id_rows = []
    for t in range(rows):
        _, logprobs_t, token_ids_t = sl[t]
        if torch.is_tensor(logprobs_t):
            logprobs_rows.append(logprobs_t[0].to(torch.float32))
        else:
            logprobs_rows.append(torch.tensor(logprobs_t[0], dtype=torch.float32))
        if torch.is_tensor(token_ids_t):
            token_id_rows.append(token_ids_t[0].to(torch.int32))
        else:
            token_id_rows.append(torch.tensor(token_ids_t[0], dtype=torch.int32))
    return torch.stack(token_id_rows, dim=0), torch.stack(logprobs_rows, dim=0)


def _is_flat_logprobs(sl):
    """Return True when vLLM flat_logprobs container is present."""
    return (
        hasattr(sl, "start_indices")
        and hasattr(sl, "end_indices")
        and hasattr(sl, "token_ids")
        and hasattr(sl, "logprobs")
    )


def _stack_flat_topk_rows(sl, rows):
    """Stack FlatLogprobs rows without materializing per-position dicts."""
    starts = sl.start_indices[:rows]
    ends = sl.end_indices[:rows]
    if rows == 0:
        return (
            torch.zeros(0, 0, dtype=torch.int32),
            torch.zeros(0, 0, dtype=torch.float32),
        )

    widths = [end - start for start, end in zip(starts, ends)]
    if len(set(widths)) == 1:
        width = widths[0]
        if width == 0:
            return (
                torch.zeros(rows, 0, dtype=torch.int32),
                torch.zeros(rows, 0, dtype=torch.float32),
            )
        limit = ends[-1]
        token_ids = torch.tensor(sl.token_ids[:limit], dtype=torch.int32).view(rows, width)
        logprobs = torch.tensor(sl.logprobs[:limit], dtype=torch.float32).view(rows, width)
        return token_ids, logprobs

    max_width = max(widths)
    token_ids = torch.zeros(rows, max_width, dtype=torch.int32)
    logprobs = torch.full((rows, max_width), -1e10, dtype=torch.float32)
    for t, (start, end) in enumerate(zip(starts, ends)):
        width = end - start
        if width <= 0:
            continue
        token_ids[t, :width] = torch.tensor(sl.token_ids[start:end], dtype=torch.int32)
        logprobs[t, :width] = torch.tensor(sl.logprobs[start:end], dtype=torch.float32)
    return token_ids, logprobs


def _extract_dense_topk(token_ids, logprobs, topk_k, *, include_logprobs, sorted):
    """Select top-k columns from dense [rows, cols] token-id/logprob tensors."""
    rows = token_ids.size(0)
    if rows == 0 or topk_k <= 0:
        return _empty_topk_result(rows, topk_k, include_logprobs=include_logprobs)

    k = min(topk_k, logprobs.size(-1))
    topk_logprobs, order = torch.topk(logprobs, k=k, dim=-1, sorted=sorted)
    topk_indices = torch.gather(token_ids, dim=-1, index=order).to(torch.int32)
    if not include_logprobs:
        return _finalize_topk_result(
            topk_indices, None, rows=rows, topk_k=topk_k, include_logprobs=False)
    return _finalize_topk_result(
        topk_indices,
        topk_logprobs.to(torch.float32),
        rows=rows,
        topk_k=topk_k,
        include_logprobs=True,
    )


def _extract_dense_top1_preferring_sampled(sampled_ids, token_ids, logprobs, *, include_logprobs):
    """Return top-1, preferring the sampled token whenever it is available.

    For K=1, sampled-token alignment is more important than pure top-1 ranking:
    it keeps the degenerate support path consistent with sampled-token PG-KL and
    avoids tiny tie/ordering mismatches between vLLM's generated token and the
    separately extracted support rows.
    """
    rows = token_ids.size(0)
    out_idx, out_logprobs = _empty_topk_result(rows, 1, include_logprobs=include_logprobs)
    if rows == 0:
        return out_idx, out_logprobs

    max_logprobs = logprobs.max(dim=-1).values
    for t in range(rows):
        sampled_id = int(sampled_ids[t])
        row_ids = token_ids[t]
        row_lps = logprobs[t]
        sampled_matches = (row_ids == sampled_id).nonzero(as_tuple=True)[0]
        if sampled_matches.numel() > 0:
            chosen_pos = int(sampled_matches[0].item())
        else:
            chosen_pos = int(torch.argmax(row_lps).item())
        out_idx[t, 0] = row_ids[chosen_pos].to(torch.int32)
        if include_logprobs and out_logprobs is not None:
            out_logprobs[t, 0] = row_lps[chosen_pos].to(torch.float32)
    return out_idx, out_logprobs


def _extract_student_topk(output, resp_len, topk_k, *, include_logprobs, sorted):
    """Shared top-k extraction backend for support/logprob and indices-only paths."""
    sl = output.outputs[0].logprobs
    if sl is None or len(sl) == 0 or topk_k <= 0:
        return _empty_topk_result(0, topk_k, include_logprobs=include_logprobs)

    rows = min(len(sl), resp_len)
    if rows <= 0:
        return _empty_topk_result(0, topk_k, include_logprobs=include_logprobs)

    if _is_flat_logprobs(sl):
        token_ids, logprobs = _stack_flat_topk_rows(sl, rows)
        if topk_k == 1:
            sampled_ids = output.outputs[0].token_ids[:rows]
            return _extract_dense_top1_preferring_sampled(
                sampled_ids, token_ids, logprobs,
                include_logprobs=include_logprobs)
        return _extract_dense_topk(
            token_ids, logprobs, topk_k,
            include_logprobs=include_logprobs, sorted=sorted)

    # Tensor format (vLLM v1): positions include both ids and logprobs.
    if hasattr(sl[0], "logprobs") and hasattr(sl[0], "logprob_token_ids"):
        token_ids, logprobs = _stack_tensor_topk_rows(sl, rows)
        if topk_k == 1:
            sampled_ids = output.outputs[0].token_ids[:rows]
            return _extract_dense_top1_preferring_sampled(
                sampled_ids, token_ids, logprobs,
                include_logprobs=include_logprobs)
        return _extract_dense_topk(
            token_ids, logprobs, topk_k,
            include_logprobs=include_logprobs, sorted=sorted)

    # Raw tuple format: (token_id, logprobs_tensor, token_ids_tensor)
    if isinstance(sl[0], tuple) and len(sl[0]) == 3:
        token_ids, logprobs = _stack_tuple_topk_rows(sl, rows)
        if topk_k == 1:
            sampled_ids = output.outputs[0].token_ids[:rows]
            return _extract_dense_top1_preferring_sampled(
                sampled_ids, token_ids, logprobs,
                include_logprobs=include_logprobs)
        return _extract_dense_topk(
            token_ids, logprobs, topk_k,
            include_logprobs=include_logprobs, sorted=sorted)

    out_idx, out_logprobs = _empty_topk_result(
        rows, topk_k, include_logprobs=include_logprobs)
    # Dict format — keep the current semantics for compatibility.
    for t in range(rows):
        if topk_k == 1:
            pos_dict = sl[t]
            actual_id = int(output.outputs[0].token_ids[t])
            if actual_id in pos_dict:
                items = [(actual_id, pos_dict[actual_id])]
            else:
                items = [max(pos_dict.items(), key=lambda kv: kv[1].logprob)]
        else:
            items = heapq.nlargest(
                topk_k,
                sl[t].items(),
                key=lambda kv: kv[1].logprob,
            )
        for j, (tok_id, logprob_obj) in enumerate(items):
            out_idx[t, j] = int(tok_id)
            if include_logprobs and out_logprobs is not None:
                out_logprobs[t, j] = float(logprob_obj.logprob)
    return out_idx, out_logprobs


def extract_student_topk_indices(output, resp_len, topk_k):
    """Extract rollout-student top-k token IDs only for teacher query support."""
    indices, _ = _extract_student_topk(
        output, resp_len, topk_k,
        include_logprobs=False, sorted=True)
    return indices


def extract_student_topk_support(output, resp_len, topk_k):
    """Extract rollout-student top-k token IDs/logprobs for each generated token.

    Returns:
        (indices, logprobs): both shaped [resp_len, topk_k]
    """
    indices, logprobs = _extract_student_topk(
        output, resp_len, topk_k,
        include_logprobs=True, sorted=True)
    return indices, logprobs


# ------------------------------------------------------------------ #
#  Shared vLLM init helpers
# ------------------------------------------------------------------ #

def setup_vllm_env(gpu_ids, tp_size, use_weight_transfer=False, native_lora=False,
                    vllm_port=None, vllm_master_port=None):
    """Set vLLM-specific environment variables BEFORE any vLLM import.

    Must be called before `from vllm import LLM` or similar.

    Args:
        vllm_port: Pre-allocated port for vLLM (from coordinator). Avoids
            EADDRINUSE when multiple subprocesses call find_free_port()
            independently with separate _allocated_ports sets.
        vllm_master_port: Pre-allocated MASTER_PORT for torch.distributed init.
    """
    if 'vllm' in sys.modules:
        return
    env = {}
    if tp_size <= 1 or native_lora:
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if tp_size > 1:
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if tp_size > 1 or use_weight_transfer:
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    # Use pre-allocated ports from coordinator when available, fall back to
    # find_free_port() for standalone/test usage.
    if vllm_port:
        env["VLLM_PORT"] = str(vllm_port)
    elif "VLLM_PORT" not in os.environ:
        from opd.utils.net import find_free_port
        env["VLLM_PORT"] = str(find_free_port("rollout.vllm.fallback"))
    if vllm_master_port:
        env["MASTER_PORT"] = str(vllm_master_port)
    elif "MASTER_PORT" not in os.environ:
        from opd.utils.net import find_free_port
        env["MASTER_PORT"] = str(find_free_port("rollout.vllm_master.fallback"))
    if "MASTER_ADDR" not in os.environ:
        env["MASTER_ADDR"] = "127.0.0.1"
    for k, v in env.items():
        os.environ[k] = v


def apply_pre_load_patches(quantization, native_lora=False, lora_cfg=None):
    """Apply FP8 and LoRA patches that must run BEFORE vLLM model load.

    Returns (is_fp8, is_blockwise) flags for post-load steps.
    """
    if native_lora and lora_cfg is not None:
        from opd.rollout.vllm.lora import hijack_lora_manager, hijack_update_weights, build_peft_config_dict
        hijack_lora_manager()
        hijack_update_weights(build_peft_config_dict(lora_cfg))

    is_fp8 = quantization in ("fp8", "fp8_blockwise")
    is_blockwise = quantization == "fp8_blockwise"
    if is_fp8:
        from opd.rollout.vllm.fp8 import patch_update_weights_for_fp8, check_fp8_hardware_support
        check_fp8_hardware_support()
        if is_blockwise:
            from opd.rollout.vllm.fp8 import apply_vllm_fp8_patches
            apply_vllm_fp8_patches()
        patch_update_weights_for_fp8()

    return is_fp8, is_blockwise


def build_common_llm_kwargs(worker):
    """Build the shared LLM constructor kwargs dict from a rollout worker instance.

    Returns a dict of kwargs common to both LLM() (batch) and
    AsyncEngineArgs (streaming). Each backend adds its own
    engine-specific keys:
      - batch: passes model_path as positional arg to LLM(model_path, **kwargs)
      - streaming: adds model=model_path, enable_log_requests=False to kwargs,
        then passes to AsyncEngineArgs(**kwargs)

    Does NOT include FP8, LoRA, or weight_transfer keys — those are
    added by each backend after calling this function.
    """
    kwargs = dict(
        tensor_parallel_size=worker.tp_size,
        trust_remote_code=resolve_trust_remote_code(
            getattr(worker, "trust_remote_code", None),
            context="vLLM rollout model loading",
        ),
        gpu_memory_utilization=worker.gpu_memory_utilization,
        max_num_seqs=worker.max_num_seqs,
        enable_chunked_prefill=False,
        enforce_eager=worker.enforce_eager,
        dtype=worker.dtype,
        max_logprobs=worker.max_logprobs,
        disable_log_stats=False,
    )
    if worker.max_model_len is not None:
        kwargs["max_model_len"] = worker.max_model_len
    if getattr(worker, 'block_size', None) is not None:
        kwargs["block_size"] = worker.block_size
    if worker.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = worker.max_num_batched_tokens
    return kwargs

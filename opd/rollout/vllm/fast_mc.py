"""Fast rollout-side MC helpers for prompt-logprob and single-pass sampling."""

from __future__ import annotations

import types

import torch

from opd.rollout.mc_utils import (
    sample_multiset_all_from_probs,
    sample_multiset_from_log_probs,
    sample_multiset_from_logits,
    sample_multiset_from_probs,
)


# ---------------------------------------------------------------------------
# Shared runner state
# ---------------------------------------------------------------------------


def _ensure_runner_fast_mc_state(runner, fallback_topk_k: int) -> None:
    runner._pending_mc_requests_by_id = getattr(
        runner, "_pending_mc_requests_by_id", {})
    runner._fastcaptured_mc = getattr(runner, "_fastcaptured_mc", [])
    runner._fast_mc_fallback_topk_k = fallback_topk_k
    runner._single_pass_mc_request_bindings = getattr(
        runner, "_single_pass_mc_request_bindings", {})
    runner._single_pass_mc_request_configs = getattr(
        runner, "_single_pass_mc_request_configs", {})
    runner._single_pass_mc_captures = getattr(
        runner, "_single_pass_mc_captures", {})


# ---------------------------------------------------------------------------
# Prompt-logprob fast path (existing 2nd-pass / score support)
# ---------------------------------------------------------------------------


def _fast_mc_get_prompt_logprobs_dict(self, hidden_states, num_scheduled_tokens):
    """Patched GPUModelRunner hook that captures prompt-logprob results on GPU."""
    num_prompt_logprobs_dict = self.num_prompt_logprobs
    if not num_prompt_logprobs_dict:
        return {}

    completed_prefill_reqs = []
    pending_queries = getattr(self, "_pending_mc_requests_by_id", {})
    captured = getattr(self, "_fastcaptured_mc", [])
    fallback_topk_k = int(getattr(self, "_fast_mc_fallback_topk_k", 1))

    for req_id, _k in list(num_prompt_logprobs_dict.items()):
        num_tokens = num_scheduled_tokens.get(req_id)
        if num_tokens is None:
            continue

        request = self.requests[req_id]
        if request.prompt_token_ids is None:
            continue

        num_prompt_tokens = len(request.prompt_token_ids)
        start_idx = request.num_computed_tokens
        start_tok = start_idx + 1
        num_remaining_tokens = num_prompt_tokens - start_tok
        if num_tokens <= num_remaining_tokens:
            num_logits = num_tokens
        else:
            num_logits = num_remaining_tokens
            completed_prefill_reqs.append(req_id)

        if num_logits <= 0:
            continue

        req_idx = self.input_batch.req_id_to_index[req_id]
        offset = self.query_start_loc.np[req_idx].item()
        hs = hidden_states[offset:offset + num_logits]
        chunk_size = 256

        request_id_key = getattr(request, "request_id", req_id)
        matched_query = pending_queries.pop(request_id_key, None)
        if matched_query is None and len(pending_queries) == 1:
            _, matched_query = pending_queries.popitem()

        chunk_mc_idx = []
        chunk_mc_logps = []
        chunk_topk_vals = []
        chunk_topk_idx = []
        chunk_actual_lps = []

        if matched_query is not None:
            prompt_len = int(matched_query["prompt_length"])
            n_total_samples = int(matched_query["n_total_samples"])
            row_start = max(prompt_len - 1, 0)
            row_end = max(num_prompt_tokens - 1, row_start)
        else:
            row_start = row_end = 0

        for ci in range(0, num_logits, chunk_size):
            ce = min(ci + chunk_size, num_logits)
            logits = self.model.compute_logits(hs[ci:ce])
            if logits is None:
                continue
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            g0 = start_idx + ci
            g1 = start_idx + ce

            if matched_query is not None:
                overlap_start = max(g0, row_start)
                overlap_end = min(g1, row_end)
                if overlap_start < overlap_end:
                    row_slice = log_probs[overlap_start - g0:overlap_end - g0]
                    actual_ids = torch.tensor(
                        request.prompt_token_ids[overlap_start + 1:overlap_end + 1],
                        dtype=torch.long,
                        device=row_slice.device,
                    )
                    sample_idx, sample_logps = sample_multiset_from_log_probs(
                        row_slice, actual_ids, n_total_samples)
                    chunk_mc_idx.append(sample_idx.cpu().to(torch.int32))
                    chunk_mc_logps.append(sample_logps.cpu().to(torch.float32))
            else:
                tv, ti = torch.topk(log_probs, fallback_topk_k, dim=-1)
                actual_ids = torch.tensor(
                    request.prompt_token_ids[start_tok + ci:start_tok + ce],
                    dtype=torch.long, device=log_probs.device,
                )
                actual_lps = log_probs.gather(1, actual_ids.unsqueeze(1)).squeeze(1)
                chunk_topk_vals.append(tv.cpu().to(torch.float32))
                chunk_topk_idx.append(ti.cpu().to(torch.int32))
                chunk_actual_lps.append(actual_lps.cpu().to(torch.float32))

            del logits, log_probs

        if matched_query is not None:
            n_total_samples = int(matched_query["n_total_samples"])
            captured.append({
                "mc_query_indices_response": (
                    torch.cat(chunk_mc_idx, dim=0)
                    if chunk_mc_idx else
                    torch.zeros(0, n_total_samples, dtype=torch.int32)
                ),
                "mc_query_old_logprobs_response": (
                    torch.cat(chunk_mc_logps, dim=0)
                    if chunk_mc_logps else
                    torch.zeros(0, n_total_samples, dtype=torch.float32)
                ),
            })
        else:
            captured.append({
                "topk_logprobs": (
                    torch.cat(chunk_topk_vals, dim=0)
                    if chunk_topk_vals else
                    torch.zeros(0, fallback_topk_k, dtype=torch.float32)
                ),
                "topk_indices": (
                    torch.cat(chunk_topk_idx, dim=0)
                    if chunk_topk_idx else
                    torch.zeros(0, fallback_topk_k, dtype=torch.int32)
                ),
                "token_logprobs": (
                    torch.cat(chunk_actual_lps, dim=0)
                    if chunk_actual_lps else
                    torch.zeros(0, dtype=torch.float32)
                ),
            })

    self._fastcaptured_mc = captured
    in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
    for req_id in completed_prefill_reqs:
        num_prompt_logprobs_dict.pop(req_id, None)
        in_progress_dict.pop(req_id, None)
    return {}


# ---------------------------------------------------------------------------
# Single-pass generation-time MC capture
# ---------------------------------------------------------------------------


def _register_single_pass_request_on_runner(
    runner,
    vllm_request_id: str,
    capture_request_id: str,
    n_total_samples: int,
) -> bool:
    _ensure_runner_fast_mc_state(runner, getattr(runner, "_fast_mc_fallback_topk_k", 1))
    bindings = getattr(runner, "_single_pass_mc_request_bindings", {})
    configs = getattr(runner, "_single_pass_mc_request_configs", {})
    bindings[vllm_request_id] = capture_request_id
    configs[capture_request_id] = {"n_total_samples": int(n_total_samples)}
    runner._single_pass_mc_request_bindings = bindings
    runner._single_pass_mc_request_configs = configs
    return True



def _clear_single_pass_requests_on_runner(runner, capture_request_ids: list[str]) -> bool:
    _ensure_runner_fast_mc_state(runner, getattr(runner, "_fast_mc_fallback_topk_k", 1))
    requested = set(capture_request_ids)
    captures = getattr(runner, "_single_pass_mc_captures", {})
    configs = getattr(runner, "_single_pass_mc_request_configs", {})
    bindings = getattr(runner, "_single_pass_mc_request_bindings", {})
    for request_id in requested:
        captures.pop(request_id, None)
        configs.pop(request_id, None)
    runner._single_pass_mc_request_bindings = {
        k: v for k, v in bindings.items() if v not in requested
    }
    runner._single_pass_mc_request_configs = configs
    runner._single_pass_mc_captures = captures
    return True



def _materialize_single_pass_capture(entry, n_total_samples: int) -> dict[str, torch.Tensor]:
    if not entry:
        return {
            "mc_query_indices_response": torch.zeros(0, n_total_samples, dtype=torch.int32),
            "mc_query_old_logprobs_response": torch.zeros(0, n_total_samples, dtype=torch.float32),
        }
    idx_rows = entry.get("mc_query_indices_response", [])
    old_rows = entry.get("mc_query_old_logprobs_response", [])
    return {
        "mc_query_indices_response": (
            torch.cat(idx_rows, dim=0).to(dtype=torch.int32, device="cpu")
            if idx_rows else torch.zeros(0, n_total_samples, dtype=torch.int32)
        ),
        "mc_query_old_logprobs_response": (
            torch.cat(old_rows, dim=0).to(dtype=torch.float32, device="cpu")
            if old_rows else torch.zeros(0, n_total_samples, dtype=torch.float32)
        ),
    }


def _align_capture_rows_to_response_ids(
    entry,
    expected_token_ids,
):
    capture_ids = entry["mc_query_indices_response"][:, 0].tolist()
    expected_ids = list(expected_token_ids)
    if capture_ids == expected_ids:
        return entry

    selected_rows = []
    cap_idx = 0
    for token_id in expected_ids:
        while cap_idx < len(capture_ids) and capture_ids[cap_idx] != token_id:
            cap_idx += 1
        if cap_idx >= len(capture_ids):
            return None
        selected_rows.append(cap_idx)
        cap_idx += 1

    return {
        "mc_query_indices_response": entry["mc_query_indices_response"][selected_rows],
        "mc_query_old_logprobs_response": entry["mc_query_old_logprobs_response"][selected_rows],
    }


def align_single_pass_capture(entry, expected_rows: int, finish_reason, expected_token_ids=None):
    """Align capture rows to the rollout-visible response length.

    vLLM can emit a final stop token internally while omitting it from the
    returned `CompletionOutput.token_ids` when `finish_reason == "stop"`.
    In that case, single-pass capture can be exactly one row longer than the
    visible response. Trim that one trailing row; otherwise fail loudly.
    """
    actual_rows = int(entry["mc_query_indices_response"].shape[0])
    if expected_token_ids is not None:
        expected_token_ids = list(expected_token_ids)[:expected_rows]
        aligned = _align_capture_rows_to_response_ids(entry, expected_token_ids)
        if aligned is not None and int(aligned["mc_query_indices_response"].shape[0]) == expected_rows:
            return aligned
    if actual_rows == expected_rows:
        return entry
    if finish_reason == "stop" and actual_rows == expected_rows + 1:
        return {
            "mc_query_indices_response": entry["mc_query_indices_response"][:expected_rows],
            "mc_query_old_logprobs_response": entry["mc_query_old_logprobs_response"][:expected_rows],
        }
    raise RuntimeError(
        "single-pass MC capture row mismatch: "
        f"expected_rows={expected_rows}, actual_rows={actual_rows}, "
        f"finish_reason={finish_reason!r}"
    )



def _pop_single_pass_captures_on_runner(runner, capture_request_ids: list[str], n_total_samples: int):
    _ensure_runner_fast_mc_state(runner, getattr(runner, "_fast_mc_fallback_topk_k", 1))
    captures = getattr(runner, "_single_pass_mc_captures", {})
    results = [
        _materialize_single_pass_capture(captures.pop(request_id, None), n_total_samples)
        for request_id in capture_request_ids
    ]
    _clear_single_pass_requests_on_runner(runner, capture_request_ids)
    return results



def _fast_mc_sample(self, logits, spec_decode_metadata):
    """Patched GPUModelRunner._sample capturing MC samples during generation."""
    req_ids = list(getattr(self.input_batch, "req_ids", []) or [])
    bindings = getattr(self, "_single_pass_mc_request_bindings", {})
    configs = getattr(self, "_single_pass_mc_request_configs", {})
    active_rows = []
    sampling_metadata = getattr(self.input_batch, "sampling_metadata", None)
    if (
        logits is not None
        and req_ids
        and (bindings or configs)
        and sampling_metadata is not None
        and getattr(sampling_metadata, "all_random", False)
    ):
        row_count = min(len(req_ids), int(logits.size(0)))
        for row_idx, raw_request_id in enumerate(req_ids[:row_count]):
            capture_request_id = bindings.get(raw_request_id)
            if capture_request_id is None:
                if raw_request_id in configs:
                    capture_request_id = raw_request_id
                else:
                    raise RuntimeError(
                        "single-pass MC capture mapping missing for vLLM request id: "
                        f"raw_request_id={raw_request_id!r}, "
                        f"bindings={sorted(bindings.keys())}, configs={sorted(configs.keys())}"
                    )
            cfg = configs[capture_request_id]
            active_rows.append((row_idx, capture_request_id, int(cfg["n_total_samples"])))
    self.sampler.topk_topp_sampler._fast_mc_active_rows = active_rows
    self.sampler.topk_topp_sampler._fast_mc_last_capture = None
    sampler_output = self._fast_mc_original_sample(logits, spec_decode_metadata)
    if logits is None:
        return sampler_output

    sampled_token_ids = getattr(sampler_output, "sampled_token_ids", None)
    if sampled_token_ids is None or sampled_token_ids.numel() == 0:
        return sampler_output

    if not req_ids:
        return sampler_output

    if sampled_token_ids.dim() == 1:
        actual_token_ids = sampled_token_ids
    else:
        actual_token_ids = sampled_token_ids[:, 0]

    row_count = min(len(req_ids), int(actual_token_ids.size(0)), int(logits.size(0)))
    if row_count <= 0:
        return sampler_output

    bindings = getattr(self, "_single_pass_mc_request_bindings", {})
    configs = getattr(self, "_single_pass_mc_request_configs", {})
    if not bindings and not configs:
        return sampler_output

    captures = getattr(self, "_single_pass_mc_captures", {})
    native_capture = getattr(self.sampler.topk_topp_sampler, "_fast_mc_last_capture", None)
    if native_capture:
        for capture_request_id, (sample_ids, sample_logps) in native_capture.items():
            bucket = captures.setdefault(capture_request_id, {
                "mc_query_indices_response": [],
                "mc_query_old_logprobs_response": [],
            })
            bucket["mc_query_indices_response"].append(sample_ids)
            bucket["mc_query_old_logprobs_response"].append(sample_logps)
        self._single_pass_mc_captures = captures
        self.sampler.topk_topp_sampler._fast_mc_last_capture = None
        self.sampler.topk_topp_sampler._fast_mc_active_rows = []
        return sampler_output

    logits = logits[:row_count].float()

    grouped_rows_by_n = {}
    for row_idx, raw_request_id in enumerate(req_ids[:row_count]):
        capture_request_id = bindings.get(raw_request_id)
        if capture_request_id is None:
            if raw_request_id in configs:
                capture_request_id = raw_request_id
            else:
                raise RuntimeError(
                    "single-pass MC capture mapping missing for vLLM request id: "
                    f"raw_request_id={raw_request_id!r}, "
                    f"bindings={sorted(bindings.keys())}, configs={sorted(configs.keys())}"
                )
        cfg = configs[capture_request_id]
        actual_token_id = int(actual_token_ids[row_idx].item())
        if actual_token_id < 0:
            continue
        n_total_samples = int(cfg["n_total_samples"])
        grouped_rows_by_n.setdefault(n_total_samples, []).append(
            (row_idx, capture_request_id, actual_token_id)
        )

    for n_total_samples, grouped_rows in grouped_rows_by_n.items():
        row_indices = [row_idx for row_idx, _, _ in grouped_rows]
        batched_actual_ids = torch.tensor(
            [actual_token_id for _, _, actual_token_id in grouped_rows],
            dtype=torch.long,
            device=logits.device,
        )
        sample_ids, sample_logps = sample_multiset_from_logits(
            logits[row_indices],
            batched_actual_ids,
            n_total_samples,
        )
        for batch_idx, (_, capture_request_id, _) in enumerate(grouped_rows):
            bucket = captures.setdefault(capture_request_id, {
                "mc_query_indices_response": [],
                "mc_query_old_logprobs_response": [],
            })
            bucket["mc_query_indices_response"].append(sample_ids[batch_idx:batch_idx + 1])
            bucket["mc_query_old_logprobs_response"].append(sample_logps[batch_idx:batch_idx + 1])

    self._single_pass_mc_captures = captures
    self.sampler.topk_topp_sampler._fast_mc_last_capture = None
    self.sampler.topk_topp_sampler._fast_mc_active_rows = []
    return sampler_output




def _resolve_capture_request_id(raw_request_id, request_id_key, bindings, configs):
    for candidate in (
        bindings.get(raw_request_id),
        bindings.get(request_id_key),
        request_id_key,
        raw_request_id,
    ):
        if candidate in configs:
            return candidate
    raise RuntimeError(
        "single-pass MC capture mapping missing for vLLM request ids: "
        f"raw_request_id={raw_request_id!r}, request_id_key={request_id_key!r}, "
        f"bindings={sorted(bindings.keys())}, configs={sorted(configs.keys())}"
    )


def _install_fast_mc_native_sampler_patch(topk_topp_sampler):
    if getattr(topk_topp_sampler, "_fast_mc_native_patch_installed", False):
        return

    original_forward_native = topk_topp_sampler.forward_native

    def _patched_forward_native(self_sampler, logits, generators, k, p):
        logits = self_sampler.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self_sampler.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self_sampler.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        random_sample = original_forward_native.__globals__["random_sample"]
        sampled = torch.full(
            (probs.shape[0],), -1, dtype=torch.long, device=probs.device
        )

        active_rows = getattr(self_sampler, "_fast_mc_active_rows", None) or []
        if active_rows:
            captures = {}
            grouped_rows_by_n = {}
            for row_idx, capture_request_id, n_total_samples in active_rows:
                grouped_rows_by_n.setdefault(n_total_samples, []).append((row_idx, capture_request_id))
            active_row_set = {row_idx for row_idx, _, _ in active_rows}
            inactive_rows = [idx for idx in range(probs.shape[0]) if idx not in active_row_set]
            if inactive_rows:
                inactive_generators = {
                    new_idx: generators[old_idx]
                    for new_idx, old_idx in enumerate(inactive_rows)
                    if old_idx in generators
                }
                sampled[inactive_rows] = random_sample(
                    probs[inactive_rows].clone(),
                    inactive_generators,
                )
            for n_total_samples, grouped_rows in grouped_rows_by_n.items():
                row_indices = [row_idx for row_idx, _ in grouped_rows]
                group_probs = probs[row_indices]
                sample_ids, sample_logps = sample_multiset_all_from_probs(
                    group_probs,
                    n_total_samples,
                )
                sampled[row_indices] = sample_ids[:, 0].long()
                for batch_idx, (_, capture_request_id) in enumerate(grouped_rows):
                    captures[capture_request_id] = (
                        sample_ids[batch_idx:batch_idx + 1],
                        sample_logps[batch_idx:batch_idx + 1],
                    )
            self_sampler._fast_mc_last_capture = captures
        else:
            sampled = random_sample(probs.clone(), generators)
            self_sampler._fast_mc_last_capture = None

        return sampled, logits_to_return

    topk_topp_sampler.forward_native = types.MethodType(
        _patched_forward_native, topk_topp_sampler
    )
    topk_topp_sampler.forward = topk_topp_sampler.forward_native
    topk_topp_sampler._fast_mc_active_rows = []
    topk_topp_sampler._fast_mc_last_capture = None
    topk_topp_sampler._fast_mc_native_patch_installed = True


# ---------------------------------------------------------------------------
# Installation helpers
# ---------------------------------------------------------------------------


def _install_fast_mc_patch_on_runner(
    runner,
    fallback_topk_k: int,
    *,
    enable_native_sampler_capture: bool,
):
    _ensure_runner_fast_mc_state(runner, fallback_topk_k)
    runner._get_prompt_logprobs_dict = types.MethodType(
        _fast_mc_get_prompt_logprobs_dict, runner)
    if enable_native_sampler_capture:
        _install_fast_mc_native_sampler_patch(runner.sampler.topk_topp_sampler)
    if not hasattr(runner, "_fast_mc_original_sample"):
        runner._fast_mc_original_sample = runner._sample
        runner._sample = types.MethodType(_fast_mc_sample, runner)
    return True



def _install_fast_mc_add_request_patch(llm, register_request_fn):
    if getattr(llm, "_fast_mc_add_request_patch_installed", False):
        return True

    original_add_request = llm._add_request

    def _patched_add_request(self_llm, prompt, params, *args, **kwargs):
        mc_request_id = None
        mc_n_total_samples = None
        if isinstance(prompt, dict) and (
            "mc_request_id" in prompt or "mc_n_total_samples" in prompt
        ):
            prompt = dict(prompt)
            mc_request_id = prompt.pop("mc_request_id", None)
            mc_n_total_samples = prompt.pop("mc_n_total_samples", None)
        result = original_add_request(prompt, params, *args, **kwargs)
        if mc_n_total_samples is not None and int(mc_n_total_samples) > 0:
            register_request_fn(
                result,
                mc_request_id or result,
                int(mc_n_total_samples),
            )
        return result

    llm._add_request = types.MethodType(_patched_add_request, llm)
    llm._fast_mc_add_request_patch_installed = True
    return True



def install_fast_mc_patch(llm, fallback_topk_k: int = 1) -> bool:
    """Install prompt-logprob + single-pass MC patches on sync LLM."""
    try:
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
    except AttributeError:
        return False

    def _register_request(vllm_request_id, capture_request_id, n_total_samples):
        return _register_single_pass_request_on_runner(
            runner, vllm_request_id, capture_request_id, n_total_samples)

    installed = _install_fast_mc_patch_on_runner(
        runner,
        fallback_topk_k,
        enable_native_sampler_capture=True,
    )
    _install_fast_mc_add_request_patch(llm, _register_request)
    return installed



def install_fast_mc_patch_rpc(
    engine_or_core,
    fallback_topk_k: int = 1,
    *,
    enable_native_sampler_capture: bool = True,
):
    """Install prompt-logprob + single-pass MC patches in worker subprocesses."""

    def _do_patch(worker, _fallback_topk_k, _enable_native_sampler_capture):
        return _install_fast_mc_patch_on_runner(
            worker.model_runner,
            _fallback_topk_k,
            enable_native_sampler_capture=_enable_native_sampler_capture,
        )

    return engine_or_core.collective_rpc(
        _do_patch,
        args=(fallback_topk_k, enable_native_sampler_capture),
    )


async def _register_single_pass_request_on_async_engine(
    engine,
    vllm_request_id: str,
    capture_request_id: str,
    n_total_samples: int,
) -> None:
    def _register(worker, _vllm_request_id, _capture_request_id, _n_total_samples):
        return _register_single_pass_request_on_runner(
            worker.model_runner,
            _vllm_request_id,
            _capture_request_id,
            _n_total_samples,
        )

    results = await engine.collective_rpc(
        _register,
        args=(vllm_request_id, capture_request_id, int(n_total_samples)),
    )
    if not results or not all(results):
        raise RuntimeError(
            "Failed to register exact AsyncLLM single-pass MC request binding for "
            f"vllm_request_id={vllm_request_id!r}"
        )


def install_fast_mc_async_add_request_patch(engine) -> bool:
    """Patch AsyncLLM.add_request so worker bindings use exact internal request ids."""
    if getattr(engine, "_fast_mc_async_add_request_patch_installed", False):
        return True

    original_add_request = engine.add_request
    g = original_add_request.__globals__
    AsyncGenerator = g["AsyncGenerator"]
    EngineCoreRequest = g["EngineCoreRequest"]
    EngineDeadError = g["EngineDeadError"]
    InputStreamError = g["InputStreamError"]
    ParentRequest = g["ParentRequest"]
    PoolingParams = g["PoolingParams"]
    RequestOutputCollector = g["RequestOutputCollector"]
    copy = g["copy"]
    extract_prompt_components = g["extract_prompt_components"]
    logger = g["logger"]
    merge_kwargs = g["merge_kwargs"]
    warnings = g["warnings"]

    async def _patched_add_request(
        self,
        request_id,
        prompt,
        params,
        arrival_time=None,
        lora_request=None,
        tokenization_kwargs=None,
        trace_headers=None,
        priority=0,
        data_parallel_rank=None,
        prompt_text=None,
    ):
        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)
        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and not is_pooling
            and params.prompt_logprobs
        ):
            raise ValueError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        if params.truncate_prompt_tokens is not None:
            params_type = type(params).__name__
            warnings.warn(
                f"The `truncate_prompt_tokens` parameter in `{params_type}` "
                "is deprecated and will be removed in v0.16. "
                "Please pass it via `tokenization_kwargs` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            tokenization_kwargs = merge_kwargs(
                tokenization_kwargs,
                dict(truncate_prompt_tokens=params.truncate_prompt_tokens),
            )

        if isinstance(prompt, AsyncGenerator):
            return await self._add_streaming_input_request(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
            )

        if isinstance(prompt, EngineCoreRequest):
            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            if prompt_text is not None:
                raise ValueError(
                    "should only provide prompt_text with EngineCoreRequest"
                )
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                supported_tasks=await self.get_supported_tasks(),
            )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        self.input_processor.assign_request_id(request)

        pending = getattr(self, "_pending_single_pass_mc_request_configs", {})
        staged = pending.pop(request_id, None)
        self._pending_single_pass_mc_request_configs = pending
        if staged is not None:
            await _register_single_pass_request_on_async_engine(
                self,
                request.request_id,
                staged["capture_request_id"],
                staged["n_total_samples"],
            )

        self._run_output_handler()
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        queue = RequestOutputCollector(params.output_kind, request.request_id)
        params = request.params

        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        parent_params = params
        assert parent_params.n >= 1
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            child_request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = child_request_id
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue

    engine.add_request = types.MethodType(_patched_add_request, engine)
    engine._pending_single_pass_mc_request_configs = getattr(
        engine, "_pending_single_pass_mc_request_configs", {}
    )
    engine._fast_mc_async_add_request_patch_installed = True
    return True



def install_fast_mc_add_request_patch_rpc(llm, engine_or_core) -> bool:
    """Patch sync LLM._add_request and register request bindings via RPC."""

    def _register_request(vllm_request_id, capture_request_id, n_total_samples):
        def _do_register(worker, _rid, _capture_id, _n_total_samples):
            return _register_single_pass_request_on_runner(
                worker.model_runner,
                _rid,
                _capture_id,
                _n_total_samples,
            )

        results = engine_or_core.collective_rpc(
            _do_register,
            args=(vllm_request_id, capture_request_id, n_total_samples),
        )
        return bool(results) and all(results)

    return _install_fast_mc_add_request_patch(llm, _register_request)


# ---------------------------------------------------------------------------
# Legacy prompt-logprob query staging (used by score path only)
# ---------------------------------------------------------------------------


def set_pending_mc_query_on_llm(llm, meta: dict) -> None:
    """Stage one prompt-logprob MC query on a synchronous LLM instance."""
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    runner._pending_mc_requests_by_id = {"__only__": meta}
    runner._fastcaptured_mc = []


async def set_pending_mc_query_on_async_engine(engine, request_id: str, meta: dict) -> None:
    """Stage one prompt-logprob MC query on every AsyncLLM worker."""

    def _set_pending(worker, _request_id, _meta):
        worker.model_runner._pending_mc_requests_by_id = {_request_id: _meta}
        worker.model_runner._fastcaptured_mc = []
        return True

    await engine.collective_rpc(_set_pending, args=(request_id, meta))


# ---------------------------------------------------------------------------
# Capture pop helpers
# ---------------------------------------------------------------------------


def pop_captured_mc_from_llm(llm):
    """Pop prompt-logprob capture rows from a sync LLM runner."""
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    captured = getattr(runner, "_fastcaptured_mc", [])
    runner._fastcaptured_mc = []
    return captured


def pop_captured_mc_from_engine_or_core(engine_or_core):
    """Pop prompt-logprob capture rows from sync TP worker subprocesses."""
    def _pop(worker):
        captured = getattr(worker.model_runner, "_fastcaptured_mc", [])
        worker.model_runner._fastcaptured_mc = []
        return captured

    results = engine_or_core.collective_rpc(_pop)
    for result in results:
        if result:
            return result
    return []


async def pop_captured_mc_from_async_engine(engine):
    """Pop prompt-logprob capture rows from AsyncLLM workers."""

    def _get_captured(worker):
        captured = getattr(worker.model_runner, "_fastcaptured_mc", [])
        worker.model_runner._fastcaptured_mc = []
        return captured

    results = await engine.collective_rpc(_get_captured)
    for result in results:
        if result:
            return result
    return []



def pop_single_pass_mc_from_llm(llm, capture_request_ids: list[str], n_total_samples: int):
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    return _pop_single_pass_captures_on_runner(
        runner,
        capture_request_ids,
        n_total_samples,
    )



def pop_single_pass_mc_from_engine_or_core(engine_or_core, capture_request_ids: list[str], n_total_samples: int):
    def _pop(worker, _capture_request_ids, _n_total_samples):
        return _pop_single_pass_captures_on_runner(
            worker.model_runner,
            list(_capture_request_ids),
            _n_total_samples,
        )

    results = engine_or_core.collective_rpc(
        _pop,
        args=(capture_request_ids, n_total_samples),
    )
    if not results:
        return [
            _materialize_single_pass_capture(None, n_total_samples)
            for _ in capture_request_ids
        ]
    for result in results:
        if any(entry["mc_query_indices_response"].numel() > 0 for entry in result):
            return result
    return results[0]


async def set_single_pass_mc_config_on_async_engine(
    engine,
    request_id: str,
    n_total_samples: int,
) -> None:
    pending = getattr(engine, "_pending_single_pass_mc_request_configs", {})
    pending[request_id] = {
        "capture_request_id": request_id,
        "n_total_samples": int(n_total_samples),
    }
    engine._pending_single_pass_mc_request_configs = pending


async def pop_single_pass_mc_from_async_engine(
    engine,
    request_id: str,
    n_total_samples: int,
):
    def _pop(worker, _request_id, _n_total_samples):
        return _pop_single_pass_captures_on_runner(
            worker.model_runner,
            [_request_id],
            _n_total_samples,
        )

    results = await engine.collective_rpc(_pop, args=(request_id, int(n_total_samples)))
    if not results:
        return _materialize_single_pass_capture(None, n_total_samples)
    for result in results:
        if result and result[0]["mc_query_indices_response"].numel() > 0:
            return result[0]
    return results[0][0]


async def clear_single_pass_mc_request_on_async_engine(engine, request_id: str) -> None:
    def _clear(worker, _request_id):
        return _clear_single_pass_requests_on_runner(worker.model_runner, [_request_id])

    await engine.collective_rpc(_clear, args=(request_id,))

"""vLLM-based teacher scoring server with ZMQ transport."""

import heapq
import os
import sys
import time
import types

import torch
import zmq

from opd.launch_specs import TeacherLaunchSpec
from opd.rollout.mc_utils import sample_teacher_multiset_from_logits
from opd.utils.config import resolve_trust_remote_code
from opd.worker.teacher.serialization import serialize, deserialize
from opd.worker.teacher.fast_logprobs import captured_topk
from opd.utils.trace import timer


class VLLMTeacherServer:
    """vLLM-based teacher scoring server with ZMQ transport."""

    def __init__(self, config: dict):
        self.launch_spec = config if isinstance(config, TeacherLaunchSpec) else None
        if isinstance(config, TeacherLaunchSpec):
            config = config.merged_config()

        model_path = config["model_path"]
        n_logprobs = config["n_logprobs"]
        tp_size = config["tp_size"]
        gpu_memory_utilization = config["gpu_memory_utilization"]
        bind_port = config["bind_port"]
        bind_address = config["bind_address"]
        max_model_len = config["max_model_len"]
        max_num_seqs = config["max_num_seqs"]
        enforce_eager = config["enforce_eager"]
        scoring_batch_size = config["scoring_batch_size"]
        dtype = config["dtype"]
        disable_fast_logprobs = config["disable_fast_logprobs"]
        block_size = config["block_size"]

        self.n_logprobs = n_logprobs
        self.tp_size = tp_size
        self.scoring_batch_size = scoring_batch_size
        self.hidden_recompute = bool(config.get("hidden_recompute", False))
        self.teacher_hidden_dtype = config.get("teacher_hidden_dtype", "bfloat16")
        self.teacher_hidden_semantics = config.get("teacher_hidden_semantics", "lm_head_input")

        from vllm import LLM

        kwargs = dict(
            tensor_parallel_size=tp_size,
            trust_remote_code=resolve_trust_remote_code(
                config.get("trust_remote_code"),
                context="vLLM teacher model loading",
            ),
            enable_chunked_prefill=False,
            enforce_eager=enforce_eager,
            max_logprobs=n_logprobs,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            dtype=dtype,
        )
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if block_size is not None:
            kwargs["block_size"] = block_size

        self.llm = LLM(model_path, **kwargs)
        self._pending_query_requests_by_query_id = {}
        self._install_query_request_id_patch()

        # Install fast prompt logprobs monkey-patch.
        # TP=1: instance-level patch (direct model_runner access in same process)
        # TP>1: RPC patch (cloudpickled callable sent to worker subprocesses)
        if disable_fast_logprobs:
            self.use_fast_logprobs = False
        elif (n_logprobs >= 1 or self.hidden_recompute) and tp_size <= 1:
            self.use_fast_logprobs = self._install_fast_prompt_logprobs()
        elif n_logprobs >= 1 and tp_size > 1:
            try:
                self.use_fast_logprobs = self._install_fast_prompt_logprobs_rpc()
            except Exception as e:
                import traceback
                print(f"[Teacher-patch] Failed to install TP>1 fast logprobs: {e}\n"
                      f"{traceback.format_exc()}", flush=True)
                self.use_fast_logprobs = False
        else:
            self.use_fast_logprobs = False

        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://{bind_address}:{bind_port}")
        print(f"[Teacher] vLLM server ready on {bind_address}:{bind_port} "
              f"(fast_logprobs={self.use_fast_logprobs}, tp={tp_size})", flush=True)

    def _set_pending_query_requests(self, query_requests):
        """Stage query metadata before request_id binding and clear old bound entries."""
        self._pending_query_requests_by_query_id = {
            q["query_request_id"]: q for q in (query_requests or [])
        }
        if self.tp_size > 1:
            def _set_query_meta(worker):
                worker.model_runner._pending_query_requests_by_id = {}
                return True
            self.llm.llm_engine.engine_core.collective_rpc(_set_query_meta)
        else:
            runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
            runner._pending_query_requests_by_id = {}

    def _clear_pending_query_requests(self):
        self._pending_query_requests_by_query_id = {}
        if self.tp_size > 1:
            def _clear_query_meta(worker):
                worker.model_runner._pending_query_requests_by_id = {}
                return True
            self.llm.llm_engine.engine_core.collective_rpc(_clear_query_meta)
        else:
            runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
            runner._pending_query_requests_by_id = {}

    def _register_query_request_binding(self, request_id: str, query_request_id: str):
        query_meta = self._pending_query_requests_by_query_id.pop(query_request_id, None)
        if query_meta is None:
            raise RuntimeError(
                f"missing staged query metadata for query_request_id={query_request_id!r}"
            )
        if self.tp_size > 1:
            def _register(worker, rid, meta):
                pending = getattr(worker.model_runner, "_pending_query_requests_by_id", {})
                pending[rid] = meta
                worker.model_runner._pending_query_requests_by_id = pending
                return True
            self.llm.llm_engine.engine_core.collective_rpc(
                _register, args=(request_id, query_meta))
        else:
            runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
            pending = getattr(runner, "_pending_query_requests_by_id", {})
            pending[request_id] = query_meta
            runner._pending_query_requests_by_id = pending

    def _install_query_request_id_patch(self):
        """Patch LLM._add_request so query_request_id binds to vLLM request_id."""
        llm = self.llm
        original_add_request = llm._add_request
        server = self

        def _patched_add_request(self_llm, prompt, params, *args, **kwargs):
            query_request_id = None
            if isinstance(prompt, dict) and "query_request_id" in prompt:
                prompt = dict(prompt)
                query_request_id = prompt.pop("query_request_id")
            result = original_add_request(
                prompt, params, *args, **kwargs)
            if query_request_id is not None:
                server._register_query_request_binding(result, query_request_id)
            return result

        llm._add_request = types.MethodType(_patched_add_request, llm)

    def _install_fast_prompt_logprobs(self):
        """Monkey-patch GPUModelRunner._get_prompt_logprobs_dict on the live LLM
        instance. The patch computes log_softmax + topk immediately inside the
        patched method, storing only the small topk results (~5MB per request)
        instead of full logits (~2.9GB per request at 10k context).

        Must be called AFTER LLM() init and with VLLM_ENABLE_V1_MULTIPROCESSING=0.
        Returns True if patch was applied. For TP=1 only (instance-level patch).
        """
        try:
            runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
        except AttributeError:
            print("[Teacher-patch] Could not find model_runner on LLM instance", flush=True)
            return False

        from vllm.v1.outputs import LogprobsTensors

        n_logprobs = self.n_logprobs
        hidden_dtype_name = getattr(self, "teacher_hidden_dtype", "bfloat16")
        hidden_semantics = getattr(self, "teacher_hidden_semantics", "lm_head_input")
        hidden_dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        hidden_dtype = hidden_dtype_map.get(hidden_dtype_name, torch.bfloat16)
        runner._opd_capture_hidden_states = bool(getattr(self, "hidden_recompute", False))

        def _fast_get_prompt_logprobs_dict(self, hidden_states, num_scheduled_tokens):
            """Replacement: compute logits -> topk immediately, skip vLLM output."""
            num_prompt_logprobs_dict = self.num_prompt_logprobs
            if not num_prompt_logprobs_dict:
                return {}

            completed_prefill_reqs = []
            for req_id, _k in num_prompt_logprobs_dict.items():
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

                # Slice this request's hidden states and compute lm_head in chunks
                # to avoid OOM from large [seq, vocab] temporary tensors.
                req_idx = self.input_batch.req_id_to_index[req_id]
                offset = self.query_start_loc.np[req_idx].item()
                hs = hidden_states[offset:offset + num_logits]
                if getattr(self, "_opd_capture_hidden_states", False):
                    token_ids = torch.tensor(
                        request.prompt_token_ids[start_tok:start_tok + num_logits],
                        dtype=torch.int32,
                    )
                    captured_topk.append({
                        "hidden_states": hs.detach().contiguous().cpu().to(hidden_dtype),
                        "teacher_token_ids": token_ids,
                        "teacher_hidden_dtype": hidden_dtype_name,
                        "teacher_hidden_semantics": hidden_semantics,
                        "hidden_size": int(hs.size(-1)),
                    })
                    continue
                _CHUNK = 256  # ~150MB per chunk for vocab=152064
                pending_queries = getattr(self, "_pending_query_requests_by_id", {})
                matched_query = None
                teacher_mc_mode = False
                request_id_key = getattr(request, "request_id", req_id)
                if pending_queries:
                    matched_query = pending_queries.pop(request_id_key, None)
                    if matched_query is None:
                        raise RuntimeError(
                            f"missing pending query metadata for request_id="
                            f"{request_id_key!r}"
                        )
                    teacher_mc_mode = "teacher_mc_n_total_samples" in matched_query

                chunk_topk_vals, chunk_topk_idx, chunk_actual_lps = [], [], []
                chunk_query_logps = []
                chunk_query_idx = []
                query_prompt_len = int(matched_query["prompt_length"]) if matched_query else 0
                query_indices = matched_query.get("query_indices_response") if matched_query else None
                query_row_start = query_prompt_len - 1 if matched_query else 0
                query_row_end = (
                    max(num_prompt_tokens - 1, 0)
                    if teacher_mc_mode else
                    (query_row_start + len(query_indices) if matched_query else 0)
                )
                for ci in range(0, num_logits, _CHUNK):
                    ce = min(ci + _CHUNK, num_logits)
                    logits = self.model.compute_logits(hs[ci:ce])
                    if matched_query:
                        g0 = start_idx + ci
                        g1 = start_idx + ce
                        overlap_start = max(g0, query_row_start)
                        overlap_end = min(g1, query_row_end)
                        if overlap_start < overlap_end:
                            if teacher_mc_mode:
                                sample_idx, sample_logps = sample_teacher_multiset_from_logits(
                                    logits[overlap_start - g0:overlap_end - g0].float(),
                                    int(matched_query["teacher_mc_n_total_samples"]),
                                )
                                chunk_query_idx.append(sample_idx.cpu().to(torch.int32))
                                chunk_query_logps.append(sample_logps.cpu().to(torch.float32))
                            else:
                                log_probs = torch.log_softmax(logits.float(), dim=-1)
                                q = torch.tensor(
                                    query_indices[overlap_start - query_row_start:overlap_end - query_row_start],
                                    dtype=torch.long, device=logits.device,
                                )
                                gathered = log_probs[overlap_start - g0:overlap_end - g0].gather(1, q)
                                chunk_query_logps.append(gathered.cpu().to(torch.float32))
                        else:
                            pass
                    else:
                        log_probs = torch.log_softmax(logits.float(), dim=-1)
                        tv, ti = torch.topk(log_probs, n_logprobs, dim=-1)
                        at = torch.tensor(
                            request.prompt_token_ids[start_tok + ci:start_tok + ce],
                            dtype=torch.long, device=logits.device,
                        )
                        al = log_probs.gather(1, at.unsqueeze(1)).squeeze(1)
                        chunk_topk_vals.append(tv.cpu().to(torch.float32))
                        chunk_topk_idx.append(ti.cpu().to(torch.int32))
                        chunk_actual_lps.append(al.cpu().to(torch.float32))
                    del logits

                if matched_query:
                    query_k = (
                        int(matched_query["teacher_mc_n_total_samples"])
                        if teacher_mc_mode else
                        matched_query.get("topk_k", len(query_indices[0]) if query_indices else 0)
                    )
                    captured_topk.append({
                        "query_logprobs": (
                            torch.cat(chunk_query_logps, dim=0)
                            if chunk_query_logps
                            else torch.zeros(0, query_k, dtype=torch.float32)
                        ),
                        "query_indices": (
                            torch.cat(chunk_query_idx, dim=0)
                            if chunk_query_idx
                            else torch.zeros(0, query_k, dtype=torch.int32)
                        ),
                    })
                else:
                    captured_topk.append({
                        "topk_logprobs": torch.cat(chunk_topk_vals, dim=0),
                        "topk_indices": torch.cat(chunk_topk_idx, dim=0),
                        "token_logprobs": torch.cat(chunk_actual_lps, dim=0),
                    })

            # Clean up completed requests (same as original)
            in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
            for req_id in completed_prefill_reqs:
                num_prompt_logprobs_dict.pop(req_id, None)
                in_progress_dict.pop(req_id, None)

            return {}

        import types
        runner._get_prompt_logprobs_dict = types.MethodType(
            _fast_get_prompt_logprobs_dict, runner
        )

        print(f"[Teacher-patch] Patched GPUModelRunner._get_prompt_logprobs_dict "
              f"(inline topk, n_logprobs={n_logprobs}, hidden_recompute={getattr(self, 'hidden_recompute', False)})", flush=True)
        return True

    def _install_fast_prompt_logprobs_rpc(self):
        """Install fast logprobs patch in worker subprocesses via collective_rpc.

        For TP>1 where workers are separate processes. Called AFTER LLM() init.
        The patch is sent via cloudpickle through the IPC chain:
        teacher process -> EngineCore subprocess -> worker subprocesses.
        """
        # Capture locally so the closure doesn't reference self (cloudpickle)
        n_logprobs = self.n_logprobs

        def _do_patch(worker, _n_logprobs):
            import types
            runner = worker.model_runner
            runner._fastcaptured_topk = []

            def _fast_get_prompt_logprobs_dict(self, hidden_states, num_scheduled_tokens):
                """Replacement: compute logits -> topk immediately, skip vLLM output.
                Handles TP>1 where compute_logits() returns None on non-rank-0."""
                num_prompt_logprobs_dict = self.num_prompt_logprobs
                if not num_prompt_logprobs_dict:
                    return {}

                completed_prefill_reqs = []
                for req_id, _k in num_prompt_logprobs_dict.items():
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

                    # Slice this request's hidden states and compute lm_head in chunks
                    # to avoid OOM from large [seq, vocab] temporary tensors.
                    req_idx = self.input_batch.req_id_to_index[req_id]
                    offset = self.query_start_loc.np[req_idx].item()
                    hs = hidden_states[offset:offset + num_logits]
                    _CHUNK = 256  # ~150MB per chunk for vocab=152064

                    # TP>1: check if compute_logits returns None (non-rank-0)
                    test_logits = self.model.compute_logits(hs[:1])
                    if test_logits is None:
                        continue
                    del test_logits

                    pending_queries = getattr(self, "_pending_query_requests_by_id", {})
                    matched_query = None
                    teacher_mc_mode = False
                    request_id_key = getattr(request, "request_id", req_id)
                    if pending_queries:
                        matched_query = pending_queries.pop(request_id_key, None)
                        if matched_query is None:
                            raise RuntimeError(
                                f"missing pending query metadata for request_id="
                                f"{request_id_key!r}"
                            )
                        teacher_mc_mode = "teacher_mc_n_total_samples" in matched_query

                    chunk_topk_vals, chunk_topk_idx, chunk_actual_lps = [], [], []
                    chunk_query_logps = []
                    chunk_query_idx = []
                    query_prompt_len = int(matched_query["prompt_length"]) if matched_query else 0
                    query_indices = matched_query.get("query_indices_response") if matched_query else None
                    query_row_start = query_prompt_len - 1 if matched_query else 0
                    query_row_end = (
                        max(num_prompt_tokens - 1, 0)
                        if teacher_mc_mode else
                        (query_row_start + len(query_indices) if matched_query else 0)
                    )
                    for ci in range(0, num_logits, _CHUNK):
                        ce = min(ci + _CHUNK, num_logits)
                        logits = self.model.compute_logits(hs[ci:ce])
                        if matched_query:
                            g0 = start_idx + ci
                            g1 = start_idx + ce
                            overlap_start = max(g0, query_row_start)
                            overlap_end = min(g1, query_row_end)
                            if overlap_start < overlap_end:
                                if teacher_mc_mode:
                                    sample_idx, sample_logps = sample_teacher_multiset_from_logits(
                                        logits[overlap_start - g0:overlap_end - g0].float(),
                                        int(matched_query["teacher_mc_n_total_samples"]),
                                    )
                                    chunk_query_idx.append(sample_idx.cpu().to(torch.int32))
                                    chunk_query_logps.append(sample_logps.cpu().to(torch.float32))
                                else:
                                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                                    q = torch.tensor(
                                        query_indices[overlap_start - query_row_start:overlap_end - query_row_start],
                                        dtype=torch.long, device=logits.device,
                                    )
                                    gathered = log_probs[overlap_start - g0:overlap_end - g0].gather(1, q)
                                    chunk_query_logps.append(gathered.cpu().to(torch.float32))
                        else:
                            log_probs = torch.log_softmax(logits.float(), dim=-1)
                            tv, ti = torch.topk(log_probs, _n_logprobs, dim=-1)
                            at = torch.tensor(
                                request.prompt_token_ids[start_tok + ci:start_tok + ce],
                                dtype=torch.long, device=logits.device,
                            )
                            al = log_probs.gather(1, at.unsqueeze(1)).squeeze(1)
                            chunk_topk_vals.append(tv.cpu().to(torch.float32))
                            chunk_topk_idx.append(ti.cpu().to(torch.int32))
                            chunk_actual_lps.append(al.cpu().to(torch.float32))
                        del logits

                    if matched_query:
                        query_k = (
                            int(matched_query["teacher_mc_n_total_samples"])
                            if teacher_mc_mode else
                            matched_query.get("topk_k", len(query_indices[0]) if query_indices else 0)
                        )
                        self._fastcaptured_topk.append({
                            "query_logprobs": (
                                torch.cat(chunk_query_logps, dim=0)
                                if chunk_query_logps
                                else torch.zeros(0, query_k, dtype=torch.float32)
                            ),
                            "query_indices": (
                                torch.cat(chunk_query_idx, dim=0)
                                if chunk_query_idx
                                else torch.zeros(0, query_k, dtype=torch.int32)
                            ),
                        })
                    else:
                        self._fastcaptured_topk.append({
                            "topk_logprobs": torch.cat(chunk_topk_vals, dim=0),
                            "topk_indices": torch.cat(chunk_topk_idx, dim=0),
                            "token_logprobs": torch.cat(chunk_actual_lps, dim=0),
                        })

                in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
                for req_id in completed_prefill_reqs:
                    num_prompt_logprobs_dict.pop(req_id, None)
                    in_progress_dict.pop(req_id, None)

                return {}

            runner._get_prompt_logprobs_dict = types.MethodType(
                _fast_get_prompt_logprobs_dict, runner
            )
            return True

        results = self.llm.llm_engine.engine_core.collective_rpc(_do_patch, args=(n_logprobs,))
        ok = all(results)
        if ok:
            print(f"[Teacher-patch] Patched {len(results)} workers via RPC "
                  f"(TP>1, inline topk, n_logprobs={n_logprobs})", flush=True)
        return ok

    def _extract_topk_via_rpc(self):
        """Retrieve captured topk from rank 0 worker via collective_rpc (TP>1)."""
        def _get_captured(worker):
            captured = getattr(worker.model_runner, '_fastcaptured_topk', [])
            results = list(captured)
            captured.clear()
            return results

        all_results = self.llm.llm_engine.engine_core.collective_rpc(_get_captured)
        return all_results[0]

    def _extract_topk_from_captured(self):
        """Return captured topk results and clear buffer (TP=1)."""
        if not captured_topk:
            return []
        results = list(captured_topk)
        captured_topk.clear()
        return results

    @staticmethod
    def _extract_prompt_logprobs(output, n_logprobs):
        """Extract prompt logprobs: top-k and actual token in a single pass.

        Returns (topk_logps, topk_indices, actual_token_logps):
            topk_logps:          [N-1, n_logprobs] float32
            topk_indices:        [N-1, n_logprobs] int32
            actual_token_logps:  [N-1] float32
        where N is prompt length (position 0 skipped -- no preceding context).
        """
        pl = output.prompt_logprobs
        if pl is None:
            z_topk = torch.zeros(0, n_logprobs)
            return z_topk, z_topk.to(torch.int32), torch.zeros(0, dtype=torch.float32)

        # Skip position 0 (no preceding context).
        positions = pl[1:]

        # Tensor format (vLLM v1) -- single LogprobsTensor
        if hasattr(positions, "logprobs"):
            lps = positions.logprobs
            return (lps[:, 1:].to(torch.float32),
                    positions.logprob_token_ids[:, 1:].to(torch.int32),
                    lps[:, 0].to(torch.float32))

        # List of LogprobsTensors
        positions = [p for p in positions if p is not None]
        if len(positions) > 0 and hasattr(positions[0], "logprobs"):
            lps = torch.cat([p.logprobs for p in positions], dim=0)
            ids = torch.cat([p.logprob_token_ids for p in positions], dim=0)
            return (lps[:, 1:].to(torch.float32),
                    ids[:, 1:].to(torch.int32),
                    lps[:, 0].to(torch.float32))

        # Dict format -- single pass for both topk and actual token
        token_ids = list(output.prompt_token_ids)
        topk_logps, topk_ids, actual_logps = [], [], []
        for t, pos_dict in enumerate(positions):
            if pos_dict is None:
                continue
            # Top-k via heapq (O(V log k) vs O(V log V) for sorted)
            tokens = heapq.nlargest(n_logprobs, pos_dict.keys(),
                                    key=lambda tk: pos_dict[tk].logprob)
            lps = [pos_dict[tk].logprob for tk in tokens]
            # Pad to n_logprobs
            while len(lps) < n_logprobs:
                lps.append(-1e10)
                tokens.append(0)
            topk_logps.append(lps)
            topk_ids.append(tokens)
            # Actual token logprob
            actual_id = token_ids[t + 1]  # positions skips first, so offset by 1
            if actual_id in pos_dict:
                actual_logps.append(pos_dict[actual_id].logprob)
            else:
                actual_logps.append(-1e10)

        if not topk_logps:
            z_topk = torch.zeros(0, n_logprobs)
            return z_topk, z_topk.to(torch.int32), torch.zeros(0, dtype=torch.float32)

        return (torch.tensor(topk_logps, dtype=torch.float32),
                torch.tensor(topk_ids, dtype=torch.int32),
                torch.tensor(actual_logps, dtype=torch.float32))

    def run(self):
        """ZMQ request loop: receive, score, respond."""
        from vllm import SamplingParams

        while True:
            msg = self.sock.recv()
            try:
                req = deserialize(msg)
            except Exception as e:
                self.sock.send(serialize({"status": "error", "reason": str(e)}))
                continue

            if not isinstance(req, dict) or "prompt_token_ids" not in req:
                self.sock.send(serialize({"status": "error", "reason": "bad request"}))
                continue

            prompt_token_ids = req["prompt_token_ids"]
            prompt_lengths = req.get("prompt_lengths")
            query_indices_response = req.get("query_indices_response")
            query_request_ids = req.get("query_request_ids")
            teacher_mc_n_total_samples = req.get("teacher_mc_n_total_samples")
            query_mode = query_indices_response is not None or teacher_mc_n_total_samples is not None
            teacher_batch_size = req.get("batch_size", self.scoring_batch_size)
            return_hidden_states = bool(req.get("return_hidden_states", False))
            if return_hidden_states:
                if query_mode:
                    self.sock.send(serialize({
                        "status": "error",
                        "reason": "hidden_recompute teacher scoring does not support query/MC modes",
                    }))
                    continue
                if not self.use_fast_logprobs or self.tp_size != 1:
                    self.sock.send(serialize({
                        "status": "error",
                        "reason": "hidden_recompute requires fast_logprobs with teacher TP=1",
                    }))
                    continue

            if query_mode and not self.use_fast_logprobs:
                self.sock.send(serialize({
                    "status": "error",
                    "reason": "rollout-support teacher scoring requires fast_logprobs",
                }))
                continue

            pl = 1 if self.use_fast_logprobs else self.n_logprobs
            sp = SamplingParams(
                temperature=1.0, top_p=0.95, detokenize=False,
                logprobs=None, prompt_logprobs=pl, max_tokens=1,
            )

            all_logps, all_indices, all_token_logps, responses = [], [], [], []
            all_query_logps = []
            all_query_indices = []
            all_hidden_states = []
            all_hidden_token_ids = []
            all_hidden_meta = []
            t_generate, t_extract, n_sub = 0.0, 0.0, 0
            captured_topk.clear()
            _use_rpc = self.use_fast_logprobs and self.tp_size > 1

            with timer() as t_score:
                try:
                    for batch_start in range(0, len(prompt_token_ids), teacher_batch_size):
                        batch_ids = prompt_token_ids[batch_start:batch_start + teacher_batch_size]
                        batch_prompt_lengths = (
                            prompt_lengths[batch_start:batch_start + teacher_batch_size]
                            if prompt_lengths is not None else None
                        )
                        batch_query_request_ids = (
                            query_request_ids[batch_start:batch_start + teacher_batch_size]
                            if query_request_ids is not None else None
                        )
                        batch_query_indices = (
                            query_indices_response[batch_start:batch_start + teacher_batch_size]
                            if query_indices_response is not None else None
                        )
                        if query_mode:
                            if batch_query_request_ids is None or any(
                                    qid is None for qid in batch_query_request_ids):
                                raise RuntimeError(
                                    "rollout-support teacher scoring requires non-empty "
                                    "query_request_ids for every request"
                                )
                            query_requests = []
                            if teacher_mc_n_total_samples is not None:
                                for ids, p_len, qid in zip(
                                        batch_ids,
                                        batch_prompt_lengths,
                                        batch_query_request_ids):
                                    query_requests.append({
                                        "query_request_id": qid,
                                        "prompt_token_ids": list(ids),
                                        "prompt_length": int(p_len),
                                        "teacher_mc_n_total_samples": int(teacher_mc_n_total_samples),
                                    })
                            else:
                                for ids, p_len, qid, q_idx in zip(
                                        batch_ids,
                                        batch_prompt_lengths,
                                        batch_query_request_ids,
                                        batch_query_indices):
                                    topk_k = int(q_idx.size(-1)) if isinstance(q_idx, torch.Tensor) and q_idx.dim() == 2 else (
                                        len(q_idx[0]) if q_idx else 0
                                    )
                                    if isinstance(q_idx, torch.Tensor):
                                        q_idx = q_idx.tolist()
                                    query_requests.append({
                                        "query_request_id": qid,
                                        "prompt_token_ids": list(ids),
                                        "prompt_length": int(p_len),
                                        "topk_k": topk_k,
                                        "query_indices_response": q_idx,
                                    })
                            self._set_pending_query_requests(query_requests)
                        if query_mode:
                            prompts = [
                                {"prompt_token_ids": ids, "query_request_id": qid}
                                for ids, qid in zip(batch_ids, batch_query_request_ids)
                            ]
                        else:
                            prompts = [{"prompt_token_ids": ids} for ids in batch_ids]
                        t0 = time.monotonic()
                        outputs = self.llm.generate(
                            prompts=prompts,
                            sampling_params=sp,
                            use_tqdm=False,
                        )
                        t1 = time.monotonic()
                        t_generate += t1 - t0
                        if query_mode:
                            self._clear_pending_query_requests()

                        if self.use_fast_logprobs and _use_rpc:
                            captured = self._extract_topk_via_rpc()
                            for i, out in enumerate(outputs):
                                responses.append(torch.tensor(
                                    out.outputs[0].token_ids, dtype=torch.int32))
                                cap = captured[i]
                                if query_mode:
                                    all_query_logps.append(cap["query_logprobs"])
                                    if teacher_mc_n_total_samples is not None:
                                        all_query_indices.append(cap["query_indices"])
                                else:
                                    all_logps.append(cap["topk_logprobs"])
                                    all_indices.append(cap["topk_indices"])
                                    all_token_logps.append(cap["token_logprobs"])
                        elif self.use_fast_logprobs and captured_topk:
                            captured = self._extract_topk_from_captured()
                            for i, out in enumerate(outputs):
                                responses.append(torch.tensor(
                                    out.outputs[0].token_ids, dtype=torch.int32))
                                cap = captured[i]
                                if return_hidden_states:
                                    all_hidden_states.append(cap["hidden_states"])
                                    all_hidden_token_ids.append(cap["teacher_token_ids"])
                                    all_hidden_meta.append({
                                        "teacher_hidden_dtype": cap.get("teacher_hidden_dtype", self.teacher_hidden_dtype),
                                        "teacher_hidden_semantics": cap.get("teacher_hidden_semantics", self.teacher_hidden_semantics),
                                        "hidden_size": cap.get("hidden_size"),
                                    })
                                elif query_mode:
                                    all_query_logps.append(cap["query_logprobs"])
                                    if teacher_mc_n_total_samples is not None:
                                        all_query_indices.append(cap["query_indices"])
                                else:
                                    all_logps.append(cap["topk_logprobs"])
                                    all_indices.append(cap["topk_indices"])
                                    all_token_logps.append(cap["token_logprobs"])
                        else:
                            for out in outputs:
                                responses.append(torch.tensor(
                                    out.outputs[0].token_ids, dtype=torch.int32))
                                if query_mode:
                                    raise RuntimeError(
                                        "rollout-support teacher scoring requires fast_logprobs"
                                    )
                                if self.n_logprobs > 0:
                                    p_logps, p_idx, p_tlp = self._extract_prompt_logprobs(out, self.n_logprobs)
                                    all_logps.append(p_logps.cpu())
                                    all_indices.append(p_idx.cpu())
                                    all_token_logps.append(p_tlp.cpu())
                        t_extract += time.monotonic() - t1
                        n_sub += 1
                except Exception as e:
                    import traceback
                    err_msg = f"{e}\n{traceback.format_exc()}"
                    print(f"[Teacher] ERROR during scoring: {err_msg}", flush=True)
                    if query_mode:
                        try:
                            self._clear_pending_query_requests()
                        except Exception:
                            pass
                    self.sock.send(serialize({"status": "error", "reason": err_msg}))
                    captured_topk.clear()
                    if _use_rpc:
                        try:
                            self._extract_topk_via_rpc()  # clear worker buffer
                        except Exception:
                            pass
                    continue

                t_score["mono_end"] = time.monotonic()
                t_ser = time.monotonic()
                resp_payload = {
                    "status": "ok",
                    "responses": responses,
                    "timing": t_score,
                }
                if return_hidden_states:
                    resp_payload["payload_kind"] = "hidden_states"
                    resp_payload["teacher_hidden_states"] = all_hidden_states
                    resp_payload["teacher_hidden_token_ids"] = all_hidden_token_ids
                    resp_payload["teacher_hidden_metadata"] = all_hidden_meta
                elif query_mode:
                    resp_payload["teacher_query_logprobs_response"] = all_query_logps
                    resp_payload["teacher_topk_logprobs"] = all_query_logps
                    resp_payload["teacher_topk_indices"] = all_query_indices if teacher_mc_n_total_samples is not None else []
                    resp_payload["teacher_token_logprobs"] = []
                else:
                    resp_payload["teacher_topk_logprobs"] = all_logps
                    resp_payload["teacher_topk_indices"] = all_indices
                    resp_payload["teacher_token_logprobs"] = all_token_logps
                resp_data = serialize(resp_payload)
                t_ser = time.monotonic() - t_ser
                self.sock.send(resp_data)
            print(f"[Teacher] {len(prompt_token_ids)} prompts scored (bs={teacher_batch_size}) | "
                  f"generate={t_generate:.1f}s  extract={t_extract:.1f}s  "
                  f"serialize={t_ser:.1f}s  ({n_sub} sub-batches, {len(resp_data)/1e6:.1f}MB)"
                  f"{'  [FAST]' if self.use_fast_logprobs else ''}",
                  flush=True)



def teacher_server_main(config: dict):
    """Entry point for teacher vLLM server subprocess."""
    if isinstance(config, TeacherLaunchSpec):
        config = config.merged_config()
    gpu_ids = config["gpu_ids"]
    tp_size = config["tp_size"]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Use pre-allocated ports from coordinator to avoid EADDRINUSE collisions
    # between teacher, rollout, and trainer subprocesses.
    if config.get("vllm_port"):
        os.environ["VLLM_PORT"] = str(config["vllm_port"])
    elif "VLLM_PORT" not in os.environ:
        from opd.utils.net import find_free_port
        os.environ["VLLM_PORT"] = str(find_free_port("teacher.vllm.fallback"))
    if config.get("vllm_master_port"):
        os.environ["MASTER_PORT"] = str(config["vllm_master_port"])
    elif "MASTER_PORT" not in os.environ:
        from opd.utils.net import find_free_port
        os.environ["MASTER_PORT"] = str(find_free_port("teacher.vllm_master.fallback"))
    # TP=1: disable EngineCore multiprocessing for direct model_runner access
    if tp_size <= 1:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    else:
        # TP>1: allow cloudpickle serialization for collective_rpc (fast logprobs patch)
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        # TP>1: use spawn for vLLM worker processes
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Ensure conda env bin is on PATH for JIT compilation tools (ninja, etc.)
    bin_dir = os.path.dirname(sys.executable)
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + ":" + os.environ.get("PATH", "")

    server = VLLMTeacherServer(config)
    server.run()

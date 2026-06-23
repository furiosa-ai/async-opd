"""Fast prompt logprobs monkey-patch for vLLM teacher server.

Patches GPUModelRunner._get_prompt_logprobs_dict to do inline GPU top-k
extraction instead of vLLM's default Python dict path.
"""

import torch


# Global buffer: patched _get_prompt_logprobs_dict stores extracted topk here.
# Only holds small tensors ([seq, k] not [seq, vocab]), ~5MB per request.
# Single-threaded (ZMQ REP/REQ), no concurrency issues.
captured_topk = []  # list of dicts with topk_logprobs, topk_indices, token_logprobs


def install_fast_prompt_logprobs(llm, n_logprobs=256):
    """Apply fast prompt logprobs patch to any vLLM LLM instance.

    Monkey-patches GPUModelRunner._get_prompt_logprobs_dict to do inline
    GPU top-k extraction instead of vLLM's default Python dict path.
    Results are stored in captured_topk (module-level buffer).

    Must be called AFTER LLM() init with VLLM_ENABLE_V1_MULTIPROCESSING=0.
    Returns True if patch was applied.
    """
    try:
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
    except AttributeError:
        print("[fast_logprobs] Could not find model_runner on LLM instance", flush=True)
        return False

    import types

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

            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            hs = hidden_states[offset:offset + num_logits]
            _CHUNK = 256
            chunk_topk_vals, chunk_topk_idx, chunk_actual_lps = [], [], []
            for ci in range(0, num_logits, _CHUNK):
                ce = min(ci + _CHUNK, num_logits)
                logits = self.model.compute_logits(hs[ci:ce])
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
                del logits, log_probs

            captured_topk.append({
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
    print(f"[fast_logprobs] Patched GPUModelRunner (inline topk, n_logprobs={n_logprobs})",
          flush=True)
    return True

"""HuggingFace Transformers rollout backend.

Proper backend alternative to vLLM (BatchRolloutWorker). Slower but fully
deterministic when seeded. Used for deterministic integration tests and
as a reference implementation for adding new backends (e.g., sglang).

Follows the same patterns as:
  - opd/worker/teacher/hf.py: HfTeacherServer (HF backend for teacher)
  - opd/rollout/vllm/batch.py: VLLMBatchRolloutWorker (vLLM backend for rollout)
"""

import os
import socket
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from opd.launch_specs import RolloutLaunchSpec
from opd.rollout.base import BaseRolloutWorker
from opd.rollout.mc_utils import sample_multiset_from_log_probs
from opd.utils.config import resolve_trust_remote_code


ALLOWED_COMMANDS = {
    "generate", "sync_weights", "init_weight_transfer",
    "compute_weight_checksum", "get_vllm_params_info", "shutdown",
}


class HFRolloutWorker(BaseRolloutWorker):
    """HuggingFace Transformers rollout backend."""

    def __init__(self, config: dict):
        super().__init__(config)
        if isinstance(config, RolloutLaunchSpec):
            config = config.merged_config()
        self.seed = config.get("seed")
        self._gen_counter = 0
        self.device = torch.device("cuda:0")

        # Deterministic mode
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        self._setup_env(self.gpu_ids)

        # Map dtype string
        torch_dtype = {
            "auto": "auto", "float16": torch.float16,
            "bfloat16": torch.bfloat16, "float32": torch.float32,
        }.get(self.dtype, "auto")

        trust_remote_code = resolve_trust_remote_code(
            config.get("trust_remote_code"),
            context="HF rollout model loading",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code, attn_implementation="eager",
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[Rollout-HF-{self.worker_id}] ready on GPU {self.gpu_ids} "
              f"(dtype={self.model.dtype}, seed={self.seed})", flush=True)

    def run(self, cmd_queue, result_queue, prompt_queue=None):
        """Command loop — blocks until shutdown."""
        try:
            while True:
                cmd = cmd_queue.get()
                if cmd[0] == "shutdown":
                    print(f"[Rollout-HF-{self.worker_id}] shutting down",
                          flush=True)
                    break
                if cmd[0] not in ALLOWED_COMMANDS:
                    print(f"[Rollout-HF-{self.worker_id}] rejected unknown "
                          f"command: {cmd[0]}", flush=True)
                    continue
                handler = getattr(self, f"handle_{cmd[0]}", None)
                if handler:
                    handler(cmd, result_queue)
                else:
                    print(f"[Rollout-HF-{self.worker_id}] unknown command: "
                          f"{cmd[0]}", flush=True)
        except Exception as e:
            import traceback
            print(f"\n{'='*60}", flush=True)
            print(f"[Rollout-HF-{self.worker_id}] FATAL ERROR: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            print(f"{'='*60}\n", flush=True)

    # ---- Per-command handlers ----

    def handle_generate(self, cmd, result_queue):
        """Handle 'generate' command — same dispatch as BatchRolloutWorker."""
        batch = cmd[1]
        is_eval = batch.pop("eval", False)
        eval_temp = batch.pop("eval_temperature", None)
        eval_n = batch.pop("eval_n_samples", None)
        return_logprobs = batch.pop("return_logprobs", False)
        response_topk_k = batch.pop("response_topk_k", 0)
        mc_n_total_samples = batch.pop("mc_n_total_samples", 0)
        batch_max_resp = (batch.pop("max_response_length", None)
                          or self.max_response_length)
        grpo_n = batch.pop("grpo_n_samples", None)
        batch.pop("use_lora", None)  # HF backend has no LoRA support
        batch.pop("ignore_eos", None)  # not supported in HF generate

        t0 = time.time()
        if grpo_n and grpo_n > 1:
            result = self._do_generate_grpo(
                batch, batch_max_resp, temperature=self.temperature,
                n=grpo_n, top_p=self.top_p)
        elif eval_n and eval_n > 1:
            result = self._do_generate_multi(
                batch, batch_max_resp,
                temperature=eval_temp or 0.6, n=eval_n)
        elif is_eval and eval_temp is None:
            result = self._do_generate(
                batch, batch_max_resp, temperature=0, top_p=1.0, top_k=-1)
        elif eval_temp is not None:
            result = self._do_generate(
                batch, batch_max_resp,
                temperature=eval_temp, top_p=0.95, top_k=-1)
        else:
            result = self._do_generate(
                batch, batch_max_resp, self.temperature, self.top_p,
                self.top_k, return_logprobs=return_logprobs,
                response_topk_k=response_topk_k,
                mc_n_total_samples=mc_n_total_samples)
        elapsed = time.time() - t0

        result["timing"] = {
            "generate_seconds": elapsed,
            "elapsed": elapsed,
            "worker_id": self.worker_id,
            "host": socket.gethostname(),
            "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
        }
        result["_vllm_stats"] = []
        result_queue.put(result)

    def handle_init_weight_transfer(self, cmd, result_queue):
        """No-op — HF backend uses CPU queue, not NCCL transfer engine."""
        result_queue.put({"status": "ok"})

    def handle_sync_weights(self, cmd, result_queue):
        """Receive state_dict via CPU queue, load into model."""
        state_dict = cmd[1]
        t0 = time.time()
        # Handle _orig_mod. prefix from torch.compile
        cleaned = {}
        for k, v in state_dict.items():
            clean_k = k.replace("_orig_mod.", "")
            cleaned[clean_k] = v
        self.model.load_state_dict(cleaned, strict=False)
        dt = time.time() - t0
        print(f"[Rollout-HF-{self.worker_id}] CPU weight sync in {dt:.2f}s",
              flush=True)
        result_queue.put({"status": "synced_cpu", "sync_seconds": dt})

    def handle_compute_weight_checksum(self, cmd, result_queue):
        """Compute checksum on CPU for bitwise consistency with trainer sync payload."""
        phi = 1.6180339887
        checksum = 0.0
        sd = self.model.state_dict()
        for i, (_, param) in enumerate(sorted(sd.items())):
            checksum += param.detach().cpu().float().abs().sum().item() * (phi ** (i % 32))
        result_queue.put({"checksum": checksum})

    def handle_get_vllm_params_info(self, cmd, result_queue):
        """Return param info — satisfies readiness ping from _build_proxies."""
        info = [(n, tuple(p.shape), p.dtype)
                for n, p in self.model.named_parameters()]
        result_queue.put({"params_info": info})

    # ---- Generation methods ----

    def _reseed(self):
        """Re-seed for deterministic generation. Uses base seed + counter."""
        if self.seed is not None:
            s = self.seed + self._gen_counter
            self._gen_counter += 1
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)

    def _do_generate(self, batch, max_response_length, temperature, top_p,
                     top_k, return_logprobs=False, response_topk_k=0,
                     mc_n_total_samples=0):
        """Run HF generation on a batch of prompts."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Extract per-prompt token IDs (unpad)
        prompts_ids = []
        prompt_lengths = []
        for i in range(input_ids.size(0)):
            mask = attention_mask[i].bool()
            ids = input_ids[i][mask]
            prompts_ids.append(ids)
            prompt_lengths.append(len(ids))

        # Pad for batched generation (left-pad)
        max_prompt_len = input_ids.size(1)
        gen_input = input_ids.to(self.device)
        gen_mask = attention_mask.to(self.device)

        # Generate
        self._reseed()
        do_sample = temperature > 0
        gen_kwargs = dict(
            input_ids=gen_input,
            attention_mask=gen_mask,
            max_new_tokens=max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            if top_p is not None and top_p < 1.0:
                gen_kwargs["top_p"] = top_p
            if top_k is not None and top_k > 0:
                gen_kwargs["top_k"] = top_k

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        # Parse outputs — output_ids includes prompt + response
        batch_size = input_ids.size(0)
        total_len = max_prompt_len + max_response_length

        full_ids = torch.zeros(batch_size, total_len, dtype=torch.long)
        full_mask = torch.zeros(batch_size, total_len, dtype=torch.bool)
        responses = torch.zeros(batch_size, max_response_length,
                                dtype=torch.long)
        response_lengths = []
        student_logprobs = (torch.zeros(batch_size, max_response_length,
                                        dtype=torch.float32)
                            if return_logprobs else None)
        query_indices_response = [] if response_topk_k > 0 else None
        query_logprobs_response = [] if response_topk_k > 0 else None
        mc_query_indices_response = [] if mc_n_total_samples > 0 else None
        mc_query_old_logprobs_response = [] if mc_n_total_samples > 0 else None

        for i in range(batch_size):
            p_len = prompt_lengths[i]
            pad_len = max_prompt_len - p_len
            resp_ids = output_ids[i, max_prompt_len:].cpu()
            # Trim trailing pad tokens
            resp_ids_list = resp_ids.tolist()
            resp_len = len(resp_ids_list)
            # Find actual response length (trim padding)
            for j in range(len(resp_ids_list) - 1, -1, -1):
                if resp_ids_list[j] != self.tokenizer.pad_token_id:
                    resp_len = j + 1
                    break
            else:
                resp_len = 0
            resp_len = min(resp_len, max_response_length)
            response_lengths.append(resp_len)

            # Fill full_ids and full_mask
            full_ids[i, :max_prompt_len] = input_ids[i]
            full_mask[i, pad_len:max_prompt_len] = True
            for j in range(resp_len):
                full_ids[i, max_prompt_len + j] = resp_ids_list[j]
                full_mask[i, max_prompt_len + j] = True
                responses[i, j] = resp_ids_list[j]

        # Build full_token_lists
        full_token_lists = []
        for i in range(batch_size):
            p_len = prompt_lengths[i]
            pad_len = max_prompt_len - p_len
            prompt = input_ids[i][pad_len:].tolist()
            resp = responses[i, :response_lengths[i]].tolist()
            full_token_lists.append(prompt + resp)

        # Extract student logprobs via forward pass if needed
        if return_logprobs or response_topk_k > 0:
            topk_result = self._extract_logprobs(
                full_ids, full_mask, prompt_lengths,
                response_lengths, max_prompt_len,
                student_logprobs=student_logprobs,
                response_topk_k=response_topk_k,
            )
            if topk_result is not None:
                query_indices_response, query_logprobs_response = topk_result
        if mc_n_total_samples > 0:
            mc_result = self._extract_mc_samples(
                full_ids, full_mask, prompt_lengths, response_lengths,
                max_prompt_len, n_total_samples=mc_n_total_samples,
            )
            mc_query_indices_response, mc_query_old_logprobs_response = mc_result

        result = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "responses": responses,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "response_lengths": torch.tensor(response_lengths,
                                             dtype=torch.long),
            "full_token_lists": full_token_lists,
        }
        if student_logprobs is not None:
            result["student_logprobs"] = student_logprobs
        if query_indices_response is not None:
            result["query_indices_response"] = query_indices_response
            result["query_logprobs_response"] = query_logprobs_response
        if mc_query_indices_response is not None:
            result["mc_query_indices_response"] = mc_query_indices_response
            result["mc_query_old_logprobs_response"] = mc_query_old_logprobs_response
        return result

    def _do_generate_multi(self, batch, max_response_length, temperature, n):
        """Generate N samples per prompt for Avg@N eval."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.size(0)
        max_prompt_len = input_ids.size(1)

        gen_input = input_ids.to(self.device)
        gen_mask = attention_mask.to(self.device)

        responses_multi = []
        for prompt_idx in range(batch_size):
            samples = []
            p_ids = gen_input[prompt_idx:prompt_idx + 1]
            p_mask = gen_mask[prompt_idx:prompt_idx + 1]
            for sample_idx in range(n):
                # Deterministic per-sample seed
                if self.seed is not None:
                    s = self.seed + self._gen_counter
                    self._gen_counter += 1
                    torch.manual_seed(s)
                    torch.cuda.manual_seed_all(s)

                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=p_ids, attention_mask=p_mask,
                        max_new_tokens=max_response_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True, temperature=temperature, top_p=0.95,
                    )
                resp = out[0, max_prompt_len:].cpu().tolist()
                # Trim trailing pad tokens
                while resp and resp[-1] == self.tokenizer.pad_token_id:
                    resp.pop()
                resp = resp[:max_response_length]
                samples.append(resp)
            responses_multi.append(samples)

        return {"responses_multi": responses_multi}

    def _do_generate_grpo(self, batch, max_response_length, temperature, n,
                          top_p=0.95):
        """Generate N samples per prompt for GRPO training with logprobs."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        num_prompts = input_ids.size(0)
        max_prompt_len = input_ids.size(1)

        B = num_prompts * n
        total_len = max_prompt_len + max_response_length

        full_ids = torch.zeros(B, total_len, dtype=torch.long)
        full_mask = torch.zeros(B, total_len, dtype=torch.bool)
        responses = torch.zeros(B, max_response_length, dtype=torch.long)
        student_logprobs = torch.zeros(B, max_response_length,
                                       dtype=torch.float32)
        response_lengths = []
        full_token_lists = []
        prompt_lengths = []

        flat_idx = 0
        for prompt_idx in range(num_prompts):
            p_mask = attention_mask[prompt_idx].bool()
            p_len = p_mask.sum().item()
            pad_len = max_prompt_len - p_len
            p_ids = input_ids[prompt_idx:prompt_idx + 1].to(self.device)
            p_attn = attention_mask[prompt_idx:prompt_idx + 1].to(self.device)

            for sample_idx in range(n):
                # Deterministic per-sample seed
                if self.seed is not None:
                    s = self.seed + self._gen_counter
                    self._gen_counter += 1
                    torch.manual_seed(s)
                    torch.cuda.manual_seed_all(s)

                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=p_ids, attention_mask=p_attn,
                        max_new_tokens=max_response_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True, temperature=temperature, top_p=top_p,
                    )

                resp_ids = out[0, max_prompt_len:].cpu().tolist()
                # Trim trailing pad tokens
                while resp_ids and resp_ids[-1] == self.tokenizer.pad_token_id:
                    resp_ids.pop()
                resp_ids = resp_ids[:max_response_length]
                resp_len = len(resp_ids)
                response_lengths.append(resp_len)
                prompt_lengths.append(p_len)

                # Fill tensors
                full_ids[flat_idx, :max_prompt_len] = input_ids[prompt_idx]
                full_mask[flat_idx, pad_len:max_prompt_len] = True
                for j in range(resp_len):
                    full_ids[flat_idx, max_prompt_len + j] = resp_ids[j]
                    full_mask[flat_idx, max_prompt_len + j] = True
                    responses[flat_idx, j] = resp_ids[j]

                # full_token_lists
                prompt_tokens = input_ids[prompt_idx][pad_len:].tolist()
                full_token_lists.append(prompt_tokens + resp_ids)

                flat_idx += 1

        # Extract student logprobs via forward pass
        self._extract_logprobs(full_ids, full_mask, prompt_lengths,
                               response_lengths, max_prompt_len,
                               student_logprobs)

        return {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "responses": responses,
            "student_logprobs": student_logprobs,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "response_lengths": torch.tensor(response_lengths,
                                             dtype=torch.long),
            "full_token_lists": full_token_lists,
        }

    def _extract_logprobs(self, full_ids, full_mask, prompt_lengths,
                          response_lengths, max_prompt_len, student_logprobs=None,
                          response_topk_k=0):
        """Extract sampled-token logprobs and optional response-local top-k support."""
        # Forward pass on full sequences (prompt + response)
        ids = full_ids.to(self.device)
        mask = full_mask.to(self.device).long()
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        query_indices_response = [] if response_topk_k > 0 else None
        query_logprobs_response = [] if response_topk_k > 0 else None

        for i in range(full_ids.size(0)):
            p_len = (prompt_lengths[i] if isinstance(prompt_lengths, list)
                     else prompt_lengths[i].item())
            r_len = (response_lengths[i] if isinstance(response_lengths, list)
                     else response_lengths[i].item())
            # Response tokens start at position max_prompt_len
            # logits at position t predict token at t+1
            # So logprob for response token j is at logits position
            # (max_prompt_len + j - 1)
            sample_topk_idx = (
                torch.zeros(r_len, response_topk_k, dtype=torch.int32)
                if response_topk_k > 0 else None
            )
            sample_topk_logps = (
                torch.zeros(r_len, response_topk_k, dtype=torch.float32)
                if response_topk_k > 0 else None
            )
            for j in range(r_len):
                logit_pos = max_prompt_len + j - 1
                if logit_pos < 0:
                    continue
                token_id = full_ids[i, max_prompt_len + j].item()
                if student_logprobs is not None:
                    student_logprobs[i, j] = log_probs[i, logit_pos,
                                                       token_id].cpu().item()
                if response_topk_k > 0:
                    topk_logps, topk_idx = torch.topk(
                        log_probs[i, logit_pos], response_topk_k, dim=-1)
                    sample_topk_idx[j] = topk_idx.to(torch.int32).cpu()
                    sample_topk_logps[j] = topk_logps.to(torch.float32).cpu()
            if response_topk_k > 0:
                query_indices_response.append(sample_topk_idx)
                query_logprobs_response.append(sample_topk_logps)

        if response_topk_k > 0:
            return query_indices_response, query_logprobs_response
        return None

    def _extract_mc_samples(
        self,
        full_ids,
        full_mask,
        prompt_lengths,
        response_lengths,
        max_prompt_len,
        n_total_samples,
    ):
        """Extract response-local multi-sample old-policy tensors via HF forward."""
        ids = full_ids.to(self.device)
        mask = full_mask.to(self.device).long()
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        mc_query_indices_response = []
        mc_query_old_logprobs_response = []
        for i in range(full_ids.size(0)):
            r_len = (response_lengths[i] if isinstance(response_lengths, list)
                     else response_lengths[i].item())
            if r_len <= 0:
                mc_query_indices_response.append(
                    torch.zeros(0, n_total_samples, dtype=torch.int32))
                mc_query_old_logprobs_response.append(
                    torch.zeros(0, n_total_samples, dtype=torch.float32))
                continue
            row_log_probs = []
            actual_token_ids = []
            for j in range(r_len):
                logit_pos = max_prompt_len + j - 1
                if logit_pos < 0:
                    continue
                row_log_probs.append(log_probs[i, logit_pos])
                actual_token_ids.append(full_ids[i, max_prompt_len + j])
            if row_log_probs:
                row_log_probs = torch.stack(row_log_probs, dim=0)
                actual_token_ids = torch.stack(actual_token_ids, dim=0).to(row_log_probs.device)
                sample_idx, sample_logps = sample_multiset_from_log_probs(
                    row_log_probs, actual_token_ids, n_total_samples)
                mc_query_indices_response.append(sample_idx.to(torch.int32).cpu())
                mc_query_old_logprobs_response.append(sample_logps.to(torch.float32).cpu())
            else:
                mc_query_indices_response.append(
                    torch.zeros(0, n_total_samples, dtype=torch.int32))
                mc_query_old_logprobs_response.append(
                    torch.zeros(0, n_total_samples, dtype=torch.float32))

        return mc_query_indices_response, mc_query_old_logprobs_response


def hf_rollout_worker_main(config, cmd_queue, result_queue):
    """Entry point for HF rollout worker subprocess."""
    worker = HFRolloutWorker(config)
    worker.run(cmd_queue, result_queue)

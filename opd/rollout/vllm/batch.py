"""Student vLLM rollout worker — runs as a subprocess on dedicated GPU(s).

Commands received via mp.Queue:
  ("generate", batch_dict)    -> generates sequences, returns result dict
  ("init_weight_transfer", init_info)  -> init vLLM native NCCL weight transfer
  ("sync_weights", update_info)        -> receive weights via vLLM NCCL engine
  ("shutdown",)               -> exits
"""

import os
import socket
import time

import torch

from opd.launch_specs import RolloutLaunchSpec
from opd.rollout.base import BaseRolloutWorker
from opd.rollout.vllm.fast_mc import (
    align_single_pass_capture,
    install_fast_mc_add_request_patch_rpc,
    install_fast_mc_patch,
    install_fast_mc_patch_rpc,
    pop_captured_mc_from_engine_or_core,
    pop_captured_mc_from_llm,
    pop_single_pass_mc_from_engine_or_core,
    pop_single_pass_mc_from_llm,
)
from opd.rollout.vllm.utils import (
    extract_student_logprobs as _extract_student_logprobs,
    extract_student_topk_support as _extract_student_topk_support,
    disable_grad_fn, get_params_info_fn, compute_checksum_fn, TraceStatLogger,
    setup_vllm_env, apply_pre_load_patches, build_common_llm_kwargs,
)
from opd.utils.trace import timer


ALLOWED_COMMANDS = {"generate", "score", "init_weight_transfer", "sync_weights", "sync_weights_collective", "get_vllm_params_info", "compute_weight_checksum", "compute_lora_checksum"}




class VLLMBatchRolloutWorker(BaseRolloutWorker):
    """Batch vLLM rollout worker.

    Loads model in __init__, processes commands in run().
    Can be subclassed or wrapped as a @ray.remote actor.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        if isinstance(config, RolloutLaunchSpec):
            config = config.merged_config()
        setup_vllm_env(self.gpu_ids, self.tp_size,
                       use_weight_transfer=self.use_weight_transfer,
                       native_lora=self.native_lora,
                       vllm_port=config.get("vllm_port"),
                       vllm_master_port=config.get("vllm_master_port"))
        self._setup_env(self.gpu_ids)
        _is_fp8, _is_blockwise = apply_pre_load_patches(
            self.quantization, self.native_lora, self.lora_cfg)
        from vllm import LLM
        llm_kwargs = build_common_llm_kwargs(self)
        if self.use_weight_transfer:
            llm_kwargs["weight_transfer_config"] = {"backend": "nccl"}
        if _is_fp8:
            from opd.rollout.vllm.fp8 import FP8_BLOCK_QUANT_KWARGS
            llm_kwargs["quantization"] = "fp8"
            if _is_blockwise:
                llm_kwargs["hf_overrides"] = {"quantization_config": FP8_BLOCK_QUANT_KWARGS}
        if self.native_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = 1
            llm_kwargs["max_lora_rank"] = self.lora_rank
        self.llm = LLM(self.model_path, **llm_kwargs)
        self._fast_mc_installed = False
        # FP8 alignment check after model load (blockwise only — per-tensor has no alignment req)
        if _is_blockwise:
            from opd.rollout.vllm.fp8 import check_fp8_alignment
            self.llm.apply_model(check_fp8_alignment)
        # Register trace stat logger for throughput monitoring
        self._trace_logger = TraceStatLogger()
        engine = self.llm.llm_engine
        if engine.logger_manager is not None:
            engine.logger_manager.stat_loggers.append(self._trace_logger)
        print(f"[Rollout-{self.worker_id}] vLLM ready on GPUs {self.gpu_ids}", flush=True)

    def run(self, cmd_queue, result_queue, prompt_queue=None):
        """Command loop — blocks until shutdown."""
        try:
            while True:
                cmd = cmd_queue.get()
                if cmd[0] == "shutdown":
                    print(f"[Rollout-{self.worker_id}] shutting down", flush=True)
                    break
                if cmd[0] not in ALLOWED_COMMANDS:
                    print(f"[Rollout-{self.worker_id}] rejected unknown command: {cmd[0]}", flush=True)
                    continue
                handler = getattr(self, f"handle_{cmd[0]}", None)
                if handler:
                    handler(cmd, result_queue)
                else:
                    print(f"[Rollout-{self.worker_id}] unknown command: {cmd[0]}", flush=True)
        except Exception as e:
            import traceback
            print(f"\n{'='*60}", flush=True)
            print(f"[Rollout-{self.worker_id}] FATAL ERROR: {type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()
            print(f"{'='*60}\n", flush=True)

    # ---- Per-command handlers ----

    def handle_generate(self, cmd, result_queue):
        """Handle 'generate' command."""
        batch = cmd[1]
        is_eval = batch.pop("eval", False)
        eval_temp = batch.pop("eval_temperature", None)
        eval_n = batch.pop("eval_n_samples", None)
        return_logprobs = batch.pop("return_logprobs", False)
        response_topk_k = batch.pop("response_topk_k", 0)
        mc_n_total_samples = batch.pop("mc_n_total_samples", 0)
        batch_max_resp = batch.pop("max_response_length", None) or self.max_response_length
        grpo_n = batch.pop("grpo_n_samples", None)
        # Use LoRA adapter for generation if loaded (not for eval or score)
        use_lora = batch.pop("use_lora", True)
        lora_req = self._lora_request if (self._lora_request and use_lora) else None
        with timer() as t:
            if grpo_n and grpo_n > 1:
                result = self._do_generate_grpo(batch, batch_max_resp,
                                                 temperature=self.temperature,
                                                 n=grpo_n, top_p=self.top_p,
                                                 lora_request=lora_req)
            elif eval_n and eval_n > 1:
                result = self._do_generate_multi(batch, batch_max_resp,
                                                 temperature=eval_temp or 0.6, n=eval_n,
                                                 lora_request=lora_req)
            elif is_eval and eval_temp is None:
                result = self._do_generate(batch, batch_max_resp,
                                           temperature=0, top_p=1.0, top_k=-1,
                                           lora_request=lora_req)
            elif eval_temp is not None:
                result = self._do_generate(batch, batch_max_resp,
                                           temperature=eval_temp, top_p=0.95, top_k=-1,
                                           lora_request=lora_req)
            else:
                result = self._do_generate(batch, batch_max_resp,
                                           self.temperature, self.top_p, self.top_k,
                                           return_logprobs=return_logprobs,
                                           response_topk_k=response_topk_k,
                                           mc_n_total_samples=mc_n_total_samples,
                                           lora_request=lora_req)
        t["generate_seconds"] = t["elapsed"]
        t["worker_id"] = self.worker_id
        t["host"] = socket.gethostname()
        t["gpu_ids"] = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        result["timing"] = t
        result["_vllm_stats"] = self._trace_logger.drain()
        result_queue.put(result)

    def handle_init_weight_transfer(self, cmd, result_queue):
        """Handle 'init_weight_transfer' command."""
        init_info = cmd[1]
        t0 = time.time()
        self.llm.init_weight_transfer_engine({"init_info": init_info})
        self.llm.apply_model(disable_grad_fn)
        dt = time.time() - t0
        print(f"[Rollout-{self.worker_id}] Weight transfer engine init in {dt:.2f}s",
              flush=True)
        result_queue.put({"status": "ok"})

    def handle_get_vllm_params_info(self, cmd, result_queue):
        """Handle 'get_vllm_params_info' command."""
        info = self.llm.apply_model(get_params_info_fn)[0]
        result_queue.put({"params_info": info})

    def handle_sync_weights(self, cmd, result_queue):
        """Handle 'sync_weights' command."""
        update_info = cmd[1]
        is_lora = update_info.get("_lora_update", False)
        t0 = time.time()
        self.llm.update_weights({"update_info": update_info})
        dt = time.time() - t0
        # After first LoRA sync, create the LoRARequest for generate() calls
        if is_lora and self._lora_request is None:
            from vllm.lora.request import LoRARequest as VLLMLoRARequest
            from opd.rollout.vllm.lora import LORA_NAME, LORA_INT_ID, LORA_PATH
            self._lora_request = VLLMLoRARequest(
                lora_name=LORA_NAME, lora_int_id=LORA_INT_ID,
                lora_path=LORA_PATH)
        kind = "LoRA" if is_lora else "NCCL"
        print(f"[Rollout-{self.worker_id}] {kind} weight sync in {dt:.2f}s",
              flush=True)
        result_queue.put({"status": "synced_nccl", "sync_seconds": dt})

    def handle_sync_weights_collective(self, cmd, result_queue):
        """Handle 'sync_weights_collective' — receive via Ray collective broadcast.

        Receives tensors with HF checkpoint names (q_proj, k_proj, v_proj, etc.)
        and copies them directly into vLLM model params, bypassing load_weights
        which has complex re-fusion logic that doesn't work for repeated updates.
        """
        info = cmd[1]
        group_name = info["group_name"]
        weights_info = info["weights_info"]
        t0 = time.time()
        from opd.worker.ray_weight_sync import _rollout_receive_weights
        tensors = _rollout_receive_weights(weights_info, group_name=group_name)
        # Direct copy into vLLM model params (bypass load_weights)
        ec = self.llm.llm_engine.engine_core
        if hasattr(ec, 'engine_core'):
            model = ec.engine_core.model_executor.driver_worker.model_runner.model
        else:
            model = ec.model_executor.driver_worker.model_runner.model
        param_dict = dict(model.named_parameters())
        loaded = 0
        for name, tensor in tensors:
            if name in param_dict:
                param_dict[name].data.copy_(tensor)
                loaded += 1
        dt = time.time() - t0
        print(f"[Rollout-{self.worker_id}] collective weight sync in {dt:.2f}s "
              f"({loaded}/{len(tensors)} params)", flush=True)
        result_queue.put({"status": "synced_collective", "sync_seconds": dt})

    def handle_compute_weight_checksum(self, cmd, result_queue):
        """Handle 'compute_weight_checksum' command."""
        checksum = self.llm.apply_model(compute_checksum_fn)[0]
        result_queue.put({"checksum": checksum})

    def handle_compute_lora_checksum(self, cmd, result_queue):
        """Handle 'compute_lora_checksum' — checksum over LoRA adapter tensors.

        Uses _last_lora_tensors stored by the hijacked update_weights.
        Order-sensitive: uses the same golden ratio weighting as compute_checksum_fn.
        """
        # Access the Worker's stored LoRA tensors (set in hijack_update_weights)
        ec = self.llm.llm_engine.engine_core
        if hasattr(ec, 'engine_core'):
            worker = ec.engine_core.model_executor.driver_worker
        else:
            worker = ec.model_executor.driver_worker
        lora_dict = getattr(worker, '_last_lora_tensors', None)
        if lora_dict is None:
            result_queue.put({"checksum": 0.0})
            return
        sorted_items = sorted(lora_dict.items())
        checksum = 0.0
        phi = 1.6180339887
        for i, (_, tensor) in enumerate(sorted_items):
            weight = phi ** (i % 32)
            checksum += tensor.float().abs().sum().item() * weight
        result_queue.put({"checksum": checksum})

    def handle_score(self, cmd, result_queue):
        """Handle 'score' command — extract prompt logprobs via vLLM prefill.

        Used by OPSD (On-Policy Self-Distillation) to score student-generated
        tokens with a privileged teacher prompt on the same vLLM instance.
        Note: score does NOT use lora_request — base model only = fixed teacher.
        """
        from vllm import SamplingParams
        from opd.worker.teacher import VLLMTeacherServer

        request = cmd[1]
        prompt_token_ids = request["prompt_token_ids"]
        n_logprobs = request.get("n_logprobs", 256)
        scoring_batch_size = request.get("scoring_batch_size", 32)

        sp = SamplingParams(
            temperature=1.0, top_p=0.95, detokenize=False,
            logprobs=None, prompt_logprobs=n_logprobs, max_tokens=1,
        )

        all_logps, all_indices, all_token_logps = [], [], []
        for batch_start in range(0, len(prompt_token_ids), scoring_batch_size):
            batch_ids = prompt_token_ids[batch_start:batch_start + scoring_batch_size]
            prompts = [{"prompt_token_ids": ids} for ids in batch_ids]
            outputs = self.llm.generate(prompts, sp, use_tqdm=False)
            if self._fast_mc_installed:
                captured = (
                    pop_captured_mc_from_engine_or_core(self.llm.llm_engine.engine_core)
                    if self.tp_size > 1 else pop_captured_mc_from_llm(self.llm)
                )
                if captured:
                    for entry in captured:
                        all_logps.append(entry["topk_logprobs"])
                        all_indices.append(entry["topk_indices"])
                        all_token_logps.append(entry["token_logprobs"])
                    continue
            for out in outputs:
                topk_lp, topk_idx, tok_lp = VLLMTeacherServer._extract_prompt_logprobs(out, n_logprobs)
                all_logps.append(topk_lp)
                all_indices.append(topk_idx)
                all_token_logps.append(tok_lp)

        result_queue.put({
            "_cmd": "score",
            "teacher_topk_logprobs": all_logps,
            "teacher_topk_indices": all_indices,
            "teacher_token_logps": all_token_logps,
        })

    # ---- Generation methods ----

    def _do_generate(self, batch, max_response_length, temperature, top_p, top_k,
                     return_logprobs=False, response_topk_k=0,
                     mc_n_total_samples=0, lora_request=None):
        """Run vLLM generation on a batch of prompts."""
        from vllm import SamplingParams

        llm = self.llm
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        prompt_lengths = []
        prompt_token_lists = []
        for i in range(input_ids.size(0)):
            mask = attention_mask[i].bool()
            ids = input_ids[i][mask].tolist()
            prompt_token_lists.append(ids)
            prompt_lengths.append(len(ids))

        mc_request_ids = None
        if mc_n_total_samples > 0:
            self._ensure_fast_mc_patch()
            import uuid
            mc_request_ids = [f"mc-{uuid.uuid4().hex}" for _ in prompt_token_lists]
            prompts = [
                {
                    "prompt_token_ids": ids,
                    "mc_request_id": request_id,
                    "mc_n_total_samples": int(mc_n_total_samples),
                }
                for ids, request_id in zip(prompt_token_lists, mc_request_ids, strict=False)
            ]
        else:
            prompts = [{"prompt_token_ids": ids} for ids in prompt_token_lists]

        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=max_response_length,
            detokenize=False,
            logprobs=max(1 if return_logprobs else 0, response_topk_k) or None,
            ignore_eos=batch.get("ignore_eos", False),
        )

        outputs = llm.generate(prompts=prompts, sampling_params=sp,
                               lora_request=lora_request)

        batch_size = len(outputs)
        max_prompt_len = input_ids.size(1)
        total_len = max_prompt_len + max_response_length

        full_ids = torch.zeros(batch_size, total_len, dtype=torch.long)
        full_mask = torch.zeros(batch_size, total_len, dtype=torch.bool)
        responses = torch.zeros(batch_size, max_response_length, dtype=torch.long)
        response_lengths = []
        student_logprobs = torch.zeros(batch_size, max_response_length,
                                       dtype=torch.float32) if return_logprobs else None
        query_indices_response = [] if response_topk_k > 0 else None
        query_logprobs_response = [] if response_topk_k > 0 else None
        mc_query_indices_response = [] if mc_n_total_samples > 0 else None
        mc_query_old_logprobs_response = [] if mc_n_total_samples > 0 else None

        for i, out in enumerate(outputs):
            resp_ids = list(out.outputs[0].token_ids)
            resp_len = min(len(resp_ids), max_response_length)
            response_lengths.append(resp_len)

            p_len = prompt_lengths[i]
            pad_len = max_prompt_len - p_len
            full_ids[i, :max_prompt_len] = input_ids[i]
            full_mask[i, pad_len:max_prompt_len] = True

            for j in range(resp_len):
                full_ids[i, max_prompt_len + j] = resp_ids[j]
                full_mask[i, max_prompt_len + j] = True
                responses[i, j] = resp_ids[j]

            if return_logprobs:
                _extract_student_logprobs(out, student_logprobs, i, resp_len)
            if response_topk_k > 0:
                topk_idx, topk_logps = _extract_student_topk_support(
                    out, resp_len=resp_len, topk_k=response_topk_k)
                query_indices_response.append(topk_idx)
                query_logprobs_response.append(topk_logps)

        full_token_lists = []
        for i, out in enumerate(outputs):
            p_len = prompt_lengths[i]
            pad_len = max_prompt_len - p_len
            prompt = input_ids[i][pad_len:].tolist()
            resp = list(out.outputs[0].token_ids)[:max_response_length]
            full_token_lists.append(prompt + resp)

        if mc_n_total_samples > 0:
            captured = self._pop_single_pass_mc_sequences(mc_request_ids, mc_n_total_samples)
            aligned = []
            for entry, out in zip(captured, outputs, strict=False):
                resp_len = min(len(out.outputs[0].token_ids), max_response_length)
                finish_reason = getattr(out.outputs[0], "finish_reason", None)
                aligned.append(
                    align_single_pass_capture(
                        entry,
                        resp_len,
                        finish_reason,
                        expected_token_ids=list(out.outputs[0].token_ids)[:resp_len],
                    )
                )
            mc_query_indices_response = [entry["mc_query_indices_response"] for entry in aligned]
            mc_query_old_logprobs_response = [entry["mc_query_old_logprobs_response"] for entry in aligned]

        result = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "responses": responses,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "response_lengths": torch.tensor(response_lengths, dtype=torch.long),
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

    def _ensure_fast_mc_patch(self):
        if self._fast_mc_installed:
            return
        if self.tp_size > 1:
            engine_core = self.llm.llm_engine.engine_core
            results = install_fast_mc_patch_rpc(
                engine_core,
                fallback_topk_k=max(1, self.max_logprobs),
            )
            self._fast_mc_installed = bool(results) and all(results)
            if self._fast_mc_installed:
                install_fast_mc_add_request_patch_rpc(self.llm, engine_core)
        else:
            self._fast_mc_installed = install_fast_mc_patch(
                self.llm, fallback_topk_k=max(1, self.max_logprobs))
        if not self._fast_mc_installed:
            raise RuntimeError("Failed to install rollout fast MC patch")

    def _pop_single_pass_mc_sequences(self, capture_request_ids, n_total_samples):
        if self.tp_size > 1:
            captured = pop_single_pass_mc_from_engine_or_core(
                self.llm.llm_engine.engine_core,
                capture_request_ids,
                n_total_samples,
            )
        else:
            captured = pop_single_pass_mc_from_llm(
                self.llm,
                capture_request_ids,
                n_total_samples,
            )
        return captured

    def _do_generate_multi(self, batch, max_response_length, temperature, n,
                           lora_request=None):
        """Generate N samples per prompt using vLLM's n parameter (for Avg@N eval).

        Returns responses as a list of lists of token ID lists:
          responses_multi[prompt_idx][sample_idx] = list of token IDs
        """
        from vllm import SamplingParams

        llm = self.llm
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        prompts = []
        for i in range(input_ids.size(0)):
            mask = attention_mask[i].bool()
            ids = input_ids[i][mask].tolist()
            prompts.append({"prompt_token_ids": ids})

        sp = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_response_length,
            n=n,
            detokenize=False,
        )

        outputs = llm.generate(prompts=prompts, sampling_params=sp,
                               lora_request=lora_request)

        responses_multi = []
        for out in outputs:
            samples = []
            for completion in out.outputs:
                resp_ids = list(completion.token_ids)[:max_response_length]
                samples.append(resp_ids)
            responses_multi.append(samples)

        return {"responses_multi": responses_multi}

    def _do_generate_grpo(self, batch, max_response_length, temperature, n,
                           top_p=0.95, lora_request=None):
        """Generate N samples per prompt for GRPO training.

        Uses vLLM's native `n` parameter for multi-sample generation.
        Returns the standard tensor schema (same as _do_generate) but with
        B = num_prompts * n rows, including student_old_logprobs.

        Rows are contiguous by prompt: [prompt0_resp0, ..., prompt0_respN-1,
        prompt1_resp0, ...].
        """
        from vllm import SamplingParams

        llm = self.llm
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        num_prompts = input_ids.size(0)
        max_prompt_len = input_ids.size(1)

        prompts = []
        prompt_lengths = []
        for i in range(num_prompts):
            m = attention_mask[i].bool()
            ids = input_ids[i][m].tolist()
            prompts.append({"prompt_token_ids": ids})
            prompt_lengths.append(len(ids))

        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_response_length,
            n=n,
            detokenize=False,
            logprobs=1,  # always need logprobs for GRPO (pi_old)
        )

        outputs = llm.generate(prompts=prompts, sampling_params=sp,
                               lora_request=lora_request)

        B = num_prompts * n
        total_len = max_prompt_len + max_response_length

        full_ids = torch.zeros(B, total_len, dtype=torch.long)
        full_mask = torch.zeros(B, total_len, dtype=torch.bool)
        responses = torch.zeros(B, max_response_length, dtype=torch.long)
        student_logprobs = torch.zeros(B, max_response_length, dtype=torch.float32)
        response_lengths = []
        full_token_lists = []

        flat_idx = 0
        for prompt_idx, out in enumerate(outputs):
            p_len = prompt_lengths[prompt_idx]
            pad_len = max_prompt_len - p_len

            for sample_idx in range(n):
                completion = out.outputs[sample_idx]
                resp_ids = list(completion.token_ids)[:max_response_length]
                resp_len = len(resp_ids)
                response_lengths.append(resp_len)

                # Fill input_ids and attention_mask (prompt part)
                full_ids[flat_idx, :max_prompt_len] = input_ids[prompt_idx]
                full_mask[flat_idx, pad_len:max_prompt_len] = True

                # Fill response part
                for j in range(resp_len):
                    full_ids[flat_idx, max_prompt_len + j] = resp_ids[j]
                    full_mask[flat_idx, max_prompt_len + j] = True
                    responses[flat_idx, j] = resp_ids[j]

                # Extract logprobs for this completion
                sl = completion.logprobs
                if sl is not None and len(sl) > 0:
                    # Tensor format (vLLM v1)
                    if hasattr(sl[0], "logprobs"):
                        lps = torch.stack([s.logprobs for s in sl])[:resp_len, 0]
                        student_logprobs[flat_idx, :resp_len] = lps.to(torch.float32)
                    # Raw tuple format
                    elif isinstance(sl[0], tuple) and len(sl[0]) == 3:
                        for t in range(min(len(sl), resp_len)):
                            _, logprobs_t, _ = sl[t]
                            student_logprobs[flat_idx, t] = logprobs_t[0][0].item()
                    # Dict format
                    else:
                        gen_ids = list(completion.token_ids)
                        for t in range(min(len(sl), resp_len)):
                            pos_dict = sl[t]
                            actual_id = gen_ids[t]
                            if actual_id in pos_dict:
                                student_logprobs[flat_idx, t] = pos_dict[actual_id].logprob

                # full_token_lists for teacher/reference scoring
                prompt_tokens = input_ids[prompt_idx][pad_len:].tolist()
                full_token_lists.append(prompt_tokens + resp_ids)

                flat_idx += 1

        # Repeat prompt_lengths for each sample
        prompt_lengths_repeated = []
        for p_len in prompt_lengths:
            prompt_lengths_repeated.extend([p_len] * n)

        return {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "responses": responses,
            "student_logprobs": student_logprobs,
            "prompt_lengths": torch.tensor(prompt_lengths_repeated, dtype=torch.long),
            "response_lengths": torch.tensor(response_lengths, dtype=torch.long),
            "full_token_lists": full_token_lists,
        }


def vllm_batch_rollout_worker_main(config, cmd_queue, result_queue):
    """Entry point for rollout worker subprocess. Thin wrapper around VLLMBatchRolloutWorker."""
    worker = VLLMBatchRolloutWorker(config)
    worker.run(cmd_queue, result_queue)

"""Async student vLLM rollout worker — runs as a subprocess on dedicated GPU(s).

Uses vLLM's AsyncLLM (V1 engine) for per-request streaming and instant
abort on weight sync. Used by fully_async scheduling mode.

Two generation modes:
  - Batch mode: asyncio.gather barrier (used for eval)
  - Streaming mode: per-sample push to result_queue (used in autonomous generation)

Two pause modes for weight sync:
  - "abort" (default): abort all in-flight requests, discard partial output
  - "keep": freeze requests, sync weights, reset KV cache, resume generation.
    Preserves in-flight requests across weight syncs. Each completed sample
    includes weight_breakpoints: [(token_idx, weight_version), ...] tracking
    which tokens were generated under which weight version.

Data samples are pickle-serialized to bytes before sending through mp.Queue
to avoid FD-passing race conditions ("received 0 items of ancdata") that
occur when multiple workers send tensor objects simultaneously.

Commands received via mp.Queue:
  ("generate", batch_dict)              -> batch generation (eval/standard)
  ("enter_autonomous", batch_dict)      -> start streaming generation
  ("exit_autonomous",)                  -> stop autonomous mode
  ("pause",)                            -> abort/keep in-flight, pause engine
  ("resume",)                           -> resume engine after pause
  ("sync_weights", update_info)         -> NCCL weight update (while paused)
  ("init_weight_transfer", init_info)   -> init vLLM NCCL engine
  ("get_vllm_params_info",)             -> return model param metadata
  ("compute_weight_checksum",)          -> return checksum of model weights
  ("shutdown",)                         -> exits
"""

import asyncio
import os
import pickle
import queue
import socket
import time
import uuid

import torch

from opd.launch_specs import RolloutLaunchSpec
from opd.rollout.base import BaseRolloutWorker
from opd.rollout.vllm.fast_mc import (
    align_single_pass_capture,
    clear_single_pass_mc_request_on_async_engine,
    install_fast_mc_async_add_request_patch,
    install_fast_mc_patch_rpc,
    pop_single_pass_mc_from_async_engine,
    set_single_pass_mc_config_on_async_engine,
)
from opd.rollout.vllm.utils import (
    extract_student_logprobs as _extract_student_logprobs,
    extract_student_topk_support as _extract_student_topk_support,
    setup_vllm_env, apply_pre_load_patches, build_common_llm_kwargs,
)


# ------------------------------------------------------------------ #
#  Module-level functions for collective_rpc (must be picklable)       #
# ------------------------------------------------------------------ #

from opd.rollout.vllm.utils import (  # noqa: E402
    disable_grad_fn as _disable_grad_fn,
    get_params_info_fn as _get_params_info_fn,
    compute_checksum_fn as _compute_checksum_fn,
)


# ------------------------------------------------------------------ #
#  StreamingRolloutWorker class                                            #
# ------------------------------------------------------------------ #

ALLOWED_COMMANDS = {"generate", "enter_autonomous", "pause", "exit_autonomous", "init_weight_transfer", "get_vllm_params_info", "sync_weights", "compute_weight_checksum"}

STREAM_METADATA_KEYS = (
    "ground_truth",
    "prompt_group_id",
    "logical_batch_id",
    "sample_in_batch_idx",
    "sample_seq_id",
    "gen_wv",
)


class VLLMStreamingRolloutWorker(BaseRolloutWorker):
    """Async vLLM rollout worker using AsyncLLM.

    Model is loaded lazily in run() since AsyncLLM requires an event loop.
    Can be subclassed or wrapped as a @ray.remote actor.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        if isinstance(config, RolloutLaunchSpec):
            config = config.merged_config()
        self._vllm_port = config.get("vllm_port")
        self._vllm_master_port = config.get("vllm_master_port")
        self.prompt_queue = config["prompt_queue"]
        pause_mode = config["pause_mode"]
        if pause_mode not in ("keep", "abort"):
            raise ValueError(
                f"Unsupported pause_mode='{pause_mode}'. "
                f"Use 'keep' (recommended) or 'abort'."
            )
        self.pause_mode = pause_mode
        # Instance state set during _async_main
        self._weight_version_ref = [0]
        self._autonomous_task = None
        # Output sink callables — set by _async_main (local) or externally (Ray).
        # Default to None; _generate_streaming/_generate_streaming_continuous
        # fall back to self._result_queue if these are not configured.
        self._output_fn = None
        self._status_fn = None
        self._sample_output_fn = None
        self._get_prompt = None
        self._fast_mc_installed = False

    def run(self, cmd_queue, result_queue, prompt_queue=None):
        """Entry point — sets up env and runs async main loop."""
        setup_vllm_env(self.gpu_ids, self.tp_size,
                       use_weight_transfer=self.use_weight_transfer,
                       native_lora=self.native_lora,
                       vllm_port=self._vllm_port,
                       vllm_master_port=self._vllm_master_port)
        self._setup_env(self.gpu_ids)
        asyncio.run(self._async_main(
            cmd_queue=cmd_queue,
            result_queue=result_queue,
            prompt_queue=prompt_queue or self.prompt_queue,
        ))

    # ---- Engine initialization (separable for Ray actor) ----

    async def _init_engine(self):
        """Initialize AsyncLLM engine, trace logger, and vLLM weight transfer imports.

        Sets self._engine, self._trace_logger, self._loop, and weight transfer
        request classes. Called by _async_main (local path) or directly by a
        Ray actor during init.
        """
        _is_fp8, _is_blockwise = apply_pre_load_patches(
            self.quantization, self.native_lora, self.lora_cfg)

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_kwargs = build_common_llm_kwargs(self)
        engine_kwargs["model"] = self.model_path
        engine_kwargs["enable_log_requests"] = False
        if self.use_weight_transfer:
            engine_kwargs["weight_transfer_config"] = {"backend": "nccl"}
        if _is_fp8:
            from opd.rollout.vllm.fp8 import FP8_BLOCK_QUANT_KWARGS
            engine_kwargs["quantization"] = "fp8"
            if _is_blockwise:
                engine_kwargs["hf_overrides"] = {"quantization_config": FP8_BLOCK_QUANT_KWARGS}
        if self.native_lora:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_loras"] = 1
            engine_kwargs["max_lora_rank"] = self.lora_rank

        engine_args = AsyncEngineArgs(**engine_kwargs)
        engine = AsyncLLM.from_engine_args(engine_args)
        # FP8 alignment check after model load (blockwise only)
        if _is_blockwise:
            from opd.rollout.vllm.fp8 import check_fp8_alignment
            await engine.collective_rpc("apply_model", args=(check_fp8_alignment,))
        # Register trace stat logger for throughput monitoring
        from opd.rollout.vllm.utils import TraceStatLogger
        self._trace_logger = TraceStatLogger()
        if engine.logger_manager is not None:
            engine.logger_manager.stat_loggers.append(self._trace_logger)
        print(f"[AsyncRollout-{self.worker_id}] AsyncLLM ready on GPUs {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

        self._engine = engine
        self._loop = asyncio.get_event_loop()

        # Import weight transfer request classes (needed by handlers)
        from vllm.distributed.weight_transfer.base import (
            WeightTransferInitRequest, WeightTransferUpdateRequest,
        )
        self._WeightTransferInitRequest = WeightTransferInitRequest
        self._WeightTransferUpdateRequest = WeightTransferUpdateRequest

    # ---- Async main loop ----

    async def _async_main(self, cmd_queue, result_queue, prompt_queue=None):
        """Async main loop — constructs AsyncLLM and runs command listener."""
        await self._init_engine()

        # Store queues for handler access (local path only)
        self._cmd_queue = cmd_queue
        self._result_queue = result_queue
        self._prompt_queue = prompt_queue

        # Configure output sinks for local (mp.Queue) path
        self._output_fn = lambda result: self._result_queue.put(pickle.dumps(result))
        self._status_fn = lambda result: self._result_queue.put(result)
        self._sample_output_fn = lambda sample: self._result_queue.put(pickle.dumps(sample))

        # Configure prompt source for local path
        loop = self._loop
        self._get_prompt = lambda: loop.run_in_executor(
            None, lambda: self._prompt_queue.get(timeout=0.5))

        # Run the command dispatch loop
        try:
            await self._dispatch_loop()
        except Exception as e:
            import traceback
            print(f"\n{'='*60}", flush=True)
            print(f"[AsyncRollout-{self.worker_id}] FATAL ERROR: {type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()
            print(f"{'='*60}\n", flush=True)
            return

        # Clean shutdown
        self._engine.shutdown()
        print(f"[AsyncRollout-{self.worker_id}] shut down", flush=True)

    # ---- Command dispatch loop ----

    async def _dispatch_loop(self):
        """Process commands from the pipeline orchestrator (local mp.Queue path)."""
        while True:
            cmd = await self._loop.run_in_executor(None, self._cmd_queue.get)

            if cmd[0] == "shutdown":
                if self._autonomous_task and not self._autonomous_task.done():
                    self._autonomous_task.cancel()
                    try:
                        await self._autonomous_task
                    except asyncio.CancelledError:
                        pass
                return

            if cmd[0] not in ALLOWED_COMMANDS:
                print(f"[AsyncRollout-{self.worker_id}] rejected unknown command: {cmd[0]}", flush=True)
                continue
            handler = getattr(self, f"_handle_{cmd[0]}", None)
            if handler:
                result = await handler(cmd)
                if result is not None:
                    # generate results are data (pickle-serialized); others are status dicts
                    if cmd[0] == "generate":
                        self._output_fn(result)
                    else:
                        self._status_fn(result)
                # After pause, enter the paused-state command loop (local path)
                if cmd[0] == "pause":
                    await self._pause_dispatch_loop()
            else:
                print(f"[AsyncRollout-{self.worker_id}] unknown command: {cmd[0]}", flush=True)

    # ---- Per-command handlers ----

    async def _handle_generate(self, cmd):
        """Handle 'generate' command — batch mode for eval/standard.

        Returns the result dict. The dispatch loop routes it via _output_fn.
        """
        batch = cmd[1]
        is_eval = batch.pop("eval", False)
        eval_temp = batch.pop("eval_temperature", None)
        eval_n = batch.pop("eval_n_samples", None)
        return_logprobs = batch.pop("return_logprobs", False)
        response_topk_k = batch.pop("response_topk_k", 0)
        mc_n_total_samples = batch.pop("mc_n_total_samples", 0)
        batch_max_resp = batch.pop("max_response_length", None) or self.max_response_length
        use_lora = batch.pop("use_lora", True)
        lora_req = self._lora_request if (self._lora_request and use_lora) else None

        t0 = time.time()
        if eval_n and eval_n > 1:
            result = await self._generate_batch_multi(
                batch, batch_max_resp,
                temperature=eval_temp or 0.6, n=eval_n,
                lora_request=lora_req)
        elif is_eval and eval_temp is None:
            result = await self._generate_batch(
                batch, batch_max_resp,
                temperature=0, top_p=1.0, top_k=-1,
                lora_request=lora_req)
        elif eval_temp is not None:
            result = await self._generate_batch(
                batch, batch_max_resp,
                temperature=eval_temp, top_p=0.95, top_k=-1,
                lora_request=lora_req)
        else:
            result = await self._generate_batch(
                batch, batch_max_resp,
                self.temperature, self.top_p, self.top_k,
                return_logprobs=return_logprobs,
                response_topk_k=response_topk_k,
                mc_n_total_samples=mc_n_total_samples,
                lora_request=lora_req)
        dt = time.time() - t0
        result["timing"] = {"generate_seconds": dt, "elapsed": dt,
                            "worker_id": self.worker_id,
                            "host": socket.gethostname(),
                            "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
                            "mono_start": time.monotonic() - dt,
                            "mono_end": time.monotonic()}
        result["_vllm_stats"] = self._trace_logger.drain()
        return result

    async def _handle_enter_autonomous(self, cmd):
        """Handle 'enter_autonomous' command — start streaming generation.

        Returns {"status": "autonomous_started"}.
        """
        batch = cmd[1]
        return_logprobs = batch.pop("return_logprobs", False)
        response_topk_k = batch.pop("response_topk_k", 0)
        mc_n_total_samples = batch.pop("mc_n_total_samples", 0)
        batch_max_resp = batch.pop("max_response_length", None) or self.max_response_length
        if self._prompt_queue is not None:
            self._autonomous_task = asyncio.create_task(
                self._generate_streaming_continuous(
                    batch, batch_max_resp,
                    return_logprobs=return_logprobs,
                    response_topk_k=response_topk_k,
                    mc_n_total_samples=mc_n_total_samples))
        else:
            self._autonomous_task = asyncio.create_task(
                self._generate_streaming(
                    batch, batch_max_resp,
                    return_logprobs=return_logprobs,
                    response_topk_k=response_topk_k,
                    mc_n_total_samples=mc_n_total_samples))
        return {"status": "autonomous_started"}

    async def _handle_pause(self, cmd):
        """Handle 'pause' command — abort/keep in-flight, then enter pause dispatch.

        Returns {"status": "paused", ...}. For local path, also enters
        _pause_dispatch_loop to handle subsequent paused-state commands.
        """
        t0 = time.time()
        if self.pause_mode == "keep":
            await self._engine.pause_generation(mode="keep", clear_cache=False)
        else:
            await self._engine.pause_generation(mode="abort")
            if self._autonomous_task and not self._autonomous_task.done():
                try:
                    await asyncio.wait_for(self._autonomous_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self._autonomous_task.cancel()
                    try:
                        await self._autonomous_task
                    except asyncio.CancelledError:
                        pass
                self._autonomous_task = None
        dt = time.time() - t0
        print(f"[AsyncRollout-{self.worker_id}] pause(mode={self.pause_mode})", flush=True)
        pause_result = {"status": "paused", "pause_seconds": dt,
                        "pause_mode": self.pause_mode,
                        "_vllm_stats": self._trace_logger.drain()}

        # Collect straggler samples from internal queue (Ray path only —
        # local path stragglers are already on result_queue via _sample_output_fn)
        stragglers = []
        if hasattr(self, '_internal_sample_queue'):
            while not self._internal_sample_queue.empty():
                try:
                    stragglers.append(self._internal_sample_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
        if stragglers:
            pause_result["_straggler_samples"] = stragglers

        return pause_result

    async def _pause_dispatch_loop(self):
        """Handle commands while paused — local mp.Queue path only.

        Reads from cmd_queue in a loop, delegating to _handle_pause_cmd_*
        methods and routing results via _status_fn.
        """
        while True:
            wake = await self._loop.run_in_executor(None, self._cmd_queue.get)
            if wake[0] == "sync_weights":
                result = await self._handle_pause_cmd_sync_weights(wake)
                self._status_fn(result)
            elif wake[0] == "resume":
                result = await self._handle_pause_cmd_resume(wake)
                self._status_fn(result)
                break
            elif wake[0] == "exit_autonomous":
                result = await self._handle_pause_cmd_exit_autonomous(wake)
                self._status_fn(result)
                break
            elif wake[0] == "compute_weight_checksum":
                result = await self._handle_compute_weight_checksum(wake)
                self._status_fn(result)
            elif wake[0] == "shutdown":
                return

    async def _handle_pause_cmd_sync_weights(self, cmd):
        """Handle sync_weights while paused. Returns {"status": "synced_nccl"}."""
        update_info = cmd[1]
        await self._engine.update_weights(
            self._WeightTransferUpdateRequest(update_info=update_info))
        self._weight_version_ref[0] += 1
        if self.pause_mode == "keep":
            await self._engine.reset_prefix_cache(
                reset_running_requests=True)
        print(f"[AsyncRollout-{self.worker_id}] sync_weights(v={self._weight_version_ref[0]}), reset_prefix_cache={self.pause_mode == 'keep'}", flush=True)
        return {"status": "synced_nccl"}

    async def _handle_pause_cmd_resume(self, cmd):
        """Handle resume while paused. Returns {"status": "resumed"}."""
        await self._engine.resume_generation()
        need_restart = (self.pause_mode == "abort" or
                        self._autonomous_task is None or
                        self._autonomous_task.done())
        print(f"[AsyncRollout-{self.worker_id}] resume(continued={not need_restart})", flush=True)
        if need_restart:
            if len(cmd) > 1 and cmd[1] is not None:
                batch = cmd[1]
                return_logprobs = batch.pop("return_logprobs", False)
                response_topk_k = batch.pop("response_topk_k", 0)
                mc_n_total_samples = batch.pop("mc_n_total_samples", 0)
                batch_max_resp = batch.pop("max_response_length", None) or self.max_response_length
                if self._prompt_queue is not None:
                    self._autonomous_task = asyncio.create_task(
                        self._generate_streaming_continuous(
                            batch, batch_max_resp,
                            return_logprobs=return_logprobs,
                            response_topk_k=response_topk_k,
                            mc_n_total_samples=mc_n_total_samples))
                else:
                    self._autonomous_task = asyncio.create_task(
                        self._generate_streaming(
                            batch, batch_max_resp,
                            return_logprobs=return_logprobs,
                            response_topk_k=response_topk_k,
                            mc_n_total_samples=mc_n_total_samples))
        return {"status": "resumed"}

    async def _handle_pause_cmd_exit_autonomous(self, cmd):
        """Handle exit_autonomous while paused. Returns {"status": "exited_autonomous", ...}."""
        n_cancelled = 0
        if self.pause_mode == "keep" and self._autonomous_task and not self._autonomous_task.done():
            # Count in-flight tasks before cancelling
            active = getattr(self, '_active_streaming_tasks', set())
            n_cancelled = sum(1 for t in active if not t.done())
            await self._engine.reset_prefix_cache(reset_running_requests=True)
            self._autonomous_task.cancel()
            try:
                await self._autonomous_task
            except asyncio.CancelledError:
                pass
            self._autonomous_task = None
        await self._engine.resume_generation()
        return {"status": "exited_autonomous",
                "n_cancelled": n_cancelled,
                "_vllm_stats": self._trace_logger.drain()}

    async def _handle_exit_autonomous(self, cmd):
        """Handle 'exit_autonomous' command — stop autonomous mode.

        Returns {"status": "exited_autonomous", ...}.
        """
        n_cancelled = 0
        if self._autonomous_task and not self._autonomous_task.done():
            active = getattr(self, '_active_streaming_tasks', set())
            n_cancelled = sum(1 for t in active if not t.done())
            await self._engine.pause_generation(mode="abort")
            try:
                await asyncio.wait_for(self._autonomous_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._autonomous_task.cancel()
                try:
                    await self._autonomous_task
                except asyncio.CancelledError:
                    pass
            await self._engine.resume_generation()
            self._autonomous_task = None
        return {"status": "exited_autonomous",
                "n_cancelled": n_cancelled,
                "_vllm_stats": self._trace_logger.drain()}

    async def _handle_init_weight_transfer(self, cmd):
        """Handle 'init_weight_transfer' command. Returns {"status": "ok"}."""
        init_info = cmd[1]
        t0 = time.time()
        await self._engine.init_weight_transfer_engine(
            self._WeightTransferInitRequest(init_info=init_info))
        await self._engine.collective_rpc("apply_model", args=(_disable_grad_fn,))
        dt = time.time() - t0
        print(f"[AsyncRollout-{self.worker_id}] Weight transfer engine init in {dt:.2f}s",
              flush=True)
        return {"status": "ok"}

    async def _handle_get_vllm_params_info(self, cmd):
        """Handle 'get_vllm_params_info' command. Returns {"params_info": ...}."""
        info = await self._engine.collective_rpc("apply_model", args=(_get_params_info_fn,))
        return {"params_info": info[0]}

    async def _handle_sync_weights(self, cmd):
        """Handle 'sync_weights' command — non-paused sync (used during init).

        Returns {"status": "synced_nccl", "sync_seconds": ...}.
        """
        update_info = cmd[1]
        is_lora = update_info.get("_lora_update", False)
        t0 = time.time()
        await self._engine.update_weights(
            self._WeightTransferUpdateRequest(update_info=update_info))
        self._weight_version_ref[0] += 1
        dt = time.time() - t0
        # After first LoRA sync, create the LoRARequest for generate() calls
        if is_lora and self._lora_request is None:
            from vllm.lora.request import LoRARequest as VLLMLoRARequest
            from opd.rollout.vllm.lora import LORA_NAME, LORA_INT_ID, LORA_PATH
            self._lora_request = VLLMLoRARequest(
                lora_name=LORA_NAME, lora_int_id=LORA_INT_ID,
                lora_path=LORA_PATH)
        kind = "LoRA" if is_lora else "NCCL"
        print(f"[AsyncRollout-{self.worker_id}] {kind} weight sync in {dt:.2f}s",
              flush=True)
        return {"status": "synced_nccl", "sync_seconds": dt}

    async def _handle_compute_weight_checksum(self, cmd):
        """Handle 'compute_weight_checksum' command. Returns {"checksum": ...}."""
        checksum = await self._engine.collective_rpc(
            "apply_model", args=(_compute_checksum_fn,))
        return {"checksum": checksum[0]}

    # ---- Batch mode generation ----

    async def _generate_batch(self, batch, max_response_length,
                              temperature, top_p, top_k,
                              return_logprobs=False, response_topk_k=0,
                              mc_n_total_samples=0, lora_request=None):
        """Batch generation using asyncio.gather — waits for all samples."""
        from vllm import SamplingParams

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=max_response_length,
            detokenize=False,
            logprobs=max(1 if return_logprobs else 0, response_topk_k) or None,
        )

        tasks = []
        prompt_id_lists = []
        request_ids = []
        if mc_n_total_samples > 0:
            await self._ensure_fast_mc_patch()
        for i in range(input_ids.size(0)):
            mask = attention_mask[i].bool()
            ids = input_ids[i][mask].tolist()
            prompt_id_lists.append(ids)
            request_id = f"gen-{i}-{uuid.uuid4().hex[:8]}"
            request_ids.append(request_id)
            if mc_n_total_samples > 0:
                await set_single_pass_mc_config_on_async_engine(
                    self._engine, request_id, mc_n_total_samples)
            tasks.append(self._run_single_request(request_id, ids, sp,
                                                  lora_request=lora_request))

        results = await asyncio.gather(*tasks)

        result = _assemble_batch_result(
            results, prompt_id_lists, input_ids, attention_mask,
            max_response_length, return_logprobs, response_topk_k)
        if mc_n_total_samples > 0:
            mc_idx = []
            mc_old = []
            for request_id, (_prompt_ids, output) in zip(request_ids, results, strict=False):
                capture = await self._pop_single_pass_mc_capture(request_id, mc_n_total_samples)
                mc_idx.append(capture["mc_query_indices_response"])
                mc_old.append(capture["mc_query_old_logprobs_response"])
            result["mc_query_indices_response"] = mc_idx
            result["mc_query_old_logprobs_response"] = mc_old
        return result

    async def _generate_batch_multi(self, batch, max_response_length,
                                    temperature, n, lora_request=None):
        """Avg@N generation — submits N separate requests per prompt."""
        from vllm import SamplingParams

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        sp = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_response_length,
            n=1,
            detokenize=False,
        )

        prompt_ids_list = []
        tasks = []
        for i in range(input_ids.size(0)):
            mask = attention_mask[i].bool()
            ids = input_ids[i][mask].tolist()
            prompt_ids_list.append(ids)
            for s in range(n):
                request_id = f"multi-{i}-s{s}-{uuid.uuid4().hex[:8]}"
                tasks.append((i, self._run_single_request(request_id, ids, sp,
                                                         lora_request=lora_request)))

        coros = [t for _, t in tasks]
        prompt_indices = [idx for idx, _ in tasks]
        results = await asyncio.gather(*coros)

        responses_multi = [[] for _ in range(input_ids.size(0))]
        for prompt_idx, (prompt_ids, output) in zip(prompt_indices, results):
            if output is None or not output.outputs:
                continue
            resp_ids = list(output.outputs[0].token_ids)[:max_response_length]
            responses_multi[prompt_idx].append(resp_ids)

        return {"responses_multi": responses_multi}

    async def _run_single_request(self, request_id, prompt_token_ids, sp,
                                   lora_request=None):
        """Run one request to completion, return (prompt_ids, final RequestOutput)."""
        from vllm import TokensPrompt

        final_output = None
        try:
            async for output in self._engine.generate(
                TokensPrompt(prompt_token_ids=prompt_token_ids), sp, request_id,
                lora_request=lora_request):
                final_output = output
        except asyncio.CancelledError:
            return (prompt_token_ids, None)
        return (prompt_token_ids, final_output)

    # ---- Streaming mode generation ----

    async def _generate_streaming(self, batch, max_response_length,
                                  return_logprobs=False, response_topk_k=0,
                                  mc_n_total_samples=0):
        """Submit all prompts, push each completed sample individually.

        No all-sample barrier — fast samples flow to the collector immediately.
        Aborted requests (from pause_generation) are silently dropped.
        Uses self._sample_output_fn to route completed samples.
        """
        from vllm import SamplingParams, TokensPrompt

        engine = self._engine
        sample_output_fn = getattr(self, '_sample_output_fn', None) or (
            lambda sample: self._result_queue.put(pickle.dumps(sample)))
        weight_version_ref = self._weight_version_ref
        worker_id = self.worker_id
        temperature = self.temperature
        top_p = self.top_p
        top_k = self.top_k

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        max_prompt_len = input_ids.size(1)

        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=max_response_length,
            detokenize=False,
            logprobs=max(1 if return_logprobs else 0, response_topk_k) or None,
        )

        completion_queue = asyncio.Queue()
        if mc_n_total_samples > 0:
            await self._ensure_fast_mc_patch()

        async def _run_and_enqueue(idx, prompt_ids, prompt_len, pad_len):
            """Run one request, push formatted sample to completion_queue when done."""
            request_id = f"stream-{worker_id}-{idx}-{uuid.uuid4().hex[:8]}"
            if mc_n_total_samples > 0:
                await set_single_pass_mc_config_on_async_engine(
                    engine, request_id, mc_n_total_samples)
            final_output = None
            rollout_t0 = time.monotonic()
            start_version = weight_version_ref[0]
            breakpoints = [(0, start_version)]
            last_version = start_version
            try:
                async for output in engine.generate(
                    TokensPrompt(prompt_token_ids=prompt_ids), sp, request_id,
                    lora_request=self._lora_request):
                    final_output = output
                    cur_version = weight_version_ref[0]
                    if cur_version != last_version:
                        num_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
                        breakpoints.append((num_tokens, cur_version))
                        last_version = cur_version
            except asyncio.CancelledError:
                if mc_n_total_samples > 0:
                    await clear_single_pass_mc_request_on_async_engine(engine, request_id)
                return
            rollout_t1 = time.monotonic()

            if final_output is None:
                if mc_n_total_samples > 0:
                    await clear_single_pass_mc_request_on_async_engine(engine, request_id)
                return
            if not final_output.finished:
                if mc_n_total_samples > 0:
                    await clear_single_pass_mc_request_on_async_engine(engine, request_id)
                return
            if (final_output.outputs
                    and final_output.outputs[0].finish_reason == "abort"):
                if mc_n_total_samples > 0:
                    await clear_single_pass_mc_request_on_async_engine(engine, request_id)
                return

            sample = _process_single_sample(
                prompt_ids, prompt_len, pad_len, final_output,
                max_prompt_len, max_response_length, return_logprobs,
                response_topk_k,
                input_ids_row=input_ids[idx],
            )
            if mc_n_total_samples > 0:
                capture = await self._pop_single_pass_mc_capture(request_id, mc_n_total_samples)
                capture = align_single_pass_capture(
                    capture,
                    len(final_output.outputs[0].token_ids),
                    getattr(final_output.outputs[0], "finish_reason", None),
                    expected_token_ids=final_output.outputs[0].token_ids,
                )
                sample["mc_query_indices_response"] = [capture["mc_query_indices_response"]]
                sample["mc_query_old_logprobs_response"] = [capture["mc_query_old_logprobs_response"]]
            total_tokens = len(final_output.outputs[0].token_ids)
            if total_tokens > 0 and len(breakpoints) > 1:
                wavg = 0.0
                for i, (tok_idx, ver) in enumerate(breakpoints):
                    next_idx = breakpoints[i + 1][0] if i + 1 < len(breakpoints) else total_tokens
                    wavg += ver * (next_idx - tok_idx)
                sample["weight_version"] = wavg / total_tokens
            else:
                sample["weight_version"] = breakpoints[0][1] if breakpoints else weight_version_ref[0]
            sample["weight_breakpoints"] = breakpoints
            sample["worker_id"] = worker_id
            sample["host"] = socket.gethostname()
            sample["gpu_ids"] = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            sample["rollout_request_id"] = request_id
            sample["rollout_mono_start"] = rollout_t0
            sample["rollout_mono_end"] = rollout_t1
            sample["rollout_elapsed"] = rollout_t1 - rollout_t0
            await completion_queue.put(sample)

        tasks = []
        for i in range(input_ids.size(0)):
            mask = attention_mask[i].bool()
            ids = input_ids[i][mask].tolist()
            p_len = len(ids)
            pad_len = max_prompt_len - p_len
            tasks.append(asyncio.create_task(
                _run_and_enqueue(i, ids, p_len, pad_len)))

        completed = 0
        total = len(tasks)
        try:
            while completed < total:
                try:
                    sample = await asyncio.wait_for(completion_queue.get(), timeout=0.5)
                    sample_output_fn(sample)
                    completed += 1
                except asyncio.TimeoutError:
                    all_done = all(t.done() for t in tasks)
                    if all_done:
                        break
        except asyncio.CancelledError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            return 0, total

        return completed, total

    async def _generate_streaming_continuous(self, first_batch, max_response_length,
                                             return_logprobs=False, response_topk_k=0,
                                             mc_n_total_samples=0):
        """Continuously generate with maximum vLLM utilization.

        Uses a semaphore to keep up to max_num_seqs requests in flight.
        Uses self._sample_output_fn to route completed samples and
        self._get_prompt to read the next prompt.
        """
        from vllm import SamplingParams, TokensPrompt

        engine = self._engine
        sample_output_fn = getattr(self, '_sample_output_fn', None) or (
            lambda sample: self._result_queue.put(pickle.dumps(sample)))
        get_prompt = getattr(self, '_get_prompt', None) or (
            lambda: asyncio.get_event_loop().run_in_executor(
                None, lambda: self._prompt_queue.get(timeout=0.5)))
        weight_version_ref = self._weight_version_ref
        worker_id = self.worker_id
        temperature = self.temperature
        top_p = self.top_p
        top_k = self.top_k

        sem = asyncio.Semaphore(self.max_num_seqs)
        if mc_n_total_samples > 0:
            await self._ensure_fast_mc_patch()
        active_tasks = set()
        self._active_streaming_tasks = active_tasks  # expose for exit_autonomous

        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=max_response_length,
            detokenize=False,
            logprobs=max(1 if return_logprobs else 0, response_topk_k) or None,
        )

        async def _run_one(prompt_info):
            """Generate one request and push result via sample_output_fn."""
            prompt_ids = prompt_info["prompt_ids"]
            prompt_len = prompt_info["prompt_len"]
            pad_len = prompt_info["pad_len"]
            input_ids_row = prompt_info["input_ids_row"]
            rl = prompt_info.get("return_logprobs", return_logprobs)
            topk_k = prompt_info.get("response_topk_k", response_topk_k)
            batch_max_resp = prompt_info.get("max_response_length", max_response_length)

            cur_sp = sp
            if (batch_max_resp != max_response_length
                    or rl != return_logprobs
                    or topk_k != response_topk_k):
                cur_sp = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k if top_k > 0 else -1,
                    max_tokens=batch_max_resp,
                    detokenize=False,
                    logprobs=max(1 if rl else 0, topk_k) or None,
                )

            request_id = f"stream-{worker_id}-{uuid.uuid4().hex[:8]}"
            mc_n = prompt_info.get("mc_n_total_samples", mc_n_total_samples)
            if mc_n > 0:
                await set_single_pass_mc_config_on_async_engine(
                    engine, request_id, mc_n)
            final_output = None
            rollout_t0 = time.monotonic()
            start_version = weight_version_ref[0]
            breakpoints = [(0, start_version)]
            last_version = start_version

            try:
                async for output in engine.generate(
                    TokensPrompt(prompt_token_ids=prompt_ids), cur_sp, request_id,
                    lora_request=self._lora_request):
                    final_output = output
                    cur_version = weight_version_ref[0]
                    if cur_version != last_version:
                        num_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
                        breakpoints.append((num_tokens, cur_version))
                        last_version = cur_version
            except asyncio.CancelledError:
                if mc_n > 0:
                    await clear_single_pass_mc_request_on_async_engine(engine, request_id)
                return
            rollout_t1 = time.monotonic()

            if final_output is None or not final_output.finished:
                if mc_n > 0:
                    await clear_single_pass_mc_request_on_async_engine(engine, request_id)
                return
            if (final_output.outputs
                    and final_output.outputs[0].finish_reason == "abort"):
                if mc_n > 0:
                    await clear_single_pass_mc_request_on_async_engine(engine, request_id)
                return

            sample = _process_single_sample(
                prompt_ids, prompt_len, pad_len, final_output,
                prompt_len + pad_len, batch_max_resp, rl, topk_k,
                input_ids_row=input_ids_row,
            )
            if mc_n > 0:
                capture = await self._pop_single_pass_mc_capture(request_id, mc_n)
                capture = align_single_pass_capture(
                    capture,
                    len(final_output.outputs[0].token_ids),
                    getattr(final_output.outputs[0], "finish_reason", None),
                    expected_token_ids=final_output.outputs[0].token_ids,
                )
                sample["mc_query_indices_response"] = [capture["mc_query_indices_response"]]
                sample["mc_query_old_logprobs_response"] = [capture["mc_query_old_logprobs_response"]]
            total_tokens = len(final_output.outputs[0].token_ids)
            if total_tokens > 0 and len(breakpoints) > 1:
                wavg = 0.0
                for i, (tok_idx, ver) in enumerate(breakpoints):
                    next_idx = breakpoints[i + 1][0] if i + 1 < len(breakpoints) else total_tokens
                    wavg += ver * (next_idx - tok_idx)
                sample["weight_version"] = wavg / total_tokens
            else:
                sample["weight_version"] = breakpoints[0][1] if breakpoints else weight_version_ref[0]
            sample["weight_breakpoints"] = breakpoints
            sample["worker_id"] = worker_id
            sample["host"] = socket.gethostname()
            sample["gpu_ids"] = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            sample["rollout_request_id"] = request_id
            sample["rollout_mono_start"] = rollout_t0
            sample["rollout_mono_end"] = rollout_t1
            sample["rollout_elapsed"] = rollout_t1 - rollout_t0
            # Forward prompt metadata for fully-async GRPO and strict
            # step-off logical-batch reassembly.
            for extra_key in STREAM_METADATA_KEYS:
                if extra_key in prompt_info:
                    sample[extra_key] = prompt_info[extra_key]
            sample_output_fn(sample)

        async def _submit(prompt_info):
            """Acquire semaphore, run one request, release on completion."""
            await sem.acquire()
            task = asyncio.create_task(_run_one(prompt_info))
            active_tasks.add(task)
            task.add_done_callback(lambda t: (active_tasks.discard(t), sem.release()))

        # Seed: submit all prompts from first_batch directly.
        # Skip seeding for GRPO streaming — seed prompts lack ground_truth
        # and prompt_group_id metadata required for reward/advantage computation.
        # The PromptFeeder will provide properly tagged prompts shortly.
        is_grpo_streaming = first_batch.get("grpo_n_samples", 1) > 1
        if not is_grpo_streaming:
            input_ids = first_batch["input_ids"]
            attention_mask = first_batch["attention_mask"]
            max_prompt_len = input_ids.size(1)
            for i in range(input_ids.size(0)):
                mask = attention_mask[i].bool()
                ids = input_ids[i][mask].tolist()
                p_len = len(ids)
                pad_len = max_prompt_len - p_len
                await _submit({
                    "prompt_ids": ids,
                    "prompt_len": p_len,
                    "pad_len": pad_len,
                    "input_ids_row": input_ids[i],
                    "return_logprobs": return_logprobs,
                    "response_topk_k": response_topk_k,
                    "mc_n_total_samples": mc_n_total_samples,
                })

        # Main loop: pull individual prompts via _get_prompt and submit
        while True:
            try:
                prompt_info = await asyncio.wait_for(
                    get_prompt(),
                    timeout=1.0)
            except (queue.Empty, asyncio.TimeoutError):
                if not active_tasks:
                    self._trace_logger.flush_idle()
                continue
            except asyncio.CancelledError:
                break

            if prompt_info is None:  # sentinel for shutdown
                break

            try:
                await _submit(prompt_info)
            except asyncio.CancelledError:
                break

        # Cancel remaining active tasks
        for t in active_tasks:
            if not t.done():
                t.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    async def _ensure_fast_mc_patch(self):
        if self._fast_mc_installed:
            return
        results = await install_fast_mc_patch_rpc(
            self._engine,
            fallback_topk_k=max(1, self.max_logprobs),
            enable_native_sampler_capture=False,
        )
        self._fast_mc_installed = bool(results) and all(results)
        if self._fast_mc_installed:
            install_fast_mc_async_add_request_patch(self._engine)
        if not self._fast_mc_installed:
            raise RuntimeError("Failed to install rollout fast MC patch for AsyncLLM")

    async def _pop_single_pass_mc_capture(self, request_id, n_total_samples):
        return await pop_single_pass_mc_from_async_engine(
            self._engine,
            request_id,
            n_total_samples,
        )


# ------------------------------------------------------------------ #
#  Static helper (used by both streaming modes)                        #
# ------------------------------------------------------------------ #

def _process_single_sample(prompt_ids, prompt_len, pad_len, output,
                           max_prompt_len, max_response_length,
                           return_logprobs, response_topk_k=0, input_ids_row=None):
    """Convert a single RequestOutput into a dict matching the batch format."""
    total_len = max_prompt_len + max_response_length

    resp_ids = list(output.outputs[0].token_ids)
    resp_len = min(len(resp_ids), max_response_length)

    full_ids = torch.zeros(1, total_len, dtype=torch.long)
    full_mask = torch.zeros(1, total_len, dtype=torch.bool)
    responses = torch.zeros(1, max_response_length, dtype=torch.long)

    full_ids[0, :max_prompt_len] = input_ids_row
    full_mask[0, pad_len:max_prompt_len] = True

    for j in range(resp_len):
        full_ids[0, max_prompt_len + j] = resp_ids[j]
        full_mask[0, max_prompt_len + j] = True
        responses[0, j] = resp_ids[j]

    student_logprobs = None
    if return_logprobs:
        student_logprobs = torch.zeros(1, max_response_length, dtype=torch.float32)
        _extract_student_logprobs(output, student_logprobs, 0, resp_len)
    query_indices_response = None
    query_logprobs_response = None
    if response_topk_k > 0:
        topk_idx, topk_logps = _extract_student_topk_support(
            output, resp_len=resp_len, topk_k=response_topk_k)
        query_indices_response = [topk_idx]
        query_logprobs_response = [topk_logps]

    full_token_list = prompt_ids + resp_ids[:resp_len]

    result = {
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "responses": responses,
        "prompt_lengths": torch.tensor([prompt_len], dtype=torch.long),
        "response_lengths": torch.tensor([resp_len], dtype=torch.long),
        "full_token_lists": [full_token_list],
    }
    if student_logprobs is not None:
        result["student_logprobs"] = student_logprobs
    if query_indices_response is not None:
        result["query_indices_response"] = query_indices_response
        result["query_logprobs_response"] = query_logprobs_response
    return result


def _assemble_batch_result(results, prompt_id_lists, input_ids, attention_mask,
                           max_response_length, return_logprobs, response_topk_k=0):
    """Assemble per-request results into a batch dict matching sync _do_generate()."""
    batch_size = len(results)
    max_prompt_len = input_ids.size(1)
    total_len = max_prompt_len + max_response_length

    full_ids = torch.zeros(batch_size, total_len, dtype=torch.long)
    full_mask = torch.zeros(batch_size, total_len, dtype=torch.bool)
    responses = torch.zeros(batch_size, max_response_length, dtype=torch.long)
    prompt_lengths = []
    response_lengths = []
    student_logprobs = (torch.zeros(batch_size, max_response_length, dtype=torch.float32)
                        if return_logprobs else None)
    query_indices_response = [] if response_topk_k > 0 else None
    query_logprobs_response = [] if response_topk_k > 0 else None

    full_token_lists = []

    for i, (prompt_ids, output) in enumerate(results):
        p_len = len(prompt_ids)
        prompt_lengths.append(p_len)

        if output is None or not output.outputs:
            response_lengths.append(0)
            full_token_lists.append(prompt_ids)
            pad_len = max_prompt_len - p_len
            full_ids[i, :max_prompt_len] = input_ids[i]
            full_mask[i, pad_len:max_prompt_len] = True
            continue

        resp_ids = list(output.outputs[0].token_ids)
        resp_len = min(len(resp_ids), max_response_length)
        response_lengths.append(resp_len)

        pad_len = max_prompt_len - p_len
        full_ids[i, :max_prompt_len] = input_ids[i]
        full_mask[i, pad_len:max_prompt_len] = True

        for j in range(resp_len):
            full_ids[i, max_prompt_len + j] = resp_ids[j]
            full_mask[i, max_prompt_len + j] = True
            responses[i, j] = resp_ids[j]

        if return_logprobs and student_logprobs is not None:
            _extract_student_logprobs(output, student_logprobs, i, resp_len)
        if response_topk_k > 0 and query_indices_response is not None:
            topk_idx, topk_logps = _extract_student_topk_support(
                output, resp_len=resp_len, topk_k=response_topk_k)
            query_indices_response.append(topk_idx)
            query_logprobs_response.append(topk_logps)

        full_token_lists.append(prompt_ids + resp_ids[:resp_len])

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
    return result


# ------------------------------------------------------------------ #
#  Entry point (thin wrapper for subprocess compatibility)              #
# ------------------------------------------------------------------ #

def vllm_streaming_rollout_worker_main(config, cmd_queue, result_queue):
    """Entry point for async rollout worker subprocess. Thin wrapper around VLLMStreamingRolloutWorker."""
    worker = VLLMStreamingRolloutWorker(config)
    worker.run(cmd_queue, result_queue)

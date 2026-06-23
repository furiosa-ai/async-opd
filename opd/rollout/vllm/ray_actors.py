"""Ray actor wrappers for vLLM rollout workers.

Contains the Ray actor classes that wrap VLLMBatchRolloutWorker and
VLLMStreamingRolloutWorker for use in Ray-based pipelines.

Ray is an optional dependency -- this module is only imported when
``pipeline.backend == "ray"`` in the config.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from opd.launch_specs import ensure_rollout_launch_spec

try:
    import ray
except ImportError:
    ray = None  # guarded at call sites


class _SplitResultQueue:
    """Routes worker put() calls to either cmd_res_q or data_q.

    Routing rule: pickle-serialized bytes (streaming samples and batch
    generate results) go to data_q.  Everything else (status dicts,
    params_info, checksums) goes to cmd_res_q.
    """

    def __init__(self, cmd_res_q: queue.Queue, data_q: queue.Queue):
        self._cmd_res_q = cmd_res_q
        self._data_q = data_q

    def put(self, item, *args, **kwargs):
        if isinstance(item, bytes):
            self._data_q.put(item, *args, **kwargs)
        else:
            self._cmd_res_q.put(item, *args, **kwargs)

    def put_nowait(self, item):
        if isinstance(item, bytes):
            self._data_q.put_nowait(item)
        else:
            self._cmd_res_q.put_nowait(item)

    def get(self, *args, **kwargs):
        return self._cmd_res_q.get(*args, **kwargs)

    def get_nowait(self):
        return self._cmd_res_q.get_nowait()

    def empty(self):
        return self._cmd_res_q.empty() and self._data_q.empty()


class VLLMBatchRolloutActor:
    """Ray actor wrapping an existing rollout worker via internal queues.

    The worker's ``run(cmd_queue, result_queue)`` method blocks on
    ``cmd_queue.get()`` in a loop.  We run it in a daemon thread and
    translate Ray method calls into queue put/get pairs.

    In streaming mode, a _SplitResultQueue splits worker output into
    _cmd_res_q (command responses) and _data_q (streaming data / generate
    results) to prevent deadlocks between concurrent callers.
    """

    def __init__(self, worker_cls, config: dict):
        # queue.Queue (stdlib) is sufficient — worker thread and actor
        # methods share the same process address space.  Avoids the
        # overhead of mp.Queue (pickle + shared-memory pipes).
        self._cmd_q = queue.Queue()
        self._cmd_res_q = queue.Queue()   # command responses
        self._data_q = queue.Queue()      # streaming data + generate results

        # Prompt queue for streaming mode — always use a local queue
        # inside the actor. The coordinator feeds prompts via the actor's
        # feed_prompt() method, NOT via a shared cross-process queue
        # (ray.util.queue.Queue deadlocks inside actor threads).
        launch_spec = ensure_rollout_launch_spec(config)
        use_prompt_queue = launch_spec.runtime.prompt_queue is not None
        worker_spec = launch_spec.with_runtime(prompt_queue=None)
        if use_prompt_queue:
            # Streaming mode: use split queue so worker output is routed
            # to the correct consumer (command vs data).
            self._prompt_queue = queue.Queue()
            result_queue = _SplitResultQueue(self._cmd_res_q, self._data_q)
        else:
            # Batch mode: no split needed — all results are command
            # responses (generate results go through generate() which
            # reads _cmd_res_q).
            self._prompt_queue = None
            result_queue = self._cmd_res_q

        # Construct worker (loads model onto this actor's GPU)
        self._worker = worker_cls(worker_spec)

        # Run the blocking command loop in a background thread
        self._thread = threading.Thread(
            target=self._worker.run,
            args=(self._cmd_q, result_queue, self._prompt_queue),
            daemon=True,
        )
        self._thread.start()

    # -- Thin RPC wrappers: put command, get result --

    def generate(self, batch_dict: dict) -> Any:
        """Send generate command and return the result.

        In batch mode the result lands on _cmd_res_q (no split).
        In streaming mode the pickled result lands on _data_q via
        the split queue, so we read from _data_q.
        """
        self._cmd_q.put(("generate", batch_dict))
        if self._prompt_queue is not None:
            # Streaming worker: generate result is pickle bytes → _data_q
            return self._data_q.get()
        return self._cmd_res_q.get()

    def command(self, cmd_name: str, *args) -> Any:
        """Send an arbitrary command and return its result."""
        self._cmd_q.put((cmd_name, *args))
        return self._cmd_res_q.get()

    def command_nowait(self, cmd_name: str, *args) -> None:
        """Send a command without waiting for a result."""
        self._cmd_q.put((cmd_name, *args))

    def collect_result(self) -> Any:
        """Collect one command result from _cmd_res_q."""
        return self._cmd_res_q.get()

    def collect_result_timeout(self, timeout: float = 30.0) -> Any:
        """Collect one data result with timeout. Returns None on timeout.

        Used by RaySampleStream to poll for streaming data samples.
        """
        try:
            return self._data_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def collect_cmd_result_timeout(self, timeout: float = 30.0) -> Any:
        """Collect one command result with timeout. Returns None on timeout.

        Used by _drain_until_status_ray to poll for status messages.
        """
        try:
            return self._cmd_res_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def feed_prompt(self, prompt_info: dict, timeout: float = 1.0) -> bool:
        """Feed a prompt to the actor's local prompt queue."""
        if self._prompt_queue is None:
            return False
        try:
            self._prompt_queue.put(prompt_info, timeout=timeout)
            return True
        except Exception:
            return False

    def feed_prompt_sentinel(self) -> None:
        """Send sentinel (None) to prompt queue for shutdown."""
        if self._prompt_queue is not None:
            try:
                self._prompt_queue.put(None, timeout=1.0)
            except Exception:
                pass

    def shutdown(self) -> None:
        self._cmd_q.put(("shutdown",))

    def get_node_ip(self):
        import socket
        return socket.gethostbyname(socket.gethostname())

    def alive(self) -> bool:
        return self._thread.is_alive()


class VLLMStreamingRolloutActor:
    """Native asyncio Ray actor for streaming rollout.

    Delegates to VLLMStreamingRolloutWorker._handle_* methods directly,
    bypassing the cmd_queue dispatch loop. No deadlock -- all async.

    No max_concurrency -- all methods are async def, so Ray runs them
    on the actor's single-threaded asyncio event loop cooperatively.
    """

    async def init(self, config):
        """Initialize worker and engine. Called after actor creation."""
        import asyncio
        from opd.rollout.vllm.streaming import VLLMStreamingRolloutWorker

        launch_spec = ensure_rollout_launch_spec(config)
        worker_spec = launch_spec.with_runtime(prompt_queue=None)

        self._worker = VLLMStreamingRolloutWorker(worker_spec)

        # Set up environment (CUDA_VISIBLE_DEVICES etc.) before engine init
        self._worker._setup_env(self._worker.gpu_ids, {
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        })

        # Internal asyncio queues for Ray-native prompt/sample flow
        self._prompt_queue = asyncio.Queue()
        self._sample_queue = asyncio.Queue()

        # Init the vLLM engine (requires running event loop -- we have one
        # since this is an async Ray actor)
        await self._worker._init_engine()

        # Configure worker for Ray mode:
        # - _get_prompt reads from asyncio.Queue (not mp.Queue)
        # - _sample_output_fn puts to asyncio.Queue (not pickle+result_queue)
        # - _prompt_queue set to a truthy value so enter_autonomous picks
        #   _generate_streaming_continuous (continuous prompt feed mode)
        self._worker._prompt_queue = self._prompt_queue  # truthy sentinel
        self._worker._get_prompt = lambda: self._prompt_queue.get()
        self._worker._sample_output_fn = lambda sample: self._sample_queue.put_nowait(sample)
        self._worker._internal_sample_queue = self._sample_queue

    async def enter_autonomous(self, batch):
        return await self._worker._handle_enter_autonomous(("enter_autonomous", batch))

    async def pause(self):
        return await self._worker._handle_pause(("pause",))

    async def resume(self, seed_batch=None):
        return await self._worker._handle_pause_cmd_resume(("resume", seed_batch))

    async def exit_autonomous(self):
        return await self._worker._handle_pause_cmd_exit_autonomous(("exit_autonomous",))

    async def sync_weights(self, update_info):
        return await self._worker._handle_pause_cmd_sync_weights(("sync_weights", update_info))

    async def generate(self, batch):
        return await self._worker._handle_generate(("generate", batch))

    async def init_weight_transfer(self, info):
        return await self._worker._handle_init_weight_transfer(("init_weight_transfer", info))

    async def get_worker_info(self):
        import os, socket
        return {"host": socket.gethostname(),
                "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "?")}

    async def get_vllm_params_info(self):
        return await self._worker._handle_get_vllm_params_info(("get_vllm_params_info",))

    async def compute_weight_checksum(self, *args):
        cmd = ("compute_weight_checksum",) + args
        return await self._worker._handle_compute_weight_checksum(cmd)

    async def feed_prompt(self, prompt_info):
        await self._prompt_queue.put(prompt_info)
        return True

    async def get_samples(self, max_n=8, timeout=1.0):
        """Return up to max_n samples. Batched to amortize Ray RPC overhead."""
        import asyncio
        samples = []
        try:
            # Get at least one (with timeout)
            sample = await asyncio.wait_for(self._sample_queue.get(), timeout=timeout)
            samples.append(sample)
            # Drain more without waiting
            while len(samples) < max_n:
                sample = self._sample_queue.get_nowait()
                samples.append(sample)
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            pass
        return samples

    async def feed_prompt_sentinel(self):
        await self._prompt_queue.put(None)

    async def shutdown(self):
        pass  # Worker cleanup happens when actor is destroyed

    async def get_node_ip(self):
        import socket
        return socket.gethostbyname(socket.gethostname())

    def alive(self):
        return True

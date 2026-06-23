"""Ray-based worker proxy implementations.

Provides Ray actor wrappers around existing worker classes (VLLMBatchRolloutWorker,
VLLMStreamingRolloutWorker, FSDPTrainer) and Ray-aware proxy implementations of the
RolloutProxy / TrainerProxy protocols.

Workers are COMPLETELY UNTOUCHED -- the Ray actor spawns an internal thread
running the worker's existing command loop, and communicates via queue.Queue
pairs (stdlib thread-safe deque).  GPU tensors stay on the actor's GPU; only
CPU-ified results cross the Ray object store.

In streaming mode the single result queue is replaced by a _SplitResultQueue
that routes worker output to two separate queues:
  - _cmd_res_q: command responses (status dicts, params_info, checksums, etc.)
  - _data_q:    streaming data samples (pickle-serialized bytes) and batch
                generate results (also pickle bytes)
This eliminates the deadlock where collect_result_timeout (data polling) could
steal command responses meant for command() / drain_until_status.

Ray is an optional dependency -- this module is only imported when
``pipeline.backend == "ray"`` in the config.
"""

import pickle
import queue
import threading
import time
from collections import deque
from typing import Any

import torch

from opd.launch_specs import ensure_teacher_launch_spec, ensure_trainer_launch_spec

try:
    import ray
except ImportError:
    ray = None  # guarded at call sites


class FSDPTrainerActor:
    """Ray actor wrapping FSDPTrainer via internal queues.

    Supports single-GPU (fsdp_world_size=1) and multi-GPU FSDP.
    For multi-GPU (fsdp_world_size>1), uses deferred init to avoid
    init_process_group deadlock: __init__ is lightweight, init() does
    the real construction. For single-GPU, __init__ constructs immediately
    (backward compat).
    """

    def __init__(self, trainer_cls, config: dict, rank_info: dict | None = None):
        self._cmd_q = queue.Queue()
        self._res_q = queue.Queue()
        self._thread = None
        self._launch_spec = ensure_trainer_launch_spec(config, rank_info)

        fsdp_world_size = self._launch_spec.runtime.rank_info.get("fsdp_world_size", 1)

        if fsdp_world_size > 1:
            # Deferred init: store config/rank_info, don't construct trainer yet.
            # init_process_group is a collective that would deadlock if
            # rank-0 is constructed before other ranks exist.
            self._trainer_cls = trainer_cls
            self._trainer = None
        else:
            # Single-GPU: construct immediately (backward compat)
            self._trainer = trainer_cls(self._launch_spec)
            self._thread = threading.Thread(
                target=self._trainer.run,
                args=(self._cmd_q, self._res_q),
                daemon=True,
            )
            self._thread.start()

    def init(self, fsdp_master_addr: str, fsdp_master_port: int):
        """Deferred init for multi-rank FSDP. Call on all actors in parallel."""
        assert self._trainer is None, "init() called but trainer already constructed"
        rank_info = dict(self._launch_spec.runtime.rank_info)
        rank_info["fsdp_master_addr"] = fsdp_master_addr
        rank_info["fsdp_master_port"] = fsdp_master_port
        self._trainer = self._trainer_cls(
            self._launch_spec.with_runtime(rank_info=rank_info)
        )
        self._thread = threading.Thread(
            target=self._trainer.run,
            args=(self._cmd_q, self._res_q),
            daemon=True,
        )
        self._thread.start()
        return True

    def get_worker_info(self):
        import os, socket
        return {"host": socket.gethostname(),
                "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "?")}

    def train(self, batch: dict) -> dict:
        self._cmd_q.put(("train", batch))
        return self._res_q.get()

    def command(self, cmd_name: str, *args) -> Any:
        self._cmd_q.put((cmd_name, *args))
        return self._res_q.get()

    def command_nowait(self, cmd_name: str, *args) -> None:
        self._cmd_q.put((cmd_name, *args))

    def collect_result(self) -> Any:
        return self._res_q.get()

    def get_node_ip(self):
        """Return routable IP. Callable before init() for deferred mode."""
        try:
            import ray
            return ray.util.get_node_ip_address()
        except Exception:
            import socket
            return socket.gethostbyname(socket.gethostname())

    def shutdown(self) -> None:
        self._cmd_q.put(("shutdown",))

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ------------------------------------------------------------------ #
#  Ray Rollout Proxy                                                   #
# ------------------------------------------------------------------ #

class RayRolloutProxy:
    """Implements the RolloutProxy protocol using Ray actor handles.

    Translates the submit/collect pattern into ray.get() on actor
    method refs.  Batch splitting mirrors QueueRolloutProxy exactly.
    """

    def __init__(self, actors: list, need_student_logprobs: bool = False,
                 rollout_support_topk_k: int = 0,
                 prompt_queue=None):
        self._actors = actors
        self._need_student_logprobs = need_student_logprobs
        self._rollout_support_topk_k = rollout_support_topk_k
        self._prompt_queue = prompt_queue
        # _pending_gens is a FIFO of batches of Ray object refs -- one entry
        # per submit_generate call.  collect_generate pops from the front.
        # This mirrors mp.Queue semantics where multiple generates can be
        # in flight (step_off > 1).
        self._pending_gens: deque[list] = deque()
        self._pending_cmd: list = []

    @property
    def n_workers(self) -> int:
        return len(self._actors)

    # --- Batch generate ---

    def submit_generate(self, batch_dict: dict) -> None:
        n = len(self._actors)
        bs = batch_dict["input_ids"].size(0)
        chunk = (bs + n - 1) // n
        refs = []
        for i, actor in enumerate(self._actors):
            s, e = i * chunk, min((i + 1) * chunk, bs)
            sub = {k: v[s:e] if isinstance(v, torch.Tensor) else v
                   for k, v in batch_dict.items()}
            if self._need_student_logprobs:
                sub["return_logprobs"] = True
            if self._rollout_support_topk_k > 0:
                sub["response_topk_k"] = self._rollout_support_topk_k
            refs.append(actor.generate.remote(sub))
        self._pending_gens.append(refs)

    def collect_generate(self) -> dict:
        refs = self._pending_gens.popleft()
        results = ray.get(refs)

        # Deserialize any pickled results
        results = [pickle.loads(r) if isinstance(r, bytes) else r for r in results]

        if len(results) == 1:
            r = results[0]
            r["_worker_timings"] = [r.get("timing", {})]
            r["_vllm_stats"] = {0: r.pop("_vllm_stats", [])}
            return r

        merged = {}
        vllm_stats = {}
        for key in results[0]:
            if key == "_vllm_stats":
                for i, r in enumerate(results):
                    wid = r.get("timing", {}).get("worker_id", i)
                    vllm_stats[wid] = r.get("_vllm_stats", [])
            elif key in {"full_token_lists", "responses_multi",
                         "query_indices_response", "query_logprobs_response"}:
                merged[key] = sum((r[key] for r in results), [])
            elif key == "timing":
                merged["_worker_timings"] = [r[key] for r in results]
                merged[key] = {
                    tk: max(r[key].get(tk, 0) for r in results)
                    for tk in results[0][key]
                    if tk != "worker_id"
                }
            elif isinstance(results[0][key], torch.Tensor):
                merged[key] = torch.cat([r[key] for r in results], dim=0)
            else:
                merged[key] = results[0][key]
        merged["_vllm_stats"] = vllm_stats
        return merged

    # --- Command broadcast ---

    def submit_command(self, command: str, *args: Any) -> None:
        self._pending_cmd = [
            actor.command.remote(command, *args) for actor in self._actors
        ]

    def collect_results(self) -> list[dict]:
        results = ray.get(self._pending_cmd)
        self._pending_cmd = []
        return results

    def submit_command_per_worker(self, command: str, per_worker_args: list[tuple]) -> None:
        self._pending_cmd = [
            actor.command.remote(command, *args)
            for actor, args in zip(self._actors, per_worker_args)
        ]

    def shutdown(self) -> None:
        for actor in self._actors:
            try:
                ray.get(actor.shutdown.remote())
            except Exception:
                pass

    # --- Streaming lifecycle ---

    def enter_autonomous(self, sub_batches: list[dict]) -> list[dict]:
        # enter_autonomous sends the command and the worker responds with
        # "autonomous_started" as the first result. Use command() which
        # does put+get atomically — the ack arrives before any data samples
        # because the worker sends it before entering the generation loop.
        refs = [
            actor.command.remote("enter_autonomous", sub_batches[i])
            for i, actor in enumerate(self._actors)
        ]
        results = ray.get(refs)
        # Results are the ack dicts (may be pickle bytes)
        status_msgs = []
        for r in results:
            if isinstance(r, bytes):
                r = pickle.loads(r)
            status_msgs.append(r)
        return status_msgs

    def pause_workers(self) -> list[dict]:
        refs = [actor.command_nowait.remote("pause") for actor in self._actors]
        ray.get(refs)  # wait for puts to complete
        return self._drain_until_status_ray("paused")

    def resume_workers(self, seed_batches: list[dict] | None = None) -> list[dict]:
        refs = []
        for i, actor in enumerate(self._actors):
            seed = seed_batches[i] if seed_batches else None
            refs.append(actor.command_nowait.remote("resume", seed))
        ray.get(refs)
        return self._drain_until_status_ray("resumed")

    def exit_autonomous(self) -> list[dict]:
        refs = [actor.command_nowait.remote("exit_autonomous") for actor in self._actors]
        ray.get(refs)
        return self._drain_until_status_ray("exited_autonomous")

    def drain_until_status(self, target_status: str,
                           forward_target=None) -> list[dict]:
        return self._drain_until_status_ray(target_status, forward_target=forward_target)

    def _drain_until_status_ray(self, target_status: str,
                                forward_target=None,
                                initial_refs=None) -> list[dict]:
        """Drain command results from actors until target_status seen from all.

        Uses collect_cmd_result_timeout which reads from the actor's _cmd_res_q
        (command response queue), NOT the _data_q (streaming data queue).
        This avoids stealing data samples meant for RaySampleStream.
        """
        status_msgs = []
        for i, actor in enumerate(self._actors):
            while True:
                ref = actor.collect_cmd_result_timeout.remote(30.0)
                raw = ray.get(ref)
                if raw is None:
                    # Timeout -- check if actor is alive
                    try:
                        alive = ray.get(actor.alive.remote())
                    except Exception:
                        alive = False
                    if not alive:
                        print(f"[RayRolloutProxy] Actor {i} died while draining "
                              f"for '{target_status}'", flush=True)
                        status_msgs.append({})
                        break
                    continue
                r = pickle.loads(raw) if isinstance(raw, bytes) else raw
                if isinstance(r, dict) and r.get("status") == target_status:
                    status_msgs.append(r)
                    break
                if isinstance(r, dict) and "status" not in r and forward_target is not None:
                    wv = r.get("weight_version", 0)
                    forward_target.put(r, weight_version=wv)
        return status_msgs

    # --- Stage interface factories ---

    def sample_stream(self):
        """Return a RaySampleStream wrapping this proxy's actors."""
        return RaySampleStream(self._actors)

    def prompt_sink(self):
        """Return a RayPromptSink that feeds prompts via actor method calls."""
        return RayPromptSink(self._actors)


# ------------------------------------------------------------------ #
#  Ray Sample Stream                                                   #
# ------------------------------------------------------------------ #

class RaySampleStream:
    """SampleStream implementation for Ray actors.

    Polls actors in round-robin for results from their internal queues.
    """

    def __init__(self, actors: list):
        self._actors = actors
        self._next_idx = 0

    def get_sample(self, timeout: float = 0.05) -> dict | None:
        n = len(self._actors)
        if n == 0:
            return None
        # Use a longer timeout inside the actor (1s) to amortize Ray RPC
        # overhead. The actor waits up to 1s for data, then returns None.
        actor_timeout = max(1.0, timeout)
        for _ in range(n):
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % n
            actor = self._actors[idx]
            try:
                raw = ray.get(actor.collect_result_timeout.remote(actor_timeout))
            except Exception:
                continue
            if raw is None:
                continue
            result = pickle.loads(raw) if isinstance(raw, bytes) else raw
            if isinstance(result, dict) and "status" in result:
                continue
            return result
        return None


# ------------------------------------------------------------------ #
#  Ray Prompt Sink                                                     #
# ------------------------------------------------------------------ #

class RayPromptSink:
    """PromptSink that feeds prompts to Ray rollout actors via method calls.

    Distributes prompts round-robin across actors. Each actor has a local
    queue.Queue that its worker thread reads from — avoids ray.util.queue
    deadlock issues inside actor threads.
    """

    def __init__(self, actors: list):
        self._actors = actors
        self._next_idx = 0

    @property
    def n_workers(self) -> int:
        return len(self._actors)

    def put(self, prompt_info: dict, timeout: float = 1.0) -> bool:
        """Feed a prompt to the next actor (round-robin). Returns True if accepted."""
        n = len(self._actors)
        if n == 0:
            return False
        idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % n
        try:
            result = ray.get(self._actors[idx].feed_prompt.remote(prompt_info, timeout))
            return result
        except Exception:
            return False

    def send_sentinel(self) -> None:
        """Send shutdown sentinel to all actors' prompt queues."""
        for actor in self._actors:
            try:
                ray.get(actor.feed_prompt_sentinel.remote())
            except Exception:
                pass


# ------------------------------------------------------------------ #
#  Native async Ray rollout proxy (for VLLMStreamingRolloutActor)       #
# ------------------------------------------------------------------ #

class RayStreamingRolloutProxy:
    """RolloutProxy implementation for VLLMStreamingRolloutActor (native async).

    Unlike RayRolloutProxy which communicates via queue.Queue pairs inside
    a threaded actor, this proxy calls async actor methods directly.
    Each method returns results synchronously via ray.get().
    """

    def __init__(self, actors: list, need_student_logprobs: bool = False,
                 rollout_support_topk_k: int = 0):
        self._actors = actors
        self._need_student_logprobs = need_student_logprobs
        self._rollout_support_topk_k = rollout_support_topk_k
        self._pending_cmd: dict[str, list] = {}  # cmd_name -> list of ObjectRefs
        self._pending_gens: deque[list] = deque()

    @property
    def n_workers(self) -> int:
        return len(self._actors)

    # --- Batch generate ---

    def submit_generate(self, batch_dict: dict) -> None:
        n = len(self._actors)
        bs = batch_dict["input_ids"].size(0)
        chunk = (bs + n - 1) // n
        refs = []
        for i, actor in enumerate(self._actors):
            s, e = i * chunk, min((i + 1) * chunk, bs)
            sub = {k: v[s:e] if isinstance(v, torch.Tensor) else v
                   for k, v in batch_dict.items()}
            if self._need_student_logprobs:
                sub["return_logprobs"] = True
            if self._rollout_support_topk_k > 0:
                sub["response_topk_k"] = self._rollout_support_topk_k
            refs.append(actor.generate.remote(sub))
        self._pending_gens.append(refs)

    def collect_generate(self) -> dict:
        refs = self._pending_gens.popleft()
        results = ray.get(refs)

        # Deserialize any pickled results
        results = [pickle.loads(r) if isinstance(r, bytes) else r for r in results]

        if len(results) == 1:
            r = results[0]
            r["_worker_timings"] = [r.get("timing", {})]
            r["_vllm_stats"] = {0: r.pop("_vllm_stats", [])}
            return r

        merged = {}
        vllm_stats = {}
        for key in results[0]:
            if key == "_vllm_stats":
                for i, r in enumerate(results):
                    wid = r.get("timing", {}).get("worker_id", i)
                    vllm_stats[wid] = r.get("_vllm_stats", [])
            elif key == "full_token_lists" or key == "responses_multi":
                merged[key] = sum((r[key] for r in results), [])
            elif key == "timing":
                merged["_worker_timings"] = [r[key] for r in results]
                merged[key] = {
                    tk: max(r[key].get(tk, 0) for r in results)
                    for tk in results[0][key]
                    if tk != "worker_id"
                }
            elif isinstance(results[0][key], torch.Tensor):
                merged[key] = torch.cat([r[key] for r in results], dim=0)
            else:
                merged[key] = results[0][key]
        merged["_vllm_stats"] = vllm_stats
        return merged

    # --- Command dispatch ---

    def submit_command(self, command: str, *args: Any) -> None:
        """Dispatch command to all actors via their native async methods."""
        cmd_map = {
            "pause": lambda a: a.pause.remote(),
            "resume": lambda a: a.resume.remote(args[0] if args else None),
            "exit_autonomous": lambda a: a.exit_autonomous.remote(),
            "sync_weights": lambda a: a.sync_weights.remote(args[0]),
            "init_weight_transfer": lambda a: a.init_weight_transfer.remote(args[0]),
            "get_vllm_params_info": lambda a: a.get_vllm_params_info.remote(),
            "compute_weight_checksum": lambda a: a.command.remote("compute_weight_checksum", *args),
            "sync_weights_collective": lambda a: a.command.remote("sync_weights_collective", args[0]),
        }
        dispatch = cmd_map.get(command)
        if dispatch is None:
            raise ValueError(f"Unknown command: {command}")
        self._pending_cmd[command] = [dispatch(a) for a in self._actors]

    def collect_results(self) -> list[dict]:
        """Collect results from the most recently submitted command."""
        if not self._pending_cmd:
            return []
        cmd, refs = next(reversed(self._pending_cmd.items()))
        results = ray.get(refs)
        del self._pending_cmd[cmd]
        return results

    def submit_command_per_worker(self, command: str, per_worker_args: list[tuple]) -> None:
        """Send command with different args to each worker."""
        cmd_map = {
            "init_weight_transfer": lambda a, args_: a.init_weight_transfer.remote(args_[0]),
            "compute_weight_checksum": lambda a, args_: a.command.remote("compute_weight_checksum", *args_),
        }
        dispatch = cmd_map.get(command)
        if dispatch is None:
            raise ValueError(f"Unknown per-worker command: {command}")
        self._pending_cmd[command] = [
            dispatch(actor, pw_args)
            for actor, pw_args in zip(self._actors, per_worker_args)
        ]

    def drain_until_status(self, target_status: str,
                           forward_target=None) -> list[dict]:
        """Resolve pending refs for the command that produces target_status.

        Unlike RayRolloutProxy which polls a queue, async actors return
        results directly from method calls -- no interleaved messages.
        Stragglers are extracted from _straggler_samples in the result dict.
        """
        cmd_key = {
            "paused": "pause",
            "resumed": "resume",
            "exited_autonomous": "exit_autonomous",
            "synced_nccl": "sync_weights",
            "autonomous_started": "enter_autonomous",
            "ok": "init_weight_transfer",
        }.get(target_status)

        refs = self._pending_cmd.pop(cmd_key, None)
        if refs is None:
            return [{}] * len(self._actors)

        results = ray.get(refs)
        status_msgs = []
        for r in results:
            if isinstance(r, bytes):
                r = pickle.loads(r)
            # Forward stragglers if present
            if forward_target and isinstance(r, dict) and "_straggler_samples" in r:
                for sample in r.pop("_straggler_samples", []):
                    wv = sample.get("weight_version", 0)
                    forward_target.put(sample, weight_version=wv)
            status_msgs.append(r)
        return status_msgs

    # --- Streaming lifecycle ---

    def enter_autonomous(self, sub_batches: list[dict]) -> list[dict]:
        refs = [
            actor.enter_autonomous.remote(sub_batches[i])
            for i, actor in enumerate(self._actors)
        ]
        results = ray.get(refs)
        status_msgs = []
        for r in results:
            if isinstance(r, bytes):
                r = pickle.loads(r)
            status_msgs.append(r)
        return status_msgs

    def pause_workers(self) -> list[dict]:
        self.submit_command("pause")
        return self.drain_until_status("paused")

    def resume_workers(self, seed_batches: list[dict] | None = None) -> list[dict]:
        refs = []
        for i, actor in enumerate(self._actors):
            seed = seed_batches[i] if seed_batches else None
            refs.append(actor.resume.remote(seed))
        self._pending_cmd["resume"] = refs
        return self.drain_until_status("resumed")

    def exit_autonomous(self) -> list[dict]:
        self.submit_command("exit_autonomous")
        return self.drain_until_status("exited_autonomous")

    def shutdown(self) -> None:
        for actor in self._actors:
            try:
                ray.get(actor.shutdown.remote())
            except Exception:
                pass

    # --- Stage interface factories ---

    def sample_stream(self):
        """Return a RayStreamingSampleStream wrapping this proxy's actors."""
        return RayStreamingSampleStream(self._actors)

    def prompt_sink(self):
        """Return a RayStreamingPromptSink that feeds prompts via actor method calls."""
        return RayStreamingPromptSink(self._actors)


# ------------------------------------------------------------------ #
#  Async Ray Sample Stream                                             #
# ------------------------------------------------------------------ #

class RayStreamingSampleStream:
    """SampleStream for VLLMStreamingRolloutActor.

    Uses get_samples() which batches multiple samples per RPC call to
    amortize Ray overhead. Polls actors in round-robin.
    """

    def __init__(self, actors: list, max_n: int = 8):
        self._actors = actors
        self._next_idx = 0
        self._max_n = max_n
        self._buffer: list = []

    def get_sample(self, timeout: float = 0.05) -> dict | None:
        # Return from buffer first
        if self._buffer:
            return self._buffer.pop(0)

        n = len(self._actors)
        if n == 0:
            return None

        # Round-robin poll actors
        for _ in range(n):
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % n
            actor = self._actors[idx]
            try:
                samples = ray.get(actor.get_samples.remote(self._max_n, max(0.5, timeout)))
            except Exception:
                continue
            if not samples:
                continue
            # First sample returned directly, rest buffered
            self._buffer.extend(samples[1:])
            return samples[0]
        return None


# ------------------------------------------------------------------ #
#  Async Ray Prompt Sink                                               #
# ------------------------------------------------------------------ #

class RayStreamingPromptSink:
    """PromptSink for VLLMStreamingRolloutActor.

    Feeds prompts round-robin via actor.feed_prompt.remote().
    """

    def __init__(self, actors: list):
        self._actors = actors
        self._next_idx = 0

    @property
    def n_workers(self) -> int:
        return len(self._actors)

    def put(self, prompt_info: dict, timeout: float = 1.0) -> bool:
        n = len(self._actors)
        if n == 0:
            return False
        idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % n
        try:
            result = ray.get(self._actors[idx].feed_prompt.remote(prompt_info))
            return result
        except Exception:
            return False

    def send_sentinel(self) -> None:
        for actor in self._actors:
            try:
                ray.get(actor.feed_prompt_sentinel.remote())
            except Exception:
                pass


# ------------------------------------------------------------------ #
#  Ray Trainer Proxy                                                   #
# ------------------------------------------------------------------ #

class RayTrainerProxy:
    """Implements TrainerProxy protocol using Ray FSDPTrainerActor(s).

    Supports single actor (backward compat) or multiple actors for multi-rank
    FSDP. When multiple actors, all commands route to actors[0] (rank-0).
    Non-rank-0 actors receive commands via torch.distributed.broadcast_object_list
    inside BaseTrainer.run().
    """

    def __init__(self, actor=None, actors=None):
        if actors is not None:
            self._actor = actors[0]      # rank-0
            self._actors = actors         # all ranks (for shutdown)
        else:
            self._actor = actor
            self._actors = [actor]
        self._pending_train = None
        self._pending_sync = None
        self._checkpoint_save_pending = False

    @property
    def proc(self):
        """Compatibility -- Ray actors don't have a proc."""
        return None

    @property
    def fsdp_procs(self):
        return []

    def submit_train(self, gen_output: dict, teacher_output: dict) -> None:
        batch = {
            "input_ids": gen_output["input_ids"],
            "attention_mask": gen_output["attention_mask"],
            "responses": gen_output["responses"],
            "prompt_lengths": gen_output["prompt_lengths"],
        }
        if "support_topk_logps" in teacher_output:
            batch["support_topk_logps"] = teacher_output["support_topk_logps"]
            batch["support_topk_indices"] = teacher_output["support_topk_indices"]
            batch["support_valid_mask"] = teacher_output["support_valid_mask"]
            if "support_student_old_logps" in teacher_output:
                batch["support_student_old_logps"] = teacher_output["support_student_old_logps"]
        else:
            batch["teacher_topk_logps"] = teacher_output["teacher_topk_logps"]
            batch["teacher_topk_indices"] = teacher_output["teacher_topk_indices"]
            batch["teacher_valid_mask"] = teacher_output["teacher_valid_mask"]
        if "teacher_token_logps" in teacher_output:
            batch["teacher_token_logps"] = teacher_output["teacher_token_logps"]
        if "student_logprobs" in gen_output:
            batch["student_logprobs"] = gen_output["student_logprobs"]
        batch["_send_mono"] = time.monotonic()
        self._pending_train = self._actor.train.remote(batch)

    def collect_train(self) -> dict:
        result = ray.get(self._pending_train)
        self._pending_train = None
        return result

    def submit_save_checkpoint(self, step: int, checkpoint_dir: str,
                               save_optimizer: bool = True) -> None:
        assert not self._checkpoint_save_pending, \
            "Previous checkpoint save not drained before dispatching new one"
        self._actor.command_nowait.remote("save_checkpoint", {
            "step": step,
            "checkpoint_dir": checkpoint_dir,
            "save_optimizer": save_optimizer,
        })
        self._checkpoint_save_pending = True

    def collect_checkpoint_save(self) -> dict | None:
        if self._checkpoint_save_pending:
            result = ray.get(self._actor.collect_result.remote())
            self._checkpoint_save_pending = False
            return result
        return None

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        result = ray.get(self._actor.command.remote("load_checkpoint", {
            "checkpoint_dir": checkpoint_dir,
        }))
        return result.get("step", 0)

    def submit_command(self, cmd_name: str, *args: Any) -> Any:
        return ray.get(self._actor.command.remote(cmd_name, *args))

    def submit_command_async(self, cmd_name: str, *args: Any) -> None:
        self._actor.command_nowait.remote(cmd_name, *args)

    def collect_command(self) -> Any:
        return ray.get(self._actor.collect_result.remote())

    def shutdown(self) -> None:
        # Shutdown rank-0 first (broadcasts shutdown to other ranks internally)
        try:
            ray.get(self._actor.shutdown.remote())
        except Exception:
            pass
        # Shutdown remaining actors (their run() loops have already exited
        # via broadcast, but we clean up the Ray actors)
        for actor in self._actors[1:]:
            try:
                ray.get(actor.shutdown.remote(), timeout=10)
            except Exception:
                pass

    # --- Async weight sync ---

    def submit_sync_weights(self, weight_merge_map: list) -> None:
        self._actor.command_nowait.remote("sync_weights", weight_merge_map)

    def collect_sync_weights(self) -> dict:
        return ray.get(self._actor.collect_result.remote())


# ------------------------------------------------------------------ #
#  Megatron Trainer Actor + Proxy                                      #
# ------------------------------------------------------------------ #

class MegatronTrainerActor:
    """Ray actor wrapping one rank of MegatronTrainer (TP×PP×DP).

    Each actor represents one global rank of the Megatron trainer.  Uses
    deferred init: __init__ stores config, init() starts the trainer thread
    after the master address is known (needed for multi-node NCCL).

    Usage:
        actor = RemoteActor.remote(...)
        master_ip = ray.get(actors[0].get_node_ip.remote())
        ray.get(actor.init.remote(megatron_master_addr=master_ip))
    """

    def __init__(self, config: dict, rank_info: dict | None = None):
        self._launch_spec = ensure_trainer_launch_spec(config, rank_info)
        self._global_rank = self._launch_spec.runtime.rank_info.get("global_rank", 0)
        self._cmd_q = queue.Queue()
        self._res_q = queue.Queue()
        self._thread = None

    def init(self, megatron_master_addr: str = "127.0.0.1"):
        """Start the trainer thread with the known master address.

        Called after all actors are placed so rank-0's IP is available.
        """
        from opd.trainer.megatron import megatron_trainer_main

        rank_info = dict(self._launch_spec.runtime.rank_info)
        rank_info["megatron_master_addr"] = megatron_master_addr
        launch_spec = self._launch_spec.with_runtime(rank_info=rank_info)
        global_rank = self._global_rank
        self._thread = threading.Thread(
            target=megatron_trainer_main,
            args=(
                launch_spec,
                self._cmd_q if global_rank == 0 else None,
                self._res_q if global_rank == 0 else None,
                None,
            ),
            daemon=True,
        )
        self._thread.start()
        return True

    # -- Thin RPC wrappers (same interface as FSDPTrainerActor) --

    def train(self, batch: dict) -> dict:
        self._cmd_q.put(("train", batch))
        return self._res_q.get()

    def command(self, cmd_name: str, *args) -> Any:
        self._cmd_q.put((cmd_name, *args))
        return self._res_q.get()

    def command_nowait(self, cmd_name: str, *args) -> None:
        self._cmd_q.put((cmd_name, *args))

    def collect_result(self) -> Any:
        return self._res_q.get()

    def shutdown(self) -> None:
        self._cmd_q.put(("shutdown",))

    def get_node_ip(self):
        import socket
        return socket.gethostbyname(socket.gethostname())

    def get_worker_info(self):
        import os
        import socket
        return {
            "host": socket.gethostname(),
            "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
            "global_rank": self._global_rank,
        }

    def alive(self) -> bool:
        return self._thread.is_alive()


class RayMegatronTrainerProxy:
    """Implements TrainerProxy protocol for a Megatron TP actor group.

    Wraps a list of MegatronTrainerActor handles (one per TP rank).
    All commands are routed to rank-0 only -- rank-0's internal
    ``MegatronTrainer.run()`` broadcasts command names to other TP ranks
    via ``torch.distributed.broadcast_object_list``.
    """

    def __init__(self, actors: list):
        self._actors = actors  # actors[0] is rank-0
        self._rank0 = actors[0]
        self._pending_train = None
        self._pending_cmd = None
        self._checkpoint_save_pending = False

    @property
    def proc(self):
        """Compatibility -- Ray actors don't have a proc."""
        return None

    @property
    def fsdp_procs(self):
        return []

    def submit_train(self, gen_output: dict, teacher_output: dict) -> None:
        batch = {
            "input_ids": gen_output["input_ids"],
            "attention_mask": gen_output["attention_mask"],
            "responses": gen_output["responses"],
            "prompt_lengths": gen_output["prompt_lengths"],
            "teacher_topk_logps": teacher_output["teacher_topk_logps"],
            "teacher_topk_indices": teacher_output["teacher_topk_indices"],
            "teacher_valid_mask": teacher_output["teacher_valid_mask"],
        }
        if "teacher_token_logps" in teacher_output:
            batch["teacher_token_logps"] = teacher_output["teacher_token_logps"]
        if "student_logprobs" in gen_output:
            batch["student_logprobs"] = gen_output["student_logprobs"]
        batch["_send_mono"] = time.monotonic()
        self._pending_train = self._rank0.train.remote(batch)

    def collect_train(self) -> dict:
        result = ray.get(self._pending_train)
        self._pending_train = None
        return result

    def submit_save_checkpoint(self, step: int, checkpoint_dir: str,
                               save_optimizer: bool = True) -> None:
        assert not self._checkpoint_save_pending, \
            "Previous checkpoint save not drained before dispatching new one"
        self._rank0.command_nowait.remote("save_checkpoint", {
            "step": step,
            "checkpoint_dir": checkpoint_dir,
            "save_optimizer": save_optimizer,
        })
        self._checkpoint_save_pending = True

    def collect_checkpoint_save(self) -> dict | None:
        if self._checkpoint_save_pending:
            result = ray.get(self._rank0.collect_result.remote())
            self._checkpoint_save_pending = False
            return result
        return None

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        result = ray.get(self._rank0.command.remote("load_checkpoint", {
            "checkpoint_dir": checkpoint_dir,
        }))
        return result.get("step", 0)

    def submit_command(self, cmd_name: str, *args: Any) -> Any:
        return ray.get(self._rank0.command.remote(cmd_name, *args))

    def submit_command_async(self, cmd_name: str, *args: Any) -> None:
        """Fire-and-forget command to rank-0. Pair with collect_command()."""
        self._rank0.command_nowait.remote(cmd_name, *args)

    def collect_command(self) -> Any:
        """Block until the pending async command completes."""
        return ray.get(self._rank0.collect_result.remote())

    def shutdown(self) -> None:
        # Shutdown rank-0 first (it broadcasts to other ranks)
        try:
            ray.get(self._rank0.shutdown.remote())
        except Exception:
            pass
        # Other ranks will exit when they receive the broadcast shutdown
        for actor in self._actors[1:]:
            try:
                ray.get(actor.shutdown.remote(), timeout=10)
            except Exception:
                pass

    # --- Async weight sync ---

    def submit_sync_weights(self, weight_merge_map: list) -> None:
        self._rank0.command_nowait.remote("sync_weights", weight_merge_map)

    def collect_sync_weights(self) -> dict:
        return ray.get(self._rank0.collect_result.remote())


# ------------------------------------------------------------------ #
#  Teacher Ray Actor                                                   #
# ------------------------------------------------------------------ #

class VLLMTeacherActor:
    """Ray actor that runs the teacher vLLM/HF server in a background thread.

    Same teacher_fn as the mp.Process path — just wrapped in a Ray actor
    so Ray can schedule it on any node.
    """

    def init(self, teacher_fn, teacher_config):
        """Start the teacher server in a daemon thread."""
        import os
        import socket
        import threading
        launch_spec = ensure_teacher_launch_spec(teacher_config)
        self._host = socket.gethostname()
        self._ip = self._resolve_ip()
        self._gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        self._port = launch_spec.runtime.bind_port
        self._thread = threading.Thread(
            target=teacher_fn, args=(launch_spec,), daemon=True)
        self._thread.start()

    @staticmethod
    def _resolve_ip():
        """Get routable IP for this node."""
        try:
            import ray
            return ray.util.get_node_ip_address()
        except Exception:
            import socket
            return socket.gethostbyname(socket.gethostname())

    def get_info(self):
        return {"host": self._host, "ip": self._ip,
                "gpu_ids": self._gpu_ids, "port": self._port}

    def alive(self):
        return self._thread.is_alive()

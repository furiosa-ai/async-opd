"""Worker proxy abstractions for coordinator <-> worker communication.

Defines Protocol interfaces (RolloutProxy, TrainerProxy, WeightSyncEngine)
and queue-based implementations that wrap the existing mp.Queue pattern.
Worker subprocesses are untouched -- this abstraction lives entirely on
the coordinator side.
"""

from __future__ import annotations

import pickle
import queue
import time
from typing import Any, Protocol, runtime_checkable

import torch

from opd.utils.net import find_free_port
from opd.worker.weight_merge import build_weight_merge_map


# ------------------------------------------------------------------ #
#  Protocols                                                          #
# ------------------------------------------------------------------ #

@runtime_checkable
class ForwardTarget(Protocol):
    """Target for forwarding straggler data samples during drain loops."""

    def put(self, item: dict, weight_version: int) -> None: ...


@runtime_checkable
class SampleStream(Protocol):
    """Stream of deserialized, status-filtered samples from rollout workers."""

    def get_sample(self, timeout: float = 0.05) -> dict | None: ...


@runtime_checkable
class PromptSink(Protocol):
    """Sink for feeding individual prompts to rollout workers."""

    def put(self, prompt_info: dict, timeout: float = 1.0) -> bool: ...
    def send_sentinel(self) -> None: ...

    @property
    def n_workers(self) -> int: ...


@runtime_checkable
class RolloutProxy(Protocol):
    """Proxy for communicating with rollout worker(s)."""

    n_workers: int

    # --- Existing ---
    def submit_generate(self, batch_dict: dict) -> None: ...
    def collect_generate(self) -> dict: ...
    def submit_command(self, command: str, *args: Any) -> None: ...
    def collect_results(self, poll_timeout: float = 30.0,
                        purpose: str = "command results") -> list[dict]: ...
    def shutdown(self) -> None: ...

    # --- Streaming lifecycle ---
    def enter_autonomous(self, sub_batches: list[dict]) -> list[dict]: ...
    def pause_workers(self) -> list[dict]: ...
    def resume_workers(self, seed_batches: list[dict] | None = None) -> list[dict]: ...
    def exit_autonomous(self) -> list[dict]: ...
    def drain_until_status(self, target_status: str,
                           forward_target: ForwardTarget | None = None) -> list[dict]: ...

    # --- Per-worker commands (for NCCLWeightSyncEngine) ---
    def submit_command_per_worker(self, command: str, per_worker_args: list[tuple]) -> None: ...

    # --- Stage interface factories ---
    def sample_stream(self) -> SampleStream: ...
    def prompt_sink(self) -> PromptSink: ...


@runtime_checkable
class TrainerProxy(Protocol):
    """Proxy for communicating with trainer worker(s)."""

    def submit_train(self, gen_output: dict, teacher_output: dict) -> None: ...
    def submit_train_direct_teacher_artifacts(
        self,
        gen_output: dict,
        *,
        teacher_buffer_id: int,
        logical_batch_id: int,
        gen_weight_version: int,
        expected_samples: int,
        timeout_s: float = 300.0,
    ) -> None: ...
    def collect_train(self) -> dict: ...
    def submit_save_checkpoint(self, step: int, checkpoint_dir: str,
                               save_optimizer: bool = True) -> None: ...
    def collect_checkpoint_save(self) -> dict | None: ...
    def load_checkpoint(self, checkpoint_dir: str) -> int: ...
    def submit_command(self, cmd_name: str, *args: Any) -> Any: ...
    def submit_command_async(self, cmd_name: str, *args: Any) -> None: ...
    def collect_command(self) -> Any: ...
    def shutdown(self) -> None: ...

    # --- Async weight sync (split submit/collect) ---
    def submit_sync_weights(self, weight_merge_map: list) -> None: ...
    def collect_sync_weights(self) -> dict: ...


@runtime_checkable
class WeightSyncEngine(Protocol):
    """Protocol for weight synchronization between trainer and rollout."""

    def initialize(self, trainer_proxy: TrainerProxy,
                   rollout_proxy: RolloutProxy,
                   master_address: str = "127.0.0.1") -> None: ...
    def sync(self, trainer_proxy: TrainerProxy,
             rollout_proxy: RolloutProxy) -> float: ...
    def verify_checksums(self, trainer_proxy: TrainerProxy,
                         rollout_proxy: RolloutProxy) -> None: ...

    @property
    def vllm_params_info(self) -> list: ...

    @property
    def trainer_weights_info(self) -> list: ...

    @property
    def weight_merge_map(self) -> list: ...


# ------------------------------------------------------------------ #
#  Queue-based stream / sink implementations                          #
# ------------------------------------------------------------------ #

class QueueSampleStream:
    """Wraps N rollout result queues into a single SampleStream.

    Round-robin polls across queues, pickle-decodes bytes, skips status
    dicts, and handles transport errors (RuntimeError/OSError).
    """

    def __init__(self, result_queues, procs):
        self._result_queues = result_queues
        self._procs = procs
        self._next_idx = 0

    def get_sample(self, timeout: float = 0.05) -> dict | None:
        """Return next data dict, or None if nothing available."""
        n = len(self._result_queues)
        if n == 0:
            return None
        per_queue_timeout = timeout / n
        for _ in range(n):
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % n
            q = self._result_queues[idx]
            try:
                raw = q.get(timeout=per_queue_timeout)
            except queue.Empty:
                continue
            except (RuntimeError, OSError):
                return None
            result = pickle.loads(raw) if isinstance(raw, bytes) else raw
            if isinstance(result, dict) and "status" in result:
                # Re-queue status messages so drain_until_status can find them.
                # Without this, the collector can race with the coordinator
                # and consume a worker's sole "paused" ack, causing a deadlock.
                q.put(raw)
                continue
            return result
        return None


class QueuePromptSink:
    """Wraps an mp.Queue for feeding prompts to rollout workers."""

    def __init__(self, prompt_queue, n_workers: int):
        self._prompt_queue = prompt_queue
        self._n_workers = n_workers

    def put(self, prompt_info: dict, timeout: float = 1.0) -> bool:
        """Put a prompt into the queue. Returns True on success, False on Full."""
        try:
            self._prompt_queue.put(prompt_info, timeout=timeout)
            return True
        except queue.Full:
            return False

    def send_sentinel(self) -> None:
        """Send one None sentinel per worker to signal shutdown."""
        for _ in range(self._n_workers):
            try:
                self._prompt_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

    @property
    def n_workers(self) -> int:
        return self._n_workers


# ------------------------------------------------------------------ #
#  Queue-based proxy implementations                                  #
# ------------------------------------------------------------------ #

class QueueRolloutProxy:
    """Wraps mp.Queue cmd/result pairs for rollout worker communication."""

    def __init__(self, cmd_queues, result_queues, procs,
                 need_student_logprobs=False, rollout_support_topk_k=0,
                 mc_n_total_samples=0, prompt_queue=None):
        self._cmd_queues = cmd_queues
        self._result_queues = result_queues
        self._procs = procs
        self._need_student_logprobs = need_student_logprobs
        self._rollout_support_topk_k = rollout_support_topk_k
        self._mc_n_total_samples = mc_n_total_samples
        self._prompt_queue = prompt_queue

    @property
    def n_workers(self) -> int:
        return len(self._cmd_queues)

    def submit_generate(self, batch_dict: dict) -> None:
        """Split batch across workers and send generate commands."""
        n = len(self._cmd_queues)
        bs = batch_dict["input_ids"].size(0)
        chunk = (bs + n - 1) // n
        for i, q in enumerate(self._cmd_queues):
            s, e = i * chunk, min((i + 1) * chunk, bs)
            sub = {k: v[s:e] if isinstance(v, torch.Tensor) else v
                   for k, v in batch_dict.items()}
            if self._need_student_logprobs:
                sub["return_logprobs"] = True
            if self._rollout_support_topk_k > 0:
                sub["response_topk_k"] = self._rollout_support_topk_k
            if self._mc_n_total_samples > 0:
                sub["mc_n_total_samples"] = self._mc_n_total_samples
            q.put(("generate", sub))

    def collect_generate(self) -> dict:
        """Wait for generate results from all workers, merge, and return."""
        results = []
        for i, q in enumerate(self._result_queues):
            while True:
                try:
                    raw = q.get(timeout=30)
                    break
                except queue.Empty:
                    if i < len(self._procs) and not self._procs[i].is_alive():
                        raise RuntimeError(
                            f"Rollout worker {i} (pid={self._procs[i].pid}) "
                            f"died with exit code {self._procs[i].exitcode} "
                            f"while waiting for generate results")
            results.append(pickle.loads(raw) if isinstance(raw, bytes) else raw)

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
                         "query_indices_response", "query_logprobs_response",
                         "mc_query_indices_response", "mc_query_old_logprobs_response"}:
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

    def submit_command(self, command: str, *args: Any) -> None:
        """Send arbitrary command to ALL workers."""
        for q in self._cmd_queues:
            q.put((command, *args))

    def _raise_if_any_worker_dead(self, purpose: str) -> None:
        for i, proc in enumerate(self._procs):
            if not proc.is_alive():
                raise RuntimeError(
                    f"Rollout worker {i} (pid={proc.pid}) died with exit code "
                    f"{proc.exitcode} while waiting for {purpose}")

    def collect_results(self, poll_timeout: float = 30.0,
                        purpose: str = "command results") -> list[dict]:
        """Get one result from each result queue, failing fast on dead workers."""
        results = []
        for i, q in enumerate(self._result_queues):
            n_timeouts = 0
            t_start = time.monotonic()
            while True:
                try:
                    raw = q.get(timeout=poll_timeout)
                    break
                except queue.Empty:
                    n_timeouts += 1
                    self._raise_if_any_worker_dead(purpose)
                    if n_timeouts % 2 == 0:
                        elapsed = time.monotonic() - t_start
                        print(f"[collect_results] waiting for worker {i} "
                              f"{purpose} ({elapsed:.0f}s, "
                              f"pid={self._procs[i].pid if i < len(self._procs) else '?'}, "
                              f"alive={self._procs[i].is_alive() if i < len(self._procs) else '?'})",
                              flush=True)
                except (RuntimeError, OSError):
                    continue  # FD passing error, retry
            results.append(pickle.loads(raw) if isinstance(raw, bytes) else raw)
        return results

    def shutdown(self) -> None:
        """Send shutdown to all workers."""
        for q in self._cmd_queues:
            try:
                q.put_nowait(("shutdown",))
            except Exception as e:
                print(f"[Shutdown] Ignoring rollout queue error: {e}", flush=True)

    # --- Streaming lifecycle ---

    def enter_autonomous(self, sub_batches: list[dict]) -> list[dict]:
        """Enter autonomous streaming mode. Sends sub_batches[i] to worker i."""
        for i, q in enumerate(self._cmd_queues):
            q.put(("enter_autonomous", sub_batches[i]))
        # Collect acks from all workers
        return self.drain_until_status("autonomous_started")

    def pause_workers(self) -> list[dict]:
        """Pause all workers and drain until all report paused."""
        for q in self._cmd_queues:
            q.put(("pause",))
        return self.drain_until_status("paused")

    def resume_workers(self, seed_batches: list[dict] | None = None) -> list[dict]:
        """Resume all workers, optionally with per-worker seed batches."""
        for i, q in enumerate(self._cmd_queues):
            seed = seed_batches[i] if seed_batches else None
            q.put(("resume", seed))
        return self.drain_until_status("resumed")

    def exit_autonomous(self) -> list[dict]:
        """Exit autonomous mode for all workers."""
        for q in self._cmd_queues:
            q.put(("exit_autonomous",))
        return self.drain_until_status("exited_autonomous")

    def drain_until_status(self, target_status: str,
                           forward_target: ForwardTarget | None = None) -> list[dict]:
        """Drain result queues until target_status seen from all workers.

        Interleaved data samples (dicts without 'status' key) are forwarded
        to forward_target if provided. This is a 1:1 port of
        StreamCoordinator._drain_result_queues().

        Returns list of status dicts (one per worker).
        """
        status_msgs = []
        t_drain_start = time.monotonic()
        for i, rq in enumerate(self._result_queues):
            t_worker_start = time.monotonic()
            n_samples = 0
            n_timeouts = 0
            while True:
                try:
                    raw = rq.get(timeout=30)
                except queue.Empty:
                    n_timeouts += 1
                    elapsed = time.monotonic() - t_worker_start
                    # Check if rollout worker is still alive
                    if i < len(self._procs) and not self._procs[i].is_alive():
                        print(f"\n{'='*60}", flush=True)
                        print(f"[Coordinator] Rollout worker {i} "
                              f"(pid={self._procs[i].pid}) died "
                              f"(exit={self._procs[i].exitcode}) "
                              f"while draining for '{target_status}'", flush=True)
                        print(f"{'='*60}\n", flush=True)
                        status_msgs.append({})
                        break
                    if n_timeouts % 2 == 0:  # log every 60s
                        print(f"[drain] waiting for worker {i} '{target_status}' "
                              f"({elapsed:.0f}s, {n_samples} samples drained, "
                              f"pid={self._procs[i].pid if i < len(self._procs) else '?'}, "
                              f"alive={self._procs[i].is_alive() if i < len(self._procs) else '?'})",
                              flush=True)
                    continue
                except (RuntimeError, OSError):
                    continue  # FD passing error, retry
                r = pickle.loads(raw) if isinstance(raw, bytes) else raw
                if isinstance(r, dict) and r.get("status") == target_status:
                    status_msgs.append(r)
                    break
                if isinstance(r, dict) and "status" in r and r["status"] != target_status:
                    print(f"[drain] worker {i}: unexpected status '{r['status']}' "
                          f"(wanted '{target_status}')", flush=True)
                if isinstance(r, dict) and "status" not in r and forward_target is not None:
                    n_samples += 1
                    wv = r.get("weight_version", 0)
                    forward_target.put(r, weight_version=wv)
        return status_msgs

    # --- Per-worker commands ---

    def submit_command_per_worker(self, command: str, per_worker_args: list[tuple]) -> None:
        """Send command with different args to each worker."""
        for i, q in enumerate(self._cmd_queues):
            q.put((command, *per_worker_args[i]))

    # --- Stage interface factories ---

    def sample_stream(self) -> QueueSampleStream:
        """Return a SampleStream wrapping this proxy's result queues."""
        return QueueSampleStream(self._result_queues, self._procs)

    def prompt_sink(self) -> QueuePromptSink:
        """Return a PromptSink wrapping this proxy's prompt queue."""
        return QueuePromptSink(self._prompt_queue, self.n_workers)


class QueueTrainerProxy:
    """Wraps single trainer cmd/result queue pair."""

    def __init__(self, cmd_queue, result_queue, proc, fsdp_procs=None):
        self._cmd_queue = cmd_queue
        self._result_queue = result_queue
        self._proc = proc
        self._fsdp_procs = fsdp_procs or []
        self._checkpoint_save_pending = False

    @property
    def proc(self):
        return self._proc

    @property
    def fsdp_procs(self):
        return self._fsdp_procs

    @staticmethod
    def _build_train_batch(gen_output: dict, teacher_output: dict) -> dict:
        from opd.data.opd_payload import build_opd_train_batch

        return build_opd_train_batch(gen_output, teacher_output)

    def submit_train(self, gen_output: dict, teacher_output: dict) -> None:
        """Send training batch to trainer subprocess (non-blocking)."""
        batch = self._build_train_batch(gen_output, teacher_output)
        batch["_send_mono"] = time.monotonic()
        self._cmd_queue.put(("train", batch))

    def submit_train_direct_teacher_artifacts(
        self,
        gen_output: dict,
        *,
        teacher_buffer_id: int,
        logical_batch_id: int,
        gen_weight_version: int,
        expected_samples: int,
        timeout_s: float = 300.0,
    ) -> None:
        """Send a train command that references trainer-side teacher artifacts."""
        batch = {
            "input_ids": gen_output["input_ids"],
            "attention_mask": gen_output["attention_mask"],
            "responses": gen_output["responses"],
            "prompt_lengths": gen_output["prompt_lengths"],
            "_direct_teacher_artifacts": True,
            "teacher_buffer_id": int(teacher_buffer_id),
            "logical_batch_id": int(logical_batch_id),
            "gen_weight_version": int(gen_weight_version),
            "expected_samples": int(expected_samples),
            "teacher_artifact_timeout_s": float(timeout_s),
        }
        for key in (
            "response_lengths", "student_logprobs", "sample_seq_ids",
            "query_indices_response", "query_logprobs_response",
            "mc_query_indices_response", "mc_query_old_logprobs_response",
        ):
            if key in gen_output:
                batch[key] = gen_output[key]
        batch["_send_mono"] = time.monotonic()
        self._cmd_queue.put(("train", batch))

    def collect_train(self) -> dict:
        """Wait for trainer subprocess to finish and return result.

        IMPORTANT: caller must drain any pending checkpoint save FIRST
        (via collect_checkpoint_save), because the save result arrives
        in the queue before the train result.
        """
        result = self._result_queue.get()
        return result

    def submit_save_checkpoint(self, step: int, checkpoint_dir: str,
                               save_optimizer: bool = True) -> None:
        """Send checkpoint save command (non-blocking).

        IMPORTANT: caller must drain any pending checkpoint save FIRST
        (via collect_checkpoint_save), because results queue is FIFO.
        """
        assert not self._checkpoint_save_pending, \
            "Previous checkpoint save not drained before dispatching new one"
        self._cmd_queue.put(("save_checkpoint", {
            "step": step,
            "checkpoint_dir": checkpoint_dir,
            "save_optimizer": save_optimizer,
        }))
        self._checkpoint_save_pending = True

    def collect_checkpoint_save(self) -> dict | None:
        """Drain pending checkpoint save result, if any. Returns result or None."""
        if self._checkpoint_save_pending:
            result = self._result_queue.get()
            self._checkpoint_save_pending = False
            return result
        return None

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """Load checkpoint and return step number.

        Does NOT sync weights -- the coordinator handles that separately.
        IMPORTANT: caller must drain any pending checkpoint save FIRST.
        """
        self._cmd_queue.put(("load_checkpoint", {
            "checkpoint_dir": checkpoint_dir,
        }))
        result = self._result_queue.get()
        return result.get("step", 0)

    def submit_command(self, cmd_name: str, *args: Any) -> Any:
        """Send arbitrary command and return result."""
        self._cmd_queue.put((cmd_name, *args))
        return self._result_queue.get()

    def submit_command_async(self, cmd_name: str, *args: Any) -> None:
        """Send command without waiting for result. Pair with collect_command()."""
        self._cmd_queue.put((cmd_name, *args))

    def collect_command(self) -> Any:
        """Collect result from a prior submit_command_async call."""
        return self._result_queue.get()

    def shutdown(self) -> None:
        """Send shutdown command."""
        try:
            self._cmd_queue.put_nowait(("shutdown",))
        except Exception as e:
            print(f"[Shutdown] Ignoring trainer queue error: {e}", flush=True)

    # --- Async weight sync (split submit/collect) ---

    def submit_sync_weights(self, weight_merge_map: list) -> None:
        """Send sync_weights command without waiting for result."""
        self._cmd_queue.put(("sync_weights", weight_merge_map))

    def collect_sync_weights(self) -> dict:
        """Wait for sync_weights result from trainer."""
        return self._result_queue.get()


# ------------------------------------------------------------------ #
#  CPU Weight Sync Engine (for HF rollout backend)                    #
# ------------------------------------------------------------------ #

class CPUWeightSyncEngine:
    """Lightweight weight sync via mp.Queue (state_dict serialization).

    Used by HF rollout backend. No NCCL, no vLLM weight transfer engine.
    """

    def __init__(self, verify_checksum=False):
        self._verify_checksum = verify_checksum

    @staticmethod
    def _compute_checksum_from_state_dict(state_dict):
        phi = 1.6180339887
        checksum = 0.0
        for i, (_, param) in enumerate(sorted(state_dict.items())):
            checksum += param.detach().cpu().float().abs().sum().item() * (phi ** (i % 32))
        return checksum

    def initialize(self, trainer_proxy, rollout_proxy, **_kwargs):
        pass  # No NCCL init needed

    def sync(self, trainer_proxy, rollout_proxy) -> float:
        """Get state_dict from trainer, send to rollout. Returns sync_seconds."""
        # submit_command on QueueTrainerProxy is synchronous (sends + returns)
        result = trainer_proxy.submit_command("get_clean_state_dict")
        state_dict = result["state_dict"]
        t0 = time.time()
        rollout_proxy.submit_command("sync_weights", state_dict)
        rollout_proxy.collect_results()
        dt = time.time() - t0
        if self._verify_checksum:
            expected_checksum = self._compute_checksum_from_state_dict(state_dict)
            self.verify_checksums(trainer_proxy, rollout_proxy, expected_checksum=expected_checksum)
        return dt

    def verify_checksums(self, trainer_proxy, rollout_proxy, expected_checksum=None):
        """Compare trainer and rollout weight checksums (sorted-name order)."""
        if expected_checksum is None:
            # Fallback path if sync() didn't supply the exact payload checksum.
            t_result = trainer_proxy.submit_command("compute_weight_checksum", None)
            t_chk = t_result["checksum"]
        else:
            t_chk = expected_checksum
        rollout_proxy.submit_command("compute_weight_checksum")
        r_results = rollout_proxy.collect_results()
        for r in r_results:
            r_chk = r["checksum"]
            if abs(t_chk - r_chk) > 1e-4:
                print(f"[CPUWeightSync] Weight checksum mismatch: "
                      f"trainer={t_chk:.6f} rollout={r_chk:.6f}", flush=True)
            else:
                print(f"[CPUWeightSync] Weight checksum OK: {t_chk:.6f}",
                      flush=True)

    def sync_paused(self, trainer_proxy, rollout_proxy, **_kwargs):
        return self.sync(trainer_proxy, rollout_proxy)

    @property
    def vllm_params_info(self):
        return None

    @property
    def trainer_weights_info(self):
        return None

    @property
    def weight_merge_map(self):
        return None


# ------------------------------------------------------------------ #
#  NCCL Weight Sync Engine                                            #
# ------------------------------------------------------------------ #

class NCCLWeightSyncEngine:
    """Wraps vLLM's native NCCL weight transfer engine.

    Creates a shared NCCL group between the trainer (rank 0) and
    vLLM EngineCore workers (rank 1+). Weights are broadcast directly
    on GPU -- no CPU copy or pickle overhead.
    """

    def __init__(self, verify_checksum: bool = False):
        self._verify_checksum = verify_checksum
        self._vllm_params_info: list | None = None
        self._trainer_weights_info: list | None = None
        self._weight_merge_map: list | None = None
        self._tp_size: int = 1
        self._lora_mode: bool = False
        self._peft_config: dict | None = None

    @property
    def vllm_params_info(self) -> list:
        return self._vllm_params_info

    @property
    def trainer_weights_info(self) -> list:
        return self._trainer_weights_info

    @property
    def weight_merge_map(self) -> list:
        return self._weight_merge_map

    def initialize(self, trainer_proxy: TrainerProxy,
                   rollout_proxy: RolloutProxy,
                   master_address: str = "127.0.0.1",
                   tp_size: int = 1,
                   lora_mode: bool = False,
                   peft_config: dict = None) -> None:
        """Initialize vLLM's native NCCL weight transfer engine.

        For TP=1: is_checkpoint_format=False, trainer pre-merges q/k/v → qkv_proj.
        For TP>1: is_checkpoint_format=True, trainer sends raw HF keys, vLLM's
                  load_weights handles merging + TP sharding internally.
                  world_size accounts for all TP ranks.
        For lora_mode: LoRA tensors transferred via queue (small ~10MB), not NCCL.
        """
        self._tp_size = tp_size
        self._lora_mode = lora_mode
        self._peft_config = peft_config

        n_rollout_workers = rollout_proxy.n_workers
        wt_world_size = 1 + n_rollout_workers * tp_size

        wt_port = find_free_port("weight_sync.nccl_master")

        init_info = {
            "master_address": master_address,
            "master_port": wt_port,
            "rank_offset": 1,
            "world_size": wt_world_size,
        }

        # Send init to trainer (rank 0) and all rollout workers SIMULTANEOUSLY.
        # NCCL init is a collective — all ranks must join together.
        # With TP>1, each rollout worker dispatches init to all its TP ranks
        # via collective_rpc. rank_offset = 1 + i * tp_size gives consecutive IDs.
        trainer_proxy.submit_command_async("init_weight_transfer", init_info)
        rollout_proxy.submit_command_per_worker(
            "init_weight_transfer",
            [(dict(init_info, rank_offset=1 + i * tp_size),) for i in range(n_rollout_workers)],
        )
        trainer_proxy.collect_command()
        rollout_proxy.collect_results()

        # Query vLLM for its internal parameter names/shapes/dtypes
        rollout_proxy.submit_command("get_vllm_params_info")
        results = rollout_proxy.collect_results()
        self._vllm_params_info = results[0]["params_info"]

        # Get trainer state_dict info
        trainer_info = trainer_proxy.submit_command("get_weights_info")
        self._trainer_weights_info = trainer_info["weights_info"]

        # Build merge mapping
        self._weight_merge_map = build_weight_merge_map(
            self._trainer_weights_info, self._vllm_params_info
        )

        print(f"[Pipeline] vLLM weight transfer ready "
              f"({len(self._vllm_params_info)} vLLM params, "
              f"{len(self._trainer_weights_info)} trainer params, "
              f"world={wt_world_size})", flush=True)

    def _build_update_info(self):
        """Build the update_info dict describing what trainer sends (bf16 tensors)."""
        vllm_lookup = {n: (s, d) for n, s, d in self._vllm_params_info}
        trainer_lookup = {n: (s, d) for n, s, d in self._trainer_weights_info}
        update_names, update_shapes, update_dtypes = [], [], []
        for vllm_name, sources in self._weight_merge_map:
            dtype = trainer_lookup[sources[0]][1] if sources[0] in trainer_lookup else vllm_lookup[vllm_name][1]
            shape = vllm_lookup[vllm_name][0] if vllm_name in vllm_lookup else trainer_lookup[sources[0]][0]
            update_names.append(vllm_name)
            update_shapes.append(list(shape))
            update_dtypes.append(str(dtype).replace("torch.", ""))
        return {
            "names": update_names,
            "dtype_names": update_dtypes,
            "shapes": update_shapes,
            "packed": False,
            "is_checkpoint_format": self._tp_size > 1,  # TP>1 needs load_weights for sharding; TP=1 uses direct copy
        }

    def sync(self, trainer_proxy: TrainerProxy,
             rollout_proxy: RolloutProxy) -> float:
        """Perform weight sync. Returns sync_seconds.

        IMPORTANT: caller must drain any pending checkpoint save BEFORE
        calling this (the coordinator does this via _wait_checkpoint_save).
        """
        if self._lora_mode:
            return self._sync_lora(trainer_proxy, rollout_proxy)
        if self._tp_size > 1:
            return self._sync_tp(trainer_proxy, rollout_proxy)

        update_info = self._build_update_info()

        # Tell trainer to send merged weights (matching vLLM param order)
        trainer_proxy.submit_sync_weights(self._weight_merge_map)
        rollout_proxy.submit_command("sync_weights", update_info)

        # Wait for all to complete
        trainer_res = trainer_proxy.collect_sync_weights()
        rollout_proxy.collect_results()

        # Optional checksum verification
        if self._verify_checksum:
            self.verify_checksums(trainer_proxy, rollout_proxy)

        return trainer_res.get("sync_seconds", 0)

    def _sync_tp(self, trainer_proxy: TrainerProxy,
                  rollout_proxy: RolloutProxy) -> float:
        """Weight sync for TP>1 using vLLM's NCCL engine with is_checkpoint_format=True.

        With is_checkpoint_format=True, vLLM's receive_weights calls model.load_weights()
        which handles TP sharding internally. The trainer sends raw HF-format weights
        (no pre-merging of q/k/v → qkv_proj). All ranks participate in the NCCL
        broadcast and each rank loads its own shard.
        """
        update_info = self._build_update_info_tp()

        # Trainer sends raw (unmerged) state dict — pass None for merge_map
        # so trainer sends individual params instead of merged qkv/gate_up
        trainer_proxy.submit_sync_weights(None)
        rollout_proxy.submit_command("sync_weights", update_info)

        trainer_res = trainer_proxy.collect_sync_weights()
        rollout_proxy.collect_results()

        if self._verify_checksum:
            self.verify_checksums(trainer_proxy, rollout_proxy)

        return trainer_res.get("sync_seconds", 0)

    def _sync_lora(self, trainer_proxy: TrainerProxy,
                   rollout_proxy: RolloutProxy) -> float:
        """Weight sync for native LoRA — transfer LoRA A/B matrices via NCCL.

        Uses the same NCCL weight transfer engine as full model sync.
        The update_info includes '_lora_update': True which the monkey-patched
        Worker.update_weights detects and routes to add_lora instead of
        loading into base model params.
        """
        update_info = self._build_update_info_lora()

        # Trainer sends LoRA tensors via NCCL (raw iter, no merge map)
        trainer_proxy.submit_sync_weights(None)
        rollout_proxy.submit_command("sync_weights", update_info)

        trainer_res = trainer_proxy.collect_sync_weights()
        rollout_proxy.collect_results()

        if self._verify_checksum:
            self._verify_lora_checksums(trainer_proxy, rollout_proxy)

        return trainer_res.get("sync_seconds", 0)

    def _verify_lora_checksums(self, trainer_proxy: TrainerProxy,
                               rollout_proxy: RolloutProxy) -> None:
        """Compare LoRA weight checksums between trainer and rollout."""
        trainer_cksum = trainer_proxy.submit_command(
            "compute_lora_checksum")["checksum"]
        rollout_proxy.submit_command("compute_lora_checksum")
        rollout_cksums = [r["checksum"] for r in rollout_proxy.collect_results()]

        for i, rc in enumerate(rollout_cksums):
            rel_err = abs(trainer_cksum - rc) / max(abs(trainer_cksum), 1e-12)
            if rel_err > 1e-6:
                print(f"[Pipeline] WARNING: LoRA checksum mismatch! "
                      f"trainer={trainer_cksum:.6f} rollout-{i}={rc:.6f} "
                      f"rel_err={rel_err:.2e}", flush=True)
            else:
                print(f"[Pipeline] LoRA checksum OK "
                      f"(trainer={trainer_cksum:.2f}, rollout-{i}={rc:.2f}, "
                      f"rel_err={rel_err:.2e})", flush=True)

    def _build_update_info_lora(self):
        """Build update_info for LoRA sync with _lora_update flag."""
        update_names, update_shapes, update_dtypes = [], [], []
        for name, shape, dtype in self._trainer_weights_info:
            update_names.append(name)
            update_shapes.append(list(shape))
            update_dtypes.append(str(dtype).replace("torch.", ""))
        return {
            "names": update_names,
            "dtype_names": update_dtypes,
            "shapes": update_shapes,
            "packed": False,
            "is_checkpoint_format": False,
            "_lora_update": True,
        }

    def _build_update_info_tp(self):
        """Build update_info for TP>1 (direct param copy with fused names).

        Uses is_checkpoint_format=False for direct param copy — avoids
        vLLM's load_weights() stacked_params_mapping overhead.  Trainer
        sends fused names (qkv_proj, gate_up_proj) that match vLLM's
        model.named_parameters() exactly.
        """
        update_names, update_shapes, update_dtypes = [], [], []
        for name, shape, dtype in self._trainer_weights_info:
            update_names.append(name)
            update_shapes.append(list(shape))
            update_dtypes.append(str(dtype).replace("torch.", ""))
        return {
            "names": update_names,
            "dtype_names": update_dtypes,
            "shapes": update_shapes,
            "packed": False,
            "is_checkpoint_format": False,
        }

    def sync_paused(self, trainer_proxy: TrainerProxy,
                    rollout_proxy: RolloutProxy,
                    forward_target: ForwardTarget | None = None) -> float:
        """Weight sync while rollout workers are paused (streaming path).

        Unlike sync(), rollout workers are in the pause dispatch loop and
        result queues may contain interleaved data + status messages.
        Uses drain_until_status to handle this correctly.

        Returns sync_seconds.
        """
        if self._lora_mode:
            # LoRA sync uses queue, not NCCL — same path as sync()
            return self._sync_lora(trainer_proxy, rollout_proxy)
        if self._tp_size > 1:
            update_info = self._build_update_info_tp()
            merge_map = None  # TP>1: raw HF keys, no pre-merge
        else:
            update_info = self._build_update_info()
            merge_map = self._weight_merge_map

        # Send sync commands (non-blocking)
        trainer_proxy.submit_sync_weights(merge_map)
        rollout_proxy.submit_command("sync_weights", update_info)

        # Wait for trainer (simple get -- trainer has no interleaved data)
        trainer_res = trainer_proxy.collect_sync_weights()

        # Drain rollout results until all report synced_nccl
        rollout_proxy.drain_until_status("synced_nccl", forward_target=forward_target)

        # Optional checksum verification
        if self._verify_checksum:
            self.verify_checksums(trainer_proxy, rollout_proxy)

        return trainer_res.get("sync_seconds", 0)

    def verify_checksums(self, trainer_proxy: TrainerProxy,
                         rollout_proxy: RolloutProxy) -> None:
        """Compare weight checksums between trainer and all rollout workers."""
        trainer_cksum = trainer_proxy.submit_command(
            "compute_weight_checksum", self._weight_merge_map)["checksum"]
        rollout_proxy.submit_command("compute_weight_checksum", self._weight_merge_map)
        rollout_cksums = [r["checksum"] for r in rollout_proxy.collect_results()]

        for i, rc in enumerate(rollout_cksums):
            rel_err = abs(trainer_cksum - rc) / max(abs(trainer_cksum), 1e-12)
            if rel_err > 1e-6:
                print(f"[Pipeline] WARNING: Weight checksum mismatch! "
                      f"trainer={trainer_cksum:.6f} rollout-{i}={rc:.6f} "
                      f"rel_err={rel_err:.2e}", flush=True)
            else:
                print(f"[Pipeline] Weight checksum OK "
                      f"(trainer={trainer_cksum:.2f}, rollout-{i}={rc:.2f}, "
                      f"rel_err={rel_err:.2e})", flush=True)

"""Pytest-only CPU stubs for exercising the real coordinator pipeline.

These helpers intentionally live under ``tests/`` rather than the ``opd`` package.
They provide lightweight process entry points that can be monkeypatched into the
normal coordinator lifecycle, so tests reuse the production coordinator,
queue/proxy, OPD payload assembly, batch preparation, weight-sync, and KL-loss
paths without starting HF/vLLM/FSDP/Megatron runtimes.
"""

from __future__ import annotations

import json
import os
import queue
import re
import time
from pathlib import Path
from typing import Any

import torch

from opd.launch_specs import RolloutLaunchSpec, TeacherLaunchSpec, TrainerLaunchSpec
from opd.trainer.base import BaseBackend
from opd.trainer.config import build_kl_config_from_algorithm_payload
from opd.trainer.opd import OPDTrainer
from opd.worker.teacher.serialization import deserialize, serialize

VOCAB_SIZE = 257
MIN_TOKEN_ID = 2


def _as_config(config: Any) -> dict:
    if hasattr(config, "merged_config"):
        return config.merged_config()
    return dict(config)


def _event_path() -> str | None:
    return os.environ.get("OPD_CPU_STUB_EVENT_LOG")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _record(role: str, event: str, **fields: Any) -> None:
    path = _event_path()
    if not path:
        return
    payload = {
        "time": time.time(),
        "pid": os.getpid(),
        "role": role,
        "event": event,
        **{k: _jsonable(v) for k, v in fields.items()},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _topk_indices_for_row(token_id: int, local_pos: int, k: int) -> torch.Tensor:
    """Return a deterministic candidate set for one sequence row."""
    k = max(1, min(int(k), VOCAB_SIZE - MIN_TOKEN_ID))
    base = int(token_id) * 17 + int(local_pos) * 31 + 13
    values: list[int] = []
    probe = 0
    while len(values) < k:
        candidate = MIN_TOKEN_ID + ((base + probe * 19) % (VOCAB_SIZE - MIN_TOKEN_ID))
        if candidate not in values:
            values.append(candidate)
        probe += 1
    return torch.tensor(values, dtype=torch.int32)


def _logps_for_indices(
    indices: torch.Tensor,
    *,
    token_id: int,
    local_pos: int,
) -> torch.Tensor:
    """Deterministic pseudo-distribution over a supplied candidate set."""
    idx = indices.to(torch.float32)
    phase = idx * 0.017 + float(local_pos) * 0.113 + float(int(token_id) % 97) * 0.031
    logits = torch.sin(phase) + 0.5 * torch.cos(phase * 1.7)
    return torch.log_softmax(logits, dim=-1).to(torch.float32)


def _score_rows(
    token_ids: list[int],
    *,
    topk: int,
    shift: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not token_ids:
        empty_idx = torch.zeros(0, max(int(topk), 1), dtype=torch.int32)
        empty_lp = torch.zeros(0, max(int(topk), 1), dtype=torch.float32)
        return empty_lp, empty_idx

    logps: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    n = len(token_ids)
    for pos, token_id in enumerate(token_ids):
        source_pos = min(pos + 1, n - 1) if shift else pos
        source_token = token_ids[source_pos]
        idx = _topk_indices_for_row(source_token, source_pos, topk)
        lp = _logps_for_indices(idx, token_id=source_token, local_pos=source_pos)
        indices.append(idx)
        logps.append(lp)
    return torch.stack(logps, dim=0), torch.stack(indices, dim=0)


def _student_topk_logps(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    teacher_indices: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute student logprobs aligned to teacher_indices for all non-pad rows."""
    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu().bool()
    teacher_indices = teacher_indices.cpu()
    if position_ids is not None:
        position_ids = position_ids.cpu().long()
    out = torch.zeros_like(teacher_indices, dtype=torch.float32)
    local_positions = (
        position_ids if position_ids is not None
        else attention_mask.long().cumsum(dim=1) - 1
    )
    batch, seq_len = input_ids.shape
    for b in range(batch):
        for s in range(seq_len):
            if not bool(attention_mask[b, s]):
                continue
            idx = teacher_indices[b, s].to(torch.int32)
            out[b, s] = _logps_for_indices(
                idx,
                token_id=int(input_ids[b, s].item()),
                local_pos=int(local_positions[b, s].item()),
            )
    return out


def _checksum(state_dict: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for i, (_, tensor) in enumerate(sorted(state_dict.items())):
        total += tensor.detach().cpu().float().abs().sum().item() * (1.6180339887 ** (i % 8))
    return float(total)


def _rollout_response_tokens(prompt_tokens: list[int], *, group_idx: int,
                             max_response_length: int) -> list[int]:
    seed = sum(int(t) for t in prompt_tokens) + 23 * (group_idx + 1)
    return [
        MIN_TOKEN_ID + ((seed + group_idx * 11 + j * 3) % (VOCAB_SIZE - MIN_TOKEN_ID))
        for j in range(max_response_length)
    ]


def _build_rollout_result(
    batch: dict,
    *,
    max_response_length: int,
    worker_id: int,
    weight_version: int = 0,
) -> dict:
    t0 = time.monotonic()
    prompt_ids = batch["input_ids"].cpu().long()
    prompt_mask = batch["attention_mask"].cpu().long()
    group_size = int(batch.get("grpo_n_samples", 1) or 1)
    return_logprobs = bool(batch.get("return_logprobs", False))
    base_bs, prompt_width = prompt_ids.shape
    out_bs = base_bs * group_size
    total_len = prompt_width + max_response_length

    full_ids = torch.zeros(out_bs, total_len, dtype=torch.long)
    full_mask = torch.zeros(out_bs, total_len, dtype=torch.long)
    responses = torch.zeros(out_bs, max_response_length, dtype=torch.long)
    prompt_lengths = torch.zeros(out_bs, dtype=torch.long)
    response_lengths = torch.full((out_bs,), max_response_length, dtype=torch.long)
    full_token_lists: list[list[int]] = []
    student_logprobs = torch.zeros(out_bs, max_response_length, dtype=torch.float32)

    out_idx = 0
    for b in range(base_bs):
        valid_prompt = prompt_ids[b][prompt_mask[b].bool()].tolist()
        for g in range(group_size):
            resp = _rollout_response_tokens(
                valid_prompt,
                group_idx=g,
                max_response_length=max_response_length,
            )
            full_ids[out_idx, :prompt_width] = prompt_ids[b]
            full_mask[out_idx, :prompt_width] = prompt_mask[b]
            full_ids[out_idx, prompt_width:] = torch.tensor(resp, dtype=torch.long)
            full_mask[out_idx, prompt_width:] = 1
            responses[out_idx] = torch.tensor(resp, dtype=torch.long)
            prompt_lengths[out_idx] = int(prompt_mask[b].sum().item())
            full_token_lists.append([int(x) for x in valid_prompt] + resp)
            for j, token_id in enumerate(resp):
                local_pos = int(prompt_lengths[out_idx].item()) + j
                idx = _topk_indices_for_row(token_id, local_pos, 1)
                student_logprobs[out_idx, j] = _logps_for_indices(
                    idx, token_id=token_id, local_pos=local_pos
                )[0]
            out_idx += 1

    t1 = time.monotonic()
    result = {
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "responses": responses,
        "prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths,
        "full_token_lists": full_token_lists,
        "timing": {
            "worker_id": worker_id,
            "mono_start": t0,
            "mono_end": t1,
            "generate_seconds": t1 - t0,
            "elapsed": t1 - t0,
        },
        "weight_version": weight_version,
        "_vllm_stats": [],
    }
    if return_logprobs:
        result["student_logprobs"] = student_logprobs
    return result


def _batch_from_prompt_info(prompt_info: dict) -> dict:
    input_row = prompt_info["input_ids_row"].cpu().long()
    attention_row = (input_row != 0).long()
    return {
        "input_ids": input_row.unsqueeze(0),
        "attention_mask": attention_row.unsqueeze(0),
        "return_logprobs": bool(prompt_info.get("return_logprobs", False)),
    }


def _sample_from_result(
    result: dict,
    idx: int,
    *,
    worker_id: int,
    weight_version: int,
    prompt_info: dict | None = None,
) -> dict:
    sample: dict[str, Any] = {}
    for key in (
        "input_ids",
        "attention_mask",
        "responses",
        "prompt_lengths",
        "response_lengths",
        "student_logprobs",
    ):
        if key in result:
            sample[key] = result[key][idx:idx + 1]
    sample["full_token_lists"] = [result["full_token_lists"][idx]]
    sample["weight_version"] = weight_version
    sample["worker_id"] = worker_id
    sample["sample_seq_id"] = f"{worker_id}-{time.monotonic_ns()}-{idx}"
    if prompt_info is not None:
        for key in ("ground_truth", "prompt_group_id"):
            if key in prompt_info:
                sample[key] = prompt_info[key]
    return sample


def cpu_stub_rollout_worker_main(config, cmd_queue, result_queue):
    """Rollout worker entry point used only by tests."""
    cfg = _as_config(config)
    static = config.static if isinstance(config, RolloutLaunchSpec) else None
    runtime = getattr(config, "runtime", None)
    worker_id = getattr(getattr(config, "runtime", None), "worker_id", 0)
    prompt_queue = getattr(runtime, "prompt_queue", None)
    max_response_length = int(
        getattr(static, "max_response_length", cfg.get("max_response_length", 1))
    )
    stored_state: dict[str, torch.Tensor] = {"weight": torch.ones(1, 1)}
    weight_version = 0
    autonomous = False
    paused = False
    _record("rollout", "started", worker_id=worker_id)

    def emit_batch(batch: dict, *, source: str) -> None:
        result = _build_rollout_result(
            batch,
            max_response_length=max_response_length,
            worker_id=worker_id,
            weight_version=weight_version,
        )
        if source == "stream":
            for i in range(result["input_ids"].size(0)):
                result_queue.put(_sample_from_result(
                    result, i, worker_id=worker_id,
                    weight_version=weight_version))
        else:
            result_queue.put(result)
        _record("rollout", "generate", worker_id=worker_id,
                prompts=int(batch["input_ids"].size(0)),
                samples=int(result["input_ids"].size(0)),
                return_logprobs=bool(batch.get("return_logprobs", False)),
                source=source, weight_version=weight_version)

    def emit_prompt(prompt_info: dict) -> None:
        batch = _batch_from_prompt_info(prompt_info)
        result = _build_rollout_result(
            batch,
            max_response_length=max_response_length,
            worker_id=worker_id,
            weight_version=weight_version,
        )
        for i in range(result["input_ids"].size(0)):
            result_queue.put(_sample_from_result(
                result, i, worker_id=worker_id,
                weight_version=weight_version,
                prompt_info=prompt_info))
        _record("rollout", "generate", worker_id=worker_id,
                prompts=1, samples=int(result["input_ids"].size(0)),
                return_logprobs=bool(prompt_info.get("return_logprobs", False)),
                source="prompt_queue", weight_version=weight_version)

    while True:
        if autonomous:
            try:
                cmd = cmd_queue.get(timeout=0.01 if not paused else 0.5)
            except queue.Empty:
                if paused or prompt_queue is None:
                    continue
                try:
                    prompt_info = prompt_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if prompt_info is None:
                    continue
                emit_prompt(prompt_info)
                continue
        else:
            cmd = cmd_queue.get()
        name = cmd[0]
        if name == "shutdown":
            _record("rollout", "shutdown", worker_id=worker_id)
            break
        if name == "enter_autonomous":
            autonomous = True
            paused = False
            result_queue.put({"status": "autonomous_started", "_vllm_stats": []})
            _record("rollout", "autonomous_started", worker_id=worker_id)
            if len(cmd) > 1 and cmd[1] is not None:
                emit_batch(cmd[1], source="stream")
            continue
        if name == "pause":
            paused = True
            result_queue.put({
                "status": "paused",
                "n_cancelled": 0,
                "_vllm_stats": [],
            })
            _record("rollout", "paused", worker_id=worker_id)
            continue
        if name == "resume":
            paused = False
            result_queue.put({"status": "resumed", "_vllm_stats": []})
            _record("rollout", "resumed", worker_id=worker_id)
            if len(cmd) > 1 and cmd[1] is not None:
                emit_batch(cmd[1], source="stream")
            continue
        if name == "exit_autonomous":
            autonomous = False
            paused = False
            result_queue.put({
                "status": "exited_autonomous",
                "n_cancelled": 0,
                "_vllm_stats": [],
            })
            _record("rollout", "exited_autonomous", worker_id=worker_id)
            continue
        if name == "get_vllm_params_info":
            result_queue.put({"params_info": [("weight", (1, 1), torch.float32)]})
            _record("rollout", "get_vllm_params_info", worker_id=worker_id)
            continue
        if name == "sync_weights":
            state_dict = cmd[1] if len(cmd) > 1 and isinstance(cmd[1], dict) else {}
            stored_state = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
            weight_version += 1
            result_queue.put({"status": "synced_cpu", "sync_seconds": 0.0})
            _record("rollout", "sync_weights", worker_id=worker_id,
                    n_tensors=len(stored_state), checksum=_checksum(stored_state),
                    weight_version=weight_version)
            continue
        if name == "compute_weight_checksum":
            result_queue.put({"checksum": _checksum(stored_state)})
            _record("rollout", "compute_weight_checksum", worker_id=worker_id)
            continue
        if name == "init_weight_transfer":
            result_queue.put({"status": "ok"})
            _record("rollout", "init_weight_transfer", worker_id=worker_id)
            continue
        if name != "generate":
            result_queue.put({"status": "error", "reason": f"unknown command: {name}"})
            continue

        emit_batch(cmd[1], source="direct")


def cpu_stub_teacher_server_main(config):
    """ZMQ teacher server entry point used only by tests."""
    import zmq

    static = config.static if isinstance(config, TeacherLaunchSpec) else None
    runtime = config.runtime if isinstance(config, TeacherLaunchSpec) else None
    topk = int(getattr(static, "n_logprobs", 8) or 8)
    bind_address = getattr(runtime, "bind_address", "127.0.0.1")
    bind_port = int(getattr(runtime, "bind_port"))
    shift = os.environ.get("OPD_CPU_STUB_SHIFT_TEACHER", "").strip().lower() in {
        "1", "true", "yes", "on"
    }

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://{bind_address}:{bind_port}")
    _record("teacher", "started", port=bind_port, topk=topk, shift=shift)

    try:
        while True:
            raw = sock.recv()
            t0 = time.monotonic()
            try:
                payload = deserialize(raw)
                token_lists = (
                    payload.get("prompt_token_ids", [])
                    if isinstance(payload, dict) else payload
                )
                all_logps = []
                all_indices = []
                all_token_logps = []
                for token_ids in token_lists:
                    logps, indices = _score_rows(
                        [int(t) for t in token_ids],
                        topk=topk,
                        shift=shift,
                    )
                    all_logps.append(logps)
                    all_indices.append(indices)
                    all_token_logps.append(logps[:, 0].clone())
                t1 = time.monotonic()
                response = {
                    "responses": ["" for _ in token_lists],
                    "teacher_topk_logprobs": all_logps,
                    "teacher_topk_indices": all_indices,
                    "teacher_token_logprobs": all_token_logps,
                    "timing": {"mono_start": t0, "mono_end": t1},
                }
                sock.send(serialize(response))
                _record("teacher", "score", prompts=len(token_lists), shift=shift)
            except Exception as exc:  # pragma: no cover - propagated to parent test
                sock.send(serialize({"status": "error", "reason": repr(exc)}))
                _record("teacher", "error", reason=repr(exc))
    finally:
        sock.close(0)
        ctx.term()


class CPUStubBackend(BaseBackend):
    """Minimal CPU trainer backend that reuses BaseBackend batch prep."""

    def __init__(self, config, rank_info=None, *, mode: str = "kl"):
        self.loss_mode = mode
        super().__init__(config, rank_info)
        self.device = torch.device("cpu")
        self.model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self.optimizer = None
        self.scheduler = None
        self._scheduler_needs_rebuild = False
        self.weights_info = self._build_weights_info()
        self._command_handlers["compute_weight_checksum"] = self._handle_compute_weight_checksum
        _record("trainer", "started", mode=mode, rank=self.rank, world_size=self.world_size)

    @property
    def use_distributed(self) -> bool:
        return False

    def _get_log_prefix(self) -> str:
        return "Trainer-CPUStub"

    def _build_weights_info(self):
        return [("weight", (1, 1), torch.float32)]

    def _train_step_impl(self, batch) -> dict:
        prepared = self._prepare_batch(batch)
        response_mask = prepared.get("response_mask")
        n_tokens = (
            int(response_mask.sum().item())
            if isinstance(response_mask, torch.Tensor) else 0
        )
        advantages = prepared.get("advantages")
        mean_adv = (
            float(advantages.float().mean().item())
            if isinstance(advantages, torch.Tensor) else 0.0
        )
        ref_token_logps = prepared.get("ref_token_logps")
        has_ref_token_logps = isinstance(ref_token_logps, torch.Tensor)
        metrics = {
            "kl_loss": 0.0,
            "n_tokens": n_tokens,
            "grad_norm": 0.0,
            "lr": 0.0,
            "n_optim_steps": 1,
            "clip_fraction": 0.0,
            "mean_advantage": mean_adv,
            "mean_kl": 0.0,
            "has_ref_token_logps": has_ref_token_logps,
        }
        _record("trainer", "train", mode=self.loss_mode, kl_loss=0.0,
                n_tokens=n_tokens, mean_advantage=mean_adv,
                has_ref_token_logps=has_ref_token_logps)
        return metrics

    def _run_train_step(self, batch, loss_fn, forward_and_loss_fn=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        teacher_indices = batch["teacher_topk_indices"]
        teacher_logps = batch["teacher_topk_logps"]
        response_mask = batch["response_mask"].bool()
        position_ids = batch.get("position_ids")
        used_packing = False
        packed_tokens = 0
        packed_max_seq_len = 0

        if self.use_sequence_packing and batch.get("prompt_lengths") is not None:
            from opd.data.packing import pack_micro_batch

            packed = pack_micro_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                teacher_topk_logps=teacher_logps,
                teacher_topk_indices=teacher_indices,
                response_mask=response_mask,
                prompt_lengths=batch["prompt_lengths"],
            )
            input_ids = packed.input_ids
            attention_mask = torch.ones_like(packed.input_ids, dtype=torch.long)
            teacher_indices = packed.teacher_topk_indices
            teacher_logps = packed.teacher_topk_logps
            response_mask = packed.response_mask.bool()
            position_ids = packed.position_ids
            used_packing = True
            packed_tokens = int(packed.input_ids.size(1))
            packed_max_seq_len = int(packed.max_seq_len)

        student_logps = _student_topk_logps(
            input_ids,
            attention_mask,
            teacher_indices,
            position_ids=position_ids,
        )
        active = response_mask & torch.isfinite(teacher_logps).all(dim=-1)
        if active.any():
            max_diff = float((student_logps[active] - teacher_logps[active]).abs().max().item())
        else:
            max_diff = 0.0
        loss_mb = {
            "teacher_topk_logps": teacher_logps,
            "teacher_topk_indices": teacher_indices,
            "response_mask": response_mask,
        }
        loss, n_tokens, extras = self._trainer._compute_loss(
            student_topk_logps=student_logps,
            mb=loss_mb,
        )
        kl_loss = float(loss.detach().cpu().item())
        metrics = {
            "kl_loss": kl_loss,
            "n_tokens": int(n_tokens),
            "grad_norm": 0.0,
            "lr": 0.0,
            "n_optim_steps": 1,
            "cpu_stub_alignment_ok": max_diff < 1e-8,
            "cpu_stub_max_abs_logp_diff": max_diff,
        }
        for key, value in extras.items():
            if key == "_raw_tensors":
                continue
            metrics[key] = float(value) if isinstance(value, (int, float)) else value
        _record("trainer", "train", mode="opd", kl_loss=kl_loss,
                n_tokens=n_tokens, alignment_ok=metrics["cpu_stub_alignment_ok"],
                max_abs_logp_diff=max_diff, used_packing=used_packing,
                packed_tokens=packed_tokens, packed_max_seq_len=packed_max_seq_len)
        return metrics

    def _get_state_dict_for_sync(self):
        return self._get_clean_state_dict()

    def _gather_full_state_dict(self):
        return self._get_clean_state_dict()

    def _gather_full_optim_state_dict(self):
        return None

    def _save_checkpoint(self, checkpoint_dir, step, state_dict,
                         save_optimizer=True, optim_state_dict=None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(checkpoint_dir, "model.pt"))
        if save_optimizer:
            torch.save({"step": step, "optim": optim_state_dict},
                       os.path.join(checkpoint_dir, "training_state.pt"))
        _record("trainer", "save_checkpoint", checkpoint_dir=checkpoint_dir, step=step)

    def _load_checkpoint(self, checkpoint_dir):
        match = re.search(r"step_(\d+)", str(checkpoint_dir))
        step = int(match.group(1)) if match else 0
        _record("trainer", "load_checkpoint", checkpoint_dir=checkpoint_dir, step=step)
        return step

    def _handle_compute_weight_checksum(self, cmd, t_recv, result_queue):
        result_queue.put({"checksum": _checksum(self._get_clean_state_dict())})

    def _handle_get_weights_info(self, cmd, t_recv, result_queue):
        _record("trainer", "get_weights_info", mode=self.loss_mode)
        return super()._handle_get_weights_info(cmd, t_recv, result_queue)

    def _handle_get_clean_state_dict(self, cmd, t_recv, result_queue):
        _record("trainer", "get_clean_state_dict", mode=self.loss_mode)
        return super()._handle_get_clean_state_dict(cmd, t_recv, result_queue)


class CPUStubOPDTrainer(OPDTrainer):
    """OPD trainer wrapper that avoids constructing FSDP/Megatron."""

    def __init__(self, config, rank_info=None):
        algo = (
            config.static.algorithm
            if isinstance(config, TrainerLaunchSpec) else config["algorithm"]
        )
        self.kl_config = build_kl_config_from_algorithm_payload(algo)
        self.launch_spec = config if isinstance(config, TrainerLaunchSpec) else None
        self._backend = CPUStubBackend(config, rank_info, mode="kl")
        self._use_decoupled_ppo = False
        self._prox_list = []
        self._prox_idx = 0
        self._use_chunked = False


def cpu_stub_opd_trainer_main(config, cmd_queue, result_queue, rank_info):
    trainer = CPUStubOPDTrainer(config, rank_info)
    trainer.run(cmd_queue, result_queue)


def _cpu_stub_simple_trainer_main(config, cmd_queue, result_queue, rank_info, *, mode: str):
    backend = CPUStubBackend(config, rank_info, mode=mode)
    backend.run(cmd_queue, result_queue)


def cpu_stub_grpo_trainer_main(config, cmd_queue, result_queue, rank_info):
    _cpu_stub_simple_trainer_main(config, cmd_queue, result_queue, rank_info, mode="grpo")


def cpu_stub_sft_trainer_main(config, cmd_queue, result_queue, rank_info):
    _cpu_stub_simple_trainer_main(config, cmd_queue, result_queue, rank_info, mode="sft")

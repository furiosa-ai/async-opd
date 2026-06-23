"""Trainer-side buffer for direct teacher artifact transport."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from opd.data.batch_utils import split_gen_teacher


_TEACHER_KEYS = {
    "teacher_topk_logps",
    "teacher_topk_indices",
    "teacher_token_logps",
    "teacher_valid_mask",
    "teacher_query_logprobs_response",
    "teacher_mc_indices_response",
    "support_topk_logps",
    "support_topk_indices",
    "support_valid_mask",
    "support_student_old_logps",
    "mc_sample_indices",
    "mc_teacher_logprobs",
    "mc_valid_mask",
    "mc_old_logprobs",
}


def estimate_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(estimate_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(estimate_tensor_bytes(v) for v in value)
    return 0


def _slice_sample(gen_output: dict, idx: int) -> dict:
    sample = {}
    for key, value in gen_output.items():
        if key.startswith("_"):
            continue
        if isinstance(value, torch.Tensor):
            if value.size(0) > idx:
                sample[key] = value[idx:idx + 1]
        elif isinstance(value, list):
            if len(value) > idx:
                sample[key] = [value[idx]]
        else:
            sample[key] = value
    if "sample_seq_ids" in gen_output:
        sample["sample_seq_id"] = gen_output["sample_seq_ids"][idx]
    sample["sample_in_batch_idx"] = idx
    return sample


@dataclass
class _BufferedLogicalBatch:
    expected: int
    sample_seq_ids: list[Any] | None = None
    artifacts: dict[int, dict] = field(default_factory=dict)
    recv_bytes: int = 0
    recv_events: list[dict] = field(default_factory=list)


class TrainerTeacherArtifactBuffer:
    """Rank-0 trainer buffer for out-of-order teacher artifacts.

    Artifacts are received on a queue that is separate from the trainer command
    queue.  The train command contains only generation tensors and a logical
    buffer id; rank 0 waits here, canonicalizes artifacts to the existing OPD
    teacher fields, then broadcasts the normal batch to the other ranks.
    """

    def __init__(self, artifact_queue=None, *, max_batches: int = 3):
        self.artifact_queue = artifact_queue
        self.max_batches = max(int(max_batches), 1)
        self._batches: dict[int, _BufferedLogicalBatch] = {}
        self._last_metrics: dict[str, Any] = {}

    @property
    def last_metrics(self) -> dict[str, Any]:
        return dict(self._last_metrics)

    def add(self, envelope: dict) -> dict:
        lbid = int(envelope["logical_batch_id"])
        idx = int(envelope["sample_in_batch_idx"])
        expected = int(envelope["n_expected"])
        sample_seq_id = envelope.get("sample_seq_id")
        rec = self._batches.setdefault(
            lbid,
            _BufferedLogicalBatch(expected=expected),
        )
        if rec.expected != expected:
            raise RuntimeError(
                f"teacher artifact expected-count mismatch for logical_batch_id={lbid}: "
                f"{expected} != {rec.expected}"
            )
        if idx in rec.artifacts:
            raise RuntimeError(
                f"duplicate teacher artifact for logical_batch_id={lbid} "
                f"sample_in_batch_idx={idx}"
            )
        payload = envelope.get("payload") or {}
        n_bytes = int(envelope.get("n_bytes") or estimate_tensor_bytes(payload))
        rec.artifacts[idx] = {
            "sample_seq_id": sample_seq_id,
            "payload": payload,
            "payload_kind": envelope.get("payload_kind"),
        }
        rec.recv_bytes += n_bytes
        rec.recv_events.append({
            "logical_batch_id": lbid,
            "sample_in_batch_idx": idx,
            "n_bytes": n_bytes,
            "mono_ts": time.monotonic(),
        })
        self._evict_old(lbid)
        return self.status(lbid)

    def status(self, logical_batch_id: int) -> dict:
        rec = self._batches.get(int(logical_batch_id))
        if rec is None:
            return {"ready_count": 0, "expected_count": 0, "complete": False}
        return {
            "ready_count": len(rec.artifacts),
            "expected_count": rec.expected,
            "complete": len(rec.artifacts) == rec.expected,
        }

    def wait_complete(self, logical_batch_id: int, *, expected_count: int,
                      timeout_s: float = 300.0) -> None:
        if self.artifact_queue is None:
            raise RuntimeError("direct teacher artifact queue is not configured")
        lbid = int(logical_batch_id)
        deadline = time.monotonic() + timeout_s
        while not self.status(lbid)["complete"]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                st = self.status(lbid)
                raise TimeoutError(
                    f"teacher artifact timeout for logical_batch_id={lbid}: "
                    f"ready={st['ready_count']} expected={expected_count}"
                )
            try:
                envelope = self.artifact_queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if isinstance(envelope, dict) and envelope.get("status") == "abort":
                raise RuntimeError(envelope.get("reason", "teacher artifact channel aborted"))
            self.add(envelope)
        st = self.status(lbid)
        if st["expected_count"] != int(expected_count):
            raise RuntimeError(
                f"teacher artifact expected-count mismatch for logical_batch_id={lbid}: "
                f"{st['expected_count']} != {expected_count}"
            )

    def assemble_canonical(self, logical_batch_id: int, gen_output: dict, *,
                           teacher_recompute_head=None) -> dict:
        lbid = int(logical_batch_id)
        rec = self._batches.get(lbid)
        if rec is None or len(rec.artifacts) != rec.expected:
            raise RuntimeError(f"teacher artifacts for logical_batch_id={lbid} are not complete")

        sample_seq_ids = gen_output.get("sample_seq_ids")
        samples = []
        for idx in range(rec.expected):
            artifact = rec.artifacts.get(idx)
            if artifact is None:
                raise RuntimeError(
                    f"missing teacher artifact for logical_batch_id={lbid} idx={idx}"
                )
            if sample_seq_ids is not None:
                expected_seq = sample_seq_ids[idx]
                actual_seq = artifact.get("sample_seq_id")
                if actual_seq != expected_seq:
                    raise RuntimeError(
                        f"teacher artifact sample_seq_id mismatch for logical_batch_id={lbid} "
                        f"idx={idx}: {actual_seq!r} != {expected_seq!r}"
                    )
            if artifact.get("payload_kind") == "canonical_teacher_output":
                samples.append(artifact["payload"])
                continue
            sample = _slice_sample(gen_output, idx)
            sample.update(artifact["payload"])
            samples.append(sample)

        payload_kinds = {
            rec.artifacts[idx].get("payload_kind") for idx in range(rec.expected)
        }
        recompute_metrics = {}
        if payload_kinds == {"canonical_teacher_output"}:
            teacher_output = self._concat_canonical_samples(samples)
        elif payload_kinds == {"hidden_states"}:
            if teacher_recompute_head is None:
                raise RuntimeError("teacher hidden-state artifacts require a trainer recompute head")
            hidden_payloads = [rec.artifacts[idx]["payload"] for idx in range(rec.expected)]
            if getattr(teacher_recompute_head, "materialization", "lazy") == "canonical":
                recomputed = teacher_recompute_head.assemble_dense_teacher_output(
                    gen_output=gen_output,
                    hidden_payloads=hidden_payloads,
                )
            else:
                recomputed = teacher_recompute_head.assemble_lazy_teacher_artifacts(
                    gen_output=gen_output,
                    hidden_payloads=hidden_payloads,
                )
            teacher_output = recomputed.teacher_output
            recompute_metrics = recomputed.metrics
        else:
            _gen, teacher_output = split_gen_teacher(samples)
        self._last_metrics = {
            "logical_batch_id": lbid,
            "recv_bytes": rec.recv_bytes,
            "recv_events": list(rec.recv_events),
            "ready_count": len(rec.artifacts),
            "expected_count": rec.expected,
            "ready_mono": time.monotonic(),
            **recompute_metrics,
        }
        return teacher_output

    @staticmethod
    def _concat_canonical_samples(samples: list[dict]) -> dict:
        """Reassemble per-row slices from an already canonical teacher batch."""
        out = {}
        keys = sorted({k for sample in samples for k in sample})
        for key in keys:
            vals = [sample[key] for sample in samples if key in sample]
            if not vals:
                continue
            first = vals[0]
            if isinstance(first, torch.Tensor):
                out[key] = torch.cat(vals, dim=0)
            elif isinstance(first, list):
                merged = []
                for v in vals:
                    merged.extend(v)
                out[key] = merged
            else:
                out[key] = vals
        return out

    def pop(self, logical_batch_id: int) -> None:
        self._batches.pop(int(logical_batch_id), None)

    def abort(self, logical_batch_id: int, reason: str) -> None:
        self._batches.pop(int(logical_batch_id), None)
        self._last_metrics = {
            "logical_batch_id": int(logical_batch_id),
            "error": reason,
        }

    def _evict_old(self, active_lbid: int) -> None:
        if len(self._batches) <= self.max_batches:
            return
        complete = sorted(
            lbid for lbid, rec in self._batches.items()
            if lbid != active_lbid and len(rec.artifacts) == rec.expected
        )
        while len(self._batches) > self.max_batches and complete:
            self._batches.pop(complete.pop(0), None)

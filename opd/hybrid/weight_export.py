"""Bucketed in-process actor-to-vLLM refresh helpers for fused hybrid OPD.

The default fused hybrid path intentionally avoids the existing trainer-to-rollout
NCCL weight-transfer communicator because the student trainer and rollout share
the same physical GPUs.  These helpers instead run inside each student rank and
load checkpoint-format tensors into the colocated vLLM worker in bounded buckets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from time import monotonic
from typing import Iterable, Iterator

import torch


@dataclass
class BucketedWeightUpdateTelemetry:
    backend: str
    bucket_count: int
    total_bytes: int
    max_bucket_bytes: int
    duration_s: float
    debug_full_state_sync: bool
    full_state_materialized: bool
    memory_before: dict
    memory_after: dict
    checksum_enabled: bool = False
    checksum_source: float | None = None
    checksum_target_local: float | None = None
    checksum_source_count: int | None = None
    checksum_target_count_local: int | None = None
    checksum_missing_source_count: int | None = None

    def to_metrics(self, prefix: str = "fused_hybrid_weight_update") -> dict:
        data = asdict(self)
        out = {}
        for key, value in data.items():
            if value is None:
                continue
            if key in {"memory_before", "memory_after"}:
                for mk, mv in value.items():
                    out[f"{prefix}_{key}_{mk}"] = mv
            else:
                out[f"{prefix}_{key}"] = value
        return out


def cuda_memory_snapshot(label: str) -> dict:
    if not torch.cuda.is_available():
        return {
            "label": label,
            "device": None,
            "allocated": 0,
            "reserved": 0,
            "max_allocated": 0,
        }
    device = torch.cuda.current_device()
    return {
        "label": label,
        "device": int(device),
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
    }


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _stable_checksum_weight(name: str) -> float:
    """Return a deterministic non-zero group weight for checksum mixing."""
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return 1.0 + (value % 1_000_003) / 1_000_003.0


def _checksum_tensor_abs(tensor: torch.Tensor, weight: float = 1.0) -> float:
    """Compute a CPU scalar checksum contribution without keeping temporaries."""
    return float(tensor.detach().float().abs().sum().item()) * float(weight)


def _is_skipped_vllm_param(name: str) -> bool:
    # FP8/quantization metadata is created on the receive side and is not part
    # of the source HF checkpoint stream.
    return name.endswith(("_scale", "_scale_inv", "_zero_point", "_amax"))


@dataclass
class WeightChecksumPlan:
    """Order-insensitive trainer↔vLLM checksum plan for fused hybrid refresh.

    ``source_weights`` maps HF/checkpoint tensor names streamed from FSDP to the
    stable target-group weight they should use.  ``target_weights`` maps vLLM
    internal parameter names to the same weights.  ``target_full_shapes`` is
    used to discount tensor-parallel replicated parameters before rank checksums
    are summed in the trainer process group.
    """

    source_weights: dict[str, float]
    target_weights: dict[str, float]
    target_full_shapes: dict[str, tuple[int, ...]]

    @property
    def source_names(self) -> set[str]:
        return set(self.source_weights)

    @property
    def target_names(self) -> set[str]:
        return set(self.target_weights)


def _shape_tuple(shape) -> tuple[int, ...]:
    return tuple(int(x) for x in shape)


def _source_plan_for_vllm_param(
    name: str,
    trainer_shapes: dict[str, tuple[int, ...]],
) -> tuple[list[str], tuple[int, ...]] | None:
    """Map one vLLM target parameter to source checkpoint tensor names."""
    if _is_skipped_vllm_param(name):
        return None

    if "qkv_proj" in name:
        base, suffix = name.split("qkv_proj", 1)
        sources = [
            base + "q_proj" + suffix,
            base + "k_proj" + suffix,
            base + "v_proj" + suffix,
        ]
        if all(src in trainer_shapes for src in sources):
            head = sum(trainer_shapes[src][0] for src in sources)
            return sources, (head, *trainer_shapes[sources[0]][1:])
        if name in trainer_shapes:
            return [name], trainer_shapes[name]
        return None

    if "gate_up_proj" in name:
        base, suffix = name.split("gate_up_proj", 1)
        sources = [base + "gate_proj" + suffix, base + "up_proj" + suffix]
        if all(src in trainer_shapes for src in sources):
            head = sum(trainer_shapes[src][0] for src in sources)
            return sources, (head, *trainer_shapes[sources[0]][1:])
        if name in trainer_shapes:
            return [name], trainer_shapes[name]
        return None

    if name in trainer_shapes:
        return [name], trainer_shapes[name]
    return None


def build_weight_checksum_plan(
    trainer_weights_info: Iterable[tuple[str, tuple[int, ...], torch.dtype]],
    vllm_params_info: Iterable[tuple[str, tuple[int, ...], torch.dtype]],
) -> WeightChecksumPlan:
    """Build the fused-hybrid checksum map from trainer/vLLM metadata.

    The checksum compares the full trainer checkpoint tensor stream against the
    tensor-parallel vLLM model after loading.  It is intentionally independent of
    iteration order: fused qkv/gate-up target groups and their source tensors get
    the same stable per-target weight.
    """
    trainer_shapes = {
        str(name): _shape_tuple(shape)
        for name, shape, _dtype in trainer_weights_info
        if not str(name).startswith("value_head.")
    }
    source_weights: dict[str, float] = {}
    target_weights: dict[str, float] = {}
    target_full_shapes: dict[str, tuple[int, ...]] = {}

    for name, _shape, _dtype in vllm_params_info:
        name = str(name)
        mapped = _source_plan_for_vllm_param(name, trainer_shapes)
        if mapped is None:
            continue
        sources, full_shape = mapped
        weight = _stable_checksum_weight(name)
        target_weights[name] = weight
        target_full_shapes[name] = full_shape
        for src in sources:
            # In normal HF/vLLM mappings a source tensor belongs to one target
            # group.  If a tied weight appears twice, keeping the first mapping
            # avoids double-counting the source tensor while still making the
            # target mismatch visible.
            source_weights.setdefault(src, weight)

    return WeightChecksumPlan(
        source_weights=source_weights,
        target_weights=target_weights,
        target_full_shapes=target_full_shapes,
    )


def iter_weight_buckets(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    bucket_bytes: int,
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Yield deterministic non-empty buckets from ``named_tensors``.

    A tensor larger than the bucket target is yielded as a singleton bucket. The
    input order is preserved; callers should pass a stable checkpoint state_dict
    order.
    """
    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be positive")
    bucket: list[tuple[str, torch.Tensor]] = []
    used = 0
    for name, tensor in named_tensors:
        nbytes = tensor_nbytes(tensor)
        if bucket and used + nbytes > bucket_bytes:
            yield bucket
            bucket = []
            used = 0
        bucket.append((name, tensor))
        used += nbytes
        if used >= bucket_bytes:
            yield bucket
            bucket = []
            used = 0
    if bucket:
        yield bucket


class LoadCheckpointBucketFn:
    """Picklable vLLM ``apply_model`` callable for one checkpoint bucket."""

    def __init__(self, bucket: Iterable[tuple[str, torch.Tensor]]):
        # Keep tensors detached. If a tensor is already on CUDA, vLLM's model
        # load path can consume it without a coordinator CPU round-trip.
        self.items = [(name, tensor.detach()) for name, tensor in bucket]

    def __call__(self, model):
        with torch.no_grad():
            loaded_raw = model.load_weights(self.items)
            for p in model.parameters():
                p.requires_grad_(False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        loaded = None
        try:
            loaded = len(list(loaded_raw)) if loaded_raw is not None else len(self.items)
        except TypeError:
            loaded = len(self.items)
        return {"loaded": loaded, "bucket_items": len(self.items)}


class ComputeTargetChecksumFn:
    """Picklable vLLM ``apply_model`` callable for target checksum telemetry."""

    def __init__(
        self,
        target_weights: dict[str, float],
        target_full_shapes: dict[str, tuple[int, ...]],
        tp_size: int,
    ):
        self.target_weights = dict(target_weights)
        self.target_full_shapes = {
            str(name): tuple(int(x) for x in shape)
            for name, shape in target_full_shapes.items()
        }
        self.tp_size = max(int(tp_size), 1)

    def __call__(self, model):
        checksum = 0.0
        count = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                weight = self.target_weights.get(name)
                if weight is None:
                    continue
                contribution = _checksum_tensor_abs(param, weight)
                full_shape = self.target_full_shapes.get(name)
                if full_shape is not None and tuple(param.shape) == tuple(full_shape):
                    # Tensor-parallel replicated weights (for example norms)
                    # appear on every TP rank.  Divide locally before the rank
                    # checksums are summed by the trainer-side process group.
                    contribution /= self.tp_size
                checksum += contribution
                count += 1
        return {"checksum": checksum, "count": count}


class BucketedInprocessWeightUpdater:
    """Load actor state into a colocated vLLM LLM object bucket-by-bucket."""

    def __init__(
        self,
        *,
        bucket_mb: int = 256,
        debug_full_state_sync: bool = False,
        backend: str = "bucketed_inprocess",
        verify_checksum: bool = False,
        checksum_plan: WeightChecksumPlan | None = None,
        vllm_params_info: Iterable[tuple[str, tuple[int, ...], torch.dtype]] | None = None,
        tp_size: int = 1,
    ):
        self.bucket_bytes = int(bucket_mb) << 20
        self.debug_full_state_sync = bool(debug_full_state_sync)
        self.backend = str(backend)
        self.verify_checksum = bool(verify_checksum)
        self.checksum_plan = checksum_plan
        self.vllm_params_info = list(vllm_params_info or [])
        self.tp_size = int(tp_size)

    def update(self, llm, state_dict: dict[str, torch.Tensor]) -> BucketedWeightUpdateTelemetry:
        """Refresh from an already materialized state dict.

        This path is retained for explicit debug/full-state callers. The
        signoff fused hybrid path should use ``update_from_named_tensors`` so
        FSDP materialization can be scoped to the current module/bucket.
        """
        return self.update_from_named_tensors(
            llm,
            state_dict.items(),
            full_state_materialized=True,
        )

    def update_from_named_tensors(
        self,
        llm,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        *,
        full_state_materialized: bool,
    ) -> BucketedWeightUpdateTelemetry:
        """Refresh vLLM weights from a streaming named-tensor iterator."""
        before = cuda_memory_snapshot("before_bucketed_weight_update")
        t0 = monotonic()
        bucket_count = 0
        total_bytes = 0
        max_bucket_bytes = 0
        source_abs_sums: dict[str, float] = {}
        source_info: dict[str, tuple[str, tuple[int, ...], torch.dtype]] = {}
        for bucket in iter_weight_buckets(named_tensors, self.bucket_bytes):
            bucket_count += 1
            bucket_bytes = sum(tensor_nbytes(t) for _, t in bucket)
            total_bytes += bucket_bytes
            max_bucket_bytes = max(max_bucket_bytes, bucket_bytes)
            if self.verify_checksum:
                for name, tensor in bucket:
                    if name.startswith("value_head."):
                        continue
                    source_abs_sums[name] = _checksum_tensor_abs(tensor)
                    source_info[name] = (name, tuple(tensor.shape), tensor.dtype)
            llm.apply_model(LoadCheckpointBucketFn(bucket))
            # Drop temporary references as soon as this bucket has been loaded.
            del bucket
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            llm.reset_prefix_cache(reset_running_requests=True)
        except TypeError:
            llm.reset_prefix_cache()
        duration = monotonic() - t0
        checksum_target_local = None
        checksum_target_count_local = None
        checksum_missing_source_count = None
        checksum_source = None
        checksum_source_count = None
        checksum_plan = self.checksum_plan if self.verify_checksum else None
        if self.verify_checksum:
            if checksum_plan is None or not checksum_plan.source_weights:
                checksum_plan = build_weight_checksum_plan(
                    source_info.values(),
                    self.vllm_params_info,
                )
            checksum_source = sum(
                source_abs_sums[name] * checksum_plan.source_weights[name]
                for name in checksum_plan.source_names
                if name in source_abs_sums
            )
            checksum_source_count = len(
                checksum_plan.source_names.intersection(source_abs_sums)
            )
            checksum_missing_source_count = len(
                checksum_plan.source_names.difference(source_abs_sums)
            )
            result = llm.apply_model(
                ComputeTargetChecksumFn(
                    checksum_plan.target_weights,
                    checksum_plan.target_full_shapes,
                    self.tp_size,
                )
            )
            if isinstance(result, list):
                result = result[0] if result else {}
            if not isinstance(result, dict):
                result = {}
            checksum_target_local = float(result.get("checksum", 0.0))
            checksum_target_count_local = int(result.get("count", 0))
        after = cuda_memory_snapshot("after_bucketed_weight_update")
        return BucketedWeightUpdateTelemetry(
            backend=self.backend,
            bucket_count=bucket_count,
            total_bytes=total_bytes,
            max_bucket_bytes=max_bucket_bytes,
            duration_s=duration,
            debug_full_state_sync=self.debug_full_state_sync,
            full_state_materialized=bool(full_state_materialized),
            memory_before=before,
            memory_after=after,
            checksum_enabled=self.verify_checksum,
            checksum_source=checksum_source,
            checksum_target_local=checksum_target_local,
            checksum_source_count=checksum_source_count,
            checksum_target_count_local=checksum_target_count_local,
            checksum_missing_source_count=checksum_missing_source_count,
        )

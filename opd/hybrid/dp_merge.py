"""Helpers for fused-hybrid data-parallel rollout sharding and merge."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


LIST_ROW_FIELDS = {
    "full_token_lists",
    "query_indices_response",
    "query_logprobs_response",
    "mc_query_indices_response",
    "mc_query_old_logprobs_response",
    "responses_multi",
    "sample_seq_ids",
}

METADATA_FIELDS = {"timing", "_worker_timings", "_vllm_stats"}


def shard_indices_for_rank(batch_size: int, rank: int, world_size: int) -> list[int]:
    """Return strided data-parallel sample indices for ``rank``."""

    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return list(range(rank, batch_size, world_size))


def slice_batch_for_indices(batch: dict[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    """Build a local sub-batch while preserving scalar options/metadata."""

    out: dict[str, Any] = {}
    batch_size = int(batch["input_ids"].size(0))
    index_list = [int(i) for i in indices]
    index_tensor_cache: dict[torch.device, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == batch_size:
            device = value.device
            if device not in index_tensor_cache:
                index_tensor_cache[device] = torch.tensor(
                    index_list,
                    dtype=torch.long,
                    device=device,
                )
            out[key] = value.index_select(0, index_tensor_cache[device])
        elif isinstance(value, list) and len(value) == batch_size:
            out[key] = [value[i] for i in index_list]
        else:
            out[key] = value
    return out


def indices_from_ranges(ranges: Sequence[tuple[int, int]]) -> list[int]:
    """Flatten half-open source ranges into an index list."""

    indices: list[int] = []
    for start, end in ranges:
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            raise ValueError(f"invalid source range [{start_i}, {end_i})")
        indices.extend(range(start_i, end_i))
    return indices


def contiguous_indices_for_rank(batch_size: int, rank: int, world_size: int) -> list[int]:
    """Return rank-first contiguous DP indices.

    This mirrors the historical trainer-side rank split for divisible batches.
    Non-divisible shapes should use the global-mini plan instead; this helper
    intentionally drops the remainder like the legacy rank-first path.
    """

    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    per_rank = batch_size // world_size
    start = rank * per_rank
    return list(range(start, start + per_rank))


def split_batch_for_dp_ranks(
    batch: dict[str, Any],
    world_size: int,
    *,
    indices_by_rank: Sequence[Sequence[int]] | None = None,
) -> list[dict[str, Any]]:
    """Split a global batch into rank-local shards with global indices.

    The returned shards preserve scalar metadata and tensor sequence width, so
    fused-hybrid DP can scatter prompt shards without broadcasting the full
    batch to every rank.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    batch_size = int(batch["input_ids"].size(0))
    if indices_by_rank is None:
        indices_by_rank = [
            contiguous_indices_for_rank(batch_size, rank, world_size)
            for rank in range(world_size)
        ]
    if len(indices_by_rank) != world_size:
        raise ValueError(
            f"indices_by_rank length {len(indices_by_rank)} does not match "
            f"world_size={world_size}"
        )

    shards: list[dict[str, Any]] = []
    seen: set[int] = set()
    for rank, indices in enumerate(indices_by_rank):
        index_list = [int(i) for i in indices]
        for idx in index_list:
            if idx < 0 or idx >= batch_size:
                raise ValueError(f"DP shard index {idx} outside batch size {batch_size}")
            if idx in seen:
                raise ValueError(f"duplicate DP shard index {idx}")
            seen.add(idx)
        local = slice_batch_for_indices(batch, index_list)
        local["_global_sample_indices"] = index_list
        local["_global_batch_size"] = batch_size
        local["_dp_rank"] = int(rank)
        local["_dp_world_size"] = int(world_size)
        shards.append(local)

    missing = sorted(set(range(batch_size)).difference(seen))
    if missing:
        raise ValueError(f"missing DP shard indices: {missing}")
    return shards


def _first_non_empty_result(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for payload in payloads:
        result = payload.get("result") or {}
        if result:
            return result
    return {}


def _merge_tensor_field(
    payloads: Sequence[dict[str, Any]],
    key: str,
    batch_size: int,
) -> torch.Tensor | None:
    exemplar = None
    for payload in payloads:
        result = payload.get("result") or {}
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            exemplar = value
            if value.size(0) > 0:
                break
    if exemplar is None:
        return None
    shape = (batch_size, *exemplar.shape[1:])
    merged = torch.zeros(shape, dtype=exemplar.dtype, device="cpu")
    for payload in payloads:
        indices = [int(i) for i in payload.get("indices", [])]
        if not indices:
            continue
        result = payload.get("result") or {}
        value = result.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        merged[torch.tensor(indices, dtype=torch.long)] = value.detach().cpu()
    return merged


def _merge_list_field(
    payloads: Sequence[dict[str, Any]],
    key: str,
    batch_size: int,
) -> list[Any] | None:
    present = any(key in (payload.get("result") or {}) for payload in payloads)
    if not present:
        return None
    merged: list[Any] = [None] * batch_size
    for payload in payloads:
        indices = [int(i) for i in payload.get("indices", [])]
        if not indices:
            continue
        result = payload.get("result") or {}
        values = result.get(key)
        if values is None:
            continue
        if len(values) != len(indices):
            raise ValueError(
                f"DP merge field {key!r} length {len(values)} does not match "
                f"indices length {len(indices)}"
            )
        for dest, value in zip(indices, values, strict=False):
            merged[dest] = value
    return merged


def _first_value_for_key(payloads: Sequence[dict[str, Any]], key: str) -> Any:
    for payload in payloads:
        result = payload.get("result") or {}
        if key in result:
            return result[key]
    return None


def _merge_timing(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    timings = [p.get("timing", {}) for p in payloads if isinstance(p.get("timing"), dict)]
    merged: dict[str, Any] = {"worker_id": 0}
    for timing in timings:
        for key, value in timing.items():
            if key == "worker_id":
                continue
            if isinstance(value, (int, float)):
                merged[key] = max(float(value), float(merged.get(key, value)))
            elif key not in merged:
                merged[key] = value
    return merged


def _validate_payload_indices(payloads: Sequence[dict[str, Any]], batch_size: int) -> None:
    seen: set[int] = set()
    for payload in payloads:
        indices = [int(i) for i in payload.get("indices", [])]
        result = payload.get("result") or {}
        row_count = 0
        for value in result.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                row_count = int(value.size(0))
                break
            if isinstance(value, list):
                row_count = len(value)
                break
        if row_count != len(indices):
            raise ValueError(
                f"DP payload rank={payload.get('rank')} has {row_count} rows "
                f"but {len(indices)} indices"
            )
        for idx in indices:
            if idx < 0 or idx >= batch_size:
                raise ValueError(f"DP payload index {idx} outside batch size {batch_size}")
            if idx in seen:
                raise ValueError(f"duplicate DP payload index {idx}")
            seen.add(idx)
    missing = sorted(set(range(batch_size)).difference(seen))
    if missing:
        raise ValueError(f"missing DP payload indices: {missing}")


def merge_indexed_generation_payloads(
    payloads: Sequence[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    """Merge per-rank generation payloads by original sample index."""

    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    _validate_payload_indices(payloads, batch_size)
    exemplar = _first_non_empty_result(payloads)
    merged: dict[str, Any] = {}
    keys = set(exemplar.keys())
    for payload in payloads:
        keys.update((payload.get("result") or {}).keys())
    keys.difference_update(METADATA_FIELDS)

    for key in sorted(keys):
        if key in LIST_ROW_FIELDS:
            value = _merge_list_field(payloads, key, batch_size)
            if value is not None:
                merged[key] = value
            continue
        exemplar_value = _first_value_for_key(payloads, key)
        if isinstance(exemplar_value, torch.Tensor):
            value = _merge_tensor_field(payloads, key, batch_size)
            if value is not None:
                merged[key] = value
            continue
        for payload in payloads:
            result = payload.get("result") or {}
            if key in result:
                merged[key] = result[key]
                break

    worker_timings = [
        payload.get("timing", {})
        for payload in sorted(payloads, key=lambda p: int(p.get("rank", 0)))
    ]
    merged["timing"] = _merge_timing(payloads)
    merged["_worker_timings"] = worker_timings
    merged["_vllm_stats"] = {
        int(payload.get("rank", idx)): payload.get("vllm_stats", [])
        for idx, payload in enumerate(payloads)
    }
    return merged

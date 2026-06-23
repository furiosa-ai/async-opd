"""NCCL-based GPU-to-GPU weight synchronization between trainer and rollout.

Architecture:
  Trainer (rank 0) and rollout workers (ranks 1..N) form a single
  torch.distributed NCCL process group. Weights are broadcast directly
  on GPU using bucketed transfers — no CPU round-trip.

  TensorBuffer packs small tensors into a contiguous buffer so that one
  NCCL broadcast covers many parameters (reduces kernel-launch overhead).
  Tensors larger than the buffer are broadcast individually.
"""
from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


# ------------------------------------------------------------------ #
#  TensorBuffer — bucketed packing for efficient broadcast            #
# ------------------------------------------------------------------ #

class TensorBuffer:
    """Contiguous GPU buffer that packs multiple tensors for batch broadcast."""

    def __init__(self, capacity_bytes: int, dtype: torch.dtype, device):
        elem_size = torch.tensor([], dtype=dtype).element_size()
        self.capacity = capacity_bytes // elem_size
        self.buffer = torch.empty(self.capacity, dtype=dtype, device=device)
        self._entries: list[tuple[str, torch.Size]] = []

    @property
    def used(self) -> int:
        return sum(s.numel() for _, s in self._entries)

    def clear(self):
        self._entries.clear()

    def pack(self, key: str, shape: torch.Size, data=None):
        """Append a tensor descriptor (and optionally copy data) into the buffer."""
        offset = self.used
        assert offset + shape.numel() <= self.capacity, \
            f"TensorBuffer overflow: {offset + shape.numel()} > {self.capacity}"
        if data is not None:
            self.buffer[offset : offset + shape.numel()] = data.reshape(-1)
        self._entries.append((key, shape))

    def unpack(self) -> list[tuple[str, torch.Tensor]]:
        """Return list of (key, view_tensor) sliced from the buffer."""
        out = []
        offset = 0
        for key, shape in self._entries:
            n = shape.numel()
            out.append((key, self.buffer[offset : offset + n].view(shape)))
            offset += n
        return out


# ------------------------------------------------------------------ #
#  Process-group initialization                                       #
# ------------------------------------------------------------------ #

def init_weight_sync_group(
    rank: int,
    world_size: int,
    master_addr: str = "127.0.0.1",
    master_port: int = 29400,
    timeout_sec: int = 300,
) -> dist.ProcessGroup:
    """Initialize (or reuse) a NCCL process group for weight sync.

    All participating processes (trainer rank-0 + every rollout worker)
    must call this with consistent rank / world_size.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_sec),
        )
        return dist.group.WORLD

    # Already initialized (e.g. multi-GPU FSDP) — create a sub-group
    # that spans all weight-sync ranks.
    pg = dist.new_group(ranks=list(range(world_size)),
                        timeout=timedelta(seconds=timeout_sec))
    return pg


# ------------------------------------------------------------------ #
#  Broadcast / receive                                                #
# ------------------------------------------------------------------ #

from typing import List, Tuple
WeightsInfo = List[Tuple[str, torch.Size, torch.dtype]]


def broadcast_weights(
    params_iter,           # iterator of (key, gpu_tensor) — only on src
    weights_info: WeightsInfo,
    group: dist.ProcessGroup,
    src_rank: int = 0,
    bucket_bytes: int = 128 << 20,   # 128 MiB default
    dtype: torch.dtype = torch.bfloat16,
    device=None,
) -> list[tuple[str, torch.Tensor]] | None:
    """Bucketed NCCL broadcast of model weights.

    Args:
        params_iter: On src_rank, yields (key, gpu_tensor) in the same order
                     as *weights_info*.  On other ranks, pass ``None``.
        weights_info: Schema — [(key, shape, dtype), ...] — shared by all ranks.
        group:  NCCL process group returned by ``init_weight_sync_group``.
        src_rank: rank that owns the authoritative weights (trainer).
        bucket_bytes: size of the packing buffer.
        dtype: dtype used for the packing buffer.
        device: CUDA device for receive-side tensor allocation.

    Returns:
        On non-src ranks: list of (key, gpu_tensor) ready for vLLM load_weights.
        On src_rank: None.
    """
    is_src = (dist.get_rank(group) == src_rank)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    buf = TensorBuffer(bucket_bytes, dtype, device)
    received: list[tuple[str, torch.Tensor]] = []

    def _flush():
        """Broadcast the current buffer contents."""
        dist.broadcast(buf.buffer, src=src_rank, group=group)
        if not is_src:
            # Clone views — buffer is reused across flushes
            received.extend((k, t.clone()) for k, t in buf.unpack())
        buf.clear()

    for key, shape, wdtype in weights_info:
        n = shape.numel()

        # --- Large tensor: standalone broadcast ---
        if n > buf.capacity:
            # flush anything already buffered
            if buf.used > 0:
                _flush()

            if is_src:
                pkey, tensor = next(params_iter)
                assert pkey == key, f"key mismatch: expected {key}, got {pkey}"
                if tensor.device != device:
                    tensor = tensor.to(device, non_blocking=True)
                dist.broadcast(tensor, src=src_rank, group=group)
            else:
                tensor = torch.empty(shape, dtype=wdtype, device=device)
                dist.broadcast(tensor, src=src_rank, group=group)
                received.append((key, tensor))
            continue

        # --- Buffer full? flush first ---
        if buf.used + n > buf.capacity:
            _flush()

        # --- Pack into buffer ---
        if is_src:
            pkey, tensor = next(params_iter)
            assert pkey == key, f"key mismatch: expected {key}, got {pkey}"
            if tensor.device != device:
                tensor = tensor.to(device, non_blocking=True)
            buf.pack(key, shape, tensor)
        else:
            buf.pack(key, shape)

    # final flush
    if buf.used > 0:
        _flush()

    return received if not is_src else None

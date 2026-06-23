"""Ray collective broadcast weight sync for trainer → rollout.

Replaces vLLM's NCCL weight transfer engine when using the Ray backend.
Uses ray.util.collective.broadcast() which creates a separate NCCL group
between trainer rank 0 and rollout workers — no interference with
Megatron's internal NCCL groups.

Protocol:
  1. initialize(): create collective group, exchange weights info
  2. sync(): trainer broadcasts gathered state dict, rollout receives + load_weights
  3. TensorBuffer packing for small params (reduces NCCL kernel launches)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from opd.worker.weight_merge import build_weight_merge_map
from opd.worker.weight_sync import TensorBuffer

if TYPE_CHECKING:
    from opd.worker.proxy import TrainerProxy, RolloutProxy

# Default bucket size for packing small tensors (128 MB)
_BUCKET_BYTES = 128 * 1024 * 1024


def _trainer_broadcast_weights(
    state_dict: dict[str, torch.Tensor],
    weights_info: list[tuple[str, torch.Size, torch.dtype]],
    group_name: str = "trainer_rollout",
    bucket_bytes: int = _BUCKET_BYTES,
) -> None:
    """Broadcast state dict from trainer (rank 0) to rollout via Ray collective.

    Iterates weights_info to determine send order. Packs small tensors into
    TensorBuffer for efficiency.
    """
    from ray.util.collective import collective

    device = next(iter(state_dict.values())).device
    dtype = next(iter(state_dict.values())).dtype
    buf = TensorBuffer(bucket_bytes, dtype, device)

    for name, shape, _ in weights_info:
        tensor = state_dict[name].to(device)
        numel = shape.numel()

        if numel > buf.capacity:
            # Large tensor: broadcast directly
            collective.broadcast(tensor.contiguous(), src_rank=0, group_name=group_name)
        else:
            # Would overflow buffer? Flush first.
            if buf.used + numel > buf.capacity:
                collective.broadcast(buf.buffer, src_rank=0, group_name=group_name)
                buf.clear()
            buf.pack(name, shape, tensor)

    # Flush remaining
    if buf.used > 0:
        collective.broadcast(buf.buffer, src_rank=0, group_name=group_name)


def _rollout_receive_weights(
    weights_info: list[tuple[str, torch.Size, str]],
    group_name: str = "trainer_rollout",
    bucket_bytes: int = _BUCKET_BYTES,
) -> list[tuple[str, torch.Tensor]]:
    """Receive weights from trainer via Ray collective broadcast.

    Returns list of (name, tensor) tuples suitable for model.load_weights().
    """
    from ray.util.collective import collective

    device = torch.cuda.current_device()
    # Determine dtype from weights_info
    dtype_str = weights_info[0][2] if weights_info else "bfloat16"
    dtype = getattr(torch, dtype_str, torch.bfloat16) if isinstance(dtype_str, str) else dtype_str
    buf = TensorBuffer(bucket_bytes, dtype, device)

    all_tensors = []

    for name, shape, _ in weights_info:
        numel = shape.numel()

        if numel > buf.capacity:
            # Large tensor: receive directly
            tensor = torch.empty(shape, dtype=dtype, device=device)
            collective.broadcast(tensor, src_rank=0, group_name=group_name)
            all_tensors.append((name, tensor))
        else:
            if buf.used + numel > buf.capacity:
                # Flush: receive packed buffer, unpack
                collective.broadcast(buf.buffer, src_rank=0, group_name=group_name)
                all_tensors.extend(buf.unpack())
                buf.clear()
            buf.pack(name, shape)  # No data on receiver

    if buf.used > 0:
        collective.broadcast(buf.buffer, src_rank=0, group_name=group_name)
        all_tensors.extend(buf.unpack())

    return all_tensors


class RayCollectiveWeightSyncEngine:
    """Weight sync via Ray collective broadcast (replaces vLLM NCCL engine).

    Used with the Ray backend for all Megatron configs (TP/PP/DP).
    The trainer gathers TP/PP weights to rank 0, then broadcasts via
    ray.util.collective. The rollout receives and loads directly via
    model.load_weights().
    """

    def __init__(self, verify_checksum: bool = False):
        self._verify_checksum = verify_checksum
        self._vllm_params_info: list | None = None
        self._trainer_weights_info: list | None = None
        self._weight_merge_map: list | None = None
        self._group_name = "trainer_rollout"

    @property
    def vllm_params_info(self) -> list:
        return self._vllm_params_info

    @property
    def trainer_weights_info(self) -> list:
        return self._trainer_weights_info

    @property
    def weight_merge_map(self) -> list:
        return self._weight_merge_map

    def initialize(self, trainer_proxy, rollout_proxy,
                   trainer_actors: list, rollout_actors: list,
                   master_address: str = "127.0.0.1") -> None:
        """Create Ray collective group and exchange weight metadata.

        Args:
            trainer_proxy: TrainerProxy for sending commands to rank 0
            rollout_proxy: RolloutProxy for sending commands to rollout workers
            trainer_actors: List of Ray actor handles for trainer rank 0 only
            rollout_actors: List of Ray actor handles for rollout workers
            master_address: unused (collective uses Ray's transport)
        """
        import ray

        # Create collective group: trainer rank 0 + all rollout workers
        # Trainer is rank 0, rollout workers are ranks 1..N
        from ray.util.collective import collective
        all_actors = trainer_actors + rollout_actors
        collective.create_collective_group(
            all_actors,
            world_size=len(all_actors),
            ranks=list(range(len(all_actors))),
            backend="nccl",
            group_name=self._group_name,
        )

        # Get trainer weights info (fused names for checksum merge_map)
        trainer_info = trainer_proxy.submit_command("get_weights_info")
        self._trainer_weights_info = trainer_info["weights_info"]

        # Collective uses same fused names as trainer (qkv_proj, gate_up_proj)
        # Rollout does direct param copy matching model.named_parameters()
        self._collective_weights_info = self._trainer_weights_info

        # Get vLLM params info
        rollout_proxy.submit_command("get_vllm_params_info")
        results = rollout_proxy.collect_results()
        self._vllm_params_info = results[0]["params_info"]

        # Build merge map (used for checksum verification — uses fused names)
        self._weight_merge_map = build_weight_merge_map(
            self._trainer_weights_info, self._vllm_params_info
        )

        n_vllm = len(self._vllm_params_info)
        n_trainer = len(self._trainer_weights_info)
        print(f"[Pipeline] Ray collective weight sync ready "
              f"({n_vllm} vLLM params, {n_trainer} trainer params, "
              f"group={self._group_name})", flush=True)

    def sync(self, trainer_proxy, rollout_proxy) -> float:
        """Broadcast weights from trainer to rollout via Ray collective."""
        import ray

        t0 = time.time()

        # Tell trainer to broadcast via collective (non-blocking)
        trainer_proxy.submit_command_async(
            "sync_weights_collective", self._group_name
        )

        # Tell rollout to receive via collective (non-blocking)
        rollout_proxy.submit_command(
            "sync_weights_collective",
            {"group_name": self._group_name,
             "weights_info": self._trainer_weights_info}
        )

        # Wait for both sides
        trainer_res = trainer_proxy.collect_command()
        rollout_proxy.collect_results()

        dt = time.time() - t0

        if self._verify_checksum:
            self.verify_checksums(trainer_proxy, rollout_proxy)

        return trainer_res.get("sync_seconds", dt)

    def verify_checksums(self, trainer_proxy, rollout_proxy) -> None:
        """Compare weight checksums between trainer and rollout."""
        trainer_cksum = trainer_proxy.submit_command(
            "compute_weight_checksum", self._weight_merge_map)["checksum"]
        rollout_proxy.submit_command("compute_weight_checksum")
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

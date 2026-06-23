"""CPU affinity helpers for NUMA-aware rollout worker placement."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import subprocess
from collections import defaultdict

_LIBNUMA = None


def expand_cpu_affinity(spec: str | None) -> list[int]:
    """Expand a compact CPU affinity string like ``0-3,8,10-11``."""
    if not spec or spec == "N/A":
        return []
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def parse_nvidia_smi_topo_matrix(text: str) -> dict[str, dict]:
    """Parse ``nvidia-smi topo -m`` output into GPU -> locality metadata."""
    topo: dict[str, dict] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not re.match(r"^GPU\d+\s", line):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 4 or (len(parts) > 1 and re.match(r"^GPU\d+$", parts[1])):
            continue
        gpu = parts[0][3:]
        cpu_affinity = parts[-3]
        numa_affinity = parts[-2]
        topo[gpu] = {
            "cpu_affinity": cpu_affinity,
            "cpus": expand_cpu_affinity(cpu_affinity),
            "numa_node": (None if numa_affinity == "N/A" else int(numa_affinity)),
        }
    return topo


def get_gpu_topology() -> dict[str, dict]:
    """Read GPU locality metadata from ``nvidia-smi topo -m``."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}
    return parse_nvidia_smi_topo_matrix(proc.stdout)


def split_cpu_set(cpus: list[int], parts: int) -> list[list[int]]:
    """Split a sorted CPU list into ``parts`` non-empty, nearly-even chunks."""
    if parts <= 1 or len(cpus) <= 1:
        return [cpus[:]]
    chunks: list[list[int]] = []
    base, extra = divmod(len(cpus), parts)
    start = 0
    for i in range(parts):
        size = base + (1 if i < extra else 0)
        chunk = cpus[start:start + size]
        if not chunk:
            chunk = cpus[:]
        chunks.append(chunk)
        start += size
    return chunks


def plan_rollout_cpu_affinities(
    rollout_gpu_list: list[str], tp: int, gpu_topology: dict[str, dict],
) -> dict[int, dict]:
    """Assign per-worker CPU affinity based on GPU NUMA locality.

    Workers using GPUs from a single NUMA node share that node's CPU pool,
    split evenly across the workers that land on the same node.
    """
    plans: dict[int, dict] = {}
    node_groups: dict[int, list[dict]] = defaultdict(list)
    n_workers = max(len(rollout_gpu_list) // max(tp, 1), 1)
    for worker_id in range(n_workers):
        worker_gpu_ids = rollout_gpu_list[worker_id * tp:(worker_id + 1) * tp]
        infos = [gpu_topology.get(gid) for gid in worker_gpu_ids if gpu_topology.get(gid)]
        if not infos:
            continue
        cpu_union = sorted(set().union(*(info["cpus"] for info in infos)))
        numa_nodes = sorted(
            {info["numa_node"] for info in infos if info["numa_node"] is not None}
        )
        plan = {
            "worker_id": worker_id,
            "gpu_ids": worker_gpu_ids,
            "cpus": cpu_union,
            "numa_nodes": numa_nodes,
        }
        if len(numa_nodes) == 1 and cpu_union:
            node_groups[numa_nodes[0]].append(plan)
        else:
            plans[worker_id] = {
                "cpu_affinity_cpus": cpu_union,
                "numa_nodes": numa_nodes,
            }

    for numa_node, group in node_groups.items():
        cpu_union = sorted(set().union(*(entry["cpus"] for entry in group)))
        chunks = split_cpu_set(cpu_union, len(group))
        for entry, chunk in zip(sorted(group, key=lambda x: x["worker_id"]), chunks):
            plans[entry["worker_id"]] = {
                "cpu_affinity_cpus": chunk,
                "numa_nodes": [numa_node],
            }
    return plans


def _load_libnuma():
    """Load libnuma lazily. Returns None if unavailable."""
    global _LIBNUMA
    if _LIBNUMA is not None:
        return _LIBNUMA
    path = ctypes.util.find_library("numa")
    if not path:
        return None
    lib = ctypes.CDLL(path)
    # bitmask helpers
    lib.numa_available.restype = ctypes.c_int
    lib.numa_num_possible_nodes.restype = ctypes.c_int
    lib.numa_allocate_nodemask.restype = ctypes.c_void_p
    lib.numa_allocate_nodemask.argtypes = []
    lib.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    lib.numa_set_bind_policy.argtypes = [ctypes.c_int]
    lib.numa_set_membind.argtypes = [ctypes.c_void_p]
    lib.numa_bitmask_free.argtypes = [ctypes.c_void_p]
    _LIBNUMA = lib
    return lib


def try_bind_numa_memory(numa_node: int) -> bool:
    """Best-effort strict NUMA memory binding for the current process."""
    lib = _load_libnuma()
    if lib is None or lib.numa_available() < 0:
        return False
    mask = lib.numa_allocate_nodemask()
    if not mask:
        return False
    try:
        lib.numa_bitmask_setbit(mask, int(numa_node))
        lib.numa_set_bind_policy(1)
        lib.numa_set_membind(mask)
    finally:
        lib.numa_bitmask_free(mask)
    return True


def run_rollout_worker_with_affinity(worker_fn, config, cmd_q, res_q):
    """Apply per-worker CPU affinity before entering the rollout worker."""
    cpus = config.get("cpu_affinity_cpus") or []
    if cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cpus))
            print(
                f"[Rollout-{config.get('worker_id', '?')}] CPU affinity pinned to "
                f"{min(cpus)}-{max(cpus)} ({len(cpus)} CPUs)",
                flush=True,
            )
        except OSError as e:
            print(
                f"[Rollout-{config.get('worker_id', '?')}] WARNING: failed to set CPU affinity: {e}",
                flush=True,
            )
    numa_nodes = config.get("numa_nodes") or []
    if config.get("bind_numa_memory") and len(numa_nodes) == 1:
        if try_bind_numa_memory(numa_nodes[0]):
            print(
                f"[Rollout-{config.get('worker_id', '?')}] NUMA memory bound to node {numa_nodes[0]}",
                flush=True,
            )
        else:
            print(
                f"[Rollout-{config.get('worker_id', '?')}] WARNING: failed to bind NUMA memory to node {numa_nodes[0]}",
                flush=True,
            )
    worker_fn(config, cmd_q, res_q)

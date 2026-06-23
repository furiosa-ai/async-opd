"""GPU metrics sampling for Perfetto trace counters."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

_NVML_AUTO = object()


def _load_pynvml():
    """Import pynvml lazily so module import doesn't require NVML bindings."""
    try:
        import pynvml  # type: ignore

        return pynvml
    except Exception:
        return None


@dataclass(frozen=True)
class GPUTraceTrack:
    """Single GPU counter track in Perfetto."""

    role: str
    gpu_id: int
    tid: int
    counter_name: str


def parse_gpu_ids(raw: str | None) -> list[int]:
    """Parse a comma-separated gpu_ids string into sorted unique ints."""
    if not raw:
        return []
    out = []
    seen = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        gid = int(part)
        if gid not in seen:
            seen.add(gid)
            out.append(gid)
    return out


def build_gpu_trace_tracks(
    *,
    teacher_gpu_ids: str | None = None,
    rollout_gpu_ids: str | None = None,
    trainer_gpu_ids: str | None = None,
    tid_base: int = 100,
) -> list[GPUTraceTrack]:
    """Build GPU trace tracks aligned with trace stage naming."""
    role_specs = [
        ("teacher_score", teacher_gpu_ids, "teacher_score.gpu"),
        ("generate", rollout_gpu_ids, "generate.gpu"),
        ("train", trainer_gpu_ids, "train.gpu"),
    ]
    tracks = []
    tid = tid_base
    for role, raw_ids, prefix in role_specs:
        for gpu_id in parse_gpu_ids(raw_ids):
            tracks.append(
                GPUTraceTrack(
                    role=role,
                    gpu_id=gpu_id,
                    tid=tid,
                    counter_name=f"{prefix}:{gpu_id}",
                )
            )
            tid += 1
    return tracks


class GPUMetricsSampler:
    """Periodically sample GPU util/power/memory and write trace counters."""

    def __init__(
        self,
        tracer,
        tracks: list[GPUTraceTrack],
        *,
        interval_sec: float = 1.0,
        nvml_module=_NVML_AUTO,
    ):
        self.tracer = tracer
        self.tracks = list(tracks)
        self.interval_sec = max(float(interval_sec), 0.1)
        self._nvml = _load_pynvml() if nvml_module is _NVML_AUTO else nvml_module
        self._thread = None
        self._stop_event = threading.Event()
        self._warned = False
        self._nvml_initialized = False

    @property
    def enabled(self) -> bool:
        return bool(self.tracks) and self._nvml is not None

    def start(self) -> bool:
        """Start the background sampler thread."""
        if not self.enabled or self._thread is not None:
            return False
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-trace-sampler",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the sampler thread."""
        if self._thread is None:
            self._shutdown_nvml()
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        self._shutdown_nvml()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            self._stop_event.wait(self.interval_sec)

    def _init_nvml(self) -> bool:
        if self._nvml is None:
            return False
        if self._nvml_initialized:
            return True
        self._nvml.nvmlInit()
        self._nvml_initialized = True
        return True

    def _shutdown_nvml(self) -> None:
        if self._nvml is None or not self._nvml_initialized:
            return
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
        self._nvml_initialized = False

    def _query_metrics_nvml(self) -> dict[int, dict[str, float]]:
        self._init_nvml()
        parsed = {}
        for track in self.tracks:
            gpu_id = track.gpu_id
            if gpu_id in parsed:
                continue
            handle = self._nvml.nvmlDeviceGetHandleByIndex(gpu_id)
            values = {}

            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = getattr(util, "gpu", None)
                if gpu_util is not None:
                    values["util_%"] = float(gpu_util)
            except Exception:
                pass

            try:
                power_mw = self._nvml.nvmlDeviceGetPowerUsage(handle)
                values["power_W"] = float(power_mw) / 1000.0
            except Exception:
                pass

            try:
                mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = getattr(mem, "used", None)
                if mem_used is not None:
                    values["mem_used_MiB"] = float(mem_used) / (1024.0 * 1024.0)
            except Exception:
                pass

            if values:
                parsed[gpu_id] = values
        return parsed

    def _query_metrics(self) -> dict[int, dict[str, float]]:
        return self._query_metrics_nvml()

    def sample_once(self, t: float | None = None) -> None:
        """Sample once and record counters for all configured GPU tracks."""
        if not self.enabled:
            return
        try:
            metrics_by_gpu = self._query_metrics()
        except Exception as e:
            if not self._warned:
                print(f"[GPUTrace] Warning: disabling GPU metrics sampling: {e}", flush=True)
                self._warned = True
            return

        sample_t = t if t is not None else time.monotonic()
        for track in self.tracks:
            values = metrics_by_gpu.get(track.gpu_id)
            if values:
                self.tracer.counter(
                    track.counter_name,
                    values,
                    tid=track.tid,
                    t=sample_t,
                )


def gpu_trace_interval_from_env(default: float = 1.0) -> float:
    """Read GPU trace sampling interval from env, tolerating bad values."""
    raw = os.environ.get("OPD_TRACE_GPU_INTERVAL_SEC")
    if raw is None:
        return default
    try:
        return max(float(raw), 0.1)
    except ValueError:
        return default

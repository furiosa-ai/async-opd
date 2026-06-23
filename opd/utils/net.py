"""Network and process utilities."""

import atexit
import hashlib
import json
import os
import signal
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import fcntl


# Port range for our services (ZMQ, vLLM/Torch distributed init, NCCL weight
# transfer, FSDP master). Keep this outside Linux's common ephemeral range
# (32768-60999) so outbound client sockets cannot race with our leased service
# ports between the bindability probe and the later server bind.
#
# Operators can override this on unusual clusters with OPD_PORT_MIN/OPD_PORT_MAX.
_PORT_MIN_ENV = "OPD_PORT_MIN"
_PORT_MAX_ENV = "OPD_PORT_MAX"
_PORT_MIN = 20000
_PORT_MAX = 29999
_REGISTRY_DIR_ENV = "OPD_PORT_LEASE_DIR"
_REGISTRY_FILENAME = "registry.json"
_LOCK_FILENAME = "registry.lock"
_DEFAULT_PURPOSE = "repo.auto"

_sysrand = None
_process_leases = {}


@dataclass(frozen=True)
class PortLease:
    """Metadata describing a leased port owned by the current process."""

    port: int
    lease_id: str
    pid: int
    purpose: str
    acquired_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


def _lease_root():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    repo_slug = os.path.basename(repo_root) or "repo"
    repo_hash = hashlib.sha1(repo_root.encode()).hexdigest()[:12]
    uid = getattr(os, "getuid", lambda: "nouid")()
    cache_root = os.environ.get(
        _REGISTRY_DIR_ENV,
        os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
    )
    root = os.environ.get(
        _REGISTRY_DIR_ENV,
        os.path.join(cache_root, f"opd-port-leases-{uid}", f"{repo_slug}-{repo_hash}"),
    )
    os.makedirs(root, mode=0o700, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return root


def _registry_path():
    return os.path.join(_lease_root(), _REGISTRY_FILENAME)


def _lock_path():
    return os.path.join(_lease_root(), _LOCK_FILENAME)


def _ensure_private_file(path):
    if not os.path.exists(path):
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _pid_is_alive(pid):
    if not pid or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _load_registry_unlocked():
    path = _registry_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {int(port): record for port, record in data.items()}


def _save_registry_unlocked(registry):
    path = _registry_path()
    tmp_path = f"{path}.tmp"
    serializable = {str(port): record for port, record in registry.items()}
    with open(tmp_path, "w") as f:
        json.dump(serializable, f, sort_keys=True)
    os.replace(tmp_path, path)
    _ensure_private_file(path)


@contextmanager
def _locked_registry():
    with open(_lock_path(), "a+") as lock_file:
        _ensure_private_file(lock_file.name)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        registry = _load_registry_unlocked()
        try:
            yield registry
            _save_registry_unlocked(registry)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _port_is_bindable(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
        return True
    except OSError:
        return False


def _log_port_event(action, *, port, lease_id=None, pid=None, purpose=None, note=None):
    details = [f"port={port}"]
    if lease_id:
        details.append(f"lease_id={lease_id}")
    if pid:
        details.append(f"pid={pid}")
    if purpose:
        details.append(f"purpose={purpose}")
    if note:
        details.append(f"note={note}")
    print(f"[PortLease] {action} " + " ".join(details), flush=True)


def _reclaim_stale_leases(registry):
    reclaimed = []
    for port, record in list(registry.items()):
        pid = int(record.get("pid", 0))
        if _pid_is_alive(pid):
            continue
        reclaimed.append((port, record))
        del registry[port]
    for port, record in reclaimed:
        _log_port_event(
            "reclaim",
            port=port,
            lease_id=record.get("lease_id"),
            pid=record.get("pid"),
            purpose=record.get("purpose"),
            note="stale_pid",
        )


def _managed_port_range():
    try:
        port_min = int(os.environ.get(_PORT_MIN_ENV, _PORT_MIN))
        port_max = int(os.environ.get(_PORT_MAX_ENV, _PORT_MAX))
    except ValueError as exc:
        raise RuntimeError(
            f"{_PORT_MIN_ENV}/{_PORT_MAX_ENV} must be integer ports") from exc
    if port_min < 1 or port_max > 65535 or port_min > port_max:
        raise RuntimeError(
            f"Invalid managed port range {port_min}-{port_max}; expected "
            "1 <= min <= max <= 65535")
    return port_min, port_max


def _choose_managed_port(registry):
    port_min, port_max = _managed_port_range()
    for port in range(port_min, port_max + 1):
        if port not in registry and _port_is_bindable(port):
            return port
    raise RuntimeError(
        f"Could not find a free managed port in range {port_min}-{port_max}")


def acquire_port_lease(purpose=_DEFAULT_PURPOSE, *, preferred_port=None, metadata=None,
                       owner_pid=None):
    """Lease a port from the shared repo-local registry."""
    global _sysrand
    if _sysrand is None:
        import random
        _sysrand = random.SystemRandom()

    owner_pid = owner_pid or os.getpid()
    metadata = metadata or {}

    with _locked_registry() as registry:
        _reclaim_stale_leases(registry)

        if preferred_port is not None:
            if preferred_port in registry:
                raise RuntimeError(f"Port {preferred_port} is already leased")
            if not _port_is_bindable(preferred_port):
                raise RuntimeError(f"Port {preferred_port} is not bindable")
            port = preferred_port
        else:
            port_min, port_max = _managed_port_range()
            port = None
            for _ in range(100):
                candidate = _sysrand.randint(port_min, port_max)
                if candidate in registry:
                    continue
                if not _port_is_bindable(candidate):
                    continue
                port = candidate
                break
            if port is None:
                port = _choose_managed_port(registry)

        lease = PortLease(
            port=port,
            lease_id=uuid.uuid4().hex,
            pid=owner_pid,
            purpose=purpose,
            metadata=dict(metadata),
        )
        registry[port] = asdict(lease)

    _process_leases[port] = lease
    _log_port_event(
        "acquire",
        port=lease.port,
        lease_id=lease.lease_id,
        pid=lease.pid,
        purpose=lease.purpose,
    )
    return lease


def release_port_lease(port=None, *, lease=None):
    """Release a locally-owned port lease. Missing leases are ignored."""
    if lease is None:
        if port is None:
            raise ValueError("release_port_lease requires port or lease")
        lease = _process_leases.get(port)
        if lease is None:
            return False
    else:
        port = lease.port

    with _locked_registry() as registry:
        _reclaim_stale_leases(registry)
        record = registry.get(port)
        if not record:
            _process_leases.pop(port, None)
            return False
        if record.get("lease_id") != lease.lease_id:
            _process_leases.pop(port, None)
            return False
        del registry[port]

    _process_leases.pop(port, None)
    _log_port_event(
        "release",
        port=lease.port,
        lease_id=lease.lease_id,
        pid=lease.pid,
        purpose=lease.purpose,
    )
    return True


def release_all_port_leases():
    """Release every lease owned by the current process."""
    for port in list(_process_leases):
        release_port_lease(port)


@contextmanager
def leased_port(purpose=_DEFAULT_PURPOSE, *, preferred_port=None, metadata=None,
                owner_pid=None):
    """Context manager that acquires and releases a shared port lease."""
    lease = acquire_port_lease(
        purpose,
        preferred_port=preferred_port,
        metadata=metadata,
        owner_pid=owner_pid,
    )
    try:
        yield lease
    finally:
        release_port_lease(lease=lease)


def find_free_port(purpose=_DEFAULT_PURPOSE):
    """Backwards-compatible shared allocator wrapper returning only the port."""
    return acquire_port_lease(purpose).port


def port_is_listening(port, host=None, timeout=1.0):
    """Check if a port is accepting connections."""
    try:
        with socket.create_connection((host or "127.0.0.1", port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def kill_tree(pid):
    """Kill a process and all its descendants (e.g. vLLM EngineCore children)."""
    # Collect child PIDs from /proc before killing the parent
    children = []
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            children = [int(c) for c in f.read().split()]
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    # Recurse into children first
    for cpid in children:
        kill_tree(cpid)
    # Kill this process
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


atexit.register(release_all_port_leases)

"""Perfetto / Chrome trace event recorder.

Produces JSON in Chrome Trace Event Format (Array of events).
Open with https://ui.perfetto.dev/ or chrome://tracing.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from errno import EDQUOT, ENOSPC


@contextmanager
def timer():
    """Context manager that records monotonic start/end/elapsed into a dict.

    Usage:
        with timer() as t:
            do_work()
        print(t["elapsed"], t["mono_start"], t["mono_end"])
    """
    t = {}
    t["mono_start"] = time.monotonic()
    yield t
    t["mono_end"] = time.monotonic()
    t["elapsed"] = t["mono_end"] - t["mono_start"]


class SpanResult:
    """Yielded by span() — collects args and elapsed time."""
    __slots__ = ("args", "elapsed")

    def __init__(self):
        self.args = {}
        self.elapsed = 0.0

    def __setitem__(self, key, value):
        self.args[key] = value


class Tracer:
    """Lightweight tracer that records duration events."""

    def __init__(self, stream_path=None, resume=False):
        self.events = []
        self._t0 = time.monotonic()
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._stream = None
        self._stream_path = stream_path
        self._seen_tids = set()
        self._disk_stream_disabled = False
        if stream_path:
            os.makedirs(os.path.dirname(stream_path) or ".", exist_ok=True)
            if resume and os.path.exists(stream_path):
                # Load existing events and continue appending
                self._load_existing(stream_path)
                self._seen_tids = {e["tid"] for e in self.events if "tid" in e}
                self._stream = open(stream_path, "w")
                self._stream.write("[\n")
                # Re-write all existing events
                for i, evt in enumerate(self.events):
                    prefix = ",\n" if i > 0 else ""
                    self._stream.write(prefix + json.dumps(evt))
                self._stream.flush()
                self._stream_count = len(self.events)
            else:
                self._stream = open(stream_path, "w")
                self._stream.write("[\n")  # Perfetto tolerates missing closing ]
                self._stream.flush()
                self._stream_count = 0

    def _load_existing(self, path):
        """Load events from an existing trace_live.json (may be incomplete)."""
        try:
            with open(path) as f:
                content = f.read().rstrip().rstrip(",")
                if not content.endswith("]"):
                    content += "\n]"
                self.events = json.loads(content)
                # Adjust _t0 so new events have timestamps relative to the original start
                if self.events:
                    earliest_ts_us = min(e.get("ts", 0) for e in self.events)
                    # _t0 = current_mono - (earliest_ts / 1e6) would make ts=0 map to original start
                    # But we want new events to continue from where the old ones ended
                    latest_end_us = max(e.get("ts", 0) + e.get("dur", 0) for e in self.events)
                    self._t0 = time.monotonic() - latest_end_us / 1_000_000
        except Exception as e:
            print(f"[Tracer] Warning: failed to load existing trace: {e}", flush=True)
            self.events = []

    def _us(self, t):
        """Convert monotonic timestamp to microseconds relative to start."""
        return int((t - self._t0) * 1_000_000)

    def _record(self, evt):
        """Append event to list and optionally stream to disk as Perfetto JSON.

        Thread-safe: all writes are serialized under self._lock so concurrent
        threads (rollout collector, teacher loop, train loop) cannot interleave.
        """
        with self._lock:
            self.events.append(evt)
            if self._stream and not self._disk_stream_disabled:
                # Emit thread_name metadata on first event for each tid
                try:
                    tid = evt.get("tid")
                    if tid is not None and tid not in self._seen_tids:
                        self._seen_tids.add(tid)
                        meta = {
                            "name": "thread_name", "ph": "M",
                            "pid": evt["pid"], "tid": tid,
                            "args": {"name": evt.get("cat") or evt.get("name", "")},
                        }
                        prefix = ",\n" if self._stream_count > 0 else ""
                        self._stream.write(prefix + json.dumps(meta))
                        self._stream_count += 1
                    prefix = ",\n" if self._stream_count > 0 else ""
                    self._stream.write(prefix + json.dumps(evt))
                    self._stream.flush()
                    self._stream_count += 1
                except OSError as e:
                    if e.errno in (ENOSPC, EDQUOT):
                        print(
                            f"[Tracer] Disabling live trace stream after quota/space error on {self._stream_path}: {e}",
                            flush=True,
                        )
                        self._disk_stream_disabled = True
                        try:
                            try:
                                self._stream.close()
                            except OSError:
                                pass
                        finally:
                            self._stream = None
                    else:
                        raise

    @contextmanager
    def span(self, name, cat="", tid=0):
        """Context manager that records a duration event.

        Usage:
            with tr.span("generate", cat="rollout", tid=1) as s:
                do_work()
                s["key"] = "value"  # optional args
            print(s.elapsed)  # seconds as float
        """
        result = SpanResult()
        t_start = time.monotonic()
        yield result
        t_end = time.monotonic()
        result.elapsed = t_end - t_start
        evt = {
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": self._us(t_start),
            "dur": self._us(t_end) - self._us(t_start),
            "pid": self._pid,
            "tid": tid,
        }
        if result.args:
            evt["args"] = result.args
        self._record(evt)

    def counter(self, name, values, tid=0, t=None):
        """Record a counter event (line chart in Perfetto).

        Args:
            name: counter series name (e.g. "pipeline")
            values: dict of {series_name: numeric_value}
            tid: thread id for the counter track
            t: monotonic timestamp (default: now)
        """
        evt = {
            "name": name,
            "ph": "C",
            "ts": self._us(t if t is not None else time.monotonic()),
            "pid": self._pid,
            "tid": tid,
            "args": values,
        }
        self._record(evt)

    def instant(self, name, cat="", tid=0, args=None):
        """Record an instant event (vertical marker in Perfetto)."""
        evt = {
            "name": name,
            "cat": cat,
            "ph": "i",
            "s": "t",  # thread scope
            "ts": self._us(time.monotonic()),
            "pid": self._pid,
            "tid": tid,
        }
        if args:
            evt["args"] = args
        self._record(evt)

    def emit(self, name, cat="", tid=0, t_start=None, t_end=None, args=None):
        """Record a completed span from a known start time (monotonic).

        Use for async spans that can't use `with` (e.g. cross-yield).
        If t_end is None, uses current time.
        """
        if t_end is None:
            t_end = time.monotonic()
        evt = {
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": self._us(t_start),
            "dur": self._us(t_end) - self._us(t_start),
            "pid": self._pid,
            "tid": tid,
        }
        if args:
            evt["args"] = args
        self._record(evt)

    def save(self, path):
        """Write trace to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._lock:
            events = list(self.events)
        # Add thread name metadata so Perfetto shows nice lane labels
        seen_tids = {e["tid"] for e in events}
        meta = []
        for tid in sorted(seen_tids):
            # Use the cat of the first event on this tid as the thread name
            for e in events:
                if e["tid"] == tid:
                    meta.append({
                        "name": "thread_name", "ph": "M",
                        "pid": self._pid, "tid": tid,
                        "args": {"name": e.get("cat") or e.get("name", "unknown")},
                    })
                    break
        try:
            with open(path, "w") as f:
                json.dump(meta + events, f)
            print(f"[Trace] Saved {len(events)} events to {path}", flush=True)
        except OSError as e:
            if e.errno in (ENOSPC, EDQUOT):
                print(
                    f"[Trace] Skipping trace save due to quota/space error on {path}: {e}",
                    flush=True,
                )
                return
            raise

"""Thread-safe FIFO queue with optional staleness tracking."""

import collections
import queue
import threading
import time


class StalenessQueue:
    """FIFO queue. Items are (weight_version, payload) pairs stored in a deque.

    No eviction — all items are returned in FIFO order regardless of staleness.
    weight_version is stored for logging/metrics purposes only.

    Thread-safe: all operations hold a single lock.
    """

    def __init__(self, version_ref, staleness_threshold):
        """
        Args:
            version_ref: mutable list [int] — current weight version (shared).
            staleness_threshold: max allowed staleness (kept for logging/metrics).
        """
        self._version_ref = version_ref
        self._threshold = staleness_threshold
        self._deque = collections.deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, item, weight_version=0):
        """Append item to the FIFO queue (never blocks).

        Returns None (kept for API compat with old FreshnessQueue).
        """
        with self._lock:
            self._deque.append((weight_version, item))
            self._not_empty.notify()
            return None

    def get(self, timeout=None):
        """Get the next FIFO item. Blocks if empty."""
        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout
        while True:
            with self._not_empty:
                if self._deque:
                    _, item = self._deque.popleft()
                    return item
                # Empty — wait for new items
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty()
                self._not_empty.wait(timeout=remaining)

    def qsize(self):
        with self._lock:
            return len(self._deque)

    def empty(self):
        with self._lock:
            return len(self._deque) == 0

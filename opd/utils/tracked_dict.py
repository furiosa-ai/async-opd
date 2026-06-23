"""TrackedDict — dict wrapper that tracks which keys are accessed.

Used to verify that config builders produce exactly the keys consumed
by __init__ methods, catching stale or missing config fields.
"""


class TrackedDict(dict):
    """A dict that records which keys have been read via [] or .get().

    Nested dicts are automatically wrapped so that access tracking is
    recursive.  Call ``unaccessed()`` to get the set of dot-path keys
    that were never read.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data or {}, **kwargs)
        self._accessed = set()
        # Recursively wrap nested dicts
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, TrackedDict):
                super().__setitem__(k, TrackedDict(v))

    def __getitem__(self, key):
        self._accessed.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._accessed.add(key)
        return super().get(key, default)

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, TrackedDict):
            value = TrackedDict(value)
        super().__setitem__(key, value)

    def unaccessed(self, prefix=""):
        """Return set of dot-path keys that were never read.

        Recurses into nested TrackedDicts.  A parent key access (e.g.
        ``config["algorithm"]``) does NOT count as accessing the children —
        only leaf-level reads matter.
        """
        result = set()
        for k in self:
            path = f"{prefix}.{k}" if prefix else str(k)
            v = super().__getitem__(k)  # bypass tracking
            if isinstance(v, TrackedDict):
                # If the nested dict itself was accessed (someone did config["algo"]),
                # still check its children for unaccessed leaves.
                result |= v.unaccessed(path)
            else:
                if k not in self._accessed:
                    result.add(path)
        return result

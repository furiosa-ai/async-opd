"""AsyncOPD public package boundary.

The stable import surface is intentionally small.  External users should treat
``opd.create_coordinator`` and the documented CLI/config files as the supported
entry points.  Most subpackages remain internal implementation details and may
change between releases.

``create_coordinator`` is resolved lazily so ``import opd`` stays lightweight
and does not import the training stack or optional GPU/runtime dependencies.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__ = ("create_coordinator",)


def __getattr__(name: str) -> Any:
    if name == "create_coordinator":
        from opd.coordinator.factory import create_coordinator

        globals()[name] = create_coordinator
        return create_coordinator
    raise AttributeError(f"module 'opd' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])

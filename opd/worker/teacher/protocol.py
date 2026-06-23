"""Typed Protocol for teacher backends."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class TeacherBackend(Protocol):
    """Core teacher backend interface."""

    def score_batch(self, prompt_token_ids: list, batch_size: int) -> dict: ...
    def shutdown(self) -> None: ...

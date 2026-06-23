"""Typed Protocols for rollout backends.

Backends implement these Protocols to declare their capabilities.
The coordinator checks capabilities at startup and errors on
unsupported combinations.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class RolloutBackend(Protocol):
    """Core rollout backend interface. All backends implement this."""

    supports_streaming: bool
    supports_weight_transfer_nccl: bool
    supports_pause_resume: bool
    supports_scoring: bool
    supports_lora: bool

    def generate(self, batch: dict, sampling_config: dict) -> dict: ...
    def init_weight_transfer(self, init_info: dict) -> dict: ...
    def sync_weights(self, update_info: dict) -> dict: ...
    def get_params_info(self) -> list: ...
    def compute_checksum(self) -> float: ...
    def shutdown(self) -> None: ...


@runtime_checkable
class StreamingBackend(Protocol):
    """Optional streaming/autonomous mode capability.

    Streaming workers own their own async dispatch loop — the base class
    does NOT dispatch these commands. This accurately models the streaming
    worker's stateful mode transitions (idle -> autonomous -> paused -> syncing).
    """

    def enter_autonomous(self, batch: dict) -> None: ...
    def exit_autonomous(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...


@runtime_checkable
class ScoringBackend(Protocol):
    """Optional scoring capability (OPSD self-distillation)."""

    def score(self, batch: dict) -> dict: ...


@runtime_checkable
class CollectiveWeightSync(Protocol):
    """Optional Ray collective weight sync."""

    def sync_weights_collective(self, update_info: dict) -> dict: ...


@runtime_checkable
class LoRABackend(Protocol):
    """Optional LoRA checksum capability."""

    def compute_lora_checksum(self) -> float: ...

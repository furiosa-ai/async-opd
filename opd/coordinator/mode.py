"""CoordinatorMode protocol — interface between mode (what) and scheduler (when/how).

Formalizes the backend adapter SimpleNamespace pattern used by
StepOffCoordinator via build_step_off_backend().

A mode defines:
  - What data to load and iterate over
  - How to generate (rollout configuration)
  - How to score (teacher ZMQ, reward functions, etc.)
  - How to assemble and send training batches
  - How to log mode-specific metrics
  - Which processes are needed (teacher, rollout, trainer)

A scheduler defines:
  - How to overlap generation, scoring, and training in time
  - When to sync weights, evaluate, and checkpoint

The scheduler consumes mode methods via a backend adapter built by
build_step_off_backend() or equivalent.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Protocol, runtime_checkable
from types import SimpleNamespace


@runtime_checkable
class CoordinatorMode(Protocol):
    """Protocol defining the mode interface for coordinator composition.

    Each method corresponds to a slot in the StepOffScheduler backend
    SimpleNamespace. Streaming stages also consume these methods (after
    Phase 3.5 alignment).
    """

    # --- Data ---

    def data_iterator(self) -> Iterator[tuple[int, dict]]:
        """Yield (epoch, batch_dict) pairs for training."""
        ...

    # --- Generation ---

    def async_generate(self, batch_dict: dict) -> None:
        """Submit batch to rollout for generation (non-blocking)."""
        ...

    def wait_generate(self) -> dict:
        """Collect generation result. Returns gen_output dict."""
        ...

    # --- Scoring ---

    def async_teacher(self, gen_output: dict) -> Any:
        """Submit gen_output for scoring. Returns future-like object.

        OPD: ZMQ teacher scoring with background thread resolution.
        GRPO: reward computation + optional reference model scoring.
        """
        ...

    def resolve_teacher(self, future: Any, timing: dict,
                        batch: int | None = None) -> dict | None:
        """Resolve scoring future. Returns teacher_output or None on failure."""
        ...

    # --- Training ---

    def async_train(self, gen_output: dict, teacher_output: dict) -> None:
        """Submit training batch to trainer subprocess (non-blocking)."""
        ...

    def wait_train(self) -> dict:
        """Collect training result from trainer subprocess."""
        ...

    # --- Logging ---

    def log_train_step(self, step: int, timing: dict, gen_out: dict,
                       result: dict) -> None:
        """Log a completed training step with mode-specific metrics."""
        ...

    # --- Lifecycle queries ---

    def needs_teacher(self) -> bool:
        """Whether this mode requires a teacher process."""
        ...

    def needs_rollout(self) -> bool:
        """Whether this mode requires rollout worker(s)."""
        ...

    def get_trainer_fn(self) -> Callable:
        """Return trainer_entry_point for process spawning."""
        ...

    # --- Streaming support ---

    def make_stream_score_fn(self, teacher_client: Any) -> Callable:
        """Return a score_fn callable for streaming TeacherScorer.

        OPD:  wraps teacher_client ZMQ scoring
        GRPO: CPU reward computation from ground_truths
        """
        ...

    def make_stream_assemble_fn(self, max_response_length: int) -> Callable:
        """Return an assemble_batch_fn callable for streaming TrainDispatcher.

        OPD:  split_gen_teacher (pad teacher logprobs)
        GRPO: group by prompt_group_id, compute advantages
        """
        ...

    @property
    def stream_batch_multiplier(self) -> int:
        """Multiplier for streaming batch size. 1 for OPD, G for GRPO."""
        ...


def build_step_off_backend(mode: CoordinatorMode, coordinator) -> SimpleNamespace:
    """Build StepOffScheduler backend from mode + coordinator infrastructure.

    Mode provides the 8 pipeline operation methods (data, generate, score,
    train, log). Coordinator provides the 3 infrastructure methods
    (sync_weights, evaluate, save_checkpoint) which are orthogonal to mode.

    This replaces the inline SimpleNamespace construction previously
    duplicated across coordinator run() methods.
    """
    backend = dict(
        async_generate=mode.async_generate,
        wait_generate=mode.wait_generate,
        async_teacher=mode.async_teacher,
        resolve_teacher=mode.resolve_teacher,
        async_train=mode.async_train,
        wait_train=mode.wait_train,
        sync_weights=coordinator._sync_weights,
        evaluate=coordinator._evaluate,
        log_train_step=mode.log_train_step,
        save_checkpoint=coordinator._save_checkpoint,
        wait_checkpoint_save=coordinator._wait_checkpoint_save,
    )
    has_direct_teacher_api = (
        getattr(type(mode), "async_train_direct_teacher_output", None) is not None
        or "async_train_direct_teacher_output" in getattr(mode, "__dict__", {})
    )
    if has_direct_teacher_api:
        backend["async_train_direct_teacher_output"] = (
            getattr(mode, "async_train_direct_teacher_output", None)
            if getattr(mode, "uses_direct_teacher_artifacts", False) is True
            else None
        )
    return SimpleNamespace(**backend)

"""Factory entry point for the OPD pipeline.

All orchestration logic lives in:
  - opd/coordinator/base.py       (CoordinatorBase — shared infra)
  - opd/coordinator/step_off.py   (StepOffCoordinator — n-step-off scheduling)
  - opd/coordinator/streaming.py  (StreamCoordinator — streaming pipeline)
  - opd/coordinator/sft.py        (SFTCoordinator — supervised fine-tuning)

Mode-specific logic (OPD, GRPO, OPSD, SFT) lives in mode classes:
  - opd/coordinator/opd_mode.py   (OPDMode — on-policy distillation)
  - opd/coordinator/grpo_mode.py  (GRPOMode — GRPO/DAPO training)
  - opd/coordinator/opd_mode.py   (OPSDMode — on-policy self-distillation)
  - opd/coordinator/sft.py        (SFTMode — supervised fine-tuning)

This module provides create_coordinator() that dispatches
to the appropriate coordinator + mode based on OPDConfig.
"""

from opd.coordinator.base import CoordinatorBase  # noqa: F401
from opd.coordinator.step_off import StepOffCoordinator  # noqa: F401
from opd.coordinator.step_off_async import StepOffAsyncCoordinator  # noqa: F401
from opd.coordinator.streaming import StreamCoordinator  # noqa: F401
from opd.coordinator.sft import SFTCoordinator  # noqa: F401
from opd.coordinator.fused_hybrid_sync import FusedHybridOPDMode, FusedHybridSyncCoordinator  # noqa: F401
from opd.coordinator.opd_mode import OPDMode, OPSDMode
from opd.coordinator.grpo_mode import GRPOMode
from opd.coordinator.sft import SFTMode
from opd.utils.config import OPDConfig, uses_step_off_streaming


def _get_mode_cls(oc: OPDConfig):
    """Determine the mode class from OPDConfig."""
    mode = oc.algorithm.mode
    if mode == "grpo":
        return GRPOMode
    if mode == "sft":
        return SFTMode
    if mode == "opsd":
        return OPSDMode
    # Auto-detect SFT: no teacher configured
    if oc.teacher is None or not oc.teacher.path:
        return SFTMode
    return OPDMode


def create_coordinator(config, *, opd_config=None, **kwargs) -> CoordinatorBase:
    """Create the appropriate coordinator for a validated OPDConfig.

    Dispatches to SFTCoordinator, StreamCoordinator, or StepOffCoordinator
    based on training mode and scheduling config. Mode-specific logic
    (OPD, GRPO, OPSD) is handled by mode_cls passed to the coordinator.

    Args:
        config: An OPDConfig dataclass.
        opd_config: Optional explicit OPDConfig override.
        **kwargs: Passed through to coordinator constructor (logger, run_dir).
    """
    oc = opd_config if opd_config is not None else config
    if not isinstance(oc, OPDConfig):
        raise TypeError(
            "create_coordinator() now requires an OPDConfig. "
            "Legacy nested dict configs are no longer supported."
        )

    mode_cls = _get_mode_cls(oc)
    mode = oc.algorithm.mode
    sched_mode = oc.pipeline.scheduling_mode
    step_off_streaming = uses_step_off_streaming(oc.pipeline)
    config_placeholder = {}

    if sched_mode == "fused_hybrid_sync":
        if mode != "opd":
            raise ValueError("fused_hybrid_sync supports OPD mode only")
        return FusedHybridSyncCoordinator(
            config_placeholder, mode_cls=FusedHybridOPDMode, opd_config=oc, **kwargs
        )

    if step_off_streaming:
        if sched_mode != "n_step_off":
            raise ValueError(
                "pipeline.n_step_off.implementation='streaming' is only supported "
                "with n_step_off scheduling"
            )
        if mode != "opd":
            raise ValueError(
                "pipeline.n_step_off.implementation='streaming' currently supports "
                "OPD mode only"
            )
        return StepOffAsyncCoordinator(config_placeholder, mode_cls=mode_cls,
                                       opd_config=oc, **kwargs)

    if mode == "grpo":
        if sched_mode == "fully_async":
            return StreamCoordinator(config_placeholder, mode_cls=mode_cls,
                                     opd_config=oc, **kwargs)
        return StepOffCoordinator(config_placeholder, mode_cls=mode_cls,
                                  opd_config=oc, **kwargs)
    if mode == "opsd":
        if sched_mode == "fully_async":
            raise NotImplementedError(
                "OPSD mode does not yet support fully_async scheduling. "
                "Use step-off mode (default)."
            )
        if oc.pipeline.n_step_off.step_off > 0:
            raise ValueError(
                f"OPSD mode requires step_off=0 (got {oc.pipeline.n_step_off.step_off}). "
                f"Scoring and generation share the same rollout worker, "
                f"so step_off>0 causes queue serialization delays."
            )
        return StepOffCoordinator(config_placeholder, mode_cls=mode_cls,
                                  opd_config=oc, **kwargs)
    if mode == "sft" or (oc.teacher is None or not oc.teacher.path):
        return SFTCoordinator(config_placeholder, mode_cls=mode_cls,
                              opd_config=oc, **kwargs)
    if sched_mode == "fully_async":
        return StreamCoordinator(config_placeholder, mode_cls=mode_cls,
                                 opd_config=oc, **kwargs)
    return StepOffCoordinator(config_placeholder, mode_cls=mode_cls,
                              opd_config=oc, **kwargs)

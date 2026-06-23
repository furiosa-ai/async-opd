"""Trainer package — backends (FSDP, Megatron) and mode trainers (OPD, SFT, GRPO)."""

from opd.trainer.base import BaseBackend, build_lr_scheduler, vllm_trainer_send
from opd.trainer.fsdp import FSDPBackend, fsdp_trainer_main
from opd.trainer.megatron import MegatronBackend, megatron_trainer_main
from opd.trainer.config import SFTConfig, GRPOConfig
from opd.trainer.ac_opd import ActorCriticOPDTrainer
from opd.trainer.opd import OPDTrainer
from opd.trainer.sft import SFTTrainer, sft_trainer_main
from opd.trainer.grpo import GRPOTrainer, grpo_trainer_main

__all__ = [
    # Backends
    "BaseBackend",
    "FSDPBackend",
    "MegatronBackend",
    # Mode trainers (composition)
    "OPDTrainer",
    "ActorCriticOPDTrainer",
    "SFTTrainer",
    "GRPOTrainer",
    # Config dataclasses
    "SFTConfig",
    "GRPOConfig",
    # Entry points
    "fsdp_trainer_main",
    "sft_trainer_main",
    "grpo_trainer_main",
    "megatron_trainer_main",
    # Utilities
    "build_lr_scheduler",
    "vllm_trainer_send",
]

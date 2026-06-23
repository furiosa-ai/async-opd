"""Typed launch-spec boundaries for trainer, rollout, and teacher workers.

These specs separate static role-owned launch inputs from runtime-owned fields
such as GPU placement, ports, and rank metadata. They are the canonical
boundary objects passed between the coordinator and child processes/actors.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Mapping

from opd.utils.config import (
    GRPOAlgorithmConfig,
    OPDAlgorithmConfig,
    RolloutConfig,
    TeacherConfig,
    TrainerConfig,
    VLLMRolloutConfig,
    VLLMTeacherConfig,
    WeightSyncConfig,
)


JSONDict = dict[str, Any]


_ROLLOUT_DEFAULTS = RolloutConfig()
_ROLLOUT_VLLM_DEFAULTS = VLLMRolloutConfig()
_TEACHER_DEFAULTS = TeacherConfig()
_TEACHER_VLLM_DEFAULTS = VLLMTeacherConfig()
_TRAINER_DEFAULTS = TrainerConfig()
_WEIGHT_SYNC_DEFAULTS = WeightSyncConfig()
_OPD_ALGO_DEFAULTS = OPDAlgorithmConfig()
_GRPO_ALGO_DEFAULTS = GRPOAlgorithmConfig()


def _copy(value: Any) -> Any:
    return deepcopy(value)


def resolve_teacher_n_logprobs(opd_config: Any) -> int:
    """Resolve teacher logprob width from the canonical config owner."""

    teacher = getattr(opd_config, "teacher", None)
    if teacher is not None and teacher.vllm.n_logprobs is not None:
        return teacher.vllm.n_logprobs
    algorithm = getattr(opd_config, "algorithm", None)
    if algorithm is not None and getattr(algorithm, "opd", None) is not None:
        return algorithm.opd.n_kl_logprobs
    return 256


@dataclass(frozen=True)
class ActorCriticLaunch:
    value_mode: str
    gae_lambda: float
    value_coef: float
    normalize_advantages: bool

    def to_payload(self) -> JSONDict:
        return {
            "pg_actor_critic": True,
            "pg_value_mode": self.value_mode,
            "pg_gae_lambda": self.gae_lambda,
            "pg_value_coef": self.value_coef,
            "pg_value_normalize_advantages": self.normalize_advantages,
        }


@dataclass(frozen=True)
class OPDTrainerAlgorithmLaunch:
    mode: str
    kl_loss_mode: str
    kl_skew_alpha: float
    pg_clip_eps: float
    use_importance_sampling: bool
    kl_token_clip: float
    pg_online_advantage: bool
    use_decoupled_loss: bool
    behave_imp_weight_cap: float
    pg_m2po_budget: float | None = None
    pg_m2po_miniclip_low: float = _OPD_ALGO_DEFAULTS.pg_m2po_miniclip_low
    pg_m2po_miniclip_high: float = _OPD_ALGO_DEFAULTS.pg_m2po_miniclip_high
    pg_token_weighted_backward: bool = _OPD_ALGO_DEFAULTS.pg_token_weighted_backward
    mof_variant: str = _OPD_ALGO_DEFAULTS.mof_variant
    mof_partition: str = _OPD_ALGO_DEFAULTS.mof_partition
    mof_eta_mass: float = _OPD_ALGO_DEFAULTS.mof_eta_mass
    mof_eta_odds: float = _OPD_ALGO_DEFAULTS.mof_eta_odds
    mof_lambda_odds: float = _OPD_ALGO_DEFAULTS.mof_lambda_odds
    mof_eps: float = _OPD_ALGO_DEFAULTS.mof_eps
    mof_deduplicate_candidates: bool = _OPD_ALGO_DEFAULTS.mof_deduplicate_candidates
    rollout_student_topk_k: int | None = None
    thunlp_loss_agg_mode: str | None = None
    actor_critic: ActorCriticLaunch | None = None

    def to_payload(self) -> JSONDict:
        payload = {
            "mode": self.mode,
            "kl_loss_mode": self.kl_loss_mode,
            "kl_skew_alpha": self.kl_skew_alpha,
            "pg_clip_eps": self.pg_clip_eps,
            "use_importance_sampling": self.use_importance_sampling,
            "kl_token_clip": self.kl_token_clip,
            "pg_online_advantage": self.pg_online_advantage,
            "use_decoupled_loss": self.use_decoupled_loss,
            "behave_imp_weight_cap": self.behave_imp_weight_cap,
            "pg_m2po_budget": self.pg_m2po_budget,
            "pg_m2po_miniclip_low": self.pg_m2po_miniclip_low,
            "pg_m2po_miniclip_high": self.pg_m2po_miniclip_high,
            "pg_token_weighted_backward": self.pg_token_weighted_backward,
            "mof_variant": self.mof_variant,
            "mof_partition": self.mof_partition,
            "mof_eta_mass": self.mof_eta_mass,
            "mof_eta_odds": self.mof_eta_odds,
            "mof_lambda_odds": self.mof_lambda_odds,
            "mof_eps": self.mof_eps,
            "mof_deduplicate_candidates": self.mof_deduplicate_candidates,
        }
        if self.rollout_student_topk_k is not None:
            payload["rollout_student_topk_k"] = self.rollout_student_topk_k
        if self.thunlp_loss_agg_mode is not None:
            payload["thunlp_loss_agg_mode"] = self.thunlp_loss_agg_mode
        if self.actor_critic is not None:
            payload.update(self.actor_critic.to_payload())
        return payload


@dataclass(frozen=True)
class SFTTrainerAlgorithmLaunch:
    mode: str
    kl_loss_mode: str
    kl_skew_alpha: float
    pg_clip_eps: float
    sft_loss_mode: str
    ce_alpha: float
    n_kl_logprobs: int

    def to_payload(self) -> JSONDict:
        return {
            "mode": self.mode,
            "kl_loss_mode": self.kl_loss_mode,
            "kl_skew_alpha": self.kl_skew_alpha,
            "pg_clip_eps": self.pg_clip_eps,
            "sft_loss_mode": self.sft_loss_mode,
            "ce_alpha": self.ce_alpha,
            "n_kl_logprobs": self.n_kl_logprobs,
        }


@dataclass(frozen=True)
class GRPOTrainerAlgorithmLaunch:
    mode: str
    kl_loss_mode: str
    kl_skew_alpha: float
    pg_clip_eps: float
    grpo_clip_eps: float
    grpo_kl_beta: float
    clip_ratio_low: float | None
    clip_ratio_high: float | None
    clip_ratio_c: float | None
    loss_agg_mode: str
    kl_type: str
    use_decoupled_loss: bool
    behave_imp_weight_cap: float

    def to_payload(self) -> JSONDict:
        return {
            "mode": self.mode,
            "kl_loss_mode": self.kl_loss_mode,
            "kl_skew_alpha": self.kl_skew_alpha,
            "pg_clip_eps": self.pg_clip_eps,
            "grpo_clip_eps": self.grpo_clip_eps,
            "grpo_kl_beta": self.grpo_kl_beta,
            "clip_ratio_low": self.clip_ratio_low,
            "clip_ratio_high": self.clip_ratio_high,
            "clip_ratio_c": self.clip_ratio_c,
            "loss_agg_mode": self.loss_agg_mode,
            "kl_type": self.kl_type,
            "use_decoupled_loss": self.use_decoupled_loss,
            "behave_imp_weight_cap": self.behave_imp_weight_cap,
        }


TrainerAlgorithmLaunch = (
    OPDTrainerAlgorithmLaunch
    | SFTTrainerAlgorithmLaunch
    | GRPOTrainerAlgorithmLaunch
)


def _is_typed_trainer_algorithm_launch(value: Any) -> bool:
    return isinstance(
        value,
        (
            OPDTrainerAlgorithmLaunch,
            SFTTrainerAlgorithmLaunch,
            GRPOTrainerAlgorithmLaunch,
        ),
    )


def build_trainer_algorithm_launch(algorithm_cfg: Any) -> TrainerAlgorithmLaunch:
    """Build the typed trainer launch payload from canonical or legacy inputs."""

    if _is_typed_trainer_algorithm_launch(algorithm_cfg):
        return algorithm_cfg

    if isinstance(algorithm_cfg, Mapping):
        mode = algorithm_cfg["mode"]
        common = dict(
            mode=mode,
            kl_loss_mode=algorithm_cfg.get("kl_loss_mode", _OPD_ALGO_DEFAULTS.kl_loss_mode),
            kl_skew_alpha=float(algorithm_cfg.get("kl_skew_alpha", _OPD_ALGO_DEFAULTS.skewed_alpha)),
            pg_clip_eps=float(algorithm_cfg.get("pg_clip_eps", _OPD_ALGO_DEFAULTS.pg_clip_eps)),
        )
        if mode in ("opd", "opsd"):
            actor_critic = None
            actor_critic_enabled = bool(
                algorithm_cfg.get(
                    "pg_actor_critic",
                    "pg_value_mode" in algorithm_cfg
                    or "pg_gae_lambda" in algorithm_cfg
                    or "pg_value_coef" in algorithm_cfg
                    or "pg_value_normalize_advantages" in algorithm_cfg,
                )
            )
            if actor_critic_enabled:
                actor_critic = ActorCriticLaunch(
                    value_mode=algorithm_cfg.get("pg_value_mode", _OPD_ALGO_DEFAULTS.pg_value_mode),
                    gae_lambda=float(algorithm_cfg.get("pg_gae_lambda", _OPD_ALGO_DEFAULTS.pg_gae_lambda)),
                    value_coef=float(algorithm_cfg.get("pg_value_coef", _OPD_ALGO_DEFAULTS.pg_value_coef)),
                    normalize_advantages=bool(
                        algorithm_cfg.get(
                            "pg_value_normalize_advantages",
                            _OPD_ALGO_DEFAULTS.pg_value_normalize_advantages,
                        )
                    ),
                )
            return OPDTrainerAlgorithmLaunch(
                **common,
                use_importance_sampling=bool(
                    algorithm_cfg.get(
                        "use_importance_sampling",
                        _OPD_ALGO_DEFAULTS.use_importance_sampling,
                    )
                ),
                kl_token_clip=float(algorithm_cfg.get("kl_token_clip", _OPD_ALGO_DEFAULTS.kl_token_clip)),
                pg_online_advantage=bool(
                    algorithm_cfg.get("pg_online_advantage", _OPD_ALGO_DEFAULTS.pg_online_advantage)
                ),
                use_decoupled_loss=bool(
                    algorithm_cfg.get("use_decoupled_loss", _OPD_ALGO_DEFAULTS.use_decoupled_loss)
                ),
                behave_imp_weight_cap=float(
                    algorithm_cfg.get("behave_imp_weight_cap", _OPD_ALGO_DEFAULTS.behave_imp_weight_cap)
                ),
                pg_m2po_budget=algorithm_cfg.get("pg_m2po_budget"),
                pg_m2po_miniclip_low=float(
                    algorithm_cfg.get("pg_m2po_miniclip_low", _OPD_ALGO_DEFAULTS.pg_m2po_miniclip_low)
                ),
                pg_m2po_miniclip_high=float(
                    algorithm_cfg.get("pg_m2po_miniclip_high", _OPD_ALGO_DEFAULTS.pg_m2po_miniclip_high)
                ),
                pg_token_weighted_backward=bool(
                    algorithm_cfg.get(
                        "pg_token_weighted_backward",
                        _OPD_ALGO_DEFAULTS.pg_token_weighted_backward,
                    )
                ),
                mof_variant=algorithm_cfg.get("mof_variant", _OPD_ALGO_DEFAULTS.mof_variant),
                mof_partition=algorithm_cfg.get("mof_partition", _OPD_ALGO_DEFAULTS.mof_partition),
                mof_eta_mass=float(algorithm_cfg.get("mof_eta_mass", _OPD_ALGO_DEFAULTS.mof_eta_mass)),
                mof_eta_odds=float(algorithm_cfg.get("mof_eta_odds", _OPD_ALGO_DEFAULTS.mof_eta_odds)),
                mof_lambda_odds=float(
                    algorithm_cfg.get("mof_lambda_odds", _OPD_ALGO_DEFAULTS.mof_lambda_odds)
                ),
                mof_eps=float(algorithm_cfg.get("mof_eps", _OPD_ALGO_DEFAULTS.mof_eps)),
                mof_deduplicate_candidates=bool(
                    algorithm_cfg.get(
                        "mof_deduplicate_candidates",
                        _OPD_ALGO_DEFAULTS.mof_deduplicate_candidates,
                    )
                ),
                rollout_student_topk_k=(
                    int(algorithm_cfg["rollout_student_topk_k"])
                    if (
                        algorithm_cfg.get("rollout_student_topk_k") is not None
                        and algorithm_cfg.get("kl_loss_mode")
                        in {"reverse_kl_rollout_student_topk", "thunlp_opd_default_loss"}
                    )
                    else None
                ),
                thunlp_loss_agg_mode=(
                    algorithm_cfg.get("thunlp_loss_agg_mode")
                    if algorithm_cfg.get("thunlp_loss_agg_mode") is not None
                    else ("token-mean" if algorithm_cfg.get("kl_loss_mode") == "thunlp_opd_default_loss" else None)
                ),
                actor_critic=actor_critic,
            )
        if mode == "sft":
            return SFTTrainerAlgorithmLaunch(
                **common,
                sft_loss_mode=algorithm_cfg["sft_loss_mode"],
                ce_alpha=float(algorithm_cfg["ce_alpha"]),
                n_kl_logprobs=int(algorithm_cfg["n_kl_logprobs"]),
            )
        if mode == "grpo":
            return GRPOTrainerAlgorithmLaunch(
                **common,
                grpo_clip_eps=float(algorithm_cfg["grpo_clip_eps"]),
                grpo_kl_beta=float(algorithm_cfg["grpo_kl_beta"]),
                clip_ratio_low=(
                    float(algorithm_cfg["clip_ratio_low"])
                    if algorithm_cfg["clip_ratio_low"] is not None
                    else None
                ),
                clip_ratio_high=(
                    float(algorithm_cfg["clip_ratio_high"])
                    if algorithm_cfg["clip_ratio_high"] is not None
                    else None
                ),
                clip_ratio_c=(
                    float(algorithm_cfg["clip_ratio_c"])
                    if algorithm_cfg.get("clip_ratio_c") is not None
                    else algorithm_cfg.get("clip_ratio_c")
                ),
                loss_agg_mode=algorithm_cfg["loss_agg_mode"],
                kl_type=algorithm_cfg["kl_type"],
                use_decoupled_loss=bool(
                    algorithm_cfg.get("use_decoupled_loss", _GRPO_ALGO_DEFAULTS.use_decoupled_loss)
                ),
                behave_imp_weight_cap=float(
                    algorithm_cfg.get("behave_imp_weight_cap", _GRPO_ALGO_DEFAULTS.behave_imp_weight_cap)
                ),
            )
        raise ValueError(f"Unsupported trainer algorithm launch mode: {mode}")

    mode = algorithm_cfg.mode
    opd = algorithm_cfg.opd
    if mode in ("opd", "opsd"):
        actor_critic = None
        if opd.pg_actor_critic:
            actor_critic = ActorCriticLaunch(
                value_mode=opd.pg_value_mode,
                gae_lambda=opd.pg_gae_lambda,
                value_coef=opd.pg_value_coef,
                normalize_advantages=opd.pg_value_normalize_advantages,
            )
        return OPDTrainerAlgorithmLaunch(
            mode=mode,
            kl_loss_mode=opd.kl_loss_mode,
            kl_skew_alpha=opd.skewed_alpha,
            pg_clip_eps=opd.pg_clip_eps,
            use_importance_sampling=opd.use_importance_sampling,
            kl_token_clip=opd.kl_token_clip,
            pg_online_advantage=opd.pg_online_advantage,
            use_decoupled_loss=opd.use_decoupled_loss,
            behave_imp_weight_cap=opd.behave_imp_weight_cap,
            pg_m2po_budget=opd.pg_m2po_budget,
            pg_m2po_miniclip_low=opd.pg_m2po_miniclip_low,
            pg_m2po_miniclip_high=opd.pg_m2po_miniclip_high,
            pg_token_weighted_backward=opd.pg_token_weighted_backward,
            mof_variant=opd.mof_variant,
            mof_partition=opd.mof_partition,
            mof_eta_mass=opd.mof_eta_mass,
            mof_eta_odds=opd.mof_eta_odds,
            mof_lambda_odds=opd.mof_lambda_odds,
            mof_eps=opd.mof_eps,
            mof_deduplicate_candidates=opd.mof_deduplicate_candidates,
            rollout_student_topk_k=(
                opd.rollout_student_topk_k
                if opd.kl_loss_mode in {"reverse_kl_rollout_student_topk", "thunlp_opd_default_loss"}
                else None
            ),
            thunlp_loss_agg_mode=(
                "token-mean" if opd.kl_loss_mode == "thunlp_opd_default_loss" else None
            ),
            actor_critic=actor_critic,
        )
    if mode == "sft":
        sft = algorithm_cfg.sft
        return SFTTrainerAlgorithmLaunch(
            mode=mode,
            kl_loss_mode=opd.kl_loss_mode,
            kl_skew_alpha=opd.skewed_alpha,
            pg_clip_eps=opd.pg_clip_eps,
            sft_loss_mode=sft.loss_mode,
            ce_alpha=sft.ce_alpha,
            n_kl_logprobs=opd.n_kl_logprobs,
        )
    if mode == "grpo":
        grpo = algorithm_cfg.grpo
        return GRPOTrainerAlgorithmLaunch(
            mode=mode,
            kl_loss_mode=opd.kl_loss_mode,
            kl_skew_alpha=opd.skewed_alpha,
            pg_clip_eps=opd.pg_clip_eps,
            grpo_clip_eps=grpo.clip_eps,
            grpo_kl_beta=grpo.kl_beta,
            clip_ratio_low=grpo.clip_ratio_low,
            clip_ratio_high=grpo.clip_ratio_high,
            clip_ratio_c=grpo.clip_ratio_c,
            loss_agg_mode=grpo.loss_agg_mode,
            kl_type=grpo.kl_type,
            use_decoupled_loss=grpo.use_decoupled_loss,
            behave_imp_weight_cap=grpo.behave_imp_weight_cap,
        )
    raise ValueError(f"Unsupported trainer algorithm launch mode: {mode}")


def serialize_algorithm_payload(algorithm_cfg: Any) -> JSONDict:
    """Serialize typed algorithm config to the legacy flat launch payload boundary."""

    return build_trainer_algorithm_launch(algorithm_cfg).to_payload()


def algorithm_mode(algorithm_cfg: TrainerAlgorithmLaunch | Mapping[str, Any]) -> str:
    if isinstance(algorithm_cfg, Mapping):
        return algorithm_cfg["mode"]
    return algorithm_cfg.mode


def algorithm_needs_student_logprobs(
    algorithm_cfg: TrainerAlgorithmLaunch | Mapping[str, Any],
) -> bool:
    if isinstance(algorithm_cfg, Mapping):
        return (
            algorithm_cfg.get("kl_loss_mode", "forward_kl") == "policy_gradient_kl"
            and bool(
                algorithm_cfg.get(
                    "use_importance_sampling",
                    _OPD_ALGO_DEFAULTS.use_importance_sampling,
                )
            )
        )
    return (
        algorithm_cfg.kl_loss_mode == "policy_gradient_kl"
        and bool(algorithm_cfg.use_importance_sampling)
    )


def algorithm_is_actor_critic(
    algorithm_cfg: TrainerAlgorithmLaunch | Mapping[str, Any],
) -> bool:
    if isinstance(algorithm_cfg, Mapping):
        return bool(algorithm_cfg.get("pg_actor_critic", False))
    return isinstance(algorithm_cfg, OPDTrainerAlgorithmLaunch) and algorithm_cfg.actor_critic is not None


def algorithm_uses_token_weighted_backward(
    algorithm_cfg: TrainerAlgorithmLaunch | Mapping[str, Any],
) -> bool:
    if isinstance(algorithm_cfg, Mapping):
        return bool(algorithm_cfg.get("pg_token_weighted_backward", False))
    return (
        isinstance(algorithm_cfg, OPDTrainerAlgorithmLaunch)
        and bool(algorithm_cfg.pg_token_weighted_backward)
    )


@dataclass(frozen=True)
class TrainerLaunchStatic:
    model_path: str
    dtype: str
    attn_implementation: str | None
    optim: JSONDict
    micro_batch_size: int
    mini_batch_size: int
    max_response_length: int
    use_sequence_packing: bool
    use_torch_compile: bool
    max_grad_norm: float
    loss_mode: str
    kl_chunk_size: int
    backend: str
    algorithm: TrainerAlgorithmLaunch
    deterministic: bool
    seed: int
    trust_remote_code: bool = False
    lora: JSONDict | None = None
    megatron: JSONDict | None = None
    teacher_model_path: str | None = None
    teacher_artifact_mode: str = "legacy"
    teacher_hidden_dtype: str = "bfloat16"
    teacher_hidden_semantics: str = "lm_head_input"
    teacher_hidden_recompute_materialization: str = "lazy"
    fused_hybrid_rollout: JSONDict | None = None
    fused_hybrid_sync: JSONDict | None = None

    def to_payload(self) -> JSONDict:
        return {
            "model_path": self.model_path,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "optim": _copy(self.optim),
            "micro_batch_size": self.micro_batch_size,
            "mini_batch_size": self.mini_batch_size,
            "max_response_length": self.max_response_length,
            "use_sequence_packing": self.use_sequence_packing,
            "use_torch_compile": self.use_torch_compile,
            "max_grad_norm": self.max_grad_norm,
            "loss_mode": self.loss_mode,
            "kl_chunk_size": self.kl_chunk_size,
            "backend": self.backend,
            "algorithm": serialize_algorithm_payload(self.algorithm),
            "deterministic": self.deterministic,
            "seed": self.seed,
            "trust_remote_code": self.trust_remote_code,
            "lora": _copy(self.lora),
            "megatron": _copy(self.megatron),
            "teacher_model_path": self.teacher_model_path,
            "teacher_artifact_mode": self.teacher_artifact_mode,
            "teacher_hidden_dtype": self.teacher_hidden_dtype,
            "teacher_hidden_semantics": self.teacher_hidden_semantics,
            "teacher_hidden_recompute_materialization": self.teacher_hidden_recompute_materialization,
            "fused_hybrid_rollout": _copy(self.fused_hybrid_rollout),
            "fused_hybrid_sync": _copy(self.fused_hybrid_sync),
        }


@dataclass(frozen=True)
class TrainerLaunchRuntime:
    gpu_ids: str | None
    total_steps: int
    nccl_timeout_hours: int
    rank_info: JSONDict
    teacher_artifact_queue: Any = None

    def to_payload(self) -> JSONDict:
        return {
            "gpu_ids": self.gpu_ids,
            "total_steps": self.total_steps,
            "nccl_timeout_hours": self.nccl_timeout_hours,
            "rank_info": _copy(self.rank_info),
            "teacher_artifact_queue": self.teacher_artifact_queue,
        }


@dataclass(frozen=True)
class TrainerLaunchSpec:
    static: TrainerLaunchStatic
    runtime: TrainerLaunchRuntime

    @property
    def backend(self) -> str:
        return self.static.backend

    @property
    def loss_mode(self) -> str:
        return self.static.loss_mode

    def merged_config(self) -> JSONDict:
        cfg = self.static.to_payload()
        cfg.update({
            "gpu_ids": self.runtime.gpu_ids,
            "total_steps": self.runtime.total_steps,
            "nccl_timeout_hours": self.runtime.nccl_timeout_hours,
        })
        return cfg

    def rank_payload(self) -> JSONDict:
        return self.runtime.to_payload()["rank_info"]

    def with_runtime(self, **runtime_updates: Any) -> "TrainerLaunchSpec":
        payload = self.runtime.to_payload()
        payload.update(runtime_updates)
        return replace(self, runtime=TrainerLaunchRuntime(**payload))


@dataclass(frozen=True)
class RolloutLaunchStatic:
    model_path: str
    tp_size: int
    max_response_length: int
    temperature: float
    top_p: float
    top_k: int
    max_num_seqs: int
    use_weight_transfer: bool
    max_model_len: int | None
    max_num_batched_tokens: int | None
    enforce_eager: bool
    dtype: str
    quantization: str | None
    native_lora: bool
    lora_rank: int
    lora_cfg: JSONDict | None
    max_logprobs: int
    block_size: int | None
    pin_cpu_affinity: bool
    bind_numa_memory: bool
    trust_remote_code: bool = False

    def to_payload(self) -> JSONDict:
        return {
            "model_path": self.model_path,
            "tp_size": self.tp_size,
            "max_response_length": self.max_response_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_num_seqs": self.max_num_seqs,
            "use_weight_transfer": self.use_weight_transfer,
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "enforce_eager": self.enforce_eager,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "native_lora": self.native_lora,
            "lora_rank": self.lora_rank,
            "lora_cfg": _copy(self.lora_cfg),
            "max_logprobs": self.max_logprobs,
            "block_size": self.block_size,
            "pin_cpu_affinity": self.pin_cpu_affinity,
            "bind_numa_memory": self.bind_numa_memory,
            "trust_remote_code": self.trust_remote_code,
        }


@dataclass(frozen=True)
class RolloutLaunchRuntime:
    gpu_ids: str | None
    worker_id: int
    gpu_memory_utilization: float
    prompt_queue: Any = None
    pause_mode: str | None = None
    cpu_affinity_cpus: tuple[int, ...] = ()
    numa_nodes: tuple[int, ...] = ()
    vllm_port: int | None = None
    vllm_master_port: int | None = None
    seed: int | None = None

    def to_payload(self) -> JSONDict:
        return {
            "gpu_ids": self.gpu_ids,
            "worker_id": self.worker_id,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "prompt_queue": self.prompt_queue,
            "pause_mode": self.pause_mode,
            "cpu_affinity_cpus": list(self.cpu_affinity_cpus),
            "numa_nodes": list(self.numa_nodes),
            "vllm_port": self.vllm_port,
            "vllm_master_port": self.vllm_master_port,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RolloutLaunchSpec:
    static: RolloutLaunchStatic
    runtime: RolloutLaunchRuntime

    def merged_config(self) -> JSONDict:
        cfg = self.static.to_payload()
        cfg.update(self.runtime.to_payload())
        return cfg

    def with_runtime(self, **runtime_updates: Any) -> "RolloutLaunchSpec":
        payload = self.runtime.to_payload()
        payload.update(runtime_updates)
        if "cpu_affinity_cpus" in payload and isinstance(payload["cpu_affinity_cpus"], list):
            payload["cpu_affinity_cpus"] = tuple(payload["cpu_affinity_cpus"])
        if "numa_nodes" in payload and isinstance(payload["numa_nodes"], list):
            payload["numa_nodes"] = tuple(payload["numa_nodes"])
        return replace(self, runtime=RolloutLaunchRuntime(**payload))


@dataclass(frozen=True)
class TeacherLaunchStatic:
    backend: str
    model_path: str
    n_logprobs: int
    tp_size: int
    gpu_memory_utilization: float | None
    max_model_len: int | None
    max_num_seqs: int | None
    enforce_eager: bool | None
    scoring_batch_size: int
    dtype: str
    disable_fast_logprobs: bool | None
    block_size: int | None
    use_torch_compile: bool | None = None
    hidden_recompute: bool = False
    teacher_hidden_dtype: str = "bfloat16"
    teacher_hidden_semantics: str = "lm_head_input"
    trust_remote_code: bool = False

    def to_payload(self) -> JSONDict:
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "n_logprobs": self.n_logprobs,
            "tp_size": self.tp_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "enforce_eager": self.enforce_eager,
            "scoring_batch_size": self.scoring_batch_size,
            "dtype": self.dtype,
            "disable_fast_logprobs": self.disable_fast_logprobs,
            "block_size": self.block_size,
            "use_torch_compile": self.use_torch_compile,
            "hidden_recompute": self.hidden_recompute,
            "teacher_hidden_dtype": self.teacher_hidden_dtype,
            "teacher_hidden_semantics": self.teacher_hidden_semantics,
            "trust_remote_code": self.trust_remote_code,
        }


@dataclass(frozen=True)
class TeacherLaunchRuntime:
    gpu_ids: str | None
    bind_port: int
    bind_address: str
    vllm_port: int | None = None
    vllm_master_port: int | None = None
    seed: int | None = None

    def to_payload(self) -> JSONDict:
        return {
            "gpu_ids": self.gpu_ids,
            "bind_port": self.bind_port,
            "bind_address": self.bind_address,
            "vllm_port": self.vllm_port,
            "vllm_master_port": self.vllm_master_port,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class TeacherLaunchSpec:
    static: TeacherLaunchStatic
    runtime: TeacherLaunchRuntime

    @property
    def backend(self) -> str:
        return self.static.backend

    def merged_config(self) -> JSONDict:
        cfg = self.static.to_payload()
        cfg.update(self.runtime.to_payload())
        return cfg

    def with_runtime(self, **runtime_updates: Any) -> "TeacherLaunchSpec":
        payload = self.runtime.to_payload()
        payload.update(runtime_updates)
        return replace(self, runtime=TeacherLaunchRuntime(**payload))


# ---------------------------------------------------------------------------
# Legacy-adapter helpers
# ---------------------------------------------------------------------------


def ensure_trainer_launch_spec(
    config_or_spec: TrainerLaunchSpec | Mapping[str, Any],
    rank_info: Mapping[str, Any] | None = None,
) -> TrainerLaunchSpec:
    if isinstance(config_or_spec, TrainerLaunchSpec):
        if rank_info is None:
            return config_or_spec
        return config_or_spec.with_runtime(rank_info=_copy(dict(rank_info)))

    cfg = _copy(dict(config_or_spec))
    actual_rank = _copy(dict(rank_info or cfg.pop("rank_info", {})))
    return TrainerLaunchSpec(
        static=TrainerLaunchStatic(
            model_path=cfg["model_path"],
            dtype=cfg["dtype"],
            attn_implementation=cfg.get("attn_implementation"),
            optim=_copy(cfg["optim"]),
            micro_batch_size=cfg["micro_batch_size"],
            mini_batch_size=cfg["mini_batch_size"],
            max_response_length=cfg["max_response_length"],
            use_sequence_packing=cfg["use_sequence_packing"],
            use_torch_compile=cfg.get("use_torch_compile", _TRAINER_DEFAULTS.use_torch_compile),
            max_grad_norm=float(cfg.get("max_grad_norm", _TRAINER_DEFAULTS.optim.max_grad_norm)),
            loss_mode=cfg.get("loss_mode", "kl"),
            kl_chunk_size=cfg.get("kl_chunk_size", _TRAINER_DEFAULTS.kl_chunk_size),
            backend=cfg.get("backend", _TRAINER_DEFAULTS.backend),
            algorithm=build_trainer_algorithm_launch(cfg["algorithm"]),
            deterministic=cfg.get("deterministic", False),
            seed=cfg.get("seed", 42),
            lora=_copy(cfg.get("lora")),
            megatron=_copy(cfg.get("megatron")),
            teacher_model_path=cfg.get("teacher_model_path"),
            teacher_artifact_mode=cfg.get("teacher_artifact_mode", "legacy"),
            teacher_hidden_dtype=cfg.get("teacher_hidden_dtype", "bfloat16"),
            teacher_hidden_semantics=cfg.get("teacher_hidden_semantics", "lm_head_input"),
            teacher_hidden_recompute_materialization=cfg.get("teacher_hidden_recompute_materialization", "lazy"),
            fused_hybrid_rollout=_copy(cfg.get("fused_hybrid_rollout")),
            fused_hybrid_sync=_copy(cfg.get("fused_hybrid_sync")),
            trust_remote_code=bool(cfg.get("trust_remote_code", _TRAINER_DEFAULTS.trust_remote_code)),
        ),
        runtime=TrainerLaunchRuntime(
            gpu_ids=cfg.get("gpu_ids"),
            total_steps=cfg["total_steps"],
            nccl_timeout_hours=int(cfg.get("nccl_timeout_hours", _WEIGHT_SYNC_DEFAULTS.nccl_timeout_hours)),
            rank_info=actual_rank,
            teacher_artifact_queue=cfg.get("teacher_artifact_queue"),
        ),
    )


def ensure_rollout_launch_spec(
    config_or_spec: RolloutLaunchSpec | Mapping[str, Any],
) -> RolloutLaunchSpec:
    if isinstance(config_or_spec, RolloutLaunchSpec):
        return config_or_spec

    cfg = _copy(dict(config_or_spec))
    return RolloutLaunchSpec(
        static=RolloutLaunchStatic(
            model_path=cfg["model_path"],
            tp_size=cfg["tp_size"],
            max_response_length=cfg["max_response_length"],
            temperature=cfg.get("temperature", _ROLLOUT_DEFAULTS.temperature),
            top_p=cfg.get("top_p", _ROLLOUT_DEFAULTS.top_p),
            top_k=cfg.get("top_k", _ROLLOUT_DEFAULTS.top_k),
            max_num_seqs=cfg.get("max_num_seqs", _ROLLOUT_VLLM_DEFAULTS.max_num_seqs),
            use_weight_transfer=cfg.get("use_weight_transfer", False),
            max_model_len=cfg.get("max_model_len"),
            max_num_batched_tokens=cfg.get("max_num_batched_tokens"),
            enforce_eager=cfg.get("enforce_eager", _ROLLOUT_VLLM_DEFAULTS.enforce_eager),
            dtype=cfg.get("dtype", _ROLLOUT_DEFAULTS.dtype),
            quantization=cfg.get("quantization"),
            native_lora=cfg.get("native_lora", False),
            lora_rank=cfg.get("lora_rank", 0),
            lora_cfg=_copy(cfg.get("lora_cfg")),
            max_logprobs=cfg.get("max_logprobs", 20),
            block_size=cfg.get("block_size"),
            pin_cpu_affinity=bool(cfg.get("pin_cpu_affinity", _ROLLOUT_DEFAULTS.pin_cpu_affinity)),
            bind_numa_memory=bool(cfg.get("bind_numa_memory", _ROLLOUT_DEFAULTS.bind_numa_memory)),
            trust_remote_code=bool(cfg.get("trust_remote_code", _ROLLOUT_DEFAULTS.trust_remote_code)),
        ),
        runtime=RolloutLaunchRuntime(
            gpu_ids=cfg.get("gpu_ids"),
            worker_id=cfg.get("worker_id", 0),
            gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", _ROLLOUT_VLLM_DEFAULTS.gpu_memory_utilization)),
            prompt_queue=cfg.get("prompt_queue"),
            pause_mode=cfg.get("pause_mode"),
            cpu_affinity_cpus=tuple(cfg.get("cpu_affinity_cpus", [])),
            numa_nodes=tuple(cfg.get("numa_nodes", [])),
            vllm_port=cfg.get("vllm_port"),
            vllm_master_port=cfg.get("vllm_master_port"),
            seed=cfg.get("seed"),
        ),
    )


def ensure_teacher_launch_spec(
    config_or_spec: TeacherLaunchSpec | Mapping[str, Any],
    *,
    backend: str | None = None,
) -> TeacherLaunchSpec:
    if isinstance(config_or_spec, TeacherLaunchSpec):
        return config_or_spec

    cfg = _copy(dict(config_or_spec))
    teacher_backend = backend or cfg.get("backend", _TEACHER_DEFAULTS.backend)
    return TeacherLaunchSpec(
        static=TeacherLaunchStatic(
            backend=teacher_backend,
            model_path=cfg["model_path"],
            n_logprobs=cfg["n_logprobs"],
            tp_size=cfg.get("tp_size", _TEACHER_VLLM_DEFAULTS.tensor_parallel_size),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization"),
            max_model_len=cfg.get("max_model_len"),
            max_num_seqs=cfg.get("max_num_seqs"),
            enforce_eager=cfg.get("enforce_eager"),
            scoring_batch_size=cfg.get(
                "scoring_batch_size",
                4 if teacher_backend == "hf" else 32,
            ),
            dtype=cfg.get("dtype", _TEACHER_DEFAULTS.dtype),
            disable_fast_logprobs=cfg.get("disable_fast_logprobs"),
            block_size=cfg.get("block_size"),
            use_torch_compile=cfg.get("use_torch_compile"),
            trust_remote_code=bool(cfg.get("trust_remote_code", _TEACHER_DEFAULTS.trust_remote_code)),
        ),
        runtime=TeacherLaunchRuntime(
            gpu_ids=cfg.get("gpu_ids"),
            bind_port=cfg["bind_port"],
            bind_address=cfg.get("bind_address", _TEACHER_DEFAULTS.bind_address),
            vllm_port=cfg.get("vllm_port"),
            vllm_master_port=cfg.get("vllm_master_port"),
            seed=cfg.get("seed"),
        ),
    )

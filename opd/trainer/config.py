"""Training mode configuration dataclasses and payload builders.

Each mode's hyperparameters are grouped into a dedicated config object,
replacing the scattered attributes that used to clutter trainer/backend
constructors. Builder helpers centralize reconstruction from launch payloads.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from opd.loss.kl import KLConfig
from opd.utils.config import GRPOAlgorithmConfig, OPDAlgorithmConfig, SFTAlgorithmConfig


_OPD_DEFAULTS = OPDAlgorithmConfig()
_GRPO_DEFAULTS = GRPOAlgorithmConfig()
_SFT_DEFAULTS = SFTAlgorithmConfig()


@dataclass
class SFTConfig:
    """Configuration for SFT (Supervised Fine-Tuning) mode.

    Attributes:
        loss_mode: "ce" (cross-entropy only), "kl" (KL only), or "mixed"
        ce_alpha: Weight of CE in mixed mode (KL weight = 1 - ce_alpha)
        n_kl_logprobs: Number of teacher top-k logprobs to use for KL
    """
    loss_mode: str | None = None
    ce_alpha: float | None = None
    n_kl_logprobs: int | None = None


@dataclass
class GRPOConfig:
    """Configuration for GRPO (Group Relative Policy Optimization) mode.

    Attributes:
        clip_eps: PPO clip epsilon for policy ratio
        kl_beta: KL penalty coefficient (0 = no KL penalty)
        clip_ratio_low: Lower clip bound (None = use 1 - clip_eps)
        clip_ratio_high: Upper clip bound (None = use 1 + clip_eps)
        clip_ratio_c: Dual-clip c parameter
        loss_agg_mode: "token-mean" or "sample-mean"
        kl_type: KL divergence type ("k1", "k2", "k3")
    """
    clip_eps: float | None = None
    kl_beta: float | None = None
    clip_ratio_low: float | None = None
    clip_ratio_high: float | None = None
    clip_ratio_c: float | None = None
    loss_agg_mode: str | None = None
    kl_type: str | None = None
    use_decoupled_loss: bool | None = None
    behave_imp_weight_cap: float | None = None


@dataclass
class ActorCriticConfig:
    """Runtime configuration for actor-critic OPD."""

    clip_eps: float | None = None
    online_advantage: bool | None = None
    value_mode: str | None = None
    gae_lambda: float | None = None
    value_coef: float | None = None
    normalize_advantages: bool | None = None
    use_decoupled_loss: bool | None = None
    behave_imp_weight_cap: float | None = None
    m2po_budget: float | None = None
    m2po_miniclip_low: float | None = None
    m2po_miniclip_high: float | None = None


def build_kl_config_from_algorithm_payload(algo: Mapping[str, Any]) -> KLConfig:
    """Build the canonical KL runtime config from the algorithm payload."""

    if isinstance(algo, Mapping):
        mode = algo["mode"]
        config = KLConfig(
            mode=algo["kl_loss_mode"],
            skew_alpha=float(algo["kl_skew_alpha"]),
            pg_clip_eps=float(algo["pg_clip_eps"]),
        )
        if mode in ("opd", "opsd"):
            config.use_importance_sampling = bool(
                algo.get("use_importance_sampling", _OPD_DEFAULTS.use_importance_sampling)
            )
            config.token_clip = float(algo["kl_token_clip"])
            config.pg_online_advantage = bool(algo["pg_online_advantage"])
            config.use_decoupled_loss = bool(algo.get("use_decoupled_loss", _OPD_DEFAULTS.use_decoupled_loss))
            config.behave_imp_weight_cap = float(algo.get("behave_imp_weight_cap", _OPD_DEFAULTS.behave_imp_weight_cap))
            config.pg_m2po_budget = algo.get("pg_m2po_budget")
            config.pg_m2po_miniclip_low = float(algo.get("pg_m2po_miniclip_low", _OPD_DEFAULTS.pg_m2po_miniclip_low))
            config.pg_m2po_miniclip_high = float(algo.get("pg_m2po_miniclip_high", _OPD_DEFAULTS.pg_m2po_miniclip_high))
            config.mof_variant = algo.get("mof_variant", _OPD_DEFAULTS.mof_variant)
            config.mof_partition = algo.get("mof_partition", _OPD_DEFAULTS.mof_partition)
            config.mof_eta_mass = float(algo.get("mof_eta_mass", _OPD_DEFAULTS.mof_eta_mass))
            config.mof_eta_odds = float(algo.get("mof_eta_odds", _OPD_DEFAULTS.mof_eta_odds))
            config.mof_lambda_odds = float(algo.get("mof_lambda_odds", _OPD_DEFAULTS.mof_lambda_odds))
            config.mof_eps = float(algo.get("mof_eps", _OPD_DEFAULTS.mof_eps))
            config.mof_deduplicate_candidates = bool(
                algo.get("mof_deduplicate_candidates", _OPD_DEFAULTS.mof_deduplicate_candidates)
            )
        return config

    from opd.launch_specs import OPDTrainerAlgorithmLaunch, build_trainer_algorithm_launch

    typed = build_trainer_algorithm_launch(algo)
    assert not isinstance(typed, Mapping)
    config = KLConfig(
        mode=typed.kl_loss_mode,
        skew_alpha=float(typed.kl_skew_alpha),
        pg_clip_eps=float(typed.pg_clip_eps),
    )
    if typed.mode in ("opd", "opsd"):
        assert isinstance(typed, OPDTrainerAlgorithmLaunch)
        config.use_importance_sampling = bool(typed.use_importance_sampling)
        config.token_clip = float(typed.kl_token_clip)
        config.pg_online_advantage = bool(typed.pg_online_advantage)
        config.use_decoupled_loss = bool(typed.use_decoupled_loss)
        config.behave_imp_weight_cap = float(typed.behave_imp_weight_cap)
        config.pg_m2po_budget = typed.pg_m2po_budget
        config.pg_m2po_miniclip_low = float(typed.pg_m2po_miniclip_low)
        config.pg_m2po_miniclip_high = float(typed.pg_m2po_miniclip_high)
        config.mof_variant = typed.mof_variant
        config.mof_partition = typed.mof_partition
        config.mof_eta_mass = float(typed.mof_eta_mass)
        config.mof_eta_odds = float(typed.mof_eta_odds)
        config.mof_lambda_odds = float(typed.mof_lambda_odds)
        config.mof_eps = float(typed.mof_eps)
        config.mof_deduplicate_candidates = bool(typed.mof_deduplicate_candidates)
    return config


def build_sft_config_from_algorithm_payload(algo: Mapping[str, Any]) -> SFTConfig:
    """Build the canonical SFT runtime config from the algorithm payload."""

    if isinstance(algo, Mapping):
        return SFTConfig(
            loss_mode=algo["sft_loss_mode"],
            ce_alpha=float(algo["ce_alpha"]),
            n_kl_logprobs=int(algo["n_kl_logprobs"]),
        )

    from opd.launch_specs import SFTTrainerAlgorithmLaunch, build_trainer_algorithm_launch

    typed = build_trainer_algorithm_launch(algo)
    assert isinstance(typed, SFTTrainerAlgorithmLaunch)
    return SFTConfig(
        loss_mode=typed.sft_loss_mode,
        ce_alpha=float(typed.ce_alpha),
        n_kl_logprobs=int(typed.n_kl_logprobs),
    )


def build_grpo_config_from_algorithm_payload(algo: Mapping[str, Any]) -> GRPOConfig:
    """Build the canonical GRPO runtime config from the algorithm payload."""

    if isinstance(algo, Mapping):
        clip_ratio_c = algo.get("clip_ratio_c")
        if clip_ratio_c is None and "clip_ratio_c" not in algo:
            clip_ratio_c = _GRPO_DEFAULTS.clip_ratio_c
        elif clip_ratio_c is not None:
            clip_ratio_c = float(clip_ratio_c)

        return GRPOConfig(
            clip_eps=float(algo["grpo_clip_eps"]),
            kl_beta=float(algo["grpo_kl_beta"]),
            clip_ratio_low=float(algo["clip_ratio_low"]) if algo["clip_ratio_low"] is not None else None,
            clip_ratio_high=float(algo["clip_ratio_high"]) if algo["clip_ratio_high"] is not None else None,
            clip_ratio_c=clip_ratio_c,
            loss_agg_mode=algo["loss_agg_mode"],
            kl_type=algo["kl_type"],
            use_decoupled_loss=bool(algo.get("use_decoupled_loss", _GRPO_DEFAULTS.use_decoupled_loss)),
            behave_imp_weight_cap=float(algo.get("behave_imp_weight_cap", _GRPO_DEFAULTS.behave_imp_weight_cap)),
        )

    from opd.launch_specs import GRPOTrainerAlgorithmLaunch, build_trainer_algorithm_launch

    typed = build_trainer_algorithm_launch(algo)
    assert isinstance(typed, GRPOTrainerAlgorithmLaunch)

    clip_ratio_c = typed.clip_ratio_c
    if clip_ratio_c is None:
        clip_ratio_c = _GRPO_DEFAULTS.clip_ratio_c
    elif clip_ratio_c is not None:
        clip_ratio_c = float(clip_ratio_c)

    return GRPOConfig(
        clip_eps=float(typed.grpo_clip_eps),
        kl_beta=float(typed.grpo_kl_beta),
        clip_ratio_low=float(typed.clip_ratio_low) if typed.clip_ratio_low is not None else None,
        clip_ratio_high=float(typed.clip_ratio_high) if typed.clip_ratio_high is not None else None,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode=typed.loss_agg_mode,
        kl_type=typed.kl_type,
        use_decoupled_loss=bool(typed.use_decoupled_loss),
        behave_imp_weight_cap=float(typed.behave_imp_weight_cap),
    )


def build_actor_critic_config_from_algorithm_payload(
    algo: Mapping[str, Any],
) -> ActorCriticConfig:
    """Build the canonical actor-critic runtime config from the algorithm payload."""

    if isinstance(algo, Mapping):
        return ActorCriticConfig(
            clip_eps=float(algo["pg_clip_eps"]),
            online_advantage=bool(algo["pg_online_advantage"]),
            value_mode=algo.get("pg_value_mode", _OPD_DEFAULTS.pg_value_mode),
            gae_lambda=float(algo.get("pg_gae_lambda", _OPD_DEFAULTS.pg_gae_lambda)),
            value_coef=float(algo.get("pg_value_coef", _OPD_DEFAULTS.pg_value_coef)),
            normalize_advantages=bool(algo.get("pg_value_normalize_advantages", _OPD_DEFAULTS.pg_value_normalize_advantages)),
            use_decoupled_loss=bool(algo.get("use_decoupled_loss", _OPD_DEFAULTS.use_decoupled_loss)),
            behave_imp_weight_cap=float(algo.get("behave_imp_weight_cap", _OPD_DEFAULTS.behave_imp_weight_cap)),
            m2po_budget=algo.get("pg_m2po_budget"),
            m2po_miniclip_low=float(algo.get("pg_m2po_miniclip_low", _OPD_DEFAULTS.pg_m2po_miniclip_low)),
            m2po_miniclip_high=float(algo.get("pg_m2po_miniclip_high", _OPD_DEFAULTS.pg_m2po_miniclip_high)),
        )

    from opd.launch_specs import OPDTrainerAlgorithmLaunch, build_trainer_algorithm_launch

    typed = build_trainer_algorithm_launch(algo)
    assert isinstance(typed, OPDTrainerAlgorithmLaunch)
    if typed.actor_critic is None:
        raise ValueError("Actor-critic config requested for algorithm launch without actor_critic state")

    return ActorCriticConfig(
        clip_eps=float(typed.pg_clip_eps),
        online_advantage=bool(typed.pg_online_advantage),
        value_mode=typed.actor_critic.value_mode,
        gae_lambda=float(typed.actor_critic.gae_lambda),
        value_coef=float(typed.actor_critic.value_coef),
        normalize_advantages=bool(typed.actor_critic.normalize_advantages),
        use_decoupled_loss=bool(typed.use_decoupled_loss),
        behave_imp_weight_cap=float(typed.behave_imp_weight_cap),
        m2po_budget=typed.pg_m2po_budget,
        m2po_miniclip_low=float(typed.pg_m2po_miniclip_low),
        m2po_miniclip_high=float(typed.pg_m2po_miniclip_high),
    )

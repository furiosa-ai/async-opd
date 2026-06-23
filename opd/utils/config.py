"""OPDConfig dataclass hierarchy with YAML loading, validation, and translation.

New flat config format -> OPDConfig dataclasses -> to_internal_dict() -> old nested format.
This is the core of the config refactor (Phase 1). The to_internal_dict() method produces
the exact old dict structure so existing Python code is unchanged.

Usage:
    from opd.utils.config import OPDConfig

    cfg = OPDConfig.from_yaml("configs/examples/opd_qwen3_1.7b.yaml", overrides=["trainer.optim.lr=2e-5"])
    old_dict = cfg.to_internal_dict()
"""

from __future__ import annotations

import copy
import dataclasses
import json
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

# ============================================================================ #
# Constants                                                                      #
# ============================================================================ #

_VALID_MODES = {"opd", "opsd", "grpo", "sft"}
_VALID_KL_MODES = {
    "forward_kl",
    "multi_sample_forward_kl",
    "reverse_kl",
    "reverse_kl_rollout_student_topk",
    "thunlp_opd_default_loss",
    "multi_sample_policy_gradient_kl",
    "mof_opd",
    "skewed_kl",
    "token_level_kl",
    "policy_gradient_kl",
}
_VALID_EVAL_MODES = {"inline", "perplexity", "post", "post_allgpu"}
_VALID_EVAL_CHECKPOINT_POLICIES = {"all", "final", "steps"}
_VALID_PROMPT_SOURCES = {"raw", "last_user_content"}
_VALID_TRAINER_BACKENDS = {"fsdp", "megatron"}
_VALID_ROLLOUT_BACKENDS = {"vllm", "hf", "dummy"}
_VALID_TEACHER_BACKENDS = {"vllm", "hf"}
_VALID_WEIGHT_SYNC_BACKENDS = {"nccl", "cpu"}
_VALID_SCHEDULING_MODES = {"n_step_off", "fully_async", "fused_hybrid_sync"}
_VALID_DEPLOYMENTS = {"local", "ray"}
_VALID_SFT_LOSS_MODES = {"ce", "kl", "mixed"}
_TRUST_REMOTE_CODE_ENV = "OPD_ALLOW_TRUST_REMOTE_CODE"

_IGNORED_LEGACY_KEYS = {
    # Top-level
    "nnodes",
    # Teacher section
    "n_server_workers", "n_dp", "use_ray_actor",
    # Training/algorithm
    "use_kl_loss", "use_kl_in_reward",
    "ppo_mini_batch_size", "ppo_micro_batch_size_per_gpu",
    # Rollout
    "router_replay", "override_config", "model_config",
    "truncation", "trust_remote_code", "name",
    # Misc
    "logger",
}

# Sentinel for required fields with no default
_MISSING = object()


# ============================================================================ #
# Helper functions                                                               #
# ============================================================================ #

def _is_dataclass_type(tp) -> bool:
    """Check if a type annotation refers to a dataclass (unwrapping Optional)."""
    if dataclasses.is_dataclass(tp):
        return True
    # Handle X | None (Union[X, None])
    origin = get_origin(tp)
    if origin is type(int | str):  # types.UnionType (Python 3.10+)
        for arg in get_args(tp):
            if arg is not type(None) and dataclasses.is_dataclass(arg):
                return True
    return False


def _unwrap_dataclass_type(tp):
    """Get the actual dataclass type from a possibly-Optional annotation."""
    if dataclasses.is_dataclass(tp):
        return tp
    origin = get_origin(tp)
    if origin is type(int | str):
        for arg in get_args(tp):
            if arg is not type(None) and dataclasses.is_dataclass(arg):
                return arg
    return None


def _from_dict(cls, d: dict, path: str = "", ignored_keys: set | None = None):
    """Load a dataclass from a dict, raising ValueError on unknown keys.

    Args:
        cls: The dataclass type to instantiate.
        d: The source dict.
        path: Dot-path prefix for error messages (e.g. "teacher.vllm").
        ignored_keys: Extra keys to silently skip (beyond _IGNORED_LEGACY_KEYS).
    """
    if d is None:
        return None
    if not isinstance(d, dict):
        raise ValueError(f"Expected dict for '{path}', got {type(d).__name__}: {d!r}")

    all_ignored = _IGNORED_LEGACY_KEYS | (ignored_keys or set())
    field_map = {f.name: f for f in fields(cls)}
    known_keys = set(field_map.keys())

    # Check for unknown keys
    for key in d:
        if key not in known_keys and key not in all_ignored:
            full_path = f"{path}.{key}" if path else key
            raise ValueError(
                f"Unknown config key '{full_path}'. "
                f"Valid keys: {sorted(known_keys)}"
            )

    kwargs = {}
    for fname, fld in field_map.items():
        if fname not in d:
            continue

        value = d[fname]
        ftype = fld.type

        # Resolve string annotations (from __future__ annotations)
        if isinstance(ftype, str):
            ftype = eval(ftype, {k: v for k, v in globals().items()})

        dc_type = _unwrap_dataclass_type(ftype)
        if dc_type is not None and isinstance(value, dict):
            child_path = f"{path}.{fname}" if path else fname
            kwargs[fname] = _from_dict(dc_type, value, path=child_path, ignored_keys=ignored_keys)
        elif dc_type is not None and value is None:
            kwargs[fname] = None
        else:
            kwargs[fname] = value

    return cls(**kwargs)


def _gpu_ids_to_n(gpu_ids: str | None) -> int | None:
    """Derive n_gpus from gpu_ids string like '0,1,2'."""
    if gpu_ids is None:
        return None
    return len([x for x in gpu_ids.split(",") if x.strip()])


def _coerce_value(value_str: str, field_type):
    """Auto-cast a string value to the appropriate Python type for apply_overrides."""
    # Handle None/null
    if value_str.lower() in ("null", "none"):
        return None

    # Resolve string type annotations
    if isinstance(field_type, str):
        field_type = eval(field_type, {k: v for k, v in globals().items()})

    # Unwrap Optional (X | None)
    origin = get_origin(field_type)
    actual_type = field_type
    if origin is type(int | str):
        args = [a for a in get_args(field_type) if a is not type(None)]
        if args:
            actual_type = args[0]

    # bool (before int, since bool is subclass of int)
    if actual_type is bool:
        low = value_str.lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
        raise ValueError(f"Cannot parse '{value_str}' as bool")

    # int
    if actual_type is int:
        return int(value_str)

    # float
    if actual_type is float:
        return float(value_str)

    # list — try JSON parse
    if actual_type is list or get_origin(actual_type) is list:
        try:
            parsed = json.loads(value_str)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        # Fall through to string
        return value_str

    # str (default)
    return value_str


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag with conservative parsing."""
    import os

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_trust_remote_code(value: bool | None = None, *, context: str = "model loading") -> bool:
    """Resolve the remote-code execution opt-in for HF/vLLM model loading.

    Public-safe default is ``False``. Existing internal flows that rely on
    custom model repository code can explicitly opt in either through the new
    config fields or by setting ``OPD_ALLOW_TRUST_REMOTE_CODE=1`` while configs
    migrate. A warning is emitted whenever the unsafe opt-in is active.
    """
    env_enabled = _env_flag(_TRUST_REMOTE_CODE_ENV, default=False)
    resolved = env_enabled if value is None else bool(value) or env_enabled
    if resolved:
        warnings.warn(
            f"{context}: trust_remote_code is enabled and executes code from the model repository. "
            "Use only with trusted, pinned model sources.",
            UserWarning,
            stacklevel=2,
        )
    return resolved


def warn_unsafe_opt_in(enabled: bool, *, context: str, detail: str) -> None:
    """Emit a consistent warning for explicit unsafe compatibility opt-ins."""
    if enabled:
        warnings.warn(
            f"{context}: {detail}",
            UserWarning,
            stacklevel=2,
        )


# ============================================================================ #
# Sub-dataclasses                                                                #
# ============================================================================ #

@dataclass
class ModelConfig:
    path: str = ""  # required — validated in OPDConfig.validate()
    eos_token_id: int | None = None
    trust_remote_code: bool = False


@dataclass
class VLLMTeacherConfig:
    tensor_parallel_size: int = 1
    n_logprobs: int | None = None
    max_model_len: int = 18688
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = True
    disable_fast_logprobs: bool = False
    block_size: int | None = None


@dataclass
class HFTeacherConfig:
    use_torch_compile: bool = False


@dataclass
class RayConfig:
    node: str | None = None
    use_ray_actor: bool = False
    n_dp: int = 1


@dataclass
class TeacherConfig:
    path: str = ""  # required — validated
    backend: str = "vllm"
    gpu_ids: str | None = None
    n_gpus: int | None = None
    n_nodes: int = 1
    scoring_batch_size: int | None = None  # default depends on backend: 4 for hf, 32 for vllm
    bind_address: str = "127.0.0.1"
    dtype: str = "auto"
    trust_remote_code: bool = False
    vllm: VLLMTeacherConfig = field(default_factory=VLLMTeacherConfig)
    hf: HFTeacherConfig = field(default_factory=HFTeacherConfig)
    ray: RayConfig = field(default_factory=RayConfig)


@dataclass
class DataConfig:
    train_files: str = ""  # required — validated
    val_files: str | None = None
    prompt_key: str = "prompt"
    prompt_source: str = "raw"
    filter_key: str | None = None
    filter_value: str | int | float | bool | None = None
    completion_key: str | None = None
    prompt_template: str | None = None
    answer_key: str | None = None
    solution_key: str | None = None
    tokenizer_path: str | None = None
    enable_thinking: bool = False
    teacher_enable_thinking: bool | None = None
    max_prompt_length: int = 2048
    max_response_length: int = 16384
    post_eval_datasets: list = field(default_factory=list)
    allow_pickle_teacher_logits: bool = False


@dataclass
class VLLMRolloutConfig:
    max_num_seqs: int = 512
    max_model_len: int = 18432
    gpu_memory_utilization: float = 0.85
    colocated_gpu_memory_utilization: float | None = None
    max_num_batched_tokens: int | None = None
    tensor_parallel_size: int = 1
    block_size: int | None = None
    enforce_eager: bool = True


@dataclass
class RolloutConfig:
    backend: str = "vllm"
    gpu_ids: str | None = None
    n_gpus: int | None = None
    n_nodes: int = 1
    pin_cpu_affinity: bool = False
    bind_numa_memory: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    dtype: str = "auto"
    quantization: str | None = None
    trust_remote_code: bool = False
    vllm: VLLMRolloutConfig = field(default_factory=VLLMRolloutConfig)
    ray: RayConfig = field(default_factory=RayConfig)


@dataclass
class OptimConfig:
    lr: float = 1e-5
    lr_decay_style: str = "constant"
    lr_warmup_steps_ratio: float = 0.0
    weight_decay: float = 0.0
    min_lr: float = 0.0
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0


@dataclass
class LoRAConfig:
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    modules_to_save: list | None = None
    native_lora: bool = True


@dataclass
class MegatronConfig:
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 1
    use_distributed_optimizer: bool = True
    use_native_megatron: bool = True
    use_transformer_engine: bool = True


@dataclass
class TrainerConfig:
    gpu_ids: str | None = None
    n_gpus: int | None = None
    n_nodes: int = 1
    backend: str = "fsdp"
    dtype: str = "bfloat16"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    optim: OptimConfig = field(default_factory=OptimConfig)
    batch_size: int = 256
    micro_batch_size: int = 2
    mini_batch_size: int | None = None
    use_sequence_packing: bool = False
    kl_chunk_size: int = 1024
    total_steps: int = 1_000_000_000
    total_epochs: int = 1
    save_freq: int = -1
    save_optimizer: bool = True
    resume_from: str | None = None
    use_torch_compile: bool = False
    lora: LoRAConfig | None = None
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    ray: RayConfig = field(default_factory=RayConfig)


@dataclass
class OPDAlgorithmConfig:
    kl_loss_mode: str = "forward_kl"
    rollout_student_topk_k: int = 64
    pg_kl_n_total_samples: int = 16
    pg_clip_eps: float = 0.2
    use_importance_sampling: bool = True
    pg_online_advantage: bool = False
    kl_token_clip: float = 0.0
    skewed_alpha: float = 0.5
    n_kl_logprobs: int = 10
    use_decoupled_loss: bool = False
    behave_imp_weight_cap: float = 5.0
    pg_m2po_budget: float | None = None
    pg_m2po_miniclip_low: float = 0.3
    pg_m2po_miniclip_high: float = 0.5
    pg_actor_critic: bool = False
    pg_value_mode: str = "gae"
    pg_gae_lambda: float = 0.95
    pg_value_coef: float = 0.5
    pg_value_normalize_advantages: bool = False
    pg_token_weighted_backward: bool = False
    mof_variant: str = "lite"  # lite | full
    mof_partition: str = "two_group"  # two_group | eos_candidate_rest
    mof_eta_mass: float = 0.0
    mof_eta_odds: float = 0.5
    mof_lambda_odds: float = 1.0
    mof_eps: float = 1e-8
    mof_deduplicate_candidates: bool = True
    teacher_artifact_mode: str = "legacy"  # legacy | direct | hidden_recompute
    teacher_hidden_dtype: str = "bfloat16"
    teacher_hidden_semantics: str = "lm_head_input"
    teacher_hidden_recompute_materialization: str = "lazy"  # lazy | canonical


@dataclass
class GRPOAlgorithmConfig:
    group_size: int = 5
    clip_eps: float = 0.2
    kl_beta: float = 0.0
    kl_type: str = "k1"
    loss_agg_mode: str = "token-mean"
    norm_adv_by_std: bool = True
    filter_groups: bool = False
    overlong_buffer_len: int = 0
    overlong_penalty_factor: float = 1.0
    clip_ratio_low: float | None = None
    clip_ratio_high: float | None = None
    clip_ratio_c: float | None = 3.0
    reward_fn: str = "correctness"
    answer_pattern: str | None = None
    use_decoupled_loss: bool = False
    behave_imp_weight_cap: float = 5.0


@dataclass
class SFTAlgorithmConfig:
    loss_mode: str = "ce"
    ce_alpha: float = 0.8


@dataclass
class RewardConfig:
    fn: str = "correctness"
    answer_pattern: str | None = None


@dataclass
class AlgorithmConfig:
    mode: str = "opd"
    reward: RewardConfig = field(default_factory=RewardConfig)
    opd: OPDAlgorithmConfig = field(default_factory=OPDAlgorithmConfig)
    grpo: GRPOAlgorithmConfig = field(default_factory=GRPOAlgorithmConfig)
    sft: SFTAlgorithmConfig = field(default_factory=SFTAlgorithmConfig)


@dataclass
class StepOffStreamingConfig:
    enabled: bool = False
    rollout_backend: str = "async_sample"
    ordering: str = "strict"
    dispatch_unit: str = "logical_batch"
    teacher_emit_unit: str = "scoring_batch"
    teacher_transport: str = "coordinator"  # coordinator | direct_trainer
    max_scored_buffer_batches: int | None = None


@dataclass
class NStepOffConfig:
    step_off: int = 2
    implementation: str = "classic"  # classic | streaming
    streaming: StepOffStreamingConfig = field(default_factory=StepOffStreamingConfig)


@dataclass
class FullyAsyncConfig:
    staleness_threshold: int = 8
    evict_stale: bool = False
    pause_mode: str = "keep"


@dataclass
class FusedHybridSyncConfig:
    rollout_parallelism: str = "spmd_tp"
    rollout_dp_size: int | None = None
    weight_update_backend: str = "bucketed_inprocess"
    debug_full_state_sync: bool = False
    update_bucket_mb: int = 256
    vllm_sleep_level: int = 2
    require_vllm_sleep: bool = True
    verify_weight_checksum: bool = False
    allow_cpu_fallback_debug: bool = False
    refresh_policy: str = "lazy_before_rollout"
    allow_teacher_gpu_overlap: bool = False
    require_multigpu_fsdp: bool = True
    allow_single_gpu_debug: bool = False
    capability_probe: bool = True
    log_memory: bool = True


@dataclass
class PipelineConfig:
    scheduling_mode: str = "n_step_off"
    deployment: str = "local"
    ray_address: str = "local"
    n_step_off: NStepOffConfig = field(default_factory=NStepOffConfig)
    fully_async: FullyAsyncConfig = field(default_factory=FullyAsyncConfig)
    fused_hybrid_sync: FusedHybridSyncConfig = field(default_factory=FusedHybridSyncConfig)
    # Deprecated compatibility alias. New configs should use
    # pipeline.n_step_off.implementation + pipeline.n_step_off.streaming.
    step_off_streaming: StepOffStreamingConfig | None = None


def get_step_off_streaming_config(pipeline: PipelineConfig) -> StepOffStreamingConfig:
    """Return canonical n-step-off streaming knobs.

    The old top-level ``pipeline.step_off_streaming`` field is still accepted
    while configs migrate, but runtime code should read the canonical nested
    section.
    """
    if pipeline.step_off_streaming is not None:
        return pipeline.step_off_streaming
    return pipeline.n_step_off.streaming


def uses_step_off_streaming(pipeline: PipelineConfig) -> bool:
    """Whether n-step-off should use the streaming implementation."""
    if pipeline.n_step_off.implementation == "streaming":
        return True
    legacy = pipeline.step_off_streaming
    return bool(legacy and legacy.enabled)


@dataclass
class EvalConfig:
    freq: int = -1
    mode: list = field(default_factory=lambda: ["post_allgpu"])
    before_train: bool = True
    batch_size: int | None = None
    n_samples: int = 32
    temperature: float = 1.0
    max_response_length: int | None = None
    # Which checkpoints post-eval should evaluate. Defaults preserve the
    # historical behavior of evaluating every saved checkpoint (plus step 0
    # when before_train=True).
    checkpoint_policy: str = "all"  # all | final | steps
    checkpoint_steps: list[int] = field(default_factory=list)
    # Whether post-eval should run the config's primary data.val_files. Set
    # false for eval-only jobs that only add data.post_eval_datasets.
    run_primary: bool = True
    allow_unsafe_code_execution: bool = False


@dataclass
class WeightSyncConfig:
    backend: str = "nccl"
    verify_checksum: bool = False
    nccl_timeout_hours: int = 2
    nccl_socket_ifname: str | None = None
    ray_collective: bool = False


@dataclass
class ClearMLConfig:
    project: str = "async-opd"
    task_name: str | None = None


@dataclass
class WandbConfig:
    project: str = "async-opd"
    name: str | None = None


@dataclass
class AimConfig:
    experiment: str = "async-opd"
    repo: str | None = None  # Path or remote URL; None = default ./.aim


@dataclass
class LoggingConfig:
    clearml: ClearMLConfig | None = None
    wandb: WandbConfig | None = None
    aim: AimConfig | None = None


# ============================================================================ #
# OPDConfig — main config class                                                  #
# ============================================================================ #

@dataclass
class OPDConfig:
    """Top-level OPD pipeline configuration.

    Flat, code-aligned config format. Use from_yaml() to load from YAML,
    to_internal_dict() to translate back to the old nested format consumed
    by existing pipeline code.
    """

    deterministic: bool = False
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    teacher: TeacherConfig | None = None
    data: DataConfig = field(default_factory=DataConfig)
    rollout: RolloutConfig | None = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Raw YAML dict — kept for _DELETED_to_internal_dict backward compat
    _raw: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    # from_yaml                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> OPDConfig:
        """Load an OPDConfig from a YAML file.

        Processing pipeline:
        1. Parse YAML
        2. Normalize (gpu_ids -> n_gpus, eval.mode string -> list)
        3. Load into dataclasses via _from_dict()
        4. Apply --set overrides
        5. Derive computed values
        6. Validate
        """
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

        # --- Normalize ---
        _normalize_gpu_ids(raw)
        _normalize_eval_mode(raw)

        # --- Load sub-dataclasses ---
        kwargs: dict[str, Any] = {}

        # Top-level scalars
        if "deterministic" in raw:
            kwargs["deterministic"] = raw["deterministic"]
        if "seed" in raw:
            kwargs["seed"] = raw["seed"]

        # model
        if "model" in raw:
            kwargs["model"] = _from_dict(ModelConfig, raw["model"], path="model")
        # teacher (optional)
        if "teacher" in raw:
            kwargs["teacher"] = _from_dict(TeacherConfig, raw["teacher"], path="teacher")
        # data
        if "data" in raw:
            kwargs["data"] = _from_dict(DataConfig, raw["data"], path="data")
        # rollout (optional)
        if "rollout" in raw:
            kwargs["rollout"] = _from_dict(RolloutConfig, raw["rollout"], path="rollout")
        # trainer
        if "trainer" in raw:
            kwargs["trainer"] = _from_dict(TrainerConfig, raw["trainer"], path="trainer")
        # algorithm
        if "algorithm" in raw:
            kwargs["algorithm"] = _from_dict(AlgorithmConfig, raw["algorithm"], path="algorithm")
        # pipeline
        if "pipeline" in raw:
            kwargs["pipeline"] = _from_dict(PipelineConfig, raw["pipeline"], path="pipeline")
        # eval
        if "eval" in raw:
            kwargs["eval"] = _from_dict(EvalConfig, raw["eval"], path="eval")
        # weight_sync
        if "weight_sync" in raw:
            kwargs["weight_sync"] = _from_dict(WeightSyncConfig, raw["weight_sync"], path="weight_sync")
        # logging
        if "logging" in raw:
            kwargs["logging"] = _from_dict(LoggingConfig, raw["logging"], path="logging")

        # Check for unknown top-level keys
        known_top = {
            "deterministic", "seed", "model", "teacher", "data", "rollout",
            "trainer", "algorithm", "pipeline", "eval", "weight_sync", "logging",
        }
        for key in raw:
            if key not in known_top and key not in _IGNORED_LEGACY_KEYS:
                raise ValueError(
                    f"Unknown top-level config key '{key}'. "
                    f"Valid keys: {sorted(known_top)}"
                )

        cfg = cls(**kwargs)

        # Store raw YAML for to_internal_dict() to know which keys were explicit
        import copy
        cfg._raw = copy.deepcopy(raw)

        # --- Apply overrides ---
        if overrides:
            cfg.apply_overrides(overrides)
            # Update _raw so overridden keys survive _prune_internal_dict()
            for ov in overrides:
                key, val = ov.split("=", 1)
                parts = key.split(".")
                d = cfg._raw
                for p in parts[:-1]:
                    d = d.setdefault(p, {})
                d[parts[-1]] = val

        # --- Derive computed values ---
        cfg._derive()

        # --- Validate ---
        cfg.validate()

        return cfg

    # ------------------------------------------------------------------ #
    # validate                                                             #
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Validate the config, raising ValueError for hard errors."""
        mode = self.algorithm.mode

        # Valid enums — check early before using mode for branching
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid algorithm.mode '{mode}'. Must be one of: {sorted(_VALID_MODES)}"
            )

        self._normalize_step_off_streaming()

        is_sft = mode == "sft"
        is_opsd = mode == "opsd"

        if self.data.prompt_source not in _VALID_PROMPT_SOURCES:
            raise ValueError(
                f"Invalid data.prompt_source {self.data.prompt_source!r}. "
                f"Must be one of: {sorted(_VALID_PROMPT_SOURCES)}"
            )
        filter_is_set = self.data.filter_key is not None or self.data.filter_value is not None
        if filter_is_set:
            if self.data.filter_key is None or self.data.filter_value is None:
                raise ValueError("data.filter_key and data.filter_value must be set together")
            if mode not in {"opd", "opsd"}:
                raise ValueError(
                    "data.filter_key/filter_value are supported only for "
                    "algorithm.mode in {'opd', 'opsd'}; "
                    f"got algorithm.mode={mode!r}"
                )

        # Required fields
        if not self.model.path:
            raise ValueError("model.path is required")
        if not self.data.train_files:
            raise ValueError("data.train_files is required")
        if self.eval.checkpoint_policy not in _VALID_EVAL_CHECKPOINT_POLICIES:
            raise ValueError(
                f"Invalid eval.checkpoint_policy '{self.eval.checkpoint_policy}'. "
                f"Must be one of: {sorted(_VALID_EVAL_CHECKPOINT_POLICIES)}"
            )
        if not isinstance(self.eval.checkpoint_steps, list):
            raise ValueError("eval.checkpoint_steps must be a list of step integers")
        for step in self.eval.checkpoint_steps:
            if not isinstance(step, int) or step < 0:
                raise ValueError(
                    "eval.checkpoint_steps must contain non-negative integers, "
                    f"got {step!r}"
                )
        resolve_trust_remote_code(self.model.trust_remote_code, context="model.trust_remote_code")
        resolve_trust_remote_code(self.trainer.trust_remote_code, context="trainer.trust_remote_code")
        if self.teacher is not None:
            resolve_trust_remote_code(self.teacher.trust_remote_code, context="teacher.trust_remote_code")
        if self.rollout is not None:
            resolve_trust_remote_code(self.rollout.trust_remote_code, context="rollout.trust_remote_code")
        warn_unsafe_opt_in(
            self.data.allow_pickle_teacher_logits,
            context="data.allow_pickle_teacher_logits",
            detail=(
                "pickle teacher-logit columns are trusted-data compatibility only; "
                "never enable for third-party datasets."
            ),
        )
        warn_unsafe_opt_in(
            self.eval.allow_unsafe_code_execution,
            context="eval.allow_unsafe_code_execution",
            detail=(
                "inline generated-code scoring runs submitted code on this host. "
                "Prefer the sandboxed grading workflow for untrusted generations."
            ),
        )

        # trainer needs n_gpus or gpu_ids
        if self.trainer.n_gpus is None and self.trainer.gpu_ids is None:
            raise ValueError("trainer.n_gpus or trainer.gpu_ids is required")

        # OPSD requires solution_key or answer_key
        if is_opsd:
            if not self.data.solution_key and not self.data.answer_key:
                raise ValueError(
                    "OPSD mode requires data.solution_key or data.answer_key "
                    "to provide privileged information for self-distillation."
                )

        # Teacher required for non-SFT modes (except GRPO with kl_beta=0)
        if not is_sft and not is_opsd:
            is_grpo_no_kl = (mode == "grpo" and self.algorithm.grpo.kl_beta == 0.0)
            if not is_grpo_no_kl and self.teacher is None:
                raise ValueError(
                    f"teacher section is required for mode='{mode}' "
                    f"(or set algorithm.grpo.kl_beta=0 for DAPO without teacher)"
                )
            if self.teacher is not None and not self.teacher.path:
                raise ValueError("teacher.path is required when teacher section is present")

        # Rollout required for non-SFT modes
        if not is_sft and self.rollout is None:
            # This is acceptable — rollout can be auto-derived
            pass

        kl_mode = self.algorithm.opd.kl_loss_mode
        if kl_mode not in _VALID_KL_MODES:
            raise ValueError(
                f"Invalid algorithm.opd.kl_loss_mode '{kl_mode}'. "
                f"Must be one of: {sorted(_VALID_KL_MODES)}"
            )
        if kl_mode in {"reverse_kl_rollout_student_topk", "thunlp_opd_default_loss"}:
            if self.trainer.backend != "fsdp":
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "requires trainer.backend='fsdp'"
                )
            if self.rollout is None or self.rollout.backend not in {"vllm", "hf"}:
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "requires rollout.backend in {'vllm', 'hf'}"
                )
            if self.teacher is None or self.teacher.backend not in {"vllm", "hf"}:
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "requires teacher.backend in {'vllm', 'hf'}"
                )
            if self.pipeline.deployment != "local":
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "requires pipeline.deployment='local'"
                )
            if self.teacher.backend == "vllm" and self.teacher.vllm.disable_fast_logprobs:
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "requires teacher.vllm.disable_fast_logprobs=False"
                )
            if self.algorithm.opd.rollout_student_topk_k < 1:
                raise ValueError(
                    "algorithm.opd.rollout_student_topk_k must be >= 1 when "
                    f"kl_loss_mode='{kl_mode}'"
                )
        if self.algorithm.opd.mof_variant not in {"lite", "full"}:
            raise ValueError(
                "algorithm.opd.mof_variant must be one of {'lite', 'full'}, "
                f"got {self.algorithm.opd.mof_variant!r}"
            )
        if self.algorithm.opd.mof_partition not in {"two_group", "eos_candidate_rest"}:
            raise ValueError(
                "algorithm.opd.mof_partition must be one of "
                "{'two_group', 'eos_candidate_rest'}, "
                f"got {self.algorithm.opd.mof_partition!r}"
            )
        if not (0.0 <= self.algorithm.opd.mof_eta_mass <= 1.0):
            raise ValueError(
                "algorithm.opd.mof_eta_mass must be between 0 and 1, "
                f"got {self.algorithm.opd.mof_eta_mass}"
            )
        if not (0.0 <= self.algorithm.opd.mof_eta_odds <= 1.0):
            raise ValueError(
                "algorithm.opd.mof_eta_odds must be between 0 and 1, "
                f"got {self.algorithm.opd.mof_eta_odds}"
            )
        if self.algorithm.opd.mof_lambda_odds < 0:
            raise ValueError(
                "algorithm.opd.mof_lambda_odds must be >= 0, "
                f"got {self.algorithm.opd.mof_lambda_odds}"
            )
        if self.algorithm.opd.mof_eps <= 0:
            raise ValueError(
                "algorithm.opd.mof_eps must be > 0, "
                f"got {self.algorithm.opd.mof_eps}"
            )

        if kl_mode in {"multi_sample_policy_gradient_kl", "multi_sample_forward_kl", "mof_opd"}:
            if self.trainer.backend != "fsdp":
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "requires trainer.backend='fsdp'"
                )
            if self.rollout is None or self.rollout.backend not in {"vllm", "hf"}:
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "requires rollout.backend in {'vllm', 'hf'}"
                )
            if self.teacher is None or self.teacher.backend not in {"vllm", "hf"}:
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "requires teacher.backend in {'vllm', 'hf'}"
                )
            if self.pipeline.deployment != "local":
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "requires pipeline.deployment='local'"
                )
            if self.teacher.backend == "vllm" and self.teacher.vllm.disable_fast_logprobs:
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "requires teacher.vllm.disable_fast_logprobs=False"
                )
            if self.algorithm.opd.pg_actor_critic:
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "does not support actor-critic in v1"
                )
            if self.algorithm.opd.use_decoupled_loss:
                raise ValueError(
                    f"algorithm.opd.kl_loss_mode='{kl_mode}' "
                    "does not support use_decoupled_loss=True in v1"
                )
            if self.algorithm.opd.pg_kl_n_total_samples < 1:
                raise ValueError(
                    "algorithm.opd.pg_kl_n_total_samples must be >= 1 when "
                    f"kl_loss_mode='{kl_mode}'"
                )
            if kl_mode == "mof_opd" and self.algorithm.opd.mof_partition == "eos_candidate_rest":
                if self.algorithm.opd.pg_kl_n_total_samples <= 1:
                    raise ValueError(
                        "algorithm.opd.mof_partition='eos_candidate_rest' currently "
                        "requires algorithm.opd.pg_kl_n_total_samples > 1 so EOS "
                        "logprobs can be routed through the query/MC path"
                    )
                if self.model.eos_token_id is None:
                    raise ValueError(
                        "algorithm.opd.mof_partition='eos_candidate_rest' requires "
                        "model.eos_token_id to be set"
                    )

        for em in self.eval.mode:
            if em not in _VALID_EVAL_MODES:
                raise ValueError(
                    f"Invalid eval.mode entry '{em}'. "
                    f"Must be one of: {sorted(_VALID_EVAL_MODES)}"
                )

        # Eval mode combination validation
        eval_modes = set(self.eval.mode)
        gen_modes = eval_modes & {"inline", "post", "post_allgpu"}
        if len(gen_modes) > 1 and "perplexity" not in eval_modes:
            raise ValueError(
                f"eval.mode {sorted(eval_modes)}: inline, post, and post_allgpu are "
                f"mutually exclusive. Use one, or combine perplexity with post_allgpu."
            )
        if "perplexity" in eval_modes and eval_modes & {"inline", "post"}:
            raise ValueError(
                f"eval.mode {sorted(eval_modes)}: perplexity cannot be combined with "
                f"inline or post. Use [perplexity, post_allgpu] instead."
            )

        if self.trainer.backend not in _VALID_TRAINER_BACKENDS:
            raise ValueError(
                f"Invalid trainer.backend '{self.trainer.backend}'. "
                f"Must be one of: {sorted(_VALID_TRAINER_BACKENDS)}"
            )

        if self.rollout is not None and self.rollout.backend not in _VALID_ROLLOUT_BACKENDS:
            raise ValueError(
                f"Invalid rollout.backend '{self.rollout.backend}'. "
                f"Must be one of: {sorted(_VALID_ROLLOUT_BACKENDS)}"
            )

        if self.teacher is not None and self.teacher.backend not in _VALID_TEACHER_BACKENDS:
            raise ValueError(
                f"Invalid teacher.backend '{self.teacher.backend}'. "
                f"Must be one of: {sorted(_VALID_TEACHER_BACKENDS)}"
            )

        if self.weight_sync.backend not in _VALID_WEIGHT_SYNC_BACKENDS:
            raise ValueError(
                f"Invalid weight_sync.backend '{self.weight_sync.backend}'. "
                f"Must be one of: {sorted(_VALID_WEIGHT_SYNC_BACKENDS)}"
            )

        if self.pipeline.scheduling_mode not in _VALID_SCHEDULING_MODES:
            raise ValueError(
                f"Invalid pipeline.scheduling_mode '{self.pipeline.scheduling_mode}'. "
                f"Must be one of: {sorted(_VALID_SCHEDULING_MODES)}"
            )

        if self.pipeline.deployment not in _VALID_DEPLOYMENTS:
            raise ValueError(
                f"Invalid pipeline.deployment '{self.pipeline.deployment}'. "
                f"Must be one of: {sorted(_VALID_DEPLOYMENTS)}"
            )
        for label, ray_cfg in [
            ("teacher.ray", self.teacher.ray if self.teacher is not None else None),
            ("trainer.ray", self.trainer.ray),
            ("rollout.ray", self.rollout.ray if self.rollout is not None else None),
        ]:
            if ray_cfg is not None and int(ray_cfg.n_dp) < 1:
                raise ValueError(f"{label}.n_dp must be >= 1")

        sft_loss = self.algorithm.sft.loss_mode
        if sft_loss not in _VALID_SFT_LOSS_MODES:
            raise ValueError(
                f"Invalid algorithm.sft.loss_mode '{sft_loss}'. "
                f"Must be one of: {sorted(_VALID_SFT_LOSS_MODES)}"
            )

        # gpu_ids must be strings
        for label, val in [
            ("teacher.gpu_ids", self.teacher.gpu_ids if self.teacher else None),
            ("trainer.gpu_ids", self.trainer.gpu_ids),
            ("rollout.gpu_ids", self.rollout.gpu_ids if self.rollout else None),
        ]:
            if val is not None and not isinstance(val, str):
                raise ValueError(
                    f"'{label}' must be a string (e.g. \"0,1,2\"), "
                    f"got {type(val).__name__}: {val!r}"
                )

        # n_gpus divisible by tensor_parallel_size
        if self.teacher is not None and self.teacher.n_gpus is not None:
            tp = self.teacher.vllm.tensor_parallel_size
            if tp > 0 and self.teacher.n_gpus % tp != 0:
                raise ValueError(
                    f"teacher.n_gpus ({self.teacher.n_gpus}) must be divisible "
                    f"by teacher.vllm.tensor_parallel_size ({tp})"
                )

        # partial_rollout + policy_gradient_kl incompatible
        # (partial_rollout is a dead-code field but still checked)

        # scheduling_mode + step_off conflict warning
        if (self.pipeline.scheduling_mode == "fully_async"
                and self.pipeline.n_step_off.step_off != 2):  # non-default
            warnings.warn(
                f"pipeline.scheduling_mode=fully_async but pipeline.n_step_off.step_off="
                f"{self.pipeline.n_step_off.step_off} is also set. "
                "n_step_off is ignored in fully_async mode."
            )

        sos = self.pipeline.n_step_off.streaming
        sos_enabled = uses_step_off_streaming(self.pipeline)
        teacher_artifact_mode = self.algorithm.opd.teacher_artifact_mode
        hidden_materialization = self.algorithm.opd.teacher_hidden_recompute_materialization
        if teacher_artifact_mode not in {"legacy", "direct", "hidden_recompute"}:
            raise ValueError(
                "algorithm.opd.teacher_artifact_mode must be one of "
                "{'legacy', 'direct', 'hidden_recompute'}"
            )
        if hidden_materialization not in {"lazy", "canonical"}:
            raise ValueError(
                "algorithm.opd.teacher_hidden_recompute_materialization must be one of "
                "{'lazy', 'canonical'}"
            )
        if self.algorithm.opd.teacher_hidden_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(
                "algorithm.opd.teacher_hidden_dtype must be one of "
                "{'bfloat16', 'float16', 'float32'}"
            )
        if self.algorithm.opd.teacher_hidden_semantics not in {"lm_head_input", "pre_final_norm"}:
            raise ValueError(
                "algorithm.opd.teacher_hidden_semantics must be one of "
                "{'lm_head_input', 'pre_final_norm'}"
            )
        if self.pipeline.scheduling_mode == "fused_hybrid_sync":
            self._validate_fused_hybrid_sync()
        if sos_enabled:
            if self.pipeline.scheduling_mode != "n_step_off":
                raise ValueError(
                    "pipeline.n_step_off.implementation='streaming' requires "
                    "pipeline.scheduling_mode='n_step_off'"
                )
            if mode != "opd":
                raise ValueError(
                    "pipeline.n_step_off.implementation='streaming' currently supports "
                    "algorithm.mode='opd' only"
                )
            if self.pipeline.deployment != "local":
                raise ValueError(
                    "pipeline.n_step_off.implementation='streaming' currently requires "
                    "pipeline.deployment='local'"
                )
            if self.rollout is None or self.rollout.backend != "vllm":
                raise ValueError(
                    "pipeline.n_step_off.streaming.rollout_backend='async_sample' "
                    "requires rollout.backend='vllm'"
                )
            if sos.rollout_backend != "async_sample":
                raise ValueError(
                    "pipeline.n_step_off.streaming.rollout_backend must be "
                    "'async_sample'"
                )
            if sos.ordering != "strict":
                raise ValueError(
                    "pipeline.n_step_off.streaming.ordering must be 'strict'"
                )
            if sos.dispatch_unit != "logical_batch":
                raise ValueError(
                    "pipeline.n_step_off.streaming.dispatch_unit must be "
                    "'logical_batch'"
                )
            if sos.teacher_emit_unit != "scoring_batch":
                raise ValueError(
                    "pipeline.n_step_off.streaming.teacher_emit_unit must be "
                    "'scoring_batch'"
                )
            if sos.teacher_transport not in {"coordinator", "direct_trainer"}:
                raise ValueError(
                    "pipeline.n_step_off.streaming.teacher_transport must be "
                    "'coordinator' or 'direct_trainer'"
                )
            if self.pipeline.n_step_off.step_off < 0:
                raise ValueError("pipeline.n_step_off.step_off must be >= 0")
        elif (
            sos.teacher_transport == "direct_trainer"
            and teacher_artifact_mode not in {"direct", "hidden_recompute"}
        ):
            raise ValueError(
                "pipeline.n_step_off.streaming.teacher_transport='direct_trainer' "
                "requires algorithm.opd.teacher_artifact_mode in {'direct', 'hidden_recompute'}"
            )

        if teacher_artifact_mode in {"direct", "hidden_recompute"}:
            if sos.teacher_transport != "direct_trainer":
                raise ValueError(
                    f"teacher_artifact_mode={teacher_artifact_mode} requires "
                    "pipeline.n_step_off.streaming.teacher_transport='direct_trainer'"
                )
            if self.pipeline.scheduling_mode != "n_step_off":
                raise ValueError(
                    f"teacher_artifact_mode={teacher_artifact_mode} requires "
                    "pipeline.scheduling_mode='n_step_off'"
                )
            if self.pipeline.deployment != "local":
                raise ValueError(
                    f"teacher_artifact_mode={teacher_artifact_mode} requires local multiprocessing; "
                    "Ray is not supported yet"
                )
            if self.trainer.backend != "fsdp":
                raise ValueError(
                    f"teacher_artifact_mode={teacher_artifact_mode} requires trainer.backend='fsdp'"
                )
            if self.teacher is None or self.teacher.backend not in {"vllm", "hf"}:
                raise ValueError(
                    f"teacher_artifact_mode={teacher_artifact_mode} requires teacher.backend in {{'vllm', 'hf'}}"
                )
            if self.rollout is None or self.rollout.backend not in {"vllm", "hf"}:
                raise ValueError(
                    f"teacher_artifact_mode={teacher_artifact_mode} requires rollout.backend in {{'vllm', 'hf'}}"
                )
        if teacher_artifact_mode == "hidden_recompute":
            if not sos_enabled:
                raise ValueError(
                    "teacher_artifact_mode=hidden_recompute requires "
                    "pipeline.n_step_off.implementation='streaming'"
                )
            if self.teacher is None or self.teacher.backend != "vllm":
                raise ValueError(
                    "teacher_artifact_mode=hidden_recompute requires teacher.backend='vllm'"
                )
            if self.teacher.vllm.tensor_parallel_size != 1:
                raise ValueError(
                    "teacher_artifact_mode=hidden_recompute requires teacher.vllm.tensor_parallel_size=1"
                )
            if self.rollout is None or self.rollout.backend != "vllm":
                raise ValueError(
                    "teacher_artifact_mode=hidden_recompute requires rollout.backend='vllm'"
                )
            if mode != "opd":
                raise ValueError(
                    "teacher_artifact_mode=hidden_recompute currently supports algorithm.mode='opd' only"
                )
            if self.algorithm.opd.kl_loss_mode not in {"forward_kl", "reverse_kl", "skewed_kl"}:
                raise ValueError(
                    "teacher_artifact_mode=hidden_recompute currently supports dense OPD KL modes "
                    "{'forward_kl', 'reverse_kl', 'skewed_kl'} only"
                )
        # LoRA + native_lora requires nccl weight_sync
        if self.trainer.lora is not None and self.trainer.lora.native_lora:
            if self.weight_sync.backend != "nccl":
                raise ValueError(
                    "native_lora requires weight_sync.backend=nccl "
                    "(NCCL engine needed for base model init, LoRA sync via queue)"
                )

        # ce_alpha in [0, 1]
        ce_alpha = self.algorithm.sft.ce_alpha
        if not (0.0 <= ce_alpha <= 1.0):
            raise ValueError(
                f"algorithm.sft.ce_alpha must be between 0 and 1, got {ce_alpha}"
            )

        # SFT loss mode warning
        if is_sft and sft_loss in ("kl", "mixed"):
            warnings.warn(
                f"algorithm.sft.loss_mode='{sft_loss}' requires data with "
                "teacher logit columns (teacher_topk_logps, teacher_topk_indices)."
            )


    @staticmethod
    def _split_explicit_gpu_ids(label: str, value: str | None,
                                *, mode_name: str = "fused_hybrid_sync") -> list[str]:
        if value is None or not str(value).strip():
            raise ValueError(
                f"{label} must be explicitly set for pipeline.scheduling_mode='{mode_name}'"
            )
        ids = [item.strip() for item in str(value).split(",") if item.strip()]
        if not ids:
            raise ValueError(
                f"{label} must contain at least one GPU id for "
                f"pipeline.scheduling_mode='{mode_name}'"
            )
        if len(ids) != len(set(ids)):
            raise ValueError(f"{label} contains duplicate GPU ids: {value!r}")
        return ids

    def _validate_fused_hybrid_sync(self) -> None:
        """Fail-closed MVP validation for OPD fused hybrid scheduling."""
        fh = self.pipeline.fused_hybrid_sync
        mode_name = "fused_hybrid_sync"

        if self.pipeline.deployment != "local":
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' currently requires "
                "pipeline.deployment='local'"
            )
        if self.algorithm.mode != "opd":
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' supports "
                "algorithm.mode='opd' only"
            )
        if self.teacher is None or not self.teacher.path:
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' requires an independent "
                "teacher section with teacher.path"
            )
        if self.rollout is None:
            raise ValueError("pipeline.scheduling_mode='fused_hybrid_sync' requires rollout config")
        if self.trainer.backend != "fsdp":
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' MVP requires trainer.backend='fsdp'"
            )
        if self.rollout.backend != "vllm":
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' MVP requires rollout.backend='vllm'"
            )
        if self.weight_sync.backend == "nccl":
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' must not use native "
                "trainer-to-rollout NCCL weight sync because trainer and rollout are "
                "colocated on the same physical GPUs. Use weight_sync.backend='cpu' "
                "and pipeline.fused_hybrid_sync.weight_update_backend='bucketed_inprocess'."
            )
        if self.algorithm.opd.teacher_artifact_mode != "legacy":
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' MVP requires "
                "algorithm.opd.teacher_artifact_mode='legacy'"
            )
        if self.pipeline.n_step_off.step_off != 0:
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' is synchronous and "
                "requires pipeline.n_step_off.step_off=0"
            )

        teacher_gpus = self._split_explicit_gpu_ids("teacher.gpu_ids", self.teacher.gpu_ids,
                                                    mode_name=mode_name)
        trainer_gpus = self._split_explicit_gpu_ids("trainer.gpu_ids", self.trainer.gpu_ids,
                                                    mode_name=mode_name)
        rollout_gpus = self._split_explicit_gpu_ids("rollout.gpu_ids", self.rollout.gpu_ids,
                                                    mode_name=mode_name)

        if trainer_gpus != rollout_gpus:
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' requires trainer.gpu_ids "
                "and rollout.gpu_ids to be identical for the colocated student group "
                f"(got trainer={self.trainer.gpu_ids!r}, rollout={self.rollout.gpu_ids!r})"
            )

        student_world_size = len(trainer_gpus)
        if fh.require_multigpu_fsdp and student_world_size < 2 and not fh.allow_single_gpu_debug:
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' MVP requires student FSDP "
                "world size >= 2. Set pipeline.fused_hybrid_sync.allow_single_gpu_debug=true "
                "only for non-signoff debug runs."
            )
        rollout_parallelism = str(fh.rollout_parallelism)
        if rollout_parallelism not in {"spmd_tp", "data_parallel"}:
            raise ValueError(
                "pipeline.fused_hybrid_sync.rollout_parallelism must be one of "
                "{'spmd_tp', 'data_parallel'}"
            )
        if fh.rollout_dp_size is not None and int(fh.rollout_dp_size) != student_world_size:
            raise ValueError(
                "pipeline.fused_hybrid_sync.rollout_dp_size must match the student "
                f"FSDP world size ({student_world_size}) for the MVP; got {fh.rollout_dp_size}"
            )
        if rollout_parallelism == "spmd_tp":
            if self.rollout.vllm.tensor_parallel_size != student_world_size:
                raise ValueError(
                    "pipeline.scheduling_mode='fused_hybrid_sync' with "
                    "rollout_parallelism='spmd_tp' requires rollout.vllm.tensor_parallel_size "
                    "to equal the student FSDP world size "
                    f"({student_world_size}); got {self.rollout.vllm.tensor_parallel_size}"
                )
        else:
            if self.rollout.vllm.tensor_parallel_size != 1:
                raise ValueError(
                    "pipeline.scheduling_mode='fused_hybrid_sync' with "
                    "rollout_parallelism='data_parallel' supports "
                    "rollout.vllm.tensor_parallel_size=1 only; DP×TP subgroup layouts "
                    f"are not supported yet (got {self.rollout.vllm.tensor_parallel_size})."
                )
            rollout_pp_size = int(getattr(self.rollout.vllm, "pipeline_parallel_size", 1))
            if rollout_pp_size != 1:
                raise ValueError(
                    "pipeline.scheduling_mode='fused_hybrid_sync' with "
                    "rollout_parallelism='data_parallel' supports pipeline_parallel_size=1 only"
                )
            rollout_n_gpus = int(self.rollout.n_gpus or len(rollout_gpus))
            trainer_n_gpus = int(self.trainer.n_gpus or len(trainer_gpus))
            if rollout_n_gpus != student_world_size or trainer_n_gpus != student_world_size:
                raise ValueError(
                    "pipeline.scheduling_mode='fused_hybrid_sync' with "
                    "rollout_parallelism='data_parallel' requires rollout.n_gpus and "
                    "trainer.n_gpus to match the colocated student world size "
                    f"({student_world_size}); got rollout.n_gpus={self.rollout.n_gpus}, "
                    f"trainer.n_gpus={self.trainer.n_gpus}"
                )
        if not fh.allow_teacher_gpu_overlap and set(teacher_gpus) & set(trainer_gpus):
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' requires teacher.gpu_ids "
                "to be disjoint from the colocated student trainer/rollout GPU set. "
                "For constrained local debug only, set "
                "pipeline.fused_hybrid_sync.allow_teacher_gpu_overlap=true."
            )
        fused_eval_modes = set(self.eval.mode)
        unsupported_fused_eval = fused_eval_modes - {"inline", "post_allgpu"}
        if unsupported_fused_eval:
            raise ValueError(
                "pipeline.scheduling_mode='fused_hybrid_sync' supports eval.mode "
                "[inline] during training or [post_allgpu] after shutdown; "
                f"unsupported modes: {sorted(unsupported_fused_eval)}"
            )
        if fh.weight_update_backend != "bucketed_inprocess":
            if fh.weight_update_backend == "full_state" and fh.debug_full_state_sync:
                pass
            else:
                raise ValueError(
                    "pipeline.fused_hybrid_sync.weight_update_backend must be "
                    "'bucketed_inprocess'. Full-state sync requires "
                    "debug_full_state_sync=true and is non-signoff."
                )
        if fh.debug_full_state_sync and fh.weight_update_backend != "full_state":
            raise ValueError(
                "pipeline.fused_hybrid_sync.debug_full_state_sync=true is only valid "
                "with weight_update_backend='full_state'"
            )
        if int(fh.update_bucket_mb) <= 0:
            raise ValueError("pipeline.fused_hybrid_sync.update_bucket_mb must be > 0")
        if int(fh.vllm_sleep_level) not in {1, 2}:
            raise ValueError("pipeline.fused_hybrid_sync.vllm_sleep_level must be 1 or 2")
        if not fh.require_vllm_sleep:
            raise ValueError(
                "pipeline.fused_hybrid_sync.require_vllm_sleep=false is not supported "
                "for signoff fused_hybrid_sync runs"
            )
        if fh.refresh_policy not in {"after_train", "lazy_before_rollout"}:
            raise ValueError(
                "pipeline.fused_hybrid_sync.refresh_policy must be 'after_train' "
                "or 'lazy_before_rollout'"
            )

    def _normalize_step_off_streaming(self) -> None:
        """Canonicalize deprecated top-level step-off streaming config.

        Older configs used ``pipeline.step_off_streaming.enabled`` as both the
        implementation selector and the knob namespace. New configs select the
        implementation via ``pipeline.n_step_off.implementation`` and store
        knobs under ``pipeline.n_step_off.streaming``.
        """
        impl = self.pipeline.n_step_off.implementation
        if impl not in {"classic", "streaming"}:
            raise ValueError(
                "pipeline.n_step_off.implementation must be 'classic' or 'streaming'"
            )

        legacy = self.pipeline.step_off_streaming
        if legacy is not None:
            default_streaming = StepOffStreamingConfig()
            if self.pipeline.n_step_off.streaming == default_streaming:
                self.pipeline.n_step_off.streaming = copy.deepcopy(legacy)
            if legacy.enabled and impl == "classic":
                self.pipeline.n_step_off.implementation = "streaming"
            self.pipeline.step_off_streaming = None

        if self.pipeline.n_step_off.implementation == "streaming":
            self.pipeline.n_step_off.streaming.enabled = True

    # ------------------------------------------------------------------ #
    # apply_overrides                                                      #
    # ------------------------------------------------------------------ #

    def apply_overrides(self, overrides: list[str]) -> None:
        """Apply dot-path overrides like ['trainer.optim.lr=2e-5', 'eval.n_samples=1'].

        Raises ValueError for unknown key paths or unparseable values.
        """
        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Override must be in 'key=value' format, got: {override!r}"
                )
            key_path, value_str = override.split("=", 1)
            parts = key_path.strip().split(".")
            value_str = value_str.strip()

            # Traverse the dataclass hierarchy
            obj = self
            for i, part in enumerate(parts[:-1]):
                if not hasattr(obj, part):
                    raise ValueError(
                        f"Unknown config path: '{'.'.join(parts[:i+1])}' "
                        f"(from override '{override}')"
                    )
                child = getattr(obj, part)
                if child is None:
                    # Auto-instantiate None optional dataclasses
                    fld = _find_field(type(obj), part)
                    if fld is not None:
                        dc_type = _unwrap_dataclass_type(fld.type if not isinstance(fld.type, str) else eval(fld.type, globals()))
                        if dc_type is not None:
                            child = dc_type()
                            setattr(obj, part, child)
                        else:
                            raise ValueError(
                                f"Cannot traverse into None value at "
                                f"'{'.'.join(parts[:i+1])}' (from override '{override}')"
                            )
                    else:
                        raise ValueError(
                            f"Cannot traverse into None value at "
                            f"'{'.'.join(parts[:i+1])}' (from override '{override}')"
                        )
                obj = child

            # Set the final field
            final_key = parts[-1]
            if not hasattr(obj, final_key):
                raise ValueError(
                    f"Unknown config key: '{key_path}' (from override '{override}')"
                )

            fld = _find_field(type(obj), final_key)
            if fld is None:
                raise ValueError(
                    f"Unknown config key: '{key_path}' (from override '{override}')"
                )

            # Resolve type annotation
            ftype = fld.type
            if isinstance(ftype, str):
                ftype = eval(ftype, globals())

            coerced = _coerce_value(value_str, ftype)
            setattr(obj, final_key, coerced)

    # ------------------------------------------------------------------ #
    # to_yaml                                                              #
    # ------------------------------------------------------------------ #

    def to_yaml(self) -> str:
        """Serialize back to new-format YAML.

        Omits None-valued optional sections (teacher, rollout, logging sub-sections).
        """
        d = self._to_ordered_dict()
        return yaml.dump(d, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _derive(self) -> None:
        """Derive computed values after loading and overrides."""
        # Derive n_gpus from gpu_ids where needed
        if self.teacher is not None:
            if self.teacher.n_gpus is None and self.teacher.gpu_ids is not None:
                self.teacher.n_gpus = _gpu_ids_to_n(self.teacher.gpu_ids)
        if self.trainer.n_gpus is None and self.trainer.gpu_ids is not None:
            self.trainer.n_gpus = _gpu_ids_to_n(self.trainer.gpu_ids)
        if self.rollout is not None:
            if self.rollout.n_gpus is None and self.rollout.gpu_ids is not None:
                self.rollout.n_gpus = _gpu_ids_to_n(self.rollout.gpu_ids)

        # Derive teacher.vllm.n_logprobs from algorithm.opd.n_kl_logprobs when null
        if self.teacher is not None and self.teacher.vllm.n_logprobs is None:
            self.teacher.vllm.n_logprobs = self.algorithm.opd.n_kl_logprobs

        # Derive teacher.n_workers (n_gpus // tp) — stored as n_gpus field
    def _to_ordered_dict(self) -> dict:
        """Convert to a plain dict preserving field order, omitting None sections."""
        d: dict[str, Any] = {}

        if self.deterministic:
            d["deterministic"] = self.deterministic
        d["seed"] = self.seed

        d["model"] = dataclasses.asdict(self.model)

        if self.teacher is not None:
            d["teacher"] = _dataclass_to_dict_skip_none_sections(self.teacher)

        d["data"] = _dataclass_to_dict_skip_none_sections(self.data)

        if self.rollout is not None:
            d["rollout"] = _dataclass_to_dict_skip_none_sections(self.rollout)

        d["trainer"] = _dataclass_to_dict_skip_none_sections(self.trainer)
        d["algorithm"] = _dataclass_to_dict_skip_none_sections(self.algorithm)
        d["pipeline"] = _dataclass_to_dict_skip_none_sections(self.pipeline)
        d["eval"] = _dataclass_to_dict_skip_none_sections(self.eval)
        d["weight_sync"] = _dataclass_to_dict_skip_none_sections(self.weight_sync)

        if (self.logging.clearml is not None or self.logging.wandb is not None
                or self.logging.aim is not None):
            d["logging"] = _dataclass_to_dict_skip_none_sections(self.logging)

        return d


# ============================================================================ #
# Module-level helpers for OPDConfig                                             #
# ============================================================================ #

def _find_field(cls, name: str):
    """Find a dataclass field by name, or None."""
    for f in fields(cls):
        if f.name == name:
            return f
    return None


def _normalize_gpu_ids(raw: dict) -> None:
    """Normalize gpu_ids <-> n_gpus across all sections."""
    for section_key in ("teacher", "trainer", "rollout"):
        section = raw.get(section_key)
        if not isinstance(section, dict):
            continue
        gpu_ids = section.get("gpu_ids")
        n_gpus = section.get("n_gpus")
        if gpu_ids is not None and n_gpus is None:
            section["n_gpus"] = _gpu_ids_to_n(str(gpu_ids))
        elif n_gpus is not None and gpu_ids is None:
            # n_gpus without gpu_ids is fine — leave as is
            pass


def _normalize_eval_mode(raw: dict) -> None:
    """Normalize eval.mode from string to list."""
    ev = raw.get("eval")
    if not isinstance(ev, dict):
        return
    mode = ev.get("mode")
    if isinstance(mode, str):
        ev["mode"] = [mode]


def _dataclass_to_dict_skip_none_sections(obj) -> dict:
    """Convert a dataclass to a dict, skipping None-valued fields that are
    Optional dataclass types (keeps None for scalar Optional fields like str|None)."""
    result = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if val is None:
            # Resolve type to check if it's an optional dataclass
            ftype = f.type
            if isinstance(ftype, str):
                ftype = eval(ftype, globals())
            dc_type = _unwrap_dataclass_type(ftype)
            if dc_type is not None:
                # Skip None optional dataclass sections
                continue
            # Keep None scalar values
            result[f.name] = None
        elif dataclasses.is_dataclass(val):
            result[f.name] = _dataclass_to_dict_skip_none_sections(val)
        elif isinstance(val, list):
            result[f.name] = list(val)  # shallow copy
        else:
            result[f.name] = val
    return result

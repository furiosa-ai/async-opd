#!/usr/bin/env python3
"""Run integration tests for all OPD/GRPO loss modes × packing × schedulers.

Generates tiny public-safe fixtures when needed, generates configs from
templates, runs each selected test, parses log.jsonl + run.log, and checks
correctness assertions.

Tests:
  - vLLM OPD / GRPO integration coverage, including:
      * packed PG-KL across sync / step-off / async schedulers
      * AReaL mini-batch behavior
      * GRPO / DAPO
      * eval pipeline
      * native LoRA sync

  - Deterministic HF-backed tests for exact golden-loss regression coverage.

Usage:
    python scripts/run_integration_tests.py
    python scripts/run_integration_tests.py --suite fsdp
    python scripts/run_integration_tests.py --suite megatron
    python scripts/run_integration_tests.py --suite both
    python scripts/run_integration_tests.py --suite fsdp --allow-skip
    python scripts/run_integration_tests.py --configs forward_kl_nopack_so2 pg_kl_packed_async
    python scripts/run_integration_tests.py --list
    python scripts/run_integration_tests.py --filter pg_kl
"""

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import textwrap
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "integration_tests"
RESULTS_DIR = PROJECT_ROOT / "results" / "integration_tests"
PYTHON = sys.executable


def _env_timeout(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer number of seconds, got {raw!r}")
    if value <= 0:
        raise SystemExit(f"{name} must be positive, got {value}")
    return value


RUN_TIMEOUT_SECONDS = _env_timeout("OPD_INTEGRATION_TIMEOUT_SECONDS", 120)
LONG_TIMEOUT_SECONDS = _env_timeout("OPD_INTEGRATION_TIMEOUT_SECONDS", 180)

# Thresholds — same-model fp32 should have near-zero numerical error
KL_TOL = 5e-5
ADV_TOL = 5e-5
RATIO_TOL = 5e-5
GRPO_KL_TOL = 5e-5
MEGATRON_KL_TOL = KL_TOL
MEGATRON_ADV_TOL = ADV_TOL
MEGATRON_RATIO_TOL = RATIO_TOL
DENSE_RECOMPUTE_BASELINE_NAME = (
    "det_dense_reverse_kl_async_hidden_recompute_baseline_full_vocab_teacher_bs2"
)
DENSE_RECOMPUTE_CLASSIC_BASELINE_NAME = (
    "det_dense_reverse_kl_async_hidden_recompute_baseline_classic_so2_full_vocab_teacher_bs2"
)
DENSE_RECOMPUTE_EQUIV_TOL = 5e-7


def detect_local_gpu_count():
    """Best-effort local GPU count for safe integration-test scheduling."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        visible = [item.strip() for item in cuda_visible.split(",") if item.strip()]
        if visible:
            return len(visible)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if gpus:
            return len(gpus)
    except Exception:
        pass
    return 0


def required_gpu_count_for_config(cfg):
    """Return max referenced local GPU index + 1 for a generated config."""
    max_id = -1
    for section in ("teacher", "rollout", "trainer"):
        raw = cfg.get(section, {}).get("gpu_ids", "")
        for item in str(raw).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                max_id = max(max_id, int(item))
            except ValueError:
                continue
    return max_id + 1


def _integration_gpu_map() -> list[int] | None:
    """Optional logical→physical GPU mapping for constrained test hosts.

    Example: ``OPD_INTEGRATION_GPU_MAP=0,1,4,6`` rewrites generated
    ``gpu_ids`` so logical IDs 0,1,2,3 use physical GPUs 0,1,4,6.  This is
    useful on shared machines where some GPUs are unavailable but configs
    need a contiguous logical layout.
    """
    raw = os.environ.get("OPD_INTEGRATION_GPU_MAP", "").strip()
    if not raw:
        return None
    mapping = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            mapping.append(int(item))
        except ValueError:
            raise SystemExit(
                f"OPD_INTEGRATION_GPU_MAP must be comma-separated integers, got {raw!r}"
            )
    if not mapping:
        raise SystemExit("OPD_INTEGRATION_GPU_MAP did not contain any GPU IDs")
    return mapping


def _remap_gpu_ids_value(value, gpu_map: list[int]):
    parts = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            logical_id = int(item)
        except ValueError:
            parts.append(item)
            continue
        if logical_id >= len(gpu_map):
            raise ValueError(
                f"gpu_ids references logical GPU {logical_id}, but "
                f"OPD_INTEGRATION_GPU_MAP only has {len(gpu_map)} entries"
            )
        parts.append(str(gpu_map[logical_id]))
    return ",".join(parts)


def apply_integration_gpu_map(cfg):
    """Rewrite every generated gpu_ids field using OPD_INTEGRATION_GPU_MAP."""
    gpu_map = _integration_gpu_map()
    if gpu_map is None:
        return cfg

    def visit(node):
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key == "gpu_ids":
                    node[key] = _remap_gpu_ids_value(value, gpu_map)
                else:
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(cfg)
    return cfg


# ──────────────────────────────────────────────────────────────
#  Config generation
# ──────────────────────────────────────────────────────────────

TINY_STUDENT = str(PROJECT_ROOT / "tests" / "fixtures" / "tiny_student")
TINY_TEACHER = str(PROJECT_ROOT / "tests" / "fixtures" / "tiny_teacher")
TINY_STUDENT_2L = str(PROJECT_ROOT / "tests" / "fixtures" / "tiny_student_2l")
TINY_TEACHER_2L = str(PROJECT_ROOT / "tests" / "fixtures" / "tiny_teacher_2l")
SFT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "sft_tiny"
SFT_TRAIN_FILE = str(SFT_FIXTURE_DIR / "train.parquet")
SFT_VAL_FILE = str(SFT_FIXTURE_DIR / "val.parquet")
GSM8K_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "integration_gsm8k_size64"
GSM8K_TRAIN_FILE = str(GSM8K_FIXTURE_DIR / "train.parquet")
GSM8K_VAL_FILE = str(GSM8K_FIXTURE_DIR / "val.parquet")
GOLDEN_GSM8K_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "gsm8k_size64"
GOLDEN_GSM8K_TRAIN_FILE = str(GOLDEN_GSM8K_FIXTURE_DIR / "train.parquet")
GOLDEN_GSM8K_VAL_FILE = str(GOLDEN_GSM8K_FIXTURE_DIR / "test.parquet")


def suite_for_test(name):
    """Return the integration suite a test belongs to."""
    return "megatron" if name.startswith("megatron_") else "fsdp"


def _selected_by_suite(names, suite):
    """Filter test names by suite. ``both``/``all`` selects every suite."""
    if suite in {"both", "all"}:
        return list(names)
    return [n for n in names if suite_for_test(n) == suite]


def _module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _truthy_env(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def preflight_name_skip_reason(name):
    """Cheap name-only dependency gates that avoid expensive config generation."""
    if not name.startswith("megatron_"):
        return None
    if not _module_available("megatron.core"):
        return "Megatron suite requires megatron-core (missing import megatron.core)"
    if "ray" in name or "multinode" in name:
        if not _module_available("ray"):
            return "Ray Megatron integration requires ray (missing import ray)"
    if "vllm" in name and not _module_available("vllm"):
        return "Megatron vLLM weight-sync integration requires vllm"
    if "multinode" in name and not _truthy_env("OPD_ENABLE_MULTINODE_INTEGRATION"):
        return (
            "multi-node Megatron integration is gated; set "
            "OPD_ENABLE_MULTINODE_INTEGRATION=1 on a prepared Ray cluster"
        )
    if "packed" in name and not _module_available("transformer_engine"):
        return "Megatron packed-sequence integration requires transformer_engine"
    return None


def preflight_config_skip_reason(name, cfg):
    """Config-aware skip gates for hardware and optional runtime dependencies."""
    if name.startswith("megatron_"):
        trainer = cfg.get("trainer", {})
        meg = trainer.get("megatron", {})
        if meg.get("use_transformer_engine") and not _module_available("transformer_engine"):
            return "Megatron config requires transformer_engine"
        if cfg.get("pipeline", {}).get("deployment") == "ray" and not _module_available("ray"):
            return "Ray deployment requires ray"

    local_gpu_count = detect_local_gpu_count()
    required_gpu_count = required_gpu_count_for_config(cfg)
    if "multinode" in name and _truthy_env("OPD_ENABLE_MULTINODE_INTEGRATION"):
        return None
    if local_gpu_count and required_gpu_count > local_gpu_count:
        return (
            f"requires local GPU index {required_gpu_count - 1} "
            f"but only {local_gpu_count} GPU(s) are visible"
        )
    return None


def _tiny_student_vocab_size():
    with open(Path(TINY_STUDENT) / "config.json") as f:
        return int(json.load(f)["vocab_size"])


def _ensure_sft_fixture():
    """Create tiny public-safe SFT parquet fixtures if the checkout lacks them."""
    train_path = Path(SFT_TRAIN_FILE)
    val_path = Path(SFT_VAL_FILE)
    if train_path.exists() and val_path.exists():
        return
    import pandas as pd

    SFT_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "prompt": f"Write a short answer for problem {i}.",
            "completion": f" The answer is {i}.",
        }
        for i in range(16)
    ]
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    pd.DataFrame(rows[:8]).to_parquet(val_path, index=False)


def _ensure_gsm8k_fixture():
    """Create tiny public-safe math parquet fixtures if the checkout lacks them."""
    train_path = Path(GSM8K_TRAIN_FILE)
    val_path = Path(GSM8K_VAL_FILE)
    required_columns = {"prompt", "answer", "solution"}
    if train_path.exists() and val_path.exists():
        try:
            import pandas as pd

            train_columns = set(pd.read_parquet(train_path).columns)
            val_columns = set(pd.read_parquet(val_path).columns)
            if required_columns.issubset(train_columns) and required_columns.issubset(val_columns):
                return
        except Exception:
            pass

    import pandas as pd

    GSM8K_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "prompt": f"problem {i}",
            "answer": str(i),
            "solution": str(i),
        }
        for i in range(64)
    ]
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    pd.DataFrame(rows[:16]).to_parquet(val_path, index=False)


def _ensure_golden_gsm8k_fixture():
    """Validate the exact GSM8K fixture used by deterministic golden losses."""
    train_path = Path(GOLDEN_GSM8K_TRAIN_FILE)
    val_path = Path(GOLDEN_GSM8K_VAL_FILE)
    required_columns = {"prompt", "reward_model", "solution"}
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "deterministic golden tests require "
            f"{train_path.relative_to(PROJECT_ROOT)} and "
            f"{val_path.relative_to(PROJECT_ROOT)}"
        )

    import pandas as pd

    for path in (train_path, val_path):
        columns = set(pd.read_parquet(path).columns)
        missing = required_columns - columns
        if missing:
            raise ValueError(
                f"{path.relative_to(PROJECT_ROOT)} is not the deterministic "
                f"golden fixture; missing columns: {sorted(missing)}"
            )


def _create_tiny_model_from_base(seed, output_path, *, num_layers):
    """Create a tiny local fixture with enough layers for pipeline-parallel tests."""
    output_path = Path(output_path)
    if (output_path / "config.json").exists():
        return

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import torch

    base_model = Path(TINY_STUDENT)
    if not (base_model / "config.json").exists():
        subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "scripts" / "create_test_models.py")],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
    config = AutoConfig.from_pretrained(
        str(base_model),
        trust_remote_code=True,
        local_files_only=True,
    )
    overrides = dict(
        num_hidden_layers=num_layers,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["full_attention"] * num_layers,
    )
    for key, value in overrides.items():
        setattr(config, key, value)

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.save_pretrained(output_path)


def _ensure_two_layer_tiny_models():
    """Create 2-layer fixtures used by Megatron PP tests if missing."""
    _create_tiny_model_from_base(42, TINY_STUDENT_2L, num_layers=2)
    _create_tiny_model_from_base(123, TINY_TEACHER_2L, num_layers=2)


OPD_TEMPLATE = {
    "model": {"path": TINY_STUDENT},
    "teacher": {
        "path": TINY_STUDENT,
        "gpu_ids": "0",
        "scoring_batch_size": 8,
        "dtype": "float32",
        "vllm": {
            "gpu_memory_utilization": 0.05,
            "tensor_parallel_size": 1,
            "n_logprobs": 64,
            "max_model_len": 512,
            "max_num_seqs": 64,
            "enforce_eager": True,
        },
    },
    "data": {
        "train_files": GSM8K_TRAIN_FILE,
        "val_files": GSM8K_VAL_FILE,
        "prompt_key": "prompt",
        "max_prompt_length": 128,
        "max_response_length": 128,
    },
    "rollout": {
        "gpu_ids": "0",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "dtype": "float32",
        "n_gpus": 1,
        "vllm": {
            "gpu_memory_utilization": 0.05,
            "max_model_len": 512,
            "tensor_parallel_size": 1,
            "max_num_seqs": 64,
            "enforce_eager": True,
        },
    },
    "trainer": {
        "gpu_ids": "1",
        "n_gpus": 1,
        "batch_size": 8,
        "micro_batch_size": 4,
        "use_torch_compile": False,
        "use_sequence_packing": False,
        "dtype": "float32",
        "total_steps": 8,
        "total_epochs": 1,
        "optim": {"lr": 1e-5, "weight_decay": 0.0},
    },
    "algorithm": {
        "mode": "opd",
    },
    "pipeline": {
        "n_step_off": {"step_off": 0},
    },
    "eval": {
        "freq": -1,
        "before_train": False,
    },
    "weight_sync": {"backend": "nccl", "verify_checksum": True},
}


def _deep_copy(d):
    """Simple deep copy for nested dicts."""
    import copy
    return copy.deepcopy(d)


def _step_off_streaming_section(cfg, *, implementation=None):
    """Return canonical n-step-off streaming config section, creating it if needed."""
    n_step = cfg.setdefault("pipeline", {}).setdefault("n_step_off", {})
    if implementation is not None:
        n_step["implementation"] = implementation
    return n_step.setdefault("streaming", {})


def _enable_async_stepoff(cfg, step_off):
    cfg["pipeline"]["scheduling_mode"] = "n_step_off"
    cfg["pipeline"]["n_step_off"]["step_off"] = step_off
    stream = _step_off_streaming_section(cfg, implementation="streaming")
    stream.update({
        "rollout_backend": "async_sample",
        "ordering": "strict",
        "dispatch_unit": "logical_batch",
        "teacher_emit_unit": "scoring_batch",
    })


def make_opd_config(loss_mode, packed, scheduler, **overrides):
    """Generate an OPD integration test config."""
    cfg = _deep_copy(OPD_TEMPLATE)
    trainer = cfg["trainer"]
    algo = cfg["algorithm"]

    if overrides.get("deterministic"):
        # Deterministic OPD tests with golden losses must use the exact
        # fixture used when GOLDEN_LOSSES was captured.
        cfg["data"]["train_files"] = GOLDEN_GSM8K_TRAIN_FILE
        cfg["data"]["val_files"] = GOLDEN_GSM8K_VAL_FILE

    # Loss mode
    algo.setdefault("opd", {})["kl_loss_mode"] = loss_mode
    if loss_mode == "skewed_kl":
        algo["opd"]["skewed_alpha"] = 0.5
    elif loss_mode == "policy_gradient_kl":
        algo["opd"]["pg_clip_eps"] = 0.2
    elif loss_mode in {"multi_sample_policy_gradient_kl", "multi_sample_forward_kl"}:
        algo["opd"]["pg_kl_n_total_samples"] = overrides.get("pg_kl_n_total_samples", 16)
        cfg["rollout"]["vllm"]["enforce_eager"] = False
    elif loss_mode == "mof_opd":
        algo["opd"]["pg_kl_n_total_samples"] = overrides.get("pg_kl_n_total_samples", 16)
        algo["opd"]["mof_variant"] = overrides.get("mof_variant", "lite")
        algo["opd"]["mof_partition"] = overrides.get("mof_partition", "two_group")
        algo["opd"]["mof_eta_mass"] = overrides.get("mof_eta_mass", 0.0)
        algo["opd"]["mof_eta_odds"] = overrides.get("mof_eta_odds", 0.5)
        algo["opd"]["mof_lambda_odds"] = overrides.get("mof_lambda_odds", 1.0)
        if algo["opd"]["pg_kl_n_total_samples"] > 1:
            cfg["rollout"]["vllm"]["enforce_eager"] = False
        else:
            cfg["teacher"]["vllm"]["n_logprobs"] = 1
    elif loss_mode == "reverse_kl_rollout_student_topk":
        algo["opd"]["rollout_student_topk_k"] = overrides.get("rollout_student_topk_k", 8)
    elif loss_mode == "thunlp_opd_default_loss":
        algo["opd"]["rollout_student_topk_k"] = overrides.get("rollout_student_topk_k", 8)
    # token_level_kl and forward/reverse need only n_logprobs=1 for token-level
    if loss_mode in ("token_level_kl", "policy_gradient_kl"):
        cfg["teacher"]["vllm"]["n_logprobs"] = 1

    # Packing
    trainer["use_sequence_packing"] = packed

    # Scheduler
    if scheduler == "so0":
        cfg["pipeline"]["n_step_off"]["step_off"] = 0
    elif scheduler == "so2":
        cfg["pipeline"]["n_step_off"]["step_off"] = 2
    elif scheduler == "async":
        cfg["pipeline"]["scheduling_mode"] = "fully_async"
        cfg["pipeline"].setdefault("fully_async", {})["staleness_threshold"] = 2
    elif scheduler == "async_stepoff0":
        _enable_async_stepoff(cfg, 0)
    elif scheduler == "async_stepoff2":
        _enable_async_stepoff(cfg, 2)

    # Overrides
    for k, v in overrides.items():
        if k == "lr":
            trainer["optim"]["lr"] = v
        elif k == "mini_batch_size":
            trainer["mini_batch_size"] = v
        elif k == "use_decoupled_loss":
            algo["opd"]["use_decoupled_loss"] = v
            algo["opd"]["behave_imp_weight_cap"] = overrides.get("behave_imp_weight_cap", 5.0)
        elif k == "pg_actor_critic":
            algo["opd"]["pg_actor_critic"] = v
        elif k == "pg_value_mode":
            algo["opd"]["pg_value_mode"] = v
        elif k == "pg_gae_lambda":
            algo["opd"]["pg_gae_lambda"] = v
        elif k == "pg_value_coef":
            algo["opd"]["pg_value_coef"] = v
        elif k == "pg_value_normalize_advantages":
            algo["opd"]["pg_value_normalize_advantages"] = v
        elif k == "pg_token_weighted_backward":
            algo["opd"]["pg_token_weighted_backward"] = v
        elif k == "rollout_student_topk_k":
            algo["opd"]["rollout_student_topk_k"] = v
        elif k == "use_importance_sampling":
            algo["opd"]["use_importance_sampling"] = v
        elif k == "teacher_scoring_batch_size":
            cfg["teacher"]["scoring_batch_size"] = v
        elif k == "teacher_path":
            cfg["teacher"]["path"] = v
        elif k == "teacher_transport":
            _step_off_streaming_section(cfg)["teacher_transport"] = v
        elif k == "teacher_artifact_mode":
            algo["opd"]["teacher_artifact_mode"] = v
        elif k == "teacher_hidden_dtype":
            algo["opd"]["teacher_hidden_dtype"] = v
        elif k == "teacher_hidden_semantics":
            algo["opd"]["teacher_hidden_semantics"] = v
        elif k == "teacher_hidden_recompute_materialization":
            algo["opd"]["teacher_hidden_recompute_materialization"] = v
        elif k == "max_prompt_length":
            cfg["data"]["max_prompt_length"] = v
        elif k == "max_response_length":
            cfg["data"]["max_response_length"] = v
        elif k == "teacher_n_logprobs":
            cfg["teacher"]["vllm"]["n_logprobs"] = v
        elif k == "total_steps":
            trainer["total_steps"] = v
        elif k == "batch_size":
            trainer["batch_size"] = v
        elif k == "micro_batch_size":
            trainer["micro_batch_size"] = v
        elif k == "deterministic":
            cfg["deterministic"] = v
        elif k == "seed":
            cfg["seed"] = v
        elif k == "model_eos_token_id":
            cfg["model"]["eos_token_id"] = v
        elif k in {
            "pg_kl_n_total_samples",
            "mof_variant",
            "mof_partition",
            "mof_eta_mass",
            "mof_eta_odds",
            "mof_lambda_odds",
            "mof_eps",
            "mof_deduplicate_candidates",
        }:
            algo["opd"][k] = v

    return cfg


def make_grpo_config(scheduler):
    """Generate a GRPO integration test config."""
    cfg = {
        "model": {"path": TINY_STUDENT},
        "teacher": {
            "path": TINY_STUDENT,
            "gpu_ids": "0",
            "vllm": {
                "tensor_parallel_size": 1,
                "n_logprobs": 1,
                "max_model_len": 1024,
                "gpu_memory_utilization": 0.15,
                "enforce_eager": True,
            },
        },
        "data": {
            "train_files": "hf:openai/gsm8k:main",
            "val_files": "hf:openai/gsm8k:main",
            "prompt_template": "{problem}",
            "answer_key": "answer",
            "enable_thinking": False,
            "prompt_key": "question",
            "max_prompt_length": 128,
            "max_response_length": 128,
        },
        "rollout": {
            "gpu_ids": "1",
            "temperature": 1.0,
            "dtype": "float32",
            "n_gpus": 1,
            "vllm": {
                "max_num_seqs": 64,
                "max_model_len": 512,
                "gpu_memory_utilization": 0.05,
            },
        },
        "trainer": {
            "n_gpus": 1,
            "gpu_ids": "0",
            "batch_size": 4,
            "micro_batch_size": 4,
            "mini_batch_size": 2,
            "use_torch_compile": False,
            "dtype": "float32",
            "total_steps": 8,
            "total_epochs": 1,
            "optim": {"lr": 0, "lr_decay_style": "constant"},
        },
        "algorithm": {
            "mode": "grpo",
            "grpo": {
                "group_size": 4,
                "clip_eps": 0.2,
                "kl_beta": 1.0,
                "reward_fn": "token_hash",
            },
        },
        "pipeline": {
            "n_step_off": {"step_off": 0},
        },
        "eval": {
            "freq": -1,
            "before_train": False,
        },
        "weight_sync": {"backend": "nccl", "verify_checksum": True},
    }
    if scheduler == "so2":
        cfg["pipeline"]["n_step_off"]["step_off"] = 2
    elif scheduler == "async":
        cfg["pipeline"]["scheduling_mode"] = "fully_async"
        cfg["pipeline"].setdefault("fully_async", {})["staleness_threshold"] = 2
    return cfg


def make_dapo_config(scheduler):
    """Generate a DAPO integration test config.

    DAPO = GRPO + asymmetric clip + dual-clip + token-mean + filter_groups + no KL.
    """
    cfg = make_grpo_config(scheduler)
    grpo = cfg["algorithm"]["grpo"]
    # DAPO-specific settings
    grpo["kl_beta"] = 0.0           # no KL penalty
    grpo["clip_ratio_low"] = 0.2    # asymmetric clip
    grpo["clip_ratio_high"] = 0.28  # asymmetric clip
    grpo["clip_ratio_c"] = 10.0     # dual-clip
    grpo["loss_agg_mode"] = "token-mean"  # DAPO aggregation
    # filter_groups disabled: zero-variance groups contribute zero gradient anyway,
    # and filtering shrinks the batch (reducing n_mini). verl-opd doesn't filter.
    grpo["norm_adv_by_std"] = True
    # DAPO doesn't need a teacher (kl_beta=0) — remove it
    del cfg["teacher"]
    return cfg


def make_deterministic_config(loss_mode, packed=False, **overrides):
    """Generate a deterministic OPD integration test config (HF rollout backend)."""
    cfg = _deep_copy(OPD_TEMPLATE)
    trainer = cfg["trainer"]
    algo = cfg["algorithm"]

    # The GOLDEN_LOSSES table was captured against this exact 64-row GSM8K
    # fixture.  Do not route deterministic/golden tests through the smaller
    # public smoke fixture unless the golden table is intentionally recaptured.
    cfg["data"]["train_files"] = GOLDEN_GSM8K_TRAIN_FILE
    cfg["data"]["val_files"] = GOLDEN_GSM8K_VAL_FILE

    # Different teacher model → non-zero KL for all loss modes
    cfg["teacher"]["path"] = TINY_TEACHER

    # HF backends + CPU weight sync
    cfg["teacher"]["backend"] = "hf"
    cfg["rollout"]["backend"] = "hf"
    cfg["weight_sync"] = {"backend": "cpu", "verify_checksum": True}

    # Deterministic mode
    cfg["deterministic"] = True
    cfg["seed"] = 42

    # Single GPU — all components share GPU 0
    cfg["teacher"]["gpu_ids"] = "0"
    cfg["rollout"]["gpu_ids"] = "0"
    trainer["gpu_ids"] = "0"

    # Short sequences for speed
    cfg["data"]["max_prompt_length"] = 32
    cfg["data"]["max_response_length"] = 32
    trainer["batch_size"] = 4

    # Minimal steps, step-off=2 matches production scheduling
    trainer["total_steps"] = 4
    cfg["pipeline"]["n_step_off"]["step_off"] = 2

    # fp32 for exact reproducibility
    trainer["dtype"] = "float32"
    cfg["rollout"]["dtype"] = "float32"
    cfg["teacher"]["dtype"] = "float32"
    trainer["use_torch_compile"] = False

    # Loss mode
    algo.setdefault("opd", {})["kl_loss_mode"] = loss_mode
    if loss_mode == "skewed_kl":
        algo["opd"]["skewed_alpha"] = 0.5
    elif loss_mode == "policy_gradient_kl":
        algo["opd"]["pg_clip_eps"] = 0.2
    elif loss_mode in {"multi_sample_policy_gradient_kl", "multi_sample_forward_kl"}:
        algo["opd"]["pg_kl_n_total_samples"] = overrides.get("pg_kl_n_total_samples", 16)
    elif loss_mode == "mof_opd":
        algo["opd"]["pg_kl_n_total_samples"] = overrides.get("pg_kl_n_total_samples", 16)
        algo["opd"]["mof_variant"] = overrides.get("mof_variant", "lite")
        algo["opd"]["mof_partition"] = overrides.get("mof_partition", "two_group")
        algo["opd"]["mof_eta_mass"] = overrides.get("mof_eta_mass", 0.0)
        algo["opd"]["mof_eta_odds"] = overrides.get("mof_eta_odds", 0.5)
        algo["opd"]["mof_lambda_odds"] = overrides.get("mof_lambda_odds", 1.0)
        if algo["opd"]["pg_kl_n_total_samples"] <= 1:
            cfg["teacher"]["vllm"]["n_logprobs"] = 1
    elif loss_mode == "reverse_kl_rollout_student_topk":
        algo["opd"]["rollout_student_topk_k"] = overrides.get("rollout_student_topk_k", 8)
    if loss_mode in ("token_level_kl", "policy_gradient_kl"):
        cfg["teacher"]["vllm"]["n_logprobs"] = 1

    # Packing
    trainer["use_sequence_packing"] = packed

    # Overrides
    for k, v in overrides.items():
        if k == "lr":
            trainer["optim"]["lr"] = v
        elif k == "mini_batch_size":
            trainer["mini_batch_size"] = v
        elif k == "use_decoupled_loss":
            algo["opd"]["use_decoupled_loss"] = v
            algo["opd"]["behave_imp_weight_cap"] = overrides.get("behave_imp_weight_cap", 5.0)
        elif k == "pg_online_advantage":
            algo["opd"]["pg_online_advantage"] = v
        elif k == "m2po_budget":
            algo["opd"]["pg_m2po_budget"] = v
            algo["opd"]["pg_m2po_miniclip_low"] = overrides.get("m2po_miniclip_low", 0.3)
            algo["opd"]["pg_m2po_miniclip_high"] = overrides.get("m2po_miniclip_high", 0.5)
        elif k == "pg_actor_critic":
            algo["opd"]["pg_actor_critic"] = v
        elif k == "pg_value_mode":
            algo["opd"]["pg_value_mode"] = v
        elif k == "pg_gae_lambda":
            algo["opd"]["pg_gae_lambda"] = v
        elif k == "pg_value_coef":
            algo["opd"]["pg_value_coef"] = v
        elif k == "pg_value_normalize_advantages":
            algo["opd"]["pg_value_normalize_advantages"] = v
        elif k == "pg_token_weighted_backward":
            algo["opd"]["pg_token_weighted_backward"] = v
        elif k == "rollout_student_topk_k":
            algo["opd"]["rollout_student_topk_k"] = v
        elif k == "use_importance_sampling":
            algo["opd"]["use_importance_sampling"] = v
        elif k == "teacher_scoring_batch_size":
            cfg["teacher"]["scoring_batch_size"] = v
        elif k == "teacher_n_logprobs":
            cfg["teacher"]["vllm"]["n_logprobs"] = v
        elif k == "teacher_transport":
            _step_off_streaming_section(cfg)["teacher_transport"] = v
        elif k == "teacher_artifact_mode":
            algo["opd"]["teacher_artifact_mode"] = v
        elif k == "teacher_hidden_dtype":
            algo["opd"]["teacher_hidden_dtype"] = v
        elif k == "teacher_hidden_semantics":
            algo["opd"]["teacher_hidden_semantics"] = v
        elif k == "teacher_hidden_recompute_materialization":
            algo["opd"]["teacher_hidden_recompute_materialization"] = v
        elif k == "step_off":
            cfg["pipeline"]["n_step_off"]["step_off"] = v
        elif k == "total_steps":
            trainer["total_steps"] = v
        elif k == "model_eos_token_id":
            cfg["model"]["eos_token_id"] = v
        elif k in {
            "pg_kl_n_total_samples",
            "mof_variant",
            "mof_partition",
            "mof_eta_mass",
            "mof_eta_odds",
            "mof_lambda_odds",
            "mof_eps",
            "mof_deduplicate_candidates",
        }:
            algo["opd"][k] = v

    return cfg


def make_deterministic_actor_critic_config():
    """Generate a deterministic actor-critic PG-KL config."""
    return make_deterministic_config(
        "policy_gradient_kl",
        False,
        pg_actor_critic=True,
        pg_token_weighted_backward=True,
        # Use TD(0) here so the zero-initialized critic reproduces
        # standard PG-KL actor advantages at the first train step.
        pg_value_mode="td0",
        pg_gae_lambda=0.95,
        pg_value_coef=0.5,
        pg_value_normalize_advantages=False,
    )


def make_actor_critic_vllm_config():
    """Generate a vLLM actor-critic PG-KL config."""
    return make_opd_config(
        "policy_gradient_kl",
        False,
        "so2",
        pg_actor_critic=True,
        pg_token_weighted_backward=True,
        pg_value_mode="gae",
        pg_gae_lambda=0.95,
        pg_value_coef=0.5,
        pg_value_normalize_advantages=False,
    )


def make_deterministic_dapo_config():
    """Generate a deterministic DAPO integration test config.

    DAPO = GRPO + asymmetric clip + dual-clip + token-mean + filter_groups, no KL.
    Uses HF backend for bitwise reproducibility.
    """
    cfg = make_dapo_config("so2")
    cfg["deterministic"] = True
    cfg["seed"] = 42
    cfg["weight_sync"] = {"backend": "cpu", "verify_checksum": True}
    cfg["rollout"]["backend"] = "hf"
    cfg["rollout"]["dtype"] = "float32"
    cfg["trainer"]["dtype"] = "float32"
    cfg["trainer"]["use_torch_compile"] = False
    # Single GPU
    cfg["rollout"]["gpu_ids"] = "0"
    cfg["trainer"]["gpu_ids"] = "0"
    # Short sequences
    cfg["data"]["max_prompt_length"] = 32
    cfg["data"]["max_response_length"] = 32
    cfg["trainer"]["batch_size"] = 4
    cfg["trainer"]["total_steps"] = 4
    cfg["pipeline"]["n_step_off"]["step_off"] = 2
    # Use non-zero lr so loss is non-trivial for golden comparison
    cfg["trainer"]["optim"]["lr"] = 1e-5
    # Use token_hash reward: tiny model always gets 0 correctness reward → zero
    # advantages → zero loss. token_hash produces binary 0/1 based on response
    # content so DAPO clip/aggregation logic is actually exercised.
    cfg["algorithm"]["grpo"]["reward_fn"] = "token_hash"
    return cfg


def make_deterministic_grpo_config():
    """Generate a deterministic GRPO integration test config."""
    cfg = make_grpo_config("so2")
    # Override for deterministic mode
    cfg["deterministic"] = True
    cfg["seed"] = 42
    cfg["teacher"]["path"] = TINY_TEACHER
    cfg["teacher"]["backend"] = "hf"
    cfg["teacher"]["dtype"] = "float32"
    cfg["weight_sync"] = {"backend": "cpu", "verify_checksum": True}
    cfg["rollout"]["backend"] = "hf"
    cfg["rollout"]["dtype"] = "float32"
    cfg["trainer"]["dtype"] = "float32"
    cfg["trainer"]["use_torch_compile"] = False
    # Single GPU
    cfg["teacher"]["gpu_ids"] = "0"
    cfg["rollout"]["gpu_ids"] = "0"
    cfg["trainer"]["gpu_ids"] = "0"
    # Short sequences
    cfg["data"]["max_prompt_length"] = 32
    cfg["data"]["max_response_length"] = 32
    cfg["trainer"]["batch_size"] = 4
    cfg["trainer"]["total_steps"] = 4
    cfg["pipeline"]["n_step_off"]["step_off"] = 2
    return cfg


def make_deterministic_dapo_packed_config():
    """Generate a deterministic DAPO + sequence packing integration test config."""
    cfg = make_deterministic_dapo_config()
    cfg["trainer"]["use_sequence_packing"] = True
    return cfg


def make_deterministic_grpo_packed_config():
    """Generate a deterministic GRPO + sequence packing integration test config.

    Uses DAPO base (kl_beta=0) because packing + kl_beta > 0 is not supported
    (ref_token_logps alignment not implemented). Removes DAPO-specific settings
    to test vanilla GRPO packing.
    """
    cfg = make_deterministic_dapo_config()
    # Strip DAPO-specific settings to get vanilla GRPO + packing
    # Keep reward_fn=token_hash for non-trivial advantages
    grpo = cfg["algorithm"]["grpo"]
    for k in ["clip_ratio_low", "clip_ratio_high", "clip_ratio_c",
              "loss_agg_mode", "filter_groups", "norm_adv_by_std"]:
        grpo.pop(k, None)
    cfg["trainer"]["use_sequence_packing"] = True
    return cfg


def make_deterministic_eval_config():
    """Generate a deterministic eval pipeline test config."""
    cfg = make_deterministic_config("forward_kl", False)
    cfg["data"]["answer_key"] = "reward_model"
    cfg["eval"]["freq"] = 3
    cfg["eval"]["mode"] = ["post"]
    cfg["eval"]["before_train"] = True
    cfg["eval"]["batch_size"] = 64
    cfg["eval"]["n_samples"] = 1
    return cfg


def make_sft_config(*, total_steps=2, eval_perplexity=True):
    """Generate a tiny SFT config for trainer-only FSDP coverage."""
    _ensure_sft_fixture()
    cfg = {
        "model": {"path": TINY_STUDENT},
        "data": {
            "train_files": SFT_TRAIN_FILE,
            "val_files": SFT_VAL_FILE,
            "prompt_key": "prompt",
            "completion_key": "completion",
            "max_prompt_length": 32,
            "max_response_length": 16,
        },
        "trainer": {
            "backend": "fsdp",
            "gpu_ids": "0",
            "n_gpus": 1,
            "batch_size": 4,
            "micro_batch_size": 2,
            "mini_batch_size": 4,
            "use_torch_compile": False,
            "use_sequence_packing": False,
            "dtype": "float32",
            "total_steps": total_steps,
            "total_epochs": 1,
            "save_freq": max(1, int(total_steps)),
            "optim": {"lr": 1e-5, "weight_decay": 0.0},
        },
        "algorithm": {
            "mode": "sft",
            "sft": {"loss_mode": "ce", "ce_alpha": 1.0},
        },
        "eval": {
            "freq": 1 if eval_perplexity else -1,
            "mode": ["perplexity"],
            "before_train": False,
            "batch_size": 8,
            "n_samples": 1,
        },
        "weight_sync": {"backend": "cpu", "verify_checksum": False},
    }
    return cfg


def make_deterministic_token_level_config(packed):
    """Deterministic HF-backed token-level KL coverage."""
    return make_deterministic_config("token_level_kl", packed)


def make_opsd_vllm_config():
    """Tiny OPSD self-distillation config: rollout self-scores, no teacher process."""
    cfg = make_opd_config(
        "policy_gradient_kl",
        False,
        "so0",
        total_steps=2,
        batch_size=4,
        micro_batch_size=2,
        mini_batch_size=4,
        max_prompt_length=64,
        max_response_length=32,
    )
    cfg["algorithm"]["mode"] = "opsd"
    cfg["data"]["train_files"] = GSM8K_TRAIN_FILE
    cfg["data"]["val_files"] = GSM8K_VAL_FILE
    cfg["data"]["solution_key"] = "solution"
    cfg["data"]["answer_key"] = "solution"
    cfg.pop("teacher", None)
    cfg["pipeline"]["n_step_off"]["step_off"] = 0
    cfg["rollout"]["vllm"]["max_num_seqs"] = 16
    cfg["rollout"]["vllm"]["max_model_len"] = 512
    cfg["rollout"]["vllm"]["gpu_memory_utilization"] = 0.05
    return cfg


def make_deterministic_dapo_filter_overlong_config():
    """Deterministic DAPO config that turns on filter_groups and overlong shaping."""
    cfg = make_deterministic_dapo_config()
    grpo = cfg["algorithm"]["grpo"]
    grpo["filter_groups"] = True
    grpo["overlong_buffer_len"] = 4
    grpo["overlong_penalty_factor"] = 0.5
    cfg["data"]["max_response_length"] = 16
    return cfg


def make_public_grpo_4gpu_smoke_config():
    """Load the public 4-GPU GRPO example and shrink it to a one-step tiny smoke."""
    with open(PROJECT_ROOT / "configs" / "examples" / "grpo_gsm8k_0.5b_4gpu.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["path"] = TINY_STUDENT
    cfg["teacher"]["path"] = TINY_TEACHER
    cfg["teacher"]["dtype"] = "float32"
    cfg["teacher"]["scoring_batch_size"] = 4
    cfg["teacher"]["vllm"].update({
        "n_logprobs": 1,
        "max_model_len": 512,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.05,
        "enforce_eager": True,
    })
    cfg["data"].update({
        "train_files": GSM8K_TRAIN_FILE,
        "val_files": GSM8K_TRAIN_FILE,
        "prompt_key": "prompt",
        "prompt_template": None,
        "answer_key": "answer",
        "max_prompt_length": 32,
        "max_response_length": 32,
    })
    cfg["rollout"].update({
        "gpu_ids": "0,1",
        "n_gpus": 2,
        "dtype": "float32",
        "temperature": 1.0,
    })
    cfg["rollout"]["vllm"].update({
        "tensor_parallel_size": 1,
        "max_model_len": 512,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.05,
        "enforce_eager": True,
    })
    cfg["trainer"].update({
        "gpu_ids": "2,3",
        "n_gpus": 2,
        "batch_size": 4,
        "micro_batch_size": 1,
        "mini_batch_size": 2,
        "use_torch_compile": False,
        "dtype": "float32",
        "total_steps": 1,
        "total_epochs": 1,
        "save_freq": -1,
    })
    cfg["algorithm"]["grpo"].update({
        "group_size": 2,
        "kl_beta": 0.0,
        "kl_type": "k1",
        "reward_fn": "token_hash",
    })
    cfg["pipeline"]["n_step_off"]["step_off"] = 0
    cfg["eval"].update({"freq": -1, "before_train": False, "n_samples": 1})
    return cfg


def _apply_megatron_trainer(cfg, *, tp_size, pp_size, dp_size=1,
                            gpu_start=0, total_steps=1, use_te=False):
    """Switch a tiny config to the Megatron trainer backend."""
    n_gpus = tp_size * pp_size * dp_size
    gpu_ids = ",".join(str(gpu_start + i) for i in range(n_gpus))
    cfg["trainer"].update({
        "backend": "megatron",
        "gpu_ids": gpu_ids,
        "n_gpus": n_gpus,
        "batch_size": max(4, 2 * dp_size),
        "micro_batch_size": 1,
        "mini_batch_size": 2,
        "use_torch_compile": False,
        "use_sequence_packing": False,
        # These release-readiness checks assert same-model KL/advantage at
        # FSDP-like tolerances, so keep Megatron smoke tests in fp32.  The
        # backend itself still supports bf16 for production configs.
        "dtype": "float32",
        "total_steps": total_steps,
        "total_epochs": 1,
        "save_freq": -1,
        "megatron": {
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": pp_size,
            "use_native_megatron": True,
            "use_transformer_engine": use_te,
        },
    })
    return cfg


def make_megatron_opd_config(*, tp_size=1, pp_size=1, dp_size=1,
                             use_two_layer_model=False, vllm_sync=False,
                             ray=False, multinode=False,
                             loss_mode="forward_kl", same_model=True):
    """Generate a tiny Megatron OPD config for local/Ray backend smoke."""
    if use_two_layer_model:
        _ensure_two_layer_tiny_models()
        student = TINY_STUDENT_2L
        teacher = student if same_model else TINY_TEACHER_2L
    else:
        student = TINY_STUDENT
        teacher = student if same_model else TINY_TEACHER

    cfg = make_deterministic_config(
        loss_mode,
        False,
        step_off=0,
        total_steps=1,
        teacher_scoring_batch_size=2,
    )
    cfg["model"]["path"] = student
    cfg["teacher"]["path"] = teacher
    cfg["data"]["max_prompt_length"] = 32
    cfg["data"]["max_response_length"] = 16
    cfg["pipeline"]["n_step_off"]["step_off"] = 0
    cfg["eval"]["freq"] = -1
    cfg["eval"]["before_train"] = False
    _apply_megatron_trainer(
        cfg,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        total_steps=1,
        use_te=False,
    )

    if vllm_sync:
        # Production-like Megatron → vLLM NCCL weight transfer.  Rollout is on
        # the first GPU after the Megatron trainer group.
        rollout_gpu = tp_size * pp_size * dp_size
        cfg["teacher"]["backend"] = "hf"
        cfg["teacher"]["gpu_ids"] = str(rollout_gpu)
        cfg["rollout"]["backend"] = "vllm"
        cfg["rollout"]["gpu_ids"] = str(rollout_gpu)
        cfg["rollout"]["n_gpus"] = 1
        cfg["rollout"]["dtype"] = "float32"
        cfg["rollout"]["vllm"].update({
            "tensor_parallel_size": 1,
            "max_model_len": 512,
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.05,
            "enforce_eager": True,
        })
        cfg["weight_sync"] = {"backend": "nccl", "verify_checksum": True}
    else:
        cfg["teacher"]["backend"] = "hf"
        cfg["rollout"]["backend"] = "hf"
        cfg["teacher"]["gpu_ids"] = "0"
        cfg["rollout"]["gpu_ids"] = "0"
        cfg["weight_sync"] = {"backend": "cpu", "verify_checksum": True}

    if ray:
        cfg["pipeline"]["deployment"] = "ray"
        cfg["pipeline"]["ray_address"] = os.environ.get("OPD_RAY_ADDRESS", "auto")
        cfg["weight_sync"]["ray_collective"] = False
        if multinode:
            cfg["trainer"].setdefault("ray", {})["node"] = ["head", "remote"]
            cfg["rollout"].setdefault("ray", {})["node"] = "remote"
            cfg["teacher"].setdefault("ray", {})["node"] = "head"
    return cfg


def make_megatron_multinode_2x2_config():
    """Two-node Megatron Ray smoke that needs only two GPUs on each node.

    Resource shape:
      - head: one Ray teacher actor + one Megatron trainer rank
      - remote: one vLLM rollout actor + one Megatron trainer rank

    Megatron parallelism is TP=1, PP=1, DP=2.  ``trainer.gpu_ids`` is used
    only to infer the two trainer ranks; Ray assigns real per-node GPU IDs.
    """
    cfg = make_megatron_opd_config(
        tp_size=1,
        pp_size=1,
        dp_size=2,
        vllm_sync=True,
        ray=True,
        multinode=True,
        same_model=True,
    )
    cfg["trainer"]["gpu_ids"] = "0,1"
    cfg["trainer"]["n_gpus"] = 2
    cfg["trainer"].setdefault("ray", {})["node"] = ["head", "remote"]

    cfg["teacher"]["backend"] = "vllm"
    cfg["teacher"]["gpu_ids"] = "0"
    cfg["teacher"]["n_gpus"] = 1
    cfg["teacher"]["dtype"] = "float32"
    cfg["teacher"].setdefault("ray", {}).update({
        "use_ray_actor": True,
        "node": "head",
    })
    cfg["teacher"]["vllm"].update({
        "tensor_parallel_size": 1,
        "n_logprobs": 64,
        "max_model_len": 512,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.05,
        "enforce_eager": True,
    })

    cfg["rollout"]["backend"] = "vllm"
    cfg["rollout"]["gpu_ids"] = "0"
    cfg["rollout"]["n_gpus"] = 1
    cfg["rollout"].setdefault("ray", {})["node"] = "remote"
    cfg["rollout"]["vllm"].update({
        "tensor_parallel_size": 1,
        "max_model_len": 512,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.05,
        "enforce_eager": True,
    })

    cfg["weight_sync"] = {
        "backend": "nccl",
        "verify_checksum": True,
        "ray_collective": False,
    }
    return cfg


def make_megatron_grpo_config(*, tp_size=1, pp_size=1, dp_size=1):
    """Generate a tiny Megatron GRPO/DAPO config without reference teacher."""
    use_two_layer = pp_size > 1
    if use_two_layer:
        _ensure_two_layer_tiny_models()
        student = TINY_STUDENT_2L
    else:
        student = TINY_STUDENT
    cfg = make_deterministic_dapo_config()
    cfg["model"]["path"] = student
    cfg["data"]["max_prompt_length"] = 32
    cfg["data"]["max_response_length"] = 16
    cfg["trainer"]["batch_size"] = 4
    cfg["trainer"]["micro_batch_size"] = 1
    cfg["trainer"]["mini_batch_size"] = 2
    cfg["pipeline"]["n_step_off"]["step_off"] = 0
    _apply_megatron_trainer(
        cfg,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        total_steps=1,
        use_te=False,
    )
    cfg["rollout"]["backend"] = "hf"
    cfg["rollout"]["gpu_ids"] = "0"
    cfg["weight_sync"] = {"backend": "cpu", "verify_checksum": True}
    return cfg


def make_deterministic_lora_config():
    """Generate a deterministic LoRA integration test config.

    Uses HF rollout + CPU weight sync so the test remains bitwise reproducible
    while still exercising the trainer-side PEFT LoRA path and merge-then-sync
    broadcast into the rollout worker.
    """
    cfg = make_deterministic_config("policy_gradient_kl", False)
    cfg["trainer"]["lora"] = {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "native_lora": False,
    }
    return cfg


def make_deterministic_fsdp2_minibatch_config():
    """Deterministic 2-GPU FSDP config to verify staleness tracks actual optimizer steps.

    Uses tiny models + HF backend (CPU weight sync), fp32, deterministic mode.
    All 3 roles colocated on 2 GPUs: teacher on 0, rollout on 1, trainer FSDP on 0+1.
    train_batch_size=8, 2 trainer GPUs, mini_batch_size=4, step_off=2.
    mini_batch_size=4 global → 8/4=2 mini-batches → 2 optimizer steps per train.
    Final staleness = step_off(2) * 2 = 4.
    """
    cfg = _deep_copy(OPD_TEMPLATE)
    trainer = cfg["trainer"]

    # Deterministic mode (fp32, eager attention)
    cfg["deterministic"] = True
    cfg["seed"] = 42

    # HF backends + CPU weight sync
    cfg["teacher"]["backend"] = "hf"
    cfg["rollout"]["backend"] = "hf"
    cfg["weight_sync"] = {"backend": "cpu", "verify_checksum": True}

    # 2 GPUs: teacher HF on 0 (colocated with trainer rank 0),
    #         rollout HF on 1 (colocated with trainer rank 1),
    #         trainer FSDP on 0,1
    cfg["teacher"]["gpu_ids"] = "0"
    cfg["model"]["path"] = TINY_STUDENT
    cfg["teacher"]["path"] = TINY_STUDENT
    cfg["rollout"]["gpu_ids"] = "1"

    # fp32 for exact reproducibility
    trainer["dtype"] = "float32"
    cfg["rollout"]["dtype"] = "float32"
    cfg["teacher"]["dtype"] = "float32"

    trainer["optim"]["lr"] = 1e-5
    trainer["micro_batch_size"] = 2
    trainer["mini_batch_size"] = 4
    trainer["use_torch_compile"] = False
    trainer["use_sequence_packing"] = False

    # Short sequences
    cfg["data"]["max_prompt_length"] = 32
    cfg["data"]["max_response_length"] = 32
    trainer["batch_size"] = 8

    # 2-GPU FSDP trainer
    trainer["gpu_ids"] = "0,1"
    trainer["n_gpus"] = 2
    trainer["total_steps"] = 8
    cfg["eval"]["freq"] = 999
    cfg["eval"]["before_train"] = False
    cfg["pipeline"]["n_step_off"]["step_off"] = 2

    cfg["algorithm"]["opd"] = {"kl_loss_mode": "forward_kl"}
    cfg["rollout"]["n_gpus"] = 1

    return cfg


def _apply_uneven_global_mini_mc_settings(cfg):
    """Configure the local 2-GPU uneven global-mini MC-PG-KL scenario.

    Shape: global batch 10, global mini 5, FSDP world 2.  Each global mini is
    sharded 3/2 across ranks, so the old rank-first path would fail
    (per-rank batch 5, per-rank mini 2) while global-mini-first preserves the
    configured two optimizer steps without dropping or padding samples.
    """
    trainer = cfg["trainer"]

    cfg["data"]["max_prompt_length"] = 32
    cfg["data"]["max_response_length"] = 32
    cfg["data"]["train_files"] = GSM8K_TRAIN_FILE
    cfg["data"]["val_files"] = GSM8K_TRAIN_FILE
    cfg["pipeline"]["n_step_off"]["step_off"] = 0
    cfg["eval"]["freq"] = -1
    cfg["eval"]["before_train"] = False

    trainer["batch_size"] = 10
    trainer["mini_batch_size"] = 5
    trainer["micro_batch_size"] = 2
    trainer["total_steps"] = 4
    trainer["use_torch_compile"] = False
    trainer["use_sequence_packing"] = True
    trainer["gpu_ids"] = "0,1"
    trainer["n_gpus"] = 2

    cfg["algorithm"].setdefault("opd", {})
    cfg["algorithm"]["opd"].update({
        "kl_loss_mode": "multi_sample_policy_gradient_kl",
        "pg_kl_n_total_samples": 16,
        "pg_online_advantage": True,
        "pg_clip_eps": 1000000.0,
    })
    return cfg


def make_deterministic_uneven_global_mini_fsdp2_config():
    """Deterministic HF-backed local-2-GPU regression for uneven mini sharding."""
    cfg = make_deterministic_config(
        "multi_sample_policy_gradient_kl",
        packed=True,
        pg_kl_n_total_samples=16,
    )
    _apply_uneven_global_mini_mc_settings(cfg)

    # HF backends are colocated with the two trainer ranks; CPU weight sync keeps
    # this deterministic and avoids vLLM/NCCL requirements for golden capture.
    cfg["teacher"]["gpu_ids"] = "0"
    cfg["rollout"]["gpu_ids"] = "1"
    cfg["rollout"]["n_gpus"] = 1
    cfg["weight_sync"] = {"backend": "cpu", "verify_checksum": True}
    return cfg


def make_vllm_uneven_global_mini_fsdp2_config():
    """vLLM local-2-GPU smoke for the production-like uneven MC-PG-KL path."""
    cfg = make_opd_config(
        "multi_sample_policy_gradient_kl",
        packed=True,
        scheduler="so0",
        pg_kl_n_total_samples=16,
        batch_size=10,
        micro_batch_size=2,
        mini_batch_size=5,
        total_steps=1,
        max_prompt_length=32,
        max_response_length=32,
    )
    _apply_uneven_global_mini_mc_settings(cfg)
    cfg["trainer"]["total_steps"] = 1
    cfg["teacher"]["gpu_ids"] = "0"
    cfg["rollout"]["gpu_ids"] = "1"
    cfg["rollout"]["n_gpus"] = 1
    cfg["trainer"]["dtype"] = "float32"
    cfg["teacher"]["dtype"] = "float32"
    cfg["rollout"]["dtype"] = "float32"
    cfg["weight_sync"] = {"backend": "nccl", "verify_checksum": True}
    return cfg


def make_fused_hybrid_sync_student_teacher_kl0_fsdp2_config():
    """Local-2-GPU fused_hybrid_sync OPD smoke: student==teacher, KL≈0.

    Local workstations have only two GPUs, so this signoff smoke explicitly
    allows teacher/student overlap while preserving the production invariant in
    config validation by default.  The student trainer and rollout are fused on
    GPUs 0,1 with FSDP world size 2 and vLLM TP 2.
    """
    cfg = make_opd_config(
        "forward_kl",
        packed=True,
        scheduler="so0",
        batch_size=10,
        micro_batch_size=2,
        mini_batch_size=5,
        total_steps=3,
        max_prompt_length=32,
        max_response_length=32,
    )
    _apply_uneven_global_mini_mc_settings(cfg)
    cfg["algorithm"]["opd"]["kl_loss_mode"] = "forward_kl"
    cfg["trainer"]["total_steps"] = 3

    cfg["trainer"]["gpu_ids"] = "0,1"
    cfg["trainer"]["n_gpus"] = 2
    cfg["rollout"]["gpu_ids"] = "0,1"
    cfg["rollout"]["n_gpus"] = 2
    cfg["rollout"]["vllm"]["tensor_parallel_size"] = 2
    cfg["rollout"]["vllm"]["gpu_memory_utilization"] = 0.05
    cfg["rollout"]["vllm"]["max_num_seqs"] = 8
    cfg["rollout"]["vllm"]["max_num_batched_tokens"] = 256
    cfg["rollout"]["vllm"]["max_model_len"] = 128

    cfg["teacher"]["path"] = TINY_STUDENT
    cfg["teacher"]["gpu_ids"] = "0"
    cfg["teacher"]["vllm"]["gpu_memory_utilization"] = 0.03
    cfg["teacher"]["vllm"]["max_num_seqs"] = 8
    cfg["teacher"]["vllm"]["max_model_len"] = 128
    cfg["teacher"]["scoring_batch_size"] = 4

    cfg["trainer"]["dtype"] = "float32"
    cfg["teacher"]["dtype"] = "float32"
    cfg["rollout"]["dtype"] = "float32"
    cfg["weight_sync"] = {"backend": "cpu", "verify_checksum": True}
    cfg["pipeline"]["scheduling_mode"] = "fused_hybrid_sync"
    cfg["pipeline"]["n_step_off"]["step_off"] = 0
    cfg["pipeline"]["fused_hybrid_sync"] = {
        "weight_update_backend": "bucketed_inprocess",
        "debug_full_state_sync": False,
        "update_bucket_mb": 1,
        "vllm_sleep_level": 2,
        "require_vllm_sleep": True,
        "verify_weight_checksum": True,
        "refresh_policy": "after_train",
        "allow_teacher_gpu_overlap": True,
        "require_multigpu_fsdp": True,
        "allow_single_gpu_debug": False,
        "log_memory": True,
    }
    cfg["eval"]["mode"] = ["inline"]
    cfg["eval"]["freq"] = -1
    cfg["eval"]["before_train"] = False
    return cfg


def make_fused_hybrid_sync_dp2_student_teacher_kl0_fsdp2_config():
    """Local fused_hybrid_sync DP smoke: 2 student GPUs + colocated 1-GPU teacher.

    The local integration lane has two GPUs, so the teacher intentionally
    overlaps GPU 0 while the fused student trainer/rollout DP replicas use
    GPUs 0,1. Production configs keep the teacher disjoint by default.
    """
    cfg = make_fused_hybrid_sync_student_teacher_kl0_fsdp2_config()
    cfg["rollout"]["vllm"]["tensor_parallel_size"] = 1
    cfg["pipeline"]["fused_hybrid_sync"]["rollout_parallelism"] = "data_parallel"
    cfg["pipeline"]["fused_hybrid_sync"]["allow_teacher_gpu_overlap"] = True
    return cfg


def make_fused_hybrid_sync_dp2_cached_colocated_teacher_kl0_fsdp2_config():
    """Cached DP fused-hybrid smoke: 2 fused rollout/trainer GPUs + 1 colocated teacher.

    This is the explicit regression lane for the cached rank-local DP path: the
    teacher is intentionally colocated on GPU 0 on local two-GPU machines, while
    FSDP/vLLM DP replicas run on GPUs 0 and 1.
    """
    return make_fused_hybrid_sync_dp2_student_teacher_kl0_fsdp2_config()


def make_fused_hybrid_sync_dp2_cached_checkpoint_save_regression_fsdp2_config():
    """Cached DP fused-hybrid regression for checkpoint-save queue draining.

    The production failure happened with ``save_freq=5``: the scheduler
    dispatched a checkpoint and immediately issued the next trainer-backed
    rollout command, so the checkpoint completion result was consumed as the
    generation result.  Saving every step keeps this integration test short
    while exercising the same result-queue ordering hazard.
    """
    cfg = make_fused_hybrid_sync_dp2_cached_colocated_teacher_kl0_fsdp2_config()
    cfg["trainer"]["total_steps"] = 3
    cfg["trainer"]["save_freq"] = 1
    return cfg


def make_fused_hybrid_sync_dp2_mc_pg_kl_student_teacher_kl0_fsdp2_config():
    """DP fused_hybrid_sync smoke for MC PG-KL response support tensors.

    This keeps the same local two-GPU colocation as the forward-KL DP smoke,
    but switches to multi-sample PG-KL so DP generate must collect and merge
    the per-token MC support fields.
    """
    cfg = make_fused_hybrid_sync_dp2_student_teacher_kl0_fsdp2_config()
    cfg["algorithm"].setdefault("opd", {})
    cfg["algorithm"]["opd"].update({
        "kl_loss_mode": "multi_sample_policy_gradient_kl",
        "pg_kl_n_total_samples": 4,
        "pg_online_advantage": True,
        "pg_clip_eps": 1000000.0,
    })
    cfg["rollout"]["vllm"]["enforce_eager"] = False
    return cfg


def make_async_staleness_config():
    """Fully-async config to verify staleness is in optimizer-step units.

    train_batch_size=8, mini_batch_size=2 → 4 mini-batches → 4 optimizer steps per train.
    With fully_async scheduling, staleness should be reported as multiples of 4
    (optimizer steps), not 1 (train steps).
    """
    cfg = make_opd_config("policy_gradient_kl", True, "async",
                          mini_batch_size=2)
    cfg["trainer"]["batch_size"] = 8
    cfg["trainer"]["total_steps"] = 8
    cfg["pipeline"]["fully_async"]["staleness_threshold"] = 4
    return cfg


def make_grpo_async_staleness_config():
    """Fully-async DAPO config to verify staleness is group-size invariant.

    Uses DAPO (kl_beta=0, no teacher) since streaming doesn't support teacher scoring.
    train_batch_size=4, grpo_group_size=4, mini_batch_size=2:
      - Correct (scaled): n_mini = 4/2 = 2 optimizer steps per train
      - Bug (unscaled): n_mini = (4*4)/2 = 8 optimizer steps per train

    Staleness in optimizer-step units should reflect n_mini=2, not 8.
    """
    cfg = make_dapo_config("async")
    cfg["pipeline"]["fully_async"]["staleness_threshold"] = 4
    cfg["trainer"]["total_steps"] = 8
    return cfg


def check_grpo_async_staleness(train_steps, run_log, name):
    """Verify async GRPO staleness is reported and pipeline runs correctly.

    Streaming GRPO batches may differ in size from step-off (batch assembly
    differs), so we don't assert n_optim_steps here. The key checks:
    1. Staleness is present and non-zero (async pipeline working)
    2. Pipeline completes without crash (GRPO + streaming integration)
    """
    failures = []
    if not train_steps:
        return ["No train steps in log.jsonl"]

    # Verify staleness is present (async should have non-zero staleness)
    stalenesses = [s.get("staleness_mean", 0) for s in train_steps if s.get("staleness_mean", 0) > 0]
    if not stalenesses:
        failures.append("No non-zero staleness in async GRPO — streaming pipeline may not be running")

    # Sanity: n_optim_steps should not be inflated by group_size (would be 8 if unscaled)
    for s in train_steps:
        n = s.get("n_optim_steps")
        if n is not None and n > 4:
            failures.append(
                f"step {s.get('step','?')}: n_optim_steps={n} > 4, "
                f"mini_batch_size may not be scaled by group_size")
            break

    return failures


def make_eval_config():
    """Generate a PG-KL config with post_allgpu eval enabled."""
    cfg = make_opd_config("policy_gradient_kl", False, "so2")
    cfg["data"]["answer_key"] = "answer"
    cfg["eval"]["freq"] = 3
    cfg["eval"]["mode"] = ["post_allgpu"]
    cfg["eval"]["before_train"] = True
    cfg["eval"]["batch_size"] = 64
    cfg["eval"]["n_samples"] = 1
    return cfg


def make_standalone_eval_cli_config():
    """Training config paired with an evaluation CLI smoke."""
    cfg = make_opd_config(
        "policy_gradient_kl", False, "so0",
        total_steps=1,
        batch_size=4,
        micro_batch_size=2,
        mini_batch_size=4,
        max_prompt_length=32,
        max_response_length=16,
    )
    cfg["data"]["val_files"] = GSM8K_TRAIN_FILE
    cfg["data"]["answer_key"] = "answer"
    cfg["eval"]["freq"] = -1
    cfg["eval"]["before_train"] = False
    return cfg


def make_eval_avg2_config():
    """Generate a small post_allgpu Avg@2 eval config."""
    cfg = make_eval_config()
    cfg["eval"]["n_samples"] = 2
    cfg["eval"]["temperature"] = 0.7
    return cfg


def make_vllm_greedy_top1_parity_config(loss_mode, online_advantage=False):
    """Small fp32 vLLM config for PG-vs-THUNLP greedy top1 parity checks."""
    cfg = make_opd_config(loss_mode, False, "so0")
    cfg["rollout"]["temperature"] = 0.0
    cfg["rollout"]["top_p"] = 1.0
    cfg["rollout"]["top_k"] = -1
    cfg["rollout"]["dtype"] = "float32"
    cfg["teacher"]["dtype"] = "float32"
    cfg["teacher"]["vllm"]["n_logprobs"] = 1
    cfg["trainer"]["dtype"] = "float32"
    cfg["trainer"]["use_torch_compile"] = False
    cfg["trainer"]["batch_size"] = 8
    cfg["trainer"]["micro_batch_size"] = 4
    cfg["trainer"]["mini_batch_size"] = 8
    cfg["trainer"]["total_steps"] = 1
    cfg["data"]["max_prompt_length"] = 32
    cfg["data"]["max_response_length"] = 32
    cfg["eval"]["freq"] = -1
    cfg["eval"]["before_train"] = False
    cfg["algorithm"]["opd"]["pg_clip_eps"] = 0.2
    cfg["algorithm"]["opd"]["pg_online_advantage"] = online_advantage
    if loss_mode == "thunlp_opd_default_loss":
        cfg["algorithm"]["opd"]["rollout_student_topk_k"] = 1
    return cfg


def make_native_lora_config():
    """Generate a small native-LoRA vLLM integration config.

    Uses the tiny local Qwen fixtures so the full native LoRA path can run on a
    local 2-GPU workstation:
      - teacher + rollout on GPU 0
      - FSDP trainer with PEFT LoRA on GPU 1
      - NCCL LoRA-only sync to rollout vLLM

    Same-model teacher/student means step-1 KL should still be ~0 before any
    optimizer update while subsequent steps exercise LoRA sync.
    """
    cfg = make_opd_config("policy_gradient_kl", True, "so2")

    cfg["teacher"]["dtype"] = "bfloat16"
    cfg["trainer"]["dtype"] = "bfloat16"
    cfg["rollout"]["dtype"] = "bfloat16"
    cfg["trainer"]["optim"]["lr"] = 1e-4
    cfg["trainer"]["micro_batch_size"] = 2
    cfg["trainer"]["use_torch_compile"] = False
    cfg["trainer"]["lora"] = {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "native_lora": True,
    }

    cfg["trainer"]["total_steps"] = 4
    cfg["eval"]["freq"] = -1
    cfg["eval"]["before_train"] = False
    return cfg


# ──────────────────────────────────────────────────────────────
#  Test registry
# ──────────────────────────────────────────────────────────────

OPD_LOSS_MODES = ["forward_kl", "reverse_kl", "skewed_kl", "policy_gradient_kl"]
SCHEDULERS = ["so2", "async"]
PACKINGS = [False, True]


def build_test_registry():
    """Build {name: (config_fn, checks)} for all tests.

    vLLM tests: minimal set (10 configs) covering all code paths without
    redundant cross-products. PG-KL is default since it's the production
    loss mode and the most complex (advantage, ratio, clipping).

    Deterministic tests: full cross-product (12 configs) since they're
    fast (HF backend, tiny models, parallel execution).
    """
    registry = {}

    # ---- vLLM integration tests (11 configs) ----
    pg_kl_checks = ["kl_step1", "weight_checksum", "trace", "adv_step1", "ratio_step1"]

    # ---- vLLM integration tests (5 configs) ----
    # Loss math is already verified by det_ tests. vLLM tests only verify
    # vLLM-specific integration: NCCL weight sync, logprob extraction,
    # streaming scheduler, eval pipeline, and GRPO multi-sample generation.

    # Zero step-off (fully synchronous): verifies sync scheduling path
    registry["pg_kl_packed_so0"] = (
        lambda: make_opd_config("policy_gradient_kl", True, "so0"),
        pg_kl_checks,
    )

    # Baseline: PG-KL packed + step-off (most complex loss mode, production config)
    registry["pg_kl_packed_so2"] = (
        lambda: make_opd_config("policy_gradient_kl", True, "so2"),
        pg_kl_checks,
    )

    # Async scheduler with vLLM
    registry["pg_kl_packed_async"] = (
        lambda: make_opd_config("policy_gradient_kl", True, "async"),
        pg_kl_checks + ["keep_mode_sequence"],
    )

    # Async-sample step-off: vLLM keeps the whole logical batch in AsyncLLM
    # while teacher scoring starts as individual samples complete.
    registry["pg_kl_packed_so2_async_sample_teacher_bs2"] = (
        lambda: make_opd_config(
            "policy_gradient_kl", True, "async_stepoff2",
            teacher_scoring_batch_size=2,
        ),
        pg_kl_checks + ["async_stepoff_trace", "async_stepoff_staleness"],
    )

    registry["pg_kl_packed_so0_async_sample_teacher_bs2"] = (
        lambda: make_opd_config(
            "policy_gradient_kl", True, "async_stepoff0",
            teacher_scoring_batch_size=2,
            total_steps=4,
        ),
        pg_kl_checks + [
            "async_stepoff_trace",
            "async_stepoff_zero_trace",
            "async_stepoff_staleness",
        ],
    )

    registry["pg_kl_packed_so2_async_direct_teacher_bs2"] = (
        lambda: make_opd_config(
            "policy_gradient_kl", True, "async_stepoff2",
            teacher_scoring_batch_size=2,
            teacher_transport="direct_trainer",
            teacher_artifact_mode="direct",
            total_steps=4,
        ),
        pg_kl_checks + [
            "async_stepoff_trace",
            "async_stepoff_staleness",
            "direct_teacher_artifacts",
        ],
    )


    registry["dense_reverse_kl_async_hidden_recompute_teacher_bs2"] = (
        lambda: make_opd_config(
            "reverse_kl", True, "async_stepoff2",
            teacher_scoring_batch_size=1,
            teacher_transport="direct_trainer",
            teacher_artifact_mode="hidden_recompute",
            teacher_hidden_dtype="float32",
            teacher_hidden_semantics="lm_head_input",
            teacher_hidden_recompute_materialization="lazy",
            teacher_path=TINY_STUDENT,
            batch_size=2,
            micro_batch_size=1,
            mini_batch_size=2,
            max_prompt_length=32,
            max_response_length=8,
            total_steps=4,
        ),
        [
            "kl_step1",
            "weight_checksum",
            "trace",
            "async_stepoff_trace",
            "async_stepoff_staleness",
            "direct_teacher_artifacts",
            "hidden_recompute_artifacts",
        ],
    )

    registry[DENSE_RECOMPUTE_BASELINE_NAME] = (
        lambda: make_opd_config(
            "reverse_kl", False, "async_stepoff2",
            teacher_scoring_batch_size=1,
            teacher_transport="direct_trainer",
            teacher_artifact_mode="direct",
            teacher_n_logprobs=_tiny_student_vocab_size(),
            teacher_path=TINY_STUDENT,
            deterministic=True,
            seed=42,
            batch_size=2,
            micro_batch_size=1,
            mini_batch_size=2,
            max_prompt_length=32,
            max_response_length=8,
            total_steps=4,
        ),
        [
            "weight_checksum",
            "trace",
            "async_stepoff_trace",
            "async_stepoff_staleness",
            "direct_teacher_artifacts",
        ],
    )

    registry[DENSE_RECOMPUTE_CLASSIC_BASELINE_NAME] = (
        lambda: make_opd_config(
            "reverse_kl", False, "so2",
            teacher_scoring_batch_size=1,
            teacher_n_logprobs=_tiny_student_vocab_size(),
            teacher_path=TINY_STUDENT,
            deterministic=True,
            seed=42,
            batch_size=2,
            micro_batch_size=1,
            mini_batch_size=2,
            max_prompt_length=32,
            max_response_length=8,
            total_steps=4,
        ),
        [
            "weight_checksum",
            "trace",
        ],
    )

    registry["det_dense_reverse_kl_async_hidden_recompute_teacher_bs2"] = (
        lambda: make_opd_config(
            "reverse_kl", True, "async_stepoff2",
            teacher_scoring_batch_size=1,
            teacher_transport="direct_trainer",
            teacher_artifact_mode="hidden_recompute",
            teacher_hidden_dtype="float32",
            teacher_hidden_semantics="lm_head_input",
            teacher_hidden_recompute_materialization="lazy",
            teacher_path=TINY_STUDENT,
            deterministic=True,
            seed=42,
            batch_size=2,
            micro_batch_size=1,
            mini_batch_size=2,
            max_prompt_length=32,
            max_response_length=8,
            total_steps=4,
        ),
        [
            "weight_checksum",
            "trace",
            "async_stepoff_trace",
            "async_stepoff_staleness",
            "direct_teacher_artifacts",
            "hidden_recompute_artifacts",
            "golden_loss",
            "dense_recompute_matches_baseline",
        ],
    )

    registry["det_dense_reverse_kl_async_hidden_recompute_canonical_teacher_bs2"] = (
        lambda: make_opd_config(
            "reverse_kl", False, "async_stepoff2",
            teacher_scoring_batch_size=1,
            teacher_transport="direct_trainer",
            teacher_artifact_mode="hidden_recompute",
            teacher_hidden_dtype="float32",
            teacher_hidden_semantics="lm_head_input",
            teacher_hidden_recompute_materialization="canonical",
            teacher_path=TINY_STUDENT,
            deterministic=True,
            seed=42,
            batch_size=2,
            micro_batch_size=1,
            mini_batch_size=2,
            max_prompt_length=32,
            max_response_length=8,
            total_steps=4,
        ),
        [
            "weight_checksum",
            "trace",
            "async_stepoff_trace",
            "async_stepoff_staleness",
            "direct_teacher_artifacts",
            "hidden_recompute_artifacts",
            "golden_loss",
            "dense_recompute_matches_baseline",
        ],
    )

    registry["det_forward_kl_async_stepoff_sample_teacher_bs2"] = (
        lambda: make_opd_config(
            "forward_kl", False, "async_stepoff2",
            teacher_scoring_batch_size=2,
            teacher_path=TINY_TEACHER,
            deterministic=True,
            seed=42,
            batch_size=4,
            micro_batch_size=4,
            total_steps=4,
        ),
        ["weight_checksum", "trace", "async_stepoff_trace", "async_stepoff_staleness"],
    )

    registry["det_forward_kl_async_stepoff0_sample_teacher_bs2"] = (
        lambda: make_opd_config(
            "forward_kl", False, "async_stepoff0",
            teacher_scoring_batch_size=2,
            teacher_path=TINY_TEACHER,
            deterministic=True,
            seed=42,
            batch_size=4,
            micro_batch_size=4,
            total_steps=4,
        ),
        [
            "weight_checksum",
            "trace",
            "async_stepoff_trace",
            "async_stepoff_zero_trace",
            "async_stepoff_staleness",
        ],
    )

    registry["det_forward_kl_hf_stepoff0_teacher_bs2"] = (
        lambda: make_deterministic_config(
            "forward_kl", False,
            teacher_scoring_batch_size=2,
            step_off=0,
            total_steps=4,
        ),
        ["weight_checksum", "golden_loss"],
    )

    registry["det_forward_kl_hf_stepoff2_teacher_bs2"] = (
        lambda: make_deterministic_config(
            "forward_kl", False,
            teacher_scoring_batch_size=2,
            step_off=2,
            total_steps=4,
        ),
        ["weight_checksum", "golden_loss"],
    )

    registry["det_forward_kl_hf_stepoff2_direct_teacher_bs2"] = (
        lambda: make_deterministic_config(
            "forward_kl", False,
            teacher_scoring_batch_size=2,
            step_off=2,
            teacher_transport="direct_trainer",
            teacher_artifact_mode="direct",
            total_steps=4,
        ),
        ["weight_checksum", "trace", "direct_teacher_artifacts", "golden_loss"],
    )

    registry["forward_kl_packed_so2"] = (
        lambda: make_opd_config("forward_kl", True, "so2"),
        ["kl_step1", "weight_checksum", "trace"],
    )

    registry["mc_pg_kl_packed_so2"] = (
        lambda: make_opd_config(
            "multi_sample_policy_gradient_kl", True, "so2",
            pg_kl_n_total_samples=16,
        ),
        pg_kl_checks + ["ratio_std_step1"],
    )

    registry["mc_forward_kl_packed_so2"] = (
        lambda: make_opd_config(
            "multi_sample_forward_kl", True, "so2",
            pg_kl_n_total_samples=16,
        ),
        ["kl_step1", "weight_checksum", "trace"],
    )

    registry["mof_opd_lite_mc_packed_so2"] = (
        lambda: make_opd_config(
            "mof_opd", True, "so2",
            pg_kl_n_total_samples=16,
            mof_variant="lite",
            mof_partition="two_group",
        ),
        ["weight_checksum", "trace", "mof_metrics"],
    )

    registry["mof_opd_full_generated_so0"] = (
        lambda: make_opd_config(
            "mof_opd", False, "so0",
            pg_kl_n_total_samples=1,
            mof_variant="full",
            mof_partition="two_group",
            mof_eta_mass=1.0,
            total_steps=4,
        ),
        ["weight_checksum", "trace", "mof_metrics"],
    )

    registry["det_mof_opd_vllm_lite_mc_packed_so2"] = (
        lambda: make_opd_config(
            "mof_opd", True, "so2",
            pg_kl_n_total_samples=16,
            mof_variant="lite",
            mof_partition="two_group",
            mof_eta_odds=1.0,
            deterministic=True,
            seed=42,
            teacher_path=TINY_TEACHER,
            total_steps=4,
        ),
        ["weight_checksum", "trace", "mof_metrics", "config_deterministic"],
    )

    registry["mc_pg_kl_packed_async"] = (
        lambda: make_opd_config(
            "multi_sample_policy_gradient_kl", True, "async",
            pg_kl_n_total_samples=16,
        ),
        pg_kl_checks + ["ratio_std_step1", "keep_mode_sequence"],
    )

    registry["mc_pg_kl_single_pass_async"] = (
        lambda: make_opd_config(
            "multi_sample_policy_gradient_kl", True, "async",
            pg_kl_n_total_samples=16,
        ),
        pg_kl_checks + ["ratio_std_step1", "keep_mode_sequence"],
    )

    # Reverse KL with rollout-student top-k support and response-local teacher queries.
    registry["reverse_kl_rollout_student_topk_async"] = (
        lambda: make_opd_config(
            "reverse_kl_rollout_student_topk", False, "async",
            rollout_student_topk_k=8,
        ),
        ["kl_step1", "weight_checksum", "trace"],
    )

    registry["thunlp_opd_default_loss_async"] = (
        lambda: make_opd_config(
            "thunlp_opd_default_loss", False, "async",
            rollout_student_topk_k=8,
        ),
        ["weight_checksum", "trace", "adv_step1", "keep_mode_sequence", "per_mini_metrics"],
    )

    registry["parity_pg_kl_greedy_top1_offline"] = (
        lambda: make_vllm_greedy_top1_parity_config(
            "policy_gradient_kl", online_advantage=False
        ),
        ["weight_checksum", "trace", "adv_step1"],
    )
    registry["parity_thunlp_greedy_top1_offline"] = (
        lambda: make_vllm_greedy_top1_parity_config(
            "thunlp_opd_default_loss", online_advantage=False
        ),
        ["weight_checksum", "trace", "parity_step1_loss"],
    )
    registry["parity_pg_kl_greedy_top1_online"] = (
        lambda: make_vllm_greedy_top1_parity_config(
            "policy_gradient_kl", online_advantage=True
        ),
        ["weight_checksum", "trace"],
    )
    registry["parity_thunlp_greedy_top1_online"] = (
        lambda: make_vllm_greedy_top1_parity_config(
            "thunlp_opd_default_loss", online_advantage=True
        ),
        ["weight_checksum", "trace", "parity_step1_loss"],
    )

    # AReaL: mini-batches + decoupled loss with vLLM
    # mini_batch_size=2, train_batch_size=8 → 4 mini-batches → ratio diverges after first
    registry["areal_packed_so2"] = (
        lambda: make_opd_config(
            "policy_gradient_kl", True, "so2",
            lr=1e-2, mini_batch_size=2, use_decoupled_loss=True,
        ),
        ["kl_step1", "weight_checksum", "trace", "behave_imp_weight_logged",
         "mini_batch_ratio_divergence"],
    )

    # GRPO: multi-sample generation with vLLM
    # Note: grpo_kl check skipped — tiny random models produce extreme KL values.
    # GRPO KL correctness is verified by det_grpo (HF backend, deterministic).
    registry["grpo_so2"] = (
        lambda: make_grpo_config("so2"),
        ["weight_checksum", "trace", "grpo_mini_batch_invariance",
         "grpo_staleness_invariance"],
    )

    # GRPO fully-async: verify async pipeline works with GRPO group expansion.
    # TrainDispatcher collects complete groups, so batch size is constant and
    # n_optim_steps = train_batch_size / mini_batch_size = 4/2 = 2.
    registry["grpo_async"] = (
        make_grpo_async_staleness_config,
        ["weight_checksum", "trace", "grpo_async_staleness",
         "grpo_mini_batch_invariance"],
    )

    registry["ac_pg_kl_so2"] = (
        make_actor_critic_vllm_config,
        ["weight_checksum", "trace", "actor_critic_metrics"],
    )

    # DAPO: GRPO + asymmetric clip + dual-clip + token-mean + filter_groups
    registry["dapo_so2"] = (
        lambda: make_dapo_config("so2"),
        ["weight_checksum", "trace", "dapo_features",
         "grpo_mini_batch_invariance", "grpo_staleness_invariance"],
    )

    # Eval pipeline with vLLM
    registry["pg_kl_eval"] = (make_eval_config, ["kl_step1", "weight_checksum", "trace", "eval_ran"])

    # Native LoRA: verify PEFT + NCCL LoRA-only sync into rollout vLLM.
    registry["pg_kl_packed_so2_native_lora"] = (
        make_native_lora_config,
        ["kl_step1", "lora_checksum", "trace"],
    )

    registry["mc_pg_kl_uneven_fsdp2_so0"] = (
        make_vllm_uneven_global_mini_fsdp2_config,
        [
            "kl_step1",
            "weight_checksum",
            "trace",
            "adv_step1",
            "ratio_step1",
            "ratio_std_step1",
            "uneven_global_mini_split",
            "per_mini_metrics",
        ],
    )

    registry["fused_hybrid_sync_student_teacher_kl0_fsdp2"] = (
        make_fused_hybrid_sync_student_teacher_kl0_fsdp2_config,
        [
            "kl_step1",
            "trace",
            "uneven_global_mini_split",
            "fused_hybrid_sync_sequence",
            "fused_hybrid_bucket_metrics",
            "fused_hybrid_weight_checksum",
            "per_mini_metrics",
        ],
    )

    registry["fused_hybrid_sync_dp2_student_teacher_kl0_fsdp2"] = (
        make_fused_hybrid_sync_dp2_student_teacher_kl0_fsdp2_config,
        [
            "kl_step1",
            "trace",
            "uneven_global_mini_split",
            "fused_hybrid_sync_sequence",
            "fused_hybrid_bucket_metrics",
            "fused_hybrid_weight_checksum",
            "fused_hybrid_dp_metrics",
            "per_mini_metrics",
        ],
    )

    registry["fused_hybrid_sync_dp2_cached_colocated_teacher_kl0_fsdp2"] = (
        make_fused_hybrid_sync_dp2_cached_colocated_teacher_kl0_fsdp2_config,
        [
            "kl_step1",
            "trace",
            "uneven_global_mini_split",
            "fused_hybrid_sync_sequence",
            "fused_hybrid_bucket_metrics",
            "fused_hybrid_weight_checksum",
            "fused_hybrid_dp_metrics",
            "per_mini_metrics",
        ],
    )

    registry["fused_hybrid_sync_dp2_cached_checkpoint_save_regression_fsdp2"] = (
        make_fused_hybrid_sync_dp2_cached_checkpoint_save_regression_fsdp2_config,
        [
            "kl_step1",
            "fused_hybrid_sync_sequence",
            "fused_hybrid_dp_metrics",
            "fused_hybrid_checkpoint_save_drained",
        ],
    )

    registry["fused_hybrid_sync_dp2_mc_pg_kl_student_teacher_kl0_fsdp2"] = (
        make_fused_hybrid_sync_dp2_mc_pg_kl_student_teacher_kl0_fsdp2_config,
        [
            "kl_step1",
            "adv_step1",
            "ratio_step1",
            "ratio_std_step1",
            "trace",
            "uneven_global_mini_split",
            "fused_hybrid_sync_sequence",
            "fused_hybrid_bucket_metrics",
            "fused_hybrid_weight_checksum",
            "fused_hybrid_dp_metrics",
            "per_mini_metrics",
        ],
    )

    # ---- Deterministic tests (HF rollout backend, bitwise reproducible) ----
    DET_LOSS_MODES = ["forward_kl", "reverse_kl", "skewed_kl", "policy_gradient_kl"]
    # Det tests use different teacher/student → no kl_step1≈0 or adv≈0 checks
    for loss in DET_LOSS_MODES:
        short_loss = loss.replace("policy_gradient_kl", "pg_kl")
        for packed in PACKINGS:
            pack_str = "packed" if packed else "nopack"
            name = f"det_{short_loss}_{pack_str}"
            registry[name] = (
                lambda l=loss, p=packed: make_deterministic_config(l, p),
                ["weight_checksum", "golden_loss"],
            )

    registry["det_reverse_kl_rollout_student_topk"] = (
        lambda: make_deterministic_config(
            "reverse_kl_rollout_student_topk",
            False,
            rollout_student_topk_k=8,
        ),
        ["weight_checksum", "golden_loss"],
    )

    registry["det_thunlp_opd_default_loss"] = (
        lambda: make_deterministic_config(
            "thunlp_opd_default_loss",
            False,
            rollout_student_topk_k=8,
        ),
        ["weight_checksum", "golden_loss"],
    )

    registry["det_pg_kl_no_is_nopack"] = (
        lambda: make_deterministic_config(
            "policy_gradient_kl",
            False,
            use_importance_sampling=False,
        ),
        ["weight_checksum", "golden_loss"],
    )

    registry["det_pg_kl_no_is_packed"] = (
        lambda: make_deterministic_config(
            "policy_gradient_kl",
            True,
            use_importance_sampling=False,
        ),
        ["weight_checksum", "golden_loss"],
    )

    registry["det_reverse_kl_rollout_student_topk_no_is"] = (
        lambda: make_deterministic_config(
            "reverse_kl_rollout_student_topk",
            False,
            rollout_student_topk_k=8,
            use_importance_sampling=False,
        ),
        ["weight_checksum", "golden_loss"],
    )

    registry["det_thunlp_opd_default_loss_no_is"] = (
        lambda: make_deterministic_config(
            "thunlp_opd_default_loss",
            False,
            rollout_student_topk_k=8,
            use_importance_sampling=False,
        ),
        ["weight_checksum", "golden_loss"],
    )

    for packed in PACKINGS:
        pack_str = "packed" if packed else "nopack"
        name = f"det_mc_pg_kl_{pack_str}"
        registry[name] = (
            lambda p=packed: make_deterministic_config(
                "multi_sample_policy_gradient_kl",
                p,
                pg_kl_n_total_samples=16,
            ),
            ["weight_checksum", "golden_loss"],
        )

    registry["det_mc_pg_kl_no_is_nopack"] = (
        lambda: make_deterministic_config(
            "multi_sample_policy_gradient_kl",
            False,
            pg_kl_n_total_samples=16,
            use_importance_sampling=False,
        ),
        ["weight_checksum", "golden_loss"],
    )

    registry["det_mc_forward_kl_nopack"] = (
        lambda: make_deterministic_config(
            "multi_sample_forward_kl",
            False,
            pg_kl_n_total_samples=16,
        ),
        ["weight_checksum", "golden_loss"],
    )

    registry["det_mof_opd_full_generated_nopack"] = (
        lambda: make_deterministic_config(
            "mof_opd",
            False,
            pg_kl_n_total_samples=1,
            mof_variant="full",
            mof_partition="two_group",
            mof_eta_mass=1.0,
        ),
        ["weight_checksum", "golden_loss", "mof_metrics"],
    )

    registry["det_mof_opd_full_generated_packed"] = (
        lambda: make_deterministic_config(
            "mof_opd",
            True,
            pg_kl_n_total_samples=1,
            mof_variant="full",
            mof_partition="two_group",
            mof_eta_mass=1.0,
        ),
        ["weight_checksum", "golden_loss", "mof_metrics"],
    )

    for packed in PACKINGS:
        pack_str = "packed" if packed else "nopack"
        name = f"det_mof_opd_lite_mc_{pack_str}"
        registry[name] = (
            lambda p=packed: make_deterministic_config(
                "mof_opd",
                p,
                pg_kl_n_total_samples=16,
                mof_variant="lite",
                mof_partition="two_group",
                mof_eta_odds=1.0,
            ),
            ["weight_checksum", "golden_loss", "mof_metrics"],
        )

    registry["det_mc_pg_kl_single_pass_packed"] = (
        lambda: make_deterministic_config(
            "multi_sample_policy_gradient_kl",
            True,
            pg_kl_n_total_samples=16,
        ),
        ["weight_checksum", "golden_loss"],
    )

    # Deterministic AReaL
    # mini_batch_size=2, train_batch_size=4 → 2 mini-batches → ratio diverges after first
    # lr=1e-2 so mini-batch ratio divergence is clearly visible
    for packed in PACKINGS:
        pack_str = "packed" if packed else "nopack"
        name = f"det_areal_{pack_str}"
        registry[name] = (
            lambda p=packed: make_deterministic_config(
                "policy_gradient_kl", p,
                lr=1e-2, mini_batch_size=2,
                use_decoupled_loss=True,
            ),
            ["weight_checksum", "golden_loss", "behave_imp_weight_logged",
             "mini_batch_ratio_divergence"],
        )

    # Deterministic M2PO (PG-KL + dynamic second-moment clipping)
    registry["det_m2po"] = (
        lambda: make_deterministic_config(
            "policy_gradient_kl", True,
            m2po_budget=0.04, m2po_miniclip_low=0.3, m2po_miniclip_high=0.5,
        ),
        ["weight_checksum", "golden_loss", "m2po_logged"],
    )

    # Deterministic PG-KL with online advantage
    # pg_online_advantage=True: advantage = teacher - current_student (not old_student)
    registry["det_pg_online"] = (
        lambda: make_deterministic_config(
            "policy_gradient_kl", True,
            pg_online_advantage=True,
        ),
        ["weight_checksum", "golden_loss"],
    )

    registry["det_pg_kl_lora"] = (
        make_deterministic_lora_config,
        ["weight_checksum", "golden_loss"],
    )

    registry["det_ac_pg_kl"] = (
        make_deterministic_actor_critic_config,
        ["weight_checksum", "golden_loss", "actor_critic_metrics",
         "actor_matches_standard_pg_init"],
    )

    # Deterministic GRPO (different teacher/student → no grpo_kl≈0 check)
    registry["det_grpo"] = (
        make_deterministic_grpo_config,
        ["weight_checksum", "golden_loss", "grpo_mini_batch_invariance",
         "grpo_staleness_invariance"],
    )

    # Deterministic GRPO with sequence packing
    registry["det_grpo_packed"] = (
        make_deterministic_grpo_packed_config,
        ["weight_checksum", "golden_loss", "grpo_mini_batch_invariance",
         "grpo_staleness_invariance"],
    )

    # Deterministic DAPO (GRPO + all DAPO features, no teacher, length reward)
    registry["det_dapo"] = (
        make_deterministic_dapo_config,
        ["weight_checksum", "golden_loss", "dapo_features",
         "grpo_mini_batch_invariance", "grpo_staleness_invariance"],
    )

    # Deterministic DAPO with sequence packing
    registry["det_dapo_packed"] = (
        make_deterministic_dapo_packed_config,
        ["weight_checksum", "golden_loss", "dapo_features",
         "grpo_mini_batch_invariance", "grpo_staleness_invariance"],
    )

    # Deterministic eval pipeline
    registry["det_eval"] = (
        make_deterministic_eval_config,
        ["weight_checksum", "golden_loss", "eval_ran"],
    )

    # ---- Fully-async staleness test (vLLM) ----
    # Verifies staleness is in optimizer-step units for fully_async mode.
    # mini_batch_size=2, train_batch_size=8 → 4 mini-batches → 4 optim steps.
    # Staleness values should be multiples of 4.
    registry["async_staleness"] = (
        make_async_staleness_config,
        ["async_staleness_check", "weight_checksum", "trace"],
    )

    # ---- 2-GPU FSDP mini-batch staleness test (deterministic) ----
    # Verifies staleness tracking uses actual n_optim_steps from trainer.
    # mini_batch_size=4 global, train_batch_size=8 → 2 mini-batches → 2 optim steps.
    # Final staleness = step_off(2) * 2 = 4.
    # Uses 2 GPUs (FSDP), but tiny models so parallel with other det_ tests is fine.
    registry["det_fsdp2_minibatch_staleness"] = (
        make_deterministic_fsdp2_minibatch_config,
        ["staleness_check", "weight_checksum", "n_optim_steps"],
    )

    registry["det_mc_pg_kl_uneven_fsdp2"] = (
        make_deterministic_uneven_global_mini_fsdp2_config,
        ["weight_checksum", "golden_loss", "uneven_global_mini_split"],
    )

    # ---- Additional FSDP release-readiness coverage ----
    registry["sft_ce_perplexity_fsdp1"] = (
        make_sft_config,
        ["finite_loss", "sft_perplexity_eval", "checkpoint_saved"],
    )

    registry["sft_ce_checkpoint_resume_fsdp1"] = (
        lambda: make_sft_config(total_steps=1, eval_perplexity=False),
        ["checkpoint_resume"],
    )

    registry["det_token_level_kl_nopack"] = (
        lambda: make_deterministic_token_level_config(False),
        ["weight_checksum", "finite_loss"],
    )

    registry["det_token_level_kl_packed"] = (
        lambda: make_deterministic_token_level_config(True),
        ["weight_checksum", "finite_loss"],
    )

    registry["opsd_vllm_self_score_so0"] = (
        make_opsd_vllm_config,
        ["weight_checksum", "trace", "finite_loss", "opsd_self_score"],
    )

    registry["det_dapo_filter_overlong"] = (
        make_deterministic_dapo_filter_overlong_config,
        ["weight_checksum", "finite_loss", "dapo_features", "dapo_filter_overlong_config"],
    )

    registry["pg_kl_standalone_eval_cli"] = (
        make_standalone_eval_cli_config,
        ["weight_checksum", "trace", "standalone_eval_cli"],
    )

    registry["pg_kl_eval_avg2"] = (
        make_eval_avg2_config,
        ["kl_step1", "weight_checksum", "trace", "eval_ran"],
    )

    registry["public_grpo_4gpu_tiny_smoke"] = (
        make_public_grpo_4gpu_smoke_config,
        ["weight_checksum", "trace", "finite_loss"],
    )

    # ---- Megatron release-readiness coverage ----
    registry["megatron_opd_tp2_hf_cpu"] = (
        lambda: make_megatron_opd_config(tp_size=2, pp_size=1, dp_size=1),
        ["weight_checksum", "finite_loss", "megatron_backend", "megatron_kl_step1_zero"],
    )

    registry["megatron_pg_kl_tp2_same_model_adv0_hf_cpu"] = (
        lambda: make_megatron_opd_config(
            tp_size=2,
            pp_size=1,
            dp_size=1,
            loss_mode="policy_gradient_kl",
            same_model=True,
        ),
        [
            "weight_checksum",
            "finite_loss",
            "megatron_backend",
            "megatron_kl_step1_zero",
            "megatron_adv_step1_zero",
            "megatron_ratio_step1_one",
        ],
    )

    registry["megatron_opd_pp2_hf_cpu"] = (
        lambda: make_megatron_opd_config(
            tp_size=1, pp_size=2, dp_size=1, use_two_layer_model=True,
        ),
        ["weight_checksum", "finite_loss", "megatron_backend", "megatron_kl_step1_zero"],
    )

    registry["megatron_opd_tp2_pp2_dp2_hf_cpu"] = (
        lambda: make_megatron_opd_config(
            tp_size=2, pp_size=2, dp_size=2, use_two_layer_model=True,
        ),
        ["weight_checksum", "finite_loss", "megatron_backend", "megatron_kl_step1_zero"],
    )

    registry["megatron_grpo_tp1_hf_cpu"] = (
        lambda: make_megatron_grpo_config(tp_size=1, pp_size=1, dp_size=1),
        ["weight_checksum", "finite_loss", "megatron_backend"],
    )

    registry["megatron_opd_tp2_vllm_nccl_weight_sync"] = (
        lambda: make_megatron_opd_config(
            tp_size=2, pp_size=1, dp_size=1, vllm_sync=True, same_model=True,
        ),
        [
            "strict_weight_checksum",
            "finite_loss",
            "trace",
            "megatron_backend",
            "megatron_kl_step1_zero",
        ],
    )

    registry["megatron_multinode_2x2_ray_dp2_vllm_smoke"] = (
        make_megatron_multinode_2x2_config,
        [
            "strict_weight_checksum",
            "finite_loss",
            "trace",
            "megatron_backend",
            "ray_multinode",
            "ray_megatron_trainer_spans_nodes",
            "megatron_kl_step1_zero",
        ],
    )

    registry["megatron_multinode_ray_tp2_pp2_dp2_smoke"] = (
        lambda: make_megatron_opd_config(
            tp_size=2,
            pp_size=2,
            dp_size=2,
            use_two_layer_model=True,
            vllm_sync=True,
            ray=True,
            multinode=True,
        ),
        [
            "strict_weight_checksum",
            "finite_loss",
            "trace",
            "megatron_backend",
            "ray_multinode",
            "ray_megatron_trainer_spans_nodes",
            "megatron_kl_step1_zero",
        ],
    )

    # Per-mini diagnostics should be validated for every integration run.
    # The check is a no-op for single-mini runs.
    for name, (factory, checks) in list(registry.items()):
        if "per_mini_metrics" not in checks:
            registry[name] = (factory, checks + ["per_mini_metrics"])

    return registry


# ──────────────────────────────────────────────────────────────
#  Check functions (operate on parsed results)
# ──────────────────────────────────────────────────────────────

def check_kl_step1(train_steps, run_log, name):
    """KL ≈ 0 at step 1 (before any weight update)."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    s1 = train_steps[0]
    kl = s1.get("kl_loss", 999)
    if abs(kl) > KL_TOL:
        return [f"step {s1.get('step','?')}: kl_loss={kl:.6f} > {KL_TOL}"]
    return []


def check_adv_step1(train_steps, run_log, name):
    """Advantage ≈ 0 at step 1 for PG-KL."""
    if not train_steps:
        return ["No train steps"]
    s1 = train_steps[0]
    adv = s1.get("adv_mean")
    if adv is None:
        # Try parsing from run.log
        for line in run_log.split("\n"):
            m = re.search(r"\[Step 1\].*adv=([-\d.]+)", line)
            if m:
                adv = float(m.group(1))
                break
    if adv is None:
        return ["adv_mean not found in step 1"]
    if abs(adv) > ADV_TOL:
        return [f"step 1: adv_mean={adv:.6f} > {ADV_TOL}"]
    return []


def check_ratio_step1(train_steps, run_log, name):
    """Ratio ≈ 1.0 at step 1 for PG-KL."""
    if not train_steps:
        return ["No train steps"]
    s1 = train_steps[0]
    r = s1.get("r_mean")
    if r is None:
        for line in run_log.split("\n"):
            m = re.search(r"\[Step 1\].*r=([-\d.]+)", line)
            if m:
                r = float(m.group(1))
                break
    if r is None:
        return ["r_mean not found in step 1"]
    if abs(r - 1.0) > RATIO_TOL:
        return [f"step 1: r_mean={r:.6f}, expected ≈1.0"]
    return []


def check_ratio_std_step1(train_steps, run_log, name):
    """Step-1 ratio std should be near zero for MC/PG no-update runs."""
    if not train_steps:
        return ["No train steps"]
    s1 = train_steps[0]
    r_std = s1.get("r_std")
    if r_std is None:
        return ["r_std not found in step 1"]
    tol = 1e-3
    if abs(r_std) > tol:
        return [f"step 1: r_std={r_std:.6f}, expected <= {tol}"]
    return []


def check_parity_step1_loss(train_steps, run_log, name):
    """THUNLP greedy top1 step-1 loss should match the PG counterpart."""
    if not train_steps:
        return ["No train steps"]
    if not name.startswith("parity_thunlp_greedy_top1_"):
        return [f"parity_step1_loss called on unsupported test name: {name}"]
    counterpart = name.replace("parity_thunlp_greedy_top1_", "parity_pg_kl_greedy_top1_", 1)
    other_path = RESULTS_DIR / counterpart / "log.jsonl"
    if not other_path.exists():
        return [f"counterpart log.jsonl not found: {other_path}"]

    other_steps = []
    with open(other_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "train":
                other_steps.append(entry)
    if not other_steps:
        return [f"No train steps in counterpart log: {counterpart}"]

    ours = train_steps[0].get("kl_loss")
    other = other_steps[0].get("kl_loss")
    if ours is None or other is None:
        return [f"Missing kl_loss for parity comparison: ours={ours}, other={other}"]
    if abs(ours - other) > 1e-4:
        return [f"step 1 parity mismatch vs {counterpart}: ours={ours:.8f}, other={other:.8f}"]
    return []


def check_weight_checksum(train_steps, run_log, name):
    """Weight checksums should match (no mismatch lines in output)."""
    failures = []
    ok_count = run_log.count("Weight checksum OK")
    fail_count = run_log.count("Weight checksum mismatch")
    if fail_count > 0:
        failures.append(f"Weight checksum mismatch found ({fail_count} times)")
    if ok_count == 0 and fail_count == 0:
        # verify_checksum might not have triggered (e.g., no sync happened)
        pass
    return failures


def check_strict_weight_checksum(train_steps, run_log, name):
    """Production sync tests must positively log at least one matching checksum."""
    failures = check_weight_checksum(train_steps, run_log, name)
    ok_count = run_log.count("Weight checksum OK")
    if ok_count == 0:
        failures.append("No Weight checksum OK logs found")
    return failures


def check_megatron_kl_step1_zero(train_steps, run_log, name):
    """Same-model Megatron tests should have near-zero step-1 KL."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    s1 = train_steps[0]
    kl = s1.get("kl_loss", s1.get("mean_kl"))
    if kl is None:
        return ["step 1: missing kl_loss/mean_kl"]
    if not isinstance(kl, (int, float)) or not math.isfinite(kl):
        return [f"step 1: non-finite KL metric {kl!r}"]
    if abs(kl) > MEGATRON_KL_TOL:
        return [f"step 1: KL={kl:.6f} > Megatron tolerance {MEGATRON_KL_TOL}"]
    return []


def check_megatron_adv_step1_zero(train_steps, run_log, name):
    """Same-model Megatron PG-KL tests should have near-zero step-1 advantage."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    s1 = train_steps[0]
    adv = s1.get("adv_mean")
    if adv is None:
        for line in run_log.split("\n"):
            m = re.search(r"\[Step 1\].*adv=([-\d.eE+]+)", line)
            if m:
                adv = float(m.group(1))
                break
    if adv is None:
        return ["step 1: missing adv_mean"]
    if not isinstance(adv, (int, float)) or not math.isfinite(adv):
        return [f"step 1: non-finite adv_mean {adv!r}"]
    if abs(adv) > MEGATRON_ADV_TOL:
        return [f"step 1: adv_mean={adv:.6f} > Megatron tolerance {MEGATRON_ADV_TOL}"]
    return []


def check_megatron_ratio_step1_one(train_steps, run_log, name):
    """Same-model Megatron PG-KL tests should have step-1 importance ratio near 1."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    s1 = train_steps[0]
    ratio = s1.get("r_mean")
    if ratio is None:
        for line in run_log.split("\n"):
            m = re.search(r"\[Step 1\].*r=([-\d.eE+]+)", line)
            if m:
                ratio = float(m.group(1))
                break
    if ratio is None:
        return ["step 1: missing r_mean"]
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio):
        return [f"step 1: non-finite r_mean {ratio!r}"]
    if abs(ratio - 1.0) > MEGATRON_RATIO_TOL:
        return [
            f"step 1: r_mean={ratio:.6f}, expected 1.0 ± {MEGATRON_RATIO_TOL}"
        ]
    return []


def check_lora_checksum(train_steps, run_log, name):
    """Native LoRA checksums should match across trainer and rollout."""
    failures = []
    ok_count = run_log.count("LoRA checksum OK")
    fail_count = run_log.count("LoRA checksum mismatch")
    if fail_count > 0:
        failures.append(f"LoRA checksum mismatch found ({fail_count} times)")
    if ok_count == 0 and fail_count == 0:
        failures.append("No LoRA checksum logs found")
    return failures


def check_grpo_kl(train_steps, run_log, name):
    """GRPO mean_kl ≈ 0 at step 1 (before any weight update)."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    s1 = train_steps[0]
    kl = s1.get("mean_kl")
    if kl is None:
        for line in run_log.split("\n"):
            m = re.search(r"\[Step 1\].*kl=([-\d.]+)", line)
            if m:
                kl = float(m.group(1))
                break
    if kl is None:
        return ["mean_kl not found in step 1"]
    if abs(kl) > GRPO_KL_TOL:
        return [f"step 1: mean_kl={kl:.6f} > {GRPO_KL_TOL}"]
    return []


def check_behave_imp_weight_logged(train_steps, run_log, name):
    """AReaL: behave_imp_weight should appear in metrics."""
    for s in train_steps:
        if "behave_imp_weight" in s:
            return []
    # Check run.log for the field
    if "behave_imp_weight" in run_log:
        return []
    return ["behave_imp_weight never logged — decoupled loss may not be active"]


def check_m2po_logged(train_steps, run_log, name):
    """M2PO: dynamic clip stats should appear in metrics."""
    for s in train_steps:
        if "m2po_clip_low" in s and "m2po_clip_high" in s:
            return []
    if "m2po_clip_low" in run_log:
        return []
    return ["m2po_clip_low/high never logged — M2PO dynamic clipping may not be active"]


def check_mof_metrics(train_steps, run_log, name):
    """MOF-OPD: mass/odds diagnostics should appear and be finite."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    required = [
        "mof_student_candidate_mass",
        "mof_teacher_candidate_mass",
        "mof_odds_loss",
        "mof_total_loss",
    ]
    failures = []
    for key in required:
        vals = [s.get(key) for s in train_steps if key in s]
        if not vals:
            failures.append(f"{key} never logged")
            continue
        if any((not isinstance(v, (int, float)) or not math.isfinite(v)) for v in vals):
            failures.append(f"{key} has non-finite values: {vals}")
    return failures


def check_config_deterministic(train_steps, run_log, name):
    """The recorded config must have deterministic mode enabled."""
    log_path = RESULTS_DIR / name / "log.jsonl"
    if not log_path.exists():
        return [f"log.jsonl not found: {log_path}"]
    try:
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("type") == "config":
                    deterministic = entry.get("config", {}).get("deterministic")
                    return [] if deterministic is True else [
                        f"config.deterministic is {deterministic!r}, expected True"
                    ]
    except (OSError, json.JSONDecodeError) as e:
        return [f"could not read config entry from log.jsonl: {e}"]
    return ["config entry not found in log.jsonl"]


def check_trace(train_steps, run_log, name):
    """Check that trace.json was written with expected events."""
    trace_path = RESULTS_DIR / name / "trace.json"
    if not trace_path.exists():
        return ["trace.json not found"]
    try:
        with open(trace_path) as f:
            trace = json.load(f)
        events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
        if len(events) < 5:
            return [f"trace.json has only {len(events)} events (expected many more)"]
    except (json.JSONDecodeError, KeyError) as e:
        return [f"trace.json parse error: {e}"]
    return []


def _load_trace_events(name):
    trace_path = RESULTS_DIR / name / "trace.json"
    if not trace_path.exists():
        return None, [f"trace.json not found: {trace_path}"]
    try:
        with open(trace_path) as f:
            trace = json.load(f)
        events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
        return events, []
    except (json.JSONDecodeError, KeyError) as e:
        return None, [f"trace.json parse error: {e}"]


def check_async_stepoff_trace(train_steps, run_log, name):
    """Async step-off trace invariants: sample events, full-batch train, teacher microbatches."""
    failures = []
    events, errs = _load_trace_events(name)
    if errs:
        return errs
    by_name = {}
    for e in events:
        by_name.setdefault(e.get("name"), []).append(e)

    submits = by_name.get("async_logical_batch_submit", [])
    samples = by_name.get("async_rollout_sample_ready", [])
    ready = by_name.get("logical_batch_ready", [])
    scored = by_name.get("scored_buffer_ready", [])
    trains = by_name.get("train", [])
    teacher = by_name.get("teacher_score", [])
    teacher_submit = by_name.get("teacher_microbatch_submit", [])

    if len(submits) != len(train_steps):
        failures.append(f"async_logical_batch_submit count={len(submits)}, expected {len(train_steps)}")
    if len(ready) != len(train_steps):
        failures.append(f"logical_batch_ready count={len(ready)}, expected {len(train_steps)}")
    if len(scored) != len(train_steps):
        failures.append(f"scored_buffer_ready count={len(scored)}, expected {len(train_steps)}")
    train_events = [
        e for e in trains
        if e.get("args", {}).get("logical_batch_id") is not None
    ]
    if len(train_events) != len(train_steps):
        failures.append(f"async train event count={len(train_events)}, expected {len(train_steps)}")

    batch_sizes = [
        e.get("args", {}).get("n_seqs")
        for e in train_events
        if e.get("args", {}).get("n_seqs") is not None
    ]
    if batch_sizes and any(bs != batch_sizes[0] for bs in batch_sizes):
        failures.append(f"train n_seqs changed across steps: {batch_sizes}")

    expected_samples = sum(
        int(e.get("args", {}).get("n_seqs", 0))
        for e in submits
    )
    if expected_samples and len(samples) != expected_samples:
        failures.append(f"async_rollout_sample_ready count={len(samples)}, expected {expected_samples}")

    if not teacher_submit:
        failures.append("missing teacher_microbatch_submit events")
    for e in teacher_submit:
        n_prompts = e.get("args", {}).get("n_prompts")
        if n_prompts is not None and n_prompts > 2 and "teacher_bs2" in name:
            failures.append(f"teacher microbatch n_prompts={n_prompts}, expected <=2")
            break

    # At least one teacher microbatch should be submitted (or observed as
    # running) before the last sample in one of its logical batches is ready.
    # ``teacher_score`` is emitted by the worker that performs scoring and can
    # lag the coordinator submission on very small/fast batches, so the
    # coordinator-side submit event is the less flaky streaming-barrier signal.
    sample_ts_by_lb = {}
    for e in samples:
        lb = e.get("args", {}).get("logical_batch_id")
        if lb is not None:
            sample_ts_by_lb.setdefault(lb, []).append(e.get("ts", 0))
    overlapped = False
    for e in teacher_submit + teacher:
        lbs = e.get("args", {}).get("logical_batch_ids") or []
        for lb in lbs:
            ts_list = sample_ts_by_lb.get(lb) or []
            if ts_list and e.get("ts", 0) < max(ts_list):
                overlapped = True
                break
        if overlapped:
            break
    if not overlapped:
        failures.append(
            "no teacher scoring microbatch was submitted before final sample_ready "
            "in any logical batch"
        )
    return failures


def check_async_stepoff_zero_trace(train_steps, run_log, name):
    """Step-off-zero async mode: no logical batch K+1 before train K completes."""
    events, errs = _load_trace_events(name)
    if errs:
        return errs
    submits = [
        e for e in events
        if e.get("name") == "async_logical_batch_submit"
    ]
    trains = [
        e for e in events
        if e.get("name") == "train"
        and e.get("args", {}).get("logical_batch_id") is not None
    ]
    failures = []
    submit_by_lb = {e.get("args", {}).get("logical_batch_id"): e for e in submits}
    train_by_lb = {e.get("args", {}).get("logical_batch_id"): e for e in trains}
    for lb, e in submit_by_lb.items():
        if lb is None or lb == 0:
            continue
        prev = train_by_lb.get(lb - 1)
        if prev is None:
            failures.append(f"missing train event for logical_batch_id={lb - 1}")
            continue
        prev_end = prev.get("ts", 0) + prev.get("dur", 0)
        if e.get("ts", 0) < prev_end:
            failures.append(
                f"logical_batch_id={lb} submitted before train {lb - 1} completed"
            )
            break
    max_inflight = 0
    active = set()
    timeline = []
    for e in submits:
        timeline.append((e.get("ts", 0), "submit", e.get("args", {}).get("logical_batch_id")))
    for e in trains:
        timeline.append((e.get("ts", 0) + e.get("dur", 0), "train_done", e.get("args", {}).get("logical_batch_id")))
    for _ts, kind, lb in sorted(timeline):
        if lb is None:
            continue
        if kind == "submit":
            active.add(lb)
            max_inflight = max(max_inflight, len(active))
        else:
            active.discard(lb)
    if max_inflight != 1:
        failures.append(f"max in-flight logical batches={max_inflight}, expected 1")
    return failures


def check_async_stepoff_staleness(train_steps, run_log, name):
    """Async step-off staleness matches classic step_off × mini-batch count."""
    if not train_steps:
        return ["No train steps in log.jsonl"]

    step_off = 0 if "stepoff0" in name or "_so0_" in name else 2
    n_optim_values = [
        int(s.get("n_optim_steps", 1) or 1)
        for s in train_steps
        if s.get("n_optim_steps", 1)
    ]
    n_optim = max(n_optim_values) if n_optim_values else 1
    expected = step_off * n_optim
    max_stale = max(
        s.get("staleness_max", s.get("staleness_mean", 0)) or 0
        for s in train_steps
    )

    if max_stale != expected:
        return [
            f"max_staleness={max_stale}, expected {expected} "
            f"(step_off={step_off} × n_optim_steps={n_optim})"
        ]
    return []


def check_direct_teacher_artifacts(train_steps, run_log, name):
    failures = []
    events, errs = _load_trace_events(name)
    if errs:
        return errs
    by_name = {}
    for e in events:
        by_name.setdefault(e.get("name"), []).append(e)
    sends = by_name.get("teacher_artifact_send", [])
    recvs = by_name.get("teacher_artifact_recv_trainer", [])
    ready = by_name.get("teacher_artifact_buffer_logical_batch_ready", [])
    dispatch = by_name.get("train_dispatch_from_teacher_buffer", [])
    if not sends:
        failures.append("missing teacher_artifact_send events")
    if not recvs:
        failures.append("missing teacher_artifact_recv_trainer events")
    if not ready:
        failures.append("missing teacher_artifact_buffer_logical_batch_ready events")
    if not dispatch:
        failures.append("missing train_dispatch_from_teacher_buffer events")
    total_send = sum((e.get("args", {}).get("n_bytes") or 0) for e in sends)
    total_recv = sum((e.get("args", {}).get("n_bytes") or 0) for e in recvs)
    if total_send <= 0:
        failures.append(f"teacher_artifact_send bytes={total_send}, expected >0")
    if total_recv <= 0:
        failures.append(f"teacher_artifact_recv_trainer bytes={total_recv}, expected >0")
    for e in ready:
        args = e.get("args", {})
        if args.get("coordinator_teacher_artifact_bytes") not in (0, None):
            failures.append(
                f"coordinator_teacher_artifact_bytes={args.get('coordinator_teacher_artifact_bytes')}, expected 0"
            )
            break
    return failures


def check_hidden_recompute_artifacts(train_steps, run_log, name):
    failures = []
    events, errs = _load_trace_events(name)
    if errs:
        return errs
    sends = [e for e in events if e.get("name") == "teacher_artifact_send"]
    if not any((e.get("args") or {}).get("payload_kind") == "hidden_states" for e in sends):
        failures.append("missing hidden_states teacher_artifact_send events")
    found_recompute_metric = False
    found_lazy_metric = False
    for step in train_steps:
        if not isinstance(step, dict):
            artifacts = {}
        else:
            artifacts = step.get("teacher_artifacts") or (step.get("metrics", {}) or {}).get("teacher_artifacts") or {}
        if artifacts.get("teacher_hidden_recv_bytes", 0) <= 0:
            failures.append("teacher_hidden_recv_bytes missing or <=0")
            break
        if artifacts.get("teacher_hidden_recompute_seconds", 0) > 0:
            found_recompute_metric = True
        if artifacts.get("teacher_hidden_prepare_seconds", 0) > 0:
            found_recompute_metric = True
        if step.get("teacher_hidden_lazy_recompute_seconds", 0) > 0:
            found_recompute_metric = True
            found_lazy_metric = True
        if step.get("teacher_hidden_fused_kl_seconds", 0) > 0:
            found_recompute_metric = True
            found_lazy_metric = True
        if step.get("teacher_hidden_fused_kl_max_logits_bytes", 0) > 0:
            found_lazy_metric = True
        if step.get("teacher_hidden_max_materialized_bytes", 0) > 0:
            found_lazy_metric = True
    if not found_recompute_metric:
        failures.append("teacher hidden recompute/prepare metric missing")
    if "canonical" not in name and not found_lazy_metric:
        failures.append("teacher hidden lazy recompute metric missing")
    return failures


def check_eval_ran(train_steps, run_log, name):
    """Check that evaluation produced validation artifacts.

    Some eval modes run as post-training subprocesses and emit their primary
    evidence as ``validation_outputs/*.jsonl`` rather than ``type=eval`` rows
    in ``log.jsonl``.
    """
    validation_dir = RESULTS_DIR / name / "validation_outputs"
    if not validation_dir.exists():
        return [f"validation_outputs directory missing: {validation_dir}"]
    outputs = list(validation_dir.glob("*.jsonl"))
    if not outputs:
        return [f"no evaluation jsonl outputs found in {validation_dir}"]
    return []


# ──────────────────────────────────────────────────────────────
#  Golden loss values for deterministic tests
# ──────────────────────────────────────────────────────────────

# Golden values: {(config_name, step): expected_loss_value}
# Populated via: python scripts/run_integration_tests.py --filter det_ --capture-golden
# Then paste the output dict here.
GOLDEN_LOSSES = {
    # Captured on kang-gpu (2x RTX 3090), step_off=2, tiny Qwen3-0.6B models, fp32
    # Teacher: tests/fixtures/tiny_teacher (seed=123)
    # Student: tests/fixtures/tiny_student (seed=42)
    # Re-capture with: python scripts/run_integration_tests.py --filter det_ --capture-golden
    ("det_areal_nopack", 1): 0.5367172062397003,
    ("det_areal_nopack", 2): 0.35089173913002014,
    ("det_areal_nopack", 3): 0.29277682304382324,
    ("det_areal_nopack", 4): 0.40535397827625275,
    ("det_areal_packed", 1): 0.5367172062397003,
    ("det_areal_packed", 2): 0.35089172422885895,
    ("det_areal_packed", 3): 0.29277682304382324,
    ("det_areal_packed", 4): 0.40535402297973633,
    ("det_eval", 1): 0.00041025158134289086,
    ("det_eval", 2): 0.0004106780979782343,
    ("det_eval", 3): 0.00041249734931625426,
    ("det_eval", 4): 0.00041062990203499794,
    ("det_forward_kl_nopack", 1): 0.00041025158134289086,
    ("det_forward_kl_nopack", 2): 0.0004106780979782343,
    ("det_forward_kl_nopack", 3): 0.00041249734931625426,
    ("det_forward_kl_nopack", 4): 0.00041062990203499794,
    ("det_forward_kl_packed", 1): 0.00041025158134289086,
    ("det_forward_kl_packed", 2): 0.0004106780979782343,
    ("det_forward_kl_packed", 3): 0.00041249734931625426,
    ("det_forward_kl_packed", 4): 0.00041062990203499794,
    ("det_forward_kl_hf_stepoff0_teacher_bs2", 1): 0.0004102516104467213,
    ("det_forward_kl_hf_stepoff0_teacher_bs2", 2): 0.00041029523708857596,
    ("det_forward_kl_hf_stepoff0_teacher_bs2", 3): 0.00041219298145733774,
    ("det_forward_kl_hf_stepoff0_teacher_bs2", 4): 0.0004107808927074075,
    ("det_forward_kl_hf_stepoff2_teacher_bs2", 1): 0.0004102516104467213,
    ("det_forward_kl_hf_stepoff2_teacher_bs2", 2): 0.00041067812708206475,
    ("det_forward_kl_hf_stepoff2_teacher_bs2", 3): 0.00041249734931625426,
    ("det_forward_kl_hf_stepoff2_teacher_bs2", 4): 0.000410629843827337,
    ("det_forward_kl_hf_stepoff2_direct_teacher_bs2", 1): 0.0004102516104467213,
    ("det_forward_kl_hf_stepoff2_direct_teacher_bs2", 2): 0.00041067812708206475,
    ("det_forward_kl_hf_stepoff2_direct_teacher_bs2", 3): 0.00041249734931625426,
    ("det_forward_kl_hf_stepoff2_direct_teacher_bs2", 4): 0.000410629843827337,
    ("det_dapo", 1): -3.3098040148615837e-06,
    ("det_dapo", 2): -2.1283747628331184e-05,
    ("det_dapo", 3): -5.1455339416861534e-05,
    ("det_dapo", 4): -2.53189355134964e-05,
    ("det_dapo_packed", 1): -3.3098040148615837e-06,
    ("det_dapo_packed", 2): -2.1283747628331184e-05,
    ("det_dapo_packed", 3): -5.1455339416861534e-05,
    ("det_dapo_packed", 4): -2.53189355134964e-05,
    ("det_grpo", 1): 0.5726408660411835,
    ("det_grpo", 2): 0.5769959092140198,
    ("det_grpo", 3): 0.5623345375061035,
    ("det_grpo", 4): 0.566938728094101,
    ("det_grpo_packed", 1): -3.3098040148615837e-06,
    ("det_grpo_packed", 2): -2.1283747628331184e-05,
    ("det_grpo_packed", 3): -5.1455339416861534e-05,
    ("det_grpo_packed", 4): -2.53189355134964e-05,
    ("det_pg_kl_nopack", 1): 0.5951197147369385,
    ("det_pg_kl_nopack", 2): 0.5984906554222107,
    ("det_pg_kl_nopack", 3): 0.5556237101554871,
    ("det_pg_kl_nopack", 4): 0.5857130289077759,
    ("det_pg_kl_packed", 1): 0.5951197147369385,
    ("det_pg_kl_packed", 2): 0.5984906554222107,
    ("det_pg_kl_packed", 3): 0.5556237101554871,
    ("det_pg_kl_packed", 4): 0.5857130289077759,
    ("det_pg_kl_no_is_nopack", 1): 0.5951197147369385,
    ("det_pg_kl_no_is_nopack", 2): 0.5985063910484314,
    ("det_pg_kl_no_is_nopack", 3): 0.5556169152259827,
    ("det_pg_kl_no_is_nopack", 4): 0.5857183337211609,
    ("det_pg_kl_no_is_packed", 1): 0.5951197147369385,
    ("det_pg_kl_no_is_packed", 2): 0.5985063910484314,
    ("det_pg_kl_no_is_packed", 3): 0.5556169152259827,
    ("det_pg_kl_no_is_packed", 4): 0.5857183337211609,
    ("det_pg_kl_lora", 1): 0.5951197147369385,
    ("det_pg_kl_lora", 2): 0.5984915494918823,
    ("det_pg_kl_lora", 3): 0.55559241771698,
    ("det_pg_kl_lora", 4): 0.5857412219047546,
    ("det_reverse_kl_nopack", 1): -0.00022220135724637657,
    ("det_reverse_kl_nopack", 2): -0.00022158713545650244,
    ("det_reverse_kl_nopack", 3): -0.00022238744713831693,
    ("det_reverse_kl_nopack", 4): -0.00022190820891410112,
    ("det_reverse_kl_packed", 1): -0.00022220135724637657,
    ("det_reverse_kl_packed", 2): -0.00022158713545650244,
    ("det_reverse_kl_packed", 3): -0.00022238744713831693,
    ("det_reverse_kl_packed", 4): -0.00022190820891410112,
    ("det_dense_reverse_kl_async_hidden_recompute_teacher_bs2", 1): 6.858582501934052e-08,
    ("det_dense_reverse_kl_async_hidden_recompute_teacher_bs2", 2): -2.5205768139358042e-08,
    ("det_dense_reverse_kl_async_hidden_recompute_teacher_bs2", 3): 8.894058645125824e-08,
    ("det_dense_reverse_kl_async_hidden_recompute_teacher_bs2", 4): 1.2911790392422517e-07,
    ("det_dense_reverse_kl_async_hidden_recompute_canonical_teacher_bs2", 1): 2.1811545991567982e-07,
    ("det_dense_reverse_kl_async_hidden_recompute_canonical_teacher_bs2", 2): -3.665192593871325e-10,
    ("det_dense_reverse_kl_async_hidden_recompute_canonical_teacher_bs2", 3): -6.035840538487491e-08,
    ("det_dense_reverse_kl_async_hidden_recompute_canonical_teacher_bs2", 4): 1.1170588010145366e-07,
    ("det_skewed_kl_nopack", 1): 9.402510477229953e-05,
    ("det_skewed_kl_nopack", 2): 9.455120743950829e-05,
    ("det_skewed_kl_nopack", 3): 9.506657806923613e-05,
    ("det_skewed_kl_nopack", 4): 9.449583012610674e-05,
    ("det_skewed_kl_packed", 1): 9.402510477229953e-05,
    ("det_skewed_kl_packed", 2): 9.455120743950829e-05,
    ("det_skewed_kl_packed", 3): 9.506657806923613e-05,
    ("det_skewed_kl_packed", 4): 9.449583012610674e-05,
    ("det_m2po", 1): 0.5951197147369385,
    ("det_m2po", 2): 0.5984906554222107,
    ("det_m2po", 3): 0.5556237101554871,
    ("det_m2po", 4): 0.5857130289077759,
    ("det_pg_online", 1): 0.5951197147369385,
    ("det_pg_online", 2): 0.5985082387924194,
    ("det_pg_online", 3): 0.5556543469429016,
    ("det_pg_online", 4): 0.5857088565826416,
    ("det_reverse_kl_rollout_student_topk", 1): 2.736132955760695e-05,
    ("det_reverse_kl_rollout_student_topk", 2): 2.584256617410574e-05,
    ("det_reverse_kl_rollout_student_topk", 3): 2.8922913770657033e-05,
    ("det_reverse_kl_rollout_student_topk", 4): 2.790280086628627e-05,
    ("det_reverse_kl_rollout_student_topk_no_is", 1): -0.00022220135724637657,
    ("det_reverse_kl_rollout_student_topk_no_is", 2): -0.00022158713545650244,
    ("det_reverse_kl_rollout_student_topk_no_is", 3): -0.00022238744713831693,
    ("det_reverse_kl_rollout_student_topk_no_is", 4): -0.00022190820891410112,
    ("det_thunlp_opd_default_loss", 1): 0.5640414357185364,
    ("det_thunlp_opd_default_loss", 2): 0.5746316909790039,
    ("det_thunlp_opd_default_loss", 3): 0.5703761577606201,
    ("det_thunlp_opd_default_loss", 4): 0.5778429508209229,
    ("det_thunlp_opd_default_loss_no_is", 1): 0.5640414953231812,
    ("det_thunlp_opd_default_loss_no_is", 2): 0.574643075466156,
    ("det_thunlp_opd_default_loss_no_is", 3): 0.5704138278961182,
    ("det_thunlp_opd_default_loss_no_is", 4): 0.577883780002594,
    ("det_mc_pg_kl_nopack", 1): 0.057057831436395645,
    ("det_mc_pg_kl_nopack", 2): 0.055751945823431015,
    ("det_mc_pg_kl_nopack", 3): 0.05582217872142792,
    ("det_mc_pg_kl_nopack", 4): 0.05664266645908356,
    ("det_mc_pg_kl_no_is_nopack", 1): 0.05705777555704117,
    ("det_mc_pg_kl_no_is_nopack", 2): 0.055754996836185455,
    ("det_mc_pg_kl_no_is_nopack", 3): 0.055815644562244415,
    ("det_mc_pg_kl_no_is_nopack", 4): 0.05648905038833618,
    ("det_mc_forward_kl_nopack", 1): 0.014432551339268684,
    ("det_mc_forward_kl_nopack", 2): 0.021436555311083794,
    ("det_mc_forward_kl_nopack", 3): 0.027548110112547874,
    ("det_mc_forward_kl_nopack", 4): 0.02907979115843773,
    ("det_mc_pg_kl_packed", 1): 0.057057831436395645,
    ("det_mc_pg_kl_packed", 2): 0.055751945823431015,
    ("det_mc_pg_kl_packed", 3): 0.05582217872142792,
    ("det_mc_pg_kl_packed", 4): 0.05664266645908356,
    ("det_mc_pg_kl_uneven_fsdp2", 1): 0.05638872707883517,
    ("det_mc_pg_kl_uneven_fsdp2", 2): 0.0518583698819081,
    ("det_mc_pg_kl_uneven_fsdp2", 3): 0.054724628726641335,
    ("det_mc_pg_kl_uneven_fsdp2", 4): 0.059903414299090706,
    ("det_mc_pg_kl_single_pass_packed", 1): 0.057057831436395645,
    ("det_mc_pg_kl_single_pass_packed", 2): 0.055751945823431015,
    ("det_mc_pg_kl_single_pass_packed", 3): 0.05582217872142792,
    ("det_mc_pg_kl_single_pass_packed", 4): 0.05664266645908356,
    ("det_mof_opd_full_generated_nopack", 1): 8.78908540471457e-05,
    ("det_mof_opd_full_generated_nopack", 2): 8.702689956407994e-05,
    ("det_mof_opd_full_generated_nopack", 3): 9.199540363624692e-05,
    ("det_mof_opd_full_generated_nopack", 4): 8.867830183589831e-05,
    ("det_mof_opd_full_generated_packed", 1): 8.78908540471457e-05,
    ("det_mof_opd_full_generated_packed", 2): 8.702689956407994e-05,
    ("det_mof_opd_full_generated_packed", 3): 9.199540363624692e-05,
    ("det_mof_opd_full_generated_packed", 4): 8.867830183589831e-05,
    ("det_mof_opd_lite_mc_nopack", 1): 0.0003144558286294341,
    ("det_mof_opd_lite_mc_nopack", 2): 0.00031535810558125377,
    ("det_mof_opd_lite_mc_nopack", 3): 0.0003167733666487038,
    ("det_mof_opd_lite_mc_nopack", 4): 0.00031490420224145055,
    ("det_mof_opd_lite_mc_packed", 1): 0.0003144558286294341,
    ("det_mof_opd_lite_mc_packed", 2): 0.00031535810558125377,
    ("det_mof_opd_lite_mc_packed", 3): 0.0003167733666487038,
    ("det_mof_opd_lite_mc_packed", 4): 0.00031490420224145055,
    ("det_ac_pg_kl", 1): 0.7896304130554199,
    ("det_ac_pg_kl", 2): 0.7943799495697021,
    ("det_ac_pg_kl", 3): 0.725966215133667,
    ("det_ac_pg_kl", 4): 0.7733651399612427,
}


def check_golden_loss(train_steps, run_log, name):
    """Bitwise loss comparison against golden values."""
    if not GOLDEN_LOSSES:
        return ["GOLDEN_LOSSES is empty — run with --capture-golden first to populate"]
    failures = []
    for step_entry in train_steps:
        step = step_entry.get("step")
        # Use kl_loss for OPD, mean_kl for GRPO
        actual = step_entry.get("kl_loss", step_entry.get("mean_kl"))
        if actual is None:
            continue
        key = (name, step)
        if key not in GOLDEN_LOSSES:
            failures.append(f"step {step}: missing GOLDEN_LOSSES entry for {key!r}")
            continue
        expected = GOLDEN_LOSSES[key]
        if actual != expected:  # bitwise equality
            failures.append(
                f"step {step}: loss={actual} != expected={expected} "
                f"(diff={abs(actual - expected):.2e})")
    return failures


def _read_train_steps_from_result(name):
    log_jsonl = RESULTS_DIR / name / "log.jsonl"
    if not log_jsonl.exists():
        return None, f"missing baseline log: {log_jsonl}"
    steps = []
    with open(log_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == "train":
                steps.append(entry)
    if not steps:
        return None, f"baseline {name} has no train steps"
    return steps, None


def check_dense_recompute_matches_baseline(train_steps, run_log, name):
    """Compare hidden recompute losses to full-vocab non-recompute baselines.

    The async/direct baseline verifies the same streaming scheduler path.  The
    classic baseline verifies that the optimized recompute+streaming path keeps
    the same loss values as the original non-streaming step-off scheduler.
    """
    failures = []
    for baseline_name in (DENSE_RECOMPUTE_BASELINE_NAME, DENSE_RECOMPUTE_CLASSIC_BASELINE_NAME):
        baseline_steps, err = _read_train_steps_from_result(baseline_name)
        if err:
            failures.append(err)
            continue
        baseline_by_step = {s.get("step"): s for s in baseline_steps}
        for step_entry in train_steps:
            step = step_entry.get("step")
            actual = step_entry.get("kl_loss")
            baseline = baseline_by_step.get(step)
            if actual is None:
                continue
            if baseline is None:
                failures.append(f"{baseline_name} step {step}: missing baseline step")
                continue
            expected = baseline.get("kl_loss")
            if expected is None:
                failures.append(f"{baseline_name} step {step}: baseline missing kl_loss")
                continue
            diff = abs(actual - expected)
            if diff > DENSE_RECOMPUTE_EQUIV_TOL:
                failures.append(
                    f"{baseline_name} step {step}: recompute loss={actual} "
                    f"baseline={expected} diff={diff:.2e} > "
                    f"{DENSE_RECOMPUTE_EQUIV_TOL:.1e}"
                )
    return failures


def check_keep_mode_sequence(train_steps, run_log, name):
    """Verify keep-mode pause/sync/resume sequence appears in order in run.log."""
    failures = []
    lines = run_log.split("\n")

    # Find line positions for each event
    pause_lines = [i for i, l in enumerate(lines)
                   if re.search(r"\[AsyncRollout-\d+\] pause\(mode=keep\)", l)]
    sync_lines = [i for i, l in enumerate(lines)
                  if re.search(r"\[AsyncRollout-\d+\] sync_weights\(.*reset_prefix_cache=True", l)]
    resume_lines = [i for i, l in enumerate(lines)
                    if re.search(r"\[AsyncRollout-\d+\] resume\(continued=True\)", l)]

    if not pause_lines:
        failures.append("Missing pause(mode=keep) log line")
    if not sync_lines:
        failures.append("Missing sync_weights with reset_prefix_cache=True log line")
    if not resume_lines:
        failures.append("Missing resume(continued=True) log line")

    # Verify at least one ordered triple: pause < sync < resume
    if pause_lines and sync_lines and resume_lines:
        found_ordered = False
        for p in pause_lines:
            for s in sync_lines:
                if s <= p:
                    continue
                for r in resume_lines:
                    if r > s:
                        found_ordered = True
                        break
                if found_ordered:
                    break
            if found_ordered:
                break
        if not found_ordered:
            failures.append("pause/sync/resume lines exist but not in correct order")

    return failures


def check_grpo_mini_batch_invariance(train_steps, run_log, name):
    """Verify n_optim_steps equals train_batch_size / mini_batch_size (prompt space).

    With mini_batch_size=2 and grpo_group_size=4:
      - Correct (scaled): mini_bs_scaled=2*4=8, expanded_batch=4*4=16, n_mini=16/8=2
      - Bug (unscaled): n_mini=16/2=8

    Expected: n_optim_steps = train_batch_size / mini_batch_size = 4/2 = 2.
    This proves mini_batch_size is scaled by group_size, so the optimizer step
    count doesn't depend on group_size.
    """
    failures = []
    if not train_steps:
        return ["No train steps in log.jsonl"]
    expected = 2  # train_batch_size(4) / mini_batch_size(2)
    for s in train_steps:
        n = s.get("n_optim_steps")
        if n is None:
            failures.append(f"step {s.get('step','?')}: n_optim_steps not logged")
            break
        if n != expected:
            failures.append(
                f"step {s.get('step','?')}: n_optim_steps={n}, expected {expected} "
                f"(train_batch_size/mini_batch_size in prompt space). "
                f"If n={n} > {expected}, mini_batch_size was not scaled by group_size.")
            break
    return failures


def check_grpo_staleness_invariance(train_steps, run_log, name):
    """Verify GRPO staleness is in optimizer-step units scaled correctly.

    With mini_batch_size=2, train_batch_size=4, grpo_group_size=4, step_off=2:
      - n_optim_steps = 4/2 = 2 (prompt space, group-size invariant)
      - max staleness = step_off * n_optim_steps = 2 * 2 = 4

    If mini_batch_size were NOT scaled by group_size:
      - n_optim_steps = (4*4)/2 = 8
      - max staleness = 2 * 8 = 16
    """
    failures = []
    max_stale = 0
    for s in train_steps:
        stale = s.get("staleness_max", s.get("staleness_mean", 0))
        if stale > max_stale:
            max_stale = stale
    expected = 4  # step_off(2) * n_optim(2)
    if max_stale == 0:
        # May not have enough steps for staleness to build up — skip
        pass
    elif max_stale > expected + 1:
        failures.append(
            f"max_staleness={max_stale}, expected <= {expected} "
            f"(step_off=2 × n_optim=2). If higher, mini_batch_size "
            f"was not scaled by group_size.")
    return failures


def check_dapo_features(train_steps, run_log, name):
    """DAPO: verify clip_fraction logged and mean_kl=0 (no KL penalty)."""
    failures = []
    if not train_steps:
        return ["No train steps in log.jsonl"]
    # DAPO has kl_beta=0: mean_kl should be 0 (or absent)
    for s in train_steps:
        kl = s.get("mean_kl", 0)
        if abs(kl) > 1e-6:
            failures.append(f"step {s.get('step','?')}: mean_kl={kl} (expected 0 for DAPO)")
            break
    # clip_fraction should be logged
    has_clip = any("clip_fraction" in s for s in train_steps)
    if not has_clip:
        # Check run.log
        if "clip=" not in run_log:
            failures.append("clip_fraction never logged — DAPO clip not active")
    # filter_groups should log when groups are filtered
    # (may not filter if all groups have variance — just check it doesn't crash)
    # All steps should complete
    if len(train_steps) < 2:
        failures.append(f"Only {len(train_steps)} train steps (expected at least 2)")
    return failures


def check_finite_loss(train_steps, run_log, name):
    """Every train step should report a finite scalar loss/KL metric."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    failures = []
    for s in train_steps:
        step = s.get("step", "?")
        value = s.get("kl_loss", s.get("mean_kl"))
        if value is None:
            failures.append(f"step {step}: missing kl_loss/mean_kl")
            break
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"step {step}: non-finite loss metric {value!r}")
            break
    return failures


def _load_result_entries(name):
    log_path = RESULTS_DIR / name / "log.jsonl"
    entries = []
    if not log_path.exists():
        return entries, f"log.jsonl not found: {log_path}"
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as e:
        return entries, f"could not read log.jsonl: {e}"
    return entries, None


def check_sft_perplexity_eval(train_steps, run_log, name):
    """SFT perplexity eval should emit finite val_loss and perplexity metrics."""
    entries, err = _load_result_entries(name)
    if err:
        return [err]
    eval_entries = [e for e in entries if e.get("type") == "eval"]
    if not eval_entries:
        return ["No SFT eval entries in log.jsonl"]
    failures = []
    for e in eval_entries:
        step = e.get("step", "?")
        for key in ("val_loss", "perplexity", "n_tokens"):
            value = e.get(key)
            if value is None:
                failures.append(f"eval step {step}: missing {key}")
                break
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append(f"eval step {step}: non-finite {key}={value!r}")
                break
        if failures:
            break
    return failures


def check_checkpoint_saved(train_steps, run_log, name):
    """A final checkpoint should be materialized for modes that auto-save."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    last_step = max(int(s.get("step", 0)) for s in train_steps)
    ckpt = RESULTS_DIR / name / "checkpoints" / f"step_{last_step}"
    if not ckpt.exists():
        return [f"checkpoint directory not found: {ckpt}"]
    if not (ckpt / "model.pt").exists():
        return [f"checkpoint model.pt not found: {ckpt / 'model.pt'}"]
    return []


def check_dapo_filter_overlong_config(train_steps, run_log, name):
    """Generated DAPO config should explicitly enable filter_groups and overlong shaping."""
    config_path = CONFIG_DIR / f"{name}.yaml"
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except OSError as e:
        return [f"could not read generated config: {e}"]
    grpo = cfg.get("algorithm", {}).get("grpo", {})
    failures = []
    if grpo.get("filter_groups") is not True:
        failures.append("algorithm.grpo.filter_groups is not true")
    if int(grpo.get("overlong_buffer_len", 0) or 0) <= 0:
        failures.append("algorithm.grpo.overlong_buffer_len is not > 0")
    if float(grpo.get("overlong_penalty_factor", 0.0) or 0.0) <= 0:
        failures.append("algorithm.grpo.overlong_penalty_factor is not > 0")
    return failures


def check_opsd_self_score(train_steps, run_log, name):
    """OPSD should run without a teacher process and use rollout self-scoring."""
    failures = []
    if "[Pipeline] Teacher backend:" in run_log:
        failures.append("OPSD started a teacher process")
    if "[OPSD]" not in run_log and "score" not in run_log:
        failures.append("run.log lacks OPSD/self-score markers")
    return failures


def check_megatron_backend(train_steps, run_log, name):
    """Megatron integration should start the Megatron trainer backend."""
    config_path = CONFIG_DIR / f"{name}.yaml"
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except OSError as e:
        return [f"could not read generated config: {e}"]

    failures = []
    trainer = cfg.get("trainer", {})
    if trainer.get("backend") != "megatron":
        failures.append(f"trainer.backend={trainer.get('backend')!r}, expected 'megatron'")
    meg = trainer.get("megatron", {})
    expectations = {
        "tp2": ("tensor_parallel_size", 2),
        "pp2": ("pipeline_parallel_size", 2),
    }
    for marker, (key, expected) in expectations.items():
        if marker in name and int(meg.get(key, 0) or 0) != expected:
            failures.append(f"trainer.megatron.{key}={meg.get(key)!r}, expected {expected}")
    if "dp2" in name:
        n_gpus = int(trainer.get("n_gpus", 0) or 0)
        tp = int(meg.get("tensor_parallel_size", 1) or 1)
        pp = int(meg.get("pipeline_parallel_size", 1) or 1)
        dp = n_gpus // max(tp * pp, 1)
        if dp != 2:
            failures.append(f"derived Megatron DP={dp}, expected 2")
    if "[Backend-Megatron]" not in run_log and "Megatron trainer" not in run_log:
        failures.append("run.log lacks Megatron backend startup markers")
    return failures


def check_ray_multinode(train_steps, run_log, name):
    """Ray multi-node smoke should detect at least one remote node when enabled."""
    failures = []
    if "Ray connected" not in run_log:
        failures.append("run.log lacks Ray connected marker")
    if "Remote nodes:" not in run_log:
        failures.append("run.log lacks remote-node discovery marker")
    return failures


def check_ray_megatron_trainer_spans_nodes(train_steps, run_log, name):
    """Megatron Ray trainer ranks should be placed on at least two nodes."""
    m = re.search(r"Megatron trainer spans (\d+) node", run_log)
    if not m:
        return ["run.log lacks Megatron trainer node-span marker"]
    n_nodes = int(m.group(1))
    if n_nodes < 2:
        return [f"Megatron trainer spans {n_nodes} node(s), expected at least 2"]
    return []


def check_staleness(train_steps, run_log, name):
    """Verify staleness reaches 4 (2 optim steps × step_off=2).

    mini_batch_size=4 globally, train_batch_size=8 → 2 mini-batches → 2 optim steps.
    With step_off=2, max staleness = 2 × 2 = 4.
    """
    failures = []
    max_stale = 0
    for s in train_steps:
        stale = s.get("staleness_mean", s.get("staleness_max", 0))
        if stale > max_stale:
            max_stale = stale

    # Also check run.log
    for line in run_log.split("\n"):
        m = re.search(r"stale=(\d+)", line)
        if m:
            stale = int(m.group(1))
            if stale > max_stale:
                max_stale = stale

    # Correct behavior: mini_batch_size=4 globally → 8/4=2 mini-batches
    # → 2 optim steps per train → staleness = step_off(2) * 2 = 4.
    # BUG: mini_batch_size is per-GPU, so per-rank=4, 4/4=1 mini-batch
    # → 1 optim step per train → staleness = step_off(2) * 1 = 2.
    expected = 4  # correct global behavior
    if max_stale == 0:
        failures.append("No staleness data found")
    elif max_stale != expected:
        failures.append(
            f"max_staleness={max_stale}, expected {expected} "
            f"(2 global mini-batches × step_off=2)")
    return failures


def check_async_staleness(train_steps, run_log, name):
    """Verify async staleness is in optimizer-step units (scaled by n_mini).

    mini_batch_size=2, train_batch_size=8 → 4 mini-batches → 4 optim steps per train.
    With keep-pause mode, token-weighted breakpoints produce fractional values,
    so we check magnitude rather than exact multiples.

    In optimizer-step units: staleness should reach >= 8 (2+ train steps × 4).
    In train-step units (old bug): staleness would be ~2-5.
    Threshold of 8 distinguishes the two.
    """
    failures = []
    stalenesses = []
    for s in train_steps:
        stale = s.get("staleness_mean", 0)
        if stale > 0:
            stalenesses.append(stale)

    if not stalenesses:
        failures.append("No non-zero staleness data found")
        return failures

    max_stale = max(stalenesses)
    # With n_mini=4, even 2 train steps of staleness = 8 optimizer steps.
    # Old (unscaled) code would report max ~5 train steps.
    # Threshold of 8 clearly separates the two.
    if max_stale < 8:
        failures.append(
            f"max staleness={max_stale}, expected >= 8 "
            f"(at least 2 train steps in optimizer-step units, n_mini=4)")

    return failures


def check_n_optim_steps(train_steps, run_log, name):
    """Verify n_optim_steps=2 is logged (mini_batch_size=4, batch=8 → 2 mini-batches)."""
    failures = []
    for s in train_steps:
        n = s.get("n_optim_steps")
        if n is None:
            failures.append(f"step {s.get('step','?')}: n_optim_steps not logged")
            break
        if n != 2:
            failures.append(f"step {s.get('step','?')}: n_optim_steps={n}, expected 2")
            break
    if not train_steps:
        failures.append("No train steps in log.jsonl")
    return failures


def check_uneven_global_mini_split(train_steps, run_log, name):
    """Verify the local-2-GPU uneven global-mini scenario kept its semantics."""
    failures = []
    if not train_steps:
        return ["No train steps in log.jsonl"]

    if "dropping" in run_log.lower():
        failures.append("run.log contains a sample-dropping warning")

    if "per-rank batch size" in run_log:
        failures.append("run.log contains the old per-rank divisibility assertion")

    if "2 mini-batches" not in run_log and "optim step 2/2" not in run_log:
        failures.append("run.log does not show the expected 2-mini optimizer schedule")

    for s in train_steps:
        step = s.get("step", "?")
        n = s.get("n_optim_steps")
        if n != 2:
            failures.append(f"step {step}: n_optim_steps={n}, expected 2")
            break
        # Rank 0 receives 3 real samples from each 5-sample global mini.
        for mi in (0, 1):
            n_seq = s.get(f"n_seqs_mini_{mi}")
            if n_seq != 3:
                failures.append(
                    f"step {step}: n_seqs_mini_{mi}={n_seq}, expected 3 "
                    "for rank0's 3/2 uneven shard")
                break
        if failures:
            break

    return failures


def check_fused_hybrid_sync_sequence(train_steps, run_log, name):
    """Verify fused_hybrid_sync phase logs show strict rollout→sleep→train→sync."""
    failures = []
    if len(train_steps) != 3:
        failures.append(f"expected 3 fused_hybrid_sync train steps, got {len(train_steps)}")
    required = [
        "[FusedHybrid] phase=rollout actor_version=0",
        "[FusedHybrid] sleep_rollout reason=after_generate",
        "[FusedHybrid] phase=rollout_quiesced",
        "[FusedHybrid] phase=trainer",
        "[FusedHybrid] phase=sync actor_version=",
        "backend=bucketed_inprocess",
    ]
    for marker in required:
        if marker not in run_log:
            failures.append(f"missing fused_hybrid_sync marker: {marker}")
    ordered = required[:5]
    positions = [run_log.find(marker) for marker in ordered]
    if all(pos >= 0 for pos in positions) and positions != sorted(positions):
        failures.append("fused_hybrid_sync phase markers are out of order")
    events, errs = _load_trace_events(name)
    if errs:
        failures.extend(errs)
    else:
        duration_events = [e for e in events if e.get("ph") == "X"]
        by_name_cat = {(e.get("name"), e.get("cat")) for e in duration_events}
        for name_cat in (("generate", "rollout"), ("train", "train"), ("sync_weights", "sync")):
            if name_cat not in by_name_cat:
                failures.append(
                    f"missing fused_hybrid_sync trace span name={name_cat[0]!r} "
                    f"cat={name_cat[1]!r}"
                )
        legacy_names = {
            "rollout_generate",
            "trainer_train",
            "actor_to_rollout_bucketed_refresh",
        }
        found_legacy = sorted(
            {e.get("name") for e in duration_events if e.get("name") in legacy_names}
        )
        if found_legacy:
            failures.append(f"legacy fused_hybrid_sync trace span names found: {found_legacy}")
        legacy_cat_count = sum(1 for e in duration_events if e.get("cat") == "fused_hybrid_sync")
        if legacy_cat_count:
            failures.append(
                f"legacy fused_hybrid_sync trace category used by {legacy_cat_count} spans"
            )
    return failures


def check_fused_hybrid_bucket_metrics(train_steps, run_log, name):
    """Default fused hybrid run must emit bucketed sync telemetry, not debug full-state."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    failures = []
    legacy_log_keys = {
        "rollout_generate",
        "trainer_train",
        "actor_to_rollout_bucketed_refresh",
    }
    for i, step in enumerate(train_steps, start=1):
        if step.get("fused_hybrid_weight_update_backend") != "bucketed_inprocess":
            failures.append(f"step {i}: missing bucketed_inprocess weight update backend metric")
        if step.get("fused_hybrid_weight_update_debug_full_state_sync") is not False:
            failures.append(
                f"step {i}: default fused_hybrid_sync used or failed to report non-debug sync"
            )
        if step.get("fused_hybrid_weight_update_full_state_materialized") is not False:
            failures.append(f"step {i}: fused_hybrid_sync materialized a full state_dict")
        for key in (
            "fused_hybrid_weight_update_bucket_count",
            "fused_hybrid_weight_update_total_bytes",
            "fused_hybrid_weight_update_max_bucket_bytes",
            "fused_hybrid_weight_update_duration_s",
            "fused_hybrid_weight_update_memory_before_allocated",
            "fused_hybrid_weight_update_memory_after_allocated",
        ):
            if key not in step:
                failures.append(f"step {i}: missing fused hybrid bucket telemetry: {key}")
        for key in ("generate_seconds", "teacher_seconds", "train_seconds",
                    "sync_seconds", "iter_seconds"):
            if key not in step:
                failures.append(f"step {i}: missing canonical log.jsonl timing key: {key}")
        sync_seconds = step.get("sync_seconds")
        update_seconds = step.get("fused_hybrid_weight_update_duration_s")
        if sync_seconds is not None and update_seconds is not None:
            if abs(float(sync_seconds) - float(update_seconds)) > 1e-6:
                failures.append(
                    f"step {i}: sync_seconds={sync_seconds} does not match "
                    f"fused_hybrid_weight_update_duration_s={update_seconds}"
                )
        found_legacy = sorted(legacy_log_keys.intersection(step.keys()))
        if found_legacy:
            failures.append(f"step {i}: legacy fused_hybrid_sync log keys found: {found_legacy}")
        if step.get("fused_hybrid_actor_version") != step.get("fused_hybrid_rollout_version"):
            failures.append(
                f"step {i}: actor/rollout version mismatch: "
                f"{step.get('fused_hybrid_actor_version')} "
                f"!= {step.get('fused_hybrid_rollout_version')}"
            )
    if "synced_cpu_state_dict" in run_log or "CPU state_dict weight sync" in run_log:
        failures.append("fused_hybrid_sync fell back to CPU state_dict sync")
    return failures


def check_fused_hybrid_weight_checksum(train_steps, run_log, name):
    """Fused hybrid integ must compare trainer stream and colocated vLLM weights."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    failures = []
    ok_marker = "[FusedHybrid] Weight checksum OK"
    mismatch_marker = "[FusedHybrid] WARNING: Weight checksum mismatch"
    if ok_marker not in run_log:
        failures.append("missing fused hybrid weight checksum OK log")
    if mismatch_marker in run_log:
        failures.append("fused hybrid weight checksum mismatch log found")

    required = (
        "fused_hybrid_weight_update_checksum_source",
        "fused_hybrid_weight_update_checksum_target",
        "fused_hybrid_weight_update_checksum_rel_error",
        "fused_hybrid_weight_update_checksum_ok",
        "fused_hybrid_weight_update_checksum_source_count",
        "fused_hybrid_weight_update_checksum_target_count",
        "fused_hybrid_weight_update_checksum_missing_source_count",
    )
    for i, step in enumerate(train_steps, start=1):
        for key in required:
            if key not in step:
                failures.append(f"step {i}: missing fused hybrid checksum metric: {key}")

        if step.get("fused_hybrid_weight_update_checksum_ok") is not True:
            failures.append(
                f"step {i}: fused_hybrid_weight_update_checksum_ok is not true "
                f"({step.get('fused_hybrid_weight_update_checksum_ok')!r})"
            )
        rel_err = step.get("fused_hybrid_weight_update_checksum_rel_error")
        if rel_err is None:
            failures.append(f"step {i}: missing fused hybrid checksum relative error")
        elif rel_err > 1e-6:
            failures.append(f"step {i}: fused hybrid checksum rel_err={rel_err:.2e} > 1e-6")
        if step.get("fused_hybrid_weight_update_checksum_missing_source_count") != 0:
            failures.append(
                f"step {i}: fused hybrid checksum missed source tensors: "
                f"{step.get('fused_hybrid_weight_update_checksum_missing_source_count')}"
            )
        source_count = step.get("fused_hybrid_weight_update_checksum_source_count")
        target_count = step.get("fused_hybrid_weight_update_checksum_target_count")
        if source_count is not None and source_count <= 0:
            failures.append(f"step {i}: fused hybrid checksum source_count is not positive")
        if target_count is not None and target_count <= 0:
            failures.append(f"step {i}: fused hybrid checksum target_count is not positive")
    return failures


def check_fused_hybrid_dp_metrics(train_steps, run_log, name):
    """Fused hybrid DP rollout must log replicated DP/TP/checksum diagnostics."""
    if not train_steps:
        return ["No train steps in log.jsonl"]
    failures = []
    if "[FusedHybrid] Weight checksum OK" not in run_log:
        failures.append("missing fused hybrid weight checksum OK log line")
    for i, step in enumerate(train_steps, start=1):
        if step.get("fused_hybrid_rollout_parallelism") != "data_parallel":
            failures.append(
                f"step {i}: fused_hybrid_rollout_parallelism is not data_parallel"
            )
        if step.get("fused_hybrid_rollout_dp_size") != 2:
            failures.append(f"step {i}: fused_hybrid_rollout_dp_size is not 2")
        if step.get("fused_hybrid_rollout_tp_size") != 1:
            failures.append(f"step {i}: fused_hybrid_rollout_tp_size is not 1")
        if step.get("fused_hybrid_dp_cached_generation") is not True:
            failures.append(f"step {i}: cached DP generation metric is not true")
        if step.get("fused_hybrid_dp_cached_teacher") is not True:
            failures.append(f"step {i}: cached DP teacher metric is not true")
        if step.get("fused_hybrid_dp_cached_teacher_prompts") is None:
            failures.append(f"step {i}: missing cached DP teacher prompt count")
        if step.get("fused_hybrid_weight_update_checksum_rank_count") != 2:
            failures.append(
                f"step {i}: checksum rank_count is not 2 "
                f"({step.get('fused_hybrid_weight_update_checksum_rank_count')!r})"
            )
        if step.get("fused_hybrid_weight_update_checksum_all_ranks_ok") is not True:
            failures.append(f"step {i}: fused DP checksum all_ranks_ok is not true")
        rank_ok = step.get("fused_hybrid_weight_update_checksum_rank_ok")
        if rank_ok != [True, True]:
            failures.append(f"step {i}: fused DP checksum rank_ok={rank_ok!r}")
        for key in (
            "fused_hybrid_weight_update_checksum_source_min",
            "fused_hybrid_weight_update_checksum_source_max",
            "fused_hybrid_weight_update_checksum_target_min",
            "fused_hybrid_weight_update_checksum_target_max",
            "fused_hybrid_weight_update_checksum_rel_error_max",
            "fused_hybrid_weight_update_checksum_target_local_by_rank",
            "fused_hybrid_weight_update_checksum_rel_error_by_rank",
        ):
            if key not in step:
                failures.append(f"step {i}: missing fused DP checksum diagnostic {key}")
        target_by_rank = step.get("fused_hybrid_weight_update_checksum_target_local_by_rank")
        if target_by_rank is not None and len(target_by_rank) != 2:
            failures.append(f"step {i}: target_local_by_rank length is not 2")
        rel_max = step.get("fused_hybrid_weight_update_checksum_rel_error_max")
        if rel_max is not None and rel_max > 1e-6:
            failures.append(f"step {i}: DP checksum max rel_err={rel_max:.2e} > 1e-6")
    return failures


def check_fused_hybrid_checkpoint_save_drained(train_steps, run_log, name):
    """Checkpoint save results must be drained before the next fused rollout command."""
    failures = []
    if len(train_steps) != 3:
        failures.append(f"expected 3 train steps for checkpoint regression, got {len(train_steps)}")

    for marker in (
        "KeyError: 'full_token_lists'",
        "fused hybrid generate returned a non-generation result",
    ):
        if marker in run_log:
            failures.append(f"found stale queue failure marker in run.log: {marker}")

    for step in range(1, 4):
        dispatch_marker = (
            "[Pipeline] Checkpoint save dispatched: "
            f"results/integration_tests/{name}/checkpoints/step_{step}"
        )
        drained_marker = (
            f"[FusedHybrid] checkpoint_save_result_drained step={step} status=saved"
        )
        saved_marker = f"[Trainer-FSDP] Checkpoint saved at step {step} ->"
        dispatch_pos = run_log.find(dispatch_marker)
        drained_pos = run_log.find(drained_marker)
        saved_pos = run_log.find(saved_marker)
        if dispatch_pos < 0:
            failures.append(f"missing checkpoint dispatch for step {step}")
            continue
        if drained_pos < 0:
            failures.append(f"missing checkpoint queue-drain marker for step {step}")
            continue
        if drained_pos < dispatch_pos:
            failures.append(f"checkpoint step {step} queue-drain marker before dispatch")
            continue
        if saved_pos < 0:
            failures.append(f"missing async checkpoint disk-write completion for step {step}")

        next_rollout_pos = run_log.find(
            "[FusedHybrid] phase=rollout actor_version=",
            dispatch_pos + len(dispatch_marker),
        )
        if step < 3 and next_rollout_pos < 0:
            failures.append(f"missing next rollout after checkpoint step {step}")
        elif step < 3 and next_rollout_pos < drained_pos:
            failures.append(
                f"checkpoint step {step} queue result was not drained before the next rollout command"
            )

    return failures


def check_mini_batch_ratio_divergence(train_steps, run_log, name):
    """Verify decoupled PPO ratio ≈ 1 in first mini-batch, != 1 in later ones.

    With mini_batch_size=2 and decoupled loss, pi_prox is frozen before the
    mini-batch loop. Mini-batch 0: ratio=pi_theta/pi_prox ≈ 1 (no update yet).
    Mini-batch 1+: ratio diverges because optimizer stepped after mini-batch 0.
    """
    failures = []
    if len(train_steps) < 2:
        failures.append("Need at least 2 train steps for ratio divergence check")
        return failures
    for s in train_steps:
        step = s.get("step", "?")
        r0 = s.get("r_mean_mini_0")
        r1 = s.get("r_mean_mini_1")
        if r0 is None or r1 is None:
            failures.append(f"step {step}: r_mean_mini_0/1 not logged")
            break
        # First mini-batch: ratio must be ≈ 1 (prox just computed, no update yet)
        if abs(r0 - 1.0) > 0.05:
            failures.append(
                f"step {step}: r_mean_mini_0={r0:.6f}, expected ≈1.0")
        # Second mini-batch at step 2+: ratio must diverge (model updated)
        if s in train_steps[1:] and abs(r1 - 1.0) < 1e-6:
            failures.append(
                f"step {step}: r_mean_mini_1={r1:.6f}, expected != 1.0")
    return failures


def check_actor_critic_metrics(train_steps, run_log, name):
    """Verify actor-critic metrics are logged and finite."""
    failures = []
    if not train_steps:
        return ["No train steps in log.jsonl"]
    for s in train_steps:
        step = s.get("step", "?")
        for key in ("actor_loss", "value_loss", "return_mean", "value_mean", "td_error_mean"):
            val = s.get(key)
            if val is None:
                failures.append(f"step {step}: missing {key}")
                break
            if not isinstance(val, (int, float)):
                failures.append(f"step {step}: non-scalar {key}={val!r}")
                break
            if math.isnan(val) or math.isinf(val):
                failures.append(f"step {step}: non-finite {key}={val}")
                break
        if failures:
            break
    return failures


def check_actor_loss_matches_standard_pg_init(train_steps, run_log, name):
    """For deterministic TD(0) AC-PG, step-1 actor loss should match baseline PG.

    With a zero-initialized value head and TD(0) targets, the initial
    advantages reduce to the standard PG-KL advantages. We only assert this
    at the first train step; later steps are expected to diverge once the
    value head is updated.
    """
    if not train_steps:
        return ["No train steps in log.jsonl"]
    step1 = train_steps[0]
    actor_loss = step1.get("actor_loss")
    if actor_loss is None:
        return ["step 1: missing actor_loss"]
    expected = GOLDEN_LOSSES[("det_pg_kl_nopack", 1)]
    if abs(actor_loss - expected) > 1e-6:
        return [
            f"step 1: actor_loss={actor_loss:.10f}, expected det_pg_kl_nopack "
            f"golden {expected:.10f}"
        ]
    return []


def check_per_mini_metrics(train_steps, run_log, name):
    """Validate per-mini diagnostics when a run uses multi-mini training."""
    failures = []
    if not train_steps:
        return ["No train steps in log.jsonl"]

    for s in train_steps:
        step = s.get("step", "?")
        n_optim = int(s.get("n_optim_steps", 1) or 1)
        mini_ids = sorted(
            int(k.rsplit("_", 1)[1])
            for k in s.keys()
            if k.startswith("n_tokens_mini_")
        )

        if n_optim <= 1:
            continue

        if len(mini_ids) != n_optim:
            failures.append(
                f"step {step}: expected {n_optim} mini diagnostics, found {len(mini_ids)}"
            )
            break

        for mi in mini_ids:
            tok = s.get(f"n_tokens_mini_{mi}")
            n_seq = s.get(f"n_seqs_mini_{mi}")
            avg_resp = s.get(f"avg_response_length_mini_{mi}")
            p90_resp = s.get(f"response_length_p90_mini_{mi}")
            if None in (tok, n_seq, avg_resp, p90_resp):
                failures.append(
                    f"step {step}: missing batch-shape mini metrics for mini {mi}"
                )
                break
            if n_seq <= 0:
                failures.append(f"step {step}: n_seqs_mini_{mi}={n_seq} (expected > 0)")
                break
            expected_tok = avg_resp * n_seq
            if abs(expected_tok - tok) > 1e-4:
                failures.append(
                    f"step {step}: n_tokens_mini_{mi}={tok}, but "
                    f"avg_response_length_mini_{mi}*n_seqs_mini_{mi}={expected_tok}"
                )
                break
            if p90_resp < 0:
                failures.append(
                    f"step {step}: response_length_p90_mini_{mi}={p90_resp} (expected >= 0)"
                )
                break

            if f"r_mean_mini_{mi}" in s and f"r_p99_mini_{mi}" not in s:
                failures.append(f"step {step}: missing r_p99_mini_{mi}")
                break
            if f"adv_mean_mini_{mi}" in s and f"adv_std_mini_{mi}" not in s:
                failures.append(f"step {step}: missing adv_std_mini_{mi}")
                break
            if f"return_mean_mini_{mi}" in s:
                needed = (
                    f"return_std_mini_{mi}",
                    f"value_mean_mini_{mi}",
                    f"value_std_mini_{mi}",
                )
                for key in needed:
                    if key not in s:
                        failures.append(f"step {step}: missing {key}")
                        break
                if failures:
                    break
        if failures:
            break
    return failures


CHECK_FNS = {
    "kl_step1": check_kl_step1,
    "adv_step1": check_adv_step1,
    "ratio_step1": check_ratio_step1,
    "ratio_std_step1": check_ratio_std_step1,
    "parity_step1_loss": check_parity_step1_loss,
    "weight_checksum": check_weight_checksum,
    "strict_weight_checksum": check_strict_weight_checksum,
    "megatron_kl_step1_zero": check_megatron_kl_step1_zero,
    "megatron_adv_step1_zero": check_megatron_adv_step1_zero,
    "megatron_ratio_step1_one": check_megatron_ratio_step1_one,
    "lora_checksum": check_lora_checksum,
    "grpo_kl": check_grpo_kl,
    "behave_imp_weight_logged": check_behave_imp_weight_logged,
    "m2po_logged": check_m2po_logged,
    "mof_metrics": check_mof_metrics,
    "config_deterministic": check_config_deterministic,
    "eval_ran": check_eval_ran,
    "trace": check_trace,
    "async_stepoff_trace": check_async_stepoff_trace,
    "async_stepoff_zero_trace": check_async_stepoff_zero_trace,
    "async_stepoff_staleness": check_async_stepoff_staleness,
    "direct_teacher_artifacts": check_direct_teacher_artifacts,
    "hidden_recompute_artifacts": check_hidden_recompute_artifacts,
    "golden_loss": check_golden_loss,
    "dense_recompute_matches_baseline": check_dense_recompute_matches_baseline,
    "keep_mode_sequence": check_keep_mode_sequence,
    "dapo_features": check_dapo_features,
    "finite_loss": check_finite_loss,
    "sft_perplexity_eval": check_sft_perplexity_eval,
    "checkpoint_saved": check_checkpoint_saved,
    "dapo_filter_overlong_config": check_dapo_filter_overlong_config,
    "opsd_self_score": check_opsd_self_score,
    "megatron_backend": check_megatron_backend,
    "ray_multinode": check_ray_multinode,
    "ray_megatron_trainer_spans_nodes": check_ray_megatron_trainer_spans_nodes,
    "grpo_mini_batch_invariance": check_grpo_mini_batch_invariance,
    "grpo_staleness_invariance": check_grpo_staleness_invariance,
    "grpo_async_staleness": check_grpo_async_staleness,
    "staleness_check": check_staleness,
    "async_staleness_check": check_async_staleness,
    "n_optim_steps": check_n_optim_steps,
    "mini_batch_ratio_divergence": check_mini_batch_ratio_divergence,
    "actor_critic_metrics": check_actor_critic_metrics,
    "actor_matches_standard_pg_init": check_actor_loss_matches_standard_pg_init,
    "per_mini_metrics": check_per_mini_metrics,
    "uneven_global_mini_split": check_uneven_global_mini_split,
    "fused_hybrid_sync_sequence": check_fused_hybrid_sync_sequence,
    "fused_hybrid_bucket_metrics": check_fused_hybrid_bucket_metrics,
    "fused_hybrid_weight_checksum": check_fused_hybrid_weight_checksum,
    "fused_hybrid_dp_metrics": check_fused_hybrid_dp_metrics,
    "fused_hybrid_checkpoint_save_drained": check_fused_hybrid_checkpoint_save_drained,
}

SPECIAL_CHECKS = {"checkpoint_resume", "standalone_eval_cli"}


# ──────────────────────────────────────────────────────────────
#  Runner
# ──────────────────────────────────────────────────────────────

def _parse_run_outputs(name):
    """Return (train_steps, eval_steps, run_log) for a completed integration run."""
    result_dir = RESULTS_DIR / name
    log_jsonl = result_dir / "log.jsonl"
    run_log_path = result_dir / "run.log"

    train_steps = []
    eval_steps = []
    if log_jsonl.exists():
        with open(log_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "train":
                    train_steps.append(entry)
                elif entry.get("type") == "eval":
                    eval_steps.append(entry)

    run_log = run_log_path.read_text() if run_log_path.exists() else ""
    return train_steps, eval_steps, run_log


def _run_standalone_eval_cli(name, config_path, cfg):
    """Run the evaluation module against the generated config and verify outputs."""
    output_dir = RESULTS_DIR / name / "standalone_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = cfg.get("rollout", {}).get("gpu_ids") or cfg.get("trainer", {}).get("gpu_ids", "0")
    first_gpu = str(gpus).split(",")[0].strip() or "0"
    eval_cmd = [
        PYTHON, "-m", "opd.cli.eval",
        "--config", str(config_path.relative_to(PROJECT_ROOT)),
        "--model", "student",
        "--gpus", first_gpu,
        "--dp", "1",
        "--eval-n-samples", "1",
        "--max-response-length", "8",
        "--datasets", GSM8K_TRAIN_FILE,
        "--output-dir", str(output_dir.relative_to(PROJECT_ROOT)),
        "--output-name", "student.jsonl",
    ]
    proc = subprocess.run(
        eval_cmd,
        cwd=str(PROJECT_ROOT),
        timeout=LONG_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        return [f"standalone eval CLI exited {proc.returncode}"]
    outputs = list(output_dir.glob("*.jsonl"))
    if not outputs:
        return [f"standalone eval produced no jsonl files in {output_dir}"]
    return []


def _run_checks(checks, train_steps, eval_steps, run_log, name):
    """Run normal check functions and return (failures, check_results)."""
    failures = []
    check_results = []
    for check_name in checks:
        if check_name in SPECIAL_CHECKS:
            continue
        fn = CHECK_FNS[check_name]
        fails = fn(train_steps, run_log, name)
        if fails:
            check_results.append((check_name, "FAIL"))
            failures.extend(fails)
        else:
            check_results.append((check_name, "ok"))

    return failures, check_results


def run_checkpoint_resume(name, config_fn, checks):
    """Run a two-phase save/resume integration test."""
    cfg = config_fn()
    try:
        apply_integration_gpu_map(cfg)
    except ValueError as exc:
        return {"name": name, "status": "SKIP", "elapsed_s": 0.0, "reason": str(exc)}
    cfg.setdefault("trainer", {})["total_steps"] = 1
    cfg["trainer"]["save_freq"] = 1
    cfg.setdefault("eval", {})["freq"] = -1
    cfg["eval"]["before_train"] = False

    config_path = CONFIG_DIR / f"{name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"  Running: {name} (checkpoint resume phase 1/2)")
    print(f"  Checks: {', '.join(checks)}")
    print(f"{'='*60}\n", flush=True)

    reason = preflight_config_skip_reason(name, cfg)
    if reason:
        return {"name": name, "status": "SKIP", "elapsed_s": 0.0, "reason": reason}

    rel_config = str(config_path.relative_to(PROJECT_ROOT))
    t0 = time.time()
    first = subprocess.run(
        [PYTHON, "-m", "opd.cli.train", "--config", rel_config, "--overwrite", "--allow-dirty"],
        cwd=str(PROJECT_ROOT),
        timeout=LONG_TIMEOUT_SECONDS,
    )
    if first.returncode != 0:
        result = {
            "name": name,
            "status": "CRASH",
            "elapsed_s": round(time.time() - t0, 1),
            "exit_code": first.returncode,
            "reason": f"phase1 exit_code={first.returncode}",
        }
        run_log_path = RESULTS_DIR / name / "run.log"
        if run_log_path.exists():
            result["tail"] = "\n".join(run_log_path.read_text().strip().split("\n")[-15:])
        return result

    ckpt1 = RESULTS_DIR / name / "checkpoints" / "step_1"
    phase_failures = []
    if not (ckpt1 / "model.pt").exists():
        phase_failures.append(f"phase1 checkpoint missing: {ckpt1 / 'model.pt'}")
    if not (ckpt1 / "training_state.pt").exists():
        phase_failures.append(f"phase1 training state missing: {ckpt1 / 'training_state.pt'}")

    cfg["trainer"]["total_steps"] = 2
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"  Running: {name} (checkpoint resume phase 2/2)")
    print(f"{'='*60}\n", flush=True)

    second = subprocess.run(
        [PYTHON, "-m", "opd.cli.train", "--config", rel_config, "--resume", "--allow-dirty"],
        cwd=str(PROJECT_ROOT),
        timeout=LONG_TIMEOUT_SECONDS,
    )
    elapsed = time.time() - t0

    result = {
        "name": name,
        "elapsed_s": round(elapsed, 1),
        "exit_code": second.returncode,
    }
    if second.returncode != 0:
        result["status"] = "CRASH"
        result["reason"] = f"phase2 exit_code={second.returncode}"
        run_log_path = RESULTS_DIR / name / "run.log"
        if run_log_path.exists():
            result["tail"] = "\n".join(run_log_path.read_text().strip().split("\n")[-15:])
        return result

    train_steps, eval_steps, run_log = _parse_run_outputs(name)
    if not any(int(s.get("step", 0)) >= 2 for s in train_steps):
        phase_failures.append("resume phase did not log/train through step 2")
    if "Resumed from checkpoint step 1" not in run_log:
        phase_failures.append("run.log lacks resume-from-step-1 marker")

    normal_checks = [c for c in checks if c != "checkpoint_resume"]
    failures, check_results = _run_checks(normal_checks, train_steps, eval_steps, run_log, name)
    failures = phase_failures + failures
    check_results.insert(0, ("checkpoint_resume", "FAIL" if phase_failures else "ok"))

    if failures:
        result["status"] = "FAIL"
        result["failures"] = failures
    else:
        result["status"] = "PASS"
        if train_steps:
            result["step1_kl"] = train_steps[0].get("kl_loss", train_steps[0].get("mean_kl", "?"))
    result["check_results"] = check_results
    result["_train_steps"] = train_steps
    return result


def run_one(name, config_fn, checks):
    """Run a single integration test. Returns result dict."""
    if "checkpoint_resume" in checks:
        return run_checkpoint_resume(name, config_fn, checks)

    # Generate config in new format
    cfg = config_fn()
    try:
        apply_integration_gpu_map(cfg)
    except ValueError as exc:
        return {"name": name, "status": "SKIP", "elapsed_s": 0.0, "reason": str(exc)}
    config_path = CONFIG_DIR / f"{name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    result_dir = RESULTS_DIR / name
    log_jsonl = result_dir / "log.jsonl"
    run_log_path = result_dir / "run.log"

    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  Checks: {', '.join(checks)}")
    print(f"{'='*60}\n", flush=True)

    reason = preflight_config_skip_reason(name, cfg)
    if reason:
        return {
            "name": name,
            "status": "SKIP",
            "elapsed_s": 0.0,
            "reason": reason,
        }

    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, "-m", "opd.cli.train",
         "--config", str(config_path.relative_to(PROJECT_ROOT)),
         "--overwrite", "--allow-dirty"],
        cwd=str(PROJECT_ROOT),
        timeout=RUN_TIMEOUT_SECONDS,
    )
    elapsed = time.time() - t0

    result = {
        "name": name,
        "elapsed_s": round(elapsed, 1),
        "exit_code": proc.returncode,
    }

    if proc.returncode != 0:
        result["status"] = "CRASH"
        result["reason"] = f"exit_code={proc.returncode}"
        if run_log_path.exists():
            lines = run_log_path.read_text().strip().split("\n")
            result["tail"] = "\n".join(lines[-15:])
        return result

    train_steps, eval_steps, run_log = _parse_run_outputs(name)

    if not train_steps and not run_log:
        result["status"] = "FAIL"
        result["reason"] = "No log.jsonl or run.log found"
        return result

    failures, check_results = _run_checks(checks, train_steps, eval_steps, run_log, name)

    if "standalone_eval_cli" in checks:
        eval_failures = _run_standalone_eval_cli(name, config_path, cfg)
        if eval_failures:
            failures.extend(eval_failures)
            check_results.append(("standalone_eval_cli", "FAIL"))
        else:
            check_results.append(("standalone_eval_cli", "ok"))

    # Summarize
    if failures:
        result["status"] = "FAIL"
        result["failures"] = failures
    else:
        result["status"] = "PASS"
        if train_steps:
            s1 = train_steps[0]
            # For GRPO, show mean_kl; for OPD, show kl_loss
            if "mean_kl" in s1:
                result["step1_kl"] = s1["mean_kl"]
            else:
                result["step1_kl"] = s1.get("kl_loss", "?")
            adv = s1.get("adv_mean")
            if adv is not None:
                result["step1_adv"] = adv

    chk_ok = run_log.count("Weight checksum OK")
    chk_fail = run_log.count("Weight checksum mismatch")
    if chk_ok or chk_fail:
        result["checksum"] = f"{chk_ok}ok/{chk_fail}fail"
    if check_results:
        result["check_results"] = check_results

    # Store train_steps for golden value capture
    result["_train_steps"] = train_steps

    return result


def main():
    parser = argparse.ArgumentParser(description="Run OPD integration tests")
    parser.add_argument("--suite", choices=["fsdp", "megatron", "both", "all"],
                        default="fsdp",
                        help="Backend suite to run/list: fsdp (default), megatron, or both")
    parser.add_argument("--configs", nargs="*", help="Specific test names to run")
    parser.add_argument("--filter", type=str, help="Substring filter for test names")
    parser.add_argument("--list", action="store_true", help="List tests and exit")
    parser.add_argument("--capture-golden", action="store_true",
                        help="Capture golden loss values for deterministic tests")
    parser.add_argument("--parallel", type=int, default=8,
                        help="Run N tests in parallel (default: 8)")
    parser.add_argument("--timeout", type=int,
                        help=("Per-test subprocess timeout in seconds "
                              "(default: 120, or OPD_INTEGRATION_TIMEOUT_SECONDS)"))
    parser.add_argument("--allow-skip", action="store_true",
                        help="Allow selected tests to skip without a non-zero exit")
    args = parser.parse_args()
    if args.timeout is not None:
        if args.timeout <= 0:
            parser.error("--timeout must be positive")
        global RUN_TIMEOUT_SECONDS, LONG_TIMEOUT_SECONDS
        RUN_TIMEOUT_SECONDS = args.timeout
        LONG_TIMEOUT_SECONDS = args.timeout

    registry = build_test_registry()
    all_names = sorted(registry.keys())
    suite_names = _selected_by_suite(all_names, args.suite)

    if args.list:
        print(f"Available integration tests ({len(suite_names)} selected by --suite {args.suite}):\n")
        for name in suite_names:
            _, checks = registry[name]
            suite = suite_for_test(name)
            print(f"  {name:45s}  suite: {suite:8s} checks: {', '.join(checks)}")
        return

    _ensure_gsm8k_fixture()
    _ensure_golden_gsm8k_fixture()
    _ensure_sft_fixture()

    # Auto-create tiny test models if not present.  This intentionally happens
    # after --list so public users can inspect the registry without downloads.
    student_path = PROJECT_ROOT / "tests" / "fixtures" / "tiny_student"
    teacher_path = PROJECT_ROOT / "tests" / "fixtures" / "tiny_teacher"
    if not (student_path / "config.json").exists() or not (teacher_path / "config.json").exists():
        print("Test models not found — creating tiny models...")
        subprocess.run([PYTHON, str(PROJECT_ROOT / "scripts" / "create_test_models.py")],
                       cwd=str(PROJECT_ROOT), check=True)

    # Select tests
    if args.configs:
        names = args.configs
        for n in names:
            if n not in registry:
                print(f"ERROR: Unknown test '{n}'. Use --list.")
                sys.exit(1)
    elif args.filter:
        names = [n for n in suite_names if args.filter in n]
        if not names:
            print(
                f"No {args.suite} tests matching filter '{args.filter}'. "
                "Use --list or choose --suite megatron/both."
            )
            sys.exit(1)
    else:
        names = suite_names

    # Free GPUs before starting — prevents hangs from orphaned processes
    free_gpus = PROJECT_ROOT / "scripts" / "free_gpus.py"
    if free_gpus.exists():
        gpu_ids = set()
        for n in names:
            if preflight_name_skip_reason(n):
                continue
            if "multinode" in n and _truthy_env("OPD_ENABLE_MULTINODE_INTEGRATION"):
                continue
            cfg_fn, _ = registry[n]
            cfg = cfg_fn()
            gpu_ids.update(cfg.get("teacher", {}).get("gpu_ids", "").split(","))
            gpu_ids.update(cfg.get("rollout", {}).get("gpu_ids", "").split(","))
            gpu_ids.update(cfg.get("trainer", {}).get("gpu_ids", "").split(","))
        gpu_ids.discard("")
        if gpu_ids:
            gpu_str = ",".join(sorted(gpu_ids))
            print(f"Freeing GPUs {gpu_str} before test run...")
            subprocess.run([PYTHON, str(free_gpus), gpu_str], timeout=15)

    parallel = args.parallel
    print(f"Running {len(names)} integration tests (suite={args.suite}, parallel={parallel})...\n")

    # Live summary file — appended after each test completes
    summary_path = os.path.join("results", "integration_tests", "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    import datetime
    with open(summary_path, "w") as f:
        f.write(f"Integration Test Run — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{len(names)} tests, suite={args.suite}, parallel={parallel}\n")
        f.write(f"{'='*60}\n")

    def _run_one_safe(name):
        reason = preflight_name_skip_reason(name)
        if reason:
            return {
                "name": name,
                "status": "SKIP",
                "elapsed_s": 0.0,
                "reason": reason,
            }
        config_fn, checks = registry[name]
        try:
            return run_one(name, config_fn, checks)
        except subprocess.TimeoutExpired:
            return {"name": name, "status": "TIMEOUT", "reason": f"Exceeded {RUN_TIMEOUT_SECONDS}s timeout"}
        except Exception as e:
            return {"name": name, "status": "ERROR", "reason": str(e)}

    def _print_result(r):
        status = r["status"]
        icon = {
            "PASS": "✓",
            "FAIL": "✗",
            "CRASH": "💥",
            "TIMEOUT": "⏱",
            "ERROR": "⚠",
            "SKIP": "↷",
        }
        elapsed = r.get("elapsed_s", "?")
        extra_parts = []
        if status == "PASS":
            if "step1_kl" in r:
                kl = r["step1_kl"]
                extra_parts.append(f"kl={kl:.3e}" if isinstance(kl, float) else f"kl={kl}")
            if "step1_adv" in r:
                extra_parts.append(f"adv={r['step1_adv']:.3e}")
            if "checksum" in r:
                extra_parts.append(f"chk={r['checksum']}")
            if "check_results" in r:
                checks_str = " ".join(f"{n}={s}" for n, s in r["check_results"])
                extra_parts.append(f"[{checks_str}]")
            extra = " " + " ".join(extra_parts) if extra_parts else ""
        elif "failures" in r:
            extra = "\n    " + "\n    ".join(r["failures"])
            if "check_results" in r:
                checks_str = " ".join(f"{n}={s}" for n, s in r["check_results"])
                extra += f"\n    [{checks_str}]"
        elif "reason" in r:
            extra = f" ({r['reason']})"
        else:
            extra = ""
        line = f"{icon.get(status, '?')} {r['name']}: {status} ({elapsed}s){extra}"
        print(f"\n{line}", flush=True)
        # Append to live summary file
        with open(summary_path, "a") as f:
            f.write(f"{line}\n")

    def _run_batch(batch_names, max_workers, label=""):
        """Run a batch of tests with given parallelism."""
        batch_results = []
        if not batch_names:
            return batch_results
        if label:
            print(f"\n--- {label} ({len(batch_names)} tests, parallel={max_workers}) ---\n",
                  flush=True)
        if max_workers <= 1:
            for name in batch_names:
                r = _run_one_safe(name)
                batch_results.append(r)
                _print_result(r)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_one_safe, n): n for n in batch_names}
                for fut in as_completed(futures):
                    r = fut.result()
                    batch_results.append(r)
                    _print_result(r)
        return batch_results

    if parallel <= 1:
        results = _run_batch(names, 1)
    else:
        # Split into HF (det_), vLLM/FSDP, and Megatron batches for optimal parallelism.
        # HF deterministic tests are GPU-bound and many share the same fixed
        # local GPU assignment (often GPU 0), so running them in parallel can
        # induce avoidable OOM/kill flakiness. Keep them sequential by default.
        # vLLM tests still use NCCL → limited parallelism to avoid deadlocks.
        megatron_names = [n for n in names if suite_for_test(n) == "megatron"]
        fsdp_names = [n for n in names if suite_for_test(n) == "fsdp"]
        det_names = [n for n in fsdp_names if n.startswith("det_")]
        vllm_names = [n for n in fsdp_names if not n.startswith("det_")]
        gpu_count = detect_local_gpu_count()
        DET_PARALLEL = 1
        VLLM_PARALLEL = 1  # vLLM + NCCL weight transfer deadlocks when parallel on shared GPUs
        MEGATRON_PARALLEL = 1  # Megatron distributed process groups need isolation
        if det_names:
            print(f"[integration] local_gpu_count={gpu_count} -> deterministic_parallel={DET_PARALLEL}",
                  flush=True)

        results = []
        if det_names:
            results.extend(_run_batch(det_names, DET_PARALLEL, "HF deterministic tests"))
        if vllm_names:
            results.extend(_run_batch(vllm_names, VLLM_PARALLEL, "vLLM integration tests"))
        if megatron_names:
            results.extend(_run_batch(megatron_names, MEGATRON_PARALLEL, "Megatron integration tests"))

    # Re-sort by original order for summary
    name_order = {n: i for i, n in enumerate(names)}
    results.sort(key=lambda r: name_order.get(r["name"], 999))

    # Retry timeouts and crashes (solo, no contention)
    retry_statuses = {"TIMEOUT", "CRASH", "ERROR"}
    retries = [r for r in results if r["status"] in retry_statuses]
    if retries and parallel > 1:
        print(f"\n{'='*60}")
        print(f"  Retrying {len(retries)} failed tests (sequential)...")
        print(f"{'='*60}")
        for r in retries:
            name = r["name"]
            print(f"\n  Retrying: {name} (was {r['status']})", flush=True)
            r2 = _run_one_safe(name)
            _print_result(r2)
            # Replace result if retry succeeded
            for i, orig in enumerate(results):
                if orig["name"] == name:
                    if r2["status"] == "PASS":
                        results[i] = r2
                        results[i]["_retried"] = True
                    break

    # Summary
    print(f"\n{'='*60}")
    print(f"  INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    crashed = sum(1 for r in results if r["status"] == "CRASH")
    other = sum(1 for r in results if r["status"] in ("TIMEOUT", "ERROR"))
    total = len(results)

    for r in results:
        status = r["status"]
        icon = {
            "PASS": "✓",
            "FAIL": "✗",
            "CRASH": "💥",
            "TIMEOUT": "⏱",
            "ERROR": "⚠",
            "SKIP": "↷",
        }
        print(f"  {icon.get(status, '?')} {r['name']:45s} {status:6s} ({r.get('elapsed_s', '?')}s)")

    print(
        f"\n  {passed}/{total} passed, {skipped} skipped, "
        f"{failed} failed, {crashed} crashed, {other} other"
    )

    explicit_selection = bool(args.configs or args.filter)
    skipped_is_error = bool(skipped and explicit_selection and not args.allow_skip)
    if skipped_is_error:
        print(
            "\nSKIPPED selected tests without --allow-skip; treating as non-success."
        )

    if failed + crashed > 0:
        print("\nFAILED:")
        for r in results:
            if r["status"] in ("FAIL", "CRASH"):
                print(f"\n  {r['name']}:")
                if "failures" in r:
                    for f in r["failures"]:
                        print(f"    {f}")
                if "reason" in r:
                    print(f"    {r['reason']}")
                if "tail" in r:
                    for line in r["tail"].split("\n")[-5:]:
                        print(f"    {line}")

    # Golden value capture for deterministic tests
    if args.capture_golden:
        print(f"\n{'='*60}")
        print(f"  GOLDEN LOSS VALUES (paste into GOLDEN_LOSSES dict)")
        print(f"{'='*60}")
        print("GOLDEN_LOSSES = {")
        for r in results:
            if not r["name"].startswith("det_") or r["status"] == "CRASH":
                continue
            for step_entry in r.get("_train_steps", []):
                step = step_entry.get("step")
                loss = step_entry.get("kl_loss", step_entry.get("mean_kl"))
                if loss is not None and step is not None:
                    print(f'    ("{r["name"]}", {step}): {loss!r},')
        print("}")

    # Append final totals to live summary file
    with open(summary_path, "a") as f:
        f.write(f"{'='*60}\n")
        f.write(
            f"{passed}/{total} passed, {skipped} skipped, "
            f"{failed} failed, {crashed} crashed, {other} other\n"
        )
        if skipped_is_error:
            f.write("SKIPPED selected tests without --allow-skip; treating as non-success.\n")
        if failed + crashed > 0:
            f.write("\nFAILED:\n")
            for r in results:
                if r["status"] in ("FAIL", "CRASH"):
                    f.write(f"  {r['name']}:\n")
                    if "failures" in r:
                        for fail in r["failures"]:
                            f.write(f"    {fail}\n")
                    if "reason" in r:
                        f.write(f"    {r['reason']}\n")
    print(f"\nSummary written to {summary_path}")

    sys.exit(0 if failed + crashed + other == 0 and not skipped_is_error else 1)


if __name__ == "__main__":
    main()

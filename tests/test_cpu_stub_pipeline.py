"""CPU-only whole-pipeline smoke tests using pytest-only process stubs.

The tests monkeypatch the normal local process lifecycle to replace HF/vLLM
rollout, HF/vLLM teacher, and FSDP trainer entry points with deterministic CPU
stubs. The production coordinator, queue proxies, OPD payload assembly, batch
preparation, CPU weight-sync path, scheduler, and OPD KL loss are still reused.
"""

from __future__ import annotations

import atexit
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from opd.coordinator.factory import create_coordinator
from opd.utils.config import OPDConfig
from tests.cpu_stub_backends import (
    cpu_stub_grpo_trainer_main,
    cpu_stub_opd_trainer_main,
    cpu_stub_rollout_worker_main,
    cpu_stub_sft_trainer_main,
    cpu_stub_teacher_server_main,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            **_kwargs):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        if tokenize:
            return self._encode(text)
        return text

    def _encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        chars = [ch for ch in str(text) if not ch.isspace()]
        tokens = [2 + ((ord(ch) + i * 17) % 200) for i, ch in enumerate(chars)]
        if add_special_tokens:
            tokens = tokens + [self.eos_token_id]
        return tokens or [self.eos_token_id]

    def __call__(self, text, max_length=None, truncation=True, padding=False,
                 add_special_tokens=True, return_tensors=None, **_kwargs):
        tokens = self._encode(str(text), add_special_tokens=add_special_tokens)
        if max_length is not None and truncation:
            tokens = tokens[: int(max_length)]
        attention = [1] * len(tokens)
        if padding == "max_length":
            if max_length is None:
                raise ValueError("padding='max_length' requires max_length")
            pad_len = max(0, int(max_length) - len(tokens))
            if self.padding_side == "left":
                tokens = [self.pad_token_id] * pad_len + tokens
                attention = [0] * pad_len + attention
            else:
                tokens = tokens + [self.pad_token_id] * pad_len
                attention = attention + [0] * pad_len
        encoded = {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
        }
        if return_tensors == "pt" or return_tensors is None:
            return encoded
        raise ValueError(f"FakeTokenizer only supports return_tensors='pt', got {return_tensors!r}")

    def decode(self, token_ids, skip_special_tokens=True):
        ids = [int(t) for t in token_ids]
        if skip_special_tokens:
            ids = [t for t in ids if t not in {self.pad_token_id, self.eos_token_id}]
        return " ".join(str(t) for t in ids)


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _write_prompt_data(path: Path, *, n_rows: int = 4,
                       prompts: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if prompts is None:
        prompts = [f"What is {i} + {i}?" for i in range(n_rows)]
    else:
        n_rows = len(prompts)
    pd.DataFrame(
        {
            "prompt": prompts,
            "answer": [str(2 * i) for i in range(n_rows)],
        }
    ).to_parquet(path)
    return path


def _write_sft_data(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "prompt": ["Say hello", "Say bye"],
            "completion": ["hello", "bye"],
        }
    ).to_parquet(path)
    return path


def _base_train_config(
    train_file: Path,
    *,
    total_steps: int,
    step_off: int = 0,
    batch_size: int = 2,
    rollout_gpu_ids: str = "0",
    trainer_gpu_ids: str = "0",
    teacher_gpu_ids: str = "0",
    max_prompt_length: int = 6,
    use_sequence_packing: bool = False,
) -> dict:
    return {
        "deterministic": True,
        "seed": 123,
        "model": {"path": "cpu-stub-student"},
        "teacher": {
            "path": "cpu-stub-teacher",
            "backend": "hf",
            "gpu_ids": teacher_gpu_ids,
            "scoring_batch_size": 2,
            "bind_address": "127.0.0.1",
            "vllm": {"n_logprobs": 8, "tensor_parallel_size": 1},
        },
        "data": {
            "train_files": str(train_file),
            "prompt_key": "prompt",
            "answer_key": "answer",
            "max_prompt_length": max_prompt_length,
            "max_response_length": 3,
        },
        "rollout": {
            "backend": "hf",
            "gpu_ids": rollout_gpu_ids,
            "temperature": 1.0,
            "vllm": {"tensor_parallel_size": 1, "max_model_len": 16, "max_num_seqs": 8},
        },
        "trainer": {
            "backend": "fsdp",
            "gpu_ids": trainer_gpu_ids,
            "batch_size": batch_size,
            "mini_batch_size": batch_size,
            "micro_batch_size": 1,
            "total_steps": total_steps,
            "total_epochs": 1,
            "save_freq": -1,
            "save_optimizer": False,
            "use_sequence_packing": use_sequence_packing,
            "optim": {"lr": 0.0, "weight_decay": 0.0},
        },
        "algorithm": {
            "mode": "opd",
            "opd": {
                "kl_loss_mode": "forward_kl",
                "n_kl_logprobs": 8,
                "use_importance_sampling": False,
            },
        },
        "pipeline": {"n_step_off": {"step_off": step_off}},
        "eval": {"freq": -1, "mode": ["inline"], "before_train": False},
        "weight_sync": {"backend": "cpu", "verify_checksum": False},
    }


def _grpo_config(train_file: Path, *, kl_beta: float = 0.0) -> dict:
    cfg = _base_train_config(train_file, total_steps=1, step_off=0)
    if kl_beta == 0:
        cfg.pop("teacher")
    cfg["algorithm"] = {
        "mode": "grpo",
        "grpo": {
            "group_size": 2,
            "clip_eps": 0.2,
            "kl_beta": kl_beta,
            "reward_fn": "token_hash",
            "norm_adv_by_std": True,
        },
    }
    return cfg


def _sft_config(train_file: Path) -> dict:
    return {
        "deterministic": True,
        "seed": 123,
        "model": {"path": "cpu-stub-student"},
        "data": {
            "train_files": str(train_file),
            "prompt_key": "prompt",
            "completion_key": "completion",
            "max_prompt_length": 6,
            "max_response_length": 3,
        },
        "trainer": {
            "backend": "fsdp",
            "gpu_ids": "0",
            "batch_size": 2,
            "mini_batch_size": 2,
            "micro_batch_size": 1,
            "total_steps": 1,
            "total_epochs": 1,
            "save_freq": -1,
            "save_optimizer": False,
            "optim": {"lr": 0.0, "weight_decay": 0.0},
        },
        "algorithm": {"mode": "sft", "sft": {"loss_mode": "ce"}},
        "eval": {"freq": -1, "mode": ["perplexity"], "before_train": False},
        "weight_sync": {"backend": "cpu", "verify_checksum": False},
    }


@pytest.fixture
def cpu_stub_runtime(monkeypatch, tmp_path):
    event_log = tmp_path / "events.jsonl"
    monkeypatch.setenv("OPD_CPU_STUB_EVENT_LOG", str(event_log))
    monkeypatch.delenv("OPD_CPU_STUB_SHIFT_TEACHER", raising=False)

    import transformers
    import opd.coordinator.base as base_mod
    import opd.coordinator.config_mixin as config_mixin
    import opd.coordinator.process_lifecycle as lifecycle

    tokenizer_factory = staticmethod(lambda *_args, **_kwargs: FakeTokenizer())
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_factory)
    monkeypatch.setattr(config_mixin.AutoTokenizer, "from_pretrained", tokenizer_factory)
    monkeypatch.setattr(base_mod.GPUMetricsSampler, "start", lambda self: False)
    monkeypatch.setattr(base_mod.GPUMetricsSampler, "stop", lambda self, timeout=2.0: None)

    def rollout_backend(_name):
        return {
            "worker_main": cpu_stub_rollout_worker_main,
            "streaming_worker_main": lambda: cpu_stub_rollout_worker_main,
            "supports_streaming": True,
            "supports_nccl": False,
        }

    def teacher_backend(_name):
        return {"server_main": cpu_stub_teacher_server_main}

    monkeypatch.setattr(lifecycle, "get_rollout_backend", rollout_backend)
    monkeypatch.setattr(lifecycle, "get_teacher_backend", teacher_backend)

    def install_trainer(trainer_fn):
        monkeypatch.setattr(
            lifecycle.ProcessLifecycleMixin,
            "_get_fsdp_trainer_fn",
            lambda self: trainer_fn,
        )

    return event_log, install_trainer


def _run_pipeline(config_path: Path, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    oc = OPDConfig.from_yaml(config_path)
    coordinator = create_coordinator(oc, run_dir=str(run_dir), logger=None)
    try:
        coordinator.start()
        coordinator.run()
    finally:
        coordinator.shutdown()
        try:
            atexit.unregister(coordinator.shutdown)
        except Exception:
            pass


def _events(event_log: Path) -> list[dict]:
    if not event_log.exists():
        return []
    return [json.loads(line) for line in event_log.read_text().splitlines() if line.strip()]


def _filter(events: list[dict], role: str, event: str) -> list[dict]:
    return [e for e in events if e.get("role") == role and e.get("event") == event]


def test_opd_cpu_stub_pipeline_hash_alignment_kl_zero(tmp_path, cpu_stub_runtime):
    event_log, install_trainer = cpu_stub_runtime
    install_trainer(cpu_stub_opd_trainer_main)
    train_file = _write_prompt_data(tmp_path / "data" / "train.parquet", n_rows=4)
    config_path = _write_yaml(
        tmp_path / "opd.yaml",
        _base_train_config(train_file, total_steps=2, step_off=1),
    )

    _run_pipeline(config_path, tmp_path / "run-opd")

    events = _events(event_log)
    trains = _filter(events, "trainer", "train")
    assert len(_filter(events, "rollout", "generate")) == 2
    assert len(_filter(events, "teacher", "score")) == 2
    assert len(trains) == 2
    assert len(_filter(events, "rollout", "sync_weights")) == 2
    assert len(_filter(events, "trainer", "get_clean_state_dict")) == 2
    assert all(abs(t["kl_loss"]) < 1e-8 for t in trains)
    assert all(t["alignment_ok"] is True for t in trains)
    assert all(t["max_abs_logp_diff"] < 1e-8 for t in trains)


def test_opd_cpu_stub_pipeline_two_rollout_workers_split_and_merge(tmp_path, cpu_stub_runtime):
    event_log, install_trainer = cpu_stub_runtime
    install_trainer(cpu_stub_opd_trainer_main)
    train_file = _write_prompt_data(tmp_path / "data" / "train.parquet", n_rows=4)
    config_path = _write_yaml(
        tmp_path / "opd-two-rollouts.yaml",
        _base_train_config(
            train_file,
            total_steps=1,
            step_off=0,
            batch_size=4,
            rollout_gpu_ids="0,1",
            trainer_gpu_ids="2",
            teacher_gpu_ids="3",
        ),
    )

    _run_pipeline(config_path, tmp_path / "run-opd-two-rollouts")

    events = _events(event_log)
    generates = _filter(events, "rollout", "generate")
    trains = _filter(events, "trainer", "train")
    syncs = _filter(events, "rollout", "sync_weights")
    assert {g["worker_id"] for g in generates} == {0, 1}
    assert all(g["prompts"] == 2 for g in generates)
    assert len(trains) == 1
    assert abs(trains[0]["kl_loss"]) < 1e-8
    assert trains[0]["alignment_ok"] is True
    assert {s["worker_id"] for s in syncs} == {0, 1}


def test_opd_cpu_stub_pipeline_sequence_packing_alignment_kl_zero(tmp_path, cpu_stub_runtime):
    event_log, install_trainer = cpu_stub_runtime
    install_trainer(cpu_stub_opd_trainer_main)
    train_file = _write_prompt_data(
        tmp_path / "data" / "train.parquet",
        prompts=[
            "a",
            "medium prompt",
            "a much longer prompt for packing coverage",
            "tiny",
        ],
    )
    config_path = _write_yaml(
        tmp_path / "opd-packing.yaml",
        _base_train_config(
            train_file,
            total_steps=1,
            step_off=0,
            batch_size=4,
            max_prompt_length=18,
            use_sequence_packing=True,
        ),
    )

    _run_pipeline(config_path, tmp_path / "run-opd-packing")

    trains = _filter(_events(event_log), "trainer", "train")
    assert len(trains) == 1
    assert trains[0]["used_packing"] is True
    assert trains[0]["packed_tokens"] > 0
    assert abs(trains[0]["kl_loss"]) < 1e-8
    assert trains[0]["alignment_ok"] is True


def test_opd_cpu_stub_pipeline_detects_shifted_teacher_rows(
    tmp_path,
    cpu_stub_runtime,
    monkeypatch,
):
    event_log, install_trainer = cpu_stub_runtime
    install_trainer(cpu_stub_opd_trainer_main)
    monkeypatch.setenv("OPD_CPU_STUB_SHIFT_TEACHER", "1")
    train_file = _write_prompt_data(tmp_path / "data" / "train.parquet", n_rows=2)
    config_path = _write_yaml(
        tmp_path / "opd-shifted.yaml",
        _base_train_config(train_file, total_steps=1, step_off=0),
    )

    _run_pipeline(config_path, tmp_path / "run-opd-shifted")

    trains = _filter(_events(event_log), "trainer", "train")
    assert len(trains) == 1
    assert trains[0]["kl_loss"] > 1e-5
    assert trains[0]["alignment_ok"] is False
    assert trains[0]["max_abs_logp_diff"] > 1e-5


def test_grpo_cpu_stub_pipeline_skips_teacher_when_kl_beta_zero(tmp_path, cpu_stub_runtime):
    event_log, install_trainer = cpu_stub_runtime
    install_trainer(cpu_stub_grpo_trainer_main)
    train_file = _write_prompt_data(tmp_path / "data" / "train.parquet", n_rows=2)
    config_path = _write_yaml(tmp_path / "grpo.yaml", _grpo_config(train_file))

    _run_pipeline(config_path, tmp_path / "run-grpo")

    events = _events(event_log)
    trains = _filter(events, "trainer", "train")
    assert _filter(events, "teacher", "started") == []
    assert _filter(events, "teacher", "score") == []
    assert len(_filter(events, "rollout", "generate")) == 1
    assert len(trains) == 1
    assert len(_filter(events, "rollout", "sync_weights")) == 1
    assert trains[0]["mode"] == "grpo"
    assert "mean_advantage" in trains[0]
    assert trains[0]["has_ref_token_logps"] is False


def test_grpo_cpu_stub_pipeline_scores_reference_when_kl_beta_positive(tmp_path, cpu_stub_runtime):
    event_log, install_trainer = cpu_stub_runtime
    install_trainer(cpu_stub_grpo_trainer_main)
    train_file = _write_prompt_data(tmp_path / "data" / "train.parquet", n_rows=2)
    config_path = _write_yaml(tmp_path / "grpo-ref.yaml", _grpo_config(train_file, kl_beta=0.1))

    _run_pipeline(config_path, tmp_path / "run-grpo-ref")

    events = _events(event_log)
    trains = _filter(events, "trainer", "train")
    assert len(_filter(events, "teacher", "started")) == 1
    assert len(_filter(events, "teacher", "score")) == 1
    assert len(trains) == 1
    assert trains[0]["mode"] == "grpo"
    assert trains[0]["has_ref_token_logps"] is True


def test_opd_cpu_stub_pipeline_resume_from_checkpoint(tmp_path, cpu_stub_runtime):
    event_log, install_trainer = cpu_stub_runtime
    install_trainer(cpu_stub_opd_trainer_main)
    train_file = _write_prompt_data(tmp_path / "data" / "train.parquet", n_rows=4)
    run_dir = tmp_path / "run-opd-resume"

    first_cfg = _base_train_config(train_file, total_steps=1, step_off=0)
    first_cfg["trainer"]["save_freq"] = 1
    first_path = _write_yaml(tmp_path / "opd-resume-first.yaml", first_cfg)
    _run_pipeline(first_path, run_dir)

    second_cfg = _base_train_config(train_file, total_steps=2, step_off=0)
    second_cfg["trainer"]["save_freq"] = 1
    second_cfg["trainer"]["resume_from"] = "latest"
    second_path = _write_yaml(tmp_path / "opd-resume-second.yaml", second_cfg)
    _run_pipeline(second_path, run_dir)

    events = _events(event_log)
    trains = _filter(events, "trainer", "train")
    loads = _filter(events, "trainer", "load_checkpoint")
    saves = _filter(events, "trainer", "save_checkpoint")
    assert (run_dir / "checkpoints" / "step_1" / "model.pt").exists()
    assert len(trains) == 2
    assert [load["step"] for load in loads] == [1]
    assert any(save["step"] == 2 for save in saves)
    assert all(abs(t["kl_loss"]) < 1e-8 for t in trains)


def test_opd_cpu_stub_pipeline_fully_async_smoke(tmp_path, cpu_stub_runtime):
    event_log, install_trainer = cpu_stub_runtime
    install_trainer(cpu_stub_opd_trainer_main)
    train_file = _write_prompt_data(tmp_path / "data" / "train.parquet", n_rows=4)
    cfg = _base_train_config(train_file, total_steps=1, step_off=0)
    cfg["pipeline"] = {
        "scheduling_mode": "fully_async",
        "fully_async": {"staleness_threshold": 0},
    }
    config_path = _write_yaml(tmp_path / "opd-fully-async.yaml", cfg)

    _run_pipeline(config_path, tmp_path / "run-opd-fully-async")

    events = _events(event_log)
    trains = _filter(events, "trainer", "train")
    assert len(_filter(events, "rollout", "autonomous_started")) == 1
    assert len(_filter(events, "rollout", "paused")) >= 1
    assert len(_filter(events, "rollout", "resumed")) >= 1
    assert len(_filter(events, "teacher", "score")) >= 1
    assert len(trains) == 1
    assert abs(trains[0]["kl_loss"]) < 1e-8
    assert trains[0]["alignment_ok"] is True


def test_sft_cpu_stub_pipeline_trainer_only(tmp_path, cpu_stub_runtime, monkeypatch):
    event_log, _install_trainer = cpu_stub_runtime
    import opd.coordinator.sft as sft_module

    monkeypatch.setattr(sft_module, "sft_trainer_main", cpu_stub_sft_trainer_main)
    monkeypatch.setattr(sft_module.time, "sleep", lambda _seconds: None)
    train_file = _write_sft_data(tmp_path / "data" / "sft.parquet")
    config_path = _write_yaml(tmp_path / "sft.yaml", _sft_config(train_file))

    _run_pipeline(config_path, tmp_path / "run-sft")

    events = _events(event_log)
    trains = _filter(events, "trainer", "train")
    assert len(trains) == 1
    assert trains[0]["mode"] == "sft"
    assert _filter(events, "teacher", "started") == []
    assert _filter(events, "rollout", "generate") == []
    assert len(_filter(events, "trainer", "save_checkpoint")) == 1

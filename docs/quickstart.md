# Quickstart

This guide gets you from installation to a first smoke test. Use the no-GPU checks to confirm the package and CLI are installed correctly, then use the GPU smoke run for real training.

Real OPD and GRPO runs require GPUs: they use model generation, distributed training, and weight synchronization. AsyncOPD does not provide a CPU-only training backend.

## Prerequisites

| Need | Guidance |
| --- | --- |
| Python | Python 3.12. |
| GPUs | NVIDIA GPUs with a CUDA/PyTorch/vLLM-compatible driver stack for training and evaluation. |
| Disk/cache | Enough space for Hugging Face model and dataset caches. |
| Network | Needed for `hf:` datasets and model downloads unless caches are already populated. |
| Security | Keep `trust_remote_code` disabled unless you trust and pin the model repository. |

The first OPD smoke test below is designed for four GPUs. Larger Qwen3 OPD examples are designed for eight GPUs.

## 1. Install

```bash
conda create -n opd python=3.12
conda activate opd
```

If conda is not available, use a standard Python 3.12 virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
```

Then install the pinned runtime:

```bash
# Default CUDA 12.8 PyTorch build used by the pinned dependency set.
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Run source-tree commands from the repository root. Example configs live under `configs/examples/`; if you installed from a wheel, copy the config you want to use into your working directory before running `opd-train` or `opd-eval`.

## 2. Run no-GPU checks

Start with CLI help:

```bash
opd-train --help
opd-eval --help
```

Then run the lightweight CPU-safe tests:

```bash
python -m pytest \
  tests/test_cli_entrypoints.py \
  tests/test_data_format.py \
  tests/test_kl_loss.py \
  tests/test_packing.py \
  tests/test_sft_loss.py \
  tests/test_tokenizer_padding.py \
  -q -o addopts="" -m "not slow"
```

The `not slow` filter avoids tokenizer tests that may download Hugging Face model artifacts.

For a broader control-plane smoke test that still does not require GPUs:

```bash
python -m pytest tests/test_cpu_stub_pipeline.py -q -o addopts=""
```

This CPU-stub test exercises coordinator, queue/proxy, batch-prep, scheduler, weight-sync, and loss plumbing with deterministic test workers. It is not a CPU training backend.

## 3. Optional: prepare DeepMath parquet

Most starter configs use Hugging Face dataset references. Some DeepMath examples expect `data/deepmath_difficulty6/train.parquet`, which is not included in the repository. Build it from the public DeepMath source with the recipe in the [Configuration guide](configuration.md#building-datadeepmath_difficulty6trainparquet).

## 4. Run a first GPU smoke test

Start with the 4-GPU Qwen2.5 GSM8K OPD config and override it to one training step:

```bash
python -m opd.cli.train \
  --config configs/examples/opd_gsm8k_0.5b_4gpu.yaml \
  --overwrite \
  --set trainer.total_steps=1 trainer.total_epochs=1 eval.freq=-1 eval.before_train=false trainer.save_freq=-1
```

This config uses the GPUs as follows:

| GPU IDs | Role |
| --- | --- |
| `0` | teacher/reference worker |
| `1` | rollout worker |
| `2,3` | FSDP trainer |

Expected output:

```text
results/examples/opd_gsm8k_0.5b_4gpu/
  log.jsonl
  run.log
  trace.json
```

If the one-step run succeeds, remove or adjust the overrides for a longer run.

## 5. Try other examples

Qwen3 OPD, intended for eight GPUs:

```bash
python -m opd.cli.train --config configs/examples/opd_qwen3_1.7b.yaml --overwrite
```

GRPO on Qwen3, intended for eight GPUs:

```bash
python -m opd.cli.train --config configs/examples/grpo_qwen3_1.7b.yaml --overwrite
```

DAPO on DeepMath, intended for eight GPUs after preparing the DeepMath parquet:

```bash
python -m opd.cli.train --config configs/examples/dapo_deepmath_qwen3_4b_8gpu.yaml --overwrite
```

SFT template, intended for two or more trainer GPUs:

```bash
python -m opd.cli.train --config configs/examples/sft_qwen3_1.7b.yaml --overwrite
```

Before running the SFT template, set `data.train_files` and `data.val_files` to parquet files you control with columns matching `data.prompt_key` and `data.completion_key`.

## 6. Run evaluation

After training produces a compatible model or checkpoint, start with a small evaluation run:

```bash
python -m opd.cli.eval \
  --config configs/examples/opd_gsm8k_0.5b_4gpu.yaml \
  --model student \
  --gpus 0 \
  --dp 1 \
  --datasets MATH-500
```

If this works, scale `--gpus`, `--dp`, and `--tp` to match your machine.

## Common first-run failures

| Symptom | First check |
| --- | --- |
| vLLM OOM during load | Lower `rollout.vllm.gpu_memory_utilization`, `rollout.vllm.max_num_seqs`, or context length. |
| NCCL hang | Verify GPU IDs, ports, and trainer/rollout placement. |
| Dataset load failure | Confirm network access or local Hugging Face cache availability. |
| SFT path failure | Replace template `data/train.parquet` and `data/val.parquet` paths. |
| Remote-code error | Keep `trust_remote_code` false unless the model repo is trusted and pinned. |

See [Troubleshooting](troubleshooting.md) for deeper diagnosis.

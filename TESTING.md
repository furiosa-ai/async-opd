# Testing guide

Use these checks to make sure a checkout is installed correctly and that the
public examples still launch. Real OPD/GRPO runs require CUDA GPUs; the no-GPU
commands only validate imports, CLI wiring, tensor utilities, and the
coordinator control plane.

## No-GPU sanity checks

```bash
opd-train --help
opd-eval --help
python -m opd.cli.train --help
python -m opd.cli.eval --help
```

```bash
python -m pytest \
  tests/test_cli_entrypoints.py \
  tests/test_data_format.py \
  tests/test_kl_loss.py \
  tests/test_packing.py \
  tests/test_pipeline_utils.py \
  tests/test_sft_loss.py \
  tests/test_tokenizer_padding.py \
  -q -o addopts="" -m "not slow"
```

The `not slow` marker avoids tests that may download tokenizer or model
artifacts. For a broader no-GPU control-plane smoke, also run:

```bash
python -m pytest tests/test_cpu_stub_pipeline.py -q -o addopts=""
```

## One-step GPU smoke

Start with a one-step OPD run and disable eval/checkpointing:

```bash
python -m opd.cli.train \
  --config configs/examples/opd_gsm8k_0.5b_4gpu.yaml \
  --overwrite \
  --set trainer.total_steps=1 trainer.total_epochs=1 eval.freq=-1 eval.before_train=false trainer.save_freq=-1
```

To check reward-training startup as well, run a one-step GRPO smoke:

```bash
python -m opd.cli.train \
  --config configs/examples/grpo_gsm8k_0.5b_4gpu.yaml \
  --overwrite \
  --set trainer.total_steps=1 trainer.total_epochs=1 eval.freq=-1 eval.before_train=false trainer.save_freq=-1
```

After a smoke run, check that these files exist under the config output
directory:

```text
results/examples/<config-name>/log.jsonl
results/examples/<config-name>/run.log
results/examples/<config-name>/trace.json
```

## Evaluation smoke

Use a small response cap when you only want to validate startup and output
writing:

```bash
python -m opd.cli.eval \
  --config configs/examples/opd_gsm8k_0.5b_4gpu.yaml \
  --model student \
  --gpus 0 \
  --dp 1 \
  --datasets MATH-500 \
  --max-response-length 64
```

## Optional integration runner

The integration runner is for deeper GPU validation with tiny generated
fixtures. Start by listing available tests:

```bash
python scripts/run_integration_tests.py --list
```

Then run a targeted config when you have exclusive GPUs:

```bash
python scripts/run_integration_tests.py --configs public_grpo_4gpu_tiny_smoke --allow-skip
```

Use `--suite fsdp`, `--suite megatron`, or `--filter <name>` for broader
coverage after the basic README smokes pass.

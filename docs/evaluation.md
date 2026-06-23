# Evaluation guide

The evaluation CLI can run configured models, generate reusable artifacts, or score existing generation artifacts. The source-tree command is `python -m opd.cli.eval`; the installed package command is `opd-eval`.

## Command forms

Evaluate a student/base model from a config:

```bash
python -m opd.cli.eval --config configs/examples/opd_qwen3_1.7b.yaml --model student --gpus 0,1,2,3 --dp 4
```

Evaluate the configured teacher/reference model:

```bash
python -m opd.cli.eval --config configs/examples/opd_qwen3_1.7b.yaml --model teacher --gpus 0 --tp 1
```

Use the installed entry point:

```bash
opd-eval --config configs/examples/grpo_gsm8k_0.5b_4gpu.yaml --model student --gpus 0 --dp 1
```

Score an existing generation artifact without loading a model:

```bash
python -m opd.cli.eval --score-only tmp/opd-generations.json --output-dir tmp/opd-score
```

## Important CLI options

| Option | Meaning |
| --- | --- |
| `--config` | YAML config to read model/data/eval settings from. |
| `--model student` | Evaluate `model.path` by default; use `--model-path` or `--run-dir` for a trained checkpoint. |
| `--model teacher` | Evaluate `teacher.path`. |
| `--gpus` | Comma-separated GPU IDs visible to evaluation. |
| `--tp` | Tensor parallel size per model instance. |
| `--dp` | Number of data-parallel eval workers. |
| `--datasets` | Override configured eval datasets or aliases. |
| `--generate-only` | Write generation artifact and skip scoring. |
| `--score-only` | Score an existing artifact and skip generation. |
| `--output-dir` | Directory for eval metrics/artifacts. |
| `--trust-remote-code` | Unsafe opt-in for trusted, pinned model repositories only. |
| `--allow-unsafe-code-execution` | Unsafe opt-in for trusted local fixture-based code scoring. |
| `--set` | Config override, e.g. `--set eval.n_samples=1`. |

## Dataset aliases

Short aliases resolve to public HuggingFace datasets and default splits where supported:

| Alias | Use |
| --- | --- |
| `AIME25` | AIME 2025-style math eval. |
| `AMC` / `AMC23` | AMC math eval. |
| `HMMT Feb25` | HMMT February 2025-style math eval. |
| `HMMT Nov25` | HMMT November 2025-style math eval. |
| `MATH-500` | MATH-500-style symbolic/numeric math eval. |

Example:

```bash
python -m opd.cli.eval --config configs/examples/grpo_qwen3_1.7b.yaml --model student \
  --gpus 0,1,2,3 --dp 4 \
  --datasets AIME25 AMC23 "HMMT Feb25" "HMMT Nov25" MATH-500
```

## Math scoring

Math-style answer extraction tries, in order:

1. `\boxed{...}`;
2. `#### <answer>`;
3. last-number fallback.

For symbolic datasets, the evaluator preserves raw symbolic ground truth when appropriate and falls back to fuller response matching helpers for common math formats.

## Greedy vs Avg@N

| Setting | Meaning |
| --- | --- |
| `eval.n_samples: 1` | Greedy/simple accuracy-style eval. |
| `eval.n_samples > 1` | Avg@N-style metric; more generation cost and larger artifacts. |
| `eval.temperature: 0.0` | Deterministic/greedy sampling. |
| `eval.temperature: 1.0` | Stochastic sampling for pass-rate style eval. |

Override from CLI:

```bash
python -m opd.cli.eval --config configs/examples/grpo_gsm8k_0.5b_4gpu.yaml --model student \
  --gpus 0 --dp 1 --set eval.n_samples=1 eval.temperature=0.0
```

## Generation artifacts

Generation can be separated from scoring:

```bash
python -m opd.cli.eval --config configs/examples/opd_qwen3_1.7b.yaml --model student \
  --gpus 0 --generate-only tmp/opd-generations.json

python -m opd.cli.eval --score-only tmp/opd-generations.json --output-dir tmp/opd-score
```

This is useful when generation is expensive or when you are validating scoring
changes. The repo-local `tmp/` directory is gitignored and is a good place for
scratch artifacts.

Artifact contents are intended for evaluator compatibility rather than a stable public data format. Expect prompt metadata, generated responses, and scoring fields.

## Code-eval safety

Fixture-based code scoring executes generated Python on the local host. It is disabled by default.

Only enable it for trusted artifacts:

```bash
python -m opd.cli.eval --score-only tmp/trusted-code-generations.json --allow-unsafe-code-execution
```

For untrusted code generations, prefer the sandboxed post-eval grader. It scans
a run/result tree for saved `*.generations.json` code artifacts, builds or
reuses a Docker image with benchmark scorers, runs with no runtime network by
default, and stages scores under `.code_eval_scored/` before merging validated
summaries:

```bash
python scripts/grade_code_generations_sandboxed.py \
  --results-dir results/my_experiment/my_config \
  --bench humaneval_plus \
  --bench mbpp_plus \
  --workers 16 \
  --yes
```

For LiveCodeBench-style artifacts, use:

```bash
python scripts/grade_code_generations_sandboxed.py \
  --results-dir results/my_experiment/my_config \
  --bench lcb_v6 \
  --workers 16 \
  --yes
```

Use `--dry-run` first to inspect Docker build/run commands and matched
artifacts. Use `--allow-network` only if your benchmark setup explicitly
requires network access and the artifacts are trusted.

## Outputs

Common output locations:

```text
results/<experiment>/<config>/eval.jsonl
results/<experiment>/<config>/validation_outputs/
tmp/opd-score/
```

Inspect:

| File/directory | Use |
| --- | --- |
| `eval.jsonl` | Per-dataset and per-step metrics. |
| `validation_outputs/` | Sample-level generations and extracted answers. |
| score-only output dir | Recomputed metrics for an existing artifact. |

## Troubleshooting quick checks

| Symptom | First checks |
| --- | --- |
| Import fails before help | `opd-eval --help` should be lightweight; reinstall the latest wheel/source checkout. |
| Model load OOM | Lower `--tp`/`--dp`, use fewer GPUs per worker, lower context length, or reduce batch/concurrency. |
| Distributed eval hangs | Test `--gpus 0 --dp 1 --tp 1` first, then scale one dimension at a time. |
| Answers look under-extracted | Check prompt format, answer format, `answer_key`, and optional `answer_pattern`. |
| Code scoring disabled | Do not bypass unless the artifact is trusted. |
| Remote-code warning/error | Keep remote code disabled unless the model source is trusted and pinned. |

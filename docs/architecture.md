# Architecture guide

AsyncOPD separates orchestration from model work. A lightweight CPU coordinator decides what should happen next, while GPU worker processes handle rollout, teacher/reference scoring, training, and weight synchronization.

The main OPD loop is:

1. the student generates responses,
2. the teacher/reference model scores those responses,
3. the trainer updates the student,
4. updated weights are synchronized back to rollout workers.

The same infrastructure also supports GRPO/DAPO reward training and SFT.

## Components

| Component | Main files | Role |
| --- | --- | --- |
| CLI | `opd/cli/train.py`, `opd/cli/eval.py` | Load configs, parse arguments, derive output directories, and start train/eval runs. |
| Coordinator | `opd/coordinator/` | Own scheduling, process lifecycle, queue movement, mode routing, evaluation hooks, and traces. |
| Rollout | `opd/rollout/` | Generate student responses from the current policy, usually with vLLM. |
| Teacher/reference | `opd/worker/teacher/` | Score generated responses for OPD or reference-KL behavior. |
| Trainer | `opd/trainer/` | Run FSDP or Megatron updates and produce fresh student weights. |
| Losses/rewards | `opd/loss/`, `opd/reward.py` | Compute OPD KL losses, GRPO/PPO losses, SFT losses, and reward terms. |
| Data utilities | `opd/data/` | Load datasets, format prompts, assemble batches, pad, and pack sequences. |
| Utilities | `opd/utils/` | Config loading, evaluation helpers, port leases, logging, tracing, and post-eval utilities. |

## Process topology

A typical OPD or GRPO run has one coordinator and several worker roles:

![AsyncOPD process topology](architecture.svg)

The diagram shows logical data flow. In the default topology, rollout workers do not directly call the teacher process. They return generated tokens and metadata to the coordinator, and the coordinator submits scoring work to the teacher/reference path. Keeping that routing centralized makes batching, retries, scheduling, and mode-specific behavior easier to control.

How the roles differ by training mode:

| Mode | Rollout | Teacher/reference | Trainer |
| --- | --- | --- | --- |
| OPD | Required | Required for teacher scoring | KL-style OPD trainer |
| GRPO/DAPO | Required | Optional reference when KL beta is nonzero | GRPO/PPO-style trainer |
| SFT | Not used | Not used | SFT trainer only |

## Communication paths

| Path | Mechanism | Payload |
| --- | --- | --- |
| Coordinator -> rollout | multiprocessing queues/control messages | rollout requests, generation settings, shutdown commands |
| Rollout -> coordinator | multiprocessing queues | generated token IDs/text and metadata |
| Coordinator -> teacher/reference | local worker/proxy path; vLLM teacher can run as a local scoring service | prompt context, response tokens, scoring requests |
| Teacher/reference -> coordinator/trainer | queues or direct trainer path, depending on mode | reference logprobs or teacher artifacts |
| Coordinator -> trainer | multiprocessing queues | prepared training batches and control commands |
| Trainer -> rollout | NCCL or configured weight-sync backend | updated student weights |

ZMQ is used in teacher service paths where the teacher is exposed as a local scoring service. Defaults bind teacher services to localhost.

## Run lifecycle

A normal OPD/GRPO run follows this sequence:

1. The CLI loads and validates the YAML config.
2. The coordinator derives role configs and GPU placement.
3. The teacher/reference worker starts if the mode needs one.
4. Rollout workers start and load the student model.
5. The trainer process group starts and loads the trainable student model.
6. The coordinator schedules rollout batches.
7. Teacher/reference scoring or reward computation enriches the generated samples.
8. The trainer consumes batches and updates the student.
9. Fresh weights are synchronized back to rollout workers.
10. Evaluation and checkpoint hooks run according to `eval.*` and `trainer.*` settings.
11. The coordinator drains queues, writes logs/traces, and shuts workers down.

SFT is shorter: load config, start the trainer, run supervised updates, optionally evaluate/checkpoint, then shut down.

## Scheduling modes

### Synchronous

```yaml
pipeline:
  n_step_off:
    step_off: 0
```

Rollout, scoring, training, and weight sync run in sequence. This is the easiest mode to debug and the best starting point for a new backend, dataset, or loss.

### N-step-off

```yaml
pipeline:
  n_step_off:
    step_off: 2
```

Rollout may run up to `N` logical batches ahead of the trainer. This overlaps generation, scoring, and training while keeping policy staleness bounded.

### N-step-off with streaming internals

```yaml
pipeline:
  n_step_off:
    implementation: streaming
    streaming:
      rollout_backend: async_sample
      teacher_transport: coordinator
```

This keeps the n-step-off scheduling model but uses streaming internals for specialized OPD paths. Use the standard n-step-off path first unless a config specifically calls for this implementation.

### Fully async

```yaml
pipeline:
  scheduling_mode: fully_async
  fully_async:
    staleness_threshold: 8
    evict_stale: false
    pause_mode: keep
```

Rollout and trainer proceed independently, and the coordinator filters work according to the staleness policy. Use fully async only after a synchronous or small-step-off run is correct.

## Weight synchronization

The main performance path is NCCL GPU-direct broadcast from trainer to rollout workers:

```yaml
weight_sync:
  backend: nccl
  nccl_timeout_hours: 2
```

## GPU placement

Configs use comma-separated GPU IDs:

```yaml
teacher:
  gpu_ids: '0'
rollout:
  gpu_ids: 1,2,3
trainer:
  gpu_ids: 4,5,6,7
```

Rules of thumb:

- Keep `gpu_ids` and `n_gpus` consistent when editing configs by hand.
- Start with local multiprocessing on one node before trying Ray or multi-node setups.
- Prefer separate GPUs for rollout and trainer roles.
- Colocation is possible for small models when memory settings are deliberately reduced, as in the 4-GPU starter configs.

## Evaluation

Evaluation can run during training, after training, after worker shutdown using all GPUs, or separately through `opd-eval` / `python -m opd.cli.eval`.

| Eval style | Config/command | Notes |
| --- | --- | --- |
| Inline | `eval.mode: [inline]` | Runs while training is active. |
| Post | `eval.mode: [post]` | Runs after checkpoints are available. |
| Post all-GPU | `eval.mode: [post_allgpu]` | Reuses rollout/trainer GPUs after worker shutdown for faster data-parallel eval. |
| Standalone | `python -m opd.cli.eval ...` | Evaluates an existing model/checkpoint or score-only artifact. |
| Perplexity | `eval.mode: [perplexity]` | SFT-style validation path. |

## Trust boundaries

Defaults are conservative where practical:

- `trust_remote_code` defaults to false for model, teacher, rollout, and trainer loading.
- Pickle-backed SFT teacher logits require `data.allow_pickle_teacher_logits: true` and should only be used with trusted datasets.
- Generated-code scoring requires `eval.allow_unsafe_code_execution: true` or `--allow-unsafe-code-execution`.
- Teacher services bind to `127.0.0.1` by default.
- Ray, multi-node setups, and external bind addresses require trusted networking.

## Output layout

A run writes under `results/`:

```text
results/<experiment>/<config>/
  log.jsonl
  run.log
  trace.json
  checkpoints/
  validation_outputs/
```

Use `log.jsonl` for metrics, `run.log` for process output, `trace.json` for Perfetto/Chrome timing, `checkpoints/` for resume/evaluation, and `validation_outputs/` for per-sample inspection.

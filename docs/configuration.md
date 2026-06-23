# Configuration guide

AsyncOPD uses YAML config files. The examples under `configs/examples/` are the best starting point: copy one, change the model/data/GPU settings you need, and run it with `opd-train` or `python -m opd.cli.train`.

This guide focuses on the fields you are most likely to edit. Some research knobs exist for advanced experiments, but you usually do not need them for a first run.

## Example configs

| Config | Use it for | Hardware |
| --- | --- | --- |
| `configs/examples/opd_gsm8k_0.5b_4gpu.yaml` | Small OPD smoke test on GSM8K | 4 GPUs |
| `configs/examples/opd_qwen3_1.7b.yaml` | Qwen3 OPD starter | 8 GPUs |
| `configs/examples/grpo_qwen3_1.7b.yaml` | Qwen3 GRPO reward training | 8 GPUs |
| `configs/examples/grpo_gsm8k_0.5b_8gpu.yaml` | Larger Qwen2.5-0.5B GSM8K GRPO layout | 8 GPUs |
| `configs/examples/grpo_gsm8k_0.5b_4gpu.yaml` | Smaller Qwen2.5-0.5B GSM8K GRPO starter | 4 GPUs |
| `configs/examples/dapo_deepmath_qwen3_4b_8gpu.yaml` | DAPO on DeepMath/AIME | 8 GPUs |
| `configs/examples/opd_deepmath_qwen3_1.7b_pg_mc64_stepoff_8gpu.yaml` | Multi-sample PG-KL OPD with bounded step-off scheduling | 8 GPUs |
| `configs/examples/opd_deepmath_qwen3_1.7b_pg_mc64_async_8gpu.yaml` | Multi-sample PG-KL OPD with fully async scheduling | 8 GPUs |
| `configs/examples/sft_qwen3_1.7b.yaml` | SFT template; replace local parquet paths before running | 2+ GPUs |

Advanced examples live under `configs/examples/advanced/`. Use them only after the nearest starter config works.

## Config shape

Most configs use these top-level sections:

```yaml
deterministic: false
seed: 42

model: {}
teacher: {}
data: {}
rollout: {}
trainer: {}
algorithm: {}
pipeline: {}
eval: {}
weight_sync: {}
logging: {}
```

Required sections depend on the training mode:

| Mode | Required sections | Common optional sections |
| --- | --- | --- |
| OPD | `model`, `teacher`, `data`, `rollout`, `trainer`, `algorithm` | `pipeline`, `eval`, `weight_sync`, `logging` |
| GRPO with `kl_beta > 0` | `model`, `teacher`, `data`, `rollout`, `trainer`, `algorithm` | `pipeline`, `eval`, `weight_sync`, `logging` |
| GRPO with `kl_beta: 0` | `model`, `data`, `rollout`, `trainer`, `algorithm` | `teacher` may be omitted, plus common sections |
| SFT | `model`, `data`, `trainer`, `algorithm` | `eval`, `logging` |

Use the top-level sections shown here for new configs.

## Model

```yaml
model:
  path: Qwen/Qwen3-1.7B
  eos_token_id: null
  trust_remote_code: false
```

| Field | Meaning | Guidance |
| --- | --- | --- |
| `path` | Student/base model path or Hugging Face repo ID | Required. Pin revisions externally if reproducibility matters. |
| `eos_token_id` | Optional explicit EOS token ID | Needed only for specialized losses. |
| `trust_remote_code` | Allow model repository Python code | Defaults false; enable only for trusted, pinned repos. |

## Data

```yaml
data:
  train_files: hf:openai/gsm8k:main
  val_files: hf:openai/gsm8k:main:test
  prompt_key: question
  answer_key: answer
  completion_key: completion
  prompt_template: '{problem}\nPlease reason step by step.'
  max_prompt_length: 512
  max_response_length: 1024
```

| Field | Meaning |
| --- | --- |
| `train_files` | Local parquet path or `hf:org/dataset[:split]` URI. Required except score-only eval flows. |
| `val_files` | Validation dataset path or URI. |
| `prompt_key` | Prompt/problem column. |
| `completion_key` | Completion column for SFT. |
| `answer_key` | Ground-truth answer column for math reward/eval. |
| `solution_key` | Optional worked-solution column for specialized modes. |
| `prompt_template` | Python format-style template used to wrap raw prompts. |
| `prompt_source` | Prompt extraction mode such as `raw` or `last_user_content` for chat-like datasets. |
| `filter_key`, `filter_value` | Optional dataset row filter. |
| `enable_thinking` | Use tokenizer/model chat-template thinking behavior where supported. |
| `teacher_enable_thinking` | Optional teacher-specific thinking override. |
| `max_prompt_length` | Prompt token cap. |
| `max_response_length` | Response token cap. |
| `post_eval_datasets` | Additional post-eval datasets, including code benchmarks. |
| `allow_pickle_teacher_logits` | Defaults false; opt-in only for trusted SFT KL datasets. |

SFT parquet files should contain columns matching `prompt_key` and `completion_key`. OPD/GRPO math examples usually need `prompt_key` and `answer_key`.

For code-training examples, `prompt_source: last_user_content` extracts the last user turn from chat-style prompt records, and `filter_key` / `filter_value` can select subsets such as `ability=code`.

For teacher-generated SFT warm-start data, use `scripts/gen_teacher_solutions.py` to write `prompt` / `completion` parquet files from a trusted teacher model. If you generate `--save-logits` parquet files for SFT-KL, keep them trusted-only and leave `allow_pickle_teacher_logits: false` unless you intentionally need that path.

### Building `data/deepmath_difficulty6/train.parquet`

Some DeepMath examples expect `data/deepmath_difficulty6/train.parquet`. That file is not included in the repository. If you need it, build it from the public Hugging Face dataset:

```bash
python - <<'PY'
from pathlib import Path

from datasets import load_dataset
import pandas as pd

output = Path("data/deepmath_difficulty6/train.parquet")
output.parent.mkdir(parents=True, exist_ok=True)

source = "zwhe99/DeepMath-103K"
dataset = load_dataset(source, split="train")
df = dataset.to_pandas()
df = df[df["difficulty"].astype(float) >= 6].copy()


def first_solution(row):
    for key in ("r1_solution_1", "r1_solution_2", "r1_solution_3"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


topic = df["topic"].astype(str) if "topic" in df.columns else ""
out = pd.DataFrame(
    {
        "problem": df["question"].astype(str),
        "answer": df["final_answer"].astype(str),
        "solution": df.apply(first_solution, axis=1),
        "difficulty": df["difficulty"],
        "topic": topic,
    }
)
out.to_parquet(output, index=False)
print(f"wrote {len(out)} rows to {output}")
PY
```

Use the generated file with matching data keys:

```yaml
data:
  train_files: data/deepmath_difficulty6/train.parquet
  prompt_key: problem
  answer_key: answer
  solution_key: solution
```

`solution_key` is optional for standard OPD/GRPO math runs, but useful for variants that consume worked solutions. Before redistributing generated data, review the upstream dataset license and terms.

## Teacher/reference

```yaml
teacher:
  path: Qwen/Qwen3-30B-A3B-Instruct-2507
  backend: vllm
  gpu_ids: '0'
  n_gpus: 1
  scoring_batch_size: 8
  bind_address: 127.0.0.1
  dtype: auto
  trust_remote_code: false
  vllm:
    tensor_parallel_size: 1
    n_logprobs: 1
    max_model_len: 18688
```

| Field | Meaning | Guidance |
| --- | --- | --- |
| `path` | Teacher/reference model path | Required for OPD and GRPO KL/reference runs. |
| `backend` | `vllm` or `hf` | Use vLLM for throughput; HF is mainly for compatibility/debugging. |
| `gpu_ids`, `n_gpus` | Teacher GPU placement | Keep count consistent with tensor parallel settings. |
| `scoring_batch_size` | Teacher scoring batch size | Lower if memory-bound. |
| `bind_address` | Service bind address | Defaults to localhost. Use external addresses only on trusted networks. |
| `vllm.tensor_parallel_size` | vLLM tensor parallel size | Usually the number of teacher GPUs. |
| `vllm.n_logprobs` | Number of logprobs to request | OPD/GRPO KL paths may need this. |
| `vllm.max_model_len` | Teacher context budget | Must fit prompt plus response. |
| `vllm.gpu_memory_utilization` | vLLM memory fraction | Lower when colocating. |
| `hf.use_torch_compile` | HF backend compile toggle | Only applies to the HF teacher backend. |
| `ray.*` | Ray placement settings | Advanced. Start with local multiprocessing first. |

## Rollout

```yaml
rollout:
  backend: vllm
  gpu_ids: 1,2,3
  n_gpus: 3
  temperature: 1.0
  top_p: 1.0
  top_k: -1
  dtype: auto
  trust_remote_code: false
  vllm:
    max_model_len: 18432
    max_num_seqs: 512
    gpu_memory_utilization: 0.85
```

| Field | Meaning | Guidance |
| --- | --- | --- |
| `backend` | Rollout backend | Public examples use vLLM. |
| `gpu_ids`, `n_gpus` | Rollout GPU placement | Prefer disjoint from trainer GPUs. |
| `temperature`, `top_p`, `top_k` | Sampling controls | GRPO usually uses nonzero temperature. |
| `quantization` | Optional rollout quantization | Advanced; verify model/backend support. |
| `vllm.max_model_len` | Prompt + response context cap | Must fit `max_prompt_length + max_response_length` with margin. |
| `vllm.max_num_seqs` | vLLM concurrency | Lower to reduce memory. |
| `vllm.gpu_memory_utilization` | vLLM memory target | Lower when GPUs are shared. |
| `vllm.tensor_parallel_size` | vLLM tensor parallel size | Use when one rollout model spans multiple GPUs. |
| `pin_cpu_affinity`, `bind_numa_memory` | CPU/NUMA tuning | Advanced throughput tuning. |
| `ray.*` | Ray placement settings | Advanced. Start local first. |

## Trainer

```yaml
trainer:
  backend: fsdp
  gpu_ids: 4,5,6,7
  n_gpus: 4
  dtype: bfloat16
  batch_size: 256
  micro_batch_size: 16
  mini_batch_size: 128
  use_sequence_packing: true
  kl_chunk_size: 1024
  total_steps: 200
  total_epochs: 10
  save_freq: -1
  optim:
    lr: 5e-6
    lr_decay_style: constant
    weight_decay: 0.01
```

| Field | Meaning | Guidance |
| --- | --- | --- |
| `backend` | `fsdp` or `megatron` | FSDP is the recommended starting point. |
| `gpu_ids`, `n_gpus` | Trainer GPU placement | Avoid rollout overlap unless explicitly designed. |
| `dtype` | Training dtype | Public examples use bf16/auto-compatible settings. |
| `batch_size` | Global training batch size | Per logical train step. |
| `micro_batch_size` | Per-device microbatch | Lower when memory-bound. |
| `mini_batch_size` | PPO/GRPO minibatch size | Optional; defaults depend on mode/backend. |
| `use_sequence_packing` | Pack examples to reduce padding | Useful for OPD/SFT; test alignment-sensitive changes carefully. |
| `kl_chunk_size` | Sequence chunk size for KL/logprob computation | Lower to reduce LM-head/log-softmax peak memory. |
| `total_steps`, `total_epochs` | Duration controls | Use `--set trainer.total_steps=1` for smoke runs. |
| `save_freq` | Checkpoint frequency | `-1` disables periodic saves. |
| `resume_from` | Resume checkpoint selector/path | Usually set by `--resume`. |
| `optim.*` | Optimizer/scheduler settings | LR, decay, warmup, weight decay, betas, grad clipping. |
| `lora.*` | Optional LoRA settings | Advanced; verify with the weight-sync path. |
| `megatron.*` | Megatron backend settings | Advanced; prefer FSDP first. |

### Loss/logit chunking (`trainer.kl_chunk_size`)

`trainer.kl_chunk_size` chunks KL/logprob LM-head work over sequence positions. It is useful when trainer memory is dominated by large `[B, S, V]` logits or log-softmax activations.

Common values:

- keep the default `1024` for most runs;
- try `512`, `256`, or `128` if trainer OOM points at LM-head, log-softmax, KL, PPO, or GRPO logprob gathering;
- use `0` only for debugging/equivalence checks, because it means one full-sequence chunk.

See [Loss/logit chunking](loss-chunking.md) for details.

## Algorithm

```yaml
algorithm:
  mode: grpo
  grpo:
    group_size: 5
    clip_eps: 0.2
    kl_beta: 0.001
    kl_type: low_var_kl
    reward_fn: correctness
```

### OPD fields

| Field | Meaning |
| --- | --- |
| `kl_loss_mode` | KL/loss variant, e.g. `forward_kl`, `reverse_kl`, or `policy_gradient_kl`. |
| `pg_clip_eps` | PPO-style clip epsilon for policy-gradient KL modes. |
| `pg_kl_n_total_samples` | Number of Monte Carlo teacher/student samples for multi-sample KL modes. |
| `pg_online_advantage` | Compute/update online advantages for PG-KL variants. |
| `use_decoupled_loss` | Decoupled-loss path used by AReaL-style examples. |
| `behave_imp_weight_cap` | Importance-weight cap for behavior-policy adjustments. |
| `pg_m2po_budget`, `pg_m2po_miniclip_low`, `pg_m2po_miniclip_high` | M2PO dynamic clipping controls. |
| `rollout_student_topk_k` | Student top-k support size for sparse/top-k reverse-KL and THUNLP-style paths. |
| `teacher_artifact_mode` | Advanced teacher-artifact path: `legacy`, `direct`, or `hidden_recompute`. |
| `teacher_hidden_dtype`, `teacher_hidden_semantics`, `teacher_hidden_recompute_materialization` | Hidden-recompute dense KL controls. |

### GRPO/DAPO fields

| Field | Meaning |
| --- | --- |
| `group_size` | Number of sampled responses per prompt. |
| `clip_eps` | PPO-style clip epsilon. |
| `kl_beta` | Reference KL penalty strength; `0.0` disables the reference-teacher requirement. |
| `kl_type` | KL estimator/variant. |
| `loss_agg_mode` | Token/sample aggregation, e.g. token-mean. |
| `norm_adv_by_std` | Normalize advantages by group standard deviation. |
| `filter_groups` | DAPO-style group filtering. |
| `clip_ratio_low`, `clip_ratio_high`, `clip_ratio_c` | Asymmetric/dual clipping controls. |
| `reward_fn` | Reward function name, typically `correctness`. |
| `answer_pattern` | Regex answer extraction override. |

### SFT fields

| Field | Meaning |
| --- | --- |
| `loss_mode` | `ce`, `kl`, or `mixed`. |
| `ce_alpha` | CE weight for mixed SFT loss. |

## Pipeline

```yaml
pipeline:
  scheduling_mode: n_step_off
  deployment: local
  n_step_off:
    step_off: 2
    implementation: classic
  fully_async:
    staleness_threshold: 8
    evict_stale: false
    pause_mode: keep
```

| Field | Meaning |
| --- | --- |
| `scheduling_mode` | `n_step_off`, `fully_async`, or specialized fused mode. |
| `deployment` | `local` by default; Ray is advanced. |
| `n_step_off.step_off` | Number of rollout batches allowed ahead of the trainer. |
| `n_step_off.implementation` | `classic` or advanced `streaming`. |
| `fully_async.staleness_threshold` | Max staleness before filtering/pausing in fully async mode. |
| `fully_async.evict_stale` | Whether to evict stale queued items. |
| `fully_async.pause_mode` | How rollout pauses under pressure. |
| `fused_hybrid_sync.rollout_parallelism` | Fused scheduler rollout layout. |
| `fused_hybrid_sync.weight_update_backend` | In-process rollout weight refresh backend for fused scheduler examples. |
| `fused_hybrid_sync.refresh_policy` | When fused rollout replicas refresh weights. |

## Evaluation

```yaml
eval:
  freq: 40
  mode: [post_allgpu]
  before_train: true
  n_samples: 32
  temperature: 1.0
  checkpoint_policy: all
  allow_unsafe_code_execution: false
```

| Field | Meaning |
| --- | --- |
| `freq` | Evaluation frequency in training steps; `-1` disables periodic eval. |
| `mode` | `inline`, `post`, `post_allgpu`, or `perplexity`. |
| `before_train` | Run eval before updates. |
| `batch_size` | Eval batch size override. |
| `n_samples` | `1` for greedy eval, `>1` for Avg@N-style metrics. |
| `temperature` | Eval generation temperature. |
| `max_response_length` | Eval response cap override. |
| `checkpoint_policy`, `checkpoint_steps` | Which checkpoints post-eval evaluates. |
| `run_primary` | Whether post-eval includes `data.val_files`. |
| `allow_unsafe_code_execution` | Generated-code scoring opt-in; default false. |

## Weight sync

```yaml
weight_sync:
  backend: nccl
  verify_checksum: false
  nccl_timeout_hours: 2
  nccl_socket_ifname: null
```

| Field | Meaning |
| --- | --- |
| `backend` | `nccl` for GPU-direct sync; other paths are debug/compatibility oriented. |
| `verify_checksum` | Optional checksum verification. |
| `nccl_timeout_hours` | NCCL operation timeout. |
| `nccl_socket_ifname` | Network interface hint for NCCL. |
| `ray_collective` | Ray collective path; advanced. |

## Logging

```yaml
logging:
  wandb:
    project: async-opd
    name: my-run
```

Optional logging integrations include ClearML, Weights & Biases, and Aim. Enable only the backend you have installed and configured.

## Command-line overrides

Both train and eval CLIs accept `--set key=value` overrides:

```bash
python -m opd.cli.train --config configs/examples/opd_gsm8k_0.5b_4gpu.yaml \
  --overwrite \
  --set trainer.total_steps=1 eval.freq=-1 eval.before_train=false
```

Use overrides for short smoke runs or local experiments. For long-running configs, copy an example and edit the YAML directly.

## Before a long run

1. Start from an example config.
2. Confirm local paths exist.
3. Confirm `model.path` and `teacher.path` are reachable or cached.
4. Confirm GPU IDs and `n_gpus` counts match.
5. Keep `trust_remote_code` false unless the source is trusted and pinned.
6. Disable eval/checkpointing for a one-step launch test if needed.
7. Run a one-step smoke before a long experiment.
8. Inspect `results/<experiment>/<config>/run.log` and `log.jsonl` before scaling.

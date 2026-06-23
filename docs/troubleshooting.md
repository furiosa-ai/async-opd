# Troubleshooting guide

This guide starts from symptoms. Prefer the smallest reproducible command before changing many config values at once.

## Installation and import issues

### `torch.cuda.is_available()` is false

Check:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('cuda build', torch.version.cuda)
PY
```

Fixes:

- Install a PyTorch wheel compatible with your NVIDIA driver/runtime.
- Avoid mixing conda and pip CUDA packages in one environment unless you know the ABI outcome.
- Confirm `CUDA_VISIBLE_DEVICES` exposes the GPUs you expect.

### vLLM import or model load fails

First checks:

- Python version is 3.12.
- PyTorch and vLLM builds are compatible.
- Model path is reachable or cached.
- The model does not require remote code unless you explicitly trust it.

For CLI/docs-only sanity checks, you do not need the full GPU stack. Use:

```bash
python -m opd.cli.train --help
python -m opd.cli.eval --help
```

## Quickstart and config issues

### `data/train.parquet` not found in SFT

The public SFT config is a template. Replace:

```yaml
data:
  train_files: data/train.parquet
  val_files: data/val.parquet
```

with real parquet files you control.

### Unknown config key

New public configs should use canonical top-level sections such as `model`, `data`, `rollout`, `trainer`, `algorithm`, and `pipeline`. Do not paste old nested `training.actor_rollout_ref.*` snippets into public configs. See [Configuration guide](configuration.md#canonical-vs-compatibility-keys).

### One-step smoke still runs eval

Disable eval in the smoke command:

```text
--set trainer.total_steps=1 trainer.total_epochs=1 eval.freq=-1 eval.before_train=false trainer.save_freq=-1
```

## vLLM memory and generation issues

### OOM while loading rollout or teacher

Common fixes:

- Lower `rollout.vllm.gpu_memory_utilization` or `teacher.vllm.gpu_memory_utilization`.
- Lower `rollout.vllm.max_num_seqs`.
- Lower `rollout.vllm.max_model_len`, `teacher.vllm.max_model_len`, or `data.max_response_length`.
- Reduce `algorithm.grpo.group_size` for GRPO.
- Avoid unintended trainer/rollout GPU overlap.
- Start with the 4-GPU Qwen2.5-0.5B GRPO example before larger Qwen3 examples.

### vLLM multiprocessing or weight-transfer errors

Weight sync paths require compatible vLLM process behavior. Checks:

- Confirm the run is using the expected vLLM version.
- Confirm rollout starts and loads before NCCL weight sync initializes.
- Keep `VLLM_ENABLE_V1_MULTIPROCESSING=0` behavior for rollout weight transfer paths that require it.
- Test with `pipeline.n_step_off.step_off: 0` or a one-step run to isolate launch from scheduling.

## NCCL and distributed training

### Trainer OOM during KL, OPD, or GRPO loss

If the OOM happens during the trainer forward/loss step rather than rollout or
teacher startup, first identify whether memory is dominated by transformer
activations or by the LM head/log-softmax. OPD and GRPO/DAPO loss paths can
avoid materializing full `[B, S, V]` logits by chunking over sequence positions.

Common fixes:

- Lower `trainer.micro_batch_size` when transformer activations dominate.
- Use transformer gradient checkpointing for transformer-block activation
  pressure; use `trainer.kl_chunk_size` for LM-head/log-softmax pressure. They
  are complementary and neither replaces the other.
- Lower `trainer.kl_chunk_size` when the stack trace or memory profile points at
  LM-head, log-softmax, KL, or PPO/GRPO logprob gathering. Try `512`, `256`, or
  `128`; the default is `1024`.
- Use `trainer.kl_chunk_size: 0` only for debugging/equivalence checks; it means
  one full-sequence chunk and is not memory-saving.
- For dense hidden-recompute OPD examples, prefer the example value
  `kl_chunk_size: 128` unless you have enough head/log-softmax memory.
- Remember that SFT CE loss normally uses full logits, so `kl_chunk_size` is not
  the primary SFT OOM lever.

See [Loss/logit chunking](loss-chunking.md) for the detailed explanation and
tuning guidance.

### Hang during process-group initialization

Checks:

- `trainer.gpu_ids`, `rollout.gpu_ids`, and `teacher.gpu_ids` match actual visible devices.
- `n_gpus` equals the number of IDs in each role.
- Trainer rank 0 is not sharing a rollout GPU unless the config was designed for colocation.
- No stale process owns the same GPU or port.
- Local firewall rules are not blocking localhost communication.

Debug path:

1. Run `nvidia-smi` and stop unrelated GPU processes.
2. Run a one-step smoke with eval disabled.
3. Set `pipeline.n_step_off.step_off: 0` if queue/scheduling overlap complicates diagnosis.
4. Lower trainer microbatch size if the hang follows an OOM/restart.

### Weight sync fails after trainer step

Common causes:

- Model architecture mismatch between trainer and rollout.
- LoRA/native weight-sync path mismatch.
- NCCL initialized before rollout/vLLM is ready.
- GPU IDs changed after config load.
- Unsupported vLLM/torch distributed combination.

Use `weight_sync.verify_checksum: true` only as a diagnostic; it can add overhead.

## GPU mapping issues

Public configs use absolute GPU IDs from the process environment. If you set `CUDA_VISIBLE_DEVICES`, the IDs inside the process are remapped.

Example:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m opd.cli.train --config configs/examples/grpo_gsm8k_0.5b_4gpu.yaml --overwrite
```

Inside that process, config GPU `0` means physical GPU `4`. When debugging, either avoid remapping or document it clearly.

## Ray and multi-node caveats

Ray/multi-node mode is advanced for the first public release. Prefer local multiprocessing until the config works.

If using Ray:

- Keep teacher bind addresses private and reviewed.
- Ensure every node has the same code, Python environment, model cache, and dataset access.
- Verify NCCL network interface selection.
- Run single-node first, then expand to multi-node.
- Treat `ray.*` config fields as experimental public surface.

## Tokenizer and padding issues

The pipeline assumes left padding for token alignment. Runtime paths force left padding in key places, but custom data/model code should preserve it.

Symptoms:

- KL loss includes prompt tokens unexpectedly.
- Response mask length does not match generated tokens.
- Packed and padded variants produce very different losses.
- Evaluation extraction sees prompt text as generated text.

Checks:

```bash
python -m pytest tests/test_tokenizer_padding.py tests/test_packing.py -q -o addopts="" -m "not slow"
```

Remove `-m "not slow"` only when you intentionally want the real-tokenizer
checks that may download or use cached HuggingFace model artifacts.

When adding new data transforms, inspect `attention_mask`, response masks, and shifted-label positions.

## SFT dataset issues

### Pickle teacher logits rejected

This is expected public-safe behavior. Pickle can execute code. Enable `data.allow_pickle_teacher_logits: true` only for trusted datasets you control.

### Missing prompt/completion columns

Check parquet schema and config keys:

```yaml
data:
  prompt_key: prompt
  completion_key: completion
```

For standard CE SFT, you do not need teacher-logit columns.

## Evaluation issues

### Eval hangs with `--dp` or `--tp`

Scale gradually:

```bash
python -m opd.cli.eval --config configs/examples/grpo_gsm8k_0.5b_4gpu.yaml --model student --gpus 0 --dp 1 --tp 1
```

Then increase `--dp` or `--tp`, but not both at once.

### Answer matching looks wrong

Check:

- Prompt asks for the expected output format.
- Dataset `answer_key` points to the right column.
- The model output includes `\boxed{...}` or `#### ...` when expected.
- `algorithm.reward.answer_pattern` or `algorithm.grpo.answer_pattern` matches the dataset format.

## Code evaluation safety

Inline fixture execution is disabled by default because it runs generated Python on the host. For trusted artifacts only:

```bash
python -m opd.cli.eval --score-only tmp/trusted-code-generations.json --allow-unsafe-code-execution
```

Do not enable this for untrusted generations.

## Checkpoint and resume issues

### Resume starts from the wrong state

Checks:

- Use `--resume` with the same config shape and model architecture.
- Confirm checkpoints exist under the expected `results/<experiment>/<config>/checkpoints/` directory.
- Do not combine `--resume` and `--overwrite`.
- Confirm optimizer state was saved if you need optimizer resume.

### Post-eval cannot find checkpoints

Checks:

- `trainer.save_freq` allowed checkpoint creation.
- `eval.checkpoint_policy` selects an existing step.
- The run directory matches the config path-derived output directory.

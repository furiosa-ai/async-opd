# Loss and logit chunking

AsyncOPD uses loss/logit chunking to keep trainer memory bounded when OPD or
GRPO/DAPO losses need logprobs from large-vocabulary causal language models.

The controlling config is:

```yaml
trainer:
  kl_chunk_size: 1024
```

The default is `1024`. Lower values reduce peak LM-head/log-softmax memory.

## What problem this solves

For hidden states `[B, S, H]`, a standard causal LM head produces logits
`[B, S, V]`, where:

- `B` = batch size
- `S` = sequence length
- `H` = hidden size
- `V` = vocabulary size

For long responses and large vocabularies, `[B, S, V]` can dominate trainer
memory. A naive implementation may also retain a full log-softmax or softmax
activation for backward.

Chunking splits the sequence dimension into spans of size `kl_chunk_size`:

```text
for s0:s1 in sequence_chunks:
  logits_chunk = lm_head(hidden[:, s0:s1, :])      # [B, C, V]
  logprobs_needed = log_softmax(logits_chunk) ...  # reduce/gather
  discard dense chunk
```

This changes the head/log-softmax peak from roughly:

```text
O(B * S * V)
```

to:

```text
O(B * kl_chunk_size * V)
```

The transformer body still has its own activation memory. If those activations
dominate, lower `trainer.micro_batch_size` or context length instead.

## Interaction with gradient checkpointing

Loss/logit chunking does **not** require transformer gradient checkpointing to be
useful. They save memory in different parts of the training step:

| Technique | Main memory target |
| --- | --- |
| Transformer gradient checkpointing | Retained transformer-block activations. |
| Loss/logit chunking | LM-head, log-softmax, KL, PPO/GRPO logprob-gather memory. |

They are often enabled together for long-context OPD/GRPO runs because both
memory regions can be large. If only gradient checkpointing is enabled, the run
can still OOM when the final LM head or log-softmax materializes large
`[B, S, V]` tensors. If only loss/logit chunking is enabled, the run can still
OOM inside transformer blocks.

The chunked LM-head path performs its own head-side recomputation during
backward: it saves hidden states, weights, and gather indices, then recomputes
per-chunk logits and softmax terms as gradients flow. That recomputation is
independent of transformer gradient checkpointing.

## Why special logprob helpers are needed

Chunking only delivers the full memory win if the trainer avoids creating full
logits before the loss. This path is memory-saving:

```text
hidden -> chunked lm_head/log_softmax/gather -> selected logprobs or reduced loss
```

This path has already paid most of the memory cost:

```text
hidden -> lm_head -> logits [B,S,V] -> chunked loss
```

If full logits already exist, chunking can still reduce log-softmax/gather
activation memory, but it no longer avoids the full `[B, S, V]` logits tensor.
That is why the FSDP trainer patches non-SFT model forwards so loss calls can
request gathered logprobs directly instead of asking the model to return logits.

Mode-specific logprob needs differ:

| Objective | What the chunked helper gathers |
| --- | --- |
| OPD top-k KL | `lm_head + log_softmax + gather` at `K` teacher support IDs per position. |
| OPD policy-gradient KL | `lm_head + log_softmax + gather` at one generated next-token ID per position. |
| GRPO/DAPO | Same `K=1` next-token logprob gather, then PPO/GRPO clipping uses those logprobs. |

This is why OPD/GRPO use custom chunked helpers: they may need teacher top-k
support, rollout support, multi-sample candidates, or dense hidden-recompute
reductions rather than only one label per token.

## What is chunked

Chunking is over **sequence positions**, not batch or vocabulary.

The code still computes an exact full-vocabulary normalization for each sequence
chunk, because exact log-softmax requires the full vocabulary denominator. It
just computes that denominator for `[B, C, V]` chunks instead of `[B, S, V]`.

Megatron tensor parallelism is different: Megatron splits the vocabulary across
tensor-parallel ranks and uses vocab-parallel KL/logit paths. The FSDP chunked
LM-head path described here is the recommended public path for most users.

## Implementation map

Core implementations live in `opd/loss/kl.py`:

| Function | Purpose |
| --- | --- |
| `chunked_lm_head_gather` | Fuses LM head, log-softmax, and gather without materializing full `[B, S, V]` logits. |
| `chunked_log_softmax_gather` | Used when full logits already exist; chunks log-softmax/gather to avoid retaining a full softmax activation, but cannot avoid the logits tensor that was already produced. |
| `chunked_dense_kl_from_hidden` | Dense hidden-recompute path; chunks student/teacher LM-head projections and reduces KL per chunk. |

## Supported loss paths

The chunked path is used by the FSDP trainer for non-SFT loss modes, including:

- OPD sparse/top-k modes: `forward_kl`, `reverse_kl`,
  `reverse_kl_rollout_student_topk`
- policy-gradient OPD: `policy_gradient_kl`
- multi-sample OPD modes
- GRPO/DAPO PPO-style logprob gathering
- dense hidden-recompute OPD via `chunked_dense_kl_from_hidden`

Notes:

- SFT does not use this chunked logprob path; tune `trainer.micro_batch_size`
  first for SFT memory issues.
- For GRPO/DAPO, chunking reduces trainer-side logprob/loss memory. It does not
  reduce rollout/vLLM generation memory.

## Tuning `trainer.kl_chunk_size`

Use the default unless trainer OOM points at the LM head, log-softmax, KL, or
PPO/GRPO logprob gathering.

Common values:

| Value | Use |
| --- | --- |
| `1024` | Default; good starting point for most OPD/GRPO runs. |
| `512` | First reduction when head/log-softmax memory is too high. |
| `256` | More conservative long-context setting. |
| `128` | Common for dense hidden-recompute examples. |
| `0` | Single full-sequence chunk; useful only for debugging/equivalence checks. |

Lower values reduce peak head/log-softmax memory but can add more LM-head
recomputation overhead. If memory is dominated by transformer activations,
reducing `kl_chunk_size` may not fix the OOM.

## Debugging symptoms

Try lowering `trainer.kl_chunk_size` when:

- OOM happens during trainer forward/loss rather than rollout or teacher load;
- stack traces mention LM head, logits, log-softmax, KL, PPO, GRPO, or logprob
  gathering;
- OOM appears only at long response lengths or larger teacher support sizes
  (`teacher.vllm.n_logprobs`);
- dense hidden-recompute OPD OOMs during trainer-side teacher/student head
  recompute.

Prefer lowering `trainer.micro_batch_size` when:

- OOM happens inside transformer blocks;
- gradient checkpointing or attention activations dominate;
- SFT CE training OOMs.

## Correctness checks

The chunked path is intended to be numerically equivalent to the full-logits
reference path up to floating-point tolerance. Public checks include KL and
packing coverage in `tests/test_kl_loss.py` and `tests/test_packing.py`.

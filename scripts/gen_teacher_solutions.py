#!/usr/bin/env python3
"""Generate teacher solutions for SFT warm-start/off-policy distillation data.

Uses vLLM to generate step-by-step solutions from the teacher model
on the same data distribution that OPD trains on. Supports data-parallel
generation across multiple GPUs.

Generated parquet files contain prompt/completion pairs suitable for SFT. With
``--save-logits`` they also contain pickle-serialized teacher top-k tensors for
trusted legacy SFT-KL experiments; do not load those columns from untrusted data.

Usage:
    python scripts/gen_teacher_solutions.py \
        --teacher Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --data data/deepmath_difficulty6/train.parquet \
        --n 20000 --gpus 0,1,2,3 --dp 4
"""
import argparse
import os
import pickle
import pandas as pd
import numpy as np
import torch
from multiprocessing import Process, Queue


def extract_topk_logprobs(output, k):
    """Extract top-k logprobs from vLLM CompletionOutput."""
    logprobs_list = output.outputs[0].logprobs  # list of dicts per token
    if logprobs_list is None:
        return None, None
    n_tokens = len(logprobs_list)
    topk_logps = torch.zeros(n_tokens, k, dtype=torch.float32)
    topk_indices = torch.zeros(n_tokens, k, dtype=torch.int32)
    for t, token_logprobs in enumerate(logprobs_list):
        # token_logprobs is a dict: {token_id: Logprob(logprob=..., ...)}
        sorted_items = sorted(token_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)[:k]
        for j, (tok_id, lp) in enumerate(sorted_items):
            topk_logps[t, j] = lp.logprob
            topk_indices[t, j] = tok_id
    return topk_logps, topk_indices


def worker_fn(worker_id, gpu_id, model, prompts, sampling_params_dict,
              max_model_len, result_queue, save_logits=False, n_logprobs=10,
              trust_remote_code=False, gpu_memory_utilization=0.90):
    """Generate solutions on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
    )

    if save_logits:
        sampling_params_dict = dict(sampling_params_dict, logprobs=n_logprobs)
    sp = SamplingParams(**sampling_params_dict)

    print(f"  [Worker {worker_id}] Generating {len(prompts)} solutions on GPU {gpu_id}...",
          flush=True)

    outputs = llm.generate(prompts, sp)
    results = []
    logits_data = []  # list of (topk_logps_bytes, topk_indices_bytes) or None
    for out in outputs:
        text = out.outputs[0].text.strip()
        if "<think>" in text:
            think_end = text.rfind("</think>")
            if think_end >= 0:
                text = text[think_end + len("</think>"):].strip()
        results.append(text)
        if save_logits:
            logps, indices = extract_topk_logprobs(out, n_logprobs)
            logits_data.append((pickle.dumps(logps), pickle.dumps(indices)))

    print(f"  [Worker {worker_id}] Done: {len(results)} solutions", flush=True)
    result_queue.put((worker_id, results, logits_data if save_logits else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--data", default="data/deepmath_difficulty6/train.parquet")
    parser.add_argument("--output-dir", default="data/sft_teacher_solutions")
    parser.add_argument("--prompt-key", default="problem")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument(
        "--prompt-template",
        default="{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    )
    parser.add_argument("--n", type=int, default=20000, help="Number of problems to solve")
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU IDs")
    parser.add_argument("--dp", type=int, default=4, help="Data parallel workers")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy (deterministic solutions)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Unsafe opt-in for trusted, pinned teacher repositories only.",
    )
    parser.add_argument("--save-logits", action="store_true", default=False,
                        help="Save teacher top-k logprobs alongside completions")
    parser.add_argument("--n-logprobs", type=int, default=10,
                        help="Number of top-k logprobs to save per token position")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    assert len(gpu_ids) >= args.dp, f"Need at least {args.dp} GPUs, got {len(gpu_ids)}"

    # Load problems
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} problems from {args.data}")
    if args.prompt_key not in df.columns:
        raise SystemExit(f"prompt key {args.prompt_key!r} not found in {args.data}")

    # Sample
    n_total = min(args.n + args.n_val, len(df))
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(df), size=n_total, replace=False)
    df = df.iloc[indices].reset_index(drop=True)
    print(f"Sampled {n_total} problems")

    # Format prompts
    prompt_template = args.prompt_template

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher,
        trust_remote_code=args.trust_remote_code,
    )

    prompts = []
    for _, row in df.iterrows():
        row_values = row.to_dict()
        row_values.setdefault("problem", row[args.prompt_key])
        text = prompt_template.format(**row_values)
        messages = [{"role": "user", "content": text}]
        if not args.enable_thinking:
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant. /no_think"})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    print(f"Formatted {len(prompts)} prompts")

    # Split prompts across workers
    max_model_len = args.max_tokens + 2048
    sampling_params_dict = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": 1.0 if args.temperature == 0 else 0.95,
    }

    result_queue = Queue()
    processes = []
    per_worker = len(prompts) // args.dp

    print(f"Launching {args.dp} workers on GPUs {gpu_ids[:args.dp]}...")

    for i in range(args.dp):
        start = i * per_worker
        end = start + per_worker if i < args.dp - 1 else len(prompts)
        worker_prompts = prompts[start:end]

        p = Process(
            target=worker_fn,
            args=(i, gpu_ids[i], args.teacher, worker_prompts,
                  sampling_params_dict, max_model_len, result_queue,
                  args.save_logits, args.n_logprobs,
                  args.trust_remote_code, args.gpu_memory_utilization),
            daemon=False,
        )
        p.start()
        processes.append(p)

    # Collect results in order
    worker_results = {}
    worker_logits = {}
    for _ in range(args.dp):
        worker_id, results, logits_data = result_queue.get()
        worker_results[worker_id] = results
        worker_logits[worker_id] = logits_data

    for p in processes:
        p.join()

    # Merge in order
    all_outputs = []
    all_logits = []
    for i in range(args.dp):
        all_outputs.extend(worker_results[i])
        if worker_logits[i] is not None:
            all_logits.extend(worker_logits[i])

    print(f"\nTotal generated: {len(all_outputs)}")

    # Build dataset
    records = []
    n_empty = 0
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= len(all_outputs):
            break
        solution = all_outputs[idx]
        if not solution or len(solution) < 10:
            n_empty += 1
            continue
        row_values = row.to_dict()
        row_values.setdefault("problem", row[args.prompt_key])
        record = {
            "prompt": prompt_template.format(**row_values),
            "completion": solution,
            "answer": str(row.get(args.answer_key, "")),
        }
        if args.save_logits and idx < len(all_logits):
            logps_bytes, indices_bytes = all_logits[idx]
            record["teacher_topk_logps"] = logps_bytes
            record["teacher_topk_indices"] = indices_bytes
        records.append(record)

    print(f"Valid solutions: {len(records)} ({n_empty} empty/short dropped)")

    result = pd.DataFrame(records)

    # Split train/val
    n_train = len(result) - args.n_val
    train_df = result.iloc[:n_train]
    val_df = result.iloc[n_train:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Val:   {val_path} ({len(val_df)} samples)")

    sol_lens = result["completion"].str.len()
    print(f"\nSolution length stats:")
    print(f"  mean={sol_lens.mean():.0f}  median={sol_lens.median():.0f}  "
          f"p95={sol_lens.quantile(0.95):.0f}")


if __name__ == "__main__":
    main()

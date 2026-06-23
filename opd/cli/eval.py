#!/usr/bin/env python3
"""Evaluate baseline performance of teacher and/or student models.

Loads model configs from a training YAML and runs Avg@N or greedy eval
on the validation set, without any training or weight sync.

Usage:
    # Student on 1 GPU
    python -m opd.cli.eval --config configs/examples/opd_qwen3_1.7b.yaml --model student --gpus 0

    # Teacher on 2 GPUs with TP=2
    python -m opd.cli.eval --config configs/examples/opd_qwen3_1.7b.yaml --model teacher --gpus 0,1 --tp 2

    # Student with DP=4 (4 GPUs, TP=1 each, split prompts across 4 workers)
    python -m opd.cli.eval --config configs/examples/opd_qwen3_1.7b.yaml --model student --gpus 0,1,2,3 --dp 4

    # Override vLLM params from CLI
    python -m opd.cli.eval --config ... --model teacher --gpus 0,1 --tp 2 --max-num-seqs 128 --gpu-mem 0.9

    # Watch mode: wait for checkpoints to appear and eval as they arrive (pipelined with rsync)
    python -m opd.cli.eval --watch --config configs/examples/opd_qwen3_1.7b.yaml \
        --model student --gpus 0,1,2,3 --dp 4 --watch-timeout 60
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import traceback
from contextlib import ExitStack, contextmanager

from opd.utils.eval import (
    extract_answer as _extract_answer,
    answers_match as _answers_match,
    score_problems as _score_problems,
    should_try_full_response_match as _should_try_full_response_match,
    should_try_full_response_candidate as _should_try_full_response_candidate,
)
from opd.utils.net import kill_tree, leased_port


def _load_opd_config(config_path, overrides=None):
    """Load OPDConfig lazily so ``--help`` works without runtime deps."""
    from opd.utils.config import OPDConfig

    return OPDConfig.from_yaml(config_path, overrides=overrides)


def resolve_trust_remote_code(value=None, *, context="model loading"):
    """Resolve trust_remote_code lazily so CLI help avoids PyYAML imports."""
    from opd.utils.config import resolve_trust_remote_code as _resolve_trust_remote_code

    return _resolve_trust_remote_code(value, context=context)


@contextmanager
def _temporary_env(**updates):
    previous = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _ground_truth_for_matching(gt_raw, answer_pattern=None):
    """Choose the ground-truth string used by answer matching.

    For symbolic eval datasets (MATH-500/HMMT), the raw answer is already the
    canonical expression.  Running the normal answer extractor over a bare
    expression such as ``9\\sqrt{15}`` falls back to the last number (``15``),
    which then forces every sample through expensive full-response parsing.
    Preserve raw symbolic answers and let ``answers_match`` compare the
    extracted prediction against the expression directly.
    """
    if answer_pattern is None and _should_try_full_response_match(gt_raw):
        return gt_raw
    return _extract_answer(gt_raw, pattern=answer_pattern) or gt_raw


def _dp_worker(worker_id, gpu_ids_str, model_path, llm_kwargs, max_model_len,
               prompt_token_ids_list, sampling_params_dict, result_queue,
               tokenizer_path=None, vllm_port=None, master_port=None):
    """Worker process for data-parallel eval. Loads vLLM, generates, decodes, returns texts."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    # Only disable multiprocessing for TP=1; TP>1 needs it for worker spawning
    if llm_kwargs.get("tensor_parallel_size", 1) <= 1:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Ports pre-allocated by parent to avoid TOCTOU races between workers.
    os.environ["VLLM_PORT"] = str(vllm_port)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    bin_dir = os.path.dirname(sys.executable)
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + ":" + os.environ.get("PATH", "")

    llm = None
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len
        tok_path = tokenizer_path or model_path
        llm = LLM(model_path, tokenizer=tok_path, **llm_kwargs)
        trust_remote_code = resolve_trust_remote_code(
            llm_kwargs.get("trust_remote_code"),
            context="eval worker tokenizer loading",
        )
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=trust_remote_code)

        prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids_list]
        sp = SamplingParams(**sampling_params_dict)

        print(f"  [Worker {worker_id}] Generating for {len(prompts)} problems on GPUs {gpu_ids_str}...",
              flush=True)
        outputs = llm.generate(prompts=prompts, sampling_params=sp)

        # Decode and return response texts per problem
        results = []
        for out in outputs:
            sample_texts = []
            for sample in out.outputs:
                resp_text = tokenizer.decode(list(sample.token_ids), skip_special_tokens=True)
                sample_texts.append(resp_text)
            results.append(sample_texts)

        result_queue.put(("ok", worker_id, results))
    except Exception:
        result_queue.put(("error", worker_id, traceback.format_exc()))
        raise
    finally:
        # Explicitly tear down vLLM + torch.distributed to release TCPStore port.
        # Without this, lingering daemons cause EADDRINUSE in subsequent eval steps.
        if llm is not None:
            del llm
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def _stable_prompt_id(*parts):
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _sampling_params_dict(n_samples, temperature, max_response_length, top_p=1.0):
    if n_samples > 1:
        return dict(temperature=temperature, top_p=top_p,
                    max_tokens=max_response_length, n=n_samples, detokenize=False)
    return dict(temperature=0, top_p=top_p,
                max_tokens=max_response_length, detokenize=False)


def _encode_prompt_texts(tokenizer, prompt_texts, max_prompt_length):
    all_prompt_ids = []
    for text in prompt_texts:
        encoded = tokenizer(
            text,
            max_length=max_prompt_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids = encoded["input_ids"][0][encoded["attention_mask"][0].bool()].tolist()
        all_prompt_ids.append(ids)
    return all_prompt_ids


def _generate_texts_from_prompt_ids(model_path, gpu_ids, tp_size, dp_size, gpu_mem,
                                    max_model_len, max_num_seqs, prompt_ids,
                                    sampling_params_dict, tokenizer,
                                    tokenizer_path=None, dtype="auto",
                                    enforce_eager=False,
                                    trust_remote_code=None):
    """Generate response texts for prompt token IDs via vLLM."""
    tok_path = tokenizer_path or model_path
    llm_kwargs = dict(
        tensor_parallel_size=tp_size,
        trust_remote_code=resolve_trust_remote_code(
            trust_remote_code,
            context="eval vLLM model loading",
        ),
        gpu_memory_utilization=gpu_mem,
        max_num_seqs=max_num_seqs,
        enable_chunked_prefill=False,
        enforce_eager=enforce_eager,
        dtype=dtype,
    )

    if dp_size > 1:
        gpu_list = gpu_ids.split(",")
        gpus_per_worker = tp_size
        all_responses = [None] * len(prompt_ids)
        worker_prompts = [[] for _ in range(dp_size)]
        worker_indices = [[] for _ in range(dp_size)]
        for i, ids in enumerate(prompt_ids):
            w = i % dp_size
            worker_prompts[w].append(ids)
            worker_indices[w].append(i)

        result_queue = mp.Queue()
        processes = []
        with ExitStack() as port_stack:
            worker_ports = []
            for w in range(dp_size):
                vllm_lease = port_stack.enter_context(
                    leased_port(f"eval.dp_worker[{w}].vllm")
                )
                master_lease = port_stack.enter_context(
                    leased_port(f"eval.dp_worker[{w}].master")
                )
                worker_ports.append((vllm_lease.port, master_lease.port))

            for w in range(dp_size):
                w_gpus = gpu_list[w * gpus_per_worker:(w + 1) * gpus_per_worker]
                w_gpu_str = ",".join(w_gpus)
                vllm_port, master_port = worker_ports[w]
                proc = mp.Process(
                    target=_dp_worker,
                    args=(w, w_gpu_str, model_path, llm_kwargs, max_model_len,
                          worker_prompts[w], sampling_params_dict, result_queue,
                          tok_path, vllm_port, master_port),
                    daemon=False,
                )
                proc.start()
                processes.append(proc)

            collected = 0
            for _ in range(dp_size):
                try:
                    msg = result_queue.get(timeout=7200)
                except Exception:
                    alive = [i for i, proc in enumerate(processes) if proc.is_alive()]
                    dead = [i for i, proc in enumerate(processes) if not proc.is_alive()]
                    print(f"[eval] Timeout collecting results. Alive workers: {alive}, "
                          f"Dead workers: {dead}. Collected {collected}/{dp_size}.",
                          flush=True)
                    break
                status, worker_id, payload = msg
                if status == "error":
                    print(f"[eval] Worker {worker_id} failed:\n{payload}", flush=True)
                    break
                collected += 1
                for local_idx, sample_texts in enumerate(payload):
                    global_idx = worker_indices[worker_id][local_idx]
                    all_responses[global_idx] = sample_texts

            for proc in processes:
                proc.join(timeout=30)
                if proc.is_alive():
                    kill_tree(proc.pid)
                    proc.join()

        if collected < dp_size:
            raise RuntimeError(f"Only collected {collected}/{dp_size} worker results")
        return all_responses

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    if tp_size <= 1:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    bin_dir = os.path.dirname(sys.executable)
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + ":" + os.environ.get("PATH", "")
    with ExitStack() as port_stack:
        if "VLLM_PORT" not in os.environ:
            vllm_port = str(port_stack.enter_context(leased_port("eval.single.vllm")).port)
        else:
            vllm_port = os.environ["VLLM_PORT"]
        if "MASTER_PORT" not in os.environ:
            master_port = str(
                port_stack.enter_context(leased_port("eval.single.master")).port
            )
        else:
            master_port = os.environ["MASTER_PORT"]
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

        with _temporary_env(
            VLLM_PORT=vllm_port,
            MASTER_PORT=master_port,
            MASTER_ADDR=master_addr,
        ):
            llm = None
            try:
                from vllm import LLM, SamplingParams

                if max_model_len:
                    llm_kwargs["max_model_len"] = max_model_len

                print(f"Loading model: {model_path} on GPUs {gpu_ids} (TP={tp_size})", flush=True)
                llm = LLM(model_path, tokenizer=tok_path, **llm_kwargs)

                prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
                sp = SamplingParams(**sampling_params_dict)

                n_samples = sampling_params_dict.get("n", 1)
                print(f"Generating {n_samples} sample(s) per problem for {len(prompts)} problems...",
                      flush=True)
                outputs = llm.generate(prompts=prompts, sampling_params=sp)

                all_responses = []
                for out in outputs:
                    sample_texts = []
                    for sample in out.outputs:
                        resp_text = tokenizer.decode(list(sample.token_ids), skip_special_tokens=True)
                        sample_texts.append(resp_text)
                    all_responses.append(sample_texts)
                return all_responses
            finally:
                if llm is not None:
                    del llm
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.destroy_process_group()
                except Exception:
                    pass


def generate_responses(model_path, gpu_ids, tp_size, dp_size, gpu_mem, max_model_len,
                       max_num_seqs, dataset, prompt_key, answer_key,
                       prompt_template, max_prompt_length, max_response_length,
                       n_samples, temperature, dtype="auto", enforce_eager=False,
                       enable_thinking=None, tokenizer_path=None, eval_type="math",
                       benchmark=None, release=None, date_start=None, date_end=None,
                       top_p=1.0, trust_remote_code=None):
    """Generate a reusable math/code evaluation artifact."""
    from transformers import AutoTokenizer
    from opd.data.prompt import format_prompt

    tok_path = tokenizer_path or model_path
    trust_remote_code = resolve_trust_remote_code(
        trust_remote_code,
        context="eval tokenizer loading",
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    date_window = {"start": date_start, "end": date_end}
    eval_type = eval_type or "math"
    if eval_type == "math":
        from opd.data.prompt import ValDataset
        val_dataset = ValDataset(
            dataset, tokenizer, max_prompt_length,
            prompt_key=prompt_key, answer_key=answer_key,
            prompt_template=prompt_template, enable_thinking=enable_thinking,
        )
        prompt_texts = [
            format_prompt(raw, tokenizer, prompt_template, enable_thinking=enable_thinking)
            for raw in val_dataset.raw_prompts
        ]
        prompt_metadata = []
        for i, (prompt_text, gt) in enumerate(zip(prompt_texts, val_dataset.ground_truths)):
            prompt_metadata.append({
                "problem_id": i,
                "prompt_id": _stable_prompt_id(dataset, i, prompt_text),
                "dataset": dataset,
                "ground_truth": gt,
                "prompt": prompt_text,
            })
        artifact_dataset = dataset
    elif eval_type == "code":
        from opd.utils.code_eval import load_code_benchmark_prompts, normalize_benchmark
        benchmark = normalize_benchmark(benchmark)
        loaded_metadata = load_code_benchmark_prompts(
            benchmark, release=release, date_start=date_start, date_end=date_end,
        )
        prompt_texts = [
            format_prompt(meta["prompt"], tokenizer, prompt_template,
                          enable_thinking=enable_thinking)
            for meta in loaded_metadata
        ]
        prompt_metadata = []
        for i, (meta, prompt_text) in enumerate(zip(loaded_metadata, prompt_texts)):
            row = dict(meta)
            row.setdefault("problem_id", i)
            row.setdefault("dataset", benchmark)
            row.setdefault("benchmark", benchmark)
            row.setdefault("prompt_id", _stable_prompt_id(benchmark, i, prompt_text))
            row["prompt"] = prompt_text
            prompt_metadata.append(row)
        artifact_dataset = benchmark
    else:
        raise ValueError("eval_type must be 'math' or 'code'")

    prompt_ids = _encode_prompt_texts(tokenizer, prompt_texts, max_prompt_length)
    sp_dict = _sampling_params_dict(n_samples, temperature, max_response_length, top_p=top_p)
    responses = _generate_texts_from_prompt_ids(
        model_path, gpu_ids, tp_size, dp_size, gpu_mem, max_model_len, max_num_seqs,
        prompt_ids, sp_dict, tokenizer, tokenizer_path=tok_path, dtype=dtype,
        enforce_eager=enforce_eager, trust_remote_code=trust_remote_code,
    )

    return {
        "schema_version": 1,
        "eval_type": eval_type,
        "dataset": artifact_dataset,
        "benchmark": benchmark if eval_type == "code" else None,
        "release": release,
        "date_window": date_window,
        "generation": {
            "n_samples": n_samples,
            "temperature": temperature,
            "top_p": top_p,
            "max_response_length": max_response_length,
        },
        "prompt_metadata": prompt_metadata,
        "responses": responses,
    }


def score_math(generation_result, output_path=None, answer_pattern=None):
    """Score a math generation artifact using the legacy answer matcher."""
    if generation_result.get("eval_type") != "math":
        raise ValueError("score_math requires a math generation artifact")
    all_responses = generation_result["responses"]
    prompt_metadata = generation_result["prompt_metadata"]
    n_samples = generation_result.get("generation", {}).get("n_samples") or max(
        (len(r) for r in all_responses), default=1,
    )

    out_file = open(output_path, "w") if output_path else None
    problem_results = []
    try:
        for i, (meta, sample_texts) in enumerate(zip(prompt_metadata, all_responses)):
            gt_raw = str(meta.get("ground_truth", "")).strip()
            gt = _ground_truth_for_matching(gt_raw, answer_pattern=answer_pattern)
            n_correct = 0
            for s_idx, resp_text in enumerate(sample_texts):
                predicted = _extract_answer(resp_text, pattern=answer_pattern)
                is_correct = _answers_match(predicted, gt)
                if (
                    not is_correct
                    and answer_pattern is None
                    and _should_try_full_response_candidate(resp_text, gt_raw)
                ):
                    is_correct = _answers_match(resp_text, gt_raw)
                n_correct += int(is_correct)

                if out_file:
                    out_file.write(json.dumps({
                        "problem_id": meta.get("problem_id", i),
                        "sample_idx": s_idx,
                        "ground_truth": gt,
                        "predicted": predicted,
                        "correct": is_correct,
                        "response": resp_text,
                    }) + "\n")

            problem_results.append({"gt": gt, "n_correct": n_correct, "n_total": len(sample_texts)})
            if i < 3:
                print(f"  [Problem {i}] gt={gt!r} pass_rate={n_correct}/{len(sample_texts)}",
                      flush=True)
    finally:
        if out_file:
            out_file.close()

    metrics = _score_problems(problem_results, n_samples)
    if n_samples == 1:
        accuracy = metrics["accuracy"]
        print(f"\nAccuracy: {metrics['correct']}/{metrics['total']} = {accuracy:.2f}%", flush=True)
    else:
        accuracy = metrics[f"avg_at_{n_samples}"]
        print(f"\nAvg@{n_samples}: {accuracy:.2f}%", flush=True)
    metrics.update({
        "eval_type": "math",
        "dataset": generation_result.get("dataset"),
    })
    return metrics


def _write_metrics_jsonl(path, metrics):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(metrics) + "\n")


def score_code(generation_result, benchmark=None, output_dir=None, output_path=None,
               code_workers=1, code_timeout=10.0, generations_path=None,
               allow_unsafe_code_execution=False):
    """Dispatch code scoring to the isolated code-eval utility module."""
    from opd.utils.code_eval import score_code as _score_code
    metrics = _score_code(
        generation_result,
        benchmark=benchmark,
        output_dir=output_dir,
        workers=code_workers,
        timeout=code_timeout,
        generations_path=generations_path,
        allow_unsafe_code_execution=allow_unsafe_code_execution,
    )
    if output_path:
        _write_metrics_jsonl(output_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def evaluate_model(model_path, gpu_ids, tp_size, dp_size, gpu_mem, max_model_len,
                   max_num_seqs, val_files, prompt_key, answer_key,
                   prompt_template, max_prompt_length, max_response_length,
                   n_samples, temperature, dtype="auto", enforce_eager=False,
                   enable_thinking=None, output_path=None, tokenizer_path=None,
                   answer_pattern=None, trust_remote_code=None):
    """Load a vLLM model and evaluate on a math validation set."""
    artifact = generate_responses(
        model_path=model_path,
        gpu_ids=gpu_ids,
        tp_size=tp_size,
        dp_size=dp_size,
        gpu_mem=gpu_mem,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dataset=val_files,
        prompt_key=prompt_key,
        answer_key=answer_key,
        prompt_template=prompt_template,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        n_samples=n_samples,
        temperature=temperature,
        dtype=dtype,
        enforce_eager=enforce_eager,
        enable_thinking=enable_thinking,
        tokenizer_path=tokenizer_path,
        eval_type="math",
        trust_remote_code=trust_remote_code,
    )
    metrics = score_math(artifact, output_path=output_path, answer_pattern=answer_pattern)
    return metrics.get("accuracy", metrics.get(f"avg_at_{n_samples}", 0.0))

def _compute_eval_steps(trainer_cfg):
    """Compute expected eval steps from trainer config."""
    test_freq = trainer_cfg.get("test_freq", 20)
    total_steps = trainer_cfg["total_training_steps"]
    val_before_train = trainer_cfg.get("val_before_train", False)
    steps = []
    if val_before_train:
        steps.append(0)
    for s in range(test_freq, total_steps + 1, test_freq):
        steps.append(s)
    return steps


def _cfg_name(config_path):
    """Extract config name: strip configs/ prefix and .yaml suffix."""
    name = config_path
    for prefix in ("configs/", "configs\\"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.endswith(".yaml"):
        name = name[:-5]
    return name


def watch_main(args):
    """Watch mode: round-robin eval across configs, waiting for checkpoints."""
    import time

    watch_timeout_secs = args.watch_timeout * 60

    # Build per-config state
    configs = []
    for config_path in args.config:
        opd_cfg = _load_opd_config(config_path, overrides=getattr(args, 'set', None))
        import dataclasses
        cfg = dataclasses.asdict(opd_cfg)

        name = _cfg_name(config_path)
        run_dir = os.path.join("results", name)
        eval_jsonl = os.path.join(run_dir, "eval.jsonl")

        # Load completed (step, dataset) pairs from eval.jsonl
        val_files = opd_cfg.data.val_files
        # Normalize to string for hashable set keys (configs may use list or string)
        if isinstance(val_files, list):
            val_files = val_files[0] if len(val_files) == 1 else ",".join(val_files)
        completed_pairs = set()
        if os.path.exists(eval_jsonl):
            with open(eval_jsonl) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "eval":
                            completed_pairs.add((entry.get("step"), entry.get("dataset")))
                    except (json.JSONDecodeError, KeyError):
                        pass

        eval_cfg = cfg["training"].get("trainer", {})  # still needed for _compute_eval_steps
        all_steps = _compute_eval_steps(eval_cfg)
        remaining = [s for s in all_steps if (s, val_files) not in completed_pairs]

        model_path = opd_cfg.model.path
        enable_thinking = opd_cfg.data.enable_thinking
        if args.enable_thinking is not None:
            enable_thinking = args.enable_thinking

        ro = opd_cfg.rollout
        cfg_trust_remote_code = (
            args.trust_remote_code
            if args.trust_remote_code is not None
            else (ro.trust_remote_code if ro else opd_cfg.model.trust_remote_code)
        )
        configs.append({
            "config_path": config_path,
            "internal_cfg": cfg,
            "name": name,
            "run_dir": run_dir,
            "eval_jsonl": eval_jsonl,
            "model_path": args.model_path or model_path,
            "tokenizer_path": args.tokenizer or opd_cfg.data.tokenizer_path or model_path,
            "remaining": remaining,
            "completed": completed_pairs,
            "timed_out": False,
            "last_progress": time.time(),
            # Eval params
            "n_samples": opd_cfg.eval.n_samples if args.eval_n_samples is None else args.eval_n_samples,
            "temperature": opd_cfg.eval.temperature if args.eval_temperature is None else args.eval_temperature,
            "max_prompt_length": opd_cfg.data.max_prompt_length,
            "max_response_length": args.max_response_length or opd_cfg.data.max_response_length,
            "prompt_key": opd_cfg.data.prompt_key,
            "prompt_template": opd_cfg.data.prompt_template,
            "answer_key": opd_cfg.data.answer_key or "auto",
            "val_files": opd_cfg.data.val_files,
            "enable_thinking": enable_thinking,
            "answer_pattern": args.answer_pattern or opd_cfg.algorithm.reward.answer_pattern,
            # vLLM params
            "gpu_mem": args.gpu_mem or (ro.vllm.gpu_memory_utilization if ro else 0.85),
            "max_model_len": args.max_model_len or (ro.vllm.max_model_len if ro else None),
            "max_num_seqs": args.max_num_seqs or (ro.vllm.max_num_seqs if ro else 64),
            "dtype": ro.dtype if ro else "auto",
            "enforce_eager": (args.enforce_eager if args.enforce_eager is not None
                              else (ro.vllm.enforce_eager if ro else False)),
            "trust_remote_code": cfg_trust_remote_code,
        })

    # Create a Logger per config for eval.jsonl + optional W&B/ClearML
    from opd.utils.logger import Logger
    for c in configs:
        c["logger"] = Logger(
            c["eval_jsonl"], config=c["internal_cfg"],
            backends=args.log or [],
            run_name=f"eval/{c['name']}",
            resume=True,
        )

    print(f"[watch] Watching {len(configs)} config(s), timeout={args.watch_timeout}min", flush=True)
    for c in configs:
        print(f"[watch]   {c['name']}: {len(c['remaining'])} steps to eval "
              f"({len(c['completed'])} already done)", flush=True)

    def _check_ready(c, step):
        """Check if checkpoint is ready, convert model.pt → HF if needed."""
        if step == 0:
            return True, c["model_path"]
        step_dir = os.path.join(c["run_dir"], "checkpoints", f"step_{step}")
        hf_dir = os.path.join(step_dir, "hf")
        model_pt = os.path.join(step_dir, "model.pt")
        if os.path.exists(os.path.join(hf_dir, "config.json")):
            return True, hf_dir
        elif os.path.exists(model_pt):
            print(f"[watch] Converting {c['name']} step {step} to HF format...",
                  flush=True)
            # Run conversion in subprocess to avoid CUDA init in main process
            import subprocess
            trust_remote_code = resolve_trust_remote_code(
                c.get("trust_remote_code"),
                context="eval checkpoint conversion",
            )
            subprocess.run([
                sys.executable, "-c",
                f"import torch; from transformers import AutoModelForCausalLM; "
                f"m = AutoModelForCausalLM.from_pretrained('{c['model_path']}', "
                f"torch_dtype='auto', trust_remote_code={trust_remote_code!r}); "
                f"m.load_state_dict(torch.load('{model_pt}', map_location='cpu', "
                f"weights_only=True)); m.save_pretrained('{hf_dir}')"
            ], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            return True, hf_dir
        return False, None

    def _eval_step(c, step, ckpt_path):
        """Run eval for one step, write results. Returns True on success."""
        print(f"\n[watch] Evaluating {c['name']} step {step}", flush=True)
        out_dir = os.path.join(c["run_dir"], "validation_outputs")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"step_{step}.jsonl")
        try:
            acc = evaluate_model(
                model_path=ckpt_path,
                gpu_ids=args.gpus,
                tp_size=args.tp,
                dp_size=args.dp,
                gpu_mem=c["gpu_mem"],
                max_model_len=c["max_model_len"],
                max_num_seqs=c["max_num_seqs"],
                val_files=c["val_files"],
                prompt_key=c["prompt_key"],
                answer_key=c["answer_key"],
                prompt_template=c["prompt_template"],
                max_prompt_length=c["max_prompt_length"],
                max_response_length=c["max_response_length"],
                n_samples=c["n_samples"],
                temperature=c["temperature"],
                dtype=c["dtype"],
                enforce_eager=c["enforce_eager"],
                enable_thinking=c["enable_thinking"],
                output_path=output_path,
                tokenizer_path=c["tokenizer_path"],
                answer_pattern=c["answer_pattern"],
                trust_remote_code=c["trust_remote_code"],
            )
        except Exception as e:
            print(f"[watch] ERROR: {c['name']} step {step} failed: {e}", flush=True)
            return True  # still counts as progress (skip this step)
        c["logger"].log_eval(step, {
            f"avg_at_{c['n_samples']}": round(acc, 4),
            "dataset": c["val_files"],
        })
        print(f"[watch] {c['name']} step {step}: {acc:.2f}%", flush=True)
        return True

    # Sequential mode: finish all steps for each config before moving to next
    try:
        for c in configs:
            print(f"\n[watch] Starting config: {c['name']} "
                  f"({len(c['remaining'])} steps)", flush=True)

            while c["remaining"]:
                step = c["remaining"][0]
                ready, ckpt_path = _check_ready(c, step)

                if not ready:
                    if time.time() - c["last_progress"] > watch_timeout_secs:
                        print(f"[watch] Timeout: {c['name']} step {step} "
                              f"(no checkpoint after {args.watch_timeout}min)", flush=True)
                        c["timed_out"] = True
                        break
                    print(f"[watch] Waiting for {c['name']} step {step}...", flush=True)
                    time.sleep(30)
                    continue

                c["last_progress"] = time.time()
                _eval_step(c, step, ckpt_path)
                c["remaining"].pop(0)
                c["completed"].add((step, c["val_files"]))
                c["last_progress"] = time.time()

    except KeyboardInterrupt:
        print("\n[watch] Interrupted by user", flush=True)

    # Close loggers
    for c in configs:
        c["logger"].close()

    # Summary
    print(f"\n[watch] {'='*60}", flush=True)
    print("[watch] Complete:", flush=True)
    for c in configs:
        n_total = len(c["completed"]) + len(c["remaining"])
        status = "timed out" if c["timed_out"] else ("done" if not c["remaining"] else "interrupted")
        print(f"[watch]   {c['name']}: {len(c['completed'])}/{n_total} steps ({status})", flush=True)



def _eval_dedupe_key(step, eval_type, dataset, benchmark=None, release=None, date_window=None):
    date_window = date_window or {"start": None, "end": None}
    return (
        step,
        eval_type or "math",
        dataset,
        benchmark,
        release,
        date_window.get("start"),
        date_window.get("end"),
    )


def _artifact_path_for_output(output_path):
    root, _ = os.path.splitext(output_path)
    return root + ".generations.json"


def _score_only_main(args):
    from opd.utils.code_eval import load_generation_artifact
    artifact = load_generation_artifact(args.score_only)
    eval_type = args.eval_type or artifact.get("eval_type")
    output_path = None
    if args.output_name:
        out_dir = args.output_dir or os.path.dirname(args.score_only) or "."
        output_path = os.path.join(out_dir, args.output_name)
    if eval_type == "code":
        benchmark = args.benchmark or artifact.get("benchmark")
        return score_code(
            artifact,
            benchmark=benchmark,
            output_dir=args.output_dir or os.path.dirname(args.score_only) or ".",
            output_path=output_path,
            code_workers=args.code_workers,
            code_timeout=args.code_timeout,
            generations_path=args.score_only,
            allow_unsafe_code_execution=args.allow_unsafe_code_execution,
        )
    metrics = score_math(artifact, output_path=output_path, answer_pattern=args.answer_pattern)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model performance")
    parser.add_argument("--config", type=str, action="append", required=False,
                        help="Path to training YAML config (repeat for multiple in --watch mode)")
    parser.add_argument("--model", type=str, required=False, choices=["student", "teacher"],
                        help="Which model to evaluate")
    parser.add_argument("--gpus", type=str, required=False,
                        help="GPU IDs to use (e.g. '0', '0,1', '0,1,2,3')")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (default: 1)")
    parser.add_argument("--dp", type=int, default=1, help="Data parallel size (default: 1)")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Max concurrent sequences per vLLM worker (default: from config)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max model context length (default: from config)")
    parser.add_argument("--gpu-mem", type=float, default=None,
                        help="GPU memory utilization 0-1 (default: from config)")
    parser.add_argument("--enforce-eager", action="store_true", default=None,
                        help="Disable CUDA graphs (default: from config)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path from config")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Override tokenizer (for prompt formatting). Useful when model's "
                             "chat template differs from desired prompt format.")
    parser.add_argument("--trust-remote-code", action="store_true", default=None,
                        help="Allow model/tokenizer loading code from remote model repositories. "
                             "Use only with trusted, pinned model sources.")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", action="store_true", default=None,
                                help="Enable thinking mode (overrides config)")
    thinking_group.add_argument("--no-enable-thinking", action="store_false", dest="enable_thinking",
                                help="Disable thinking mode (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for per-sample outputs (default: results/<config>/baselines/)")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Output filename (default: {model}.jsonl)")
    parser.add_argument("--eval-n-samples", type=int, default=None,
                        help="Override eval_n_samples from config (1=greedy, N=Avg@N)")
    parser.add_argument("--eval-temperature", type=float, default=None,
                        help="Override eval_temperature from config")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Override val_files: evaluate on each dataset separately. "
                             "Supports aliases: AIME25, AMC23, 'HMMT Feb25', "
                             "'HMMT Nov25', MATH-500. "
                             "E.g. 'hf:Maxwell-Jia/AIME_2024' MATH-500")
    parser.add_argument("--max-response-length", type=int, default=None,
                        help="Override max_response_length from config (max tokens to generate)")
    parser.add_argument("--answer-pattern", type=str, default=None,
                        help="Regex with capture group for strict answer extraction. "
                             "E.g. '#### (\\-?[0-9\\.\\,]+)' for GSM8K strict format.")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run results dir — auto-discover checkpoints and eval each. "
                             "Results saved to <run_dir>/eval/<step>/<dataset>.jsonl")
    parser.add_argument("--eval-jsonl", type=str, default=None,
                        help="Path to eval.jsonl — append a summary line per evaluation. "
                             "Useful for remote eval where log.jsonl isn't updated.")
    parser.add_argument("--step", type=int, default=None,
                        help="Training step for this checkpoint (written to eval.jsonl)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode: wait for checkpoints to appear and eval as they arrive")
    parser.add_argument("--watch-timeout", type=int, default=60,
                        help="Minutes to wait for a new checkpoint before giving up (default: 60)")
    parser.add_argument("--log", type=str, nargs="*", default=None,
                        help="Logging backends: wandb, clearml, aim (default: JSONL only)")
    parser.add_argument("--set", nargs="*", default=None,
                        help="Override config values: --set trainer.optim.lr=2e-5 eval.n_samples=1")
    parser.add_argument("--eval-type", type=str, default=None, choices=["math", "code"],
                        help="Evaluation type (default: math; score-only can infer from artifact)")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["humaneval_plus", "mbpp_plus", "lcb_v6"],
                        help="Code benchmark for --eval-type code")
    parser.add_argument("--generate-only", type=str, default=None,
                        help="Write reusable generation artifact to this path and skip scoring")
    parser.add_argument("--score-only", type=str, default=None,
                        help="Score an existing generation artifact without model/config/GPU setup")
    parser.add_argument("--release", type=str, default=None,
                        help="Benchmark release, e.g. release_v6 for LiveCodeBench")
    parser.add_argument("--date-start", type=str, default=None,
                        help="Optional benchmark date window start (YYYY-MM-DD)")
    parser.add_argument("--date-end", type=str, default=None,
                        help="Optional benchmark date window end (YYYY-MM-DD)")
    parser.add_argument("--code-workers", type=int, default=1,
                        help="CPU workers for local code fixture scoring")
    parser.add_argument("--code-timeout", type=float, default=10.0,
                        help="Per-sample timeout in seconds for local code fixture scoring")
    parser.add_argument("--allow-unsafe-code-execution", action="store_true",
                        help="Allow inline local execution of generated Python code during "
                             "fixture-based code scoring. Prefer sandboxed grading for "
                             "untrusted generations.")
    args = parser.parse_args()

    if args.eval_type == "code" and not args.benchmark:
        parser.error("--eval-type code requires --benchmark")

    if args.score_only:
        return _score_only_main(args)

    if args.eval_type is None:
        args.eval_type = "math"

    missing = []
    if not args.config:
        missing.append("--config")
    if not args.model:
        missing.append("--model")
    if not args.gpus:
        missing.append("--gpus")
    if missing:
        parser.error("required unless --score-only: " + ", ".join(missing))

    n_gpus = len(args.gpus.split(","))
    expected = args.tp * args.dp
    if expected > n_gpus:
        parser.error(f"TP={args.tp} x DP={args.dp} = {expected} GPUs needed, but only {n_gpus} provided")

    if args.watch:
        if args.eval_type != "math" or args.generate_only:
            parser.error("--watch supports math scoring only")
        return watch_main(args)

    if len(args.config) > 1:
        parser.error("Multiple --config only supported in --watch mode")

    config_path = args.config[0]
    oc = _load_opd_config(config_path, overrides=getattr(args, 'set', None))
    import dataclasses
    config = dataclasses.asdict(oc)

    n_samples = args.eval_n_samples or oc.eval.n_samples
    temperature = args.eval_temperature or oc.eval.temperature
    max_prompt_length = oc.data.max_prompt_length
    max_response_length = args.max_response_length or oc.eval.max_response_length or oc.data.max_response_length
    prompt_key = oc.data.prompt_key
    prompt_template = oc.data.prompt_template
    answer_key = oc.data.answer_key or "auto"
    val_files = oc.data.val_files
    enable_thinking = oc.data.enable_thinking
    config_answer_pattern = oc.algorithm.reward.answer_pattern
    answer_pattern = args.answer_pattern or config_answer_pattern

    if args.eval_type == "math" and not val_files and not args.datasets:
        parser.error("No val_files in config and no --datasets specified")

    # Get model path from config
    if args.model == "student":
        model_path = oc.model.path
    else:
        model_path = oc.teacher.path if oc.teacher else ""

    # Use config defaults for vLLM params, but GPU layout from CLI
    if args.model == "student":
        ro = oc.rollout
        gpu_mem = ro.vllm.gpu_memory_utilization if ro else 0.85
        max_model_len = ro.vllm.max_model_len if ro else None
        max_num_seqs = ro.vllm.max_num_seqs if ro else 64
        dtype = ro.dtype if ro else "auto"
        enforce_eager = ro.vllm.enforce_eager if ro else False
        trust_remote_code = (ro.trust_remote_code if ro else oc.model.trust_remote_code)
    else:
        t = oc.teacher
        gpu_mem = t.vllm.gpu_memory_utilization if t else 0.85
        max_model_len = t.vllm.max_model_len if t else None
        max_num_seqs = t.vllm.max_num_seqs if t else 64
        dtype = t.dtype if t else "auto"
        enforce_eager = t.vllm.enforce_eager if t else False
        trust_remote_code = (t.trust_remote_code if t else oc.model.trust_remote_code)

    # CLI overrides
    if args.max_num_seqs is not None:
        max_num_seqs = args.max_num_seqs
    if args.max_model_len is not None:
        max_model_len = args.max_model_len
    if args.gpu_mem is not None:
        gpu_mem = args.gpu_mem
    if args.enforce_eager is not None:
        enforce_eager = args.enforce_eager
    if args.model_path is not None:
        model_path = args.model_path
    if args.trust_remote_code is not None:
        trust_remote_code = args.trust_remote_code
    if args.enable_thinking is not None:
        enable_thinking = args.enable_thinking

    # Default tokenizer to config's model path (not checkpoint path) to avoid
    # corrupted tokenizer saved in checkpoint hf/ dirs.
    tokenizer_path = args.tokenizer or model_path

    # --- Build list of (model_path, step_label) pairs ---
    checkpoints = []
    if args.run_dir:
        ckpt_dir = os.path.join(args.run_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            parser.error(f"No checkpoints dir found in {args.run_dir}")
        for entry in sorted(os.listdir(ckpt_dir)):
            ckpt_path = os.path.join(ckpt_dir, entry)
            if os.path.isdir(ckpt_path) and entry.startswith("step_"):
                # Prefer /hf subdir if it exists (HF-format checkpoint)
                hf_path = os.path.join(ckpt_path, "hf")
                if os.path.isdir(hf_path):
                    ckpt_path = hf_path
                checkpoints.append((ckpt_path, entry))
    else:
        step_label = None
        if args.model_path and "step_" in args.model_path:
            # Extract step label from path like .../checkpoints/step_80
            for part in args.model_path.replace("\\", "/").split("/"):
                if part.startswith("step_"):
                    step_label = part
                    break
        checkpoints.append((model_path, step_label))

    # --- Build list of datasets/benchmarks ---
    if args.eval_type == "code":
        dataset_list = [args.benchmark]
    elif args.datasets:
        dataset_list = args.datasets
    else:
        dataset_list = [val_files] if isinstance(val_files, str) else val_files

    # --- Derive run dir from config path ---
    config_rel = config_path
    for prefix in ("configs/", "configs\\"):
        if config_rel.startswith(prefix):
            config_rel = config_rel[len(prefix):]
            break
    run_base, _ = os.path.splitext(config_rel)
    run_dir = os.path.join("results", run_base)

    # --- Derive base output dir ---
    if args.output_dir:
        base_out_dir = args.output_dir
    elif args.run_dir:
        base_out_dir = os.path.join(args.run_dir, "eval")
    else:
        base_out_dir = os.path.join(run_dir, "baselines")

    # --- Default eval.jsonl path ---
    if args.eval_jsonl is None:
        args.eval_jsonl = os.path.join(run_dir, "eval.jsonl")

    # --- Create Logger for eval.jsonl + optional W&B/ClearML ---
    from opd.utils.logger import Logger
    eval_logger = Logger(
        args.eval_jsonl, config=config,
        backends=args.log or [],
        run_name=f"eval/{run_base}",
        resume=True,
    )

    # --- Load completed evals from eval.jsonl to skip duplicates ---
    completed_evals = set()
    if os.path.exists(args.eval_jsonl):
        with open(args.eval_jsonl) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("type") != "eval":
                        continue
                    completed_evals.add(_eval_dedupe_key(
                        entry.get("step"),
                        entry.get("eval_type", "math"),
                        entry.get("dataset"),
                        entry.get("benchmark"),
                        entry.get("release"),
                        entry.get("date_window"),
                    ))
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

    # --- Eval loop: checkpoints × datasets ---
    all_results = []
    multi_mode = len(checkpoints) > 1 or len(dataset_list) > 1

    for ckpt_path, step_label in checkpoints:
        for ds in dataset_list:
            # Derive dataset name for output file
            ds_name = ds.replace("hf:", "").replace("/", "_").replace("\\", "_")
            if "." in ds_name:
                ds_name = os.path.splitext(ds_name)[0]

            # Skip if already evaluated (recorded in eval.jsonl)
            if args.step is not None:
                step_val = args.step
            elif step_label and step_label.startswith("step_"):
                step_val = int(step_label.split("_", 1)[1])
            elif step_label and step_label.isdigit():
                step_val = int(step_label)
            else:
                step_val = step_label
            date_window = {"start": args.date_start, "end": args.date_end}
            eval_key = _eval_dedupe_key(
                step_val, args.eval_type, ds,
                args.benchmark if args.eval_type == "code" else None,
                args.release, date_window,
            )
            if eval_key in completed_evals:
                print(f"[skip] Step {step_val}, {ds}: already in eval.jsonl", flush=True)
                continue

            # Output path
            if multi_mode:
                if step_label:
                    out_dir = os.path.join(base_out_dir, step_label)
                else:
                    out_dir = base_out_dir
                output_path = os.path.join(out_dir, f"{ds_name}.jsonl")
            else:
                out_dir = base_out_dir
                output_name = args.output_name or f"{args.model}.jsonl"
                output_path = os.path.join(out_dir, output_name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n{'='*60}", flush=True)
            print(f"Evaluating: {ckpt_path}", flush=True)
            print(f"  Dataset: {ds}", flush=True)
            if tokenizer_path:
                print(f"  Tokenizer: {tokenizer_path}", flush=True)
            print(f"  GPUs: {args.gpus}, TP={args.tp}, DP={args.dp}", flush=True)
            print(f"  Eval: n_samples={n_samples}, temp={temperature}, "
                  f"enable_thinking={enable_thinking}", flush=True)
            print(f"  Output: {output_path}", flush=True)
            print(f"{'='*60}", flush=True)

            if args.eval_type == "code":
                generation_path = args.generate_only or _artifact_path_for_output(output_path)
                artifact = generate_responses(
                    model_path=ckpt_path,
                    gpu_ids=args.gpus,
                    tp_size=args.tp,
                    dp_size=args.dp,
                    gpu_mem=gpu_mem,
                    max_model_len=max_model_len,
                    max_num_seqs=max_num_seqs,
                    dataset=ds,
                    prompt_key=prompt_key,
                    answer_key=answer_key,
                    prompt_template=prompt_template,
                    max_prompt_length=max_prompt_length,
                    max_response_length=max_response_length,
                    n_samples=n_samples,
                    temperature=temperature,
                    dtype=dtype,
                    enforce_eager=enforce_eager,
                    enable_thinking=enable_thinking,
                    tokenizer_path=tokenizer_path,
                    eval_type="code",
                    benchmark=args.benchmark,
                    release=args.release,
                    date_start=args.date_start,
                    date_end=args.date_end,
                    trust_remote_code=trust_remote_code,
                )
                from opd.utils.code_eval import write_generation_artifact
                write_generation_artifact(artifact, generation_path)
                print(f"Saved generation artifact to: {generation_path}", flush=True)
                if args.generate_only:
                    all_results.append({
                        "checkpoint": ckpt_path,
                        "step": step_label,
                        "dataset": ds,
                        "output": generation_path,
                    })
                    continue
                metrics = score_code(
                    artifact,
                    benchmark=args.benchmark,
                    output_dir=out_dir,
                    output_path=output_path,
                    code_workers=args.code_workers,
                    code_timeout=args.code_timeout,
                    generations_path=generation_path,
                    allow_unsafe_code_execution=args.allow_unsafe_code_execution,
                )
                acc = metrics.get("pass_at_1")
                if acc is None:
                    acc = metrics.get(f"avg_at_{n_samples}", 0.0)
                all_results.append({
                    "checkpoint": ckpt_path,
                    "step": step_label,
                    "dataset": ds,
                    "accuracy": acc,
                    "output": output_path,
                })
                print(f"Saved to: {output_path}", flush=True)
                eval_logger.log_eval(step_val, metrics)
                continue

            # When evaluating on a dataset override, use auto-detection
            # for column names since the override dataset may differ from config.
            # Also ignore config-level strict answer patterns unless the user
            # explicitly supplied one, because GSM8K/AIME/MATH-style datasets
            # often need different extraction behavior.
            ds_is_override = args.datasets and ds in args.datasets
            ds_prompt_key = "prompt" if ds_is_override else prompt_key
            ds_answer_key = "auto" if ds_is_override else answer_key
            ds_answer_pattern = (
                args.answer_pattern
                if args.answer_pattern
                else (None if ds_is_override else answer_pattern)
            )

            if args.generate_only:
                artifact = generate_responses(
                    model_path=ckpt_path,
                    gpu_ids=args.gpus,
                    tp_size=args.tp,
                    dp_size=args.dp,
                    gpu_mem=gpu_mem,
                    max_model_len=max_model_len,
                    max_num_seqs=max_num_seqs,
                    dataset=ds,
                    prompt_key=ds_prompt_key,
                    answer_key=ds_answer_key,
                    prompt_template=prompt_template,
                    max_prompt_length=max_prompt_length,
                    max_response_length=max_response_length,
                    n_samples=n_samples,
                    temperature=temperature,
                    dtype=dtype,
                    enforce_eager=enforce_eager,
                    enable_thinking=enable_thinking,
                    tokenizer_path=tokenizer_path,
                    eval_type="math",
                    trust_remote_code=trust_remote_code,
                )
                from opd.utils.code_eval import write_generation_artifact
                write_generation_artifact(artifact, args.generate_only)
                print(f"Saved generation artifact to: {args.generate_only}", flush=True)
                all_results.append({
                    "checkpoint": ckpt_path,
                    "step": step_label,
                    "dataset": ds,
                    "output": args.generate_only,
                })
                continue

            acc = evaluate_model(
                model_path=ckpt_path,
                gpu_ids=args.gpus,
                tp_size=args.tp,
                dp_size=args.dp,
                gpu_mem=gpu_mem,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                val_files=ds,
                prompt_key=ds_prompt_key,
                answer_key=ds_answer_key,
                prompt_template=prompt_template,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                n_samples=n_samples,
                temperature=temperature,
                dtype=dtype,
                enforce_eager=enforce_eager,
                enable_thinking=enable_thinking,
                output_path=output_path,
                tokenizer_path=tokenizer_path,
                answer_pattern=ds_answer_pattern,
                trust_remote_code=trust_remote_code,
            )
            all_results.append({
                "checkpoint": ckpt_path,
                "step": step_label,
                "dataset": ds,
                "accuracy": acc,
                "output": output_path,
            })
            print(f"Saved to: {output_path}", flush=True)

            eval_logger.log_eval(step_val, {
                f"avg_at_{n_samples}": round(acc, 4),
                "dataset": ds,
                "eval_type": "math",
                "benchmark": None,
                "release": None,
                "date_window": {"start": None, "end": None},
            })

    # --- Summary ---
    if multi_mode and all_results:
        print(f"\n{'='*60}", flush=True)
        print("Summary:", flush=True)
        for r in all_results:
            step_str = f" ({r['step']})" if r['step'] else ""
            print(f"  {r['dataset']}{step_str}: {r['accuracy']:.2f}%", flush=True)
        # Save summary CSV
        summary_path = os.path.join(base_out_dir, "eval_summary.csv")
        with open(summary_path, "w") as f:
            f.write("checkpoint,step,dataset,accuracy\n")
            for r in all_results:
                f.write(f"{r['checkpoint']},{r['step'] or ''},{r['dataset']},{r['accuracy']:.2f}\n")
        print(f"Summary saved to: {summary_path}", flush=True)

    eval_logger.close()


if __name__ == "__main__":
    main()

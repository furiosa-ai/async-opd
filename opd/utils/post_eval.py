"""Post-training all-GPU evaluation.

Evaluates all checkpoints using all GPUs in data-parallel mode after
pipeline.shutdown() frees all GPUs. Each eval runs as a subprocess
via eval.py to get a fresh CUDA context.
"""

import json
import os
import subprocess
import sys
import time

from opd.utils.net import leased_port

# Inline script for subprocess checkpoint conversion.
# Runs in a fresh process to avoid corrupted NCCL/torch state from pipeline shutdown.
# Takes model_path followed by pairs of (ckpt_pt, hf_dir) — loads base model once.
_CONVERT_CKPT_SCRIPT = """\
import sys, torch
from transformers import AutoModelForCausalLM
from opd.utils.config import resolve_trust_remote_code
model_path = sys.argv[1]
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype="auto",
    trust_remote_code=resolve_trust_remote_code(context="post-eval checkpoint conversion"),
    local_files_only=True)
for i, (ckpt_pt, hf_dir) in enumerate(pairs):
    print(f"  Converting {i+1}/{len(pairs)}: {ckpt_pt}", flush=True)
    sd = torch.load(ckpt_pt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.save_pretrained(hf_dir)
    del sd
"""


def collect_gpu_ids(config):
    """Extract all GPU IDs used by teacher, rollout, and trainer from config.

    Accepts either an OPDConfig dataclass or old-format dict.
    """
    from opd.utils.config import OPDConfig
    gpu_ids = set()
    if isinstance(config, OPDConfig):
        gpu_strs = [
            config.teacher.gpu_ids if config.teacher else "",
            config.rollout.gpu_ids if config.rollout else "",
            config.trainer.gpu_ids or "",
        ]
    else:
        gpu_strs = [
            config.get("teacher", {}).get("gpu_ids", ""),
            config.get("training", {}).get("actor_rollout_ref", {}).get("rollout", {}).get("gpu_ids", ""),
            config.get("training", {}).get("rollout", {}).get("gpu_ids", ""),
            config.get("training", {}).get("trainer", {}).get("gpu_ids", ""),
        ]
    for gpu_str in gpu_strs:
        for g in str(gpu_str or "").split(","):
            g = g.strip()
            if g.isdigit():
                gpu_ids.add(int(g))
    return gpu_ids


def select_post_eval_checkpoint_steps(steps, policy="all", requested_steps=None):
    """Select checkpoint steps for post-eval.

    Args:
        steps: Iterable of checkpoint step integers found on disk. Step 0 may
            be included by callers when they explicitly want base-model eval.
        policy: "all", "final", or "steps".
        requested_steps: Explicit steps for policy="steps".

    Returns:
        Sorted list of selected step integers, preserving the requested order
        for explicit steps while dropping unavailable checkpoints.
    """
    steps = sorted({int(s) for s in steps})
    requested_steps = requested_steps or []

    if policy == "all":
        return steps
    if policy == "final":
        positive_steps = [s for s in steps if s > 0]
        return [positive_steps[-1]] if positive_steps else []
    if policy == "steps":
        available = set(steps)
        selected = []
        for step in requested_steps:
            step = int(step)
            if step in available and step not in selected:
                selected.append(step)
        return selected
    raise ValueError(f"unsupported eval checkpoint policy: {policy!r}")


def resolve_post_eval_checkpoint_source(ckpt_dir, step, model_path):
    """Resolve the model source needed to evaluate one checkpoint step.

    Returns ``(hf_path, convert_item, missing_path)``:
      - ``hf_path`` when the checkpoint can be evaluated immediately.
      - ``convert_item`` as ``(step, model_pt, hf_dir)`` when a raw
        ``model.pt`` must be converted to HF format first.
      - ``missing_path`` when neither an existing HF checkpoint nor
        ``model.pt`` is available.

    HF-only checkpoints are valid: remote cleanup may delete ``model.pt`` after
    conversion to save disk, while eval.py only needs the HF directory.
    """
    if step == 0:
        return model_path, None, None

    step_dir = os.path.join(ckpt_dir, f"step_{step}")
    hf_dir = os.path.join(step_dir, "hf")
    if os.path.exists(os.path.join(hf_dir, "config.json")):
        return hf_dir, None, None

    ckpt_pt = os.path.join(step_dir, "model.pt")
    if not os.path.exists(ckpt_pt):
        return None, None, ckpt_pt

    return None, (step, ckpt_pt, hf_dir), None



def _is_code_post_eval_entry(entry):
    return isinstance(entry, dict) and entry.get("type") == "code"


def _code_post_eval_name(entry):
    parts = [entry["benchmark"]]
    if entry.get("release"):
        parts.append(str(entry["release"]))
    if entry.get("date_start") or entry.get("date_end"):
        parts.append(f"{entry.get('date_start') or 'start'}_{entry.get('date_end') or 'end'}")
    return "_".join(parts).replace("/", "_").replace(":", "_")


def _math_post_eval_dataset(entry):
    return entry if isinstance(entry, str) else entry["dataset"]


def _read_code_summary(path):
    with open(path) as f:
        first = f.readline().strip()
    return json.loads(first) if first else None


def _code_post_eval_executes_code(entry):
    """Return whether a code post-eval entry opts into inline code execution.

    Code benchmark generation can be GPU-heavy while sandboxed execution is
    CPU-heavy.  Default post_allgpu behavior keeps dispatcher throughput high
    by generating artifacts only; users must explicitly opt in when they want
    pass@k scoring before run.py exits.
    """
    return bool(entry.get("execute_code", False))


def _code_generation_artifact_path(output_path):
    root, _ext = os.path.splitext(output_path)
    return f"{root}.generations.json"


def _write_code_generation_summary(output_path, generation_path, *, step, benchmark,
                                   release=None, date_start=None, date_end=None):
    """Write a one-line eval summary for generation-only code post-eval."""
    metrics = {
        "eval_type": "code",
        "dataset": benchmark,
        "benchmark": benchmark,
        "release": release,
        "date_window": {"start": date_start, "end": date_end},
        "step": step,
        "score_skipped": True,
        "code_execution": "disabled",
        "generations_path": generation_path,
        "pass_at_1": None,
        "pass_at_4": None,
    }
    try:
        from opd.utils.code_eval import load_generation_artifact
        artifact = load_generation_artifact(generation_path)
        metrics["n_tasks"] = len(artifact.get("prompt_metadata", []))
        metrics["n_samples"] = max(
            (len(samples) for samples in artifact.get("responses", [])),
            default=0,
        )
    except Exception:
        # The eval subprocess succeeded, so keep a durable pointer even if the
        # artifact cannot be inspected in this parent process.
        pass

    with open(output_path, "w") as f:
        f.write(json.dumps(metrics, sort_keys=True) + "\n")
    return metrics


def run_allgpu_post_eval(config, config_path, run_dir, logger, tracer=None):
    """Evaluate requested checkpoints using all GPUs in data-parallel mode.

    Called after pipeline.shutdown() so all GPUs are free. Each eval runs as
    a subprocess via eval.py to get a fresh CUDA context.

    Args:
        config: OPDConfig dataclass or old-format dict.
    """
    from opd.utils.config import OPDConfig
    if isinstance(config, OPDConfig):
        oc = config
        test_freq = oc.eval.freq
        val_before_train = oc.eval.before_train
        n_samples = oc.eval.n_samples
        checkpoint_policy = oc.eval.checkpoint_policy
        checkpoint_steps = oc.eval.checkpoint_steps
        run_primary = oc.eval.run_primary
        model_path = oc.model.path
        tokenizer_path = oc.data.tokenizer_path or model_path
        val_files = oc.data.val_files
        # Compat dicts for downstream code that still reads them.
        data_cfg = {
            "val_files": val_files,
            "post_eval_datasets": oc.data.post_eval_datasets,
            "tokenizer_path": oc.data.tokenizer_path,
        }
        eval_cfg = {"eval_max_response_length": oc.eval.max_response_length}
        train_cfg = {
            "data": {
                "max_prompt_length": oc.data.max_prompt_length,
                "max_response_length": oc.data.max_response_length,
            }
        }
    else:
        train_cfg = config["training"]
        data_cfg = config["data"]
        eval_cfg = train_cfg.get("trainer", {})
        test_freq = eval_cfg.get("test_freq", -1)
        val_before_train = eval_cfg.get("val_before_train", True)
        n_samples = eval_cfg.get("eval_n_samples", 1)
        checkpoint_policy = eval_cfg.get("checkpoint_policy", "all")
        checkpoint_steps = eval_cfg.get("checkpoint_steps", [])
        run_primary = eval_cfg.get("run_primary", True)
        model_path = train_cfg["actor_rollout_ref"]["model"]["path"]
        tokenizer_path = data_cfg.get("tokenizer_path", model_path)
        val_files = data_cfg.get("val_files")

    extra_datasets = data_cfg.get("post_eval_datasets", [])
    if not val_files and run_primary:
        print("[AllGPU-PostEval] No val_files configured for primary eval.", flush=True)
        run_primary = False
    if not run_primary and not extra_datasets:
        print("[AllGPU-PostEval] No primary eval or post_eval_datasets requested; skipping.", flush=True)
        return
    if test_freq <= 0 and checkpoint_policy == "all" and not extra_datasets:
        print("[AllGPU-PostEval] test_freq <= 0 and no extra datasets, skipping.", flush=True)
        return

    # Collect all GPU IDs.
    all_gpus = sorted(collect_gpu_ids(config))
    n_gpus = len(all_gpus)
    gpu_ids_str = ",".join(str(g) for g in all_gpus)
    print(f"[AllGPU-PostEval] Using {n_gpus} GPUs: {gpu_ids_str}", flush=True)

    # Discover checkpoint steps.
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    steps = []
    if os.path.exists(ckpt_dir):
        for d in os.listdir(ckpt_dir):
            if d.startswith("step_"):
                try:
                    steps.append(int(d.split("_")[1]))
                except (ValueError, IndexError):
                    pass
    steps.sort()

    selected_steps = select_post_eval_checkpoint_steps(
        steps,
        policy=checkpoint_policy,
        requested_steps=checkpoint_steps,
    )

    include_step0 = (
        val_before_train and checkpoint_policy == "all"
    ) or (
        checkpoint_policy == "steps" and 0 in {int(s) for s in checkpoint_steps}
    )

    def _maybe_with_step0(step_list):
        if include_step0 and 0 not in step_list:
            return [0] + list(step_list)
        return list(step_list)

    primary_candidate_steps = _maybe_with_step0(selected_steps) if run_primary else []
    extra_candidate_steps = _maybe_with_step0(selected_steps) if extra_datasets else []

    if not primary_candidate_steps and not extra_candidate_steps:
        print("[AllGPU-PostEval] No requested checkpoint steps found, skipping.", flush=True)
        return

    print(
        f"[AllGPU-PostEval] Checkpoint policy={checkpoint_policy}; "
        f"primary candidates={primary_candidate_steps}; "
        f"extra candidates={extra_candidate_steps}",
        flush=True,
    )

    # Skip already-evaluated primary steps.
    val_dir = os.path.join(run_dir, "validation_outputs")
    os.makedirs(val_dir, exist_ok=True)
    eval_steps = []
    for s in primary_candidate_steps:
        val_file = os.path.join(val_dir, f"step_{s}.jsonl")
        if os.path.exists(val_file) and os.path.getsize(val_file) > 0:
            continue
        eval_steps.append(s)

    # Check if extra datasets still need evaluation.
    has_pending_extra = False
    if extra_datasets:
        for ds_entry in extra_datasets:
            if _is_code_post_eval_entry(ds_entry):
                ds_name = _code_post_eval_name(ds_entry)
            else:
                ds_path = _math_post_eval_dataset(ds_entry)
                ds_name = ds_path.replace("hf:", "").replace("/", "_")
            ds_dir = os.path.join(run_dir, "eval", ds_name)
            for s in extra_candidate_steps:
                out = os.path.join(ds_dir, f"step_{s}.jsonl")
                if not (os.path.exists(out) and os.path.getsize(out) > 0):
                    has_pending_extra = True
                    break
            if has_pending_extra:
                break

    if not eval_steps and not has_pending_extra:
        print("[AllGPU-PostEval] All requested checkpoints already evaluated.", flush=True)
        print("[AllGPU-PostEval] Done — evaluated 0 checkpoints.", flush=True)
        return

    print(f"[AllGPU-PostEval] Evaluating primary steps: {eval_steps}", flush=True)

    # Pre-cache HF datasets so eval subprocesses work with HF_HUB_OFFLINE=1.
    # load_dataframe resolves short aliases (e.g. AMC23, MATH-500) before
    # deciding whether to hit HuggingFace or local parquet.
    from opd.data.prompt import load_dataframe
    extra_dataset_refs = [
        _math_post_eval_dataset(d)
        for d in extra_datasets
        if not _is_code_post_eval_entry(d)
    ]
    for ds in ([data_cfg.get("val_files", "")] if run_primary else []) + extra_dataset_refs:
        if ds:
            try:
                load_dataframe(ds)
            except Exception:
                pass
    for ds_entry in extra_datasets:
        if _is_code_post_eval_entry(ds_entry):
            try:
                from opd.utils.code_eval import load_code_benchmark_prompts
                load_code_benchmark_prompts(
                    ds_entry["benchmark"],
                    release=ds_entry.get("release"),
                    date_start=ds_entry.get("date_start"),
                    date_end=ds_entry.get("date_end"),
                )
            except Exception:
                pass

    # Resolve HF checkpoint dirs for vLLM. If an HF dir already exists, use it
    # directly even when model.pt was deleted to save disk. Otherwise convert
    # model.pt state dicts to HF model dirs.
    # Run in a single subprocess to avoid corrupted NCCL/torch state from
    # pipeline shutdown. Loads the base model once and reuses it for requested checkpoints.
    python = sys.executable
    hf_dirs = {}  # step -> path
    all_steps_to_check = sorted(set(eval_steps) | set(extra_candidate_steps))
    to_convert = []  # (step, ckpt_pt, hf_dir)
    for step in all_steps_to_check:
        hf_path, convert_item, missing_path = resolve_post_eval_checkpoint_source(
            ckpt_dir, step, model_path,
        )
        if hf_path is not None:
            hf_dirs[step] = hf_path
        elif convert_item is not None:
            to_convert.append(convert_item)
        else:
            print(f"[AllGPU-PostEval] WARNING: {missing_path} not found and "
                  f"no HF checkpoint exists, skipping step {step}", flush=True)

    if to_convert:
        print(f"[AllGPU-PostEval] Converting {len(to_convert)} checkpoints to HF format...",
              flush=True)
        # Build args: model_path ckpt1 hf1 ckpt2 hf2 ...
        conv_args = [model_path]
        for _, ckpt_pt, hf_dir in to_convert:
            conv_args.extend([ckpt_pt, hf_dir])
        result = subprocess.run(
            [python, "-c", _CONVERT_CKPT_SCRIPT] + conv_args,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        if result.returncode != 0:
            print(f"[AllGPU-PostEval] WARNING: batch conversion failed "
                  f"(exit code {result.returncode})", flush=True)
        # Check which ones actually converted.
        for step, _, hf_dir in to_convert:
            if os.path.exists(os.path.join(hf_dir, "config.json")):
                hf_dirs[step] = hf_dir
            else:
                print(f"[AllGPU-PostEval] WARNING: step {step} conversion incomplete",
                      flush=True)
        # No tokenizer saved — eval loads from original model path to
        # avoid HF tokenizer serialization bug (corrupted regex patterns).

    # --- Helper: run eval.py subprocess and parse accuracy ---
    def _run_eval_step(step, hf_path, out_dir, out_name, dataset_override=None,
                       label="", trace_args=None,
                       n_samples_override=None, temperature_override=None,
                       eval_type="math", benchmark=None, release=None,
                       date_start=None, date_end=None, execute_code=True,
                       code_workers=None, code_timeout=None,
                       max_response_override=None):
        """Run eval.py for one step, return (accuracy, metrics) or None on failure."""
        output_path = os.path.join(out_dir, out_name)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return None  # already evaluated

        step_n = n_samples_override or n_samples
        generate_only_path = None
        if eval_type == "code" and not execute_code:
            generate_only_path = _code_generation_artifact_path(output_path)

        print(f"  {label}Step {step}: evaluating with {n_gpus} GPUs...", flush=True)
        cmd = [
            python, "eval.py",
            "--config", config_path,
            "--model", "student",
            "--gpus", gpu_ids_str,
            "--dp", str(n_gpus),
            "--model-path", hf_path,
            "--tokenizer", tokenizer_path,
            "--output-dir", out_dir,
            "--output-name", out_name,
            "--gpu-mem", "0.85",
            "--step", str(step),
        ]
        if eval_type == "code":
            cmd.extend(["--eval-type", "code", "--benchmark", benchmark])
            if release:
                cmd.extend(["--release", str(release)])
            if date_start:
                cmd.extend(["--date-start", str(date_start)])
            if date_end:
                cmd.extend(["--date-end", str(date_end)])
            if generate_only_path:
                cmd.extend(["--generate-only", generate_only_path])
            if execute_code and code_workers:
                cmd.extend(["--code-workers", str(code_workers)])
            if execute_code and code_timeout:
                cmd.extend(["--code-timeout", str(code_timeout)])

        eval_max_resp = max_response_override or eval_cfg.get("eval_max_response_length")
        if eval_max_resp:
            cmd.extend(["--max-response-length", str(eval_max_resp)])
            # Ensure vLLM model length fits the longer eval responses.
            max_prompt = train_cfg["data"]["max_prompt_length"]
            cmd.extend(["--max-model-len", str(max_prompt + eval_max_resp)])
        if dataset_override and eval_type == "math":
            cmd.extend(["--datasets", dataset_override])
        if n_samples_override:
            cmd.extend(["--eval-n-samples", str(n_samples_override)])
        if temperature_override:
            cmd.extend(["--eval-temperature", str(temperature_override)])

        # Prevent HF Hub network calls in eval workers — models/data are
        # already cached from training. HF timeouts cause worker crashes.
        # Use our port range (55000-59999) to avoid vLLM's default 45xxx
        # range where zombie processes from prior eval steps may linger.
        with leased_port(
            "post_eval.eval_subprocess.master",
            metadata={"step": step, "output": out_name},
        ) as master_lease:
            eval_env = {
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "MASTER_PORT": str(master_lease.port),
                "TORCH_DISABLE_ADDR2LINE": "1",
            }
            t_start = time.monotonic()
            result = subprocess.run(cmd, env=eval_env)
            t_end = time.monotonic()

        if result.returncode != 0:
            # Retry once — transient port conflicts (EADDRINUSE) from
            # lingering TCPStore daemons resolve after a brief delay.
            print(f"  {label}WARNING: eval failed for step {step} "
                  f"(exit code {result.returncode}), retrying in 10s...",
                  flush=True)
            time.sleep(10)
            with leased_port(
                "post_eval.eval_subprocess.master_retry",
                metadata={"step": step, "output": out_name},
            ) as retry_lease:
                eval_env = {
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "MASTER_PORT": str(retry_lease.port),
                    "TORCH_DISABLE_ADDR2LINE": "1",
                }
                t_start = time.monotonic()
                result = subprocess.run(cmd, env=eval_env)
                t_end = time.monotonic()
            if result.returncode != 0:
                print(f"  {label}WARNING: eval retry also failed for step {step} "
                      f"(exit code {result.returncode})", flush=True)
                return None

        if tracer is not None:
            tracer.emit("eval", cat="eval", tid=13,
                        t_start=t_start, t_end=t_end,
                        args={"step": step, "mode": "post_allgpu",
                              **(trace_args or {})})

        # Parse metrics.
        if generate_only_path:
            if not os.path.exists(generate_only_path):
                return None
            metrics = _write_code_generation_summary(
                output_path,
                generate_only_path,
                step=step,
                benchmark=benchmark,
                release=release,
                date_start=date_start,
                date_end=date_end,
            )
            return None, metrics

        if not os.path.exists(output_path):
            return None
        if eval_type == "code":
            metrics = _read_code_summary(output_path)
            if not metrics:
                return None
            accuracy = metrics.get("pass_at_1")
            if accuracy is None:
                accuracy = metrics.get(f"avg_at_{step_n}", 0)
            return accuracy, metrics

        from opd.utils.eval import score_problems
        problem_pass = {}
        with open(output_path) as f:
            for line in f:
                d = json.loads(line)
                pid = d["problem_id"]
                if pid not in problem_pass:
                    problem_pass[pid] = [0, 0]
                problem_pass[pid][1] += 1
                if d.get("correct"):
                    problem_pass[pid][0] += 1

        problem_results = [{"n_correct": v[0], "n_total": v[1]}
                           for v in problem_pass.values()]
        metrics = score_problems(problem_results, step_n)
        metrics.setdefault("eval_type", "math")
        metrics.setdefault("dataset", dataset_override or val_files)
        accuracy = metrics.get("accuracy", metrics.get(f"avg_at_{step_n}", 0))
        return accuracy, metrics

    # --- Main val_files eval ---
    n_primary_evaluated = 0
    for step in eval_steps:
        hf_path = hf_dirs.get(step)
        if hf_path is None:
            continue
        result = _run_eval_step(
            step, hf_path, val_dir, f"step_{step}.jsonl",
            label="[AllGPU-PostEval] ")
        if result is not None:
            accuracy, metrics = result
            logger.log_eval(step, metrics)
            n_primary_evaluated += 1
            print(f"[AllGPU-PostEval] Step {step}: {accuracy:.2f}%", flush=True)

    # --- Additional datasets (post_eval_datasets in config) ---
    # Supports string/dict math entries and dict code entries:
    #   post_eval_datasets:
    #     - "hf:Maxwell-Jia/AIME_2024"
    #     - dataset: "hf:Maxwell-Jia/AIME_2024"
    #       eval_n_samples: 32
    #     - type: code
    #       benchmark: humaneval_plus
    #       eval_n_samples: 4
    #       # Default is generation-only; set execute_code: true to run
    #       # CPU code scoring inline before run.py exits.
    #       execute_code: true
    n_extra_evaluated = 0
    for ds_entry in extra_datasets:
        if _is_code_post_eval_entry(ds_entry):
            ds_name = _code_post_eval_name(ds_entry)
            ds_dir = os.path.join(run_dir, "eval", ds_name)
            os.makedirs(ds_dir, exist_ok=True)
            ds_n_samples = ds_entry.get("eval_n_samples")
            ds_temperature = ds_entry.get("eval_temperature")
            ds_max_response = ds_entry.get("max_response_length")
            print(f"\n[AllGPU-PostEval] Evaluating code benchmark {ds_name}...", flush=True)
            for step in extra_candidate_steps:
                hf_path = hf_dirs.get(step)
                if hf_path is None:
                    continue
                result = _run_eval_step(
                    step, hf_path, ds_dir, f"step_{step}.jsonl",
                    label=f"  [{ds_name}] ",
                    trace_args={"eval_type": "code", "benchmark": ds_entry["benchmark"]},
                    n_samples_override=ds_n_samples,
                    temperature_override=ds_temperature,
                    eval_type="code",
                    benchmark=ds_entry["benchmark"],
                    release=ds_entry.get("release"),
                    date_start=ds_entry.get("date_start"),
                    date_end=ds_entry.get("date_end"),
                    execute_code=_code_post_eval_executes_code(ds_entry),
                    code_workers=ds_entry.get("code_workers"),
                    code_timeout=ds_entry.get("code_timeout"),
                    max_response_override=ds_max_response,
                )
                if result is not None:
                    accuracy, metrics = result
                    logger.log_eval(step, metrics)
                    n_extra_evaluated += 1
                    if metrics.get("score_skipped"):
                        print(
                            f"  Step {step} [{ds_name}]: generated "
                            f"{metrics.get('generations_path')} (code execution disabled)",
                            flush=True,
                        )
                    else:
                        print(f"  Step {step} [{ds_name}]: {accuracy:.2f}%", flush=True)
            print(f"[AllGPU-PostEval] {ds_name} done.", flush=True)
            continue

        if isinstance(ds_entry, str):
            ds_path = ds_entry
            ds_n_samples = None
            ds_temperature = None
        else:
            ds_path = ds_entry["dataset"]
            ds_n_samples = ds_entry.get("eval_n_samples")
            ds_temperature = ds_entry.get("eval_temperature")

        ds_name = ds_path.replace("hf:", "").replace("/", "_")
        ds_dir = os.path.join(run_dir, "eval", ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        ds_label = f"{ds_name}"
        if ds_n_samples:
            ds_label += f" (Avg@{ds_n_samples})"
        print(f"\n[AllGPU-PostEval] Evaluating on {ds_label}...", flush=True)

        for step in extra_candidate_steps:
            hf_path = hf_dirs.get(step)
            if hf_path is None:
                continue
            result = _run_eval_step(
                step, hf_path, ds_dir, f"step_{step}.jsonl",
                dataset_override=ds_path, label=f"  [{ds_name}] ",
                trace_args={"dataset": ds_name},
                n_samples_override=ds_n_samples,
                temperature_override=ds_temperature)
            if result is not None:
                accuracy, metrics = result
                logger.log_eval(step, metrics)
                n_extra_evaluated += 1
                print(f"  Step {step} [{ds_name}]: {accuracy:.2f}%", flush=True)

        print(f"[AllGPU-PostEval] {ds_name} done.", flush=True)

    print(
        f"[AllGPU-PostEval] Done — evaluated "
        f"{n_primary_evaluated} primary and {n_extra_evaluated} extra checkpoints.",
        flush=True,
    )

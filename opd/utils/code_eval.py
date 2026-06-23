"""Utilities for code-generation evaluation artifacts and scoring.

This module is intentionally stdlib-only at import time. Optional benchmark
packages (EvalPlus, LiveCodeBench, Hugging Face datasets) are imported lazily
inside the functions that need them.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

SUPPORTED_BENCHMARKS = {"humaneval_plus", "mbpp_plus", "lcb_v6"}
_EVALPLUS_HF_DATASETS = {
    "humaneval_plus": "evalplus/humanevalplus",
    "mbpp_plus": "evalplus/mbppplus",
}
_LCB_HF_REPO = "livecodebench/code_generation_lite"
_LCB_TEST_FILES = [
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
]


class CodeEvalError(RuntimeError):
    """Raised when code-eval artifacts or optional runners are invalid."""


@dataclasses.dataclass(frozen=True)
class CodeBenchmarkProblem:
    """Prompt metadata needed by eval.py for code benchmark generation."""

    problem_id: int
    prompt_id: str
    dataset: str
    benchmark: str
    prompt: str
    task_id: str | None = None
    question_id: str | None = None
    contest_date: str | None = None
    entry_point: str | None = None
    tests: str | list[str] | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


def _stable_id(*parts: Any) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _as_date(value: str | None) -> _dt.date | None:
    if not value:
        return None
    return _dt.date.fromisoformat(str(value))


def _in_date_window(value: str | None, start: str | None, end: str | None) -> bool:
    if not value or (not start and not end):
        return True
    date = _as_date(str(value)[:10])
    start_date = _as_date(start)
    end_date = _as_date(end)
    if start_date and date < start_date:
        return False
    if end_date and date > end_date:
        return False
    return True


def normalize_benchmark(benchmark: str | None) -> str:
    if not benchmark:
        raise CodeEvalError("code eval requires a benchmark")
    benchmark = benchmark.lower()
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise CodeEvalError(
            f"unsupported code benchmark {benchmark!r}; expected one of {sorted(SUPPORTED_BENCHMARKS)}"
        )
    return benchmark


def canonical_evalplus_task_id(benchmark: str, task_id: Any) -> str | None:
    """Return the task id format required by the official EvalPlus runner."""
    if task_id is None:
        return None
    task_id_text = str(task_id)
    if benchmark == "humaneval_plus":
        return task_id_text if task_id_text.startswith("HumanEval/") else f"HumanEval/{task_id_text}"
    if benchmark == "mbpp_plus":
        return task_id_text if task_id_text.startswith("Mbpp/") else f"Mbpp/{task_id_text}"
    return task_id_text


def _load_dataset(*args, **kwargs):
    """Import Hugging Face datasets lazily so module import stays lightweight."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise CodeEvalError(
            "Hugging Face datasets is required to load code benchmark prompts."
        ) from exc
    return load_dataset(*args, **kwargs)


def _hf_hub_download(*args, **kwargs):
    """Import Hugging Face Hub lazily so module import stays lightweight."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise CodeEvalError(
            "huggingface_hub is required to load code benchmark prompts."
        ) from exc
    return hf_hub_download(*args, **kwargs)


def extract_python_code(response: str) -> str:
    """Extract Python code from a model response.

    Selection is deterministic: the last fenced Python block wins, then the last
    generic fenced block, then the raw response text.
    """
    text = "" if response is None else str(response)
    fence_re = re.compile(r"```[ \t]*(?P<info>[^\n`]*)\n(?P<code>.*?)```", re.DOTALL)
    python_blocks: list[str] = []
    generic_blocks: list[str] = []
    for match in fence_re.finditer(text):
        info = match.group("info").strip().lower()
        code = match.group("code").strip("\n")
        if info.startswith(("python", "py")):
            python_blocks.append(code)
        elif not info:
            generic_blocks.append(code)
    if python_blocks:
        return python_blocks[-1].strip()
    if generic_blocks:
        return generic_blocks[-1].strip()
    return text.strip()


def validate_generation_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Validate common generation artifact structure and return it."""
    required = ["schema_version", "eval_type", "prompt_metadata", "responses"]
    missing = [k for k in required if k not in artifact]
    if missing:
        raise CodeEvalError(f"generation artifact missing required keys: {missing}")
    if artifact["schema_version"] != 1:
        raise CodeEvalError(f"unsupported schema_version: {artifact['schema_version']!r}")
    if artifact["eval_type"] not in {"math", "code"}:
        raise CodeEvalError("artifact eval_type must be 'math' or 'code'")
    prompt_metadata = artifact["prompt_metadata"]
    responses = artifact["responses"]
    if not isinstance(prompt_metadata, list) or not isinstance(responses, list):
        raise CodeEvalError("prompt_metadata and responses must be lists")
    if len(prompt_metadata) != len(responses):
        raise CodeEvalError(
            "response count does not match prompt_metadata count: "
            f"{len(responses)} != {len(prompt_metadata)}"
        )
    for idx, sample_texts in enumerate(responses):
        if not isinstance(sample_texts, list):
            raise CodeEvalError(f"responses[{idx}] must be a list of sample strings")
    if artifact["eval_type"] == "code":
        benchmark = normalize_benchmark(artifact.get("benchmark"))
        for idx, meta in enumerate(prompt_metadata):
            if not isinstance(meta, dict):
                raise CodeEvalError(f"prompt_metadata[{idx}] must be a dict")
            if benchmark in {"humaneval_plus", "mbpp_plus"} and meta.get("task_id") is None:
                raise CodeEvalError(f"{benchmark} metadata row {idx} requires task_id")
            if benchmark == "lcb_v6" and not meta.get("question_id"):
                raise CodeEvalError(f"lcb_v6 metadata row {idx} requires question_id")
    return artifact


def load_generation_artifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path) as f:
        return validate_generation_artifact(json.load(f))


def write_generation_artifact(artifact: dict[str, Any], path: str | os.PathLike[str]) -> str:
    validate_generation_artifact(artifact)
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def iter_code_samples(artifact: dict[str, Any]):
    validate_generation_artifact(artifact)
    for problem_id, (meta, sample_texts) in enumerate(
        zip(artifact["prompt_metadata"], artifact["responses"])
    ):
        for sample_idx, response in enumerate(sample_texts):
            code = extract_python_code(response)
            yield {
                "problem_id": meta.get("problem_id", problem_id),
                "sample_idx": sample_idx,
                "task_id": meta.get("task_id"),
                "question_id": meta.get("question_id"),
                "prompt_id": meta.get("prompt_id"),
                "prompt": meta.get("prompt"),
                "raw_response": response,
                "extracted_code": code,
                "metadata": meta,
            }


def write_evalplus_samples(
    artifact: dict[str, Any], path: str | os.PathLike[str], *, use_completion: bool = False
) -> str:
    """Write EvalPlus-compatible JSONL samples."""
    validate_generation_artifact(artifact)
    benchmark = normalize_benchmark(artifact.get("benchmark") or artifact.get("dataset"))
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    code_key = "completion" if use_completion else "solution"
    with open(path, "w") as f:
        for sample in iter_code_samples(artifact):
            task_id = canonical_evalplus_task_id(benchmark, sample.get("task_id"))
            if task_id is None:
                raise CodeEvalError("EvalPlus samples require task_id")
            row = {
                "task_id": task_id,
                code_key: sample["extracted_code"],
                "problem_id": sample["problem_id"],
                "sample_idx": sample["sample_idx"],
            }
            f.write(json.dumps(row) + "\n")
    return path


def write_lcb_samples(artifact: dict[str, Any], path: str | os.PathLike[str]) -> str:
    """Write LiveCodeBench custom-evaluator JSON samples."""
    validate_generation_artifact(artifact)
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for problem_id, (meta, sample_texts) in enumerate(
        zip(artifact["prompt_metadata"], artifact["responses"])
    ):
        question_id = meta.get("question_id")
        if not question_id:
            raise CodeEvalError("LiveCodeBench samples require question_id")
        rows.append({
            "question_id": question_id,
            "code_list": [extract_python_code(r) for r in sample_texts],
            "problem_id": meta.get("problem_id", problem_id),
            "prompt_id": meta.get("prompt_id"),
        })
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _metadata_tests(meta: dict[str, Any]) -> str | None:
    tests = meta.get("tests", meta.get("test"))
    if tests is None:
        tests = meta.get("assertions")
    if tests is None:
        return None
    if isinstance(tests, list):
        return "\n".join(str(t) for t in tests)
    return str(tests)


def artifact_has_fixture_tests(artifact: dict[str, Any]) -> bool:
    validate_generation_artifact(artifact)
    return any(_metadata_tests(meta) for meta in artifact["prompt_metadata"])


def _run_one_fixture(sample: dict[str, Any], timeout: float) -> dict[str, Any]:
    meta = sample["metadata"]
    tests = _metadata_tests(meta)
    if not tests:
        return {**sample, "passed": False, "status": "missing_tests", "stdout": "", "stderr": ""}
    with tempfile.TemporaryDirectory(prefix="opd-code-eval-") as tmpdir:
        program = Path(tmpdir) / "submission_test.py"
        program.write_text(
            sample["extracted_code"].rstrip() + "\n\n" + tests.rstrip() + "\n",
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [sys.executable, str(program)],
                cwd=tmpdir,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            passed = completed.returncode == 0
            status = "passed" if passed else "failed"
            return {
                **sample,
                "passed": passed,
                "status": status,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                **sample,
                "passed": False,
                "status": "timeout",
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            }


def run_fixture_tests(
    artifact: dict[str, Any],
    *,
    workers: int = 1,
    timeout: float = 10.0,
    allow_unsafe_code_execution: bool = False,
) -> list[dict[str, Any]]:
    """Run fixture tests in bounded subprocesses, one subprocess per sample.

    This executes generated Python code on the local host. It is disabled by
    default for public-safe behavior; callers must explicitly opt in for
    trusted/internal generations or use the sandboxed grading workflow.
    """
    validate_generation_artifact(artifact)
    if not allow_unsafe_code_execution:
        raise CodeEvalError(
            "Inline fixture code execution is disabled by default. Use the "
            "sandboxed grader for untrusted generations, or pass "
            "allow_unsafe_code_execution=True only for trusted code."
        )
    warnings.warn(
        "code_eval.run_fixture_tests: executing generated Python code on the "
        "local host. Prefer the sandboxed grader for untrusted generations.",
        UserWarning,
        stacklevel=2,
    )
    samples = list(iter_code_samples(artifact))
    max_workers = max(1, min(int(workers or 1), max(1, len(samples))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_one_fixture, sample, timeout) for sample in samples]
        return [future.result() for future in futures]


def _percent(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f * 100.0 if 0.0 <= f <= 1.0 else f


def _metrics_from_sample_results(
    artifact: dict[str, Any], sample_results: list[dict[str, Any]]
) -> dict[str, Any]:
    by_problem: dict[Any, list[bool]] = {}
    for result in sample_results:
        by_problem.setdefault(result["problem_id"], []).append(bool(result.get("passed")))
    n_tasks = len(artifact.get("prompt_metadata", []))
    n_samples_per_task = [len(r) for r in artifact.get("responses", [])]
    n_samples = max(n_samples_per_task) if n_samples_per_task else 0

    def pass_at(k: int) -> float | None:
        if n_samples < k:
            return None
        if n_tasks == 0:
            return 0.0
        return 100.0 * sum(any(vals[:k]) for vals in by_problem.values()) / n_tasks

    if n_tasks == 0:
        sample_accuracy = 0.0
    else:
        sample_accuracy = 100.0 * sum(
            (sum(vals) / len(vals)) if vals else 0.0 for vals in by_problem.values()
        ) / n_tasks
    return {
        "pass_at_1": pass_at(1),
        "pass_at_4": pass_at(4),
        f"avg_at_{n_samples}": sample_accuracy if n_samples else 0.0,
        f"sample_accuracy_at_{n_samples}": sample_accuracy if n_samples else 0.0,
        "n_tasks": n_tasks,
        "n_samples": n_samples,
        "total_samples": sum(n_samples_per_task),
    }


def _parse_pass_metrics(text: str) -> dict[str, float]:
    metrics = {}
    for match in re.finditer(r"pass[@_]([0-9]+)[^0-9]+([0-9]+(?:\.[0-9]+)?)", text, re.I):
        metrics[f"pass_at_{match.group(1)}"] = _percent(match.group(2))
    return metrics


def _estimate_pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k estimate used by code-generation benchmarks."""
    if n < k:
        return None
    if n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def _evalplus_results_path(samples_path: str | os.PathLike[str]) -> Path:
    path = Path(samples_path)
    return path.with_name(f"{path.stem}_eval_results.json")


def _evalplus_generation_passed(generation: dict[str, Any]) -> bool:
    # HumanEval+/MBPP+ headline scores are based on the stricter plus tests.
    # Fall back to base_status only for compatibility with non-plus EvalPlus
    # payloads that do not include plus_status.
    status = generation.get("plus_status")
    if status is None:
        status = generation.get("status", generation.get("base_status"))
    return str(status).lower() == "pass"


def _read_evalplus_eval_metrics(eval_path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(eval_path, encoding="utf-8") as f:
        payload = json.load(f)
    eval_payload = payload.get("eval") if isinstance(payload, dict) else None
    if not isinstance(eval_payload, dict):
        raise CodeEvalError(f"unexpected EvalPlus eval payload in {eval_path}")

    def pass_at(k: int) -> float | None:
        values = []
        for generations in eval_payload.values():
            if not isinstance(generations, list):
                continue
            n = len(generations)
            c = sum(
                _evalplus_generation_passed(g)
                for g in generations
                if isinstance(g, dict)
            )
            estimate = _estimate_pass_at_k(n, c, k)
            if estimate is None:
                return None
            values.append(estimate)
        if not values:
            return None
        return 100.0 * sum(values) / len(values)

    return {
        "pass_at_1": pass_at(1),
        "pass_at_4": pass_at(4),
    }


def run_evalplus_official(
    samples_path: str, benchmark: str, *, workers: int | None = None, timeout: float | None = None
) -> dict[str, Any]:
    """Run EvalPlus in a subprocess and parse pass metrics when available."""
    dataset = {"humaneval_plus": "humaneval", "mbpp_plus": "mbpp"}[benchmark]
    cmd = [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", samples_path]
    if workers:
        cmd.extend(["--parallel", str(workers)])
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise CodeEvalError(
            "EvalPlus is required for official HumanEval+/MBPP+ scoring; install evalplus "
            "or provide fixture tests in the generation artifact."
        ) from exc
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise CodeEvalError(
            "EvalPlus official runner failed with exit code "
            f"{result.returncode}. Command: {' '.join(cmd)}\n{output}"
        )
    metrics = _parse_pass_metrics(output)
    eval_path = _evalplus_results_path(samples_path)
    if eval_path.exists():
        metrics.update(_read_evalplus_eval_metrics(eval_path))
    metrics.update({"official_runner": "evalplus"})
    return metrics


def _lcb_runner_cwd() -> str | None:
    try:
        import importlib.util

        spec = importlib.util.find_spec("lcb_runner")
    except Exception:
        return None
    locations = list(spec.submodule_search_locations or []) if spec else []
    if not locations:
        return None
    return str(Path(locations[0]).parent)


def _pass_at_k_from_lcb_results(results: dict[str, Any], k: int) -> float | None:
    values = []
    for generations in results.values():
        total = len(generations)
        correct = sum(all(test_result > 0 for test_result in generation) for generation in generations)
        estimate = _estimate_pass_at_k(total, correct, k)
        if estimate is None:
            return None
        values.append(estimate)
    if not values:
        return None
    return 100.0 * sum(values) / len(values)


def _read_lcb_eval_metrics(eval_path: Path) -> dict[str, Any]:
    with eval_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list) or len(payload) < 2:
        raise CodeEvalError(f"unexpected LiveCodeBench eval payload in {eval_path}")
    metrics = payload[0] if isinstance(payload[0], dict) else {}
    results = payload[1] if isinstance(payload[1], dict) else {}
    parsed: dict[str, Any] = {}
    parsed["pass_at_1"] = (
        _percent(metrics.get("pass@1"))
        if metrics.get("pass@1") is not None
        else _pass_at_k_from_lcb_results(results, 1)
    )
    parsed["pass_at_4"] = _pass_at_k_from_lcb_results(results, 4)
    parsed["official_runner"] = "livecodebench"
    return parsed


def _lcb_hub_loader_patch_script() -> str:
    repo = json.dumps(_LCB_HF_REPO)
    files = json.dumps(_LCB_TEST_FILES)
    return f"""
import json, re, sys
from datetime import datetime
from huggingface_hub import hf_hub_download
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
import lcb_runner.benchmarks.code_generation as code_generation
import lcb_runner.benchmarks as benchmarks
import lcb_runner.runner.scenario_router as scenario_router

LCB_REPO = {repo}
LCB_TEST_FILES = {files}

def _files_for_release(release):
    if release in {{"release_latest", "latest"}}:
        return list(LCB_TEST_FILES)
    match = re.fullmatch(r"release_v([1-6])", release)
    if match:
        return list(LCB_TEST_FILES[: int(match.group(1))])
    match = re.fullmatch(r"v([1-6])", release)
    if match:
        return [LCB_TEST_FILES[int(match.group(1)) - 1]]
    match = re.fullmatch(r"v([1-6])_v([1-6])", release)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end:
            return list(LCB_TEST_FILES[start - 1:end])
    raise ValueError(f"unsupported LiveCodeBench release {{release!r}}")

def _load_from_hub(release_version="release_v1", start_date=None, end_date=None):
    rows = []
    p_start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    p_end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    for filename in _files_for_release(release_version):
        path = hf_hub_download(LCB_REPO, filename, repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                problem = CodeGenerationProblem(**json.loads(line))
                if p_start is not None and problem.contest_date < p_start:
                    continue
                if p_end is not None and problem.contest_date > p_end:
                    continue
                rows.append(problem)
    print(f"Loaded {{len(rows)}} problems")
    return rows

code_generation.load_code_generation_dataset = _load_from_hub
benchmarks.load_code_generation_dataset = _load_from_hub
scenario_router.load_code_generation_dataset = _load_from_hub
""".strip()


def run_lcb_official(
    samples_path: str,
    *,
    release: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run LiveCodeBench custom evaluator in a subprocess."""
    eval_path = Path(str(samples_path).removesuffix(".json") + "_codegeneration_output_eval.json")
    runner = _lcb_hub_loader_patch_script() + "\n" + r"""
import lcb_runner.runner.custom_evaluator as custom_evaluator

original_argv = sys.argv[:]
sample_path_arg = original_argv[1]
release_arg = original_argv[2]
start_date_arg = original_argv[3]
end_date_arg = original_argv[4]
sys.argv = [
    "custom_evaluator",
    "--custom_output_file", sample_path_arg,
    "--scenario", "codegeneration",
]
if release_arg != "":
    sys.argv.extend(["--release_version", release_arg])
if start_date_arg != "":
    sys.argv.extend(["--start_date", start_date_arg])
if end_date_arg != "":
    sys.argv.extend(["--end_date", end_date_arg])
custom_evaluator.main()
"""
    cmd = [
        sys.executable,
        "-c",
        runner,
        samples_path,
        release or "release_v6",
        date_start or "",
        date_end or "",
    ]
    result = subprocess.run(
        cmd, text=True, capture_output=True, timeout=timeout, cwd=_lcb_runner_cwd()
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise CodeEvalError(
            "LiveCodeBench official runner failed with exit code "
            f"{result.returncode}. Command: {' '.join(cmd[:3])} <runner> {samples_path}\n{output}"
        )
    metrics = _read_lcb_eval_metrics(eval_path) if eval_path.exists() else {}
    metrics.update(_parse_pass_metrics(output))
    metrics.setdefault("official_runner", "livecodebench")
    return metrics


def _default_paths(
    output_dir: str | os.PathLike[str] | None,
    benchmark: str,
    generations_path: str | os.PathLike[str] | None,
) -> tuple[str, str]:
    if output_dir is None:
        base_dir = os.path.dirname(str(generations_path)) if generations_path else "."
    else:
        base_dir = str(output_dir)
    os.makedirs(base_dir or ".", exist_ok=True)
    samples_ext = "json" if benchmark == "lcb_v6" else "jsonl"
    return (
        os.path.join(base_dir, f"{benchmark}_samples.{samples_ext}"),
        os.path.join(base_dir, f"{benchmark}_results.json"),
    )


def score_code(
    generation_result: dict[str, Any],
    *,
    benchmark: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    workers: int = 1,
    timeout: float = 10.0,
    generations_path: str | os.PathLike[str] | None = None,
    samples_path: str | os.PathLike[str] | None = None,
    results_path: str | os.PathLike[str] | None = None,
    allow_unsafe_code_execution: bool = False,
) -> dict[str, Any]:
    """Score a code generation artifact.

    Fixture artifacts with per-problem tests can run locally in bounded CPU
    subprocesses only when ``allow_unsafe_code_execution`` is true. Benchmark
    artifacts without tests are converted to official runner sample files and
    delegated to EvalPlus/LiveCodeBench subprocesses.
    """
    artifact = validate_generation_artifact(generation_result)
    benchmark = normalize_benchmark(benchmark or artifact.get("benchmark"))
    default_samples, default_results = _default_paths(output_dir, benchmark, generations_path)
    samples_path = str(samples_path or default_samples)
    results_path = str(results_path or default_results)
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    if benchmark == "lcb_v6":
        write_lcb_samples(artifact, samples_path)
    else:
        write_evalplus_samples(artifact, samples_path)

    base_metrics = {
        "eval_type": "code",
        "dataset": artifact.get("dataset") or benchmark,
        "benchmark": benchmark,
        "release": artifact.get("release"),
        "date_window": artifact.get("date_window") or {"start": None, "end": None},
        "generations_path": str(generations_path) if generations_path else None,
        "samples_path": samples_path,
        "results_path": results_path,
    }

    if artifact_has_fixture_tests(artifact):
        sample_results = run_fixture_tests(
            artifact,
            workers=workers,
            timeout=timeout,
            allow_unsafe_code_execution=allow_unsafe_code_execution,
        )
        metrics = _metrics_from_sample_results(artifact, sample_results)
        payload = {"metrics": {**base_metrics, **metrics}, "sample_results": sample_results}
    elif benchmark in {"humaneval_plus", "mbpp_plus"}:
        metrics = run_evalplus_official(samples_path, benchmark, workers=workers, timeout=None)
        metrics = {"pass_at_1": metrics.get("pass_at_1"), "pass_at_4": metrics.get("pass_at_4"), **metrics}
        metrics.update({"n_tasks": len(artifact["prompt_metadata"]), "n_samples": max((len(r) for r in artifact["responses"]), default=0)})
        payload = {"metrics": {**base_metrics, **metrics}}
    else:
        date_window = artifact.get("date_window") or {}
        metrics = run_lcb_official(
            samples_path,
            release=artifact.get("release"),
            date_start=date_window.get("start"),
            date_end=date_window.get("end"),
            timeout=None,
        )
        metrics = {"pass_at_1": metrics.get("pass_at_1"), "pass_at_4": metrics.get("pass_at_4"), **metrics}
        metrics.update({"n_tasks": len(artifact["prompt_metadata"]), "n_samples": max((len(r) for r in artifact["responses"]), default=0)})
        payload = {"metrics": {**base_metrics, **metrics}}

    summary = payload["metrics"]
    # Keep required pass_at_4 field even when only fewer samples were generated.
    summary.setdefault("pass_at_4", None)
    summary.setdefault("pass_at_1", None)
    summary.setdefault("n_tasks", len(artifact["prompt_metadata"]))
    summary.setdefault("n_samples", max((len(r) for r in artifact["responses"]), default=0))
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def load_code_benchmark_prompts(
    benchmark: str,
    *,
    release: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> list[dict[str, Any]]:
    """Load prompt metadata for supported code benchmarks using lazy deps."""
    benchmark = normalize_benchmark(benchmark)
    if benchmark in {"humaneval_plus", "mbpp_plus"}:
        return _load_evalplus_prompts(benchmark)
    return _load_lcb_prompts(release=release, date_start=date_start, date_end=date_end)


def _load_evalplus_prompts(benchmark: str) -> list[dict[str, Any]]:
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except Exception as exc:
        return _load_evalplus_prompts_from_hf(benchmark, import_error=exc)
    problems = get_human_eval_plus() if benchmark == "humaneval_plus" else get_mbpp_plus()
    rows = []
    for i, (task_id, problem) in enumerate(problems.items()):
        prompt = problem.get("prompt") or problem.get("text") or problem.get("question") or ""
        rows.append(CodeBenchmarkProblem(
            problem_id=i,
            prompt_id=_stable_id(benchmark, task_id, prompt),
            dataset=benchmark,
            benchmark=benchmark,
            task_id=canonical_evalplus_task_id(benchmark, task_id),
            entry_point=problem.get("entry_point"),
            prompt=str(prompt),
        ).to_metadata())
    return rows


def _load_evalplus_prompts_from_hf(
    benchmark: str, *, import_error: Exception | None = None
) -> list[dict[str, Any]]:
    """Load EvalPlus prompt metadata from data-only HF datasets.

    Some cluster environments intentionally do not install the optional
    ``evalplus`` package because these runs are generation-only.  The prompt
    data is also mirrored as normal Hugging Face datasets, which are cacheable
    by the same ``datasets.config.HF_DATASETS_CACHE`` path used elsewhere in
    OPD.  Do not include the public/private tests in metadata here: keeping the
    artifact test-free preserves the existing generation-only/official-runner
    behavior when ``execute_code`` is disabled/enabled.
    """
    dataset_name = _EVALPLUS_HF_DATASETS[benchmark]
    try:
        ds = _load_dataset(dataset_name, split="test")
    except Exception as exc:
        raise CodeEvalError(
            "EvalPlus is required to load HumanEval+/MBPP+ prompts, or the "
            f"{dataset_name!r} Hugging Face dataset must be available in the "
            "datasets cache."
        ) from (import_error or exc)

    rows = []
    for i, problem in enumerate(ds):
        prompt = problem.get("prompt") or problem.get("text") or problem.get("question") or ""
        task_id = canonical_evalplus_task_id(benchmark, problem.get("task_id", i))
        rows.append(CodeBenchmarkProblem(
            problem_id=i,
            prompt_id=_stable_id(benchmark, task_id, prompt),
            dataset=benchmark,
            benchmark=benchmark,
            task_id=task_id,
            entry_point=problem.get("entry_point"),
            prompt=str(prompt),
        ).to_metadata())
    return rows


def _load_lcb_prompts(
    *, release: str | None = None, date_start: str | None = None, date_end: str | None = None
) -> list[dict[str, Any]]:
    release = release or "release_v6"
    try:
        ds = _load_dataset("livecodebench/code_generation_lite", release, split="test")
    except Exception:
        return _load_lcb_prompts_from_hf_hub(
            release=release, date_start=date_start, date_end=date_end,
        )
    return _lcb_rows_to_metadata(
        ds, release=release, date_start=date_start, date_end=date_end,
    )


def _lcb_rows_to_metadata(
    rows_iterable,
    *,
    release: str,
    date_start: str | None,
    date_end: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for row in rows_iterable:
        contest_date = row.get("contest_date")
        if not _in_date_window(contest_date, date_start, date_end):
            continue
        title = row.get("question_title") or ""
        content = row.get("question_content") or ""
        starter = row.get("starter_code") or ""
        prompt_parts = [part for part in [title.strip(), content.strip()] if part]
        if starter.strip():
            prompt_parts.append("Starter code:\n" + starter.strip())
        prompt = "\n\n".join(prompt_parts)
        question_id = str(row.get("question_id"))
        rows.append(CodeBenchmarkProblem(
            problem_id=len(rows),
            prompt_id=_stable_id("lcb_v6", release, question_id, prompt),
            dataset="lcb_v6",
            benchmark="lcb_v6",
            question_id=question_id,
            contest_date=contest_date,
            prompt=prompt,
        ).to_metadata())
    if not rows:
        raise CodeEvalError(
            "LiveCodeBench prompt filter matched 0 rows "
            f"for release={release!r}, date_start={date_start!r}, date_end={date_end!r}"
        )
    return rows


def _lcb_files_for_release(release: str) -> list[str]:
    """Match the file selection in LiveCodeBench's HF dataset script."""
    if release in {"release_latest", "latest"}:
        return list(_LCB_TEST_FILES)

    match = re.fullmatch(r"release_v([1-6])", release)
    if match:
        return list(_LCB_TEST_FILES[: int(match.group(1))])

    match = re.fullmatch(r"v([1-6])", release)
    if match:
        return [_LCB_TEST_FILES[int(match.group(1)) - 1]]

    match = re.fullmatch(r"v([1-6])_v([1-6])", release)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end:
            return list(_LCB_TEST_FILES[start - 1:end])

    raise CodeEvalError(
        f"unsupported LiveCodeBench release {release!r}; expected release_v1..release_v6, "
        "release_latest, v1..v6, or ranges like v4_v6"
    )


def _lcb_processed_cache_path(
    *, release: str, date_start: str | None, date_end: str | None
) -> Path:
    """Return an OPD prompt-cache path under HF_DATASETS_CACHE.

    Newer ``datasets`` releases no longer execute dataset scripts, so the LCB
    fallback downloads the raw JSONL files via Hugging Face Hub and stores the
    filtered prompt metadata under the same datasets cache root OPD uses for
    ``load_dataset``.  That keeps repeated post-eval subprocesses offline and
    avoids rereading multi-GB LCB JSONL files every checkpoint.
    """
    try:
        from datasets import config as datasets_config
    except Exception as exc:
        raise CodeEvalError(
            "Hugging Face datasets is required to determine the LiveCodeBench cache path."
        ) from exc
    key = hashlib.sha1(
        json.dumps(
            {
                "repo": _LCB_HF_REPO,
                "release": release,
                "date_start": date_start,
                "date_end": date_end,
                "schema": 1,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (
        Path(datasets_config.HF_DATASETS_CACHE)
        / "opd_code_benchmarks"
        / "livecodebench_code_generation_lite"
        / f"{key}.jsonl"
    )


def _read_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_metadata_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _iter_lcb_hub_rows(release: str):
    for filename in _lcb_files_for_release(release):
        path = _hf_hub_download(_LCB_HF_REPO, filename, repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _load_lcb_prompts_from_hf_hub(
    *, release: str, date_start: str | None = None, date_end: str | None = None
) -> list[dict[str, Any]]:
    cache_path = _lcb_processed_cache_path(
        release=release, date_start=date_start, date_end=date_end,
    )
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return _read_metadata_jsonl(cache_path)

    rows = _lcb_rows_to_metadata(
        _iter_lcb_hub_rows(release),
        release=release,
        date_start=date_start,
        date_end=date_end,
    )
    _write_metadata_jsonl(cache_path, rows)
    return rows

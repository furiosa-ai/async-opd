#!/usr/bin/env python3
"""Batch-score saved OPD code generation artifacts.

This is the efficient non-Docker inner loop used by
``scripts/grade_code_generations_sandboxed.py``. It runs once, scans a result
tree for ``*.generations.json`` files, skips already-scored summaries, and calls
``opd.utils.code_eval.score_code`` directly instead of launching the evaluation
CLI once per artifact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opd.utils.code_eval import CodeEvalError, load_generation_artifact, score_code

_PRINT_LOCK = threading.Lock()


def log(message: str, *, file=sys.stdout) -> None:
    with _PRINT_LOCK:
        print(message, file=file, flush=True)


def read_first_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open(encoding="utf-8") as f:
            line = f.readline().strip()
        return json.loads(line) if line else None
    except Exception:
        return None


def has_score_metrics(item: dict[str, Any]) -> bool:
    return any(
        item.get(key) is not None
        for key in item
        if key.startswith(("pass_at_", "avg_at_", "sample_accuracy_at_"))
    )


def looks_scored(path: Path) -> bool:
    item = read_first_json(path)
    if not isinstance(item, dict):
        return False
    if item.get("score_skipped") is True:
        return False
    return has_score_metrics(item)


def output_paths(
    artifact_path: Path,
    *,
    results_dir: Path,
    scored_dir: Path,
    in_place: bool,
    benchmark: str | None = None,
) -> tuple[Path, Path, Path]:
    rel = artifact_path.relative_to(results_dir)
    base = artifact_path.name.removesuffix(".generations.json")
    out_dir = artifact_path.parent if in_place else scored_dir / rel.parent
    output_jsonl = out_dir / f"{base}.jsonl"
    samples_ext = "json" if benchmark == "lcb_v6" else "jsonl"
    samples_path = out_dir / f"{base}.samples.{samples_ext}"
    results_path = out_dir / f"{base}.results.json"
    return output_jsonl, samples_path, results_path


def path_matches_bench(path: Path, benches: set[str]) -> bool:
    if not benches:
        return True
    parts = set(path.parts)
    return any(bench in parts or bench in str(path) for bench in benches)


def atomic_write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
    os.replace(tmp, path)


def score_one(
    artifact_path: Path,
    *,
    results_dir: Path,
    scored_dir: Path,
    benches: set[str],
    in_place: bool,
    force: bool,
    workers: int,
    timeout: float,
) -> dict[str, Any]:
    source_summary = artifact_path.with_name(
        artifact_path.name.removesuffix(".generations.json") + ".jsonl"
    )
    staged_summary, _samples_unused, _results_unused = output_paths(
        artifact_path,
        results_dir=results_dir,
        scored_dir=scored_dir,
        in_place=in_place,
    )
    if not force and (looks_scored(source_summary) or looks_scored(staged_summary)):
        return {"status": "skipped", "artifact": str(artifact_path), "reason": "already_scored"}

    artifact = load_generation_artifact(artifact_path)
    benchmark = str(artifact.get("benchmark") or "")
    if benches and benchmark not in benches and not path_matches_bench(artifact_path, benches):
        return {"status": "skipped", "artifact": str(artifact_path), "reason": "bench_filter"}

    output_jsonl, samples_path, results_path = output_paths(
        artifact_path,
        results_dir=results_dir,
        scored_dir=scored_dir,
        in_place=in_place,
        benchmark=benchmark,
    )
    if not force and (looks_scored(output_jsonl) or looks_scored(source_summary)):
        return {"status": "skipped", "artifact": str(artifact_path), "reason": "already_scored"}

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    log(f"[score] {artifact_path} -> {output_jsonl}")
    metrics = score_code(
        artifact,
        benchmark=benchmark,
        output_dir=output_jsonl.parent,
        workers=workers,
        timeout=timeout,
        generations_path=artifact_path,
        samples_path=samples_path,
        results_path=results_path,
    )
    if metrics.get("score_skipped") is True:
        raise CodeEvalError(f"scorer returned score_skipped for {artifact_path}")
    if not has_score_metrics(metrics):
        raise CodeEvalError(f"scorer returned no pass/avg metrics for {artifact_path}: {metrics}")
    atomic_write_jsonl(output_jsonl, metrics)
    return {
        "status": "scored",
        "artifact": str(artifact_path),
        "output_jsonl": str(output_jsonl),
        "results_json": str(results_path),
        "benchmark": benchmark,
        "pass_at_1": metrics.get("pass_at_1"),
        "pass_at_4": metrics.get("pass_at_4"),
    }


def iter_artifacts(results_dir: Path, benches: set[str]) -> list[Path]:
    artifacts = []
    for path in sorted(results_dir.rglob("*.generations.json")):
        if ".code_eval_scored" in path.parts:
            continue
        if path_matches_bench(path, benches):
            artifacts.append(path)
    return artifacts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--scored-dir", type=Path, default=None)
    parser.add_argument("--bench", action="append", default=[])
    parser.add_argument("--workers", type=int, default=16, help="workers passed to each official scorer")
    parser.add_argument("--jobs", type=int, default=1, help="number of artifacts to score concurrently")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--keep-going", action="store_true", default=True)
    parser.add_argument("--fail-fast", dest="keep_going", action="store_false")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()
    scored_dir = (args.scored_dir or (results_dir / ".code_eval_scored")).resolve()
    benches = {str(bench) for bench in args.bench if str(bench)}
    if not results_dir.is_dir():
        raise SystemExit(f"results dir does not exist: {results_dir}")
    if not args.in_place:
        scored_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or (scored_dir / "score_manifest.jsonl")
    artifacts = iter_artifacts(results_dir, benches)
    log(
        f"[batch-score] artifacts={len(artifacts)} benches={','.join(sorted(benches)) or 'all'} "
        f"jobs={args.jobs} workers={args.workers} in_place={args.in_place} force={args.force}"
    )
    if not artifacts:
        return 0

    results: list[dict[str, Any]] = []
    failures = 0

    def run(path: Path) -> dict[str, Any]:
        try:
            return score_one(
                path,
                results_dir=results_dir,
                scored_dir=scored_dir,
                benches=benches,
                in_place=args.in_place,
                force=args.force,
                workers=max(1, args.workers),
                timeout=args.timeout,
            )
        except Exception as exc:
            return {
                "status": "failed",
                "artifact": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    jobs = max(1, min(int(args.jobs or 1), len(artifacts)))
    if jobs == 1:
        for artifact in artifacts:
            item = run(artifact)
            results.append(item)
            if item["status"] == "failed":
                failures += 1
                log(f"[failed] {artifact}: {item['error']}", file=sys.stderr)
                if not args.keep_going:
                    break
            elif item["status"] == "skipped":
                log(f"[skip] {artifact}: {item.get('reason')}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            future_to_path = {pool.submit(run, path): path for path in artifacts}
            for future in concurrent.futures.as_completed(future_to_path):
                artifact = future_to_path[future]
                item = future.result()
                results.append(item)
                if item["status"] == "failed":
                    failures += 1
                    log(f"[failed] {artifact}: {item['error']}", file=sys.stderr)
                    if not args.keep_going:
                        break
                elif item["status"] == "skipped":
                    log(f"[skip] {artifact}: {item.get('reason')}")

    if manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    log("[batch-score] summary " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

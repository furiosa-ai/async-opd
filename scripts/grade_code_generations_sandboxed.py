#!/usr/bin/env python3
"""Grade saved code-eval generations inside a locked-down Docker sandbox.

Default target is ``results/``. For routine use, pass ``--results-dir`` to a
specific run directory or to a parent directory containing code-eval artifacts.

The wrapper builds/reuses one Docker image with EvalPlus and, by default, a
minimal LiveCodeBench custom-evaluator install. It then runs the batch scorer in
an isolated container with no runtime network, no GPUs, dropped capabilities, a
read-only root filesystem, source/results mounted read-only, and only a staging
score/cache directory mounted writable. The generated code is scored by
``scripts/score_code_generation_artifacts.py`` in one Python process, so we avoid
spawning one full eval CLI process per artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import stat
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results"
METRIC_PREFIXES = ("pass_at_", "avg_at_", "sample_accuracy_at_")
STEP_SUMMARY_RE = re.compile(r"^step_(\d+)\.jsonl$")

DOCKERFILE = r"""
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG INSTALL_LCB=1
ARG LCB_INSTALL_MODE=minimal
ARG EXTRA_PIP=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="" \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      bash ca-certificates coreutils git tini \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir \
      datasets huggingface_hub numpy pandas tqdm pebble anthropic \
      psutil termcolor fire rich appdirs wget tempdir multipledispatch \
      tree-sitter tree-sitter-python attrs annotated-types \
 && python -m pip install --no-cache-dir --no-deps evalplus \
 && if [ -n "$EXTRA_PIP" ]; then python -m pip install --no-cache-dir $EXTRA_PIP; fi \
 && if [ "$INSTALL_LCB" = "1" ] && [ "$LCB_INSTALL_MODE" = "full" ]; then \
      python -m pip install --no-cache-dir 'git+https://github.com/LiveCodeBench/LiveCodeBench.git'; \
    elif [ "$INSTALL_LCB" = "1" ]; then \
      python -m pip install --no-cache-dir --no-deps 'git+https://github.com/LiveCodeBench/LiveCodeBench.git'; \
      tmp_lcb=$(mktemp -d); \
      git clone --depth=1 https://github.com/LiveCodeBench/LiveCodeBench.git "$tmp_lcb"; \
      dst=$(python -c 'import pathlib, site; print(pathlib.Path(site.getsitepackages()[0]) / "lcb_runner")'); \
      mkdir -p "$dst"; \
      cp -a "$tmp_lcb/lcb_runner/." "$dst/"; \
      rm -rf "$tmp_lcb"; \
      python -c "import importlib.util, pathlib, site; p=pathlib.Path(site.getsitepackages()[0])/'torch.py'; p.write_text('class _Cuda:\\n    @staticmethod\\n    def device_count():\\n        return 0\\ncuda = _Cuda()\\n', encoding='utf-8') if importlib.util.find_spec('torch') is None else None"; \
    fi

WORKDIR /workspace/src
ENTRYPOINT ["/usr/bin/tini", "--"]
""".strip() + "\n"

PREFETCH_SCRIPT = r"""
set -euo pipefail
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
python - <<'PY'
from evalplus.data import get_human_eval_plus, get_mbpp_plus
get_human_eval_plus()
get_mbpp_plus()
print("[prefetch] evalplus ok")
PY
if [[ "${INSTALL_LCB:-0}" == "1" ]]; then
  python - <<'PY' || true
from huggingface_hub import snapshot_download
snapshot_download("livecodebench/code_generation_lite", repo_type="dataset")
print("[prefetch] livecodebench/code_generation_lite snapshot ok")
PY
fi
""".strip()

CONTAINER_SCRIPT = r"""
set -euo pipefail

bench_args=()
if [[ -n "${BENCH_FILTER:-}" ]]; then
  IFS=',' read -r -a benches <<< "$BENCH_FILTER"
  for bench in "${benches[@]}"; do
    [[ -n "$bench" ]] && bench_args+=(--bench "$bench")
  done
fi
force_arg=()
[[ "${FORCE:-0}" == "1" ]] && force_arg=(--force)
in_place_arg=()
[[ "${IN_PLACE:-0}" == "1" ]] && in_place_arg=(--in-place)

python /workspace/src/scripts/score_code_generation_artifacts.py \
  --results-dir /workspace/results \
  --scored-dir /workspace/scored \
  --workers "${CODE_WORKERS:-16}" \
  --jobs "${SCORE_JOBS:-1}" \
  --timeout "${CODE_TIMEOUT:-10}" \
  --keep-going \
  "${bench_args[@]}" \
  "${force_arg[@]}" \
  "${in_place_arg[@]}"
""".strip()


class LCBModeAction(argparse.Action):
    """argparse action whose last occurrence wins for LCB mode flags."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        install_lcb, mode = self.const
        setattr(namespace, "install_lcb", install_lcb)
        setattr(namespace, "lcb_install_mode", mode)


def abs_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def quote_cmd(cmd: Sequence[str | os.PathLike[str]]) -> str:
    return shlex.join(str(part) for part in cmd)


def run_cmd(cmd: Sequence[str | os.PathLike[str]], *, dry_run: bool) -> None:
    print(quote_cmd(cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], check=True)


def has_score_metrics(item: dict[str, Any]) -> bool:
    return any(item.get(key) is not None for key in item if key.startswith(METRIC_PREFIXES))


def read_first_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        line = handle.readline().strip()
    if not line:
        raise ValueError(f"empty scored summary: {path}")
    item = json.loads(line)
    if not isinstance(item, dict):
        raise ValueError(f"first JSONL item is not an object: {path}")
    return item


def validate_scored_summary(path: Path) -> None:
    item = read_first_json(path)
    if item.get("score_skipped") is True:
        raise ValueError(f"refusing to merge unscored summary: {path}")
    if not has_score_metrics(item):
        raise ValueError(f"refusing to merge non-metric JSONL: {path}")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def eval_dedupe_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    date_window = entry.get("date_window") or {}
    return (
        entry.get("step"),
        entry.get("eval_type", "math"),
        entry.get("dataset"),
        entry.get("benchmark"),
        entry.get("release"),
        date_window.get("start"),
        date_window.get("end"),
    )


def step_from_summary_path(path: Path) -> int | None:
    match = STEP_SUMMARY_RE.match(path.name)
    return int(match.group(1)) if match else None


def run_dir_for_eval_summary(path: Path) -> Path | None:
    # Expected code-eval layout:
    #   results/<experiment>/<run>/eval/<benchmark_name>/step_N.jsonl
    if len(path.parents) < 3 or path.parent.parent.name != "eval":
        return None
    return path.parents[2]


def code_eval_entry_from_summary(summary_path: Path) -> dict[str, Any] | None:
    step = step_from_summary_path(summary_path)
    run_dir = run_dir_for_eval_summary(summary_path)
    if step is None or run_dir is None:
        return None
    try:
        metrics = read_first_json(summary_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if metrics.get("eval_type") != "code":
        return None
    if metrics.get("score_skipped") is True:
        return None
    if not has_score_metrics(metrics):
        return None

    base = summary_path.name.removesuffix(".jsonl")
    generation_path = summary_path.with_name(f"{base}.generations.json")
    entry = dict(metrics)
    entry["type"] = "eval"
    entry["step"] = step
    entry.setdefault("eval_type", "code")
    # The detailed runner outputs remain next to the staged score artifacts.
    # Avoid putting container-internal /workspace/scored paths into eval.jsonl.
    entry.pop("samples_path", None)
    entry.pop("results_path", None)
    entry["generations_path"] = display_path(generation_path)
    return entry


def iter_code_eval_summary_entries(results_dir: Path):
    for summary_path in sorted(results_dir.rglob("eval/*/step_*.jsonl")):
        if ".code_eval_scored" in summary_path.parts:
            continue
        if summary_path.name.endswith(".samples.jsonl"):
            continue
        entry = code_eval_entry_from_summary(summary_path)
        if entry is not None:
            run_dir = run_dir_for_eval_summary(summary_path)
            if run_dir is not None:
                yield run_dir, entry


def load_existing_eval_jsonl(path: Path) -> list[tuple[dict[str, Any] | None, str]]:
    if not path.exists():
        return []
    entries: list[tuple[dict[str, Any] | None, str]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                entries.append((None, line))
                continue
            entries.append((item if isinstance(item, dict) else None, line))
    return entries


def write_eval_jsonl(path: Path, preserved_lines: list[str], new_entries: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for line in preserved_lines:
            handle.write(line.rstrip("\n") + "\n")
        for entry in sorted(new_entries, key=eval_entry_sort_key):
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    os.replace(tmp, path)


def eval_entry_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    step = entry.get("step")
    step_key = step if isinstance(step, int) else -1
    return (
        step_key,
        str(entry.get("dataset") or ""),
        str(entry.get("benchmark") or ""),
        str(entry.get("release") or ""),
        str((entry.get("date_window") or {}).get("start") or ""),
        str((entry.get("date_window") or {}).get("end") or ""),
    )


def rebuild_code_eval_jsonl(*, results_dir: Path) -> int:
    """Rebuild each run's eval.jsonl code-eval index from scored step summaries.

    The per-step ``eval/<bench>/step_N.jsonl`` files remain the durable score
    summaries. ``eval.jsonl`` is a convenience index for existing report and
    monitor tools. Existing non-code eval lines are preserved, while code eval
    lines for the same dedupe key are replaced idempotently.
    """
    entries_by_run: dict[Path, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for run_dir, entry in iter_code_eval_summary_entries(results_dir):
        entries_by_run.setdefault(run_dir, {})[eval_dedupe_key(entry)] = entry

    total_entries = 0
    for run_dir in sorted(entries_by_run):
        new_entries_by_key = entries_by_run[run_dir]
        eval_jsonl = run_dir / "eval.jsonl"
        preserved_lines: list[str] = []
        for item, raw_line in load_existing_eval_jsonl(eval_jsonl):
            if (
                isinstance(item, dict)
                and item.get("type") == "eval"
                and item.get("eval_type") == "code"
                and eval_dedupe_key(item) in new_entries_by_key
            ):
                continue
            preserved_lines.append(raw_line)
        new_entries = list(new_entries_by_key.values())
        write_eval_jsonl(eval_jsonl, preserved_lines, new_entries)
        print(f"[sandbox-grade] indexed {len(new_entries)} code eval rows in {eval_jsonl}")
        total_entries += len(new_entries)
    print(f"[sandbox-grade] eval_jsonl_indexed={total_entries}")
    return total_entries


def path_matches_bench(path: Path, benches: Sequence[str]) -> bool:
    if not benches:
        return True
    parts = set(path.parts)
    path_text = str(path)
    return any(bench in parts or bench in path_text for bench in benches)


def iter_generation_artifacts(results_dir: Path, benches: Sequence[str]) -> list[Path]:
    return [
        path
        for path in sorted(results_dir.rglob("*.generations.json"))
        if ".code_eval_scored" not in path.parts and path_matches_bench(path, benches)
    ]


def print_artifact_summary(
    *,
    results_dir: Path,
    scored_dir: Path,
    cache_dir: Path,
    args: argparse.Namespace,
    benches: Sequence[str],
    artifacts: Sequence[Path],
) -> None:
    total_bytes = sum(path.stat().st_size for path in artifacts)
    bench_filter = ",".join(benches)
    print(f"[sandbox-grade] results_dir={results_dir}")
    print(
        f"[sandbox-grade] scored_dir={scored_dir} cache_dir={cache_dir} "
        f"in_place={int(args.in_place)} merge={int(args.merge)}"
    )
    print(
        f"[sandbox-grade] image={args.image} build={int(args.build)} "
        f"install_lcb={int(args.install_lcb)} lcb_mode={args.lcb_install_mode} "
        f"prefetch={int(args.prefetch)} workers={args.workers} jobs={args.jobs}"
    )
    print(
        f"[sandbox-grade] artifacts={len(artifacts)} "
        f"bytes={total_bytes / 1024 / 1024 / 1024:.1f}GB benches={bench_filter or 'all'}"
    )
    print("[sandbox-grade] largest artifacts:")
    for path in sorted(artifacts, key=lambda item: item.stat().st_size, reverse=True)[: args.print_largest]:
        print(f"  {path.stat().st_size / 1024 / 1024:8.1f}MB  {path}")


def prompt_for_confirmation(*, dry_run: bool, yes: bool) -> None:
    if yes or dry_run:
        return
    if not sys.stdin.isatty():
        raise SystemExit("Refusing non-interactive grading without --yes.")
    print("\nThis will execute untrusted model-generated Python in Docker.")
    answer = input("Type 'grade' to continue: ")
    if answer != "grade":
        raise SystemExit("Cancelled.")


def ensure_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"directory is not writable by the host user: {path} ({exc})") from exc
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass


def safe_chmod_tree(path: Path, mode_file: int = 0o600, mode_dir: int = 0o700) -> None:
    """Tighten permissions after a run without following untrusted symlinks."""
    if not path.exists():
        return
    for root, dirs, files in os.walk(path, topdown=False, followlinks=False):
        root_path = Path(root)
        for name in files:
            child = root_path / name
            try:
                if child.is_symlink():
                    continue
                os.chmod(child, mode_file)
            except OSError:
                pass
        for name in dirs:
            child = root_path / name
            try:
                if child.is_symlink():
                    continue
                os.chmod(child, mode_dir)
            except OSError:
                pass
        try:
            if not root_path.is_symlink():
                os.chmod(root_path, mode_dir)
        except OSError:
            pass


def docker_build(args: argparse.Namespace, *, dry_run: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="opd-code-eval-build-") as tmpdir:
        dockerfile_path = Path(tmpdir) / "Dockerfile"
        dockerfile_path.write_text(DOCKERFILE, encoding="utf-8")
        cmd = [
            "docker",
            "build",
            "--build-arg",
            f"BASE_IMAGE={args.base_image}",
            "--build-arg",
            f"INSTALL_LCB={int(args.install_lcb)}",
            "--build-arg",
            f"LCB_INSTALL_MODE={args.lcb_install_mode}",
            "--build-arg",
            f"EXTRA_PIP={' '.join(args.extra_pip)}",
            "-t",
            args.image,
            tmpdir,
        ]
        print("[sandbox-grade] docker build command:")
        run_cmd(cmd, dry_run=dry_run)


def cache_env(cache_dir: Path) -> list[str]:
    return [
        "-e",
        "HOME=/workspace/cache/home",
        "-e",
        "XDG_CACHE_HOME=/workspace/cache/xdg",
        "-e",
        "HF_HOME=/workspace/cache/hf",
        "-e",
        "HF_HUB_CACHE=/workspace/cache/hf/hub",
        "-e",
        "HF_DATASETS_CACHE=/workspace/cache/hf/datasets",
        "-e",
        "HF_HUB_DISABLE_TELEMETRY=1",
        "-v",
        f"{cache_dir}:/workspace/cache:rw",
    ]


def docker_user_args() -> list[str]:
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


def docker_prefetch(args: argparse.Namespace, *, cache_dir: Path, dry_run: bool) -> None:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        *docker_user_args(),
        "--cpus",
        str(args.cpus),
        "--memory",
        args.memory,
        "--pids-limit",
        str(args.pids_limit),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        f"/tmp:rw,nosuid,nodev,size={args.tmpfs_size}",
        *cache_env(cache_dir),
        "-e",
        f"INSTALL_LCB={int(args.install_lcb)}",
        args.image,
        "bash",
        "-lc",
        PREFETCH_SCRIPT,
    ]
    print("[sandbox-grade] docker prefetch command:")
    run_cmd(cmd, dry_run=dry_run)


def docker_run_score(
    args: argparse.Namespace,
    *,
    results_dir: Path,
    scored_dir: Path,
    cache_dir: Path,
    benches: Sequence[str],
    dry_run: bool,
) -> None:
    network_args: list[str] = [] if args.allow_network else ["--network", "none"]
    if args.in_place:
        result_mount = [
            "-v",
            f"{results_dir}:/workspace/results:rw",
            "-v",
            f"{scored_dir}:/workspace/scored:rw",
        ]
    else:
        result_mount = [
            "-v",
            f"{results_dir}:/workspace/results:ro",
            "-v",
            f"{scored_dir}:/workspace/scored:rw",
        ]
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        *network_args,
        *docker_user_args(),
        "--cpus",
        str(args.cpus),
        "--memory",
        args.memory,
        "--pids-limit",
        str(args.pids_limit),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        f"/tmp:rw,nosuid,nodev,size={args.tmpfs_size}",
        "--shm-size",
        "2g",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        *cache_env(cache_dir),
        "-e",
        f"BENCH_FILTER={','.join(benches)}",
        "-e",
        f"FORCE={int(args.force)}",
        "-e",
        f"IN_PLACE={int(args.in_place)}",
        "-e",
        f"CODE_WORKERS={args.workers}",
        "-e",
        f"SCORE_JOBS={args.jobs}",
        "-e",
        f"CODE_TIMEOUT={args.timeout}",
        "-e",
        "KEEP_GOING=1",
        "-v",
        f"{ROOT}:/workspace/src:ro",
        *result_mount,
        "-w",
        "/workspace/src",
        args.image,
        "bash",
        "-lc",
        CONTAINER_SCRIPT,
    ]
    print("[sandbox-grade] docker run command:")
    run_cmd(cmd, dry_run=dry_run)


def merge_scored_outputs(*, scored_dir: Path, results_dir: Path) -> int:
    print(f"[sandbox-grade] merging scored step_*.jsonl files back into {results_dir}")
    merged = 0
    for scored in sorted(scored_dir.rglob("step_*.jsonl")):
        if scored.name.endswith(".samples.jsonl"):
            continue
        try:
            rel = scored.relative_to(scored_dir)
        except ValueError as exc:
            raise RuntimeError(f"scored path escaped staging dir: {scored}") from exc
        if scored.is_symlink():
            raise RuntimeError(f"refusing to merge symlink: {scored}")
        st = os.stat(scored, follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode):
            continue
        validate_scored_summary(scored)
        dest = results_dir / rel
        if dest.exists() and dest.is_symlink():
            raise RuntimeError(f"refusing to overwrite symlink destination: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scored, dest)
        print(f"[merge] {dest}")
        merged += 1
    print(f"[sandbox-grade] merged={merged}")
    return merged


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=textwrap.dedent(
            """
            Examples:
              python scripts/grade_code_generations_sandboxed.py --results-dir results/my_run --dry-run
              python scripts/grade_code_generations_sandboxed.py --results-dir results/my_run --workers 32 --jobs 1 --yes
              python scripts/grade_code_generations_sandboxed.py --results-dir results/my_run --bench humaneval_plus --bench mbpp_plus --workers 32 --yes
              python scripts/grade_code_generations_sandboxed.py --results-dir results/my_run --bench lcb_v6 --workers 16 --yes
              python scripts/grade_code_generations_sandboxed.py --results-dir results/my_run --no-build --force --yes
              python scripts/grade_code_generations_sandboxed.py --results-dir results --index-only

            Safety model:
              The container still executes untrusted Python, but with no network by default,
              no GPU devices, dropped Linux capabilities, no-new-privileges, read-only source,
              read-only root filesystem, and only the staged score/cache directories mounted
              writable unless --in-place is requested.
            """
        ),
    )
    parser.set_defaults(build=True, install_lcb=True, lcb_install_mode="minimal", prefetch=True, merge=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--image", default="opd-code-eval:latest")
    parser.add_argument("--scored-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--base-image", default="python:3.11-slim")
    parser.add_argument("--build", dest="build", action="store_true")
    parser.add_argument("--no-build", dest="build", action="store_false")
    parser.add_argument("--with-lcb", nargs=0, action=LCBModeAction, const=(True, "minimal"))
    parser.add_argument("--full-lcb", nargs=0, action=LCBModeAction, const=(True, "full"))
    parser.add_argument("--without-lcb", nargs=0, action=LCBModeAction, const=(False, "none"))
    parser.add_argument("--prefetch", dest="prefetch", action="store_true")
    parser.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    parser.add_argument("--extra-pip", action="append", default=[])
    parser.add_argument("--bench", action="append", default=[])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cpus", default="16")
    parser.add_argument("--memory", default="32g")
    parser.add_argument("--pids-limit", type=int, default=2048)
    parser.add_argument("--tmpfs-size", default="8g")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--no-merge", dest="merge", action="store_false")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Rebuild run eval.jsonl indexes from existing scored code step summaries and exit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--print-largest", type=int, default=10, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> tuple[Path, Path, Path, list[str]]:
    benches = [bench for bench in args.bench if bench]
    if not args.install_lcb:
        if not benches:
            benches = ["humaneval_plus", "mbpp_plus"]
            print(
                "[sandbox-grade] --without-lcb selected; defaulting benchmark filter "
                "to humaneval_plus,mbpp_plus"
            )
        elif any("lcb" in bench for bench in benches):
            raise SystemExit("LCB benchmark filters require --with-lcb or --full-lcb.")

    results_dir = abs_path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory does not exist: {results_dir}")
    scored_dir = abs_path(args.scored_dir) if args.scored_dir else results_dir / ".code_eval_scored"
    cache_dir = abs_path(args.cache_dir) if args.cache_dir else scored_dir / "cache"
    return results_dir, scored_dir, cache_dir, benches


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir, scored_dir, cache_dir, benches = normalize_args(args)
    if args.index_only:
        rebuild_code_eval_jsonl(results_dir=results_dir)
        return 0

    artifacts = iter_generation_artifacts(results_dir, benches)
    if not artifacts:
        raise SystemExit(f"No artifacts matched benchmark filter: {','.join(benches) or '<none>'}")
    print_artifact_summary(
        results_dir=results_dir,
        scored_dir=scored_dir,
        cache_dir=cache_dir,
        args=args,
        benches=benches,
        artifacts=artifacts,
    )
    prompt_for_confirmation(dry_run=args.dry_run, yes=args.yes)

    if not args.dry_run:
        ensure_writable_dir(scored_dir)
        ensure_writable_dir(cache_dir)

    if args.build:
        docker_build(args, dry_run=args.dry_run)
    if args.prefetch:
        docker_prefetch(args, cache_dir=cache_dir, dry_run=args.dry_run)
    run_error: subprocess.CalledProcessError | None = None
    try:
        docker_run_score(
            args,
            results_dir=results_dir,
            scored_dir=scored_dir,
            cache_dir=cache_dir,
            benches=benches,
            dry_run=args.dry_run,
        )
    except subprocess.CalledProcessError as exc:
        run_error = exc
        print(
            f"[sandbox-grade] scoring container failed with exit code {exc.returncode}; "
            "will still merge valid staged summaries",
            file=sys.stderr,
        )

    if args.dry_run:
        print("[sandbox-grade] dry-run complete; no container started")
        return 0

    if not args.in_place:
        safe_chmod_tree(scored_dir)
        safe_chmod_tree(cache_dir)
    if args.merge:
        if not args.in_place:
            merge_scored_outputs(scored_dir=scored_dir, results_dir=results_dir)
        rebuild_code_eval_jsonl(results_dir=results_dir)
    return run_error.returncode if run_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

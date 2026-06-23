"""Reusable plotting helpers for OPD experiment analysis.

Usage (notebook)::

    from opd.utils.plots import apply_dark_theme, load_runs, plot_overview
    apply_dark_theme()
    data = load_runs(run_names, labels=RUN_LABELS, results_dir=Path("results"))
    fig = plot_overview(data)

Usage (headless / Slack bot)::

    import matplotlib
    matplotlib.use("Agg")
    from opd.utils.plots import apply_dark_theme, load_runs, plot_overview
    apply_dark_theme()
    data = load_runs(...)
    fig = plot_overview(data)
    fig.savefig("overview.png", dpi=150)

Do NOT import this module from ``opd/utils/__init__.py`` — it pulls in
matplotlib/seaborn which are heavy and unwanted in trainer processes.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# ── Data types ───────────────────────────────────────────────────


@dataclasses.dataclass
class RunData:
    """Container for all data needed by the plot functions."""

    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    rollout_df: pd.DataFrame
    run_palette: dict[str, str]  # run_label -> hex color
    run_order: list[str]  # sorted run labels
    totals: dict[str, list[dict]]  # run_label -> list of per-source totals dicts
    run_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    run_dirs: list[Path] = dataclasses.field(default_factory=list)
    results_dir: Path = Path("results")
    labels: dict[str, str] = dataclasses.field(default_factory=dict)


# ── Theme ────────────────────────────────────────────────────────

_DARK_THEME_RC = {
    "figure.facecolor": "#1e1e1e",
    "axes.facecolor": "#2a2a2a",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#ccc",
    "text.color": "#ccc",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "legend.facecolor": "#1e1e1e",
    "legend.edgecolor": "#444",
}


def apply_dark_theme() -> None:
    """Apply the dark seaborn theme used by run_summary.ipynb."""
    sns.set_theme(style="darkgrid", font_scale=0.9, rc=_DARK_THEME_RC)


# ── Ordered color palette ────────────────────────────────────────

# Colormap for smooth gradient across any number of runs (cool→warm).
_PALETTE_CMAP = "coolwarm"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_LABEL_PRESETS_PATH = PROJECT_ROOT / "notebooks" / "run_summary_labels.yaml"


# ── Helpers ──────────────────────────────────────────────────────


def _extract_staleness(run_name: str) -> int:
    """Extract staleness number from run name like '… /st4_r3_t4 …'."""
    m = re.search(r"/st(\d+)", run_name)
    return int(m.group(1)) if m else 999


def load_run_label_presets(path: Path | str = RUN_LABEL_PRESETS_PATH):
    """Load notebook run-label presets from YAML.

    YAML shape:

    ```yaml
    group_name:
      - run: path/to/run
        label: Display label
      - runs:
          - path/one
          - path/two
        label: Aggregated display label
    ```

    Returns a dict of ``group_name -> list[(run_spec, label)]`` where
    ``run_spec`` is either a string path or a tuple of paths, suitable for
    ``prepare_run_spec``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Run label presets YAML not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Run label presets YAML must contain a top-level mapping: {path}")

    presets: dict[str, list[tuple[str | tuple[str, ...], str]]] = {}
    for group, entries in data.items():
        if not isinstance(entries, list):
            raise ValueError(f"Preset group '{group}' must be a list of entries")
        normalized_entries: list[tuple[str | tuple[str, ...], str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Preset entry in group '{group}' must be a mapping")
            label = entry.get("label")
            run = entry.get("run")
            runs = entry.get("runs")
            if not isinstance(label, str) or not label:
                raise ValueError(f"Preset entry in group '{group}' is missing a non-empty 'label'")
            if isinstance(run, str) and run:
                normalized_entries.append((run, label))
                continue
            if isinstance(runs, list) and runs and all(isinstance(x, str) and x for x in runs):
                normalized_entries.append((tuple(runs), label))
                continue
            raise ValueError(
                f"Preset entry in group '{group}' must define either 'run' or non-empty 'runs'"
            )
        presets[str(group)] = normalized_entries
    return presets


# ── Data loading ─────────────────────────────────────────────────


@lru_cache(maxsize=256)
def _load_trace_cached(path_str: str) -> list[dict]:
    """Load trace_live.json (may be incomplete -- no closing bracket)."""
    path = Path(path_str)
    text = path.read_text().rstrip().rstrip(",")
    if not text.endswith("]"):
        text += "]"
    return json.loads(text)


def load_trace(path: Path) -> list[dict]:
    """Load trace_live.json with process-local caching."""
    return _load_trace_cached(str(path.resolve()))


@lru_cache(maxsize=256)
def _load_config_cached(run_dir_str: str, results_dir_str: str) -> dict | None:
    """Load the config for a run.

    Prefer the resolved config embedded in ``log.jsonl`` because many older
    experiment YAMLs use the legacy flat schema while plotting code expects the
    normalized internal config shape. Fall back to ``configs/<run_name>.yaml``
    when the log is missing or does not contain a config record.
    """
    run_dir = Path(run_dir_str)
    results_dir = Path(results_dir_str)
    log_path = run_dir / "log.jsonl"
    if log_path.exists():
        try:
            with open(log_path) as f:
                first = json.loads(f.readline())
            cfg = first.get("config")
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass

    import yaml

    run_name = run_dir.relative_to(results_dir).as_posix()
    cfg_path = results_dir.parent / "configs" / f"{run_name}.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_config(run_dir: Path, results_dir: Path) -> dict | None:
    """Load the config for a run with process-local caching."""
    return _load_config_cached(str(run_dir.resolve()), str(results_dir.resolve()))


def _get_train_batch_size(cfg: dict | None, default: int = 1) -> int:
    """Read train batch size from either normalized or legacy config shapes."""
    if not isinstance(cfg, dict):
        return default
    return (
        cfg.get("training", {}).get("data", {}).get("train_batch_size")
        or cfg.get("trainer", {}).get("batch_size")
        or default
    )


def _get_actor_mini_batch_size(cfg: dict | None, default: int = 0) -> int:
    """Read actor mini batch size from normalized or legacy config shapes."""
    if not isinstance(cfg, dict):
        return default
    return (
        cfg.get("training", {})
        .get("actor_rollout_ref", {})
        .get("actor", {})
        .get("mini_batch_size")
        or cfg.get("trainer", {}).get("mini_batch_size")
        or default
    )


def _get_staleness_or_stepoff(cfg: dict | None, run_name: str) -> int | None:
    """Extract the intended x-axis threshold for best-accuracy-vs-staleness plots."""
    if isinstance(cfg, dict):
        candidates = [
            cfg.get("trainer", {}).get("scheduler_step_off"),
            cfg.get("pipeline", {}).get("n_step_off", {}).get("step_off"),
            cfg.get("training", {}).get("trainer", {}).get("scheduler_step_off"),
            cfg.get("training", {}).get("pipeline", {}).get("n_step_off", {}).get("step_off"),
            cfg.get("trainer", {}).get("staleness_threshold"),
            cfg.get("pipeline", {}).get("fully_async", {}).get("staleness_threshold"),
            cfg.get("training", {}).get("trainer", {}).get("staleness_threshold"),
            cfg.get("training", {}).get("pipeline", {}).get("fully_async", {}).get("staleness_threshold"),
        ]
        for value in candidates:
            if value is not None:
                return int(value)

    m = re.search(r"/st(\d+)", run_name)
    if m:
        return int(m.group(1))
    m = re.search(r"/so(\d+)", run_name)
    if m:
        return int(m.group(1))
    return None


def _is_run_dir(p: Path) -> bool:
    """Check if a directory is a valid run (has trace or log)."""
    return p.is_dir() and ((p / "trace_live.json").exists() or (p / "log.jsonl").exists())


def _flatten_run_entries(entries) -> list[str]:
    """Flatten nested run/group entries into a plain list of string paths/globs."""
    flat: list[str] = []
    for entry in entries:
        if isinstance(entry, Path):
            flat.append(entry.as_posix())
        elif isinstance(entry, str):
            flat.append(entry)
        elif isinstance(entry, (list, tuple, set)):
            flat.extend(_flatten_run_entries(list(entry)))
        else:
            raise TypeError(f"Unsupported run entry type: {type(entry)!r}")
    return flat


def _normalize_label_map(labels) -> dict[str, str]:
    """Expand grouped label specs into a concrete run-path -> display-label map."""
    if not labels:
        return {}

    if hasattr(labels, "items"):
        items = labels.items()
    else:
        items = labels

    normalized: dict[str, str] = {}
    for key, label in items:
        for path in _flatten_run_entries([key]):
            normalized[path] = label
    return normalized


def prepare_run_spec(labels) -> tuple[list[str], dict[str, str]]:
    """Normalize notebook-friendly label specs into RUNS + concrete label map.

    Supports:
    - dict[path -> label]
    - dict[(path1, path2) -> label]
    - sequence of ``(paths, label)`` where ``paths`` may be a string or list/tuple
    """
    if not labels:
        return [], {}

    if hasattr(labels, "items"):
        run_entries = [k for k, _ in labels.items()]
    else:
        run_entries = [k for k, _ in labels]
    return _flatten_run_entries(run_entries), _normalize_label_map(labels)


def discover_runs(run_names: list[str], results_dir: Path | str) -> list[Path]:
    """Expand run names (including globs) into concrete run directories."""
    results_dir = Path(results_dir)
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for entry in _flatten_run_entries(run_names):
        if any(c in entry for c in "*?["):
            matches = sorted(results_dir.glob(entry))
            if not matches:
                continue
            for p in matches:
                if _is_run_dir(p):
                    rp = p.resolve()
                    if rp not in seen:
                        run_dirs.append(p)
                        seen.add(rp)
                elif p.is_dir():
                    for child in sorted(p.iterdir()):
                        if _is_run_dir(child):
                            rc = child.resolve()
                            if rc not in seen:
                                run_dirs.append(child)
                                seen.add(rc)
        else:
            p = results_dir / entry
            if not p.exists():
                continue
            if _is_run_dir(p):
                rp = p.resolve()
                if rp not in seen:
                    run_dirs.append(p)
                    seen.add(rp)
            else:
                for child in sorted(p.iterdir()):
                    if _is_run_dir(child):
                        rc = child.resolve()
                        if rc not in seen:
                            run_dirs.append(child)
                            seen.add(rc)
    return run_dirs


def _std0(values: pd.Series) -> float:
    """Population std with graceful empty/singleton handling."""
    vals = pd.Series(values).dropna()
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=0))


def _ordered_runs(df: pd.DataFrame, run_order: list[str]) -> list[str]:
    present = set(df["run"].unique()) if not df.empty and "run" in df.columns else set()
    ordered = [r for r in run_order if r in present]
    extras = sorted(present - set(ordered), key=_extract_staleness)
    return ordered + extras


def _aggregate_line_stats(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    smooth_window: int | None = None,
) -> pd.DataFrame:
    """Aggregate one label's replica lines into mean/std stats by x."""
    if df.empty:
        return pd.DataFrame(columns=[x_col, "mean", "std", "count"])

    replica_col = "source_run" if "source_run" in df.columns else None
    pieces: list[pd.DataFrame] = []

    if replica_col:
        replica_groups = df.groupby(replica_col, sort=False)
    else:
        replica_groups = [("__single__", df)]

    for replica, grp in replica_groups:
        g = grp[[x_col, y_col]].dropna().copy()
        if g.empty:
            continue
        g = g.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)
        if smooth_window and smooth_window > 1:
            g[y_col] = (
                g[y_col]
                .rolling(smooth_window, min_periods=1, center=True)
                .mean()
            )
        g["source_run"] = replica
        pieces.append(g)

    if not pieces:
        return pd.DataFrame(columns=[x_col, "mean", "std", "count"])

    combined = pd.concat(pieces, ignore_index=True)
    stats = (
        combined.groupby(x_col)[y_col]
        .agg(mean="mean", std=_std0, count="count")
        .reset_index()
        .sort_values(x_col)
    )
    return stats


def _plot_mean_std_lines(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    palette: dict[str, str],
    run_order: list[str],
    smooth_window: int | None = None,
    linewidth: float = 1.5,
    markers: list[str] | None = None,
    linestyles: list | None = None,
    markersize: float = 4,
    markevery: int | None = None,
    shade_alpha: float = 0.18,
) -> None:
    """Plot mean line + std band for each run label."""
    if df.empty:
        return

    markers = markers or ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "d"]
    linestyles = linestyles or ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    for j, run_name in enumerate(_ordered_runs(df, run_order)):
        grp = df[df["run"] == run_name]
        stats = _aggregate_line_stats(grp, x_col, y_col, smooth_window=smooth_window)
        if stats.empty:
            continue
        color = palette.get(run_name)
        ls = linestyles[j % len(linestyles)]
        mk = markers[j % len(markers)]
        x_vals = np.asarray(stats[x_col], dtype=float)
        mean_vals = np.asarray(stats["mean"], dtype=float)
        line = ax.plot(
            x_vals,
            mean_vals,
            linewidth=linewidth,
            label=run_name,
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=markersize,
            markevery=markevery,
        )
        if stats["count"].max() > 1:
            lower = np.asarray(stats["mean"] - stats["std"], dtype=float)
            upper = np.asarray(stats["mean"] + stats["std"], dtype=float)
            ax.fill_between(
                x_vals,
                lower,
                upper,
                alpha=shade_alpha,
                color=line[0].get_color(),
            )


def _format_mean_std(values, fmt: str = "{:.1f}") -> str:
    vals = pd.Series(list(values)).dropna()
    if vals.empty:
        return "—"
    mean = float(vals.mean())
    std = _std0(vals)
    return f"{fmt.format(mean)} ± {fmt.format(std)}" if len(vals) > 1 else fmt.format(mean)


def _build_eval_with_time(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Attach wall-time to eval rows while preserving per-source replicas."""
    rows: list[dict] = []
    group_cols = ["run", "source_run"] if "source_run" in eval_df.columns else ["run"]
    for keys, grp in eval_df.groupby(group_cols):
        run_name = keys[0] if isinstance(keys, tuple) else keys
        source_run = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else "__single__"
        train_grp = train_df[train_df["run"] == run_name]
        if "source_run" in train_df.columns:
            train_grp = train_grp[train_grp["source_run"] == source_run]
        for _, row in grp.iterrows():
            step = row["step"]
            if step == 0:
                t_min = 0.0
            else:
                match = train_grp[train_grp["step"] == step]
                t_min = match["cumtime_min"].values[0] if len(match) > 0 else np.nan
            rows.append(
                {
                    "run": run_name,
                    "source_run": source_run,
                    "step": step,
                    "samples_seen": row["samples_seen"],
                    "avg_at_n": row["avg_at_n"],
                    "dataset": row.get("dataset", "primary"),
                    "wall_min": t_min,
                }
            )
    return pd.DataFrame(rows)


def _efficiency_dataset_priority(dataset: str) -> int:
    """Tie-breaker for efficiency plots when datasets have equal coverage."""
    ds = str(dataset).lower()
    if ds == "primary":
        return 3
    if "aime_2024" in ds or "aime24" in ds:
        return 2
    if "aime_2025" in ds or "aime25" in ds or "aime_25" in ds:
        return 1
    return 0


def _select_efficiency_dataset(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> str | None:
    """Choose the dataset with enough time-series eval points for efficiency plots.

    Post-eval benchmark rows are often final-checkpoint-only.  If we pick the
    first dataset alphabetically, a final-only dataset such as AIME_25 can hide
    the primary AIME24 time series and leave AUC with no runs containing at
    least two time-aligned eval points.
    """
    if eval_df.empty or "dataset" not in eval_df.columns:
        return None

    best_dataset: str | None = None
    best_score: tuple[int, int, int, int, str] | None = None
    for dataset, ds_eval in eval_df.groupby("dataset"):
        ewt = _build_eval_with_time(train_df, ds_eval)
        timed = ewt.dropna(subset=["wall_min"]) if "wall_min" in ewt.columns else ewt
        if timed.empty:
            score = (0, 0, 0, _efficiency_dataset_priority(str(dataset)), str(dataset))
        else:
            group_cols = ["run", "source_run"] if "source_run" in timed.columns else ["run"]
            n_auc_groups = sum(len(grp) >= 2 for _, grp in timed.groupby(group_cols))
            score = (
                n_auc_groups,
                int(timed["step"].nunique()),
                len(timed),
                _efficiency_dataset_priority(str(dataset)),
                str(dataset),
            )
        if best_score is None or score > best_score:
            best_score = score
            best_dataset = str(dataset)
    return best_dataset


def _build_step_aligned(
    run_dir: Path, rollout_grp: pd.DataFrame, results_dir: Path
) -> pd.DataFrame:
    """Assign each rollout counter to its training step."""
    if rollout_grp.empty:
        return pd.DataFrame()
    tp = run_dir / "trace_live.json"
    events = load_trace(tp)
    x_events = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    if not x_events:
        return pd.DataFrame()
    t_min = min(e["ts"] for e in x_events)
    train_spans = sorted(
        [e for e in x_events if e.get("name") == "train"], key=lambda e: e["ts"]
    )
    boundaries: list[tuple[int, float, float]] = []
    prev_end = t_min
    for i, ts in enumerate(train_spans):
        boundaries.append((i + 1, prev_end / 1e6, (ts["ts"] + ts["dur"]) / 1e6))
        prev_end = ts["ts"] + ts["dur"]
    if not boundaries:
        return pd.DataFrame()

    starts = np.array([b[1] for b in boundaries], dtype=float)
    ends = np.array([b[2] for b in boundaries], dtype=float)
    abs_times = rollout_grp["t_s"].to_numpy(dtype=float) + (t_min / 1e6)
    idx = np.searchsorted(ends, abs_times, side="left")
    valid = idx < len(ends)
    valid &= abs_times >= starts[np.clip(idx, 0, len(starts) - 1)]
    if not np.any(valid):
        return pd.DataFrame()

    aligned = rollout_grp.loc[valid, ["running_reqs", "gen_tok_s"]].copy()
    aligned.insert(0, "step", idx[valid] + 1)
    aligned.reset_index(drop=True, inplace=True)
    return aligned


def _get_per_step_data_from_log(
    run_dir: Path, results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, dict]:
    """Fast path: build train_df from log.jsonl only (no trace parsing)."""
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, {}

    cfg = load_config(run_dir, results_dir)
    batch_size = _get_train_batch_size(cfg)

    rows, eval_data, eval_seen = [], [], set()
    cumtime_s = 0.0
    with open(log_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "train":
                iter_s = d.get("iter_seconds", 0)
                cumtime_s += iter_s
                n_tok = d.get("n_tokens", 0)
                rows.append({
                    "step": d["step"],
                    "samples_seen": d["step"] * batch_size,
                    "iter_s": iter_s,
                    "train_s": d.get("train_seconds", 0),
                    "gap_s": iter_s - d.get("train_seconds", 0),
                    "n_tokens": n_tok,
                    "s/sample_iter": iter_s / batch_size if batch_size else 0,
                    "s/sample_gen": d.get("generate_seconds", 0) / batch_size if batch_size else 0,
                    "tok/sample": n_tok / batch_size if batch_size else 0,
                    "tok_per_s": n_tok / iter_s if iter_s > 0 else 0,
                    "kl_loss": d.get("kl_loss"),
                    "avg_resp_len": d.get("avg_response_length"),
                    "staleness_mean": d.get("staleness_mean"),
                    "generate_s": d.get("generate_seconds"),
                    "teacher_s": d.get("teacher_seconds"),
                    "train_stage_s": d.get("train_seconds"),
                    "r_mean": d.get("r_mean"),
                    "r_p95": d.get("r_p95"),
                    "r_p99": d.get("r_p99"),
                    "clip_frac_high": d.get("clip_frac_high"),
                    "clip_frac_low": d.get("clip_frac_low"),
                    "cumtime_min": cumtime_s / 60,
                })
            elif d.get("type") == "eval":
                acc_val = next((d[k] for k in d if k.startswith("avg_at_")), None)
                if acc_val is not None:
                    dataset = d.get("dataset", "primary")
                    step = d["step"]
                    eval_data.append({"step": step, "avg_at_n": acc_val, "dataset": dataset})
                    eval_seen.add((step, dataset))

    # Also load from eval.jsonl
    eval_jsonl_path = run_dir / "eval.jsonl"
    if eval_jsonl_path.exists():
        with open(eval_jsonl_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") != "eval":
                    continue
                acc_val = next((d[k] for k in d if k.startswith("avg_at_")), None)
                if acc_val is None:
                    continue
                dataset = d.get("dataset", "primary")
                step = d.get("step")
                if step is None:
                    step = 0  # null step = baseline
                if (step, dataset) not in eval_seen:
                    eval_data.append({"step": step, "avg_at_n": acc_val, "dataset": dataset})

    # Drop "primary" entries if eval.jsonl provided explicit dataset names
    has_named = any(e["dataset"] != "primary" for e in eval_data)
    if has_named:
        eval_data = [e for e in eval_data if e["dataset"] != "primary"]

    edf = pd.DataFrame(eval_data)
    if not edf.empty:
        edf["samples_seen"] = edf["step"] * batch_size

    totals = {"total_train_s": cumtime_s, "total_eval_s": 0, "total_s": cumtime_s}
    return pd.DataFrame(rows), edf, pd.DataFrame(), batch_size, totals


def get_per_step_data(
    run_dir: Path, results_dir: Path, skip_trace: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, dict]:
    """Get per-step metrics from trace + log.

    Returns (train_df, eval_df, rollout_stats_df, batch_size, totals).
    rollout_stats_df has columns: t_s, running_reqs, gen_tok_s (accumulated over workers).

    If skip_trace=True, builds train_df from log.jsonl only (faster, no trace parsing).
    """
    trace_path = run_dir / "trace_live.json"
    log_path = run_dir / "log.jsonl"

    if skip_trace:
        return _get_per_step_data_from_log(run_dir, results_dir)

    if not trace_path.exists():
        return _get_per_step_data_from_log(run_dir, results_dir)

    # Load batch_size from config
    cfg = load_config(run_dir, results_dir)
    batch_size = _get_train_batch_size(cfg)

    events = load_trace(trace_path)
    x_events = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    if not x_events:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), batch_size, {}
    t_min = min(e["ts"] for e in x_events)

    # Compute iter times from trace
    boundary_events = sorted(
        [e for e in x_events if e.get("name") in ("train", "eval")],
        key=lambda e: e["ts"] + e["dur"],
    )
    iter_data: list[dict] = []
    total_train_s = 0.0
    total_eval_s = 0.0
    prev_end = t_min
    for e in boundary_events:
        if e["name"] == "train":
            cur_end = e["ts"] + e["dur"]
            train_dur = e["dur"] / 1e6
            iter_s = (cur_end - prev_end) / 1e6
            gap_s = iter_s - train_dur
            iter_data.append({"iter_s": iter_s, "train_s": train_dur, "gap_s": gap_s})
            total_train_s += iter_s
            prev_end = cur_end
        elif e["name"] == "eval":
            total_eval_s += e["dur"] / 1e6
            prev_end = e["ts"] + e["dur"]

    totals = {
        "total_train_s": total_train_s,
        "total_eval_s": total_eval_s,
        "total_s": total_train_s + total_eval_s,
    }

    # --- Extract per-step stage times from trace spans ---
    gen_spans = sorted(
        [e for e in x_events if e.get("name") == "generate"], key=lambda e: e["ts"]
    )
    train_spans = sorted(
        [e for e in x_events if e.get("name") == "train"], key=lambda e: e["ts"]
    )
    teacher_spans = sorted(
        [e for e in x_events if e.get("name") == "teacher_score"],
        key=lambda e: e["ts"],
    )

    # Build per-step stage durations
    step_stages: dict[int, dict] = {}
    for i in range(len(train_spans)):
        step_num = i + 1
        gen_s = gen_spans[i]["dur"] / 1e6 if i < len(gen_spans) else 0
        train_s = train_spans[i]["dur"] / 1e6
        gen_start = gen_spans[i]["ts"] if i < len(gen_spans) else train_spans[i]["ts"]
        train_end = train_spans[i]["ts"] + train_spans[i]["dur"]
        teach_s = sum(
            e["dur"] / 1e6
            for e in teacher_spans
            if e["ts"] >= gen_start and e["ts"] + e["dur"] <= train_end
        )
        step_stages[step_num] = {
            "generate_s": gen_s,
            "teacher_s": teach_s,
            "train_stage_s": train_s,
        }

    # --- Extract rollout stats from trace counters (fully_async mode) ---
    rollout_counters = [
        e
        for e in events
        if e.get("ph") == "C" and e.get("name", "").startswith("rollout-w")
    ]
    rollout_stats_rows: list[dict] = []
    if rollout_counters:
        ts_buckets: dict[int, dict] = defaultdict(
            lambda: {"running_reqs": 0, "gen_tok_s": 0}
        )
        for c in rollout_counters:
            bucket = round(c["ts"] / 100000) * 100000  # 100ms buckets
            args = c.get("args", {})
            ts_buckets[bucket]["running_reqs"] += args.get("running_reqs", 0)
            ts_buckets[bucket]["gen_tok_s"] += args.get("gen_tok/s", 0)
        for ts, vals in sorted(ts_buckets.items()):
            t_rel = (ts - t_min) / 1e6
            rollout_stats_rows.append(
                {
                    "t_s": t_rel,
                    "running_reqs": vals["running_reqs"],
                    "gen_tok_s": vals["gen_tok_s"],
                }
            )
    rollout_stats_df = pd.DataFrame(rollout_stats_rows)

    # Get per-step data from log
    step_data: dict[int, dict] = {}
    eval_data: list[dict] = []
    eval_seen: set[tuple] = set()
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") == "train":
                    step_data[d["step"]] = d
                if d.get("type") == "eval":
                    acc_val = next((d[k] for k in d if k.startswith("avg_at_")), None)
                    if acc_val is not None:
                        dataset = d.get("dataset", "primary")
                        step = d["step"]
                        eval_data.append({"step": step, "avg_at_n": acc_val, "dataset": dataset})
                        eval_seen.add((step, dataset))
    # Also load from eval.jsonl
    eval_jsonl_path = run_dir / "eval.jsonl"
    if eval_jsonl_path.exists():
        with open(eval_jsonl_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") != "eval":
                    continue
                acc_val = next((d[k] for k in d if k.startswith("avg_at_")), None)
                if acc_val is None:
                    continue
                dataset = d.get("dataset", "primary")
                step = d.get("step")
                if step is None:
                    step = 0
                if (step, dataset) not in eval_seen:
                    eval_data.append({"step": step, "avg_at_n": acc_val, "dataset": dataset})

    # Drop "primary" entries if eval.jsonl provided explicit dataset names
    has_named = any(e["dataset"] != "primary" for e in eval_data)
    if has_named:
        eval_data = [e for e in eval_data if e["dataset"] != "primary"]

    rows: list[dict] = []
    cumtime_s = 0.0
    for i, d in enumerate(iter_data):
        cumtime_s += d["iter_s"]
        step_num = i + 1
        log_d = step_data.get(step_num, {})
        n_tok = log_d.get("n_tokens", 0)
        # Workaround: results before 2026-03-24 logged per-rank n_tokens.
        if cfg and n_tok > 0 and "008_packing/" in str(run_dir):
            n_tgpus = len(
                str(cfg["training"]["trainer"].get("gpu_ids", "0")).split(",")
            )
            if n_tgpus > 1:
                n_tok *= n_tgpus
        stages = step_stages.get(step_num, {})
        rows.append(
            {
                "step": step_num,
                "samples_seen": step_num * batch_size,
                "iter_s": d["iter_s"],
                "train_s": d["train_s"],
                "gap_s": d["gap_s"],
                "n_tokens": n_tok,
                "s/sample_iter": d["iter_s"] / batch_size,
                "s/sample_gen": d.get("gap_s", 0) / batch_size,
                "tok/sample": n_tok / batch_size if batch_size else 0,
                "tok_per_s": n_tok / d["iter_s"] if d["iter_s"] > 0 else 0,
                "kl_loss": log_d.get("kl_loss"),
                "avg_resp_len": log_d.get("avg_response_length"),
                "staleness_mean": log_d.get("staleness_mean"),
                "generate_s": stages.get("generate_s"),
                "teacher_s": stages.get("teacher_s"),
                "train_stage_s": stages.get("train_stage_s"),
                "r_mean": log_d.get("r_mean"),
                "r_p95": log_d.get("r_p95"),
                "r_p99": log_d.get("r_p99"),
                "clip_frac_high": log_d.get("clip_frac_high"),
                "clip_frac_low": log_d.get("clip_frac_low"),
                "cumtime_min": cumtime_s / 60,
            }
        )

    edf = pd.DataFrame(eval_data)
    if not edf.empty:
        edf["samples_seen"] = edf["step"] * batch_size

    return pd.DataFrame(rows), edf, rollout_stats_df, batch_size, totals


def load_runs(
    run_names: list[str],
    labels: dict[str, str] | list[tuple[object, str]] | None = None,
    results_dir: Path = Path("results"),
    skip_trace: bool = False,
) -> RunData:
    """Load all runs and assemble into a RunData container.

    Parameters
    ----------
    run_names : list[str]
        Folder names under *results_dir*, or specific run paths.  Globs allowed.
    labels : dict | sequence[(paths, label)] | None
        Mapping/spec of run path(s) -> display label. Dict keys may be a single
        path or a tuple of paths; sequence entries may use a single path or a
        nested list/tuple of paths. Runs not in this mapping use
        ``"<path> (bs=N)"`` as the label.
    results_dir : Path
        Root results directory (default ``Path("results")``).

    Returns
    -------
    RunData
    """
    import yaml as _yaml

    label_map = _normalize_label_map(labels)
    run_dirs = discover_runs(run_names, results_dir)

    all_train_rows: list[pd.DataFrame] = []
    all_eval_rows: list[pd.DataFrame] = []
    all_rollout_stats: list[pd.DataFrame] = []
    run_totals: dict[str, list[dict]] = defaultdict(list)
    run_counts: dict[str, int] = defaultdict(int)
    source_to_label: dict[str, str] = {}

    for d in run_dirs:
        name = d.relative_to(results_dir).as_posix()
        pdf, edf, rdf, bs, totals = get_per_step_data(d, results_dir, skip_trace=skip_trace)
        if pdf.empty:
            continue
        label = label_map.get(name, f"{name} (bs={bs})")
        source_to_label[name] = label
        pdf["run"] = label
        pdf["source_run"] = name
        all_train_rows.append(pdf)
        run_totals[label].append(totals)
        run_counts[label] += 1
        if not edf.empty:
            edf["run"] = label
            edf["source_run"] = name
            all_eval_rows.append(edf)
        if not rdf.empty:
            rdf["run"] = label
            rdf["source_run"] = name
            all_rollout_stats.append(rdf)

    train_df = (
        pd.concat(all_train_rows, ignore_index=True)
        if all_train_rows
        else pd.DataFrame()
    )

    # Fix staleness for 015_step_off_k_4mini* runs: staleness was logged per
    # optimizer step (Nx per training step) due to mini-batch bug.
    if not train_df.empty and "staleness_mean" in train_df.columns:
        _stale_fix_folders = (
            "015_step_off_k_4mini/",
            "015_step_off_k_4mini_areal/",
        )
        for d in run_dirs:
            name = d.relative_to(results_dir).as_posix()
            if any(name.startswith(f) for f in _stale_fix_folders):
                label = source_to_label.get(name, name)
                mask = (train_df["run"] == label) & (train_df["source_run"] == name)
                if mask.any():
                    cfg = None
                    cfg_yaml_path = d / "config.yaml"
                    if cfg_yaml_path.exists():
                        cfg = _yaml.safe_load(open(cfg_yaml_path))
                    if cfg is None:
                        log_path = d / "log.jsonl"
                        if log_path.exists():
                            cfg = json.loads(
                                open(log_path).readline()
                            ).get("config", {})
                    if cfg:
                        tbs = _get_train_batch_size(cfg)
                        mbs = _get_actor_mini_batch_size(cfg)
                        n_mini = tbs // mbs if mbs > 0 else 1
                        if n_mini > 1:
                            train_df.loc[mask, "staleness_mean"] = (
                                train_df.loc[mask, "staleness_mean"] / n_mini
                            )

    eval_df = (
        pd.concat(all_eval_rows, ignore_index=True)
        if all_eval_rows
        else pd.DataFrame()
    )
    rollout_df = (
        pd.concat(all_rollout_stats, ignore_index=True)
        if all_rollout_stats
        else pd.DataFrame()
    )

    # Build palette: sort by staleness, sample smooth gradient from colormap
    run_names_sorted = sorted(
        train_df["run"].unique(), key=_extract_staleness
    ) if not train_df.empty else []
    cmap = plt.get_cmap(_PALETTE_CMAP)
    n = max(len(run_names_sorted), 1)
    run_palette = {
        name: cmap(i / max(n - 1, 1))
        for i, name in enumerate(run_names_sorted)
    }

    # Apply palette to seaborn defaults
    if run_names_sorted:
        sns.set_palette([run_palette[r] for r in run_names_sorted])

    return RunData(
        train_df=train_df,
        eval_df=eval_df,
        rollout_df=rollout_df,
        run_palette=run_palette,
        run_order=run_names_sorted,
        totals=run_totals,
        run_counts=dict(run_counts),
        run_dirs=run_dirs,
        results_dir=Path(results_dir),
        labels=label_map,
    )


# ── Plot functions ───────────────────────────────────────────────
# Each returns a matplotlib Figure (or list of Figures for multi-figure plots).


# Available metric names for plot_overview's metrics parameter
OVERVIEW_METRICS = [
    "eval_accuracy", "elapsed_time", "time_per_sample", "tokens_per_sample",
    "throughput", "kl_loss", "response_length", "staleness",
    "ppo_ratio_mean", "ppo_ratio_p95", "ppo_clip_high", "ppo_clip_low",
    "rollout_requests", "rollout_throughput", "rollout_queue", "rollout_idle",
]

# Map metric names to (column, ylabel, title)
_TRAIN_METRIC_SPECS = {
    "elapsed_time": ("cumtime_min", "Elapsed time (min)", "Elapsed wall time"),
    "time_per_sample": ("s/sample_iter", "s / sample", "Wall time per sample"),
    "tokens_per_sample": ("tok/sample", "Resp tokens / sample", "Response tokens per sample"),
    "throughput": ("tok_per_s", "Tokens/s", "Step throughput (tok/s)"),
    "kl_loss": ("kl_loss", "KL Loss", "KL Loss"),
    "response_length": ("avg_resp_len", "Avg response length", "Average response length (tokens)"),
    "staleness": ("staleness_mean", "Staleness", "Staleness"),
}

_PPO_METRIC_SPECS = {
    "ppo_ratio_mean": ("r_mean", "PPO ratio (mean)", "PPO importance sampling ratio (mean)"),
    "ppo_ratio_p95": ("r_p95", "PPO ratio (P95)", "PPO importance sampling ratio (P95)"),
    "ppo_clip_high": ("clip_frac_high", "Clip fraction (high)", "PPO clip fraction (high)"),
    "ppo_clip_low": ("clip_frac_low", "Clip fraction (low)", "PPO clip fraction (low)"),
}

_ROLLOUT_METRICS = {"rollout_requests", "rollout_throughput", "rollout_queue", "rollout_idle"}


def plot_overview(
    data: RunData,
    smooth_window: int = 10,
    metrics: list[str] | None = None,
    *,
    _run_dirs: list[Path] | None = None,
    _results_dir: Path | None = None,
    _labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Overview plot with selectable metrics.

    Args:
        metrics: list of metric names to include, or None for all.
                 See OVERVIEW_METRICS for available names.
    Returns the Figure.
    """
    train_df = data.train_df
    eval_df = data.eval_df
    rollout_df = data.rollout_df
    palette = data.run_palette
    run_order = data.run_order

    # Filter specs by requested metrics
    if metrics is not None:
        m_set = set(metrics)
    else:
        m_set = None  # means all

    include_eval = (m_set is None or "eval_accuracy" in m_set) and not eval_df.empty
    # Discover distinct eval datasets
    eval_datasets = []
    if include_eval:
        eval_datasets = sorted(eval_df["dataset"].unique()) if "dataset" in eval_df.columns else ["primary"]
    n_eval_plots = len(eval_datasets) if include_eval else 0
    plot_specs = [
        (col, ylabel, title)
        for name, (col, ylabel, title) in _TRAIN_METRIC_SPECS.items()
        if m_set is None or name in m_set
    ]
    ppo_plot_specs = [
        (col, ylabel, title)
        for name, (col, ylabel, title) in _PPO_METRIC_SPECS.items()
        if m_set is None or name in m_set
    ]
    # Use RunData's stored run_dirs/results_dir if private kwargs not passed
    if _run_dirs is None:
        _run_dirs = data.run_dirs
    if _results_dir is None:
        _results_dir = data.results_dir
    if _labels is None:
        _labels = data.labels

    has_rollout = (
        not rollout_df.empty
        and bool(_run_dirs)
        and _results_dir is not None
        and (m_set is None or m_set & _ROLLOUT_METRICS)
    )

    n_ppo_plots = (
        len(ppo_plot_specs)
        if not train_df.empty and "r_mean" in train_df.columns
        else 0
    )
    extra_plots = 4 if has_rollout else 0
    n_plots = len(plot_specs) + n_ppo_plots + extra_plots + n_eval_plots

    if n_plots == 0:
        return plt.figure()

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    _ls_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    _markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "d"]

    # Eval accuracy plots (one per dataset)
    for ds_idx, ds_name in enumerate(eval_datasets):
        ds_df = eval_df[eval_df["dataset"] == ds_name] if "dataset" in eval_df.columns else eval_df
        # Derive a short label for the dataset
        ds_short = ds_name.split("/")[-1] if "/" in ds_name else ds_name
        # Detect avg_at_N from the metric key
        acc_col_name = "Avg@N"
        _plot_mean_std_lines(
            axes[ds_idx],
            ds_df,
            "samples_seen",
            "avg_at_n",
            palette=palette,
            run_order=run_order,
            linestyles=_ls_styles,
            markers=_markers,
            markersize=4,
        )
        axes[ds_idx].set_xlabel("Samples seen")
        axes[ds_idx].set_ylabel(f"{acc_col_name} (%)")
        axes[ds_idx].set_title(f"Eval accuracy — {ds_short}")
        axes[ds_idx].legend(fontsize=8)

    for i, (col, ylabel, title) in enumerate(plot_specs, start=n_eval_plots):
        subset = train_df.dropna(subset=[col])
        if subset.empty:
            axes[i].set_title(f"{title} (no data)")
            continue
        ax = axes[i]
        _plot_mean_std_lines(
            ax,
            subset,
            "samples_seen",
            col,
            palette=palette,
            run_order=run_order,
            linestyles=_ls_styles,
            markers=_markers,
            markersize=3,
            markevery=max(len(subset) // max(len(run_order), 1) // 10, 1),
        )
        ax.set_xlabel("Samples seen")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        # Clamp staleness y-axis
        if col == "staleness_mean":
            p90 = subset[col].quantile(0.90)
            if p90 > 0:
                ax.set_ylim(bottom=0, top=p90 * 1.5)

    ax_idx = len(plot_specs) + n_eval_plots

    # PPO ratio plots
    for col, ylabel, title in ppo_plot_specs:
        if col not in train_df.columns:
            continue
        subset = train_df.dropna(subset=[col])
        if subset.empty:
            axes[ax_idx].set_title(f"{title} (no data)")
            ax_idx += 1
            continue
        ax = axes[ax_idx]
        _plot_mean_std_lines(
            ax,
            subset,
            "samples_seen",
            col,
            palette=palette,
            run_order=run_order,
            linestyles=_ls_styles,
            markers=_markers,
            markersize=3,
            markevery=max(len(subset) // max(len(run_order), 1) // 10, 1),
        )
        ax.set_xlabel("Samples seen")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        if col in ("r_mean", "r_p95"):
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax_idx += 1

    # Rollout stats: per-step aggregates
    if has_rollout and _run_dirs is not None and _results_dir is not None:
        _labels = _labels or {}
        step_aligned_all: list[pd.DataFrame] = []
        for d in _run_dirs:
            rname = d.relative_to(_results_dir).as_posix()
            cfg = load_config(d, _results_dir)
            label = _labels.get(
                rname,
                f"{rname} (bs={_get_train_batch_size(cfg)})"
                if cfg
                else rname,
            )
            if label in rollout_df["run"].unique():
                sa = _build_step_aligned(d, rollout_df[rollout_df["run"] == label], _results_dir)
                if not sa.empty:
                    sa["run"] = label
                    sa["source_run"] = rname
                    step_aligned_all.append(sa)

        if step_aligned_all:
            sa_df = pd.concat(step_aligned_all, ignore_index=True)
            for col, ylabel, title in [
                ("running_reqs", "Running requests", "Rollout running requests per step"),
                ("gen_tok_s", "Gen tokens/s", "Rollout generation throughput per step"),
            ]:
                ax = axes[ax_idx]
                _plot_mean_std_lines(
                    ax,
                    sa_df,
                    "step",
                    col,
                    palette=palette,
                    run_order=run_order,
                )
                ax.set_xlabel("Step")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend(fontsize=8)
                ax_idx += 1

        # Rollout stats: histograms
        for col, xlabel, title in [
            ("running_reqs", "Running requests", "Rollout running requests distribution"),
            ("gen_tok_s", "Gen tokens/s", "Rollout generation throughput distribution"),
        ]:
            ax = axes[ax_idx]
            vals = rollout_df[col].dropna()
            bins = np.linspace(vals.min(), vals.max(), 401)
            for run_name, grp in rollout_df.groupby("run"):
                ax.hist(
                    grp[col].dropna(),
                    bins=bins,
                    alpha=0.5,
                    label=run_name,
                    edgecolor="none",
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.set_title(title)
            ax.set_xlim(left=0, right=rollout_df[col].quantile(0.99) * 1.05)
            ax.legend(fontsize=8)
            ax_idx += 1

    fig.tight_layout()
    return fig


def plot_diagnostics(data: RunData) -> plt.Figure | None:
    """2x3 diagnostics grid: throughput, acc vs time/tokens, clip, idle, violin.

    Returns Figure, or None if data is insufficient.
    """
    train_df = data.train_df
    eval_df = data.eval_df
    palette = data.run_palette

    if train_df.empty or eval_df.empty:
        return None

    # Use only the first (primary) dataset
    if "dataset" in eval_df.columns:
        primary_ds = sorted(eval_df["dataset"].unique())[0]
        eval_df = eval_df[eval_df["dataset"] == primary_ds]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # --- 1. Effective training throughput (tok/min) ---
    ax = axes[0, 0]
    throughput_df = train_df.copy()
    throughput_df["tok_per_min_m"] = throughput_df["n_tokens"] / (
        throughput_df["iter_s"] / 60
    ) / 1e6
    _plot_mean_std_lines(
        ax,
        throughput_df,
        "samples_seen",
        "tok_per_min_m",
        palette=palette,
        run_order=data.run_order,
        smooth_window=10,
    )
    ax.set_xlabel("Samples seen")
    ax.set_ylabel("M tokens / min")
    ax.set_title("Effective training throughput")
    ax.legend(fontsize=7)

    # --- 2. Accuracy vs wall time ---
    ax = axes[0, 1]
    acc_time_rows: list[dict] = []
    for run_name, egrp in eval_df.groupby("run"):
        for source_run, source_egrp in egrp.groupby("source_run" if "source_run" in egrp.columns else lambda _: "__single__"):
            tgrp = train_df[(train_df["run"] == run_name)]
            if "source_run" in train_df.columns:
                tgrp = tgrp[tgrp["source_run"] == source_run]
            wall_mins: list[float] = []
            for _, row in source_egrp.iterrows():
                if row["step"] == 0:
                    wall_mins.append(0)
                else:
                    m = tgrp[tgrp["step"] == row["step"]]
                    wall_mins.append(
                        m["cumtime_min"].values[0] if len(m) > 0 else np.nan
                    )
            tmp = source_egrp.copy()
            tmp["wall_min"] = wall_mins
            acc_time_rows.append(tmp[["run", "source_run", "wall_min", "avg_at_n"]])
    if acc_time_rows:
        acc_time_df = pd.concat(acc_time_rows, ignore_index=True)
        _plot_mean_std_lines(
            ax,
            acc_time_df.dropna(subset=["wall_min"]),
            "wall_min",
            "avg_at_n",
            palette=palette,
            run_order=data.run_order,
        )
    ax.set_xlabel("Wall time (min)")
    ax.set_ylabel("Avg@32 (%)")
    ax.set_title("Accuracy vs wall time")
    ax.legend(fontsize=7)

    # --- 3. Cumulative tokens trained vs accuracy ---
    ax = axes[0, 2]
    acc_tok_rows: list[dict] = []
    for run_name, egrp in eval_df.groupby("run"):
        for source_run, source_egrp in egrp.groupby("source_run" if "source_run" in egrp.columns else lambda _: "__single__"):
            tgrp = train_df[(train_df["run"] == run_name)]
            if "source_run" in train_df.columns:
                tgrp = tgrp[tgrp["source_run"] == source_run]
            cum_tok: list[float] = []
            for _, row in source_egrp.iterrows():
                if row["step"] == 0:
                    cum_tok.append(0)
                else:
                    m = tgrp[tgrp["step"] <= row["step"]]
                    cum_tok.append(m["n_tokens"].sum() / 1e9)
            tmp = source_egrp.copy()
            tmp["cum_tok_b"] = cum_tok
            acc_tok_rows.append(tmp[["run", "source_run", "cum_tok_b", "avg_at_n"]])
    if acc_tok_rows:
        acc_tok_df = pd.concat(acc_tok_rows, ignore_index=True)
        _plot_mean_std_lines(
            ax,
            acc_tok_df,
            "cum_tok_b",
            "avg_at_n",
            palette=palette,
            run_order=data.run_order,
        )
    ax.set_xlabel("Cumulative tokens trained (B)")
    ax.set_ylabel("Avg@32 (%)")
    ax.set_title("Accuracy vs tokens trained")
    ax.legend(fontsize=7)

    # --- 4. Clip fraction vs staleness ---
    ax = axes[1, 0]
    clip_summary: list[dict] = []
    group_cols = ["run", "source_run"] if "source_run" in train_df.columns else ["run"]
    for keys, grp in train_df.groupby(group_cols):
        run_name = keys[0] if isinstance(keys, tuple) else keys
        clip_summary.append(
            {
                "run": run_name.replace(" (bs=256)", ""),
                "source_run": keys[1] if isinstance(keys, tuple) and len(keys) > 1 else "__single__",
                "clip_high": grp["clip_frac_high"].mean(),
                "clip_low": grp["clip_frac_low"].mean(),
            }
        )
    cdf = pd.DataFrame(clip_summary)
    clip_stats = cdf.groupby("run").agg(
        clip_high_mean=("clip_high", "mean"),
        clip_high_std=("clip_high", _std0),
        clip_low_mean=("clip_low", "mean"),
        clip_low_std=("clip_low", _std0),
    ).reset_index()
    ax.bar(clip_stats["run"], clip_stats["clip_high_mean"], yerr=clip_stats["clip_high_std"], label="clip high", alpha=0.8)
    ax.bar(
        clip_stats["run"],
        clip_stats["clip_low_mean"],
        bottom=clip_stats["clip_high_mean"],
        yerr=clip_stats["clip_low_std"],
        label="clip low",
        alpha=0.8,
    )
    ax.set_ylabel("Clip fraction")
    ax.set_title("PPO clipping vs staleness config")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    # --- 5. Train idle % vs config ---
    ax = axes[1, 1]
    idle_data: list[dict] = []
    for keys, grp in train_df.groupby(group_cols):
        run_name = keys[0] if isinstance(keys, tuple) else keys
        train = grp["train_stage_s"].mean()
        it = grp["iter_s"].mean()
        idle_pct = max(0, (1 - train / it)) * 100 if pd.notna(train) and pd.notna(it) and it > 0 else np.nan
        idle_data.append(
            {
                "run": run_name.replace(" (bs=256)", ""),
                "source_run": keys[1] if isinstance(keys, tuple) and len(keys) > 1 else "__single__",
                "train_idle_pct": idle_pct,
                "tok_per_min": grp["n_tokens"].sum() / grp["cumtime_min"].max() / 1e6 if grp["cumtime_min"].max() > 0 else np.nan,
            }
        )
    idf = pd.DataFrame(idle_data)
    idle_stats = idf.groupby("run").agg(
        train_idle_pct_mean=("train_idle_pct", "mean"),
        train_idle_pct_std=("train_idle_pct", _std0),
        tok_per_min_mean=("tok_per_min", "mean"),
    ).reset_index()
    bars = ax.bar(idle_stats["run"], idle_stats["train_idle_pct_mean"], yerr=idle_stats["train_idle_pct_std"], alpha=0.8)
    ax.set_ylabel("Train idle (%)")
    ax.set_title("Trainer idle time (waiting for data)")
    ax.tick_params(axis="x", rotation=30)
    for bar, row in zip(bars, idle_stats.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{row.tok_per_min_mean:.1f}M t/m",
            ha="center",
            fontsize=8,
        )

    # --- 6. Per-step r_p95 distribution (violin) ---
    ax = axes[1, 2]
    vdata = train_df[["run", "r_p95"]].dropna()
    vdata = vdata.copy()
    vdata["run_short"] = vdata["run"].str.replace(" (bs=256)", "")
    order = sorted(vdata["run_short"].unique())
    sns.violinplot(
        data=vdata, x="run_short", y="r_p95", order=order, inner="quart", ax=ax, cut=0
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("PPO ratio (P95)")
    ax.set_title("Importance sampling ratio distribution")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    return fig


def plot_kl_analysis(
    data: RunData, smooth_window: int = 10
) -> plt.Figure | None:
    """2x2 KL analysis: KL vs time, KL vs tokens, time-to-KL-threshold, convergence speed.

    Returns Figure, or None if data is insufficient.
    """
    train_df = data.train_df
    palette = data.run_palette
    run_order = data.run_order

    if train_df.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    group_cols = ["run", "source_run"] if "source_run" in train_df.columns else ["run"]

    # --- 1. KL loss vs wall time ---
    ax = axes[0, 0]
    _plot_mean_std_lines(
        ax,
        train_df.dropna(subset=["kl_loss"]),
        "cumtime_min",
        "kl_loss",
        palette=palette,
        run_order=run_order,
        smooth_window=smooth_window,
    )
    ax.set_xlabel("Wall time (min)")
    ax.set_ylabel("KL loss (smoothed)")
    ax.set_title("KL loss vs wall time")
    ax.legend(fontsize=7)

    # --- 2. KL loss vs cumulative tokens trained ---
    ax = axes[0, 1]
    kl_tok_rows: list[pd.DataFrame] = []
    for keys, grp in train_df.groupby(group_cols):
        run_name = keys[0] if isinstance(keys, tuple) else keys
        source_run = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else "__single__"
        g = grp.sort_values("step").dropna(subset=["kl_loss"]).copy()
        if g.empty:
            continue
        g["cum_tok_b"] = g["n_tokens"].cumsum() / 1e9
        g["run"] = run_name
        g["source_run"] = source_run
        kl_tok_rows.append(g[["run", "source_run", "cum_tok_b", "kl_loss"]])
    if kl_tok_rows:
        _plot_mean_std_lines(
            ax,
            pd.concat(kl_tok_rows, ignore_index=True),
            "cum_tok_b",
            "kl_loss",
            palette=palette,
            run_order=run_order,
            smooth_window=smooth_window,
        )
    ax.set_xlabel("Cumulative tokens trained (B)")
    ax.set_ylabel("KL loss (smoothed)")
    ax.set_title("KL loss vs tokens trained (data quality signal)")
    ax.legend(fontsize=7)

    # --- 3. Time to KL threshold ---
    ax = axes[1, 0]
    kl_thresholds = [3.0, 2.0, 1.5, 1.0, 0.5]
    kl_ttt_data: list[dict] = []
    for keys, grp in train_df.groupby(group_cols):
        run_name = keys[0] if isinstance(keys, tuple) else keys
        source_run = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else "__single__"
        grp = grp.sort_values("cumtime_min").dropna(subset=["kl_loss"])
        for thresh in kl_thresholds:
            reached = grp[grp["kl_loss"] <= thresh]
            t = reached["cumtime_min"].min() if len(reached) > 0 else np.nan
            kl_ttt_data.append(
                {"run": run_name, "source_run": source_run, "kl_threshold": thresh, "time_min": t}
            )
    kl_ttt_df = pd.DataFrame(kl_ttt_data)
    _plot_mean_std_lines(
        ax,
        kl_ttt_df.dropna(subset=["time_min"]),
        "kl_threshold",
        "time_min",
        palette=palette,
        run_order=run_order,
        markers=["s"],
        linestyles=["-"],
        markersize=5,
    )
    ax.set_xlabel("KL loss threshold (lower = better)")
    ax.set_ylabel("Time to reach (min)")
    ax.set_title("Time to KL threshold")
    ax.invert_xaxis()
    ax.legend(fontsize=7)

    # --- 4. KL-based efficiency summary bar chart ---
    ax = axes[1, 1]
    kl_eff: list[dict] = []
    for keys, grp in train_df.groupby(group_cols):
        run_name = keys[0] if isinstance(keys, tuple) else keys
        grp = grp.sort_values("cumtime_min").dropna(subset=["kl_loss"])
        if grp.empty:
            continue
        total_min = grp["cumtime_min"].max()
        final_kl = grp["kl_loss"].iloc[-1]
        initial_kl = grp["kl_loss"].iloc[0]
        kl_reduction_rate = (initial_kl - final_kl) / total_min if total_min > 0 else 0
        kl_eff.append(
            {"run": run_name.replace(" (bs=256)", ""), "kl_reduction_rate": kl_reduction_rate}
        )
    kl_eff_df = (
        pd.DataFrame(kl_eff)
        .groupby("run", as_index=False)
        .agg(
            kl_reduction_rate_mean=("kl_reduction_rate", "mean"),
            kl_reduction_rate_std=("kl_reduction_rate", _std0),
        )
        .sort_values("kl_reduction_rate_mean", ascending=True)
    )
    bars = ax.barh(
        kl_eff_df["run"],
        kl_eff_df["kl_reduction_rate_mean"],
        xerr=kl_eff_df["kl_reduction_rate_std"],
    )
    ax.set_xlabel("KL reduction rate (KL drop / min)")
    ax.set_title("KL convergence speed")
    for bar, val, std in zip(
        bars,
        kl_eff_df["kl_reduction_rate_mean"],
        kl_eff_df["kl_reduction_rate_std"],
    ):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}±{std:.3f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def plot_kl_pareto(data: RunData) -> plt.Figure | None:
    """Pareto scatter: total time vs best (lowest) KL.

    Returns Figure, or None if data is insufficient.
    """
    train_df = data.train_df
    palette = data.run_palette
    run_order = data.run_order

    if train_df.empty:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    group_cols = ["run", "source_run"] if "source_run" in train_df.columns else ["run"]
    pareto_kl: list[dict] = []
    for keys, grp in train_df.groupby(group_cols):
        run_name = keys[0] if isinstance(keys, tuple) else keys
        grp = grp.sort_values("cumtime_min").dropna(subset=["kl_loss"])
        if grp.empty:
            continue
        pareto_kl.append(
            {
                "run": run_name,
                "best_kl": grp["kl_loss"].min(),
                "total_min": grp["cumtime_min"].max(),
            }
        )
    pkdf = (
        pd.DataFrame(pareto_kl)
        .groupby("run", as_index=False)
        .agg(
            best_kl_mean=("best_kl", "mean"),
            best_kl_std=("best_kl", _std0),
            total_min_mean=("total_min", "mean"),
            total_min_std=("total_min", _std0),
        )
    )
    ax.errorbar(
        pkdf["total_min_mean"],
        pkdf["best_kl_mean"],
        xerr=pkdf["total_min_std"],
        yerr=pkdf["best_kl_std"],
        fmt="none",
        ecolor="gray",
        alpha=0.5,
        zorder=3,
    )
    sns.scatterplot(
        data=pkdf,
        x="total_min_mean",
        y="best_kl_mean",
        hue="run",
        s=100,
        zorder=5,
        ax=ax,
        legend=False,
        palette=palette,
        hue_order=run_order,
    )
    for _, row in pkdf.iterrows():
        name = row["run"].replace(" (bs=256)", "")
        ax.annotate(
            name,
            (row["total_min_mean"], row["best_kl_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    # Pareto frontier (lower-left is better)
    pkdf_sorted = pkdf.sort_values("total_min_mean")
    frontier_x: list[float] = []
    frontier_y: list[float] = []
    best_kl_so_far = float("inf")
    for _, row in pkdf_sorted.iterrows():
        if row["best_kl_mean"] < best_kl_so_far:
            frontier_x.append(row["total_min_mean"])
            frontier_y.append(row["best_kl_mean"])
            best_kl_so_far = row["best_kl_mean"]
    ax.plot(
        frontier_x,
        frontier_y,
        "r--",
        alpha=0.5,
        linewidth=1.5,
        label="Pareto frontier",
    )
    ax.set_xlabel("Total wall time (min)")
    ax.set_ylabel("Best KL loss (lower = better)")
    ax.set_title("Pareto: Time vs Best KL Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pareto_evolution(
    data: RunData, metric: str = "accuracy"
) -> plt.Figure | None:
    """Per-eval-step Pareto evolution.

    Parameters
    ----------
    metric : str
        ``"accuracy"`` or ``"kl"``.

    Returns Figure, or None if data is insufficient.
    """
    train_df = data.train_df
    eval_df = data.eval_df
    palette = data.run_palette
    run_order = data.run_order
    SMOOTH_WINDOW = 10

    if train_df.empty or eval_df.empty:
        return None

    # Use only the first (primary) dataset
    if "dataset" in eval_df.columns:
        primary_ds = sorted(eval_df["dataset"].unique())[0]
        eval_df = eval_df[eval_df["dataset"] == primary_ds]

    eval_steps = sorted(eval_df[eval_df["step"] > 0]["step"].unique())
    n_eval = len(eval_steps)
    if n_eval == 0:
        return None

    group_cols = ["run", "source_run"] if "source_run" in train_df.columns else ["run"]

    if metric == "accuracy":
        fig, axes_arr = plt.subplots(
            1, n_eval, figsize=(4 * n_eval, 4), sharey=True
        )
        if n_eval == 1:
            axes_arr = [axes_arr]
        for ax_i, eval_step in enumerate(eval_steps):
            ax = axes_arr[ax_i]
            pts: list[dict] = []
            for keys, egrp in eval_df.groupby(group_cols):
                run_name = keys[0] if isinstance(keys, tuple) else keys
                source_run = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else "__single__"
                tgrp = train_df[train_df["run"] == run_name]
                if "source_run" in train_df.columns:
                    tgrp = tgrp[tgrp["source_run"] == source_run]
                row = egrp[egrp["step"] == eval_step]
                if row.empty:
                    continue
                acc = row["avg_at_n"].values[0]
                m = tgrp[tgrp["step"] == eval_step]
                wt = m["cumtime_min"].values[0] if len(m) > 0 else np.nan
                if np.isnan(wt):
                    continue
                pts.append({"run": run_name, "acc": acc, "wall_min": wt})
            if not pts:
                continue
            pf = (
                pd.DataFrame(pts)
                .groupby("run", as_index=False)
                .agg(
                    acc_mean=("acc", "mean"),
                    acc_std=("acc", _std0),
                    wall_min_mean=("wall_min", "mean"),
                    wall_min_std=("wall_min", _std0),
                )
            )
            ax.errorbar(
                pf["wall_min_mean"],
                pf["acc_mean"],
                xerr=pf["wall_min_std"],
                yerr=pf["acc_std"],
                fmt="none",
                ecolor="gray",
                alpha=0.5,
                zorder=3,
            )
            sns.scatterplot(
                data=pf,
                x="wall_min_mean",
                y="acc_mean",
                hue="run",
                s=80,
                zorder=5,
                ax=ax,
                legend=(ax_i == n_eval - 1),
                palette=palette,
                hue_order=run_order,
            )
            for _, r in pf.iterrows():
                ax.annotate(
                    r["run"].replace(" (bs=256)", "").split("/")[-1],
                    (r["wall_min_mean"], r["acc_mean"]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=7,
                )
            # Pareto frontier (upper-left is better)
            pf_s = pf.sort_values("wall_min_mean")
            fx: list[float] = []
            fy: list[float] = []
            best = -1.0
            for _, r in pf_s.iterrows():
                if r["acc_mean"] > best:
                    fx.append(r["wall_min_mean"])
                    fy.append(r["acc_mean"])
                    best = r["acc_mean"]
            ax.plot(fx, fy, "r--", alpha=0.5, linewidth=1.5)
            ax.set_title(f"Step {eval_step}")
            ax.set_xlabel("Wall time (min)")
            if ax_i == 0:
                ax.set_ylabel("Avg@32 (%)")
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            "Pareto frontier evolution: Accuracy vs Time", fontsize=12, y=1.02
        )
        fig.tight_layout()
        return fig

    elif metric == "kl":
        fig, axes_arr = plt.subplots(
            1, n_eval, figsize=(4 * n_eval, 4), sharey=True
        )
        if n_eval == 1:
            axes_arr = [axes_arr]
        for ax_i, eval_step in enumerate(eval_steps):
            ax = axes_arr[ax_i]
            pts = []
            for keys, tgrp in train_df.groupby(group_cols):
                run_name = keys[0] if isinstance(keys, tuple) else keys
                tgrp = tgrp.sort_values("step")
                smoothed_kl = tgrp["kl_loss"].rolling(
                    SMOOTH_WINDOW, min_periods=1, center=True
                ).mean()
                row_idx = tgrp[tgrp["step"] == eval_step].index
                if len(row_idx) == 0:
                    continue
                kl = smoothed_kl.loc[row_idx[0]]
                wt = tgrp.loc[row_idx[0], "cumtime_min"]
                if pd.isna(kl) or pd.isna(wt):
                    continue
                pts.append({"run": run_name, "kl": kl, "wall_min": wt})
            if not pts:
                continue
            pf = (
                pd.DataFrame(pts)
                .groupby("run", as_index=False)
                .agg(
                    kl_mean=("kl", "mean"),
                    kl_std=("kl", _std0),
                    wall_min_mean=("wall_min", "mean"),
                    wall_min_std=("wall_min", _std0),
                )
            )
            ax.errorbar(
                pf["wall_min_mean"],
                pf["kl_mean"],
                xerr=pf["wall_min_std"],
                yerr=pf["kl_std"],
                fmt="none",
                ecolor="gray",
                alpha=0.5,
                zorder=3,
            )
            sns.scatterplot(
                data=pf,
                x="wall_min_mean",
                y="kl_mean",
                hue="run",
                s=80,
                zorder=5,
                ax=ax,
                legend=(ax_i == n_eval - 1),
                palette=palette,
                hue_order=run_order,
            )
            for _, r in pf.iterrows():
                ax.annotate(
                    r["run"].replace(" (bs=256)", "").split("/")[-1],
                    (r["wall_min_mean"], r["kl_mean"]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=7,
                )
            # Pareto frontier (lower-left is better)
            pf_s = pf.sort_values("wall_min_mean")
            fx = []
            fy = []
            best_val = float("inf")
            for _, r in pf_s.iterrows():
                if r["kl_mean"] < best_val:
                    fx.append(r["wall_min_mean"])
                    fy.append(r["kl_mean"])
                    best_val = r["kl_mean"]
            ax.plot(fx, fy, "r--", alpha=0.5, linewidth=1.5)
            ax.set_title(f"Step {eval_step}")
            ax.set_xlabel("Wall time (min)")
            if ax_i == 0:
                ax.set_ylabel("KL loss")
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            "Pareto frontier evolution: KL loss vs Time", fontsize=12, y=1.02
        )
        fig.tight_layout()
        return fig

    return None


def plot_efficiency(data: RunData) -> plt.Figure | None:
    """2x2 efficiency: acc vs wall time, time-to-threshold, AUC, Pareto.

    Returns Figure, or None if data is insufficient.
    """
    train_df = data.train_df
    eval_df = data.eval_df
    palette = data.run_palette
    run_order = data.run_order

    if eval_df.empty or train_df.empty:
        return None

    # Use the dataset with real time-series coverage for efficiency plots.
    # Extra post-eval benchmark rows are frequently final-only and can sort
    # before the primary eval dataset (e.g. AIME_25 before AIME_2024).
    if "dataset" in eval_df.columns:
        primary_ds = _select_efficiency_dataset(train_df, eval_df)
        if primary_ds is None:
            return None
        eval_df = eval_df[eval_df["dataset"] == primary_ds]

    ewt = _build_eval_with_time(train_df, eval_df)
    group_cols = ["run", "source_run"] if "source_run" in ewt.columns else ["run"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Plot 1: Accuracy vs Wall Time ---
    ax = axes[0, 0]
    _plot_mean_std_lines(
        ax,
        ewt.dropna(subset=["wall_min"]),
        "wall_min",
        "avg_at_n",
        palette=palette,
        run_order=run_order,
        markersize=5,
    )
    ax.set_xlabel("Wall time (min)")
    ax.set_ylabel("Avg@32 (%)")
    ax.set_title("Accuracy vs Wall Time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Time-to-Threshold ---
    ax = axes[0, 1]
    thresholds = [25, 28, 30, 33, 35]
    ttt_data: list[dict] = []
    for keys, grp in ewt.groupby(group_cols):
        run_name = keys[0] if isinstance(keys, tuple) else keys
        source_run = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else "__single__"
        grp = grp.dropna(subset=["wall_min"]).sort_values("wall_min")
        for thresh in thresholds:
            reached = grp[grp["avg_at_n"] >= thresh]
            t = reached["wall_min"].min() if len(reached) > 0 else np.nan
            ttt_data.append(
                {"run": run_name, "source_run": source_run, "threshold": thresh, "time_min": t}
            )
    ttt_df = pd.DataFrame(ttt_data)
    _plot_mean_std_lines(
        ax,
        ttt_df.dropna(subset=["time_min"]),
        "threshold",
        "time_min",
        palette=palette,
        run_order=run_order,
        markers=["s"],
        linestyles=["-"],
        markersize=5,
    )
    ax.set_xlabel("Accuracy threshold (%)")
    ax.set_ylabel("Time to reach (min)")
    ax.set_title("Time-to-Threshold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: AUC (Accuracy x Time) ---
    ax = axes[1, 0]
    auc_data: list[dict] = []
    for keys, grp in ewt.groupby(group_cols):
        run_name = keys[0] if isinstance(keys, tuple) else keys
        grp = grp.dropna(subset=["wall_min"]).sort_values("wall_min")
        if len(grp) < 2:
            continue
        auc = np.trapz(grp["avg_at_n"].values, grp["wall_min"].values)
        total_min = grp["wall_min"].max()
        avg_acc = auc / total_min if total_min > 0 else 0
        auc_data.append({"run": run_name, "avg_acc": avg_acc})
    ax.set_xlabel("Time-weighted avg accuracy (%)")
    ax.set_title("AUC: Time-weighted Average Accuracy")
    if auc_data:
        auc_df = (
            pd.DataFrame(auc_data)
            .groupby("run", as_index=False)
            .agg(avg_acc_mean=("avg_acc", "mean"), avg_acc_std=("avg_acc", _std0))
            .sort_values("avg_acc_mean", ascending=True)
        )
        bars = ax.barh(
            auc_df["run"].str.replace(" (bs=256)", ""),
            auc_df["avg_acc_mean"],
            xerr=auc_df["avg_acc_std"],
        )
        for bar, val, std in zip(bars, auc_df["avg_acc_mean"], auc_df["avg_acc_std"]):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} ± {std:.1f}",
                va="center",
                fontsize=9,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Need ≥2 time-aligned eval points per run",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            alpha=0.8,
        )

    # --- Plot 4: Pareto Frontier (Time vs Best Accuracy) ---
    ax = axes[1, 1]
    pareto_data: list[dict] = []
    for keys, grp in ewt.groupby(group_cols):
        run_name = keys[0] if isinstance(keys, tuple) else keys
        grp = grp.dropna(subset=["wall_min"])
        if grp.empty:
            continue
        pareto_data.append(
            {"run": run_name, "best_acc": grp["avg_at_n"].max(), "total_min": grp["wall_min"].max()}
        )
    ax.set_xlabel("Total wall time (min)")
    ax.set_ylabel("Best Avg@32 (%)")
    ax.set_title("Pareto: Time vs Best Accuracy")
    if pareto_data:
        pdf = (
            pd.DataFrame(pareto_data)
            .groupby("run", as_index=False)
            .agg(
                best_acc_mean=("best_acc", "mean"),
                best_acc_std=("best_acc", _std0),
                total_min_mean=("total_min", "mean"),
                total_min_std=("total_min", _std0),
            )
        )
        ax.errorbar(
            pdf["total_min_mean"],
            pdf["best_acc_mean"],
            xerr=pdf["total_min_std"],
            yerr=pdf["best_acc_std"],
            fmt="none",
            ecolor="gray",
            alpha=0.5,
            zorder=3,
        )
        sns.scatterplot(
            data=pdf,
            x="total_min_mean",
            y="best_acc_mean",
            hue="run",
            s=100,
            zorder=5,
            ax=ax,
            legend=False,
            palette=palette,
            hue_order=run_order,
        )
        for _, row in pdf.iterrows():
            name = row["run"].replace(" (bs=256)", "")
            ax.annotate(
                name,
                (row["total_min_mean"], row["best_acc_mean"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        # Pareto frontier
        pdf_sorted = pdf.sort_values("total_min_mean")
        frontier_x = [pdf_sorted.iloc[0]["total_min_mean"]]
        frontier_y = [pdf_sorted.iloc[0]["best_acc_mean"]]
        best_so_far = pdf_sorted.iloc[0]["best_acc_mean"]
        for _, row in pdf_sorted.iloc[1:].iterrows():
            if row["best_acc_mean"] > best_so_far:
                frontier_x.append(row["total_min_mean"])
                frontier_y.append(row["best_acc_mean"])
                best_so_far = row["best_acc_mean"]
        ax.plot(
            frontier_x,
            frontier_y,
            "r--",
            alpha=0.5,
            linewidth=1.5,
            label="Pareto frontier",
        )
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No time-aligned eval points",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            alpha=0.8,
        )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _dataset_display_name(dataset: str) -> str:
    """Return compact labels for common eval datasets."""
    if dataset in {"hf:Maxwell-Jia/AIME_2024", "hf:HuggingFaceH4/aime_2024", "hf:yentinglin/aime_2024"}:
        return "AIME24"
    if dataset in {"hf:yentinglin/aime_2025", "hf:math-ai/aime25"}:
        return "AIME25"
    return dataset.split("/")[-1] if "/" in dataset else dataset


def _near_zero_filtered_lower_ylim(values, threshold: float = 1.0) -> float | None:
    """Return a lower y-limit while ignoring only near-zero values.

    This keeps failed/empty eval points from compressing the plot, without
    clipping legitimate low-performing runs. The plotted data is unchanged.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    inliers = arr[arr > threshold]
    if inliers.size == 0:
        inliers = arr

    ymin = float(np.min(inliers))
    ymax = float(np.max(inliers))
    span = max(ymax - ymin, 1.0)
    return max(0.0, ymin - 0.08 * span)


def plot_accuracy_vs_staleness(
    data: RunData,
    datasets: str | list[str] | tuple[str, ...] | set[str] | None = None,
    checkpoint: str = "best",
    ax: plt.Axes | None = None,
    title: str | None = None,
    robust_ymin: bool = False,
    near_zero_threshold: float = 1.0,
) -> plt.Figure | None:
    """Plot best or final eval accuracy against configured staleness / step-off.

    Args:
        data: Loaded run/eval data.
        datasets: Optional dataset name or collection of dataset names to keep.
            When provided, only matching eval datasets are included.
        checkpoint: ``"best"`` for max accuracy across eval steps, or
            ``"final"`` for the last evaluated checkpoint in each run.
        ax: Optional Matplotlib axis to draw into. When omitted, a new figure is
            created.
        title: Optional axis title.
        robust_ymin: If true, set the lower y-axis bound after ignoring only
            near-zero accuracy values.
        near_zero_threshold: Accuracy values at or below this percentage are
            ignored when ``robust_ymin`` chooses the lower axis bound.
    """
    checkpoint = checkpoint.lower()
    if checkpoint not in {"best", "final"}:
        raise ValueError(f"checkpoint must be 'best' or 'final', got {checkpoint!r}")

    eval_df = data.eval_df

    if eval_df.empty or not data.run_dirs:
        return None

    if datasets is None:
        dataset_filter = None
    elif isinstance(datasets, str):
        dataset_filter = {datasets}
    else:
        dataset_filter = set(datasets)

    rows: list[dict] = []
    results_dir = data.results_dir
    labels = data.labels or {}

    for run_dir in data.run_dirs:
        run_name = run_dir.relative_to(results_dir).as_posix()
        run_label = labels.get(run_name, run_name)
        cfg = load_config(run_dir, results_dir)
        threshold = _get_staleness_or_stepoff(cfg, run_name)
        if threshold is None:
            continue

        run_eval = eval_df[eval_df["run"] == run_label]
        if "source_run" in run_eval.columns:
            run_eval = run_eval[run_eval["source_run"] == run_name]
        if run_eval.empty:
            continue

        if dataset_filter is not None and "dataset" in run_eval.columns:
            run_eval = run_eval[run_eval["dataset"].isin(dataset_filter)]
            if run_eval.empty:
                continue

        for dataset, grp in run_eval.groupby(
            "dataset" if "dataset" in run_eval.columns else lambda _: "primary"
        ):
            grp = grp.dropna(subset=["avg_at_n"])
            if grp.empty:
                continue
            if checkpoint == "best":
                eval_accuracy = grp["avg_at_n"].max()
            else:
                sort_cols = ["step"] if "step" in grp.columns else None
                final_grp = grp.sort_values(sort_cols, na_position="last") if sort_cols else grp
                eval_accuracy = final_grp["avg_at_n"].iloc[-1]
            rows.append(
                {
                    "run": run_label,
                    "source_run": run_name,
                    "family": re.split(r"\s+(?:st|so)=", run_label, maxsplit=1)[0].strip(),
                    "dataset": dataset,
                    "staleness_threshold": threshold,
                    "accuracy": eval_accuracy,
                }
            )

    if not rows:
        return None

    summary = (
        pd.DataFrame(rows)
        .groupby(["family", "dataset", "staleness_threshold"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", _std0),
        )
        .sort_values(["dataset", "family", "staleness_threshold"])
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    families = list(dict.fromkeys(summary["family"]))
    family_colors = {
        family: data.run_palette.get(
            next((r for r in data.run_order if r.startswith(family)), None), None
        )
        for family in families
    }
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    multi_dataset = summary["dataset"].nunique() > 1
    for idx, ((family, dataset), grp) in enumerate(summary.groupby(["family", "dataset"], sort=False)):
        dataset_short = _dataset_display_name(dataset) if isinstance(dataset, str) else dataset
        label = f"{family} — {dataset_short}" if multi_dataset else family
        ax.plot(
            grp["staleness_threshold"],
            grp["accuracy_mean"],
            marker=markers[idx % len(markers)],
            linewidth=1.8,
            markersize=6,
            label=label,
            color=family_colors.get(family),
        )
        if (grp["accuracy_std"] > 0).any():
            ax.fill_between(
                grp["staleness_threshold"],
                grp["accuracy_mean"] - grp["accuracy_std"],
                grp["accuracy_mean"] + grp["accuracy_std"],
                alpha=0.18,
                color=family_colors.get(family),
            )

    xticks = sorted(summary["staleness_threshold"].unique())
    if any(x <= 0 for x in xticks):
        ax.set_xscale("symlog", base=2, linthresh=1)
        ax.set_xlim(left=0, right=max(xticks) * 1.05 if max(xticks) > 0 else 1)
    else:
        ax.set_xscale("log", base=2)
        ax.set_xlim(left=min(xticks) / 1.05, right=max(xticks) * 1.05)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _pos: (
                f"{int(round(x))}" if np.isclose(x, round(x)) else f"{x:g}"
            )
        )
    )
    ax.set_xlabel("Staleness / step-off")
    checkpoint_label = "Best checkpoint" if checkpoint == "best" else "Final evaluated checkpoint"
    ax.set_ylabel(f"{checkpoint_label} accuracy (%)")
    if robust_ymin:
        ymin = _near_zero_filtered_lower_ylim(
            summary["accuracy_mean"], threshold=near_zero_threshold)
        if ymin is not None:
            _, ymax = ax.get_ylim()
            ax.set_ylim(bottom=ymin, top=ymax)
    if title is None:
        dataset_note = ""
        if dataset_filter is not None and len(dataset_filter) == 1:
            only_dataset = next(iter(dataset_filter))
            dataset_note = f" — {_dataset_display_name(only_dataset)} only"
        title = f"{checkpoint_label} eval accuracy vs staleness / step-off{dataset_note}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_best_accuracy_vs_staleness(
    data: RunData,
    datasets: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> plt.Figure | None:
    """Plot best eval accuracy against configured staleness / step-off."""
    return plot_accuracy_vs_staleness(data, datasets=datasets, checkpoint="best")


def plot_final_accuracy_vs_staleness(
    data: RunData,
    datasets: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> plt.Figure | None:
    """Plot final-checkpoint eval accuracy against configured staleness / step-off."""
    return plot_accuracy_vs_staleness(data, datasets=datasets, checkpoint="final")


def get_efficiency_summary(data: RunData) -> pd.DataFrame:
    """Compute efficiency summary table.

    Returns a DataFrame with one row per run, with per-dataset accuracy columns.
    """
    train_df = data.train_df
    eval_df = data.eval_df
    run_totals = data.totals
    run_counts = data.run_counts

    if eval_df.empty or train_df.empty:
        return pd.DataFrame()

    # Discover datasets
    datasets = sorted(eval_df["dataset"].unique()) if "dataset" in eval_df.columns else ["primary"]

    ewt = _build_eval_with_time(train_df, eval_df)

    # Use the dataset with real time-series coverage for time-based metrics;
    # final-only post-eval benchmarks still get their own accuracy columns.
    primary_ds = _select_efficiency_dataset(train_df, eval_df) or datasets[0]

    eff_rows: list[dict] = []
    for run_name, run_grp in sorted(
        ewt.groupby("run"), key=lambda x: _extract_staleness(x[0])
    ):
        row_dict: dict = {
            "Run": run_name.replace(" (bs=256)", ""),
            "Runs": run_counts.get(run_name, run_grp["source_run"].nunique() if "source_run" in run_grp.columns else 1),
        }

        # Per-dataset best accuracy columns
        for ds in datasets:
            ds_grp = run_grp[run_grp["dataset"] == ds]
            ds_short = ds.split("/")[-1] if "/" in ds else ds
            if ds_grp.empty:
                row_dict[f"Best% {ds_short}"] = "—"
                row_dict[f"Final% {ds_short}"] = "—"
            else:
                per_source = (
                    ds_grp.dropna(subset=["wall_min"])
                    .sort_values(["source_run", "step"] if "source_run" in ds_grp.columns else ["step"])
                    .groupby("source_run" if "source_run" in ds_grp.columns else lambda _: "__single__")
                    .agg(best_acc=("avg_at_n", "max"), final_acc=("avg_at_n", "last"))
                    .reset_index(drop=True)
                )
                row_dict[f"Best% {ds_short}"] = _format_mean_std(per_source["best_acc"])
                row_dict[f"Final% {ds_short}"] = _format_mean_std(per_source["final_acc"])

        # Time-based metrics from primary dataset
        grp = run_grp[run_grp["dataset"] == primary_ds]
        grp = grp.dropna(subset=["wall_min"]).sort_values("wall_min")
        if grp.empty:
            eff_rows.append(row_dict)
            continue

        per_source_wall = (
            grp.groupby("source_run" if "source_run" in grp.columns else lambda _: "__single__")
            .agg(total_min=("wall_min", "max"))
            .reset_index(drop=True)
        )
        train_per_source = (
            train_df[train_df["run"] == run_name]
            .groupby("source_run" if "source_run" in train_df.columns else lambda _: "__single__")
            .agg(train_end_min=("cumtime_min", "max"))
            .reset_index(drop=True)
        )
        totals_list = run_totals.get(run_name, [])
        eval_minutes = [t.get("total_eval_s", np.nan) / 60 for t in totals_list]
        row_dict.update({
            "Total (min)": _format_mean_std(per_source_wall["total_min"], "{:.0f}"),
            "Train end (min)": _format_mean_std(train_per_source["train_end_min"], "{:.0f}"),
            "Eval time (min)": _format_mean_std(eval_minutes, "{:.1f}"),
        })
        eff_rows.append(row_dict)
    return pd.DataFrame(eff_rows)

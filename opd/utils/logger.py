"""Logging to filesystem (JSONL) and optionally ClearML / W&B / Aim.

Each log call writes one JSON line to the results file and (if enabled)
reports scalars to remote backends.
"""

import json
import os
import time
from errno import EDQUOT, ENOSPC


class Logger:
    """Multi-backend logger: JSONL file + optional ClearML / W&B / Aim."""

    def __init__(self, results_path, config=None, backends=None, run_name=None,
                 resume=False):
        """
        Args:
            results_path: Path to JSONL output file.
            config: Full config dict (logged as first line + connected to remote backends).
            backends: List of backend names to enable, e.g. ["clearml", "wandb", "aim"].
                      None or empty = JSONL only.
            run_name: Human-readable run name for remote backends (e.g. "001_first_real_test/run_3gpu_one_step_off").
            resume: If True, append to existing log file instead of overwriting.
        """
        self._path = results_path
        self._f = None
        self._clearml_logger = None
        self._wandb_run = None
        self._aim_run = None
        self._t0 = time.time()
        self._disk_logging_disabled = False

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        self._f = open(results_path, "a" if resume else "w")

        # Write config as first line for fresh runs only.
        # Resume mode appends step/eval records without duplicating config.
        if config is not None and not resume:
            self._write({"type": "config", "config": config})

        backends = backends or []

        if "clearml" in backends:
            self._init_clearml(config, run_name)

        if "wandb" in backends:
            self._init_wandb(config, run_name)

        if "aim" in backends:
            self._init_aim(config, run_name)

    def _init_clearml(self, config, run_name=None):
        try:
            from clearml import Task
            project = config.get("clearml", {}).get("project", "async-opd") if config else "async-opd"
            task_name = run_name or config.get("clearml", {}).get("task_name") if config else run_name
            task = Task.init(
                project_name=project,
                task_name=task_name,
                auto_connect_frameworks=False,
                reuse_last_task_id=False,
            )
            if config is not None:
                task.connect(config, name="config")
            self._clearml_logger = task.get_logger()
            print(f"[Logger] ClearML task: {task.id} ({task_name})", flush=True)
        except Exception as e:
            print(f"[Logger] ClearML init failed: {e}", flush=True)

    def _init_wandb(self, config, run_name=None):
        try:
            import wandb
            project = config.get("wandb", {}).get("project", "async-opd") if config else "async-opd"
            name = run_name or (config.get("wandb", {}).get("name") if config else None)
            self._wandb_run = wandb.init(
                project=project,
                name=name,
                config=config,
            )
            print(f"[Logger] W&B run: {self._wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[Logger] W&B init failed: {e}", flush=True)

    def _init_aim(self, config, run_name=None):
        try:
            from aim import Run
            aim_cfg = {}
            if config:
                aim_cfg = (config.get("logging") or {}).get("aim") or config.get("aim") or {}
            experiment = aim_cfg.get("experiment", "async-opd")
            repo = aim_cfg.get("repo")  # None = default ./.aim repo
            self._aim_run = Run(experiment=experiment, repo=repo)
            if run_name:
                self._aim_run.name = run_name
            if config is not None:
                # Assign via tuple keys so aim stores leaves as typed scalars
                # instead of wrapping the whole dict under hparams._raw.
                for path, value in _flatten_for_aim(config, ("hparams",)):
                    self._aim_run[path] = value
            print(
                f"[Logger] Aim run: {self._aim_run.hash} (experiment={experiment}, repo={repo or 'default'})",
                flush=True,
            )
        except Exception as e:
            print(f"[Logger] Aim init failed: {e}", flush=True)

    def _write(self, record):
        if self._disk_logging_disabled or self._f is None:
            return
        record["wall_time"] = time.time() - self._t0
        record["timestamp"] = time.time()
        try:
            self._f.write(json.dumps(record) + "\n")
            self._f.flush()
        except OSError as e:
            if e.errno in (ENOSPC, EDQUOT):
                print(
                    f"[Logger] Disabling file logging after quota/space error on {self._path}: {e}",
                    flush=True,
                )
                self._disk_logging_disabled = True
                try:
                    try:
                        self._f.close()
                    except OSError:
                        pass
                finally:
                    self._f = None
                return
            raise

    def _report_remote(self, step, metrics, series):
        if self._clearml_logger:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._clearml_logger.report_scalar(
                        title=k, series=series, iteration=step, value=v,
                    )
        if self._wandb_run:
            self._wandb_run.log(
                {f"{series}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                step=step,
            )
        if self._aim_run:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._aim_run.track(
                        v, name=k, step=step, context={"subset": series},
                    )

    def log_step(self, step, metrics):
        """Log a training step's metrics."""
        self._write({"type": "train", "step": step, **metrics})
        self._report_remote(step, metrics, "train")

    def log_eval(self, step, metrics):
        """Log evaluation metrics."""
        self._write({"type": "eval", "step": step, **metrics})
        self._report_remote(step, metrics, "eval")

    def close(self):
        if self._f:
            self._f.close()
            self._f = None
        if self._wandb_run:
            self._wandb_run.finish()
            self._wandb_run = None
        if self._aim_run:
            self._aim_run.close()
            self._aim_run = None


def _flatten_for_aim(value, path):
    """Yield (tuple_path, leaf_value) so aim stores typed scalars rather than
    wrapping a nested dict under `<path>._raw`. Lists of primitives are kept
    intact; lists containing dicts are stringified per element to keep aim happy.
    Skips None values."""
    if value is None:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            ks = str(k)
            if ks.startswith("_"):  # skip private/round-trip fields like OPDConfig._raw
                continue
            yield from _flatten_for_aim(v, path + (ks,))
        return
    if isinstance(value, (list, tuple)):
        if all(isinstance(x, (str, int, float, bool)) or x is None for x in value):
            yield path, list(value)
        else:
            yield path, [str(x) for x in value]
        return
    if isinstance(value, (str, int, float, bool)):
        yield path, value
        return
    yield path, str(value)

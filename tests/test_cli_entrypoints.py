import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_console_scripts_point_to_packaged_cli_modules():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    scripts = pyproject["project"]["scripts"]
    assert scripts["opd-train"] == "opd.cli.train:main"
    assert scripts["opd-eval"] == "opd.cli.eval:main"
    assert scripts["opd-train"] != "run:main"
    assert scripts["opd-eval"] != "eval:main"
    py_modules = set(pyproject.get("tool", {}).get("setuptools", {}).get("py-modules", []))
    assert "run" not in py_modules
    assert "eval" not in py_modules


def test_packaged_cli_entry_modules_import():
    for module_name in ("opd.cli.train", "opd.cli.eval"):
        module = importlib.import_module(module_name)
        assert callable(module.main)


def test_packaged_cli_modules_own_implementations():
    train_source = (REPO_ROOT / "opd/cli/train.py").read_text()
    eval_source = (REPO_ROOT / "opd/cli/eval.py").read_text()

    assert "def _derive_run_dir" in train_source
    assert "def evaluate_model" in eval_source
    assert "from importlib import import_module" not in train_source
    assert "from importlib import import_module" not in eval_source


def test_training_cli_docstring_names_installed_and_module_commands():
    train_source = (REPO_ROOT / "opd/cli/train.py").read_text()

    assert "Installed usage:" in train_source
    assert "opd-train --config configs/examples/opd_qwen3_1.7b.yaml" in train_source
    assert "Module usage:" in train_source
    assert "python -m opd.cli.train --config configs/examples/opd_qwen3_1.7b.yaml" in train_source


def _run_help(command, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def _run_module_help(module_name, tmp_path):
    return _run_help([sys.executable, "-m", module_name, "--help"], tmp_path)


def test_train_entry_module_help_exits_without_launching_training(tmp_path):
    result = _run_module_help("opd.cli.train", tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "On-Policy Distillation" in result.stdout
    assert "--config" in result.stdout


def test_eval_entry_module_help_exits_without_launching_eval(tmp_path):
    result = _run_module_help("opd.cli.eval", tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Evaluate baseline model performance" in result.stdout
    assert "--score-only" in result.stdout

"""Public API boundary regression tests."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_package_exposes_minimal_stable_api() -> None:
    import opd
    from opd.coordinator.factory import create_coordinator

    assert opd.__version__ == "0.1.0"
    assert opd.__all__ == ("create_coordinator",)
    assert opd.create_coordinator is create_coordinator


def test_root_package_import_is_lightweight(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, opd; print(opd.__all__); print('torch' in sys.modules); print('vllm' in sys.modules)",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines == ["('create_coordinator',)", "False", "False"]


def test_pipeline_compatibility_exports_are_explicit() -> None:
    pipeline = importlib.import_module("opd.pipeline")

    assert pipeline.__all__ == pipeline.PUBLIC_API + pipeline.COMPATIBILITY_API
    assert pipeline.PUBLIC_API == ("create_coordinator",)
    assert "FusedHybridSyncCoordinator" in pipeline.COMPATIBILITY_API
    assert "pad_teacher" in pipeline.COMPATIBILITY_API
    assert "RayRolloutProxy" not in pipeline.__all__
    for name in pipeline.__all__:
        assert hasattr(pipeline, name), name

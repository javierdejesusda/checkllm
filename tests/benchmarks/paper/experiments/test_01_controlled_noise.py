"""Tests for the controlled-noise validation experiment (Task C1-zero).

The experiment lives at ``benchmarks/paper/experiments/01_controlled_noise.py``.
Because the module name starts with a digit it is not a valid Python
identifier and cannot be imported with ``import``; we load it via
:func:`importlib.util.spec_from_file_location`.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path
from types import ModuleType


_EXPERIMENT_PATH = (
    Path(__file__).resolve().parents[4]
    / "benchmarks"
    / "paper"
    / "experiments"
    / "01_controlled_noise.py"
)


def _load_experiment_module() -> ModuleType:
    """Load the experiment script as a module by file path.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the experiment script is missing.
        ImportError: If the module spec cannot be built.
    """
    module_name = "controlled_noise_experiment"
    spec = importlib.util.spec_from_file_location(module_name, _EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {_EXPERIMENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so dataclass annotations can resolve via
    # ``sys.modules[cls.__module__]``.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_runs_to_completion_on_smoke_subset(tmp_path: Path) -> None:
    """Smoke run with --limit-tasks=2 must produce a manifest with the expected shape."""
    module = _load_experiment_module()
    output_dir = tmp_path / "01_controlled_noise"
    summary = module.run_experiment(output_dir=output_dir, limit_tasks=2)

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json was not written"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 2 domains x 2 tasks x 3 seeds x 5 noise levels = 60.
    assert len(manifest["rows"]) == 60
    assert summary["total_rows"] == 60

    expected_keys = {
        "experiment_id",
        "model",
        "benchmark",
        "domain",
        "seed",
        "task_id",
        "noise_level",
        "noise_params",
        "trajectory_score",
        "passed",
        "reference_action_count",
        "predicted_action_count",
        "trajectory_path",
        "timestamp_utc",
        "model_version_sha",
        "benchmark_sha",
        "temperature",
        "mean_latency_ms",
        "total_cost_usd",
    }
    for row in manifest["rows"]:
        missing = expected_keys - set(row.keys())
        assert not missing, f"Row missing required keys: {missing}"


def test_clean_noise_level_produces_perfect_overall(tmp_path: Path) -> None:
    """All ``clean`` noise rows must score >= 0.99 on overall."""
    module = _load_experiment_module()
    output_dir = tmp_path / "01_controlled_noise"
    module.run_experiment(output_dir=output_dir, limit_tasks=2)

    jsonl_path = output_dir / "trajectories.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line]
    clean_rows = [r for r in rows if r["noise_level"] == "clean"]
    assert clean_rows, "no clean rows recorded"
    for row in clean_rows:
        assert row["overall"] >= 0.99, (
            f"clean noise produced non-perfect overall: "
            f"{row['domain']}/{row['task_id']}/seed={row['seed']} -> {row['overall']}"
        )


def test_severe_noise_degrades_overall(tmp_path: Path) -> None:
    """Mean overall at ``severe`` must be strictly less than at ``clean`` per domain."""
    module = _load_experiment_module()
    output_dir = tmp_path / "01_controlled_noise"
    module.run_experiment(output_dir=output_dir, limit_tasks=2)

    jsonl_path = output_dir / "trajectories.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line]
    for domain in ("airline", "retail"):
        clean_scores = [r["overall"] for r in rows if r["domain"] == domain and r["noise_level"] == "clean"]
        severe_scores = [r["overall"] for r in rows if r["domain"] == domain and r["noise_level"] == "severe"]
        assert clean_scores and severe_scores
        assert statistics.mean(severe_scores) < statistics.mean(clean_scores), (
            f"severe noise did not degrade overall in {domain}: "
            f"clean={statistics.mean(clean_scores):.3f} severe={statistics.mean(severe_scores):.3f}"
        )


def test_monotonicity_check_in_summary(tmp_path: Path) -> None:
    """The aggregate summary must report monotonic degradation in both domains."""
    module = _load_experiment_module()
    output_dir = tmp_path / "01_controlled_noise"
    module.run_experiment(output_dir=output_dir, limit_tasks=2)

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "monotonicity_check" in summary
    for domain in ("airline", "retail"):
        assert summary["monotonicity_check"][domain] is True, (
            f"monotonicity check failed for {domain}: "
            f"{summary.get('domain_noise_means', {}).get(domain)}"
        )


def test_jsonl_rows_match_manifest_count(tmp_path: Path) -> None:
    """Number of rows in trajectories.jsonl must equal manifest row count."""
    module = _load_experiment_module()
    output_dir = tmp_path / "01_controlled_noise"
    module.run_experiment(output_dir=output_dir, limit_tasks=2)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    jsonl_lines = [
        line
        for line in (output_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(jsonl_lines) == len(manifest["rows"])

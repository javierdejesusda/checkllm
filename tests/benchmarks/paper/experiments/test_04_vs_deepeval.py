"""Tests for the head-to-head DeepEval comparison experiment (Task C4-zero).

The experiment lives at ``benchmarks/paper/experiments/04_vs_deepeval.py``.
Because the module name starts with a digit it is not a valid Python
identifier and cannot be imported with ``import``; we load it via
:func:`importlib.util.spec_from_file_location`.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip(
    "deepeval",
    reason="deepeval is an optional dev dep used only for the C4-zero head-to-head experiment",
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPERIMENT_PATH = _REPO_ROOT / "benchmarks" / "paper" / "experiments" / "04_vs_deepeval.py"
_C1_INPUT_FILE = (
    _REPO_ROOT / "benchmarks" / "paper" / "results" / "01_controlled_noise" / "trajectories.jsonl"
)


def _load_experiment_module() -> ModuleType:
    """Load the C4 experiment script as a module by file path.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the experiment script is missing.
        ImportError: If the module spec cannot be built.
    """
    module_name = "vs_deepeval_experiment"
    spec = importlib.util.spec_from_file_location(module_name, _EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {_EXPERIMENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_runs_to_completion_on_smoke_subset(tmp_path: Path) -> None:
    """Smoke run with --limit-trajectories=10 must produce summary.json with both blocks."""
    module = _load_experiment_module()
    output_dir = tmp_path / "04_vs_deepeval"
    module.run_experiment(
        input_file=_C1_INPUT_FILE,
        output_dir=output_dir,
        limit_trajectories=10,
        n_bootstrap=50,
    )

    summary_path = output_dir / "summary.json"
    assert summary_path.exists(), "summary.json was not written"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "checkllm" in summary, "summary missing 'checkllm' block"
    assert "deepeval" in summary, "summary missing 'deepeval' block"


def test_summary_has_required_statistics(tmp_path: Path) -> None:
    """Both metric blocks must carry pearson_r/spearman_rho/auroc + scalars."""
    module = _load_experiment_module()
    output_dir = tmp_path / "04_vs_deepeval"
    module.run_experiment(
        input_file=_C1_INPUT_FILE,
        output_dir=output_dir,
        limit_trajectories=10,
        n_bootstrap=50,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    for framework in ("checkllm", "deepeval"):
        block = summary[framework]
        for key in ("pearson_r", "spearman_rho", "auroc"):
            assert key in block, f"missing {framework}.{key}"
            stat_block = block[key]
            for sub in ("value", "ci_lower", "ci_upper"):
                assert sub in stat_block, f"missing {framework}.{key}.{sub}"
                assert isinstance(
                    stat_block[sub], (int, float)
                ), f"{framework}.{key}.{sub} must be numeric"
        for scalar_key in (
            "coverage",
            "mean_latency_ms_per_trajectory",
            "mean_abs_error",
            "cost_usd_per_trajectory",
        ):
            assert scalar_key in block, f"missing {framework}.{scalar_key}"
            assert isinstance(
                block[scalar_key], (int, float)
            ), f"{framework}.{scalar_key} must be numeric"


def test_head_to_head_holm_correction(tmp_path: Path) -> None:
    """head_to_head.claims must have 4 items, raw <= corrected, and meta fields."""
    module = _load_experiment_module()
    output_dir = tmp_path / "04_vs_deepeval"
    module.run_experiment(
        input_file=_C1_INPUT_FILE,
        output_dir=output_dir,
        limit_trajectories=10,
        n_bootstrap=50,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    h2h = summary["head_to_head"]
    assert h2h["n_claims"] == 4
    assert h2h["holm_alpha"] == 0.05
    claims = h2h["claims"]
    assert len(claims) == 4, f"expected 4 claims, got {len(claims)}"
    for claim in claims:
        for field in (
            "name",
            "p_value_raw",
            "p_value_holm_corrected",
            "significant_at_0_05",
        ):
            assert field in claim, f"claim missing {field}: {claim}"
        # Holm only inflates p-values.
        assert claim["p_value_holm_corrected"] >= claim["p_value_raw"], (
            f"Holm-corrected p-value < raw p-value for {claim['name']!r}: "
            f"raw={claim['p_value_raw']} corrected={claim['p_value_holm_corrected']}"
        )
        assert isinstance(claim["significant_at_0_05"], bool)


def test_per_trajectory_jsonl_count_matches_summary(tmp_path: Path) -> None:
    """per_trajectory.jsonl row count must equal summary.n_trajectories."""
    module = _load_experiment_module()
    output_dir = tmp_path / "04_vs_deepeval"
    module.run_experiment(
        input_file=_C1_INPUT_FILE,
        output_dir=output_dir,
        limit_trajectories=10,
        n_bootstrap=50,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    per_traj_path = output_dir / "per_trajectory.jsonl"
    rows = [
        json.loads(line)
        for line in per_traj_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == summary["n_trajectories"], (
        f"per_trajectory.jsonl has {len(rows)} rows, summary says " f"{summary['n_trajectories']}"
    )
    # Sanity-check that each row has the fields we promise.
    required = {
        "domain",
        "task_id",
        "seed",
        "noise_level",
        "label",
        "checkllm_score",
        "checkllm_latency_ms",
        "checkllm_abs_error",
        "deepeval_score",
        "deepeval_latency_ms",
        "deepeval_abs_error",
    }
    for row in rows:
        missing = required - set(row.keys())
        assert not missing, f"per_trajectory row missing keys: {missing}"


def test_deepeval_version_recorded(tmp_path: Path) -> None:
    """summary.deepeval_version must be a non-empty semver-ish string."""
    module = _load_experiment_module()
    output_dir = tmp_path / "04_vs_deepeval"
    module.run_experiment(
        input_file=_C1_INPUT_FILE,
        output_dir=output_dir,
        limit_trajectories=10,
        n_bootstrap=50,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    version = summary.get("deepeval_version")
    assert isinstance(
        version, str
    ), f"deepeval_version must be a string, got {type(version).__name__}"
    assert version, "deepeval_version is empty"
    # Loosely match semver-like: e.g. 3.9.7, 1.0.0a1, 2.5.0.dev3.
    assert re.match(r"^\d+\.\d+(\.\d+)?", version), f"deepeval_version not semver-ish: {version!r}"


def test_coverage_is_float_in_unit_interval(tmp_path: Path) -> None:
    """Both metrics' coverage field must be a float in [0, 1]."""
    module = _load_experiment_module()
    output_dir = tmp_path / "04_vs_deepeval"
    module.run_experiment(
        input_file=_C1_INPUT_FILE,
        output_dir=output_dir,
        limit_trajectories=10,
        n_bootstrap=50,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    for framework in ("checkllm", "deepeval"):
        coverage = summary[framework]["coverage"]
        assert isinstance(coverage, (int, float)), f"{framework}.coverage must be numeric"
        assert 0.0 <= coverage <= 1.0, f"{framework}.coverage out of [0, 1]: {coverage}"

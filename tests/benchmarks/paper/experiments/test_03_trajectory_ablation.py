"""Tests for the TrajectoryMetric weight ablation experiment (Task C3-zero).

The experiment lives at ``benchmarks/paper/experiments/03_trajectory_ablation.py``.
Because the module name starts with a digit it is not a valid Python
identifier and cannot be imported with ``import``; we load it via
:func:`importlib.util.spec_from_file_location`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPERIMENT_PATH = (
    _REPO_ROOT
    / "benchmarks"
    / "paper"
    / "experiments"
    / "03_trajectory_ablation.py"
)
_C2_INPUT_DIR = _REPO_ROOT / "benchmarks" / "paper" / "results" / "02_metric_vs_truth"


def _load_experiment_module() -> ModuleType:
    """Load the ablation script as a module by file path.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the experiment script is missing.
        ImportError: If the module spec cannot be built.
    """
    module_name = "trajectory_ablation_experiment"
    spec = importlib.util.spec_from_file_location(module_name, _EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {_EXPERIMENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_runs_to_completion_on_smoke_subset(tmp_path: Path) -> None:
    """Smoke run with --limit-cells=20 must produce 20 grid rows and a summary."""
    module = _load_experiment_module()
    output_dir = tmp_path / "03_ablation"
    module.run_experiment(
        input_dir=_C2_INPUT_DIR,
        output_dir=output_dir,
        limit_cells=20,
    )

    grid_path = output_dir / "grid.jsonl"
    summary_path = output_dir / "summary.json"
    heatmap_path = output_dir / "heatmap.json"
    assert grid_path.exists(), "grid.jsonl was not written"
    assert summary_path.exists(), "summary.json was not written"
    assert heatmap_path.exists(), "heatmap.json was not written"

    rows = [
        json.loads(line)
        for line in grid_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 20, f"expected 20 rows, got {len(rows)}"


def test_grid_size_is_correct(tmp_path: Path) -> None:
    """Full run must yield exactly 1875 - 3 = 1872 grid rows; n_skipped == 3."""
    module = _load_experiment_module()
    output_dir = tmp_path / "03_ablation"
    module.run_experiment(input_dir=_C2_INPUT_DIR, output_dir=output_dir)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_skipped_degenerate"] == 3, (
        f"expected 3 skipped degenerate cells, got {summary['n_skipped_degenerate']}"
    )
    assert summary["n_cells"] == 1872, (
        f"expected 1872 valid cells, got {summary['n_cells']}"
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "grid.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1872, f"grid.jsonl row count: {len(rows)}"


def test_default_config_present_in_summary(tmp_path: Path) -> None:
    """summary['default_config'] must include the 5 knobs + spearman_rho + auroc."""
    module = _load_experiment_module()
    output_dir = tmp_path / "03_ablation"
    module.run_experiment(input_dir=_C2_INPUT_DIR, output_dir=output_dir)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    default = summary["default_config"]
    for key in (
        "ordering_weight",
        "loop_weight",
        "coverage_weight",
        "unexpected_weight",
        "loop_threshold",
        "spearman_rho",
        "auroc",
    ):
        assert key in default, f"default_config missing {key}"


def test_top_5_correctly_sorted(tmp_path: Path) -> None:
    """summary['top_5_by_spearman'] must be exactly 5 rows sorted descending."""
    module = _load_experiment_module()
    output_dir = tmp_path / "03_ablation"
    module.run_experiment(input_dir=_C2_INPUT_DIR, output_dir=output_dir)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    top5 = summary["top_5_by_spearman"]
    assert len(top5) == 5, f"expected 5 rows, got {len(top5)}"
    rhos = [row["spearman_rho"] for row in top5]
    assert rhos == sorted(rhos, reverse=True), (
        f"top_5_by_spearman not sorted descending: {rhos}"
    )

    top5_auroc = summary["top_5_by_auroc"]
    assert len(top5_auroc) == 5
    aurocs = [row["auroc"] for row in top5_auroc]
    assert aurocs == sorted(aurocs, reverse=True), (
        f"top_5_by_auroc not sorted descending: {aurocs}"
    )


def test_default_within_5pct_of_best(tmp_path: Path) -> None:
    """The library default must be within 5% (relative) of the best Spearman cell."""
    module = _load_experiment_module()
    output_dir = tmp_path / "03_ablation"
    module.run_experiment(input_dir=_C2_INPUT_DIR, output_dir=output_dir)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rel_gap = summary["best_vs_default_gap"]["spearman_rho_relative"]
    assert abs(rel_gap) < 0.05, (
        f"default vs best Spearman relative gap exceeded 5%: {rel_gap:.4f}"
    )


def test_pareto_check_is_bool(tmp_path: Path) -> None:
    """default_is_pareto_optimal must be a Python bool (not numpy bool)."""
    module = _load_experiment_module()
    output_dir = tmp_path / "03_ablation"
    module.run_experiment(input_dir=_C2_INPUT_DIR, output_dir=output_dir)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    val = summary["default_is_pareto_optimal"]
    assert isinstance(val, bool), (
        f"default_is_pareto_optimal must be bool, got {type(val).__name__}"
    )


def test_skips_all_zero_weights(tmp_path: Path) -> None:
    """The (0,0,0,0) cell must be flagged degenerate; the full grid skips it.

    Directly probes the ``_is_degenerate`` helper for every loop_threshold
    value, then confirms via a full-grid run that no degenerate cell
    leaks into ``grid.jsonl``.
    """
    module = _load_experiment_module()

    # Direct probe of the skip predicate for the (0,0,0,0) weight tuple.
    assert module._is_degenerate((0.0, 0.0, 0.0, 0.0)) is True
    # Any non-zero weight should make the cell non-degenerate.
    assert module._is_degenerate((0.1, 0.0, 0.0, 0.0)) is False
    assert module._is_degenerate((0.0, 0.0, 0.0, 1.0)) is False

    # And verify the constructor itself rejects all-zero (matches spec
    # rationale: ``TrajectoryMetricConfig.normalized_weights`` raises).
    import pytest

    with pytest.raises(ValueError):
        module._make_config(0.0, 0.0, 0.0, 0.0, 1).normalized_weights()

    output_dir = tmp_path / "03_ablation"
    module.run_experiment(input_dir=_C2_INPUT_DIR, output_dir=output_dir)

    rows = [
        json.loads(line)
        for line in (output_dir / "grid.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        weights = (
            row["ordering_weight"],
            row["loop_weight"],
            row["coverage_weight"],
            row["unexpected_weight"],
        )
        assert any(w > 0 for w in weights), (
            f"degenerate (0,0,0,0) cell not skipped: {row}"
        )


def test_heatmap_has_six_pairs(tmp_path: Path) -> None:
    """heatmap.json must have 6 keys (one per pair) each holding a 5x5 grid."""
    module = _load_experiment_module()
    output_dir = tmp_path / "03_ablation"
    module.run_experiment(input_dir=_C2_INPUT_DIR, output_dir=output_dir)

    heatmap = json.loads((output_dir / "heatmap.json").read_text(encoding="utf-8"))
    expected_pairs = {
        "ordering__loop",
        "ordering__coverage",
        "ordering__unexpected",
        "loop__coverage",
        "loop__unexpected",
        "coverage__unexpected",
    }
    assert set(heatmap.keys()) == expected_pairs, (
        f"unexpected heatmap keys: {sorted(heatmap.keys())}"
    )
    for pair, payload in heatmap.items():
        grid = payload["grid"]
        assert len(grid) == 5, f"{pair}: outer dim not 5 (got {len(grid)})"
        for row in grid:
            assert len(row) == 5, f"{pair}: inner dim not 5 (got {len(row)})"

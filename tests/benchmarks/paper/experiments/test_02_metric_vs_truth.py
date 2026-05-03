"""Tests for the metric-vs-synthetic-truth correlation experiment (Task C2-zero).

The experiment lives at ``benchmarks/paper/experiments/02_metric_vs_truth.py``.
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


_EXPERIMENT_PATH = (
    Path(__file__).resolve().parents[4]
    / "benchmarks"
    / "paper"
    / "experiments"
    / "02_metric_vs_truth.py"
)


def _load_experiment_module() -> ModuleType:
    """Load the experiment script as a module by file path.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the experiment script is missing.
        ImportError: If the module spec cannot be built.
    """
    module_name = "metric_vs_truth_experiment"
    spec = importlib.util.spec_from_file_location(module_name, _EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {_EXPERIMENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_runs_to_completion_on_smoke_subset(tmp_path: Path) -> None:
    """Smoke run with --limit-tasks=2 must produce a manifest with the expected row count."""
    module = _load_experiment_module()
    output_dir = tmp_path / "02_metric_vs_truth"
    summary = module.run_experiment(
        output_dir=output_dir, limit_tasks=2, n_bootstrap=50
    )

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json was not written"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 2 domains x 2 tasks x 10 seeds x 5 noise levels = 200.
    assert len(manifest["rows"]) == 200
    assert summary["n_trajectories"] == 200


def test_summary_contains_required_statistics(tmp_path: Path) -> None:
    """summary.json must carry pearson_r / spearman_rho / auroc with CIs at the top level."""
    module = _load_experiment_module()
    output_dir = tmp_path / "02_metric_vs_truth"
    module.run_experiment(output_dir=output_dir, limit_tasks=2, n_bootstrap=50)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    overall = summary["overall"]
    for key in ("pearson_r", "spearman_rho", "auroc"):
        assert key in overall, f"missing overall.{key}"
        block = overall[key]
        for sub in ("value", "ci_lower", "ci_upper"):
            assert sub in block, f"missing overall.{key}.{sub}"
            assert isinstance(block[sub], (int, float)), (
                f"overall.{key}.{sub} must be numeric, got {type(block[sub]).__name__}"
            )


def test_synthetic_truth_yields_strong_auroc(tmp_path: Path) -> None:
    """Synthetic truth + monotone metric should produce a very strong signal (AUROC >= 0.85)."""
    module = _load_experiment_module()
    output_dir = tmp_path / "02_metric_vs_truth"
    module.run_experiment(output_dir=output_dir, limit_tasks=2, n_bootstrap=50)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    auroc = summary["overall"]["auroc"]["value"]
    assert auroc >= 0.85, f"AUROC too low for synthetic truth: {auroc:.3f}"


def test_per_domain_breakdown_present(tmp_path: Path) -> None:
    """per_domain must contain airline and retail with the same statistic structure."""
    module = _load_experiment_module()
    output_dir = tmp_path / "02_metric_vs_truth"
    module.run_experiment(output_dir=output_dir, limit_tasks=2, n_bootstrap=50)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    per_domain = summary["per_domain"]
    for domain in ("airline", "retail"):
        assert domain in per_domain, f"missing per_domain.{domain}"
        block = per_domain[domain]
        for key in ("pearson_r", "spearman_rho", "auroc"):
            assert key in block, f"missing per_domain.{domain}.{key}"
            for sub in ("value", "ci_lower", "ci_upper"):
                assert sub in block[key], (
                    f"missing per_domain.{domain}.{key}.{sub}"
                )


def test_per_metric_breakdown_present(tmp_path: Path) -> None:
    """per_metric must contain all 5 sub-score keys, each with the statistic block."""
    module = _load_experiment_module()
    output_dir = tmp_path / "02_metric_vs_truth"
    module.run_experiment(output_dir=output_dir, limit_tasks=2, n_bootstrap=50)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    per_metric = summary["per_metric"]
    for sub_metric in ("ordering", "loops", "coverage", "unexpected", "overall"):
        assert sub_metric in per_metric, f"missing per_metric.{sub_metric}"
        block = per_metric[sub_metric]
        for key in ("pearson_r", "spearman_rho", "auroc"):
            assert key in block, f"missing per_metric.{sub_metric}.{key}"
            for sub in ("value", "ci_lower", "ci_upper"):
                assert sub in block[key], (
                    f"missing per_metric.{sub_metric}.{key}.{sub}"
                )


def test_auroc_gate_evaluation(tmp_path: Path) -> None:
    """auroc_gate_passed must be a bool dict per domain (not just truthy)."""
    module = _load_experiment_module()
    output_dir = tmp_path / "02_metric_vs_truth"
    module.run_experiment(output_dir=output_dir, limit_tasks=2, n_bootstrap=50)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    gate = summary["auroc_gate_passed"]
    assert isinstance(gate, dict), "auroc_gate_passed must be a dict per domain"
    for domain in ("airline", "retail"):
        assert domain in gate, f"auroc_gate_passed missing {domain}"
        # Strict bool check: not just truthy.
        assert isinstance(gate[domain], bool), (
            f"auroc_gate_passed[{domain}] must be a bool, got "
            f"{type(gate[domain]).__name__}"
        )


def test_label_assignment(tmp_path: Path) -> None:
    """Every row in scores.jsonl must satisfy label == 1 iff noise_level == 'clean'."""
    module = _load_experiment_module()
    output_dir = tmp_path / "02_metric_vs_truth"
    module.run_experiment(output_dir=output_dir, limit_tasks=2, n_bootstrap=50)

    scores_path = output_dir / "scores.jsonl"
    rows = [
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "no score rows recorded"
    for row in rows:
        expected_label = 1 if row["noise_level"] == "clean" else 0
        assert row["label"] == expected_label, (
            f"label mismatch: noise_level={row['noise_level']!r} "
            f"label={row['label']} (expected {expected_label})"
        )

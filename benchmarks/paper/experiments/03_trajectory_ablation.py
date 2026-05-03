"""Task C3-zero: TrajectoryMetric weight ablation.

This experiment sweeps the four ``TrajectoryMetric`` sub-score weights
(ordering / loops / coverage / unexpected) and the ``loop_threshold``
across a discrete grid, re-scoring the same 500 trajectories used in
Task C2-zero. For each grid cell we compute Spearman rho and AUROC
between the cell's ``overall`` score and the synthetic binary truth
(``label = 1`` iff ``noise_level == "clean"``). We then identify the
top-5 cells, compare against the library default, and produce
heatmap-ready aggregates for plotting.

Re-scoring is zero API cost: trajectories are regenerated
deterministically by replaying the :class:`StubAgent` with the same
``(seed, drop, repeat, extra)`` configuration that produced the
C2-zero output. We do not call the StubAgent more than once per
trajectory; predictions are cached and re-scored under each grid cell.

Outputs in ``output_dir``:

* ``grid.jsonl`` -- one row per (non-degenerate) grid cell with the
  seven fields ``ordering_weight, loop_weight, coverage_weight,
  unexpected_weight, loop_threshold, spearman_rho, auroc``.
* ``summary.json`` -- aggregate report (``default_config``,
  ``top_5_by_spearman``, ``top_5_by_auroc``, ``best_vs_default_gap``,
  ``default_is_pareto_optimal``, ``weight_correlations``).
* ``heatmap.json`` -- 5x5 mean-Spearman grids for each of the six
  weight pairs, ready for plotting.

Run via:

    python benchmarks/paper/experiments/03_trajectory_ablation.py \
        --input-dir benchmarks/paper/results/02_metric_vs_truth \
        --output-dir benchmarks/paper/results/03_ablation

Note: because the file name starts with a digit it is not a valid
Python identifier; ``python -m benchmarks.paper.experiments.03_trajectory_ablation``
is therefore not supported. Invoke the script by file path instead.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

# Ensure ``benchmarks`` and ``src/checkllm`` are importable when the
# script is invoked by file path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from checkllm.benchmarks import TauBenchTask, load_tau_bench  # noqa: E402
from checkllm.metrics.trajectory_metric import (  # noqa: E402
    TrajectoryMetric,
    TrajectoryMetricConfig,
)

from benchmarks.paper.agents import StubAgent  # noqa: E402


EXPERIMENT_ID = "03_trajectory_ablation"

# Grid axes per spec.
WEIGHT_VALUES: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
LOOP_THRESHOLDS: tuple[int, ...] = (1, 2, 3)
WEIGHT_AXES: tuple[str, ...] = (
    "ordering_weight",
    "loop_weight",
    "coverage_weight",
    "unexpected_weight",
)

# Library default knobs (match TrajectoryMetricConfig defaults).
DEFAULT_ORDERING_WEIGHT = 0.4
DEFAULT_LOOP_WEIGHT = 0.2
DEFAULT_COVERAGE_WEIGHT = 0.25
DEFAULT_UNEXPECTED_WEIGHT = 0.15
DEFAULT_LOOP_THRESHOLD = 2

# Pairs used to build the 6 heatmap projections. Names mirror the
# ``TrajectoryMetricConfig`` field stems (drop the ``_weight`` suffix).
HEATMAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("ordering", "loop"),
    ("ordering", "coverage"),
    ("ordering", "unexpected"),
    ("loop", "coverage"),
    ("loop", "unexpected"),
    ("coverage", "unexpected"),
)


@dataclass(frozen=True)
class NoiseLevel:
    """One named noise regime for the StubAgent (mirrors C1/C2)."""

    name: str
    drop: float
    repeat: float
    extra: float


# Same five regimes as Task C2-zero; trajectories are reproduced
# deterministically by replaying StubAgent with these knobs and the
# C2-zero seeds.
NOISE_LEVELS: tuple[NoiseLevel, ...] = (
    NoiseLevel("clean", drop=0.0, repeat=0.0, extra=0.0),
    NoiseLevel("light", drop=0.10, repeat=0.10, extra=0.20),
    NoiseLevel("medium", drop=0.25, repeat=0.25, extra=0.50),
    NoiseLevel("heavy", drop=0.50, repeat=0.50, extra=0.80),
    NoiseLevel("severe", drop=0.80, repeat=0.80, extra=1.00),
)
SEEDS: tuple[int, ...] = (42, 123, 2026, 7, 11, 17, 31, 53, 97, 1009)
DOMAINS: tuple[str, ...] = ("airline", "retail")


def _load_c2_inputs(
    input_dir: Path,
) -> tuple[list[int], list[NoiseLevel], list[str], int | None]:
    """Read C2-zero's ``manifest.json`` to recover its seeds + noise grid.

    Args:
        input_dir: Path to the C2-zero results directory containing
            ``manifest.json``.

    Returns:
        Tuple ``(seeds, noise_levels, domains, limit_tasks)`` extracted
        from C2-zero. Falls back to module-level defaults when the file
        is missing.
    """
    manifest_path = Path(input_dir) / "manifest.json"
    if not manifest_path.exists():
        return list(SEEDS), list(NOISE_LEVELS), list(DOMAINS), None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = list(manifest.get("seeds", SEEDS))
    raw_levels = manifest.get("noise_levels", [])
    noise_levels = [
        NoiseLevel(
            name=str(lvl["name"]),
            drop=float(lvl["drop"]),
            repeat=float(lvl["repeat"]),
            extra=float(lvl["extra"]),
        )
        for lvl in raw_levels
    ] or list(NOISE_LEVELS)
    domains = list(manifest.get("domains", DOMAINS))
    limit_tasks = manifest.get("limit_tasks")
    return seeds, noise_levels, domains, limit_tasks


def _build_trajectory_pool(
    seeds: list[int],
    noise_levels: list[NoiseLevel],
    domains: list[str],
    limit_tasks: int | None,
) -> list[dict[str, Any]]:
    """Replay StubAgent across the C2-zero grid to recover all trajectories.

    Args:
        seeds: Seeds matching the C2-zero design.
        noise_levels: Five-regime noise schedule.
        domains: Tau-bench domains to enumerate.
        limit_tasks: Optional truncation (used for testing parity).

    Returns:
        A list of dicts, each with ``predicted_names`` (list[str]),
        ``reference_names`` (list[str]), and the binary ``label``.
    """
    domain_tasks: dict[str, list[TauBenchTask]] = {
        d: load_tau_bench(d, limit=limit_tasks) for d in domains
    }
    pool: list[dict[str, Any]] = []
    for domain in domains:
        tasks = domain_tasks[domain]
        for noise in noise_levels:
            for seed in seeds:
                agent = StubAgent(
                    seed=seed,
                    drop=noise.drop,
                    repeat=noise.repeat,
                    extra=noise.extra,
                )
                for task in tasks:
                    predicted = agent.run(
                        reference_actions=task.reference_actions,
                        user_instruction=task.user_instruction,
                        tools=task.tools,
                    )
                    pool.append(
                        {
                            "domain": domain,
                            "task_id": task.task_id,
                            "seed": seed,
                            "noise_level": noise.name,
                            "predicted_names": [
                                str(a.get("name", "")) for a in predicted
                            ],
                            "reference_names": [
                                str(a.get("name", ""))
                                for a in task.reference_actions
                            ],
                            "label": 1 if noise.name == "clean" else 0,
                        }
                    )
    return pool


def _iter_grid_cells() -> Iterable[tuple[float, float, float, float, int]]:
    """Yield every ``(ord_w, loop_w, cov_w, unexp_w, loop_threshold)`` tuple."""
    for combo in itertools.product(WEIGHT_VALUES, repeat=4):
        for thr in LOOP_THRESHOLDS:
            yield (*combo, thr)


def _is_degenerate(weights: tuple[float, float, float, float]) -> bool:
    """Return True iff all four weights are zero (config would raise)."""
    return all(w == 0.0 for w in weights)


def _score_cell(
    pool: list[dict[str, Any]],
    config: TrajectoryMetricConfig,
) -> tuple[float, float]:
    """Compute Spearman rho and AUROC for one grid cell.

    Args:
        pool: Trajectory pool from :func:`_build_trajectory_pool`.
        config: Configured weights + loop_threshold for this cell.

    Returns:
        ``(spearman_rho, auroc)``. Returns ``(nan, nan)`` if either
        statistic is undefined (zero variance or single-class labels).
    """
    scores = np.empty(len(pool), dtype=float)
    labels = np.empty(len(pool), dtype=int)
    for i, traj in enumerate(pool):
        metric = TrajectoryMetric(
            expected_trajectory=traj["reference_names"],
            config=config,
        )
        sub = metric.compute_subscores(traj["predicted_names"])
        scores[i] = float(sub.overall)
        labels[i] = int(traj["label"])
    return _spearman_rho(scores, labels), _auroc_mann_whitney(scores, labels)


def _spearman_rho(scores: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rho between scores and binary labels."""
    if scores.std() == 0.0 or labels.std() == 0.0:
        return float("nan")
    result = stats.spearmanr(scores, labels)
    statistic = getattr(result, "statistic", None)
    if statistic is None:
        statistic = result[0]  # type: ignore[index]
    return float(statistic)


def _auroc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via Mann-Whitney U on average ranks (no sklearn)."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores, method="average")
    sum_ranks_pos = float(ranks[labels == 1].sum())
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def _make_config(
    ordering_w: float,
    loop_w: float,
    coverage_w: float,
    unexpected_w: float,
    loop_threshold: int,
) -> TrajectoryMetricConfig:
    """Construct a :class:`TrajectoryMetricConfig` with the given knobs."""
    return TrajectoryMetricConfig(
        ordering_weight=ordering_w,
        loop_weight=loop_w,
        coverage_weight=coverage_w,
        unexpected_weight=unexpected_w,
        loop_threshold=loop_threshold,
    )


def _row_to_dict(
    ordering_w: float,
    loop_w: float,
    coverage_w: float,
    unexpected_w: float,
    loop_threshold: int,
    spearman: float,
    auroc: float,
) -> dict[str, Any]:
    """Pack one grid cell's result as a JSONL-friendly dict."""
    return {
        "ordering_weight": float(ordering_w),
        "loop_weight": float(loop_w),
        "coverage_weight": float(coverage_w),
        "unexpected_weight": float(unexpected_w),
        "loop_threshold": int(loop_threshold),
        "spearman_rho": float(spearman),
        "auroc": float(auroc),
    }


def _build_heatmap(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build the six pairwise 5x5 mean-Spearman grids.

    Each cell averages over the other two weights and ``loop_threshold``.

    Args:
        rows: All grid rows from this experiment.

    Returns:
        Dict keyed by ``"<a>__<b>"`` (e.g. ``"ordering__loop"``) where
        each value carries ``axes`` (the two axis names), ``values``
        (the discrete weight values), and a 5x5 ``grid`` of mean
        Spearman rho values (NaN entries become ``None``).
    """
    out: dict[str, dict[str, Any]] = {}
    for axis_a, axis_b in HEATMAP_PAIRS:
        key_a = f"{axis_a}_weight"
        key_b = f"{axis_b}_weight"
        grid: list[list[float | None]] = []
        for va in WEIGHT_VALUES:
            row: list[float | None] = []
            for vb in WEIGHT_VALUES:
                vals = [
                    r["spearman_rho"]
                    for r in rows
                    if r[key_a] == va
                    and r[key_b] == vb
                    and not np.isnan(r["spearman_rho"])
                ]
                row.append(float(np.mean(vals)) if vals else None)
            grid.append(row)
        out[f"{axis_a}__{axis_b}"] = {
            "axes": [axis_a, axis_b],
            "values": list(WEIGHT_VALUES),
            "grid": grid,
        }
    return out


def _weight_correlations(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-axis Spearman rho between weight value and grid Spearman rho.

    Indicates which axes drive the metric's correlation with truth.

    Args:
        rows: Per-cell result rows.

    Returns:
        Mapping from each weight axis to ``{"with_spearman": <float>}``.
        Uses ``nan`` when the inputs are degenerate.
    """
    out: dict[str, dict[str, float]] = {}
    rhos = np.asarray([r["spearman_rho"] for r in rows], dtype=float)
    valid = ~np.isnan(rhos)
    for axis in WEIGHT_AXES:
        weights = np.asarray([r[axis] for r in rows], dtype=float)
        x = weights[valid]
        y = rhos[valid]
        if x.std() == 0.0 or y.std() == 0.0 or x.size < 2:
            out[axis] = {"with_spearman": float("nan")}
            continue
        result = stats.spearmanr(x, y)
        statistic = getattr(result, "statistic", None)
        if statistic is None:
            statistic = result[0]  # type: ignore[index]
        out[axis] = {"with_spearman": float(statistic)}
    return out


def _is_pareto_optimal(
    rows: list[dict[str, Any]], default_row: dict[str, Any]
) -> bool:
    """True iff no other cell strictly dominates the default on both metrics.

    A cell strictly dominates iff its Spearman rho and its AUROC are
    both strictly greater than the default's.

    Args:
        rows: All non-degenerate grid rows.
        default_row: The library-default cell (must be in ``rows``).

    Returns:
        Plain Python bool.
    """
    default_rho = default_row["spearman_rho"]
    default_auc = default_row["auroc"]
    if np.isnan(default_rho) or np.isnan(default_auc):
        return False
    for r in rows:
        if r is default_row:
            continue
        if np.isnan(r["spearman_rho"]) or np.isnan(r["auroc"]):
            continue
        if (
            r["spearman_rho"] > default_rho
            and r["auroc"] > default_auc
        ):
            return False
    return True


def run_experiment(
    input_dir: Path | str,
    output_dir: Path | str,
    limit_cells: int | None = None,
) -> dict[str, Any]:
    """Run the trajectory-metric ablation grid.

    Args:
        input_dir: Directory containing the C2-zero ``manifest.json``;
            seeds / noise levels / domains / limit_tasks are sourced
            from there for parity. Falls back to module defaults if the
            manifest is missing (useful in tests).
        output_dir: Output directory. Created if missing; existing
            files are overwritten.
        limit_cells: Optional cap on the number of grid cells (random
            sample without replacement). Used for smoke tests.

    Returns:
        A small summary dict useful for the CLI.
    """
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    grid_path = output_root / "grid.jsonl"
    summary_path = output_root / "summary.json"
    heatmap_path = output_root / "heatmap.json"

    seeds, noise_levels, domains, limit_tasks = _load_c2_inputs(Path(input_dir))
    pool = _build_trajectory_pool(seeds, noise_levels, domains, limit_tasks)

    all_cells = list(_iter_grid_cells())
    n_skipped_degenerate = sum(
        1 for cell in all_cells if _is_degenerate(cell[:4])
    )
    valid_cells = [cell for cell in all_cells if not _is_degenerate(cell[:4])]

    if limit_cells is not None and limit_cells < len(valid_cells):
        rng = random.Random(20260503)
        valid_cells = rng.sample(valid_cells, limit_cells)
        # When limited, the degenerate-skip count reported in summary
        # reflects the (limited) sample.
        n_skipped_degenerate = 0

    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with grid_path.open("w", encoding="utf-8") as fh:
        for ord_w, loop_w, cov_w, unexp_w, thr in valid_cells:
            config = _make_config(ord_w, loop_w, cov_w, unexp_w, thr)
            rho, auc = _score_cell(pool, config)
            row = _row_to_dict(ord_w, loop_w, cov_w, unexp_w, thr, rho, auc)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
    wall_seconds = time.perf_counter() - t0

    # Score the library default as its own cell. The default weights
    # (0.4 / 0.15) are NOT on the {0, 0.1, 0.25, 0.5, 1.0} grid, so we
    # always score it explicitly rather than searching the rows. We do
    # NOT add this row to grid.jsonl (it's not a grid cell), but we
    # surface it in summary.json under "default_config".
    default_config = _make_config(
        DEFAULT_ORDERING_WEIGHT,
        DEFAULT_LOOP_WEIGHT,
        DEFAULT_COVERAGE_WEIGHT,
        DEFAULT_UNEXPECTED_WEIGHT,
        DEFAULT_LOOP_THRESHOLD,
    )
    default_rho, default_auc = _score_cell(pool, default_config)
    default_row = {
        "ordering_weight": DEFAULT_ORDERING_WEIGHT,
        "loop_weight": DEFAULT_LOOP_WEIGHT,
        "coverage_weight": DEFAULT_COVERAGE_WEIGHT,
        "unexpected_weight": DEFAULT_UNEXPECTED_WEIGHT,
        "loop_threshold": DEFAULT_LOOP_THRESHOLD,
        "spearman_rho": default_rho,
        "auroc": default_auc,
    }

    sorted_by_rho = sorted(
        (r for r in rows if not np.isnan(r["spearman_rho"])),
        key=lambda r: r["spearman_rho"],
        reverse=True,
    )
    sorted_by_auc = sorted(
        (r for r in rows if not np.isnan(r["auroc"])),
        key=lambda r: r["auroc"],
        reverse=True,
    )
    top_5_by_spearman = sorted_by_rho[:5]
    top_5_by_auroc = sorted_by_auc[:5]

    if sorted_by_rho:
        best_rho = sorted_by_rho[0]["spearman_rho"]
        best_auc_for_rho_winner = sorted_by_rho[0]["auroc"]
    else:
        best_rho = float("nan")
        best_auc_for_rho_winner = float("nan")
    if sorted_by_auc:
        best_auc = sorted_by_auc[0]["auroc"]
    else:
        best_auc = float("nan")

    rho_gap = best_rho - default_rho
    rho_rel = (
        rho_gap / default_rho
        if default_rho not in (0.0,) and not np.isnan(default_rho)
        else float("nan")
    )
    auc_gap = best_auc - default_auc

    pareto = _is_pareto_optimal(rows, default_row)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(rows),
        "n_skipped_degenerate": n_skipped_degenerate,
        "limit_cells": limit_cells,
        "input_dir": str(Path(input_dir)),
        "n_trajectories": len(pool),
        "default_config": {
            "ordering_weight": DEFAULT_ORDERING_WEIGHT,
            "loop_weight": DEFAULT_LOOP_WEIGHT,
            "coverage_weight": DEFAULT_COVERAGE_WEIGHT,
            "unexpected_weight": DEFAULT_UNEXPECTED_WEIGHT,
            "loop_threshold": DEFAULT_LOOP_THRESHOLD,
            "spearman_rho": float(default_rho),
            "auroc": float(default_auc),
        },
        "top_5_by_spearman": top_5_by_spearman,
        "top_5_by_auroc": top_5_by_auroc,
        "best_vs_default_gap": {
            "spearman_rho": float(rho_gap),
            "spearman_rho_relative": float(rho_rel),
            "auroc": float(auc_gap),
            "best_spearman_rho": float(best_rho),
            "best_auroc": float(best_auc),
        },
        "default_is_pareto_optimal": bool(pareto),
        "weight_correlations": _weight_correlations(rows),
        "wall_seconds": float(wall_seconds),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    heatmap = _build_heatmap(rows)
    heatmap_path.write_text(
        json.dumps(heatmap, indent=2, sort_keys=True), encoding="utf-8"
    )

    return {
        "n_cells": len(rows),
        "n_skipped_degenerate": n_skipped_degenerate,
        "default_spearman_rho": float(default_rho),
        "default_auroc": float(default_auc),
        "best_spearman_rho": float(best_rho),
        "best_auroc": float(best_auc),
        "default_is_pareto_optimal": bool(pareto),
        "wall_seconds": float(wall_seconds),
        "output_dir": str(output_root),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argparse namespace for the CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TrajectoryMetric weight ablation (Task C3-zero).",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "Path to the C2-zero results directory. Used to recover the "
            "seeds / noise schedule / domains for parity."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write grid.jsonl, summary.json, heatmap.json.",
    )
    parser.add_argument(
        "--limit-cells",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of grid cells (random sample "
            "without replacement); useful for smoke testing."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Console entry point.

    Args:
        argv: Optional explicit argv for testing. ``None`` uses ``sys.argv``.

    Returns:
        Exit code; ``0`` on success.
    """
    args = _parse_args(argv)
    summary = run_experiment(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit_cells=args.limit_cells,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Correlation analysis for CheckLLM snapshots.

Two flavors are provided:

* :func:`correlate_metrics` — Pearson and Spearman correlation between every
  pair of metrics observed in a snapshot. Useful for detecting metrics that
  move together and metrics that are redundant.
* :func:`correlate_to_pass` — per-metric correlation of the metric score
  against the final pass/fail label of the run. Useful for answering
  "which metric is the strongest predictor of pass?".

The snapshot schema already exposes per-test score and pass lists via
``Snapshot.get_scores`` and ``Snapshot.get_pass_results``. This module is a
thin, numerically safe wrapper over :mod:`scipy.stats` that degrades
gracefully when there is insufficient data (returns ``nan`` rather than
raising).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from scipy import stats as scipy_stats

from checkllm.regression.snapshot import Snapshot


@dataclass
class CorrelationResult:
    """Pairwise correlation between two series."""

    metric_a: str
    metric_b: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n: int


@dataclass
class CorrelationMatrix:
    """Symmetric correlation matrix for a collection of metrics."""

    metrics: list[str]
    pairs: list[CorrelationResult] = field(default_factory=list)
    per_pass: dict[str, CorrelationResult] = field(default_factory=dict)

    def get(self, metric_a: str, metric_b: str) -> CorrelationResult | None:
        """Return the pair entry for ``metric_a`` and ``metric_b`` if present."""
        for pair in self.pairs:
            names = {pair.metric_a, pair.metric_b}
            if metric_a in names and metric_b in names:
                return pair
        return None

    def top_pairs(self, n: int = 5) -> list[CorrelationResult]:
        """Return the ``n`` strongest (by |Pearson r|) metric pairs."""
        finite = [p for p in self.pairs if not math.isnan(p.pearson_r)]
        finite.sort(key=lambda p: abs(p.pearson_r), reverse=True)
        return finite[:n]

    def best_predictor(self) -> CorrelationResult | None:
        """Return the metric most strongly correlated with pass/fail."""
        finite = [r for r in self.per_pass.values() if not math.isnan(r.pearson_r)]
        if not finite:
            return None
        finite.sort(key=lambda r: abs(r.pearson_r), reverse=True)
        return finite[0]


def _pearson(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Wrap :func:`scipy.stats.pearsonr` with degenerate-input handling."""
    if len(a) < 2 or len(b) < 2:
        return math.nan, math.nan
    if _is_constant(a) or _is_constant(b):
        return math.nan, math.nan
    r, p = scipy_stats.pearsonr(a, b)
    return float(r), float(p)


def _spearman(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Wrap :func:`scipy.stats.spearmanr` with degenerate-input handling."""
    if len(a) < 2 or len(b) < 2:
        return math.nan, math.nan
    if _is_constant(a) or _is_constant(b):
        return math.nan, math.nan
    result = scipy_stats.spearmanr(a, b)
    # ``spearmanr`` returns either a named tuple or an object with
    # ``.correlation``/``.pvalue`` depending on scipy version.
    r = float(getattr(result, "correlation", result[0]))
    p = float(getattr(result, "pvalue", result[1]))
    return r, p


def _is_constant(values: Iterable[float]) -> bool:
    vals = list(values)
    if not vals:
        return True
    first = vals[0]
    return all(v == first for v in vals)


def _flatten_snapshot(
    snapshot: Snapshot,
) -> tuple[list[str], dict[str, list[float]], dict[str, list[bool]]]:
    """Collapse a snapshot into per-metric flat score and pass lists.

    The snapshot groups tests into runs. Tests that omit a metric are skipped
    for that metric only — other metrics on the same run are still included.
    For pass correlation, a run's "label" is True iff every metric on that
    run passed (conservative default matching pytest semantics).
    """
    metrics: set[str] = set()
    for runs in snapshot.tests.values():
        for run in runs:
            metrics.update(run.metrics.keys())

    metric_order = sorted(metrics)
    scores: dict[str, list[float]] = {m: [] for m in metric_order}
    passes: dict[str, list[bool]] = {m: [] for m in metric_order}

    for runs in snapshot.tests.values():
        for run in runs:
            run_passed = all(rec.passed for rec in run.metrics.values())
            for metric_name in metric_order:
                record = run.metrics.get(metric_name)
                if record is None:
                    continue
                scores[metric_name].append(record.score)
                passes[metric_name].append(run_passed)

    return metric_order, scores, passes


def correlate_metrics(snapshot: Snapshot) -> list[CorrelationResult]:
    """Pairwise Pearson/Spearman correlation across every metric in a run.

    Only test cases where both metrics have a recorded score contribute to
    the correlation (intersect per pair). Pairs with fewer than two joint
    observations or where one series is constant return ``nan``.
    """
    metric_order, _, _ = _flatten_snapshot(snapshot)

    # Build per-test joint series keyed by test_name -> run_index -> metric -> score.
    joint: dict[str, dict[str, list[float]]] = {m: {} for m in metric_order}
    # For pairwise we need aligned per-run pairs. Rebuild an aligned view:
    # a list of run dicts, preserving order, each mapping metric -> score.
    aligned_runs: list[dict[str, float]] = []
    for runs in snapshot.tests.values():
        for run in runs:
            aligned_runs.append({k: v.score for k, v in run.metrics.items()})
    # ``joint`` unused sentinel kept for readability; drop it.
    del joint

    results: list[CorrelationResult] = []
    for i, metric_a in enumerate(metric_order):
        for metric_b in metric_order[i + 1 :]:
            a_scores: list[float] = []
            b_scores: list[float] = []
            for run in aligned_runs:
                if metric_a in run and metric_b in run:
                    a_scores.append(run[metric_a])
                    b_scores.append(run[metric_b])
            pr, pp = _pearson(a_scores, b_scores)
            sr, sp = _spearman(a_scores, b_scores)
            results.append(
                CorrelationResult(
                    metric_a=metric_a,
                    metric_b=metric_b,
                    pearson_r=pr,
                    pearson_p=pp,
                    spearman_r=sr,
                    spearman_p=sp,
                    n=len(a_scores),
                )
            )
    return results


def correlate_to_pass(snapshot: Snapshot) -> dict[str, CorrelationResult]:
    """Correlate each metric's score against the run-level pass/fail label.

    Returns a mapping of metric name -> :class:`CorrelationResult` where
    ``metric_b`` is the sentinel ``"__pass__"``.
    """
    metric_order, scores, passes = _flatten_snapshot(snapshot)
    out: dict[str, CorrelationResult] = {}
    for metric in metric_order:
        metric_scores = scores[metric]
        pass_bits = [1.0 if p else 0.0 for p in passes[metric]]
        pr, pp = _pearson(metric_scores, pass_bits)
        sr, sp = _spearman(metric_scores, pass_bits)
        out[metric] = CorrelationResult(
            metric_a=metric,
            metric_b="__pass__",
            pearson_r=pr,
            pearson_p=pp,
            spearman_r=sr,
            spearman_p=sp,
            n=len(metric_scores),
        )
    return out


def build_correlation_matrix(snapshot: Snapshot) -> CorrelationMatrix:
    """Compute a full correlation matrix for a snapshot."""
    metric_order, _, _ = _flatten_snapshot(snapshot)
    return CorrelationMatrix(
        metrics=metric_order,
        pairs=correlate_metrics(snapshot),
        per_pass=correlate_to_pass(snapshot),
    )


def summarize_correlation_matrix(matrix: CorrelationMatrix) -> Mapping[str, object]:
    """Return a JSON-friendly summary of a correlation matrix."""
    best = matrix.best_predictor()
    return {
        "metrics": list(matrix.metrics),
        "n_pairs": len(matrix.pairs),
        "top_pairs": [
            {
                "metric_a": p.metric_a,
                "metric_b": p.metric_b,
                "pearson_r": p.pearson_r,
                "pearson_p": p.pearson_p,
                "spearman_r": p.spearman_r,
                "spearman_p": p.spearman_p,
                "n": p.n,
            }
            for p in matrix.top_pairs(5)
        ],
        "per_pass": {
            name: {
                "pearson_r": r.pearson_r,
                "pearson_p": r.pearson_p,
                "spearman_r": r.spearman_r,
                "spearman_p": r.spearman_p,
                "n": r.n,
            }
            for name, r in matrix.per_pass.items()
        },
        "best_predictor": best.metric_a if best else None,
    }

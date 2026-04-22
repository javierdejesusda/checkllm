"""Scoring utilities: AUC, best-F1, and Spearman correlation."""

from __future__ import annotations

from statistics import mean
from typing import Mapping

from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from bench.schema import BenchmarkScore


def _aligned(
    scores: list[BenchmarkScore], labels: Mapping[str, float]
) -> tuple[list[float], list[float]]:
    """Align scores with ground-truth labels by sample_id.

    Args:
        scores: Evaluated benchmark scores.
        labels: Ground-truth labels keyed by sample_id.

    Returns:
        A tuple (y_score, y_true) with only the samples present in both.
    """
    y_score: list[float] = []
    y_true: list[float] = []
    for s in scores:
        if s.sample_id in labels:
            y_score.append(s.score)
            y_true.append(labels[s.sample_id])
    return y_score, y_true


def roc_auc(scores: list[BenchmarkScore], labels: Mapping[str, float]) -> float:
    """Compute ROC-AUC between predicted scores and binary ground-truth.

    Args:
        scores: Evaluated benchmark scores.
        labels: Binary ground-truth labels (0.0 or 1.0) keyed by sample_id.

    Returns:
        ROC-AUC in [0, 1], or NaN when only one class is present.
    """
    y_score, y_true = _aligned(scores, labels)
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def best_f1(scores: list[BenchmarkScore], labels: Mapping[str, float]) -> tuple[float, float]:
    """Find the threshold that maximises F1 over the score range.

    Args:
        scores: Evaluated benchmark scores.
        labels: Binary ground-truth labels (0.0 or 1.0) keyed by sample_id.

    Returns:
        A (best_f1, best_threshold) tuple.
    """
    y_score, y_true = _aligned(scores, labels)
    if not y_score:
        return 0.0, 0.5
    best: tuple[float, float] = (-1.0, 0.5)
    for threshold in sorted(set(y_score)):
        y_pred = [1.0 if s >= threshold else 0.0 for s in y_score]
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best[0]:
            best = (f1, threshold)
    return best


def spearman(scores: list[BenchmarkScore], labels: Mapping[str, float]) -> float:
    """Compute Spearman rank correlation between scores and numeric labels.

    Args:
        scores: Evaluated benchmark scores.
        labels: Numeric ground-truth labels keyed by sample_id.

    Returns:
        Spearman correlation in [-1, 1], or NaN when fewer than two samples.
    """
    y_score, y_true = _aligned(scores, labels)
    if len(y_score) < 2:
        return float("nan")
    corr, _ = spearmanr(y_score, y_true)
    return float(corr)


def summarize_scores(scores: list[BenchmarkScore], labels: Mapping[str, float]) -> dict[str, float]:
    """Compute a summary of all scoring metrics for a set of benchmark results.

    Args:
        scores: Evaluated benchmark scores.
        labels: Ground-truth labels keyed by sample_id.

    Returns:
        A dict with keys: auc, best_f1, best_threshold, spearman, n,
        mean_latency_ms, total_cost_usd.
    """
    f1, threshold = best_f1(scores, labels)
    return {
        "auc": roc_auc(scores, labels),
        "best_f1": f1,
        "best_threshold": threshold,
        "spearman": spearman(scores, labels),
        "n": float(len(scores)),
        "mean_latency_ms": float(mean(s.latency_ms for s in scores)) if scores else 0.0,
        "total_cost_usd": float(sum(s.cost_usd for s in scores)),
    }

"""Tests for checkllm.analysis.correlation."""

from __future__ import annotations

import math
import random

from checkllm.analysis.correlation import (
    build_correlation_matrix,
    correlate_metrics,
    correlate_to_pass,
    summarize_correlation_matrix,
)
from checkllm.regression.snapshot import MetricRecord, Snapshot, TestRunRecord


def _snapshot(metric_scores: dict[str, list[float]], threshold: float = 0.5) -> Snapshot:
    max_len = max(len(v) for v in metric_scores.values())
    runs = []
    for i in range(max_len):
        metrics = {}
        for name, scores in metric_scores.items():
            if i < len(scores):
                score = scores[i]
                metrics[name] = MetricRecord(score=score, passed=score >= threshold)
        runs.append(TestRunRecord(metrics=metrics))
    return Snapshot(tests={"test_a": runs})


class TestCorrelateMetrics:
    def test_linearly_related_metrics_have_pearson_close_to_one(self):
        rng = random.Random(1)
        xs = [rng.random() for _ in range(100)]
        ys = [2 * x + 0.01 for x in xs]
        snap = _snapshot({"a": xs, "b": ys})
        results = correlate_metrics(snap)
        assert len(results) == 1
        pair = results[0]
        assert pair.n == 100
        assert pair.pearson_r > 0.99
        assert pair.spearman_r > 0.99
        assert pair.pearson_p < 1e-50

    def test_uncorrelated_metrics_have_small_pearson(self):
        rng = random.Random(2)
        xs = [rng.random() for _ in range(200)]
        ys = [rng.random() for _ in range(200)]
        snap = _snapshot({"a": xs, "b": ys})
        results = correlate_metrics(snap)
        assert abs(results[0].pearson_r) < 0.2

    def test_constant_metric_returns_nan(self):
        snap = _snapshot({"a": [0.5] * 10, "b": [i / 10 for i in range(10)]})
        results = correlate_metrics(snap)
        assert math.isnan(results[0].pearson_r)

    def test_anticorrelated_metrics(self):
        rng = random.Random(3)
        xs = [rng.random() for _ in range(80)]
        ys = [1 - x for x in xs]
        snap = _snapshot({"a": xs, "b": ys})
        results = correlate_metrics(snap)
        assert results[0].pearson_r < -0.99


class TestCorrelateToPass:
    def test_metric_that_drives_pass_has_positive_correlation(self):
        # Metric 'strong' is high on passing runs and low on failing runs.
        strong = [0.9] * 10 + [0.1] * 10
        # Metric 'noise' is random w.r.t. pass label.
        rng = random.Random(9)
        noise = [rng.random() for _ in range(20)]
        # Pass label is derived from 'strong' (>= 0.5). Since snapshot
        # pass label = all metrics on run pass, we must make 'noise' not
        # cause failures on the strong-high runs: set noise in [0.5, 1.0].
        # Simplest: use threshold=0.0 so noise always "passes" per-metric.
        snap = _snapshot({"strong": strong, "noise": noise}, threshold=0.0)
        # Now every run passes -> pass label is constant -> correlation nan.
        # Adjust: override per-record passed field directly to reflect
        # a specific pass/fail pattern tied only to the 'strong' metric.
        runs = snap.tests["test_a"]
        for i, run in enumerate(runs):
            label = strong[i] >= 0.5
            for rec in run.metrics.values():
                rec.passed = label

        per_pass = correlate_to_pass(snap)
        assert "strong" in per_pass
        assert per_pass["strong"].pearson_r > 0.9
        assert per_pass["strong"].pearson_p < 1e-6
        # Noise should have weak correlation to the pass label.
        assert abs(per_pass["noise"].pearson_r) < 0.5


class TestCorrelationMatrix:
    def test_build_and_summarize(self):
        rng = random.Random(4)
        xs = [rng.random() for _ in range(50)]
        ys = [x + rng.gauss(0, 0.01) for x in xs]
        zs = [rng.random() for _ in range(50)]
        snap = _snapshot({"x": xs, "y": ys, "z": zs})

        matrix = build_correlation_matrix(snap)
        assert set(matrix.metrics) == {"x", "y", "z"}
        assert len(matrix.pairs) == 3
        top = matrix.top_pairs(1)[0]
        assert {top.metric_a, top.metric_b} == {"x", "y"}

        summary = summarize_correlation_matrix(matrix)
        assert summary["n_pairs"] == 3
        assert "per_pass" in summary

    def test_get_pair_is_order_independent(self):
        rng = random.Random(5)
        xs = [rng.random() for _ in range(30)]
        ys = [rng.random() for _ in range(30)]
        snap = _snapshot({"a": xs, "b": ys})
        matrix = build_correlation_matrix(snap)
        ab = matrix.get("a", "b")
        ba = matrix.get("b", "a")
        assert ab is not None and ba is not None
        assert ab is ba

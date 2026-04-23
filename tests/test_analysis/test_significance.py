"""Tests for checkllm.analysis.significance.

Uses deterministically seeded random draws from known distributions so
p-values and effect sizes land in predictable ranges.
"""

from __future__ import annotations

import random

import pytest

from checkllm.analysis.significance import (
    analyze_runs,
    bootstrap_ci,
    cohens_d,
    mann_whitney_u,
    significance_of,
    welchs_t_test,
)
from checkllm.regression.snapshot import MetricRecord, Snapshot, TestRunRecord


def _draw(mu: float, sigma: float, n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


class TestWelchsT:
    def test_same_distribution_not_significant(self):
        a = _draw(0.8, 0.05, 100, seed=1)
        b = _draw(0.8, 0.05, 100, seed=2)
        _, p = welchs_t_test(a, b)
        assert p > 0.05

    def test_different_means_significant(self):
        a = _draw(0.9, 0.05, 100, seed=3)
        b = _draw(0.5, 0.05, 100, seed=4)
        _, p = welchs_t_test(a, b)
        assert p < 1e-20

    def test_degenerate_sample_returns_safe_defaults(self):
        t, p = welchs_t_test([0.5], [0.5, 0.6])
        assert t == 0.0
        assert p == 1.0


class TestCohensD:
    def test_identical_samples_zero(self):
        a = [0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5]
        assert cohens_d(a, b) == 0.0

    def test_large_effect(self):
        # One SD apart means |d| ~ 1.0 (large effect in Cohen's taxonomy).
        a = _draw(1.0, 0.1, 200, seed=5)
        b = _draw(0.9, 0.1, 200, seed=6)
        d = cohens_d(a, b)
        assert 0.6 < d < 1.4

    def test_sign_reflects_direction(self):
        a = _draw(0.3, 0.05, 50, seed=7)
        b = _draw(0.8, 0.05, 50, seed=8)
        assert cohens_d(a, b) < -1.0


class TestMannWhitney:
    def test_large_shift_rejects(self):
        a = _draw(0.9, 0.05, 100, seed=9)
        b = _draw(0.5, 0.05, 100, seed=10)
        _, p = mann_whitney_u(a, b)
        assert p < 1e-10

    def test_identical_distribution_not_significant(self):
        a = _draw(0.8, 0.05, 100, seed=11)
        b = _draw(0.8, 0.05, 100, seed=12)
        _, p = mann_whitney_u(a, b)
        assert p > 0.05


class TestBootstrapCI:
    def test_ci_brackets_true_difference(self):
        a = _draw(0.9, 0.05, 200, seed=13)
        b = _draw(0.7, 0.05, 200, seed=14)
        low, high = bootstrap_ci(a, b, n_resamples=1000, seed=123)
        # True delta is ~0.2; 95% CI should cover it and exclude zero.
        assert low > 0.1
        assert high < 0.3

    def test_ci_contains_zero_for_equal_populations(self):
        a = _draw(0.5, 0.1, 200, seed=15)
        b = _draw(0.5, 0.1, 200, seed=16)
        low, high = bootstrap_ci(a, b, n_resamples=1000, seed=321)
        assert low < 0.0 < high

    def test_degenerate_returns_point(self):
        low, high = bootstrap_ci([], [], n_resamples=10, seed=0)
        assert low == high == 0.0


class TestSignificanceOf:
    def test_significant_result_has_bounded_ci(self):
        a = _draw(0.9, 0.05, 100, seed=17)
        b = _draw(0.5, 0.05, 100, seed=18)
        result = significance_of("score", a, b, seed=99, n_bootstrap=500)
        assert result.significant is True
        assert result.p_value < 1e-20
        # Cohen's d for ~8 SD apart with sd=0.05 -> very large; positive because a > b.
        assert result.effect_size > 4.0
        # CI is strictly positive (delta > 0).
        assert result.ci_low > 0
        assert result.ci_high > result.ci_low
        assert result.method == "welch"
        assert result.mann_whitney_p < 1e-10

    def test_bootstrap_method_uses_ci(self):
        a = _draw(0.8, 0.1, 40, seed=19)
        b = _draw(0.8, 0.1, 40, seed=20)
        result = significance_of("noise", a, b, method="bootstrap", seed=7, n_bootstrap=500)
        # Bootstrap CI should include 0 for effectively identical populations.
        assert result.significant is False
        assert result.ci_low <= 0.0 <= result.ci_high

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            significance_of("m", [0.1], [0.2], method="bogus")  # type: ignore[arg-type]


def _build_snapshot(scores_by_metric: dict[str, list[float]]) -> Snapshot:
    runs = []
    max_len = max(len(v) for v in scores_by_metric.values())
    for i in range(max_len):
        metrics = {}
        for metric, scores in scores_by_metric.items():
            if i < len(scores):
                metrics[metric] = MetricRecord(score=scores[i], passed=scores[i] >= 0.5)
        runs.append(TestRunRecord(metrics=metrics))
    return Snapshot(tests={"test_case": runs})


class TestAnalyzeRuns:
    def test_surfaces_biggest_movers_first(self):
        snap_a = _build_snapshot(
            {
                "accuracy": _draw(0.9, 0.05, 50, seed=21),
                "faithfulness": _draw(0.7, 0.05, 50, seed=22),
            }
        )
        snap_b = _build_snapshot(
            {
                "accuracy": _draw(0.5, 0.05, 50, seed=23),  # large negative shift
                "faithfulness": _draw(0.7, 0.05, 50, seed=24),  # no shift
            }
        )
        results = analyze_runs(snap_a, snap_b, seed=0, n_bootstrap=200)
        # Biggest effect first after sorting by |effect_size|.
        assert results[0].metric == "accuracy"
        assert results[0].significant is True
        # delta = mean_a - mean_b; A is the baseline at 0.9, B is the
        # candidate at 0.5 -> positive delta, large effect size.
        assert results[0].delta > 0.2
        assert results[0].effect_size > 1.0
        assert results[1].metric == "faithfulness"
        assert results[1].significant is False

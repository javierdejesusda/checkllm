import pytest

from checkllm.regression.stats import (
    ComparisonResult,
    compare_scores,
    confidence_interval,
    pass_rate,
)


class TestConfidenceInterval:
    def test_returns_mean_and_bounds(self):
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        ci = confidence_interval(scores, confidence=0.95)
        assert ci.mean == pytest.approx(0.85, abs=0.01)
        assert ci.lower < ci.mean
        assert ci.upper > ci.mean

    def test_single_value_has_zero_width(self):
        ci = confidence_interval([0.9], confidence=0.95)
        assert ci.mean == 0.9
        assert ci.lower == ci.upper == 0.9

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            confidence_interval([], confidence=0.95)


class TestPassRate:
    def test_all_pass(self):
        results = [True, True, True, True, True]
        rate = pass_rate(results)
        assert rate == 1.0

    def test_some_fail(self):
        results = [True, True, False, True, False]
        rate = pass_rate(results)
        assert rate == pytest.approx(0.6)

    def test_all_fail(self):
        results = [False, False, False]
        rate = pass_rate(results)
        assert rate == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pass_rate([])


class TestCompareScores:
    def test_detects_significant_regression(self):
        baseline = [0.9, 0.92, 0.88, 0.91, 0.89, 0.90, 0.91, 0.88, 0.92, 0.90]
        current = [0.5, 0.55, 0.48, 0.52, 0.50, 0.53, 0.49, 0.51, 0.54, 0.50]
        result = compare_scores(baseline, current, p_threshold=0.05)
        assert result.is_regression is True
        assert result.p_value < 0.05
        assert result.delta < 0

    def test_no_regression_when_similar(self):
        baseline = [0.9, 0.88, 0.91, 0.89, 0.90]
        current = [0.89, 0.91, 0.88, 0.90, 0.92]
        result = compare_scores(baseline, current, p_threshold=0.05)
        assert result.is_regression is False

    def test_no_regression_when_improved(self):
        baseline = [0.5, 0.52, 0.48, 0.51, 0.50]
        current = [0.9, 0.92, 0.88, 0.91, 0.89]
        result = compare_scores(baseline, current, p_threshold=0.05)
        assert result.is_regression is False

    def test_result_contains_stats(self):
        baseline = [0.9, 0.88, 0.91]
        current = [0.5, 0.52, 0.48]
        result = compare_scores(baseline, current, p_threshold=0.05)
        assert isinstance(result, ComparisonResult)
        assert result.baseline_mean == pytest.approx(0.8967, abs=0.01)
        assert result.current_mean == pytest.approx(0.5, abs=0.01)
        assert result.delta == pytest.approx(result.current_mean - result.baseline_mean, abs=0.01)
        assert 0 <= result.p_value <= 1.0

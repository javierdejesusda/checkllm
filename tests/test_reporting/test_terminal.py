from io import StringIO

from checkllm.models import CheckResult
from checkllm.regression.compare import RegressionItem
from checkllm.regression.stats import ComparisonResult
from checkllm.reporting.terminal import render_results, render_regression_report


class TestRenderResults:
    def test_renders_passing_results(self):
        results = [
            CheckResult(
                passed=True, score=0.95, reasoning="All good",
                cost=0.002, latency_ms=450, metric_name="hallucination",
            ),
        ]
        output = render_results(results, to_string=True)
        assert "PASS" in output
        assert "hallucination" in output
        assert "0.95" in output

    def test_renders_failing_results(self):
        results = [
            CheckResult(
                passed=False, score=0.3, reasoning="Hallucinated content",
                cost=0.002, latency_ms=500, metric_name="hallucination",
            ),
        ]
        output = render_results(results, to_string=True)
        assert "FAIL" in output
        assert "hallucination" in output

    def test_renders_cost_summary(self):
        results = [
            CheckResult(
                passed=True, score=0.9, reasoning="ok",
                cost=0.003, latency_ms=100, metric_name="hallucination",
            ),
            CheckResult(
                passed=True, score=0.8, reasoning="ok",
                cost=0.002, latency_ms=200, metric_name="relevance",
            ),
        ]
        output = render_results(results, to_string=True)
        assert "0.005" in output or "$0.01" in output or "cost" in output.lower()


class TestRenderRegressionReport:
    def test_renders_regression(self):
        items = [
            RegressionItem(
                test_name="test_foo",
                metric_name="hallucination",
                comparison=ComparisonResult(
                    is_regression=True,
                    baseline_mean=0.92,
                    current_mean=0.71,
                    delta=-0.21,
                    p_value=0.003,
                    baseline_std=0.02,
                    current_std=0.05,
                ),
            )
        ]
        output = render_regression_report(items, to_string=True)
        assert "REGRESSION" in output
        assert "test_foo" in output
        assert "hallucination" in output

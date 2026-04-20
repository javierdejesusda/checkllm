"""Additional tests for checkllm.models covering format_failure branches."""
from __future__ import annotations

import pytest

from checkllm.models import CheckFailedError, CheckResult, JudgeResponse


class TestCheckResultFormatFailure:
    def test_format_failure_with_threshold(self):
        result = CheckResult(
            passed=False,
            score=0.4,
            reasoning="Not grounded",
            cost=0.002,
            latency_ms=200,
            metric_name="hallucination",
            threshold=0.8,
        )
        text = result.format_failure()
        assert "hallucination" in text
        assert "0.40" in text
        assert "threshold" in text
        assert "0.80" in text

    def test_format_failure_without_threshold(self):
        result = CheckResult(
            passed=False,
            score=0.3,
            reasoning="Not relevant",
            cost=0.0,
            latency_ms=100,
            metric_name="relevance",
            threshold=None,
        )
        text = result.format_failure()
        assert "relevance" in text
        assert "threshold" not in text

    def test_format_failure_with_input_preview_short(self):
        result = CheckResult(
            passed=False,
            score=0.2,
            reasoning="Toxic",
            cost=0.001,
            latency_ms=50,
            metric_name="toxicity",
            input_preview="Short input",
        )
        text = result.format_failure()
        assert "Short input" in text
        assert "..." not in text

    def test_format_failure_with_input_preview_truncated(self):
        long_input = "A" * 200
        result = CheckResult(
            passed=False,
            score=0.1,
            reasoning="Bad",
            cost=0.0,
            latency_ms=10,
            metric_name="test",
            input_preview=long_input,
        )
        text = result.format_failure()
        assert "..." in text
        # Should contain first 120 chars
        assert "A" * 120 in text

    def test_format_failure_with_cost(self):
        result = CheckResult(
            passed=False,
            score=0.5,
            reasoning="Mediocre",
            cost=0.0050,
            latency_ms=300,
            metric_name="coherence",
        )
        text = result.format_failure()
        assert "Cost:" in text
        assert "Latency:" in text

    def test_format_failure_no_cost(self):
        result = CheckResult(
            passed=False,
            score=0.5,
            reasoning="Mediocre",
            cost=0.0,
            latency_ms=300,
            metric_name="coherence",
        )
        text = result.format_failure()
        assert "Cost:" not in text


class TestJudgeResponseExtra:
    def test_default_cost_is_zero(self):
        resp = JudgeResponse(score=0.7, reasoning="ok")
        assert resp.cost == 0.0

    def test_with_cost(self):
        resp = JudgeResponse(score=0.9, reasoning="good", cost=0.005)
        assert resp.cost == 0.005

    def test_raw_output_optional(self):
        resp = JudgeResponse(score=0.8, reasoning="fine")
        assert resp.raw_output is None


class TestCheckFailedErrorExtra:
    def test_multiple_failed(self):
        results = [
            CheckResult(
                passed=False, score=0.1, reasoning="bad1", cost=0.0,
                latency_ms=0, metric_name="metric1",
            ),
            CheckResult(
                passed=False, score=0.2, reasoning="bad2", cost=0.0,
                latency_ms=0, metric_name="metric2",
            ),
        ]
        error = CheckFailedError(results)
        assert len(error.failed_results) == 2
        assert "2 check(s) failed" in str(error)
        assert "metric1" in str(error)
        assert "metric2" in str(error)

    def test_format_failure_in_error_message(self):
        results = [
            CheckResult(
                passed=False,
                score=0.3,
                reasoning="Not grounded at all",
                cost=0.001,
                latency_ms=100,
                metric_name="hallucination",
                threshold=0.8,
            )
        ]
        error = CheckFailedError(results)
        error_str = str(error)
        assert "FAILED" in error_str
        assert "hallucination" in error_str
        assert "Not grounded at all" in error_str

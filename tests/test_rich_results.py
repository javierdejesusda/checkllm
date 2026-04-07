"""Verify enhanced CheckResult carries diagnostic context."""
from checkllm.models import CheckResult, CheckFailedError


class TestEnhancedCheckResult:
    def test_check_result_with_context(self):
        r = CheckResult(
            passed=False, score=0.65, reasoning="Output contains unsupported claims",
            cost=0.002, latency_ms=450, metric_name="hallucination",
            threshold=0.8, input_preview="The capital of France is Berlin...",
        )
        assert r.threshold == 0.8
        assert r.input_preview == "The capital of France is Berlin..."

    def test_check_result_backwards_compatible(self):
        r = CheckResult(
            passed=True, score=1.0, reasoning="OK", cost=0.0,
            latency_ms=0, metric_name="contains",
        )
        assert r.threshold is None
        assert r.input_preview is None

    def test_check_result_format_failure(self):
        r = CheckResult(
            passed=False, score=0.3, reasoning="Output is not relevant to the query",
            cost=0.001, latency_ms=200, metric_name="relevance",
            threshold=0.8, input_preview="What is Python?",
        )
        text = r.format_failure()
        assert "relevance" in text
        assert "0.30" in text
        assert "0.80" in text
        assert "not relevant" in text

    def test_format_failure_without_threshold(self):
        r = CheckResult(
            passed=False, score=0.0, reasoning="Not found",
            cost=0.0, latency_ms=0, metric_name="contains",
        )
        text = r.format_failure()
        assert "contains" in text
        assert "Score: 0.00" in text

    def test_check_failed_error_shows_details(self):
        results = [
            CheckResult(
                passed=False, score=0.3, reasoning="Not relevant",
                cost=0.001, latency_ms=200, metric_name="relevance",
                threshold=0.8, input_preview="What is Python?",
            ),
            CheckResult(
                passed=True, score=0.95, reasoning="OK",
                cost=0.001, latency_ms=150, metric_name="toxicity",
            ),
        ]
        err = CheckFailedError(results)
        text = str(err)
        assert "1 check(s) failed" in text
        assert "relevance" in text
        assert "0.3" in text or "0.30" in text

    def test_check_failed_error_backwards_compatible(self):
        results = [
            CheckResult(
                passed=False, score=0.0, reasoning="Not found",
                cost=0.0, latency_ms=0, metric_name="contains",
            ),
        ]
        err = CheckFailedError(results)
        assert "1 check(s) failed" in str(err)

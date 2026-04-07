"""Verify CheckFailedError produces rich output."""
from checkllm.models import CheckFailedError, CheckResult


class TestCheckFailedErrorRich:
    def test_str_shows_per_check_detail(self):
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
        assert "relevance" in text
        assert "0.3" in text or "0.30" in text

    def test_backwards_compatible_message(self):
        results = [
            CheckResult(
                passed=False, score=0.0, reasoning="Not found",
                cost=0.0, latency_ms=0, metric_name="contains",
            ),
        ]
        err = CheckFailedError(results)
        assert "1 check(s) failed" in str(err)

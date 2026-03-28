from checkllm.models import CheckResult, CheckFailedError, JudgeResponse


class TestCheckResult:
    def test_create_passing_result(self):
        result = CheckResult(
            passed=True,
            score=0.95,
            reasoning="Output is grounded in context",
            cost=0.002,
            latency_ms=450,
            metric_name="hallucination",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "hallucination"

    def test_create_failing_result(self):
        result = CheckResult(
            passed=False,
            score=0.3,
            reasoning="Output contains claims not in context",
            cost=0.002,
            latency_ms=500,
            metric_name="hallucination",
        )
        assert result.passed is False
        assert result.score == 0.3

    def test_score_must_be_between_0_and_1(self):
        import pytest

        with pytest.raises(ValueError):
            CheckResult(
                passed=True,
                score=1.5,
                reasoning="bad",
                cost=0.0,
                latency_ms=0,
                metric_name="test",
            )

    def test_cost_must_be_non_negative(self):
        import pytest

        with pytest.raises(ValueError):
            CheckResult(
                passed=True,
                score=0.5,
                reasoning="bad",
                cost=-0.01,
                latency_ms=0,
                metric_name="test",
            )


class TestJudgeResponse:
    def test_create_judge_response(self):
        resp = JudgeResponse(
            score=0.85,
            reasoning="The output is relevant to the query",
            raw_output="score: 0.85\nreasoning: The output is relevant",
        )
        assert resp.score == 0.85
        assert resp.reasoning == "The output is relevant to the query"
        assert resp.raw_output is not None


class TestCheckFailedError:
    def test_error_contains_results(self):
        results = [
            CheckResult(
                passed=False,
                score=0.3,
                reasoning="failed",
                cost=0.0,
                latency_ms=0,
                metric_name="hallucination",
            ),
            CheckResult(
                passed=True,
                score=0.9,
                reasoning="passed",
                cost=0.0,
                latency_ms=0,
                metric_name="relevance",
            ),
        ]
        error = CheckFailedError(results)
        assert len(error.results) == 2
        assert len(error.failed_results) == 1
        assert error.failed_results[0].metric_name == "hallucination"
        assert "1 check(s) failed" in str(error)

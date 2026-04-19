"""Tests for checkllm.errors and checkllm.models — error formatting and models."""

from __future__ import annotations

import pytest

from checkllm.errors import format_budget_error, format_missing_dependency_error
from checkllm.models import CheckFailedError, CheckResult, JudgeResponse


class TestFormatBudgetError:
    def test_contains_budget(self):
        msg = format_budget_error(budget=5.0, spent=5.01, completed=3, total=10)
        assert "$5.00" in msg

    def test_contains_spent(self):
        msg = format_budget_error(budget=5.0, spent=5.01, completed=3, total=10)
        assert "5.0100" in msg

    def test_contains_completed_total(self):
        msg = format_budget_error(budget=5.0, spent=5.01, completed=3, total=10)
        assert "3/10" in msg

    def test_contains_fix_guidance(self):
        msg = format_budget_error(budget=5.0, spent=5.01, completed=3, total=10)
        assert "fix" in msg.lower() or "budget" in msg.lower()

    def test_suggests_doubled_budget(self):
        msg = format_budget_error(budget=5.0, spent=5.01, completed=3, total=10)
        assert "10" in msg  # doubled budget suggestion

    def test_contains_cheaper_model_suggestion(self):
        msg = format_budget_error(budget=1.0, spent=1.01, completed=1, total=5)
        assert "mini" in msg.lower() or "cheaper" in msg.lower()


class TestFormatMissingDependencyError:
    def test_contains_package_name(self):
        msg = format_missing_dependency_error("anthropic", "AnthropicJudge")
        assert "anthropic" in msg

    def test_contains_class_name(self):
        msg = format_missing_dependency_error("anthropic", "AnthropicJudge")
        assert "AnthropicJudge" in msg

    def test_contains_pip_install(self):
        msg = format_missing_dependency_error("anthropic", "AnthropicJudge")
        assert "pip install" in msg

    def test_known_extra_anthropic(self):
        msg = format_missing_dependency_error("anthropic", "AnthropicJudge")
        assert "checkllm[anthropic]" in msg

    def test_known_extra_gemini(self):
        msg = format_missing_dependency_error("gemini", "GeminiJudge")
        assert "checkllm[gemini]" in msg

    def test_known_extra_litellm(self):
        msg = format_missing_dependency_error("litellm", "LiteLLMJudge")
        assert "checkllm[litellm]" in msg

    def test_known_extra_embeddings(self):
        msg = format_missing_dependency_error("embeddings", "EmbeddingClient")
        assert "checkllm[embeddings]" in msg

    def test_known_extra_sentence_transformers(self):
        msg = format_missing_dependency_error("sentence-transformers", "SentenceTransformer")
        assert "checkllm[embeddings]" in msg

    def test_known_extra_google_generativeai(self):
        msg = format_missing_dependency_error("google-generativeai", "GeminiClient")
        assert "checkllm[gemini]" in msg

    def test_unknown_extra_fallback(self):
        msg = format_missing_dependency_error("somepackage", "SomeClass")
        assert "checkllm[somepackage]" in msg

    def test_contains_all_extra(self):
        msg = format_missing_dependency_error("anthropic", "AnthropicJudge")
        assert "checkllm[all]" in msg


class TestCheckResult:
    def test_create_basic(self):
        result = CheckResult(
            passed=True,
            score=0.9,
            reasoning="Good output",
            cost=0.001,
            latency_ms=50,
            metric_name="relevance",
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.reasoning == "Good output"
        assert result.cost == 0.001
        assert result.latency_ms == 50
        assert result.metric_name == "relevance"

    def test_format_failure_basic(self):
        result = CheckResult(
            passed=False,
            score=0.3,
            reasoning="Missing key information",
            cost=0.0,
            latency_ms=0,
            metric_name="hallucination",
        )
        text = result.format_failure()
        assert "FAILED" in text
        assert "hallucination" in text
        assert "0.30" in text
        assert "Missing key information" in text

    def test_format_failure_with_threshold(self):
        result = CheckResult(
            passed=False,
            score=0.5,
            reasoning="Below threshold",
            cost=0.0,
            latency_ms=0,
            metric_name="relevance",
            threshold=0.8,
        )
        text = result.format_failure()
        assert "threshold: 0.80" in text
        assert "0.50" in text

    def test_format_failure_with_input_preview(self):
        result = CheckResult(
            passed=False,
            score=0.2,
            reasoning="Bad",
            cost=0.0,
            latency_ms=0,
            metric_name="test",
            input_preview="This is the input",
        )
        text = result.format_failure()
        assert "This is the input" in text

    def test_format_failure_with_long_input_preview(self):
        long_input = "x" * 200
        result = CheckResult(
            passed=False,
            score=0.2,
            reasoning="Bad",
            cost=0.0,
            latency_ms=0,
            metric_name="test",
            input_preview=long_input,
        )
        text = result.format_failure()
        assert "..." in text

    def test_format_failure_with_cost(self):
        result = CheckResult(
            passed=False,
            score=0.2,
            reasoning="Bad",
            cost=0.005,
            latency_ms=120,
            metric_name="test",
        )
        text = result.format_failure()
        assert "0.0050" in text
        assert "120ms" in text

    def test_format_failure_no_cost_shown_when_zero(self):
        result = CheckResult(
            passed=False,
            score=0.2,
            reasoning="Bad",
            cost=0.0,
            latency_ms=0,
            metric_name="test",
        )
        text = result.format_failure()
        assert "Cost:" not in text

    def test_score_validation(self):
        with pytest.raises(Exception):
            CheckResult(
                passed=True,
                score=1.5,
                reasoning="ok",
                cost=0.0,
                latency_ms=0,
                metric_name="test",
            )

    def test_threshold_none_by_default(self):
        result = CheckResult(
            passed=True,
            score=0.9,
            reasoning="ok",
            cost=0.0,
            latency_ms=0,
            metric_name="test",
        )
        assert result.threshold is None

    def test_input_preview_none_by_default(self):
        result = CheckResult(
            passed=True,
            score=0.9,
            reasoning="ok",
            cost=0.0,
            latency_ms=0,
            metric_name="test",
        )
        assert result.input_preview is None


class TestJudgeResponse:
    def test_create_basic(self):
        response = JudgeResponse(score=0.8, reasoning="Good response")
        assert response.score == 0.8
        assert response.reasoning == "Good response"
        assert response.raw_output is None
        assert response.cost == 0.0

    def test_create_with_all_fields(self):
        response = JudgeResponse(
            score=0.95,
            reasoning="Excellent",
            raw_output='{"score": 0.95, "reasoning": "Excellent"}',
            cost=0.002,
        )
        assert response.score == 0.95
        assert response.raw_output is not None
        assert response.cost == 0.002


class TestCheckFailedError:
    def _make_result(self, passed: bool, name: str, score: float = 0.5) -> CheckResult:
        return CheckResult(
            passed=passed,
            score=score,
            reasoning="test reasoning",
            cost=0.0,
            latency_ms=0,
            metric_name=name,
        )

    def test_message_contains_failed_count(self):
        results = [
            self._make_result(False, "relevance"),
            self._make_result(False, "hallucination"),
            self._make_result(True, "safety"),
        ]
        err = CheckFailedError(results)
        assert "2 check(s) failed" in str(err)

    def test_message_contains_metric_names(self):
        results = [
            self._make_result(False, "relevance"),
            self._make_result(False, "hallucination"),
        ]
        err = CheckFailedError(results)
        msg = str(err)
        assert "relevance" in msg
        assert "hallucination" in msg

    def test_failed_results_attribute(self):
        results = [
            self._make_result(False, "metric1"),
            self._make_result(True, "metric2"),
            self._make_result(False, "metric3"),
        ]
        err = CheckFailedError(results)
        assert len(err.failed_results) == 2
        names = {r.metric_name for r in err.failed_results}
        assert "metric1" in names
        assert "metric3" in names

    def test_results_attribute_contains_all(self):
        results = [
            self._make_result(False, "metric1"),
            self._make_result(True, "metric2"),
        ]
        err = CheckFailedError(results)
        assert len(err.results) == 2

    def test_is_exception(self):
        err = CheckFailedError([self._make_result(False, "test")])
        assert isinstance(err, Exception)

    def test_single_failure(self):
        err = CheckFailedError([self._make_result(False, "relevance")])
        assert "1 check(s) failed" in str(err)

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from checkllm.dual_judge import (
    AggregationMethod,
    DualJudge,
    DualJudgeMetric,
    DualJudgeResult,
)
from checkllm.models import CheckResult, JudgeResponse


def _mock_judge(score: float, reasoning: str = "ok", cost: float = 0.01) -> AsyncMock:
    """Create a mock judge that returns a fixed response."""
    judge = AsyncMock()
    judge.evaluate = AsyncMock(
        return_value=JudgeResponse(score=score, reasoning=reasoning, cost=cost)
    )
    return judge


class TestDualJudgeAverage:
    @pytest.mark.asyncio
    async def test_average_aggregation(self):
        dual = DualJudge(
            primary=_mock_judge(0.8, "good", 0.01),
            secondary=_mock_judge(0.6, "decent", 0.02),
            aggregation=AggregationMethod.AVERAGE,
        )

        result = await dual.evaluate("test prompt")

        assert isinstance(result, DualJudgeResult)
        assert abs(result.score - 0.7) < 1e-10
        assert result.primary_score == 0.8
        assert result.secondary_score == 0.6
        assert result.primary_reasoning == "good"
        assert result.secondary_reasoning == "decent"

    @pytest.mark.asyncio
    async def test_cost_aggregated(self):
        dual = DualJudge(
            primary=_mock_judge(0.5, cost=0.03),
            secondary=_mock_judge(0.5, cost=0.07),
        )

        result = await dual.evaluate("test")
        assert abs(result.cost - 0.10) < 1e-10


class TestDualJudgeMinMax:
    @pytest.mark.asyncio
    async def test_min_aggregation(self):
        dual = DualJudge(
            primary=_mock_judge(0.9),
            secondary=_mock_judge(0.4),
            aggregation=AggregationMethod.MIN,
        )

        result = await dual.evaluate("test")
        assert abs(result.score - 0.4) < 1e-10

    @pytest.mark.asyncio
    async def test_max_aggregation(self):
        dual = DualJudge(
            primary=_mock_judge(0.3),
            secondary=_mock_judge(0.7),
            aggregation=AggregationMethod.MAX,
        )

        result = await dual.evaluate("test")
        assert abs(result.score - 0.7) < 1e-10


class TestDualJudgeWeighted:
    @pytest.mark.asyncio
    async def test_weighted_aggregation(self):
        dual = DualJudge(
            primary=_mock_judge(1.0),
            secondary=_mock_judge(0.0),
            aggregation=AggregationMethod.WEIGHTED,
            primary_weight=0.6,
        )

        result = await dual.evaluate("test")
        assert abs(result.score - 0.6) < 1e-10

    @pytest.mark.asyncio
    async def test_weighted_custom_weight(self):
        dual = DualJudge(
            primary=_mock_judge(0.8),
            secondary=_mock_judge(0.2),
            aggregation=AggregationMethod.WEIGHTED,
            primary_weight=0.75,
        )

        result = await dual.evaluate("test")
        expected = 0.75 * 0.8 + 0.25 * 0.2
        assert abs(result.score - expected) < 1e-10


class TestRequireAgreement:
    @pytest.mark.asyncio
    async def test_agreeing_judges(self):
        dual = DualJudge(
            primary=_mock_judge(0.8),
            secondary=_mock_judge(0.7),
            aggregation=AggregationMethod.REQUIRE_AGREEMENT,
            agreement_threshold=0.2,
        )

        result = await dual.evaluate("test")
        assert result.agreement is True
        assert abs(result.score - 0.75) < 1e-10
        assert result.score_difference == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_disagreeing_judges(self):
        dual = DualJudge(
            primary=_mock_judge(0.9),
            secondary=_mock_judge(0.3),
            aggregation=AggregationMethod.REQUIRE_AGREEMENT,
            agreement_threshold=0.2,
        )

        result = await dual.evaluate("test")
        assert result.agreement is False
        assert abs(result.score - 0.3) < 1e-10
        assert "Warning" in result.reasoning
        assert result.score_difference == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_exact_threshold_boundary(self):
        dual = DualJudge(
            primary=_mock_judge(0.7),
            secondary=_mock_judge(0.5),
            aggregation=AggregationMethod.REQUIRE_AGREEMENT,
            agreement_threshold=0.2,
        )

        result = await dual.evaluate("test")
        assert result.agreement is True


class TestDualJudgeConcurrency:
    @pytest.mark.asyncio
    async def test_both_judges_called(self):
        primary = _mock_judge(0.8)
        secondary = _mock_judge(0.6)

        dual = DualJudge(primary=primary, secondary=secondary)
        await dual.evaluate("test prompt", system_prompt="system")

        primary.evaluate.assert_awaited_once_with("test prompt", "system")
        secondary.evaluate.assert_awaited_once_with("test prompt", "system")


class TestDualJudgeResult:
    def test_result_model(self):
        result = DualJudgeResult(
            score=0.7,
            reasoning="combined",
            cost=0.05,
            primary_score=0.8,
            secondary_score=0.6,
            primary_reasoning="good",
            secondary_reasoning="ok",
            agreement=True,
            score_difference=0.2,
        )
        assert result.primary_score == 0.8
        assert result.secondary_score == 0.6
        assert result.agreement is True
        assert result.score_difference == 0.2


class TestDualJudgeMetric:
    @pytest.mark.asyncio
    async def test_metric_wrapper(self):
        """Test DualJudgeMetric wraps a metric class and evaluates in parallel."""

        class FakeMetric:
            def __init__(self, judge, threshold=0.8):
                self.judge = judge
                self.threshold = threshold
                self.system_prompt = "test"

            async def evaluate(self, output: str, context: str) -> CheckResult:
                response = await self.judge.evaluate(output)
                return CheckResult(
                    passed=response.score >= self.threshold,
                    score=response.score,
                    reasoning=response.reasoning,
                    cost=response.cost,
                    latency_ms=10,
                    metric_name="fake",
                    threshold=self.threshold,
                )

        dual_metric = DualJudgeMetric(
            metric_class=FakeMetric,
            primary_judge=_mock_judge(0.9, "primary says good", 0.01),
            secondary_judge=_mock_judge(0.7, "secondary says ok", 0.02),
            aggregation=AggregationMethod.AVERAGE,
        )

        result = await dual_metric.evaluate(output="test output", context="test ctx")

        assert isinstance(result, CheckResult)
        assert abs(result.score - 0.8) < 1e-10
        assert result.metric_name == "fake"
        assert abs(result.cost - 0.03) < 1e-10
        assert "[Primary]" in result.reasoning
        assert "[Secondary]" in result.reasoning

    @pytest.mark.asyncio
    async def test_metric_wrapper_min_aggregation(self):
        class SimpleMetric:
            def __init__(self, judge, threshold=0.5):
                self.judge = judge
                self.threshold = threshold
                self.system_prompt = "eval"

            async def evaluate(self, output: str) -> CheckResult:
                response = await self.judge.evaluate(output)
                return CheckResult(
                    passed=response.score >= self.threshold,
                    score=response.score,
                    reasoning=response.reasoning,
                    cost=response.cost,
                    latency_ms=5,
                    metric_name="simple",
                    threshold=self.threshold,
                )

        dual_metric = DualJudgeMetric(
            metric_class=SimpleMetric,
            primary_judge=_mock_judge(0.8),
            secondary_judge=_mock_judge(0.4),
            aggregation=AggregationMethod.MIN,
        )

        result = await dual_metric.evaluate(output="test")
        assert abs(result.score - 0.4) < 1e-10
        assert result.passed is False


class TestAggregationMethodEnum:
    def test_all_values(self):
        assert AggregationMethod.AVERAGE.value == "average"
        assert AggregationMethod.MIN.value == "min"
        assert AggregationMethod.MAX.value == "max"
        assert AggregationMethod.REQUIRE_AGREEMENT.value == "require_agreement"
        assert AggregationMethod.WEIGHTED.value == "weighted"

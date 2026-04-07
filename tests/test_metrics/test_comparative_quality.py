from __future__ import annotations

import pytest

from checkllm.metrics.comparative_quality import ComparativeQualityMetric
from checkllm.testing import MockJudge


class TestComparativeQualityMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_a_is_better(self, judge):
        judge.set_default(score=0.85, reasoning="Response A is clearly better")
        metric = ComparativeQualityMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(
            output_a="Python is a high-level, interpreted language created in 1991 by Guido van Rossum.",
            output_b="Python is a language.",
            criteria="Evaluate for completeness, accuracy, and informativeness.",
        )
        assert result.passed is True
        assert result.score == 0.85
        assert result.metric_name == "comparative_quality"

    @pytest.mark.asyncio
    async def test_passes_when_equal(self, judge):
        judge.set_default(score=0.5, reasoning="Both responses are equally good")
        metric = ComparativeQualityMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(
            output_a="Python is great.",
            output_b="Python is wonderful.",
            criteria="Evaluate for informativeness.",
        )
        assert result.passed is True
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_fails_when_b_is_better(self, judge):
        judge.set_default(score=0.2, reasoning="Response B is significantly better")
        metric = ComparativeQualityMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(
            output_a="Python is a language.",
            output_b="Python is a high-level, interpreted language created in 1991 by Guido van Rossum.",
            criteria="Evaluate for completeness, accuracy, and informativeness.",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.7, reasoning="A is somewhat better")
        metric = ComparativeQualityMetric(judge=judge, threshold=0.6)
        result = await metric.evaluate(
            output_a="a", output_b="b", criteria="quality"
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_all_inputs(self, judge):
        judge.set_default(score=0.5, reasoning="ok")
        metric = ComparativeQualityMetric(judge=judge, threshold=0.5)
        await metric.evaluate(
            output_a="first response",
            output_b="second response",
            criteria="my evaluation criteria",
        )
        last_call = judge.calls[-1]
        assert "first response" in last_call["prompt"]
        assert "second response" in last_call["prompt"]
        assert "my evaluation criteria" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_default_threshold_is_half(self, judge):
        judge.set_default(score=0.5, reasoning="Equal")
        metric = ComparativeQualityMetric(judge=judge)
        assert metric.threshold == 0.5
        result = await metric.evaluate(
            output_a="a", output_b="b", criteria="quality"
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.5, reasoning="ok")
        metric = ComparativeQualityMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(
            output_a="a", output_b="b", criteria="quality"
        )
        assert result.latency_ms >= 0

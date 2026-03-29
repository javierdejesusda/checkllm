from __future__ import annotations

import pytest

from checkllm.metrics.faithfulness import FaithfulnessMetric
from checkllm.testing import MockJudge


class TestFaithfulnessMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_faithful(self, judge):
        judge.set_default(score=0.95, reasoning="Answer is fully faithful to context")
        metric = FaithfulnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python was created by Guido van Rossum.",
            context="Python is a programming language created by Guido van Rossum in 1991.",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "faithfulness"

    @pytest.mark.asyncio
    async def test_fails_when_unfaithful(self, judge):
        judge.set_default(score=0.2, reasoning="Answer introduces unsupported claims")
        metric = FaithfulnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python was created in 1985 and is the fastest language.",
            context="Python is a programming language created by Guido van Rossum in 1991.",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = FaithfulnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", context="test context")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", context="test context")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Partially faithful")
        metric = FaithfulnessMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(output="test", context="test context")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, judge):
        metric = FaithfulnessMetric(judge=judge, threshold=0.8)
        metric.system_prompt = "Custom faithfulness prompt"
        judge.set_default(score=0.9, reasoning="ok")
        await metric.evaluate(output="test", context="test context")
        last_call = judge.calls[-1]
        assert last_call["system_prompt"] == "Custom faithfulness prompt"

    @pytest.mark.asyncio
    async def test_score_range(self, judge):
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            judge.set_default(score=score, reasoning=f"Score {score}")
            metric = FaithfulnessMetric(judge=judge, threshold=0.8)
            result = await metric.evaluate(output="test", context="test context")
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_optional_query_parameter(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = FaithfulnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="test output",
            context="test context",
            query="test query",
        )
        assert result.passed is True
        last_call = judge.calls[-1]
        assert "test query" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = FaithfulnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", context="test context")
        assert result.latency_ms >= 0

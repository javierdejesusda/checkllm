from __future__ import annotations

import pytest

from checkllm.metrics.groundedness import GroundednessMetric
from checkllm.testing import MockJudge


class TestGroundednessMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_grounded(self, judge):
        judge.set_default(score=0.95, reasoning="All claims grounded in sources")
        metric = GroundednessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python was created by Guido van Rossum and released in 1991.",
            sources=[
                "Python is a programming language created by Guido van Rossum.",
                "Python was first released in 1991.",
            ],
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "groundedness"

    @pytest.mark.asyncio
    async def test_fails_when_ungrounded(self, judge):
        judge.set_default(score=0.2, reasoning="Most claims lack source support")
        metric = GroundednessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python is the fastest language and was created by Linus Torvalds.",
            sources=[
                "Python is a programming language created by Guido van Rossum.",
            ],
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = GroundednessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", sources=["source"])
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", sources=["source"])
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Partially grounded")
        metric = GroundednessMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(output="test", sources=["source"])
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, judge):
        metric = GroundednessMetric(judge=judge, threshold=0.8)
        metric.system_prompt = "Custom groundedness prompt"
        judge.set_default(score=0.9, reasoning="ok")
        await metric.evaluate(output="test", sources=["source"])
        last_call = judge.calls[-1]
        assert last_call["system_prompt"] == "Custom groundedness prompt"

    @pytest.mark.asyncio
    async def test_score_range(self, judge):
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            judge.set_default(score=score, reasoning=f"Score {score}")
            metric = GroundednessMetric(judge=judge, threshold=0.8)
            result = await metric.evaluate(output="test", sources=["source"])
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_multiple_sources(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = GroundednessMetric(judge=judge, threshold=0.8)
        await metric.evaluate(
            output="test output",
            sources=["source one", "source two", "source three"],
        )
        last_call = judge.calls[-1]
        assert "source one" in last_call["prompt"]
        assert "source two" in last_call["prompt"]
        assert "source three" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_single_source(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = GroundednessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", sources=["single source"])
        assert result.passed is True
        last_call = judge.calls[-1]
        assert "single source" in last_call["prompt"]

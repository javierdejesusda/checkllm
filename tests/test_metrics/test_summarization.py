from __future__ import annotations

import pytest

from checkllm.metrics.summarization import SummarizationMetric
from checkllm.testing import MockJudge


class TestSummarizationMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_good_summary(self, judge):
        judge.set_default(score=0.95, reasoning="Excellent summary, accurate and concise")
        metric = SummarizationMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python is a popular, readable programming language created in 1991.",
            source="Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability. It was created by Guido van Rossum and first released in 1991. Python is consistently ranked as one of the most popular programming languages.",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "summarization"

    @pytest.mark.asyncio
    async def test_fails_when_poor_summary(self, judge):
        judge.set_default(score=0.2, reasoning="Summary misses key information")
        metric = SummarizationMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Something about code.",
            source="Python is a high-level programming language created by Guido van Rossum in 1991.",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = SummarizationMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", source="test source")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", source="test source")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Adequate summary")
        metric = SummarizationMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(output="test", source="test source")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, judge):
        metric = SummarizationMetric(judge=judge, threshold=0.8)
        metric.system_prompt = "Custom summarization prompt"
        judge.set_default(score=0.9, reasoning="ok")
        await metric.evaluate(output="test", source="test source")
        last_call = judge.calls[-1]
        assert last_call["system_prompt"] == "Custom summarization prompt"

    @pytest.mark.asyncio
    async def test_score_range(self, judge):
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            judge.set_default(score=score, reasoning=f"Score {score}")
            metric = SummarizationMetric(judge=judge, threshold=0.8)
            result = await metric.evaluate(output="test", source="test source")
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_source(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = SummarizationMetric(judge=judge, threshold=0.8)
        await metric.evaluate(output="my summary", source="my source material")
        last_call = judge.calls[-1]
        assert "my summary" in last_call["prompt"]
        assert "my source material" in last_call["prompt"]

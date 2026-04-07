from __future__ import annotations

import pytest

from checkllm.metrics.response_completeness import ResponseCompletenessMetric
from checkllm.testing import MockJudge


class TestResponseCompletenessMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_complete(self, judge):
        judge.set_default(score=0.95, reasoning="All parts of the question addressed")
        metric = ResponseCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output=(
                "Python was created in 1991. It uses indentation for blocks. "
                "It is dynamically typed."
            ),
            query=(
                "When was Python created? How does it handle code blocks? "
                "Is it statically or dynamically typed?"
            ),
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "response_completeness"

    @pytest.mark.asyncio
    async def test_fails_when_incomplete(self, judge):
        judge.set_default(score=0.3, reasoning="Only one of three parts addressed")
        metric = ResponseCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python was created in 1991.",
            query=(
                "When was Python created? How does it handle code blocks? "
                "Is it statically or dynamically typed?"
            ),
        )
        assert result.passed is False
        assert result.score == 0.3

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = ResponseCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", query="test query")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", query="test query")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_query(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = ResponseCompletenessMetric(judge=judge, threshold=0.8)
        await metric.evaluate(output="my output text", query="my query text")
        last_call = judge.calls[-1]
        assert "my output text" in last_call["prompt"]
        assert "my query text" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = ResponseCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", query="test query")
        assert result.latency_ms >= 0

from __future__ import annotations

import pytest

from checkllm.metrics.answer_completeness import AnswerCompletenessMetric
from checkllm.testing import MockJudge


class TestAnswerCompletenessMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_complete(self, judge):
        judge.set_default(score=0.95, reasoning="Answer addresses all parts")
        metric = AnswerCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python is a high-level, interpreted language created by Guido van Rossum in 1991.",
            query="What is Python and who created it?",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "answer_completeness"

    @pytest.mark.asyncio
    async def test_fails_when_incomplete(self, judge):
        judge.set_default(score=0.3, reasoning="Answer only addresses part of the question")
        metric = AnswerCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python is a programming language.",
            query="What is Python, who created it, and when was it first released?",
        )
        assert result.passed is False
        assert result.score == 0.3

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = AnswerCompletenessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", query="test")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", query="test")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Partially complete")
        metric = AnswerCompletenessMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(output="test", query="test")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, judge):
        metric = AnswerCompletenessMetric(judge=judge, threshold=0.8)
        metric.system_prompt = "Custom completeness prompt"
        judge.set_default(score=0.9, reasoning="ok")
        await metric.evaluate(output="test", query="test")
        last_call = judge.calls[-1]
        assert last_call["system_prompt"] == "Custom completeness prompt"

    @pytest.mark.asyncio
    async def test_score_range(self, judge):
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            judge.set_default(score=score, reasoning=f"Score {score}")
            metric = AnswerCompletenessMetric(judge=judge, threshold=0.8)
            result = await metric.evaluate(output="test", query="test")
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_query(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = AnswerCompletenessMetric(judge=judge, threshold=0.8)
        await metric.evaluate(output="my output", query="my query")
        last_call = judge.calls[-1]
        assert "my output" in last_call["prompt"]
        assert "my query" in last_call["prompt"]

from __future__ import annotations

import pytest

from checkllm.metrics.context_relevance import ContextRelevanceMetric
from checkllm.testing import MockJudge


class TestContextRelevanceMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_relevant(self, judge):
        judge.set_default(score=0.95, reasoning="Context is highly relevant")
        metric = ContextRelevanceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            context="Python is a programming language known for its readability.",
            query="What is Python?",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "context_relevance"

    @pytest.mark.asyncio
    async def test_fails_when_irrelevant(self, judge):
        judge.set_default(score=0.1, reasoning="Context is unrelated to query")
        metric = ContextRelevanceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            context="The recipe calls for two cups of flour.",
            query="What is Python?",
        )
        assert result.passed is False
        assert result.score == 0.1

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = ContextRelevanceMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(context="test", query="test")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(context="test", query="test")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Partially relevant")
        metric = ContextRelevanceMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(context="test", query="test")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, judge):
        metric = ContextRelevanceMetric(judge=judge, threshold=0.8)
        metric.system_prompt = "Custom relevance prompt"
        judge.set_default(score=0.9, reasoning="ok")
        await metric.evaluate(context="test", query="test")
        last_call = judge.calls[-1]
        assert last_call["system_prompt"] == "Custom relevance prompt"

    @pytest.mark.asyncio
    async def test_score_range(self, judge):
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            judge.set_default(score=score, reasoning=f"Score {score}")
            metric = ContextRelevanceMetric(judge=judge, threshold=0.8)
            result = await metric.evaluate(context="test", query="test")
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_prompt_contains_context_and_query(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = ContextRelevanceMetric(judge=judge, threshold=0.8)
        await metric.evaluate(context="my context", query="my query")
        last_call = judge.calls[-1]
        assert "my context" in last_call["prompt"]
        assert "my query" in last_call["prompt"]

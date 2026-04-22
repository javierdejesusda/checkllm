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

    @pytest.mark.asyncio
    async def test_answer_aware_prompt_includes_answer(self, judge):
        """When an answer is supplied, it must flow into the judge prompt so
        the grader can check whether the context justifies that specific
        claim — not just topical overlap with the query.
        """
        judge.set_default(score=0.7, reasoning="ok")
        metric = ContextRelevanceMetric(judge=judge, threshold=0.8)
        await metric.evaluate(
            context="Python was created by Guido van Rossum in 1991.",
            query="Who created Python?",
            answer="Guido van Rossum",
        )
        last_call = judge.calls[-1]
        assert "Guido van Rossum" in last_call["prompt"]
        assert "System Answer" in last_call["prompt"]
        assert "justify the answer" in last_call["prompt"].lower()

    @pytest.mark.asyncio
    async def test_answerless_prompt_omits_answer_section(self, judge):
        """Backwards compatibility: when no answer is supplied the prompt
        must not mention a system answer at all — callers that pass only
        context + query keep the original behaviour.
        """
        judge.set_default(score=0.9, reasoning="ok")
        metric = ContextRelevanceMetric(judge=judge, threshold=0.8)
        await metric.evaluate(
            context="Python is a programming language.",
            query="What is Python?",
        )
        last_call = judge.calls[-1]
        assert "System Answer" not in last_call["prompt"]
        assert "justify the answer" not in last_call["prompt"].lower()

    @pytest.mark.asyncio
    async def test_system_prompt_is_precision_oriented(self, judge):
        """System prompt must push the judge toward a precision/ratio rubric
        so scores spread out instead of clustering near 1.0 for any context
        that merely touches the query topic.
        """
        judge.set_default(score=0.9, reasoning="ok")
        metric = ContextRelevanceMetric(judge=judge, threshold=0.8)
        await metric.evaluate(context="irrelevant filler", query="q")
        system_prompt = judge.calls[-1]["system_prompt"].lower()
        assert (
            "precision" in system_prompt
        ), "system prompt must frame grading as precision / signal-to-noise"
        assert "ratio" in system_prompt, "system prompt must mention the relevant/total ratio"

from __future__ import annotations

import pytest

from checkllm.metrics.context_entity_recall import ContextEntityRecallMetric
from checkllm.testing import MockJudge


class TestContextEntityRecallMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_entities_present(self, judge):
        judge.set_default(score=0.95, reasoning="All entities found in context")
        metric = ContextEntityRecallMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            context="Guido van Rossum created Python in 1991 at CWI in the Netherlands.",
            reference="Python was created by Guido van Rossum in 1991.",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "context_entity_recall"

    @pytest.mark.asyncio
    async def test_fails_when_entities_missing(self, judge):
        judge.set_default(score=0.2, reasoning="Most reference entities missing from context")
        metric = ContextEntityRecallMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            context="Programming languages are useful tools for software development.",
            reference="Python was created by Guido van Rossum in 1991.",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = ContextEntityRecallMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(context="test context", reference="test ref")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(context="test context", reference="test ref")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Partial entity coverage")
        metric = ContextEntityRecallMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(context="test context", reference="test ref")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_context_and_reference(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = ContextEntityRecallMetric(judge=judge, threshold=0.8)
        await metric.evaluate(context="my context text", reference="my reference text")
        last_call = judge.calls[-1]
        assert "my context text" in last_call["prompt"]
        assert "my reference text" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = ContextEntityRecallMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(context="test", reference="test ref")
        assert result.latency_ms >= 0

from __future__ import annotations

import pytest

from checkllm.metrics.factual_correctness import FactualCorrectnessMetric
from checkllm.testing import MockJudge


class TestFactualCorrectnessMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_factually_correct(self, judge):
        judge.set_default(score=0.95, reasoning="All claims match the reference")
        metric = FactualCorrectnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python was created by Guido van Rossum in 1991.",
            reference="Python is a programming language created by Guido van Rossum in 1991.",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "factual_correctness"

    @pytest.mark.asyncio
    async def test_fails_when_factually_incorrect(self, judge):
        judge.set_default(score=0.2, reasoning="Multiple claims contradict the reference")
        metric = FactualCorrectnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="Python was created by James Gosling in 1985.",
            reference="Python is a programming language created by Guido van Rossum in 1991.",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = FactualCorrectnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", reference="test ref")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="test", reference="test ref")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Partially correct")
        metric = FactualCorrectnessMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(output="test", reference="test ref")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_reference(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = FactualCorrectnessMetric(judge=judge, threshold=0.8)
        await metric.evaluate(output="my output text", reference="my reference text")
        last_call = judge.calls[-1]
        assert "my output text" in last_call["prompt"]
        assert "my reference text" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = FactualCorrectnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="test", reference="test ref")
        assert result.latency_ms >= 0

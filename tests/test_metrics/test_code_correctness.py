from __future__ import annotations

import pytest

from checkllm.metrics.code_correctness import CodeCorrectnessMetric
from checkllm.testing import MockJudge


class TestCodeCorrectnessMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_correct(self, judge):
        judge.set_default(score=0.95, reasoning="Code correctly implements requirements")
        metric = CodeCorrectnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="def add(a, b):\n    return a + b",
            requirements="Write a function that adds two numbers and returns the result.",
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "code_correctness"

    @pytest.mark.asyncio
    async def test_fails_when_incorrect(self, judge):
        judge.set_default(score=0.2, reasoning="Code has logical errors")
        metric = CodeCorrectnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            output="def add(a, b):\n    return a - b",
            requirements="Write a function that adds two numbers and returns the result.",
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = CodeCorrectnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="code", requirements="requirements")
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(output="code", requirements="requirements")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Partially correct")
        metric = CodeCorrectnessMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(output="code", requirements="requirements")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_prompt_contains_output_and_requirements(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = CodeCorrectnessMetric(judge=judge, threshold=0.8)
        await metric.evaluate(
            output="def my_func(): pass",
            requirements="Implement a function that does X",
        )
        last_call = judge.calls[-1]
        assert "def my_func(): pass" in last_call["prompt"]
        assert "Implement a function that does X" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = CodeCorrectnessMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(output="code", requirements="requirements")
        assert result.latency_ms >= 0

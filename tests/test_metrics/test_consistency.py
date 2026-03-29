from __future__ import annotations

import pytest

from checkllm.metrics.consistency import ConsistencyMetric
from checkllm.testing import MockJudge


class TestConsistencyMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_passes_when_consistent(self, judge):
        judge.set_default(score=0.95, reasoning="All outputs are consistent")
        metric = ConsistencyMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            outputs=[
                "Python was created by Guido van Rossum in 1991.",
                "Guido van Rossum created Python, which was first released in 1991.",
                "In 1991, Guido van Rossum released Python.",
            ],
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "consistency"

    @pytest.mark.asyncio
    async def test_fails_when_inconsistent(self, judge):
        judge.set_default(score=0.2, reasoning="Outputs contradict each other")
        metric = ConsistencyMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(
            outputs=[
                "Python was created in 1991.",
                "Python was created in 2000.",
                "Python was created in 1985.",
            ],
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, judge):
        judge.set_default(score=0.8, reasoning="Exactly at threshold")
        metric = ConsistencyMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(outputs=["a", "b"])
        assert result.passed is True

        judge.set_default(score=0.79, reasoning="Just below threshold")
        result = await metric.evaluate(outputs=["a", "b"])
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self, judge):
        judge.set_default(score=0.6, reasoning="Partially consistent")
        metric = ConsistencyMetric(judge=judge, threshold=0.5)
        result = await metric.evaluate(outputs=["a", "b"])
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, judge):
        metric = ConsistencyMetric(judge=judge, threshold=0.8)
        metric.system_prompt = "Custom consistency prompt"
        judge.set_default(score=0.9, reasoning="ok")
        await metric.evaluate(outputs=["a", "b"])
        last_call = judge.calls[-1]
        assert last_call["system_prompt"] == "Custom consistency prompt"

    @pytest.mark.asyncio
    async def test_score_range(self, judge):
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            judge.set_default(score=score, reasoning=f"Score {score}")
            metric = ConsistencyMetric(judge=judge, threshold=0.8)
            result = await metric.evaluate(outputs=["a", "b"])
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_prompt_contains_all_outputs(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = ConsistencyMetric(judge=judge, threshold=0.8)
        await metric.evaluate(outputs=["first output", "second output", "third output"])
        last_call = judge.calls[-1]
        assert "first output" in last_call["prompt"]
        assert "second output" in last_call["prompt"]
        assert "third output" in last_call["prompt"]

    @pytest.mark.asyncio
    async def test_two_outputs(self, judge):
        judge.set_default(score=0.9, reasoning="Consistent pair")
        metric = ConsistencyMetric(judge=judge, threshold=0.8)
        result = await metric.evaluate(outputs=["output A", "output B"])
        assert result.passed is True

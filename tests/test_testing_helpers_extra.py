"""Additional tests for checkllm.testing to cover remaining branches."""

from __future__ import annotations

import pytest

from checkllm.testing import MockJudge, make_collector, assert_score_above


class TestMockJudgeSetDefault:
    @pytest.mark.asyncio
    async def test_set_default_changes_score(self):
        judge = MockJudge(default_score=0.5)
        judge.set_default(0.95, reasoning="Updated default")
        assert judge.default_score == 0.95
        assert judge.default_reasoning == "Updated default"
        response = await judge.evaluate("test prompt")
        assert response.score == 0.95

    @pytest.mark.asyncio
    async def test_set_default_default_reasoning(self):
        judge = MockJudge()
        judge.set_default(0.7)
        assert judge.default_score == 0.7
        assert judge.default_reasoning == "Mock evaluation"


class TestMockJudgeAssertCalledWithMetric:
    @pytest.mark.asyncio
    async def test_assert_called_with_metric_pass(self):
        judge = MockJudge()
        # Call with a system_prompt containing "hallucination" keyword
        await judge.evaluate("test", system_prompt="You are a hallucination evaluator.")
        # Should not raise
        judge.assert_called(metric="hallucination")

    @pytest.mark.asyncio
    async def test_assert_called_with_metric_fail(self):
        judge = MockJudge()
        await judge.evaluate("test", system_prompt="You are a relevance evaluator.")
        with pytest.raises(AssertionError, match="never called for metric"):
            judge.assert_called(metric="hallucination")


class TestMockJudgeAssertCallCountWithMetric:
    @pytest.mark.asyncio
    async def test_assert_call_count_with_metric(self):
        judge = MockJudge()
        await judge.evaluate("a", system_prompt="hallucination")
        await judge.evaluate("b", system_prompt="hallucination")
        await judge.evaluate("c", system_prompt="relevance checker")
        judge.assert_call_count(2, metric="hallucination")
        judge.assert_call_count(1, metric="relevance")

    @pytest.mark.asyncio
    async def test_assert_call_count_with_metric_fail(self):
        judge = MockJudge()
        await judge.evaluate("a")
        with pytest.raises(AssertionError):
            judge.assert_call_count(5, metric="hallucination")


class TestMakeCollectorExtra:
    def test_make_collector_with_config_kwargs(self):
        collector = make_collector(threshold=0.7, budget=5.0)
        assert collector.config.default_threshold == 0.7
        assert collector.config.budget == 5.0


class TestAssertScoreAboveExtra:
    def test_assert_score_above_passes_multiple_results(self):
        collector = make_collector()
        collector.contains("hello world", "hello")
        collector.contains("hello earth", "hello")
        assert_score_above(collector, "contains", 0.5)

    def test_assert_score_above_fails_when_score_low(self):
        collector = make_collector()
        collector.not_contains("hello", "hello")  # score=0.0 (fails)
        with pytest.raises(AssertionError, match="scored"):
            assert_score_above(collector, "not_contains", 0.5)

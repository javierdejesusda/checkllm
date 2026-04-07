"""Tests for MockJudge and testing helpers."""
import pytest

from checkllm.testing import MockJudge, make_collector, assert_all_passed, assert_score_above


class TestMockJudge:
    @pytest.mark.asyncio
    async def test_default_response(self):
        judge = MockJudge(default_score=0.9)
        response = await judge.evaluate("test prompt")
        assert response.score == 0.9
        assert len(judge.calls) == 1

    @pytest.mark.asyncio
    async def test_queued_responses(self):
        judge = MockJudge()
        judge.add_response("hallucination", score=0.95, reasoning="Perfect")
        judge.add_response("hallucination", score=0.7, reasoning="Weak")

        # First call returns 0.95
        r1 = await judge.evaluate("test", system_prompt="hallucination evaluator")
        assert r1.score == 0.95
        # Second call returns 0.7
        r2 = await judge.evaluate("test", system_prompt="hallucination evaluator")
        assert r2.score == 0.7
        # Third call falls back to default
        r3 = await judge.evaluate("test", system_prompt="hallucination evaluator")
        assert r3.score == judge.default_score

    @pytest.mark.asyncio
    async def test_assert_called(self):
        judge = MockJudge()
        with pytest.raises(AssertionError):
            judge.assert_called()
        await judge.evaluate("test")
        judge.assert_called()

    @pytest.mark.asyncio
    async def test_assert_call_count(self):
        judge = MockJudge()
        await judge.evaluate("a")
        await judge.evaluate("b")
        judge.assert_call_count(2)

    @pytest.mark.asyncio
    async def test_reset(self):
        judge = MockJudge()
        await judge.evaluate("test")
        judge.reset()
        assert len(judge.calls) == 0

    def test_repr(self):
        judge = MockJudge(default_score=0.5)
        assert "0.5" in repr(judge)


class TestMakeCollector:
    def test_creates_collector_with_mock(self):
        collector = make_collector()
        collector.contains("hello world", "hello")
        assert len(collector.results) == 1

    def test_with_custom_judge(self):
        judge = MockJudge(default_score=0.95)
        collector = make_collector(judge=judge, threshold=0.9)
        assert collector.config.default_threshold == 0.9

    def test_cache_disabled_by_default(self):
        collector = make_collector()
        assert collector.config.cache_enabled is False


class TestAssertHelpers:
    def test_assert_all_passed(self):
        collector = make_collector()
        collector.contains("hello world", "hello")
        collector.not_contains("hello world", "bye")
        assert_all_passed(collector)

    def test_assert_all_passed_fails(self):
        collector = make_collector()
        collector.contains("hello", "bye")  # fails
        with pytest.raises(AssertionError, match="1 check.*failed"):
            assert_all_passed(collector)

    def test_assert_score_above(self):
        collector = make_collector()
        collector.contains("hello world", "hello")  # score=1.0
        assert_score_above(collector, "contains", 0.5)

    def test_assert_score_above_fails(self):
        collector = make_collector()
        collector.contains("hello", "bye")  # score=0.0
        with pytest.raises(AssertionError):
            assert_score_above(collector, "contains", 0.5)

    def test_assert_score_above_missing_metric(self):
        collector = make_collector()
        with pytest.raises(AssertionError, match="No results found"):
            assert_score_above(collector, "nonexistent", 0.5)

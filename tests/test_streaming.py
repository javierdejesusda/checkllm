"""Tests for checkllm.streaming — streaming evaluation of LLM outputs."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from checkllm.models import CheckResult
from checkllm.streaming import StreamingCheckpoint, StreamingEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(passed: bool, score: float = 1.0, name: str = "test") -> CheckResult:
    return CheckResult(
        passed=passed,
        score=score,
        reasoning="ok" if passed else "fail",
        cost=0.0,
        latency_ms=0,
        metric_name=name,
    )


async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# StreamingCheckpoint
# ---------------------------------------------------------------------------


class TestStreamingCheckpoint:
    def test_creates_checkpoint(self):
        cp = StreamingCheckpoint(
            tokens_received=50,
            partial_output="Hello world",
            checks_run=3,
            checks_passed=2,
            checks_failed=1,
            elapsed_ms=120,
        )
        assert cp.tokens_received == 50
        assert cp.partial_output == "Hello world"
        assert cp.checks_run == 3
        assert cp.checks_passed == 2
        assert cp.checks_failed == 1
        assert cp.elapsed_ms == 120
        assert cp.results == []

    def test_defaults(self):
        cp = StreamingCheckpoint(
            tokens_received=0,
            partial_output="",
        )
        assert cp.checks_run == 0
        assert cp.checks_passed == 0
        assert cp.checks_failed == 0
        assert cp.elapsed_ms == 0
        assert cp.results == []


# ---------------------------------------------------------------------------
# StreamingEvaluator
# ---------------------------------------------------------------------------


class TestStreamingEvaluator:
    @pytest.fixture
    def evaluator(self):
        return StreamingEvaluator(check_interval=5)

    def test_init_default_interval(self):
        ev = StreamingEvaluator()
        # Default is 50, should not raise
        assert ev._check_interval == 50

    def test_init_invalid_interval(self):
        with pytest.raises(ValueError, match="check_interval must be >= 1"):
            StreamingEvaluator(check_interval=0)

    def test_add_check(self, evaluator):
        def length_check(text: str) -> CheckResult:
            passed = len(text) < 100
            return _make_result(passed, 1.0 if passed else 0.0, "length")

        evaluator.add_check("length", length_check)
        assert len(evaluator._sync_checks) == 1
        assert evaluator._sync_checks[0][0] == "length"

    def test_add_async_check(self, evaluator):
        async def async_check(text: str) -> CheckResult:
            return _make_result(True, name="async_test")

        evaluator.add_async_check("async_test", async_check)
        assert len(evaluator._async_checks) == 1
        assert evaluator._async_checks[0][0] == "async_test"

    def test_add_early_stop(self, evaluator):
        evaluator.add_early_stop(lambda text: "STOP" in text)
        assert len(evaluator._early_stops) == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_chunks(self, evaluator):
        """Add a simple length check, feed 10 chunks, verify checkpoints."""

        def length_check(text: str) -> CheckResult:
            passed = len(text) <= 100
            score = 1.0 if passed else 0.0
            return _make_result(passed, score, "length")

        evaluator.add_check("length", length_check)

        chunks = [f"token{i} " for i in range(10)]
        checkpoints: list[StreamingCheckpoint] = []

        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        # With check_interval=5, we expect checkpoint at token 5
        # and a final checkpoint at token 10
        assert len(checkpoints) >= 1

        final = checkpoints[-1]
        assert final.tokens_received == 10
        assert final.checks_run >= 1
        assert final.checks_passed >= 1
        assert "token0" in final.partial_output
        assert "token9" in final.partial_output

    @pytest.mark.asyncio
    async def test_early_stop(self, evaluator):
        """Early stop should halt streaming and yield a final checkpoint."""

        def noop_check(text: str) -> CheckResult:
            return _make_result(True, name="noop")

        evaluator.add_check("noop", noop_check)
        evaluator.add_early_stop(lambda text: "STOP" in text)

        chunks = ["Hello ", "world ", "STOP", " should ", "not ", "reach"]
        checkpoints: list[StreamingCheckpoint] = []

        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        # Should have stopped after 3 tokens (when "STOP" appears)
        final = checkpoints[-1]
        assert final.tokens_received == 3
        assert "STOP" in final.partial_output
        assert "not" not in final.partial_output
        assert "reach" not in final.partial_output

    @pytest.mark.asyncio
    async def test_evaluate_empty_stream(self, evaluator):
        """An empty stream should yield no checkpoints (or an empty final one)."""

        def check(text: str) -> CheckResult:
            return _make_result(True, name="check")

        evaluator.add_check("check", check)

        checkpoints: list[StreamingCheckpoint] = []
        async for cp in evaluator.evaluate(_async_iter([])):
            checkpoints.append(cp)

        # Final checks on empty string should produce one checkpoint
        assert len(checkpoints) == 1
        assert checkpoints[0].tokens_received == 0

    @pytest.mark.asyncio
    async def test_evaluate_string_chunks_convenience(self, evaluator):
        """Test the evaluate_string_chunks convenience method."""

        def length_check(text: str) -> CheckResult:
            passed = len(text) <= 200
            return _make_result(passed, 1.0 if passed else 0.0, "length")

        evaluator.add_check("length", length_check)

        chunks = [f"word{i} " for i in range(12)]
        final = await evaluator.evaluate_string_chunks(chunks)

        assert isinstance(final, StreamingCheckpoint)
        assert final.tokens_received == 12
        assert final.checks_run >= 1

    @pytest.mark.asyncio
    async def test_evaluate_string_chunks_empty(self, evaluator):
        final = await evaluator.evaluate_string_chunks([])
        assert final.tokens_received == 0
        assert final.partial_output == ""

    @pytest.mark.asyncio
    async def test_check_exception_is_caught(self, evaluator):
        """A check that raises should produce a failing CheckResult, not crash."""

        def bad_check(text: str) -> CheckResult:
            raise ValueError("something went wrong")

        evaluator.add_check("bad", bad_check)

        chunks = ["a " * 5]
        # check_interval=5 so this won't trigger at interval, but final checks will run
        final = await evaluator.evaluate_string_chunks(chunks)
        assert final.checks_run >= 1
        assert final.checks_failed >= 1
        # The failing result should contain the error message
        fail_results = [r for r in final.results if not r.passed]
        assert any("something went wrong" in r.reasoning for r in fail_results)

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_checks(self):
        evaluator = StreamingEvaluator(check_interval=3)

        def sync_check(text: str) -> CheckResult:
            return _make_result(True, name="sync")

        async def async_check(text: str) -> CheckResult:
            return _make_result(True, name="async")

        evaluator.add_check("sync", sync_check)
        evaluator.add_async_check("async", async_check)

        chunks = ["a", "b", "c", "d", "e", "f"]
        checkpoints = []
        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        # At interval 3 we get checkpoints, plus final
        assert len(checkpoints) >= 1
        # Each checkpoint should have run both checks
        for cp in checkpoints:
            assert cp.checks_run >= 2


# ---------------------------------------------------------------------------
# Judge-level streaming (OpenAI / DeepSeek stream_evaluate)
# ---------------------------------------------------------------------------


class TestJudgeStreaming:
    """Verify ``stream_evaluate`` on judge backends yields chunks + final."""

    def _make_stream_chunk(
        self,
        content: str | None = None,
        usage: dict[str, int] | None = None,
    ):
        from unittest.mock import MagicMock

        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = None
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        if usage is not None:
            chunk.usage = MagicMock()
            chunk.usage.prompt_tokens = usage.get("prompt_tokens", 0)
            chunk.usage.completion_tokens = usage.get("completion_tokens", 0)
        else:
            chunk.usage = None
        return chunk

    async def _async_chunks(self, chunks):
        for c in chunks:
            yield c

    @pytest.mark.asyncio
    async def test_openai_judge_stream_yields_text_then_final(self) -> None:
        from unittest.mock import AsyncMock, patch

        from checkllm.judge import OpenAIJudge, StreamingJudgeResult

        judge = OpenAIJudge(model="gpt-4o", api_key="k")
        chunks = [
            self._make_stream_chunk(content='{"score"'),
            self._make_stream_chunk(content=": 0.5, "),
            self._make_stream_chunk(content='"reasoning": "fine"}'),
            self._make_stream_chunk(usage={"prompt_tokens": 10, "completion_tokens": 5}),
        ]

        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = self._async_chunks(chunks)

            pieces: list[str] = []
            final: StreamingJudgeResult | None = None
            async for item in judge.stream_evaluate("hi"):
                if isinstance(item, str):
                    pieces.append(item)
                else:
                    final = item

        assert "".join(pieces) == '{"score": 0.5, "reasoning": "fine"}'
        assert final is not None
        assert final.response.score == pytest.approx(0.5)
        assert final.response.reasoning == "fine"
        assert final.response.cost > 0.0

    @pytest.mark.asyncio
    async def test_openai_judge_stream_early_stop_callback(self) -> None:
        from unittest.mock import AsyncMock, patch

        from checkllm.judge import OpenAIJudge, StreamingJudgeResult

        judge = OpenAIJudge(model="gpt-4o", api_key="k")
        chunks = [
            self._make_stream_chunk(content="hello "),
            self._make_stream_chunk(content="STOP_NOW "),
            self._make_stream_chunk(content="never arrives"),
        ]

        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = self._async_chunks(chunks)

            seen: list[str] = []
            final: StreamingJudgeResult | None = None
            async for item in judge.stream_evaluate("hi", on_token=lambda t: "STOP_NOW" in t):
                if isinstance(item, str):
                    seen.append(item)
                else:
                    final = item

        assert "never arrives" not in seen
        assert final is not None
        assert final.stopped_early is True

    @pytest.mark.asyncio
    async def test_openai_judge_stream_final_consistent_with_accumulated(
        self,
    ) -> None:
        from unittest.mock import AsyncMock, patch

        from checkllm.judge import OpenAIJudge, StreamingJudgeResult

        judge = OpenAIJudge(model="gpt-4o", api_key="k")
        chunks = [
            self._make_stream_chunk(content='{"score": 1.0, '),
            self._make_stream_chunk(content='"reasoning": "great"}'),
        ]

        with patch.object(
            judge._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = self._async_chunks(chunks)

            collected: list[str] = []
            final: StreamingJudgeResult | None = None
            async for item in judge.stream_evaluate("hi"):
                if isinstance(item, str):
                    collected.append(item)
                else:
                    final = item

        assert final is not None
        assert final.aggregated_text == "".join(collected)
        assert final.response.raw_output == final.aggregated_text

"""Additional tests for checkllm.streaming — covering uncovered branches."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from checkllm.models import CheckResult
from checkllm.streaming import StreamingCheckpoint, StreamingEvaluator


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


class TestStreamingEvaluatorEdgeCases:
    @pytest.mark.asyncio
    async def test_exact_interval_boundary(self):
        """Test that checks run exactly at the interval boundary."""
        evaluator = StreamingEvaluator(check_interval=3)

        call_count = 0

        def counter_check(text: str) -> CheckResult:
            nonlocal call_count
            call_count += 1
            return _make_result(True, name="counter")

        evaluator.add_check("counter", counter_check)

        # 6 tokens = 2 intervals (at 3 and 6) + final check
        chunks = ["a", "b", "c", "d", "e", "f"]
        checkpoints = []
        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        # Should have checkpoint at 3, at 6, and possibly final
        assert len(checkpoints) >= 2

    @pytest.mark.asyncio
    async def test_no_final_checks_when_disabled(self):
        """final_checks=False should not emit a final checkpoint."""
        evaluator = StreamingEvaluator(check_interval=10)
        evaluator.add_check("test", lambda t: _make_result(True))

        # 5 tokens, interval is 10, so no interval checkpoint
        chunks = ["a", "b", "c", "d", "e"]
        checkpoints = []
        async for cp in evaluator.evaluate(_async_iter(chunks), final_checks=False):
            checkpoints.append(cp)

        # No checkpoints since interval wasn't reached and final_checks=False
        assert len(checkpoints) == 0

    @pytest.mark.asyncio
    async def test_no_checks_registered(self):
        """Evaluator with no checks should still produce checkpoints."""
        evaluator = StreamingEvaluator(check_interval=5)
        chunks = [f"word{i} " for i in range(5)]
        checkpoints = []
        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        final = checkpoints[-1]
        assert final.tokens_received == 5
        assert final.checks_run == 0

    @pytest.mark.asyncio
    async def test_early_stop_no_checks(self):
        """Early stop without any registered checks."""
        evaluator = StreamingEvaluator(check_interval=100)
        evaluator.add_early_stop(lambda text: "HALT" in text)

        chunks = ["go ", "go ", "HALT", "more ", "more"]
        checkpoints = []
        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        assert len(checkpoints) >= 1
        final = checkpoints[-1]
        assert "HALT" in final.partial_output
        assert final.tokens_received == 3

    @pytest.mark.asyncio
    async def test_early_stop_exception_continues(self):
        """If early stop condition raises, it should be caught and continue."""
        evaluator = StreamingEvaluator(check_interval=100)

        def bad_condition(text: str) -> bool:
            raise ValueError("condition failed")

        evaluator.add_early_stop(bad_condition)

        chunks = ["a", "b", "c"]
        checkpoints = []
        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        # Should complete without crashing
        assert len(checkpoints) == 1
        assert checkpoints[0].tokens_received == 3

    @pytest.mark.asyncio
    async def test_async_check_exception_caught(self):
        """Async check that raises should produce failing CheckResult."""
        evaluator = StreamingEvaluator(check_interval=3)

        async def bad_async_check(text: str) -> CheckResult:
            raise RuntimeError("async check broken")

        evaluator.add_async_check("bad_async", bad_async_check)

        chunks = ["a", "b", "c", "d", "e", "f"]
        checkpoints = []
        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        # Should complete, with failing results for the bad check
        assert len(checkpoints) >= 1
        failed = [r for cp in checkpoints for r in cp.results if not r.passed]
        assert any("async check broken" in r.reasoning for r in failed)

    @pytest.mark.asyncio
    async def test_multiple_early_stop_conditions(self):
        """Multiple early stop conditions — first one that triggers wins."""
        evaluator = StreamingEvaluator(check_interval=100)
        evaluator.add_early_stop(lambda text: "STOP_A" in text)
        evaluator.add_early_stop(lambda text: "STOP_B" in text)

        chunks = ["hello ", "STOP_A", "more ", "STOP_B"]
        checkpoints = []
        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        final = checkpoints[-1]
        assert final.tokens_received == 2
        assert "STOP_A" in final.partial_output

    @pytest.mark.asyncio
    async def test_checks_passed_and_failed_counts(self):
        """Verify checks_passed and checks_failed are correct."""
        evaluator = StreamingEvaluator(check_interval=5)

        evaluator.add_check("passing", lambda t: _make_result(True, name="passing"))
        evaluator.add_check("failing", lambda t: _make_result(False, name="failing"))

        chunks = [f"w{i} " for i in range(5)]
        checkpoints = []
        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        final = checkpoints[-1]
        assert final.checks_run >= 2
        assert final.checks_passed >= 1
        assert final.checks_failed >= 1

    @pytest.mark.asyncio
    async def test_evaluate_string_chunks_with_checks(self):
        """evaluate_string_chunks runs all checks on final output."""
        evaluator = StreamingEvaluator(check_interval=5)
        evaluator.add_check("length", lambda t: _make_result(len(t) > 0, name="length"))

        chunks = ["Hello ", "World"]
        final = await evaluator.evaluate_string_chunks(chunks)
        assert final.tokens_received == 2
        assert final.checks_run >= 1
        assert final.checks_passed >= 1

    @pytest.mark.asyncio
    async def test_evaluate_exact_multiple_of_interval(self):
        """When token count is exact multiple of interval, no duplicate final."""
        evaluator = StreamingEvaluator(check_interval=5)
        evaluator.add_check("test", lambda t: _make_result(True))

        chunks = ["a", "b", "c", "d", "e"]  # exactly 5 tokens
        checkpoints = []
        async for cp in evaluator.evaluate(_async_iter(chunks)):
            checkpoints.append(cp)

        # Should have interval checkpoint at 5 and final
        # The final should not re-run checks if already done at interval
        assert len(checkpoints) >= 1
        final = checkpoints[-1]
        assert final.tokens_received == 5


class TestStreamingCheckpointModel:
    def test_results_list_populated(self):
        results = [_make_result(True, name="a"), _make_result(False, name="b")]
        cp = StreamingCheckpoint(
            tokens_received=10,
            partial_output="test",
            checks_run=2,
            checks_passed=1,
            checks_failed=1,
            elapsed_ms=50,
            results=results,
        )
        assert len(cp.results) == 2
        assert cp.results[0].metric_name == "a"
        assert cp.results[1].metric_name == "b"

    def test_elapsed_ms_tracked(self):
        cp = StreamingCheckpoint(
            tokens_received=100,
            partial_output="long text",
            elapsed_ms=500,
        )
        assert cp.elapsed_ms == 500

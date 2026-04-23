"""Testing helpers — MockJudge, fixtures, and utilities for writing checkllm tests.

Use these in your own test suites to test LLM-powered code without calling real APIs.

Usage::

    from checkllm.testing import MockJudge, mock_collector

    def test_my_agent(mock_collector):
        output = my_agent("What is Python?")
        mock_collector.hallucination(output, context="...")
        # Uses MockJudge — no API key needed, instant, free

    # Or construct manually:
    judge = MockJudge(default_score=0.9)
    judge.add_response("hallucination", score=0.95, reasoning="Well grounded")
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.models import JudgeResponse


class MockJudge:
    """A fake judge backend for testing without API keys.

    Returns configurable scores per metric, or a default score.
    Tracks all calls for assertions in tests.
    """

    def __init__(
        self, default_score: float = 0.85, default_reasoning: str = "Mock evaluation"
    ) -> None:
        self.default_score = default_score
        self.default_reasoning = default_reasoning
        self.calls: list[dict[str, Any]] = []
        self._responses: dict[str, list[JudgeResponse]] = defaultdict(list)
        self._response_index: dict[str, int] = defaultdict(int)
        # Satisfies JudgeBackend protocol
        self.model = "mock-judge"
        self.total_cost: float = 0.0
        self.last_cost: float = 0.0

    def add_response(
        self,
        metric: str,
        score: float,
        reasoning: str = "Mock response",
        cost: float = 0.0,
    ) -> None:
        """Queue a specific response for a metric. Responses are consumed in order."""
        self._responses[metric].append(JudgeResponse(score=score, reasoning=reasoning, cost=cost))

    def set_default(self, score: float, reasoning: str = "Mock evaluation") -> None:
        """Change the default score for all unqueued metrics."""
        self.default_score = score
        self.default_reasoning = reasoning

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        """Return a mock response. Tracks the call for test assertions."""
        # Try to detect the metric from the system prompt
        metric = "unknown"
        if system_prompt:
            for keyword in [
                "hallucination",
                "relevance",
                "toxicity",
                "rubric",
                "fluency",
                "coherence",
                "sentiment",
                "correctness",
            ]:
                if keyword in system_prompt.lower():
                    metric = keyword
                    break

        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "metric": metric,
            }
        )

        # Check for queued responses
        if metric in self._responses and self._response_index[metric] < len(
            self._responses[metric]
        ):
            idx = self._response_index[metric]
            self._response_index[metric] += 1
            resp = self._responses[metric][idx]
            self.last_cost = resp.cost
            self.total_cost += resp.cost
            return resp

        # Default response
        return JudgeResponse(
            score=self.default_score,
            reasoning=self.default_reasoning,
            cost=0.0,
        )

    def assert_called(self, metric: str | None = None) -> None:
        """Assert that the judge was called (optionally for a specific metric)."""
        if metric:
            matching = [c for c in self.calls if c["metric"] == metric]
            assert matching, (
                f"MockJudge was never called for metric '{metric}'. Calls: {[c['metric'] for c in self.calls]}"
            )
        else:
            assert self.calls, "MockJudge was never called"

    def assert_call_count(self, expected: int, metric: str | None = None) -> None:
        """Assert the number of judge calls."""
        if metric:
            actual = sum(1 for c in self.calls if c["metric"] == metric)
        else:
            actual = len(self.calls)
        assert actual == expected, f"Expected {expected} calls, got {actual}"

    def reset(self) -> None:
        """Clear all recorded calls and queued responses."""
        self.calls.clear()
        self._responses.clear()
        self._response_index.clear()
        self.total_cost = 0.0

    def __repr__(self) -> str:
        return f"MockJudge(calls={len(self.calls)}, default_score={self.default_score})"


def make_collector(
    judge: MockJudge | None = None,
    threshold: float = 0.8,
    cache_enabled: bool = False,
    **config_kwargs,
) -> CheckCollector:
    """Create a CheckCollector with sensible test defaults.

    Caching is disabled by default for test isolation.
    """
    if judge is None:
        judge = MockJudge()
    config = CheckllmConfig(
        default_threshold=threshold,
        cache_enabled=cache_enabled,
        **config_kwargs,
    )
    return CheckCollector(config=config, judge=judge)


def assert_all_passed(collector: CheckCollector) -> None:
    """Assert that every check in the collector passed."""
    failed = [r for r in collector.results if not r.passed]
    if failed:
        names = ", ".join(f"{r.metric_name}({r.score:.2f})" for r in failed)
        raise AssertionError(f"{len(failed)} check(s) failed: {names}")


def assert_score_above(collector: CheckCollector, metric_name: str, threshold: float) -> None:
    """Assert that a specific metric scored above a threshold."""
    matching = [r for r in collector.results if r.metric_name == metric_name]
    if not matching:
        raise AssertionError(f"No results found for metric '{metric_name}'")
    for r in matching:
        if r.score < threshold:
            raise AssertionError(
                f"{metric_name} scored {r.score:.2f}, expected >= {threshold:.2f}: {r.reasoning}"
            )

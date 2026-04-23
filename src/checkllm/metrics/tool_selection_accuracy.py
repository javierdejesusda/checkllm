"""Deterministic tool selection accuracy metric.

Evaluates, at each step of an agent trajectory, whether the tool the
agent selected matches the expected tool for that step. Unlike the
LLM-judged :class:`~checkllm.metrics.tool_accuracy.ToolAccuracyMetric`,
this metric is cheap, deterministic, and suitable for CI pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Union

from checkllm.agents import ToolCall, ToolCallTrace
from checkllm.models import CheckResult

_TRACE_LIKE = Union[ToolCall, ToolCallTrace, Mapping[str, Any], str]


def _tool_name(call: _TRACE_LIKE) -> str:
    """Extract the tool name from any accepted representation."""
    if isinstance(call, str):
        return call
    if isinstance(call, ToolCall):
        return call.name
    if isinstance(call, ToolCallTrace):
        return call.tool_name
    if isinstance(call, Mapping):
        return str(call.get("tool_name") or call.get("name") or "")
    raise TypeError(f"Unsupported tool call type: {type(call).__name__}")


class ToolSelectionAccuracyMetric:
    """Step-wise tool selection accuracy.

    Given the sequence of tools actually invoked by an agent and the
    sequence of expected tools, counts how many positions match exactly
    and divides by the length of the expected sequence.

    Extra trailing calls made by the agent are tracked in ``reasoning``
    but do not improve the score. By default a missed step (the agent
    skipped a position) counts as a selection miss; when
    ``allow_equivalents`` is provided, tools listed as equivalents for a
    given expected tool also count as a match.
    """

    metric_name = "tool_selection_accuracy"

    def __init__(
        self,
        threshold: float = 0.8,
        allow_equivalents: Mapping[str, list[str]] | None = None,
        penalize_extras: bool = True,
    ) -> None:
        """Construct the metric.

        Args:
            threshold: Minimum score required for ``passed=True``.
            allow_equivalents: Optional map from an expected tool name to a
                list of names that should be accepted as equivalent at
                that position.
            penalize_extras: When ``True``, any extra actual tool calls
                beyond the expected sequence reduce the final score.
        """
        self.threshold = threshold
        self._equivalents: dict[str, set[str]] = {
            k: {k, *v} for k, v in (allow_equivalents or {}).items()
        }
        self.penalize_extras = penalize_extras

    def _matches(self, expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        group = self._equivalents.get(expected)
        return actual in group if group else False

    def evaluate(
        self,
        actual_calls: list[_TRACE_LIKE],
        expected_sequence: list[str],
    ) -> CheckResult:
        """Score step-wise tool selection accuracy.

        Args:
            actual_calls: Ordered tool calls the agent made.
            expected_sequence: Ordered expected tool names.

        Returns:
            A :class:`CheckResult` describing matches, misses, and extras.
        """
        if not expected_sequence:
            return CheckResult(
                passed=True,
                score=1.0,
                reasoning="No expected tool sequence specified; nothing to validate.",
                cost=0.0,
                latency_ms=0,
                metric_name=self.metric_name,
                threshold=self.threshold,
            )

        actual_names = [_tool_name(c) for c in actual_calls]
        matches = 0
        misses: list[str] = []
        for i, expected in enumerate(expected_sequence):
            actual = actual_names[i] if i < len(actual_names) else None
            if actual is not None and self._matches(expected, actual):
                matches += 1
            else:
                misses.append(f"step {i}: expected '{expected}', got {actual!r}")

        base_score = matches / len(expected_sequence)
        extras = max(0, len(actual_names) - len(expected_sequence))
        score = base_score
        if self.penalize_extras and extras > 0:
            penalty = extras / (len(expected_sequence) + extras)
            score = max(0.0, base_score * (1.0 - penalty))

        passed = (
            score >= self.threshold and not misses and (not self.penalize_extras or extras == 0)
        )
        parts = [f"{matches}/{len(expected_sequence)} selections correct ({base_score:.0%})."]
        if extras > 0:
            parts.append(
                f"{extras} extra trailing tool call(s): {actual_names[len(expected_sequence) :]}."
            )
        if misses:
            parts.append("Misses: " + "; ".join(misses[:6]))
            if len(misses) > 6:
                parts.append(f"(+{len(misses) - 6} more)")

        return CheckResult(
            passed=passed,
            score=round(score, 4),
            reasoning=" ".join(parts),
            cost=0.0,
            latency_ms=0,
            metric_name=self.metric_name,
            threshold=self.threshold,
        )

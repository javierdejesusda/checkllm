"""Tests for :class:`ToolSelectionAccuracyMetric`."""

from __future__ import annotations

from checkllm.agents import ToolCallTrace
from checkllm.metrics.tool_selection_accuracy import ToolSelectionAccuracyMetric


def _trace(idx: int, name: str) -> ToolCallTrace:
    return ToolCallTrace(step_index=idx, tool_name=name)


class TestToolSelectionAccuracyMetric:
    def test_empty_expected_passes(self) -> None:
        metric = ToolSelectionAccuracyMetric()
        result = metric.evaluate(actual_calls=[_trace(0, "a")], expected_sequence=[])
        assert result.passed is True
        assert result.score == 1.0
        assert result.metric_name == "tool_selection_accuracy"

    def test_perfect_match(self) -> None:
        metric = ToolSelectionAccuracyMetric()
        actual = [_trace(0, "search"), _trace(1, "summarize")]
        result = metric.evaluate(actual_calls=actual, expected_sequence=["search", "summarize"])
        assert result.passed is True
        assert result.score == 1.0

    def test_wrong_tool_at_step(self) -> None:
        metric = ToolSelectionAccuracyMetric()
        actual = [_trace(0, "search"), _trace(1, "translate")]
        result = metric.evaluate(actual_calls=actual, expected_sequence=["search", "summarize"])
        assert result.passed is False
        assert result.score == 0.5
        assert "step 1: expected 'summarize'" in result.reasoning

    def test_skipped_step(self) -> None:
        metric = ToolSelectionAccuracyMetric()
        actual = [_trace(0, "search")]
        result = metric.evaluate(actual_calls=actual, expected_sequence=["search", "summarize"])
        assert result.passed is False
        assert result.score == 0.5

    def test_extra_trailing_calls_penalized(self) -> None:
        metric = ToolSelectionAccuracyMetric(threshold=0.95)
        actual = [_trace(0, "search"), _trace(1, "summarize"), _trace(2, "post")]
        result = metric.evaluate(actual_calls=actual, expected_sequence=["search", "summarize"])
        assert result.score < 1.0
        assert result.passed is False
        assert "1 extra trailing tool call" in result.reasoning

    def test_extras_not_penalized_when_disabled(self) -> None:
        metric = ToolSelectionAccuracyMetric(penalize_extras=False)
        actual = [_trace(0, "search"), _trace(1, "summarize"), _trace(2, "post")]
        result = metric.evaluate(actual_calls=actual, expected_sequence=["search", "summarize"])
        assert result.score == 1.0
        assert result.passed is True

    def test_equivalent_tools(self) -> None:
        metric = ToolSelectionAccuracyMetric(allow_equivalents={"search": ["web_search"]})
        actual = [_trace(0, "web_search"), _trace(1, "summarize")]
        result = metric.evaluate(actual_calls=actual, expected_sequence=["search", "summarize"])
        assert result.passed is True
        assert result.score == 1.0

    def test_string_input_supported(self) -> None:
        metric = ToolSelectionAccuracyMetric()
        result = metric.evaluate(
            actual_calls=["search", "summarize"],
            expected_sequence=["search", "summarize"],
        )
        assert result.passed is True

"""Tests for :class:`ToolParameterAccuracyMetric`."""

from __future__ import annotations

from checkllm.agents import ToolCall, ToolCallTrace
from checkllm.metrics.tool_parameter_accuracy import ToolParameterAccuracyMetric


def _trace(idx: int, name: str, **params: object) -> ToolCallTrace:
    return ToolCallTrace(step_index=idx, tool_name=name, parameters=dict(params))


class TestToolParameterAccuracyMetric:
    def test_no_expected_calls_passes(self) -> None:
        metric = ToolParameterAccuracyMetric()
        result = metric.evaluate(actual_calls=[_trace(0, "search", q="hi")], expected_calls=[])
        assert result.passed is True
        assert result.score == 1.0
        assert result.metric_name == "tool_parameter_accuracy"

    def test_perfect_params(self) -> None:
        metric = ToolParameterAccuracyMetric(threshold=0.9)
        actual = [_trace(0, "search", query="weather", limit=5)]
        expected = [
            {
                "tool_name": "search",
                "required": ["query"],
                "schema": {"query": "string", "limit": "integer"},
                "values": {"query": "weather"},
            }
        ]
        result = metric.evaluate(actual_calls=actual, expected_calls=expected)
        assert result.passed is True
        assert result.score == 1.0

    def test_missing_required_param(self) -> None:
        metric = ToolParameterAccuracyMetric()
        actual = [_trace(0, "search", limit=5)]
        expected = [
            {
                "tool_name": "search",
                "required": ["query"],
                "schema": {"query": "string"},
            }
        ]
        result = metric.evaluate(actual_calls=actual, expected_calls=expected)
        assert result.passed is False
        assert result.score < 1.0
        assert "missing required param 'query'" in result.reasoning

    def test_wrong_value(self) -> None:
        metric = ToolParameterAccuracyMetric()
        actual = [_trace(0, "search", query="weather")]
        expected = [
            {
                "tool_name": "search",
                "required": ["query"],
                "values": {"query": "news"},
            }
        ]
        result = metric.evaluate(actual_calls=actual, expected_calls=expected)
        assert result.passed is False
        assert "!= expected 'news'" in result.reasoning

    def test_wrong_type(self) -> None:
        metric = ToolParameterAccuracyMetric()
        actual = [_trace(0, "search", limit="five")]
        expected = [
            {
                "tool_name": "search",
                "schema": {"limit": "integer"},
            }
        ]
        result = metric.evaluate(actual_calls=actual, expected_calls=expected)
        assert result.passed is False
        assert "does not satisfy type" in result.reasoning

    def test_tool_call_input_type_supported(self) -> None:
        metric = ToolParameterAccuracyMetric()
        actual = [ToolCall(name="search", parameters={"query": "hi"})]
        expected = [{"tool_name": "search", "required": ["query"]}]
        result = metric.evaluate(actual_calls=list(actual), expected_calls=expected)
        assert result.passed is True

    def test_missing_call_counts_as_misses(self) -> None:
        metric = ToolParameterAccuracyMetric()
        actual: list[ToolCallTrace] = []
        expected = [
            {
                "tool_name": "search",
                "required": ["query"],
                "schema": {"query": "string"},
                "values": {"query": "weather"},
            }
        ]
        result = metric.evaluate(actual_calls=actual, expected_calls=expected)
        assert result.passed is False
        assert result.score == 0.0
        assert "missing call to 'search'" in result.reasoning

    def test_strict_extras_flag(self) -> None:
        metric = ToolParameterAccuracyMetric(strict_extras=True)
        actual = [_trace(0, "search", query="weather", debug=True)]
        expected = [{"tool_name": "search", "required": ["query"]}]
        result = metric.evaluate(actual_calls=actual, expected_calls=expected)
        assert result.passed is False
        assert "unexpected extra param 'debug'" in result.reasoning

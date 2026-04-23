"""Tests for :class:`TrajectoryMetric`."""

from __future__ import annotations

import pytest

from checkllm.agents import ToolCallTrace
from checkllm.metrics.trajectory_metric import TrajectoryMetric


def _trace(idx: int, name: str, **params: object) -> ToolCallTrace:
    return ToolCallTrace(step_index=idx, tool_name=name, parameters=dict(params))


class TestTrajectoryMetric:
    def test_perfect_trajectory(self) -> None:
        metric = TrajectoryMetric(expected_trajectory=["search", "fetch", "summarize"])
        actual = [
            _trace(0, "search"),
            _trace(1, "fetch"),
            _trace(2, "summarize"),
        ]
        result = metric.evaluate(actual)
        assert result.passed is True
        assert result.score == 1.0
        assert result.metric_name == "trajectory"

    def test_out_of_order_trajectory(self) -> None:
        metric = TrajectoryMetric(expected_trajectory=["search", "fetch", "summarize"])
        actual = [
            _trace(0, "summarize"),
            _trace(1, "fetch"),
            _trace(2, "search"),
        ]
        result = metric.evaluate(actual)
        # Coverage is full but ordering score should drop.
        assert result.score < 1.0
        sub = metric.compute_subscores(actual)
        assert sub.coverage == 1.0
        assert sub.ordering < 1.0

    def test_looping_trajectory(self) -> None:
        metric = TrajectoryMetric(
            expected_trajectory=["search", "summarize"],
            loop_threshold=1,
        )
        actual = [
            _trace(0, "search"),
            _trace(1, "search"),
            _trace(2, "search"),
            _trace(3, "summarize"),
        ]
        result = metric.evaluate(actual)
        sub = metric.compute_subscores(actual)
        assert sub.loops < 1.0
        assert "max consecutive run=3" in result.reasoning
        assert result.passed is False

    def test_extra_tool_calls(self) -> None:
        metric = TrajectoryMetric(expected_trajectory=["search", "summarize"])
        actual = [
            _trace(0, "search"),
            _trace(1, "translate"),
            _trace(2, "summarize"),
        ]
        sub = metric.compute_subscores(actual)
        assert sub.unexpected < 1.0
        assert sub.coverage == 1.0

    def test_missing_tool_calls(self) -> None:
        metric = TrajectoryMetric(expected_trajectory=["search", "fetch", "summarize"])
        actual = [_trace(0, "search")]
        sub = metric.compute_subscores(actual)
        assert sub.coverage == pytest.approx(1 / 3, abs=1e-3)
        assert sub.ordering < 1.0

    def test_wrong_params_do_not_affect_score(self) -> None:
        # Params are not considered by the trajectory metric itself;
        # pair with ToolParameterAccuracyMetric if you need that.
        metric = TrajectoryMetric(expected_trajectory=["search"])
        actual = [_trace(0, "search", q="wrong")]
        result = metric.evaluate(actual)
        assert result.score == 1.0

    def test_string_input(self) -> None:
        metric = TrajectoryMetric(expected_trajectory=["a", "b", "c"])
        result = metric.evaluate(["a", "b", "c"])
        assert result.passed is True

    def test_weights_override(self) -> None:
        metric = TrajectoryMetric(
            expected_trajectory=["a", "b"],
            weights={"ordering": 1.0, "loops": 0.0, "coverage": 0.0, "unexpected": 0.0},
        )
        # Reverse order should now dominate the score.
        assert metric.evaluate(["b", "a"]).score < 1.0

    def test_invalid_loop_threshold(self) -> None:
        with pytest.raises(ValueError):
            TrajectoryMetric(expected_trajectory=["a"], loop_threshold=0)

    def test_invalid_weight_key(self) -> None:
        with pytest.raises(ValueError):
            TrajectoryMetric(expected_trajectory=["a"], weights={"bogus": 1.0})

    def test_negative_weight(self) -> None:
        with pytest.raises(ValueError):
            TrajectoryMetric(expected_trajectory=["a"], weights={"ordering": -1.0})

    def test_all_zero_weights(self) -> None:
        with pytest.raises(ValueError):
            TrajectoryMetric(
                expected_trajectory=["a"],
                weights={"ordering": 0, "loops": 0, "coverage": 0, "unexpected": 0},
            )

    def test_empty_trajectory(self) -> None:
        metric = TrajectoryMetric(expected_trajectory=[])
        result = metric.evaluate([])
        assert result.passed is True
        assert result.score == 1.0

    def test_subscores_as_dict(self) -> None:
        metric = TrajectoryMetric(expected_trajectory=["a"])
        d = metric.compute_subscores([_trace(0, "a")]).as_dict()
        assert set(d) == {"ordering", "loops", "coverage", "unexpected", "overall"}
        assert d["overall"] == 1.0

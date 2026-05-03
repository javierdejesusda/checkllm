"""Tests for the ``TrajectoryMetricConfig`` dataclass (Task A3)."""

import math

import pytest

from checkllm.metrics.trajectory_metric import TrajectoryMetric, TrajectoryMetricConfig


def test_config_with_only_ordering_weight_reduces_to_ordering_subscore():
    # When ordering_weight=1 and all others 0, the overall score must equal the ordering sub-score.
    config = TrajectoryMetricConfig(
        ordering_weight=1.0,
        loop_weight=0.0,
        coverage_weight=0.0,
        unexpected_weight=0.0,
        loop_threshold=2,
    )
    metric = TrajectoryMetric(
        expected_trajectory=["search", "calc", "format"],
        config=config,
    )
    # Actual has one extra (unexpected) and a loop; ordering-only weight should ignore those penalties.
    subs = metric.compute_subscores(["search", "search", "calc", "format"])
    assert math.isclose(subs.overall, subs.ordering, abs_tol=1e-4)


def test_config_with_only_loop_weight_reduces_to_loop_subscore():
    config = TrajectoryMetricConfig(
        ordering_weight=0.0,
        loop_weight=1.0,
        coverage_weight=0.0,
        unexpected_weight=0.0,
    )
    metric = TrajectoryMetric(expected_trajectory=["a"], config=config)
    subs = metric.compute_subscores(["a", "a", "a", "a"])  # a loop of 4
    assert math.isclose(subs.overall, subs.loops, abs_tol=1e-4)


def test_config_default_matches_library_default():
    # The dataclass defaults must equal the library's _DEFAULT_WEIGHTS constants.
    config = TrajectoryMetricConfig()
    assert config.ordering_weight == 0.4
    assert config.loop_weight == 0.2
    assert config.coverage_weight == 0.25
    assert config.unexpected_weight == 0.15
    assert config.loop_threshold == 2


def test_config_normalized_weights_sum_to_one():
    config = TrajectoryMetricConfig(
        ordering_weight=4.0, loop_weight=2.0, coverage_weight=2.5, unexpected_weight=1.5
    )
    normalized = config.normalized_weights()
    assert math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-6)
    # Ratios preserved.
    assert math.isclose(normalized["ordering"] / normalized["loops"], 2.0, abs_tol=1e-6)


def test_config_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="at least one weight"):
        TrajectoryMetricConfig(
            ordering_weight=0.0, loop_weight=0.0, coverage_weight=0.0, unexpected_weight=0.0
        ).normalized_weights()


def test_config_rejects_negative_weight():
    with pytest.raises(ValueError, match="non-negative"):
        TrajectoryMetricConfig(ordering_weight=-0.1).normalized_weights()


def test_config_rejects_invalid_loop_threshold():
    with pytest.raises(ValueError, match="loop_threshold"):
        TrajectoryMetric(expected_trajectory=["a"], config=TrajectoryMetricConfig(loop_threshold=0))


def test_config_construction_rejects_invalid_loop_threshold():
    with pytest.raises(ValueError, match="loop_threshold"):
        TrajectoryMetricConfig(loop_threshold=0)


def test_backward_compat_weights_mapping_still_accepted():
    # Existing callers that pass weights={...} must continue to work unchanged.
    metric = TrajectoryMetric(
        expected_trajectory=["a", "b"],
        weights={"ordering": 1.0, "loops": 0.0, "coverage": 0.0, "unexpected": 0.0},
    )
    subs = metric.compute_subscores(["a", "b"])
    assert math.isclose(subs.overall, subs.ordering, abs_tol=1e-4)


def test_config_and_weights_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot specify both"):
        TrajectoryMetric(
            expected_trajectory=["a"],
            config=TrajectoryMetricConfig(),
            weights={"ordering": 1.0},
        )


def test_config_overrides_loop_threshold():
    config = TrajectoryMetricConfig(loop_threshold=3)
    metric = TrajectoryMetric(expected_trajectory=["a"], config=config)
    assert metric.loop_threshold == 3

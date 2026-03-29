"""Tests for checkllm.experiments — experiment tracking with prompt versioning."""

from __future__ import annotations

import pytest

from checkllm.experiments import (
    ExperimentComparison,
    ExperimentRun,
    ExperimentTracker,
)
from checkllm.models import CheckResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    passed: bool,
    score: float,
    name: str = "test",
    cost: float = 0.001,
) -> CheckResult:
    return CheckResult(
        passed=passed,
        score=score,
        reasoning="ok" if passed else "fail",
        cost=cost,
        latency_ms=50,
        metric_name=name,
    )


def _make_results(pass_scores: list[tuple[bool, float]], name: str = "test") -> list[CheckResult]:
    return [_make_result(p, s, name) for p, s in pass_scores]


# ---------------------------------------------------------------------------
# ExperimentRun
# ---------------------------------------------------------------------------


class TestExperimentRun:
    def test_pass_rate(self):
        run = ExperimentRun(
            run_id="r1",
            experiment_name="exp",
            results=[
                _make_result(True, 0.9),
                _make_result(True, 0.8),
                _make_result(False, 0.3),
            ],
        )
        assert run.pass_rate == pytest.approx(2.0 / 3.0)

    def test_pass_rate_empty(self):
        run = ExperimentRun(run_id="r2", experiment_name="exp")
        assert run.pass_rate == 0.0

    def test_avg_score(self):
        run = ExperimentRun(
            run_id="r3",
            experiment_name="exp",
            results=[
                _make_result(True, 0.9),
                _make_result(True, 0.6),
                _make_result(False, 0.3),
            ],
        )
        assert run.avg_score == pytest.approx((0.9 + 0.6 + 0.3) / 3.0)

    def test_avg_score_empty(self):
        run = ExperimentRun(run_id="r4", experiment_name="exp")
        assert run.avg_score == 0.0

    def test_total_cost(self):
        run = ExperimentRun(
            run_id="r5",
            experiment_name="exp",
            results=[
                _make_result(True, 0.9, cost=0.01),
                _make_result(True, 0.8, cost=0.02),
            ],
        )
        assert run.total_cost == pytest.approx(0.03)

    def test_total_cost_empty(self):
        run = ExperimentRun(run_id="r6", experiment_name="exp")
        assert run.total_cost == 0.0

    def test_fields(self):
        run = ExperimentRun(
            run_id="abc",
            experiment_name="my-exp",
            model="gpt-4o",
            prompt_template="Answer: {{input}}",
            prompt_version="v2",
            tags=["prod", "fast"],
            parameters={"temperature": 0.7},
        )
        assert run.run_id == "abc"
        assert run.experiment_name == "my-exp"
        assert run.model == "gpt-4o"
        assert run.prompt_version == "v2"
        assert "prod" in run.tags
        assert run.parameters["temperature"] == 0.7


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class TestExperimentTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        t = ExperimentTracker(db_path=tmp_path / "test.db")
        yield t
        t.close()

    def test_start_run(self, tracker):
        run = tracker.start_run(
            "my-experiment",
            model="gpt-4o",
            prompt_version="v1",
            tags=["test"],
            parameters={"temperature": 0.5},
        )
        assert run.experiment_name == "my-experiment"
        assert run.model == "gpt-4o"
        assert run.prompt_version == "v1"
        assert "test" in run.tags
        assert run.parameters["temperature"] == 0.5
        assert run.run_id  # UUID should be set

    def test_log_results(self, tracker):
        run = tracker.start_run("exp1")
        results = _make_results([(True, 0.9), (False, 0.4)])
        tracker.log_results(run, results)

        assert len(run.results) == 2
        assert run.results[0].passed is True
        assert run.results[1].passed is False

        # Verify persistence
        reloaded = tracker.get_run(run.run_id)
        assert reloaded is not None
        assert len(reloaded.results) == 2

    def test_log_results_appends(self, tracker):
        run = tracker.start_run("exp1")
        tracker.log_results(run, [_make_result(True, 0.9)])
        tracker.log_results(run, [_make_result(False, 0.3)])
        assert len(run.results) == 2

    def test_end_run(self, tracker):
        run = tracker.start_run("exp1")
        tracker.log_results(run, _make_results([(True, 0.8), (True, 0.9)]))
        tracker.end_run(run)

        assert "ended_at" in run.metadata
        assert run.metadata["pass_rate"] == 1.0
        assert run.metadata["avg_score"] == pytest.approx(0.85)
        assert "total_cost" in run.metadata

    def test_get_run(self, tracker):
        run = tracker.start_run("exp2", model="claude")
        fetched = tracker.get_run(run.run_id)
        assert fetched is not None
        assert fetched.run_id == run.run_id
        assert fetched.model == "claude"

    def test_get_run_not_found(self, tracker):
        assert tracker.get_run("nonexistent-id") is None

    def test_list_runs(self, tracker):
        tracker.start_run("alpha")
        tracker.start_run("alpha")
        tracker.start_run("beta")

        all_runs = tracker.list_runs()
        assert len(all_runs) == 3

        alpha_runs = tracker.list_runs(experiment_name="alpha")
        assert len(alpha_runs) == 2
        assert all(r.experiment_name == "alpha" for r in alpha_runs)

    def test_list_runs_with_tags(self, tracker):
        tracker.start_run("exp", tags=["gpu", "fast"])
        tracker.start_run("exp", tags=["cpu"])
        tracker.start_run("exp", tags=["gpu", "slow"])

        gpu_runs = tracker.list_runs(experiment_name="exp", tags=["gpu"])
        assert len(gpu_runs) == 2

        fast_gpu = tracker.list_runs(experiment_name="exp", tags=["gpu", "fast"])
        assert len(fast_gpu) == 1

    def test_list_runs_limit(self, tracker):
        for _ in range(5):
            tracker.start_run("exp")
        runs = tracker.list_runs(limit=3)
        assert len(runs) == 3

    def test_compare(self, tracker):
        run_a = tracker.start_run("cmp", model="gpt-4o")
        tracker.log_results(
            run_a,
            [
                _make_result(True, 0.8, "relevance", cost=0.01),
                _make_result(False, 0.4, "hallucination", cost=0.01),
            ],
        )
        tracker.end_run(run_a)

        run_b = tracker.start_run("cmp", model="gpt-4o-mini")
        tracker.log_results(
            run_b,
            [
                _make_result(True, 0.9, "relevance", cost=0.005),
                _make_result(True, 0.7, "hallucination", cost=0.005),
            ],
        )
        tracker.end_run(run_b)

        comparison = tracker.compare(run_a.run_id, run_b.run_id)
        assert isinstance(comparison, ExperimentComparison)

        # run_b has higher avg_score
        assert comparison.score_diff > 0
        # run_b has higher pass_rate (1.0 vs 0.5)
        assert comparison.pass_rate_diff > 0
        # run_b is cheaper
        assert comparison.cost_diff < 0

        # Both metrics improved in run_b
        assert "relevance" in comparison.improved_metrics
        assert "hallucination" in comparison.improved_metrics
        assert len(comparison.degraded_metrics) == 0

    def test_compare_not_found(self, tracker):
        run = tracker.start_run("exp")
        with pytest.raises(ValueError, match="Run not found"):
            tracker.compare(run.run_id, "nonexistent")

    def test_best_run(self, tracker):
        run1 = tracker.start_run("best-exp")
        tracker.log_results(run1, [_make_result(True, 0.7)])
        tracker.end_run(run1)

        run2 = tracker.start_run("best-exp")
        tracker.log_results(run2, [_make_result(True, 0.95)])
        tracker.end_run(run2)

        run3 = tracker.start_run("best-exp")
        tracker.log_results(run3, [_make_result(True, 0.5)])
        tracker.end_run(run3)

        best = tracker.best_run("best-exp", metric="avg_score")
        assert best is not None
        assert best.run_id == run2.run_id

    def test_best_run_by_pass_rate(self, tracker):
        run1 = tracker.start_run("pr-exp")
        tracker.log_results(run1, [_make_result(True, 0.9), _make_result(False, 0.3)])
        tracker.end_run(run1)

        run2 = tracker.start_run("pr-exp")
        tracker.log_results(run2, [_make_result(True, 0.8), _make_result(True, 0.7)])
        tracker.end_run(run2)

        best = tracker.best_run("pr-exp", metric="pass_rate")
        assert best is not None
        assert best.run_id == run2.run_id

    def test_best_run_by_cost(self, tracker):
        run1 = tracker.start_run("cost-exp")
        tracker.log_results(run1, [_make_result(True, 0.9, cost=0.05)])
        tracker.end_run(run1)

        run2 = tracker.start_run("cost-exp")
        tracker.log_results(run2, [_make_result(True, 0.9, cost=0.01)])
        tracker.end_run(run2)

        best = tracker.best_run("cost-exp", metric="total_cost")
        assert best is not None
        assert best.run_id == run2.run_id

    def test_best_run_no_runs(self, tracker):
        assert tracker.best_run("nonexistent") is None

    def test_delete_run(self, tracker):
        run = tracker.start_run("del-exp")
        assert tracker.get_run(run.run_id) is not None

        deleted = tracker.delete_run(run.run_id)
        assert deleted is True
        assert tracker.get_run(run.run_id) is None

    def test_delete_run_not_found(self, tracker):
        deleted = tracker.delete_run("nonexistent-id")
        assert deleted is False

    def test_best_run_by_custom_metric(self, tracker):
        run1 = tracker.start_run("custom-exp")
        tracker.log_results(run1, [_make_result(True, 0.6, "fluency")])
        tracker.end_run(run1)

        run2 = tracker.start_run("custom-exp")
        tracker.log_results(run2, [_make_result(True, 0.95, "fluency")])
        tracker.end_run(run2)

        best = tracker.best_run("custom-exp", metric="fluency")
        assert best is not None
        assert best.run_id == run2.run_id

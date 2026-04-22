"""Tests for historical run tracking."""

import pytest

from checkllm.history import RunHistory
from checkllm.models import CheckResult


@pytest.fixture
def history(tmp_path):
    db_path = tmp_path / "history.db"
    h = RunHistory(db_path=db_path)
    yield h
    h.close()


def _make_results():
    return {
        "test_foo": [
            CheckResult(
                passed=True,
                score=0.9,
                reasoning="good",
                cost=0.001,
                latency_ms=100,
                metric_name="hallucination",
            ),
            CheckResult(
                passed=True,
                score=0.85,
                reasoning="ok",
                cost=0.001,
                latency_ms=120,
                metric_name="relevance",
            ),
        ],
        "test_bar": [
            CheckResult(
                passed=False,
                score=0.3,
                reasoning="bad",
                cost=0.002,
                latency_ms=200,
                metric_name="toxicity",
            ),
        ],
    }


class TestRunHistory:
    def test_record_and_list(self, history):
        results = _make_results()
        run_id = history.record_run(results, label="v1")
        assert run_id >= 1

        runs = history.list_runs()
        assert len(runs) == 1
        assert runs[0].run_id == run_id
        assert runs[0].label == "v1"
        assert runs[0].total_checks == 3
        assert runs[0].passed_checks == 2
        assert runs[0].failed_checks == 1
        assert runs[0].total_cost == pytest.approx(0.004)

    def test_get_run(self, history):
        results = _make_results()
        run_id = history.record_run(results, label="test")
        record = history.get_run(run_id)
        assert record is not None
        assert record.label == "test"
        assert "test_foo" in record.results
        assert len(record.results["test_foo"]) == 2

    def test_get_nonexistent_run(self, history):
        assert history.get_run(999) is None

    def test_multiple_runs_ordered(self, history):
        history.record_run(_make_results(), label="first")
        history.record_run(_make_results(), label="second")
        history.record_run(_make_results(), label="third")

        runs = history.list_runs()
        assert len(runs) == 3
        assert runs[0].label == "third"  # newest first
        assert runs[2].label == "first"

    def test_list_limit(self, history):
        for i in range(5):
            history.record_run(_make_results(), label=f"run-{i}")
        runs = history.list_runs(limit=3)
        assert len(runs) == 3

    def test_metric_trend(self, history):
        for i in range(3):
            results = {
                "test_foo": [
                    CheckResult(
                        passed=True,
                        score=0.7 + i * 0.1,
                        reasoning="ok",
                        cost=0.001,
                        latency_ms=100,
                        metric_name="hallucination",
                    ),
                ],
            }
            history.record_run(results, label=f"v{i}")

        trend = history.get_metric_trend("test_foo", "hallucination")
        assert len(trend) == 3
        # Chronological order (oldest first)
        assert trend[0]["score"] == pytest.approx(0.7)
        assert trend[1]["score"] == pytest.approx(0.8)
        assert trend[2]["score"] == pytest.approx(0.9)

    def test_delete_run(self, history):
        run_id = history.record_run(_make_results(), label="deleteme")
        assert history.delete_run(run_id) is True
        assert history.get_run(run_id) is None
        assert history.delete_run(999) is False

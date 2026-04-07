
from checkllm.regression.compare import compare_snapshot
from checkllm.regression.snapshot import (
    MetricRecord,
    Snapshot,
    TestRunRecord,
)


def _make_snapshot(test_name: str, metric_name: str, scores: list[float]) -> Snapshot:
    """Helper to create a snapshot with N runs of a single metric."""
    runs = [
        TestRunRecord(
            metrics={metric_name: MetricRecord(score=s, passed=s >= 0.8)}
        )
        for s in scores
    ]
    return Snapshot(version=1, tests={test_name: runs})


class TestCompareSnapshot:
    def test_detects_regression(self):
        baseline = _make_snapshot("test_a", "hallucination", [0.9, 0.92, 0.88, 0.91, 0.90])
        current = _make_snapshot("test_a", "hallucination", [0.5, 0.52, 0.48, 0.51, 0.50])
        report = compare_snapshot(baseline, current, p_threshold=0.05)
        assert len(report.regressions) == 1
        assert report.regressions[0].test_name == "test_a"
        assert report.regressions[0].metric_name == "hallucination"
        assert report.has_regressions is True

    def test_no_regression_when_stable(self):
        baseline = _make_snapshot("test_a", "hallucination", [0.9, 0.88, 0.91])
        current = _make_snapshot("test_a", "hallucination", [0.89, 0.91, 0.88])
        report = compare_snapshot(baseline, current, p_threshold=0.05)
        assert report.has_regressions is False

    def test_multiple_tests(self):
        baseline = Snapshot(
            version=1,
            tests={
                "test_a": [
                    TestRunRecord(metrics={"h": MetricRecord(score=0.9, passed=True)})
                    for _ in range(5)
                ],
                "test_b": [
                    TestRunRecord(metrics={"h": MetricRecord(score=0.9, passed=True)})
                    for _ in range(5)
                ],
            },
        )
        current = Snapshot(
            version=1,
            tests={
                "test_a": [
                    TestRunRecord(metrics={"h": MetricRecord(score=0.9, passed=True)})
                    for _ in range(5)
                ],
                "test_b": [
                    TestRunRecord(metrics={"h": MetricRecord(score=0.4, passed=False)})
                    for _ in range(5)
                ],
            },
        )
        report = compare_snapshot(baseline, current, p_threshold=0.05)
        assert report.has_regressions is True
        assert len(report.regressions) == 1
        assert report.regressions[0].test_name == "test_b"

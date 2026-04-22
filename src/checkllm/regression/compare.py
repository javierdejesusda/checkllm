from __future__ import annotations

from dataclasses import dataclass, field

from checkllm.regression.snapshot import Snapshot
from checkllm.regression.stats import ComparisonResult, compare_scores


@dataclass
class RegressionItem:
    test_name: str
    metric_name: str
    comparison: ComparisonResult


@dataclass
class RegressionReport:
    regressions: list[RegressionItem] = field(default_factory=list)
    comparisons: list[RegressionItem] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0


def compare_snapshot(
    baseline: Snapshot,
    current: Snapshot,
    p_threshold: float = 0.05,
) -> RegressionReport:
    """Compare current snapshot against baseline and detect regressions."""
    report = RegressionReport()

    all_tests = set(baseline.tests.keys()) | set(current.tests.keys())
    for test_name in sorted(all_tests):
        if test_name not in baseline.tests or test_name not in current.tests:
            continue

        baseline_runs = baseline.tests[test_name]
        current_runs = current.tests[test_name]
        if not baseline_runs or not current_runs:
            continue

        all_metrics = set()
        for run in baseline_runs:
            all_metrics.update(run.metrics.keys())
        for run in current_runs:
            all_metrics.update(run.metrics.keys())

        for metric_name in sorted(all_metrics):
            baseline_scores = baseline.get_scores(test_name, metric_name)
            current_scores = current.get_scores(test_name, metric_name)
            if not baseline_scores or not current_scores:
                continue

            comparison = compare_scores(baseline_scores, current_scores, p_threshold=p_threshold)
            item = RegressionItem(
                test_name=test_name,
                metric_name=metric_name,
                comparison=comparison,
            )
            report.comparisons.append(item)
            if comparison.is_regression:
                report.regressions.append(item)

    return report

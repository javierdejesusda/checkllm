from __future__ import annotations

import statistics
from dataclasses import dataclass

from scipy import stats as scipy_stats


@dataclass
class ConfidenceIntervalResult:
    mean: float
    lower: float
    upper: float
    std_dev: float


@dataclass
class ComparisonResult:
    is_regression: bool
    baseline_mean: float
    current_mean: float
    delta: float
    p_value: float
    baseline_std: float
    current_std: float


def confidence_interval(scores: list[float], confidence: float = 0.95) -> ConfidenceIntervalResult:
    """Calculate mean and confidence interval for a set of scores."""
    if not scores:
        raise ValueError("Cannot compute confidence interval for empty list")
    n = len(scores)
    mean = statistics.mean(scores)
    if n == 1:
        return ConfidenceIntervalResult(mean=mean, lower=mean, upper=mean, std_dev=0.0)
    std_dev = statistics.stdev(scores)
    se = std_dev / (n**0.5)
    t_val = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * se
    return ConfidenceIntervalResult(
        mean=mean,
        lower=mean - margin,
        upper=mean + margin,
        std_dev=std_dev,
    )


def pass_rate(results: list[bool]) -> float:
    """Calculate the pass rate from a list of boolean results."""
    if not results:
        raise ValueError("Cannot compute pass rate for empty list")
    return sum(results) / len(results)


def compare_scores(
    baseline: list[float],
    current: list[float],
    p_threshold: float = 0.05,
) -> ComparisonResult:
    """Compare current scores against baseline using Welch's t-test.

    A regression is detected when the current mean is significantly lower
    than the baseline mean (one-tailed test).
    """
    baseline_mean = statistics.mean(baseline)
    current_mean = statistics.mean(current)
    delta = current_mean - baseline_mean
    baseline_std = statistics.stdev(baseline) if len(baseline) > 1 else 0.0
    current_std = statistics.stdev(current) if len(current) > 1 else 0.0

    if len(baseline) < 2 or len(current) < 2:
        return ComparisonResult(
            is_regression=delta < 0,
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            delta=delta,
            p_value=0.0 if delta < 0 else 1.0,
            baseline_std=baseline_std,
            current_std=current_std,
        )

    t_stat, two_tailed_p = scipy_stats.ttest_ind(baseline, current, equal_var=False)
    one_tailed_p = two_tailed_p / 2 if t_stat > 0 else 1.0 - two_tailed_p / 2

    return ComparisonResult(
        is_regression=bool(delta < 0 and one_tailed_p < p_threshold),
        baseline_mean=baseline_mean,
        current_mean=current_mean,
        delta=float(delta),
        p_value=float(one_tailed_p),
        baseline_std=baseline_std,
        current_std=current_std,
    )

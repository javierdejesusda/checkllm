"""A/B significance testing for CheckLLM runs.

Unlike :mod:`checkllm.regression.stats`, which performs a one-sided
regression check used to gate CI, this module answers the scientific
question "is run B different from run A, and by how much?". It reports:

* Welch's t-test p-value (two-sided, unequal variances),
* Cohen's d effect size (pooled standard deviation),
* Mann-Whitney U p-value (non-parametric, two-sided),
* Percentile bootstrap confidence interval for the mean difference.

Results are packaged as :class:`SignificanceResult`. The convenience
function :func:`analyze_runs` walks two snapshots and produces one result
per shared metric, flagging significance based on the configured alpha.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from scipy import stats as scipy_stats

from checkllm.regression.snapshot import Snapshot

TestMethod = Literal["welch", "mann_whitney", "bootstrap"]


@dataclass
class SignificanceResult:
    """Outcome of a single A/B comparison on one metric."""

    metric: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    delta: float
    p_value: float
    effect_size: float
    ci_low: float
    ci_high: float
    method: TestMethod
    significant: bool
    mann_whitney_p: float = math.nan


def welchs_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sided Welch's t-test.

    Returns ``(t_statistic, p_value)``. Falls back to ``(0.0, 1.0)`` when
    either sample is too small to compute variance.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    t_stat, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d effect size using pooled standard deviation.

    Positive values mean mean(a) > mean(b). Returns ``0.0`` if the pooled
    standard deviation is zero (both samples identical / degenerate).
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_sd


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test.

    Returns ``(U_statistic, p_value)``. Falls back to ``(0.0, 1.0)`` if
    either sample is empty.
    """
    if not a or not b:
        return 0.0, 1.0
    try:
        u_stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        # All identical values make the test degenerate.
        return 0.0, 1.0
    return float(u_stat), float(p)


def bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``mean(a) - mean(b)``.

    Uses independent resampling with replacement on each sample. Falls back
    to point estimates when either sample is empty.
    """
    if not a or not b:
        mean_a = statistics.mean(a) if a else 0.0
        mean_b = statistics.mean(b) if b else 0.0
        delta = mean_a - mean_b
        return delta, delta

    rng = random.Random(seed)
    a_list = list(a)
    b_list = list(b)
    n_a, n_b = len(a_list), len(b_list)
    diffs: list[float] = []
    for _ in range(n_resamples):
        sample_a = [a_list[rng.randrange(n_a)] for _ in range(n_a)]
        sample_b = [b_list[rng.randrange(n_b)] for _ in range(n_b)]
        diffs.append(statistics.mean(sample_a) - statistics.mean(sample_b))
    diffs.sort()
    alpha = 1.0 - confidence
    lower_idx = int(math.floor((alpha / 2) * n_resamples))
    upper_idx = min(n_resamples - 1, int(math.ceil((1 - alpha / 2) * n_resamples)) - 1)
    return diffs[lower_idx], diffs[upper_idx]


def significance_of(
    metric: str,
    a: Sequence[float],
    b: Sequence[float],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int | None = None,
    method: TestMethod = "welch",
) -> SignificanceResult:
    """Full significance analysis for a single metric across two runs.

    The ``method`` argument selects which p-value drives the ``significant``
    flag — Welch's t-test (parametric), Mann-Whitney U (non-parametric), or
    the bootstrap CI (significant iff the CI excludes zero).
    """
    mean_a = statistics.mean(a) if a else 0.0
    mean_b = statistics.mean(b) if b else 0.0
    delta = mean_a - mean_b

    _, welch_p = welchs_t_test(a, b)
    _, mw_p = mann_whitney_u(a, b)
    d = cohens_d(a, b)
    ci_low, ci_high = bootstrap_ci(a, b, n_resamples=n_bootstrap, seed=seed)

    if method == "welch":
        p_value = welch_p
        significant = welch_p < alpha
    elif method == "mann_whitney":
        p_value = mw_p
        significant = mw_p < alpha
    elif method == "bootstrap":
        p_value = welch_p  # Report Welch for information; significance is CI-based.
        significant = not (ci_low <= 0.0 <= ci_high)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown method: {method!r}")

    return SignificanceResult(
        metric=metric,
        n_a=len(a),
        n_b=len(b),
        mean_a=mean_a,
        mean_b=mean_b,
        delta=delta,
        p_value=p_value,
        effect_size=d,
        ci_low=ci_low,
        ci_high=ci_high,
        method=method,
        significant=significant,
        mann_whitney_p=mw_p,
    )


def analyze_runs(
    run_a: Snapshot,
    run_b: Snapshot,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int | None = None,
    method: TestMethod = "welch",
) -> list[SignificanceResult]:
    """Per-metric significance tests for two snapshots.

    Scores are pooled across every shared ``(test, metric)`` cell. Metrics
    that exist only on one side are skipped. Results are sorted by
    descending absolute effect size so the biggest movers surface first.
    """
    results: list[SignificanceResult] = []
    shared_tests = set(run_a.tests) & set(run_b.tests)

    metric_scores_a: dict[str, list[float]] = {}
    metric_scores_b: dict[str, list[float]] = {}

    for test_name in shared_tests:
        for snap, bucket in ((run_a, metric_scores_a), (run_b, metric_scores_b)):
            for run in snap.tests[test_name]:
                for metric_name, record in run.metrics.items():
                    bucket.setdefault(metric_name, []).append(record.score)

    shared_metrics = set(metric_scores_a) & set(metric_scores_b)
    for metric in sorted(shared_metrics):
        a = metric_scores_a[metric]
        b = metric_scores_b[metric]
        results.append(
            significance_of(
                metric,
                a,
                b,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                seed=seed,
                method=method,
            )
        )

    results.sort(key=lambda r: abs(r.effect_size), reverse=True)
    return results

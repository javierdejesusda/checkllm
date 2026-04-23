"""Statistical analysis utilities for CheckLLM runs.

Provides correlation analysis between metrics within a single run and
significance testing for run-vs-run comparisons.
"""

from __future__ import annotations

from checkllm.analysis.correlation import (
    CorrelationMatrix,
    CorrelationResult,
    correlate_metrics,
    correlate_to_pass,
)
from checkllm.analysis.significance import (
    SignificanceResult,
    analyze_runs,
    bootstrap_ci,
    cohens_d,
    mann_whitney_u,
    welchs_t_test,
)

__all__ = [
    "CorrelationMatrix",
    "CorrelationResult",
    "SignificanceResult",
    "analyze_runs",
    "bootstrap_ci",
    "cohens_d",
    "correlate_metrics",
    "correlate_to_pass",
    "mann_whitney_u",
    "welchs_t_test",
]

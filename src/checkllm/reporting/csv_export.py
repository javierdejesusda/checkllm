"""CSV export — flat tabular format for spreadsheets and data pipelines."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from checkllm.models import CheckResult


def results_to_dataframe(
    results: dict[str, list[CheckResult]],
) -> list[dict[str, object]]:
    """Convert results to a list of flat dicts (one per check).

    Useful for constructing a ``pandas.DataFrame``::

        import pandas as pd
        df = pd.DataFrame(results_to_dataframe(results))
    """
    rows: list[dict[str, object]] = []
    for test_name, checks in results.items():
        for c in checks:
            rows.append(
                {
                    "test_name": test_name,
                    "metric_name": c.metric_name,
                    "passed": c.passed,
                    "score": c.score,
                    "reasoning": c.reasoning,
                    "cost": c.cost,
                    "latency_ms": c.latency_ms,
                }
            )
    return rows


_FIELDNAMES = [
    "test_name",
    "metric_name",
    "passed",
    "score",
    "reasoning",
    "cost",
    "latency_ms",
]


def write_csv(
    results: dict[str, list[CheckResult]],
    path: Path,
) -> None:
    """Write results to a CSV file.

    Columns: test_name, metric_name, passed, score, reasoning, cost, latency_ms.
    """
    rows = results_to_dataframe(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_string(
    results: dict[str, list[CheckResult]],
) -> str:
    """Return the CSV content as a string (useful for testing or streaming)."""
    rows = results_to_dataframe(results)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()

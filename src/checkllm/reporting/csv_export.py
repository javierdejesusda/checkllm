"""CSV export -- flat tabular format for spreadsheets and data pipelines."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from checkllm.models import CheckResult


_FIELDNAMES = [
    "test_name",
    "metric_name",
    "passed",
    "score",
    "threshold",
    "reasoning",
    "cost",
    "latency_ms",
    "input_preview",
    "run_id",
    "timestamp_utc",
]


def _row_for_check(
    test_name: str,
    c: CheckResult,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Build a single flat row for one CheckResult.

    Args:
        test_name: Name of the owning test node.
        c: The CheckResult to serialize.
        extra_fields: Optional extra metadata to merge (e.g. ``run_id``,
            ``timestamp_utc``) that is not part of ``CheckResult``.

    Returns:
        A flat dict representing the row. Missing optional columns are
        rendered as ``None`` for backward compatibility with spreadsheet
        tools that expect stable column shapes.
    """
    row: dict[str, object] = {
        "test_name": test_name,
        "metric_name": c.metric_name,
        "passed": c.passed,
        "score": c.score,
        "threshold": c.threshold,
        "reasoning": c.reasoning,
        "cost": c.cost,
        "latency_ms": c.latency_ms,
        "input_preview": c.input_preview,
        "run_id": None,
        "timestamp_utc": None,
    }
    if extra_fields:
        for k, v in extra_fields.items():
            row[k] = v
    return row


def results_to_dataframe(
    results: dict[str, list[CheckResult]],
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    """Convert results to a list of flat dicts (one per check).

    Useful for constructing a ``pandas.DataFrame``::

        import pandas as pd
        df = pd.DataFrame(results_to_dataframe(results))

    Args:
        results: Mapping of test node IDs to lists of CheckResults.
        extra_fields: Optional metadata merged into every row
            (e.g. ``run_id`` or ``timestamp_utc``).

    Returns:
        List of row dicts suitable for CSV or DataFrame consumption.
    """
    rows: list[dict[str, object]] = []
    for test_name, checks in results.items():
        for c in checks:
            rows.append(_row_for_check(test_name, c, extra_fields))
    return rows


def _fieldnames_with_extras(extra_fields: dict[str, Any] | None) -> list[str]:
    """Return base fieldnames plus any extra keys not already present."""
    if not extra_fields:
        return list(_FIELDNAMES)
    fields = list(_FIELDNAMES)
    for key in extra_fields:
        if key not in fields:
            fields.append(key)
    return fields


def write_csv(
    results: dict[str, list[CheckResult]],
    path: Path,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Write results to a CSV file.

    Args:
        results: Mapping of test node IDs to lists of CheckResults.
        path: Destination file path.
        extra_fields: Optional additional metadata columns (e.g.
            ``{"run_id": 42, "timestamp_utc": "2026-04-22T00:00:00Z"}``)
            appended to every row.
    """
    rows = results_to_dataframe(results, extra_fields=extra_fields)
    fieldnames = _fieldnames_with_extras(extra_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_string(
    results: dict[str, list[CheckResult]],
    extra_fields: dict[str, Any] | None = None,
) -> str:
    """Return the CSV content as a string (useful for testing or streaming).

    Args:
        results: Mapping of test node IDs to lists of CheckResults.
        extra_fields: Optional additional metadata columns merged into rows.

    Returns:
        A CSV-formatted string including the header row.
    """
    rows = results_to_dataframe(results, extra_fields=extra_fields)
    fieldnames = _fieldnames_with_extras(extra_fields)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _iter_rows(
    results_iter: Iterable[tuple[str, CheckResult]] | dict[str, list[CheckResult]],
    extra_fields: dict[str, Any] | None = None,
) -> Iterator[dict[str, object]]:
    """Yield row dicts from either a dict-of-lists or a (test_name, check) iter.

    Accepts two shapes to accommodate both in-memory result maps and
    generator-based streaming producers that cannot hold all results in
    memory at once.
    """
    if isinstance(results_iter, dict):
        for test_name, checks in results_iter.items():
            for c in checks:
                yield _row_for_check(test_name, c, extra_fields)
    else:
        for test_name, c in results_iter:
            yield _row_for_check(test_name, c, extra_fields)


def write_csv_streaming(
    results_iter: Iterable[tuple[str, CheckResult]] | dict[str, list[CheckResult]],
    path: Path,
    extra_fields: dict[str, Any] | None = None,
) -> int:
    """Stream CSV rows to ``path`` without loading everything in memory.

    Args:
        results_iter: Either a dict-of-lists (eager) or an iterable of
            ``(test_name, CheckResult)`` tuples (lazy).
        path: Destination file path.
        extra_fields: Optional additional metadata columns merged into rows.

    Returns:
        Number of data rows written (header row excluded).
    """
    fieldnames = _fieldnames_with_extras(extra_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in _iter_rows(results_iter, extra_fields):
            writer.writerow(row)
            count += 1
    return count

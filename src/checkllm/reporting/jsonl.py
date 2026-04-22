"""JSONL export -- one JSON object per line, ideal for data pipelines."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from checkllm.models import CheckResult


def _record_for_check(
    test_name: str,
    c: CheckResult,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-check JSONL record.

    Args:
        test_name: Name of the owning test node.
        c: The CheckResult to serialize.
        extra_fields: Optional extra metadata merged into the record
            (e.g. ``run_id``, ``timestamp_utc``).

    Returns:
        Dict that is one line in the JSONL stream.
    """
    record: dict[str, Any] = {
        "test": test_name,
        "metric": c.metric_name,
        "passed": c.passed,
        "score": c.score,
        "threshold": c.threshold,
        "reasoning": c.reasoning,
        "cost": c.cost,
        "latency_ms": c.latency_ms,
        "input_preview": c.input_preview,
    }
    if extra_fields:
        record.update(extra_fields)
    return record


def export_jsonl(
    results: dict[str, list[CheckResult]],
    output_path: Path | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> str:
    """Export results as JSONL (one JSON object per line).

    Args:
        results: Mapping of test node IDs to lists of CheckResults.
        output_path: Optional destination path; when given, the file is
            created and written in addition to returning the string.
        extra_fields: Optional metadata merged into every record.

    Returns:
        The JSONL string (may be empty).
    """
    lines: list[str] = []
    for test_name, checks in results.items():
        for c in checks:
            record = _record_for_check(test_name, c, extra_fields)
            lines.append(json.dumps(record, ensure_ascii=False))

    text = "\n".join(lines) + "\n" if lines else ""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    return text


def iter_jsonl_records(
    results_iter: Iterable[tuple[str, CheckResult]] | dict[str, list[CheckResult]],
    extra_fields: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield JSONL record dicts from either a dict map or a lazy iterator.

    Accepts both in-memory maps and ``(test_name, CheckResult)`` generators
    so large datasets can be streamed without materializing.
    """
    if isinstance(results_iter, dict):
        for test_name, checks in results_iter.items():
            for c in checks:
                yield _record_for_check(test_name, c, extra_fields)
    else:
        for test_name, c in results_iter:
            yield _record_for_check(test_name, c, extra_fields)


def stream_export_jsonl(
    results_iter: Iterable[tuple[str, CheckResult]] | dict[str, list[CheckResult]],
    output_path: Path,
    extra_fields: dict[str, Any] | None = None,
) -> int:
    """Stream JSONL to ``output_path`` without buffering the full output.

    Args:
        results_iter: Either a dict-of-lists or an iterable of
            ``(test_name, CheckResult)`` pairs.
        output_path: Destination file path.
        extra_fields: Optional metadata merged into every record.

    Returns:
        The number of JSON lines written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for record in iter_jsonl_records(results_iter, extra_fields):
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count

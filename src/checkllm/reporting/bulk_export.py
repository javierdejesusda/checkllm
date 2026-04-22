"""Unified bulk export entry point.

Dispatches a single ``results`` mapping to one of several backends
(CSV, Parquet, JSONL, JSON) based on either an explicit ``format`` or
the destination file's suffix. Returns an :class:`ExportSummary`
describing the written artifact.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from checkllm.models import CheckResult
from checkllm.reporting.csv_export import write_csv, write_csv_streaming
from checkllm.reporting.jsonl import export_jsonl, stream_export_jsonl


Format = Literal["csv", "parquet", "jsonl", "json"]


_SUFFIX_MAP: dict[str, Format] = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "json",
}


class ExportSummary(BaseModel):
    """Summary metadata for a completed bulk export.

    Attributes:
        format: The export format that was written.
        path: Absolute path to the written file.
        row_count: Number of result rows (one per CheckResult) written.
        size_bytes: File size in bytes after writing.
        duration_ms: Wall-clock duration of the write in milliseconds.
    """

    format: Format
    path: str
    row_count: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    duration_ms: int = Field(ge=0)


def _infer_format(path: Path, explicit: Format | None) -> Format:
    """Resolve the export format from an explicit value or path suffix.

    Raises:
        ValueError: If neither an explicit format nor a recognizable
            file suffix is provided.
    """
    if explicit is not None:
        if explicit not in {"csv", "parquet", "jsonl", "json"}:
            raise ValueError(f"Unknown export format: {explicit!r}")
        return explicit
    suffix = path.suffix.lower()
    if suffix not in _SUFFIX_MAP:
        raise ValueError(
            f"Cannot infer export format from path suffix {suffix!r}. "
            "Pass format= explicitly or use one of: "
            f"{sorted(_SUFFIX_MAP)}"
        )
    return _SUFFIX_MAP[suffix]


def _count_rows(results: dict[str, list[CheckResult]]) -> int:
    """Return the total number of CheckResults across all tests."""
    return sum(len(v) for v in results.values())


def _write_json(
    results: dict[str, list[CheckResult]],
    path: Path,
    extra_fields: dict[str, Any] | None,
) -> int:
    """Write the results mapping as a single structured JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "results": {
            test_name: [c.model_dump() for c in checks]
            for test_name, checks in results.items()
        },
    }
    if extra_fields:
        payload["metadata"] = dict(extra_fields)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return _count_rows(results)


def export_results(
    results: dict[str, list[CheckResult]],
    path: Path,
    *,
    format: Format | None = None,
    compression: str = "snappy",
    extra_fields: dict[str, Any] | None = None,
) -> ExportSummary:
    """Export ``results`` to ``path`` in the chosen format.

    Args:
        results: Mapping of test node IDs to lists of CheckResults.
        path: Destination file path.
        format: Explicit format selector. When ``None``, inferred from
            ``path.suffix``.
        compression: Parquet-only compression codec. Ignored for other
            formats.
        extra_fields: Optional metadata merged into rows (CSV/JSONL) or
            attached under a ``metadata`` key (single-document JSON).

    Returns:
        An :class:`ExportSummary` describing the write.

    Raises:
        ValueError: If the format cannot be resolved.
        ImportError: If a Parquet export is requested but pyarrow is
            not installed.
    """
    resolved = _infer_format(path, format)
    path = Path(path)
    started = time.perf_counter()

    if resolved == "csv":
        write_csv(results, path, extra_fields=extra_fields)
        row_count = _count_rows(results)
    elif resolved == "jsonl":
        export_jsonl(results, output_path=path, extra_fields=extra_fields)
        row_count = _count_rows(results)
    elif resolved == "json":
        row_count = _write_json(results, path, extra_fields)
    elif resolved == "parquet":
        # Lazy import so the module stays importable without pyarrow.
        from checkllm.reporting.parquet_export import (  # noqa: PLC0415
            write_parquet,
        )

        run_id = None
        timestamp_utc = None
        if extra_fields:
            run_id = extra_fields.get("run_id")
            timestamp_utc = extra_fields.get("timestamp_utc")
        write_parquet(
            results,
            path,
            compression=compression,
            run_id=run_id,
            timestamp_utc=timestamp_utc,
        )
        row_count = _count_rows(results)
    else:  # pragma: no cover -- guarded by _infer_format
        raise ValueError(f"Unsupported format: {resolved!r}")

    duration_ms = int((time.perf_counter() - started) * 1000)
    size_bytes = path.stat().st_size if path.exists() else 0
    return ExportSummary(
        format=resolved,
        path=str(path.resolve()),
        row_count=row_count,
        size_bytes=size_bytes,
        duration_ms=duration_ms,
    )


def stream_jsonl(
    results_iter: Iterable[tuple[str, CheckResult]] | dict[str, list[CheckResult]],
    path: Path,
    *,
    extra_fields: dict[str, Any] | None = None,
) -> ExportSummary:
    """Stream JSONL to ``path`` without buffering the whole dataset.

    Designed for very large result sets (generator-based producers) where
    an eager ``dict[str, list[CheckResult]]`` would not fit in memory.

    Args:
        results_iter: Either an eager dict-of-lists or an iterable of
            ``(test_name, CheckResult)`` pairs.
        path: Destination file path.
        extra_fields: Optional metadata merged into every record.

    Returns:
        An :class:`ExportSummary` describing the stream.
    """
    path = Path(path)
    started = time.perf_counter()
    row_count = stream_export_jsonl(results_iter, path, extra_fields=extra_fields)
    duration_ms = int((time.perf_counter() - started) * 1000)
    size_bytes = path.stat().st_size if path.exists() else 0
    return ExportSummary(
        format="jsonl",
        path=str(path.resolve()),
        row_count=row_count,
        size_bytes=size_bytes,
        duration_ms=duration_ms,
    )


def stream_csv(
    results_iter: Iterable[tuple[str, CheckResult]] | dict[str, list[CheckResult]],
    path: Path,
    *,
    extra_fields: dict[str, Any] | None = None,
) -> ExportSummary:
    """Stream CSV to ``path`` without buffering the whole dataset.

    Complement to :func:`stream_jsonl` for tabular consumers.
    """
    path = Path(path)
    started = time.perf_counter()
    row_count = write_csv_streaming(results_iter, path, extra_fields=extra_fields)
    duration_ms = int((time.perf_counter() - started) * 1000)
    size_bytes = path.stat().st_size if path.exists() else 0
    return ExportSummary(
        format="csv",
        path=str(path.resolve()),
        row_count=row_count,
        size_bytes=size_bytes,
        duration_ms=duration_ms,
    )

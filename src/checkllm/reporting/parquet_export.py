"""Parquet export -- columnar format for ML pipelines and analytics.

Parquet is the preferred format when exporting checkllm results into data
lakes, feature stores, or notebook analyses (pandas / polars / duckdb).
``pyarrow`` is an optional dependency; importing it is deferred so the
rest of the reporting package works without it installed.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from checkllm.models import CheckResult

if TYPE_CHECKING:  # pragma: no cover -- type-only imports
    import pyarrow as pa


_PARQUET_COLUMNS: tuple[str, ...] = (
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
    "metadata",
)


_INSTALL_HINT = (
    "pyarrow is required for Parquet export. "
    "Install with: pip install checkllm[parquet] "
    "(or: pip install pyarrow)"
)


def _require_pyarrow() -> Any:
    """Import ``pyarrow`` lazily with a clean error if missing.

    Returns:
        The imported ``pyarrow`` module.

    Raises:
        ImportError: If ``pyarrow`` is not installed.
    """
    try:
        import pyarrow  # noqa: PLC0415 -- intentional lazy import
    except ImportError as exc:  # pragma: no cover -- env-dependent
        raise ImportError(_INSTALL_HINT) from exc
    return pyarrow


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _build_column_arrays(
    results: dict[str, list[CheckResult]],
    run_id: int | str | None,
    timestamp_utc: str | None,
) -> dict[str, list[Any]]:
    """Build per-column lists in the canonical order.

    Args:
        results: Mapping of test node IDs to lists of CheckResults.
        run_id: Optional run identifier to stamp into every row.
        timestamp_utc: Optional ISO 8601 UTC timestamp to stamp into
            every row. When ``None``, the current UTC time is recorded.

    Returns:
        Dict keyed by column name with parallel value lists.
    """
    ts = timestamp_utc if timestamp_utc is not None else _now_iso()
    cols: dict[str, list[Any]] = {name: [] for name in _PARQUET_COLUMNS}
    for test_name, checks in results.items():
        for c in checks:
            cols["test_name"].append(test_name)
            cols["metric_name"].append(c.metric_name)
            cols["passed"].append(bool(c.passed))
            cols["score"].append(float(c.score))
            cols["threshold"].append(None if c.threshold is None else float(c.threshold))
            cols["reasoning"].append(c.reasoning)
            cols["cost"].append(float(c.cost))
            cols["latency_ms"].append(int(c.latency_ms))
            cols["input_preview"].append(c.input_preview)
            cols["run_id"].append(None if run_id is None else str(run_id))
            cols["timestamp_utc"].append(ts)
            extras = c.model_dump(
                exclude={
                    "passed",
                    "score",
                    "reasoning",
                    "cost",
                    "latency_ms",
                    "metric_name",
                    "threshold",
                    "input_preview",
                },
            )
            cols["metadata"].append(json.dumps(extras, ensure_ascii=False, default=str))
    return cols


def results_to_arrow_table(
    results: dict[str, list[CheckResult]],
    *,
    run_id: int | str | None = None,
    timestamp_utc: str | None = None,
) -> pa.Table:
    """Build a ``pyarrow.Table`` from a results mapping.

    Args:
        results: Mapping of test node IDs to lists of CheckResults.
        run_id: Optional run identifier stamped into every row.
        timestamp_utc: Optional ISO 8601 UTC timestamp stamped into
            every row (defaults to ``datetime.now(UTC)``).

    Returns:
        A ``pyarrow.Table`` whose schema matches the canonical checkllm
        Parquet column order, including empty tables when ``results`` is
        empty.

    Raises:
        ImportError: If ``pyarrow`` is not installed.
    """
    pa = _require_pyarrow()
    cols = _build_column_arrays(results, run_id=run_id, timestamp_utc=timestamp_utc)

    schema = pa.schema(
        [
            ("test_name", pa.string()),
            ("metric_name", pa.string()),
            ("passed", pa.bool_()),
            ("score", pa.float64()),
            ("threshold", pa.float64()),
            ("reasoning", pa.string()),
            ("cost", pa.float64()),
            ("latency_ms", pa.int64()),
            ("input_preview", pa.string()),
            ("run_id", pa.string()),
            ("timestamp_utc", pa.string()),
            ("metadata", pa.string()),
        ]
    )
    arrays = [pa.array(cols[name], type=schema.field(name).type) for name in _PARQUET_COLUMNS]
    return pa.Table.from_arrays(arrays, schema=schema)


def write_parquet(
    results: dict[str, list[CheckResult]],
    path: Path,
    *,
    compression: str = "snappy",
    run_id: int | str | None = None,
    timestamp_utc: str | None = None,
) -> None:
    """Write results to a Parquet file.

    Args:
        results: Mapping of test node IDs to lists of CheckResults.
        path: Destination file path.
        compression: Parquet compression codec; one of ``"snappy"``,
            ``"gzip"``, ``"brotli"``, ``"zstd"``, ``"lz4"``, or
            ``"none"``. Defaults to ``"snappy"``.
        run_id: Optional run identifier stamped into every row.
        timestamp_utc: Optional ISO 8601 UTC timestamp stamped into
            every row.

    Raises:
        ImportError: If ``pyarrow`` is not installed.
    """
    pa = _require_pyarrow()
    import pyarrow.parquet as pq  # noqa: PLC0415 -- lazy import

    _ = pa  # referenced for side-effect of verifying install
    table = results_to_arrow_table(results, run_id=run_id, timestamp_utc=timestamp_utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = None if compression in {"none", "uncompressed"} else compression
    pq.write_table(table, str(path), compression=normalized)  # type: ignore[no-untyped-call, unused-ignore]


def read_parquet(path: Path) -> dict[str, list[CheckResult]]:
    """Read a checkllm-written Parquet file back into a results mapping.

    Supports round-trip comparisons against a previous regression run.
    ``run_id`` and ``timestamp_utc`` columns are not preserved on
    ``CheckResult`` (they are stored on ``RunHistory`` instead).

    Args:
        path: Path to a Parquet file previously written by
            :func:`write_parquet` (or any file with the same schema).

    Returns:
        A dict mapping ``test_name`` to a list of ``CheckResult``
        instances, preserving row order within each test.

    Raises:
        ImportError: If ``pyarrow`` is not installed.
    """
    _require_pyarrow()
    import pyarrow.parquet as pq  # noqa: PLC0415 -- lazy import

    table = pq.read_table(str(path))  # type: ignore[no-untyped-call, unused-ignore]
    rows = table.to_pylist()
    out: dict[str, list[CheckResult]] = {}
    for row in rows:
        test_name = row.get("test_name", "")
        result = CheckResult(
            passed=bool(row.get("passed", False)),
            score=float(row.get("score") or 0.0),
            reasoning=row.get("reasoning") or "",
            cost=float(row.get("cost") or 0.0),
            latency_ms=int(row.get("latency_ms") or 0),
            metric_name=row.get("metric_name") or "",
            threshold=row.get("threshold"),
            input_preview=row.get("input_preview"),
        )
        out.setdefault(test_name, []).append(result)
    return out

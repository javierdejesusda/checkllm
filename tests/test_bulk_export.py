"""Tests for bulk_export: format dispatch, streaming, summaries."""

from __future__ import annotations

import json

import pytest

from checkllm.models import CheckResult
from checkllm.reporting.bulk_export import (
    ExportSummary,
    export_results,
    stream_csv,
    stream_jsonl,
)


def _r(metric: str = "relevance", score: float = 0.8) -> CheckResult:
    return CheckResult(
        passed=True,
        score=score,
        reasoning="ok",
        cost=0.0,
        latency_ms=1,
        metric_name=metric,
    )


def _simple_results() -> dict[str, list[CheckResult]]:
    return {
        "t1": [_r("relevance"), _r("toxicity", score=0.1)],
        "t2": [_r("rubric", score=0.9)],
    }


def test_format_dispatch_by_csv_suffix(tmp_path):
    path = tmp_path / "out.csv"
    summary = export_results(_simple_results(), path)
    assert isinstance(summary, ExportSummary)
    assert summary.format == "csv"
    assert summary.row_count == 3
    assert path.exists() and path.read_text(encoding="utf-8").startswith("test_name,")


def test_format_dispatch_by_jsonl_suffix(tmp_path):
    path = tmp_path / "out.jsonl"
    summary = export_results(_simple_results(), path)
    assert summary.format == "jsonl"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 3
    # Each line must parse as JSON.
    for line in lines:
        json.loads(line)


def test_format_dispatch_by_json_suffix(tmp_path):
    path = tmp_path / "out.json"
    summary = export_results(_simple_results(), path, extra_fields={"run_id": 7})
    assert summary.format == "json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "results" in payload
    assert payload["metadata"]["run_id"] == 7


def test_format_dispatch_by_parquet_suffix(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "out.parquet"
    summary = export_results(_simple_results(), path)
    assert summary.format == "parquet"
    assert path.exists() and path.stat().st_size > 0


def test_explicit_format_overrides_suffix(tmp_path):
    path = tmp_path / "out.bin"
    summary = export_results(_simple_results(), path, format="csv")
    assert summary.format == "csv"


def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError):
        export_results(_simple_results(), tmp_path / "out.csv", format="xml")  # type: ignore[arg-type]


def test_unknown_suffix_raises(tmp_path):
    with pytest.raises(ValueError):
        export_results(_simple_results(), tmp_path / "out.bin")


def test_export_summary_has_expected_fields(tmp_path):
    path = tmp_path / "out.csv"
    summary = export_results(_simple_results(), path)
    assert summary.row_count == 3
    assert summary.size_bytes > 0
    assert summary.duration_ms >= 0
    # path is absolute.
    from pathlib import Path as _P

    assert _P(summary.path).is_absolute()


def test_stream_jsonl_handles_large_generator(tmp_path):
    # 10k synthetic rows from a generator -- memory-bounded writer.
    def gen():
        for i in range(10_000):
            yield (f"test_{i % 50}", _r("relevance", score=(i % 100) / 100))

    path = tmp_path / "stream.jsonl"
    summary = stream_jsonl(gen(), path)
    assert summary.row_count == 10_000
    assert summary.format == "jsonl"
    # Sanity: line count matches row count.
    line_count = 0
    with path.open("r", encoding="utf-8") as fh:
        for _ in fh:
            line_count += 1
    assert line_count == 10_000


def test_stream_csv_handles_generator(tmp_path):
    def gen():
        for i in range(1000):
            yield (f"t_{i % 10}", _r("rubric", score=0.5))

    path = tmp_path / "stream.csv"
    summary = stream_csv(gen(), path)
    assert summary.row_count == 1000
    assert summary.format == "csv"
    # Header + 1000 data rows.
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == 1001


def test_extra_fields_flow_through_csv(tmp_path):
    path = tmp_path / "out.csv"
    export_results(
        _simple_results(),
        path,
        extra_fields={"run_id": 9, "timestamp_utc": "2026-04-22T00:00:00Z"},
    )
    text = path.read_text(encoding="utf-8")
    assert "run_id" in text
    assert "2026-04-22T00:00:00Z" in text


def test_extra_fields_flow_through_jsonl(tmp_path):
    path = tmp_path / "out.jsonl"
    export_results(
        _simple_results(),
        path,
        extra_fields={"run_id": 9},
    )
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    assert record["run_id"] == 9

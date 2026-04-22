"""Tests for parquet_export: schema, round-trip, edge cases."""

from __future__ import annotations

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402 -- after importorskip

from checkllm.models import CheckResult  # noqa: E402
from checkllm.reporting.parquet_export import (  # noqa: E402
    _PARQUET_COLUMNS,
    read_parquet,
    results_to_arrow_table,
    write_parquet,
)


def _make_result(
    metric: str = "relevance",
    score: float = 0.8,
    passed: bool = True,
    threshold: float | None = 0.7,
    input_preview: str | None = "hello",
) -> CheckResult:
    return CheckResult(
        passed=passed,
        score=score,
        reasoning=f"score={score}",
        cost=0.001,
        latency_ms=12,
        metric_name=metric,
        threshold=threshold,
        input_preview=input_preview,
    )


def test_arrow_table_has_canonical_columns():
    results = {"test_1": [_make_result()]}
    table = results_to_arrow_table(results)
    assert list(table.schema.names) == list(_PARQUET_COLUMNS)


def test_arrow_table_row_count_matches_input():
    results = {
        "t1": [_make_result("relevance"), _make_result("toxicity", score=0.1)],
        "t2": [_make_result("rubric", score=0.9)],
    }
    table = results_to_arrow_table(results)
    assert table.num_rows == 3


def test_arrow_table_empty_results_yields_empty_table():
    table = results_to_arrow_table({})
    assert table.num_rows == 0
    assert list(table.schema.names) == list(_PARQUET_COLUMNS)


def test_roundtrip_write_then_read(tmp_path):
    results = {
        "tests/api.py::test_echo": [
            _make_result("relevance", score=0.82),
            _make_result("toxicity", score=0.05, passed=True, threshold=0.2),
        ],
    }
    path = tmp_path / "out.parquet"
    write_parquet(results, path)
    assert path.exists() and path.stat().st_size > 0

    loaded = read_parquet(path)
    assert set(loaded.keys()) == {"tests/api.py::test_echo"}
    checks = loaded["tests/api.py::test_echo"]
    assert len(checks) == 2
    metrics = {c.metric_name for c in checks}
    assert metrics == {"relevance", "toxicity"}


def test_roundtrip_preserves_scalar_fields(tmp_path):
    results = {"t": [_make_result("rubric", score=0.77, threshold=0.5)]}
    path = tmp_path / "out.parquet"
    write_parquet(results, path)
    back = read_parquet(path)
    r = back["t"][0]
    assert r.score == pytest.approx(0.77, abs=1e-6)
    assert r.threshold == pytest.approx(0.5, abs=1e-6)
    assert r.metric_name == "rubric"
    assert r.input_preview == "hello"


def test_none_threshold_roundtrips(tmp_path):
    results = {"t": [_make_result("m", threshold=None, input_preview=None)]}
    path = tmp_path / "out.parquet"
    write_parquet(results, path)
    back = read_parquet(path)
    r = back["t"][0]
    assert r.threshold is None
    assert r.input_preview is None


def test_compression_options_accepted(tmp_path):
    results = {"t": [_make_result()]}
    for codec in ("snappy", "gzip", "zstd", "none"):
        path = tmp_path / f"out_{codec}.parquet"
        write_parquet(results, path, compression=codec)
        assert path.exists() and path.stat().st_size > 0


def test_run_id_and_timestamp_are_stamped(tmp_path):
    results = {"t": [_make_result()]}
    path = tmp_path / "out.parquet"
    write_parquet(
        results, path, run_id=42, timestamp_utc="2026-04-22T00:00:00+00:00"
    )
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    run_ids = table.column("run_id").to_pylist()
    timestamps = table.column("timestamp_utc").to_pylist()
    assert run_ids == ["42"]
    assert timestamps == ["2026-04-22T00:00:00+00:00"]


def test_metadata_column_is_json_string():
    results = {"t": [_make_result()]}
    table = results_to_arrow_table(results)
    meta = table.column("metadata").to_pylist()
    assert isinstance(meta[0], str)
    # Must be valid JSON.
    import json as _json

    assert isinstance(_json.loads(meta[0]), dict)


def test_arrow_types_are_as_expected():
    results = {"t": [_make_result()]}
    table = results_to_arrow_table(results)
    field = {f.name: f.type for f in table.schema}
    assert field["score"] == pa.float64()
    assert field["passed"] == pa.bool_()
    assert field["latency_ms"] == pa.int64()
    assert field["test_name"] == pa.string()

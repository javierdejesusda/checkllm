# Exporting results

checkllm stores every eval result as a structured `CheckResult`, and the
`checkllm.reporting` package turns those results into the file formats
downstream tools expect: CSV for spreadsheets, Parquet for ML pipelines,
JSONL for streaming consumers, and a single-document JSON for quick
inspection. This guide shows how to pick and use each format.

## Why bulk export matters

Eval runs generate a lot of data. A single test suite can produce
thousands of `CheckResult` rows across dozens of metrics, and those
rows are where all the useful information lives: pass/fail state,
per-metric scores, thresholds, costs, latencies, reasoning strings
from judges, and the input preview that triggered the check. To do
anything useful with that data past the terminal summary, you need to
get it out of memory and into a file your analytics stack can load.

The `checkllm.reporting.bulk_export` module is the one-stop entry
point. It dispatches to the right backend based on the file suffix
(or an explicit `format=` argument), stamps optional metadata such as
`run_id` and `timestamp_utc` into every row, and returns an
`ExportSummary` describing what it wrote.

```python
from pathlib import Path

from checkllm.reporting.bulk_export import export_results

summary = export_results(results, Path("out/run-42.parquet"))
print(summary.row_count, summary.size_bytes, summary.duration_ms)
```

## CSV (default)

CSV is the safest lowest-common-denominator format. Spreadsheets open
it, every database can import it, and `pandas.read_csv` handles it in
one line. checkllm writes a stable column order so older tooling
never breaks when new columns are added.

```python
from pathlib import Path

from checkllm.reporting.csv_export import write_csv

write_csv(results, Path("out/run.csv"))
```

Columns: `test_name, metric_name, passed, score, threshold, reasoning,
cost, latency_ms, input_preview, run_id, timestamp_utc`.

The optional `extra_fields=` argument appends additional metadata
columns to every row, useful for joining multiple runs into a single
CSV:

```python
write_csv(
    results,
    Path("out/run.csv"),
    extra_fields={"run_id": 42, "timestamp_utc": "2026-04-22T00:00:00Z"},
)
```

For very large suites, use `write_csv_streaming(results_iter, path)`.
It accepts either the usual `dict[str, list[CheckResult]]` or a lazy
iterator of `(test_name, CheckResult)` tuples, so the full result set
never needs to be held in memory.

## Parquet (ML pipelines)

Parquet is the best choice when results flow into a data lake, a
feature store, or a notebook-based analysis. It is columnar,
compressed, and ~10x smaller than CSV for the same data.

```python
from pathlib import Path

from checkllm.reporting.parquet_export import write_parquet

write_parquet(results, Path("out/run.parquet"), compression="zstd")
```

`pyarrow` is an optional dependency. If it is not installed, the
function raises a clean `ImportError` with the install hint
`pip install checkllm[parquet]`.

Read a Parquet file back into a `dict[str, list[CheckResult]]` for
regression comparisons:

```python
from checkllm.reporting.parquet_export import read_parquet

baseline = read_parquet(Path("out/baseline.parquet"))
```

Supported `compression` codecs: `snappy` (default), `gzip`, `brotli`,
`zstd`, `lz4`, and `none`.

## JSONL (streaming)

JSONL (one JSON object per line) is the right format when downstream
consumers stream records line-by-line: a log shipper, a message
queue, a serverless loader. The `stream_jsonl` helper writes rows
directly from a generator so memory stays bounded even for very large
result sets.

```python
from pathlib import Path

from checkllm.reporting.bulk_export import stream_jsonl


def produce_results():
    for test_name, check in expensive_eval_iterator():
        yield (test_name, check)


summary = stream_jsonl(produce_results(), Path("out/stream.jsonl"))
print(summary.row_count)
```

The eager API is also available:

```python
from checkllm.reporting.jsonl import export_jsonl

export_jsonl(results, Path("out/run.jsonl"))
```

## JSON (single-file structured)

For configuration-style consumers that want one file they can load
whole, use the single-document JSON format. The payload is
`{"results": {test_name: [check_dict, ...]}}` with an optional
`metadata` block.

```python
from pathlib import Path

from checkllm.reporting.bulk_export import export_results

export_results(
    results,
    Path("out/run.json"),
    extra_fields={"run_id": 42, "git_commit": "abc123"},
)
```

## Comparing two runs in the dashboard

The dashboard ships a side-by-side comparison view for any two runs
stored in `RunHistory`. Start the dashboard and navigate to:

```
http://localhost:8484/compare?a=<snapshot_a_id>&b=<snapshot_b_id>
```

The HTML page shows a summary panel (metrics compared, improved,
regressed, unchanged, average delta) and a per-metric delta table.
Rows are color-coded: green for improvements, red for regressions,
neutral for unchanged metrics. Direction is shown as `+`, `-`, or
`=`.

Under the hood the page is backed by
`build_comparison(history, a, b)`, which returns a `ComparisonView`
suitable for programmatic use:

```python
from checkllm.dashboard import build_comparison
from checkllm.history import RunHistory

history = RunHistory()
view = build_comparison(history, snapshot_a_id=41, snapshot_b_id=42)
print(view.improved)
print(view.regressed)
print(view.metrics_diff)
```

The JSON endpoint is available at `POST /api/compare` with either
query parameters (`?a=<id>&b=<id>`) or a JSON body
(`{"a": <id>, "b": <id>}`). The response is a serialized
`ComparisonView` plus a `summary` block.

## Integrating with pandas / polars / duckdb

All four formats load into the major analytics libraries in a single
line.

pandas:

```python
import pandas as pd

df = pd.read_parquet("out/run.parquet")
df = pd.read_csv("out/run.csv")
df = pd.read_json("out/run.jsonl", lines=True)
```

polars (fastest for Parquet):

```python
import polars as pl

df = pl.read_parquet("out/run.parquet")
df = pl.read_csv("out/run.csv")
df = pl.read_ndjson("out/run.jsonl")
```

duckdb (no load, queries directly against the file):

```python
import duckdb

duckdb.sql("SELECT metric_name, AVG(score) FROM 'out/run.parquet' GROUP BY 1")
duckdb.sql("SELECT * FROM read_csv_auto('out/run.csv') WHERE passed = false")
duckdb.sql("SELECT * FROM read_json_auto('out/run.jsonl') WHERE score < 0.5")
```

## CI regression gates

Export on every CI run with a deterministic filename, then diff the
most recent against the previous artifact to fail the build on
regressions.

```yaml
- name: Run evals
  run: checkllm run --export out/run-${{ github.sha }}.parquet

- name: Compare against baseline
  run: |
    python - <<'PY'
    from checkllm.reporting.parquet_export import read_parquet
    from checkllm.dashboard import build_comparison_view

    new = read_parquet("out/run-${{ github.sha }}.parquet")
    base = read_parquet("out/baseline.parquet")
    view = build_comparison_view(base, new, label_a="baseline", label_b="pr")
    regressions = view.regressed
    if regressions:
        print("Regressed metrics:", regressions)
        raise SystemExit(1)
    PY
```

Because `read_parquet` is a pure Python function returning the same
`dict[str, list[CheckResult]]` shape as a live run, the same
comparison utilities work equally well against in-memory results and
stored artifacts. Keep the most recent "known good" Parquet file in
the repo or object storage and every PR becomes a zero-extra-work
regression test.

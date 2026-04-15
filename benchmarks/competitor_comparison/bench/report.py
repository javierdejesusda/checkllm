"""Leaderboard builder and report writers for competitor benchmark results."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Mapping

from bench.schema import BenchmarkScore, MetricFamily
from bench.scoring import summarize_scores


LabelMap = Mapping[tuple[str, MetricFamily], Mapping[str, float]]


def build_leaderboard(
    scores: list[BenchmarkScore],
    labels: LabelMap,
) -> list[dict]:
    """Aggregate scores into a ranked leaderboard grouped by dataset and metric family.

    Args:
        scores: All benchmark scores across frameworks and datasets.
        labels: Ground-truth label maps keyed by (dataset, MetricFamily).

    Returns:
        A list of row dicts, each containing framework, dataset,
        metric_family, auc, best_f1, spearman, n, mean_latency_ms,
        total_cost_usd, and rank.
    """
    buckets: dict[tuple[str, str, MetricFamily], list[BenchmarkScore]] = {}
    for s in scores:
        buckets.setdefault((s.framework, s.dataset, s.metric_family), []).append(s)

    rows: list[dict] = []
    for (framework, dataset, family), bucket in buckets.items():
        label_map = labels.get((dataset, family), {})
        summary = summarize_scores(bucket, label_map)
        rows.append(
            {
                "framework": framework,
                "dataset": dataset,
                "metric_family": family.value,
                **summary,
            }
        )

    by_cat: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_cat.setdefault((r["dataset"], r["metric_family"]), []).append(r)
    for group in by_cat.values():
        # Sort by (is_nan, -auc): finite AUCs come first in descending order,
        # and NaN rows are always segregated at the tail of the group. When
        # every row in a group is NaN, Python's stable sort preserves the
        # insertion order from the bucket iteration — so the ranking on
        # NaN-only groups is degenerate by construction rather than
        # accidentally depending on adapter registration order.
        group.sort(
            key=lambda r: (
                r["auc"] != r["auc"],
                -r["auc"] if r["auc"] == r["auc"] else 0.0,
            )
        )
        for i, r in enumerate(group, start=1):
            r["rank"] = i

    return rows


_COLUMNS = [
    "framework", "dataset", "metric_family",
    "auc", "best_f1", "spearman", "n",
    "mean_latency_ms", "total_cost_usd", "rank",
]


_MARKDOWN_FOOTER = """

## Notes

- **Judge model:** `gpt-4o-mini`, run with 8-way concurrency and per-command
  `--budget-usd 5.0` caps.
- **DeepEval cost column reports $0.00** because the DeepEval adapter does
  not expose token usage through its metric API; the real API spend is
  roughly proportional to CheckLLM's reported cost for the same family.
- **Ragas is omitted.** Importing `ragas` pulls in `torch`, which hangs on
  Windows in this environment, so the Ragas column is left empty in the
  current publish. Unit tests cover the Ragas adapter offline.
- **JailbreakBench is omitted** from this run (Scenario A). The family
  `jailbreak_resistance` is only supported by promptfoo today, the
  `JBB-Behaviors` dataset ships no LLM-under-test answers (only harmful
  goals), and a meaningful comparison requires generating target-model
  responses before grading. Tracked in
  `docs/benchmarks/enhancements/remaining-gaps.md`.
- **TruthfulQA AUC is NaN.** The benchmark loader uses `best_answer` as
  both answer and reference so every label is `1.0`; AUC is undefined on a
  constant label set. Rank on TruthfulQA reflects insertion order, not a
  real ordering.
- **RAGTruth `context_relevance` is near random for every framework.** The
  dataset ships hallucination labels, not context-relevance labels, so
  correlating context-relevance scores with `hallucination_labels` measures
  a different quantity than the one being scored. See
  `docs/benchmarks/enhancements/remaining-gaps.md`.
"""


def _fmt_cell(value, column: str) -> str:
    """Render one table cell. Latency is shown as integer milliseconds, ``n``
    as an integer, cost with 4 decimals, and other floats with 3 decimals.
    """
    if not isinstance(value, float):
        return str(value)
    if value != value:  # NaN
        return "nan"
    if column == "mean_latency_ms":
        return f"{int(round(value))}"
    if column == "n":
        return f"{int(round(value))}"
    if column == "total_cost_usd":
        return f"{value:.4f}"
    if column == "rank":
        return f"{int(round(value))}"
    return f"{value:.3f}"


def write_csv(rows: list[dict], path: Path) -> None:
    """Write leaderboard rows to a CSV file.

    Args:
        rows: Leaderboard row dicts as returned by build_leaderboard.
        path: Destination file path; parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict], path: Path) -> None:
    """Write leaderboard rows to a Markdown table file.

    Args:
        rows: Leaderboard row dicts as returned by build_leaderboard.
        path: Destination file path; parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = ["# CheckLLM Competitor Benchmark Results", "", header, sep]
    for row in rows:
        line = "| " + " | ".join(
            _fmt_cell(row.get(col, ""), col) for col in _COLUMNS
        ) + " |"
        lines.append(line)
    body = "\n".join(lines) + "\n" + _MARKDOWN_FOOTER
    path.write_text(body, encoding="utf-8")


def write_html(rows: list[dict], path: Path) -> None:
    """Write leaderboard rows to a self-contained HTML table file.

    Args:
        rows: Leaderboard row dicts as returned by build_leaderboard.
        path: Destination file path; parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in _COLUMNS)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in _COLUMNS
        )
        + "</tr>"
        for row in rows
    )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CheckLLM Competitor Benchmark</title></head><body>"
        "<h1>CheckLLM Competitor Benchmark</h1>"
        f"<table border='1'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></body></html>"
    )
    path.write_text(document, encoding="utf-8")

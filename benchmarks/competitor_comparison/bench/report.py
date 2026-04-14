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
        group.sort(key=lambda r: (-(r["auc"] if r["auc"] == r["auc"] else -1)))
        for i, r in enumerate(group, start=1):
            r["rank"] = i

    return rows


_COLUMNS = [
    "framework", "dataset", "metric_family",
    "auc", "best_f1", "spearman", "n",
    "mean_latency_ms", "total_cost_usd", "rank",
]


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
            f"{row.get(col, ''):.3f}"
            if isinstance(row.get(col), float)
            else str(row.get(col, ""))
            for col in _COLUMNS
        ) + " |"
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CheckLLM Competitor Benchmark</title></head><body>"
        "<h1>CheckLLM Competitor Benchmark</h1>"
        f"<table border='1'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></body></html>"
    )
    path.write_text(html, encoding="utf-8")

"""A/B comparison reports — compare two sets of check results side-by-side."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from checkllm.models import CheckResult


@dataclass
class ComparisonReport:
    """Container for an A/B comparison between two result sets."""

    results_a: dict[str, list[CheckResult]]
    results_b: dict[str, list[CheckResult]]
    label_a: str = "A"
    label_b: str = "B"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _all_checks(results: dict[str, list[CheckResult]]) -> list[CheckResult]:
    return [c for checks in results.values() for c in checks]


def _summary_stats(results: dict[str, list[CheckResult]]) -> dict[str, float]:
    all_c = _all_checks(results)
    total = len(all_c)
    passed = sum(1 for c in all_c if c.passed)
    avg_score = sum(c.score for c in all_c) / total if total else 0.0
    total_cost = sum(c.cost for c in all_c)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "avg_score": avg_score,
        "total_cost": total_cost,
    }


def _paired_rows(
    report: ComparisonReport,
) -> list[dict[str, object]]:
    """Build paired rows keyed by (test_name, metric_name)."""
    lookup_a: dict[tuple[str, str], CheckResult] = {}
    for test, checks in report.results_a.items():
        for c in checks:
            lookup_a[(test, c.metric_name)] = c

    lookup_b: dict[tuple[str, str], CheckResult] = {}
    for test, checks in report.results_b.items():
        for c in checks:
            lookup_b[(test, c.metric_name)] = c

    all_keys = sorted(set(lookup_a.keys()) | set(lookup_b.keys()))
    rows: list[dict[str, object]] = []
    for key in all_keys:
        test_name, metric_name = key
        a = lookup_a.get(key)
        b = lookup_b.get(key)
        score_a = a.score if a else None
        score_b = b.score if b else None
        delta = None
        if score_a is not None and score_b is not None:
            delta = score_b - score_a
        rows.append(
            {
                "test_name": test_name,
                "metric_name": metric_name,
                "a": a,
                "b": b,
                "score_a": score_a,
                "score_b": score_b,
                "delta": delta,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def generate_comparison_html(
    report: ComparisonReport,
    path: Path,
) -> None:
    """Generate a self-contained HTML comparison report."""
    stats_a = _summary_stats(report.results_a)
    stats_b = _summary_stats(report.results_b)
    rows = _paired_rows(report)

    # Determine overall winner
    if stats_a["avg_score"] > stats_b["avg_score"]:
        overall = report.label_a
    elif stats_b["avg_score"] > stats_a["avg_score"]:
        overall = report.label_b
    else:
        overall = "Tie"

    def _delta_color(d: float | None) -> str:
        if d is None:
            return "#888"
        if d > 0.001:
            return "#27ae60"
        if d < -0.001:
            return "#e74c3c"
        return "#888"

    def _fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "-"

    def _delta_fmt(d: float | None) -> str:
        if d is None:
            return "-"
        return f"{d:+.3f}"

    table_rows = ""
    for r in rows:
        color = _delta_color(r["delta"])
        a_pass = ""
        b_pass = ""
        if r["a"] is not None:
            a_pass = "PASS" if r["a"].passed else "FAIL"
        if r["b"] is not None:
            b_pass = "PASS" if r["b"].passed else "FAIL"
        table_rows += (
            f"<tr>"
            f"<td>{r['test_name']}</td>"
            f"<td>{r['metric_name']}</td>"
            f"<td>{a_pass}</td>"
            f"<td>{_fmt(r['score_a'])}</td>"
            f"<td>{b_pass}</td>"
            f"<td>{_fmt(r['score_b'])}</td>"
            f'<td style="color:{color};font-weight:bold">{_delta_fmt(r["delta"])}</td>'
            f"</tr>\n"
        )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>checkllm Comparison: {report.label_a} vs {report.label_b}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #f5f5f5; }}
.summary {{ display: flex; gap: 2rem; margin: 1rem 0; }}
.summary-card {{ padding: 1rem; border-radius: 8px; background: #f9f9f9; border: 1px solid #ddd; flex: 1; }}
.summary-card h3 {{ margin-top: 0; }}
.winner {{ background: #e8f5e9; border-color: #27ae60; }}
.improvement {{ color: #27ae60; }}
.regression {{ color: #e74c3c; }}
</style>
</head>
<body>
<h1>checkllm Comparison Report</h1>
<p><strong>Overall:</strong> {overall} {"is better" if overall != "Tie" else ""}</p>

<div class="summary">
  <div class="summary-card {"winner" if overall == report.label_a else ""}">
    <h3>{report.label_a}</h3>
    <p>Pass rate: {stats_a["pass_rate"]:.0%} ({stats_a["passed"]:.0f}/{stats_a["total"]:.0f})</p>
    <p>Avg score: {stats_a["avg_score"]:.3f}</p>
    <p>Total cost: ${stats_a["total_cost"]:.4f}</p>
  </div>
  <div class="summary-card {"winner" if overall == report.label_b else ""}">
    <h3>{report.label_b}</h3>
    <p>Pass rate: {stats_b["pass_rate"]:.0%} ({stats_b["passed"]:.0f}/{stats_b["total"]:.0f})</p>
    <p>Avg score: {stats_b["avg_score"]:.3f}</p>
    <p>Total cost: ${stats_b["total_cost"]:.4f}</p>
  </div>
</div>

<h2>Detailed Results</h2>
<table>
<thead>
<tr>
  <th>Test</th><th>Metric</th>
  <th>{report.label_a} Status</th><th>{report.label_a} Score</th>
  <th>{report.label_b} Status</th><th>{report.label_b} Score</th>
  <th>Delta</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def generate_comparison_markdown(
    report: ComparisonReport,
    path: Path,
) -> str:
    """Generate a Markdown comparison report. Writes to *path* and returns the text."""
    stats_a = _summary_stats(report.results_a)
    stats_b = _summary_stats(report.results_b)
    rows = _paired_rows(report)

    if stats_a["avg_score"] > stats_b["avg_score"]:
        overall = report.label_a
    elif stats_b["avg_score"] > stats_a["avg_score"]:
        overall = report.label_b
    else:
        overall = "Tie"

    lines: list[str] = []
    lines.append(f"# checkllm Comparison: {report.label_a} vs {report.label_b}")
    lines.append("")
    lines.append(f"**Overall:** {overall}" + (" is better" if overall != "Tie" else ""))
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | {report.label_a} | {report.label_b} |")
    lines.append("|--------|------:|------:|")
    lines.append(f"| Pass rate | {stats_a['pass_rate']:.0%} | {stats_b['pass_rate']:.0%} |")
    lines.append(f"| Avg score | {stats_a['avg_score']:.3f} | {stats_b['avg_score']:.3f} |")
    lines.append(f"| Total cost | ${stats_a['total_cost']:.4f} | ${stats_b['total_cost']:.4f} |")
    lines.append("")

    # Detail table
    lines.append("## Detailed Results")
    lines.append("")
    lines.append(f"| Test | Metric | {report.label_a} | {report.label_b} | Delta |")
    lines.append("|------|--------|------:|------:|------:|")
    for r in rows:
        sa = f"{r['score_a']:.2f}" if r["score_a"] is not None else "-"
        sb = f"{r['score_b']:.2f}" if r["score_b"] is not None else "-"
        delta = f"{r['delta']:+.3f}" if r["delta"] is not None else "-"
        lines.append(f"| {r['test_name']} | {r['metric_name']} | {sa} | {sb} | {delta} |")
    lines.append("")

    md = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md)
    return md


# ---------------------------------------------------------------------------
# Terminal (Rich)
# ---------------------------------------------------------------------------


def render_comparison_terminal(
    report: ComparisonReport,
    to_string: bool = False,
) -> str | None:
    """Render a comparison report to the terminal using Rich."""
    buf = StringIO() if to_string else None
    console = Console(file=buf, force_terminal=True, width=120) if buf else Console()

    stats_a = _summary_stats(report.results_a)
    stats_b = _summary_stats(report.results_b)

    if stats_a["avg_score"] > stats_b["avg_score"]:
        overall = report.label_a
    elif stats_b["avg_score"] > stats_a["avg_score"]:
        overall = report.label_b
    else:
        overall = "Tie"

    console.print(
        f"\n[bold]Comparison:[/] {report.label_a} vs {report.label_b}  |  "
        f"[bold]Overall:[/] {overall}" + (" is better" if overall != "Tie" else "")
    )

    # Summary table
    summary = Table(title="Summary", show_lines=True)
    summary.add_column("Metric")
    summary.add_column(report.label_a, justify="right")
    summary.add_column(report.label_b, justify="right")
    summary.add_row("Pass rate", f"{stats_a['pass_rate']:.0%}", f"{stats_b['pass_rate']:.0%}")
    summary.add_row("Avg score", f"{stats_a['avg_score']:.3f}", f"{stats_b['avg_score']:.3f}")
    summary.add_row("Total cost", f"${stats_a['total_cost']:.4f}", f"${stats_b['total_cost']:.4f}")
    console.print(summary)

    # Detail table
    detail = Table(title="Detailed Results", show_lines=True)
    detail.add_column("Test")
    detail.add_column("Metric")
    detail.add_column(f"{report.label_a} Score", justify="right")
    detail.add_column(f"{report.label_b} Score", justify="right")
    detail.add_column("Delta", justify="right")

    rows = _paired_rows(report)
    for r in rows:
        sa = f"{r['score_a']:.2f}" if r["score_a"] is not None else "-"
        sb = f"{r['score_b']:.2f}" if r["score_b"] is not None else "-"
        if r["delta"] is not None:
            d = r["delta"]
            if d > 0.001:
                delta_text = Text(f"{d:+.3f}", style="bold green")
            elif d < -0.001:
                delta_text = Text(f"{d:+.3f}", style="bold red")
            else:
                delta_text = Text(f"{d:+.3f}")
        else:
            delta_text = Text("-")
        detail.add_row(r["test_name"], r["metric_name"], sa, sb, delta_text)

    console.print(detail)

    if buf:
        return buf.getvalue()
    return None

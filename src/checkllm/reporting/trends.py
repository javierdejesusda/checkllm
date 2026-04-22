"""Trend reporting — track scores, pass rates and costs over time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.table import Table

from checkllm.models import CheckResult


@dataclass
class TrendData:
    """One snapshot of results for a particular run."""

    run_id: str
    timestamp: datetime
    label: str
    results: dict[str, list[CheckResult]]

    # Pre-computed convenience attributes
    @property
    def all_checks(self) -> list[CheckResult]:
        return [c for checks in self.results.values() for c in checks]

    @property
    def pass_rate(self) -> float:
        total = len(self.all_checks)
        if total == 0:
            return 0.0
        return sum(1 for c in self.all_checks if c.passed) / total

    @property
    def avg_score(self) -> float:
        total = len(self.all_checks)
        if total == 0:
            return 0.0
        return sum(c.score for c in self.all_checks) / total

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.all_checks)

    def metric_avg(self, metric_name: str) -> float | None:
        """Average score for a specific metric across all tests."""
        scores = [
            c.score
            for checks in self.results.values()
            for c in checks
            if c.metric_name == metric_name
        ]
        if not scores:
            return None
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# SVG chart helpers (no external dependencies)
# ---------------------------------------------------------------------------


def _build_line_svg(
    points: list[tuple[str, float]],
    title: str,
    width: int = 600,
    height: int = 250,
    y_min: float = 0.0,
    y_max: float = 1.0,
    fmt: str = ".2f",
) -> str:
    """Build an inline SVG line chart from (label, value) pairs."""
    margin_left = 60
    margin_right = 20
    margin_top = 40
    margin_bottom = 50
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    y_range = y_max - y_min if y_max != y_min else 1.0

    def _x(i: int) -> float:
        if len(points) <= 1:
            return margin_left + plot_w / 2
        return margin_left + i * plot_w / (len(points) - 1)

    def _y(v: float) -> float:
        return margin_top + plot_h - ((v - y_min) / y_range) * plot_h

    # Build polyline
    coords = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, (_, v) in enumerate(points))

    # Data point circles
    circles = "\n".join(
        f'<circle cx="{_x(i):.1f}" cy="{_y(v):.1f}" r="4" fill="#3498db" />'
        for i, (_, v) in enumerate(points)
    )

    # X-axis labels
    x_labels = "\n".join(
        f'<text x="{_x(i):.1f}" y="{height - 5}" text-anchor="middle" '
        f'font-size="10" fill="#666">{lbl}</text>'
        for i, (lbl, _) in enumerate(points)
    )

    # Y-axis gridlines + labels (5 ticks)
    gridlines = ""
    for tick in range(5):
        frac = tick / 4
        yv = y_min + frac * y_range
        yp = _y(yv)
        gridlines += (
            f'<line x1="{margin_left}" y1="{yp:.1f}" '
            f'x2="{width - margin_right}" y2="{yp:.1f}" '
            f'stroke="#eee" stroke-width="1" />\n'
            f'<text x="{margin_left - 8}" y="{yp + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="#666">{yv:{fmt}}</text>\n'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#fff;border:1px solid #ddd;border-radius:4px;margin:8px 0">\n'
        f'<text x="{width // 2}" y="20" text-anchor="middle" font-size="14" '
        f'font-weight="bold" fill="#333">{title}</text>\n'
        f"{gridlines}"
        f'<polyline points="{coords}" fill="none" stroke="#3498db" stroke-width="2" />\n'
        f"{circles}\n"
        f"{x_labels}\n"
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# HTML trend report
# ---------------------------------------------------------------------------


def generate_trend_html(
    data: list[TrendData],
    path: Path,
) -> None:
    """Generate a self-contained HTML report with inline SVG trend charts.

    Charts included:
    - Average score over time
    - Pass rate over time
    - Total cost over time
    - One line chart per unique metric
    """
    if not data:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html><body><p>No trend data.</p></body></html>")
        return

    sorted_data = sorted(data, key=lambda d: d.timestamp)
    labels = [d.label for d in sorted_data]

    # Collect unique metric names
    metrics: set[str] = set()
    for d in sorted_data:
        for checks in d.results.values():
            for c in checks:
                metrics.add(c.metric_name)
    metric_names = sorted(metrics)

    charts: list[str] = []

    # 1. Average score over time
    avg_points = [(lbl, d.avg_score) for lbl, d in zip(labels, sorted_data)]
    charts.append(_build_line_svg(avg_points, "Average Score"))

    # 2. Pass rate over time
    pr_points = [(lbl, d.pass_rate) for lbl, d in zip(labels, sorted_data)]
    charts.append(_build_line_svg(pr_points, "Pass Rate"))

    # 3. Cost over time
    cost_vals = [d.total_cost for d in sorted_data]
    cost_max = max(cost_vals) if cost_vals else 1.0
    cost_points = [(lbl, d.total_cost) for lbl, d in zip(labels, sorted_data)]
    charts.append(
        _build_line_svg(
            cost_points,
            "Total Cost",
            y_min=0.0,
            y_max=cost_max * 1.2 if cost_max > 0 else 1.0,
            fmt=".4f",
        )
    )

    # 4. Per-metric charts
    for metric in metric_names:
        points: list[tuple[str, float]] = []
        for lbl, d in zip(labels, sorted_data):
            avg = d.metric_avg(metric)
            if avg is not None:
                points.append((lbl, avg))
        if points:
            charts.append(_build_line_svg(points, f"Metric: {metric}"))

    charts_html = "\n".join(charts)

    # Build run table
    run_rows = ""
    for d in sorted_data:
        run_rows += (
            f"<tr>"
            f"<td>{d.run_id}</td>"
            f"<td>{d.label}</td>"
            f"<td>{d.timestamp.isoformat()}</td>"
            f"<td>{d.pass_rate:.0%}</td>"
            f"<td>{d.avg_score:.3f}</td>"
            f"<td>${d.total_cost:.4f}</td>"
            f"</tr>\n"
        )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>checkllm Trend Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #f5f5f5; }}
.charts {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
</style>
</head>
<body>
<h1>checkllm Trend Report</h1>
<p>{len(sorted_data)} run(s) tracked</p>

<div class="charts">
{charts_html}
</div>

<h2>Run History</h2>
<table>
<thead>
<tr><th>Run ID</th><th>Label</th><th>Timestamp</th><th>Pass Rate</th><th>Avg Score</th><th>Cost</th></tr>
</thead>
<tbody>
{run_rows}
</tbody>
</table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


# ---------------------------------------------------------------------------
# Terminal sparkline with Rich
# ---------------------------------------------------------------------------

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], lo: float = 0.0, hi: float = 1.0) -> str:
    """Return a sparkline string for a list of float values."""
    if not values:
        return ""
    rng = hi - lo if hi != lo else 1.0
    chars = []
    for v in values:
        idx = int((v - lo) / rng * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


def render_trend_terminal(
    data: list[TrendData],
    to_string: bool = False,
) -> str | None:
    """Render trend data with sparklines in the terminal using Rich."""
    buf = StringIO() if to_string else None
    console = Console(file=buf, force_terminal=True, width=120) if buf else Console()

    if not data:
        console.print("[dim]No trend data to display.[/]")
        if buf:
            return buf.getvalue()
        return None

    sorted_data = sorted(data, key=lambda d: d.timestamp)

    table = Table(title="checkllm Trend", show_lines=True)
    table.add_column("Metric")
    table.add_column("Trend")
    table.add_column("Latest", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    # Avg score sparkline
    avg_scores = [d.avg_score for d in sorted_data]
    table.add_row(
        "Avg Score",
        _sparkline(avg_scores),
        f"{avg_scores[-1]:.3f}",
        f"{min(avg_scores):.3f}",
        f"{max(avg_scores):.3f}",
    )

    # Pass rate sparkline
    pass_rates = [d.pass_rate for d in sorted_data]
    table.add_row(
        "Pass Rate",
        _sparkline(pass_rates),
        f"{pass_rates[-1]:.0%}",
        f"{min(pass_rates):.0%}",
        f"{max(pass_rates):.0%}",
    )

    # Cost sparkline
    costs = [d.total_cost for d in sorted_data]
    cost_max = max(costs) if costs else 1.0
    table.add_row(
        "Cost",
        _sparkline(costs, lo=0.0, hi=cost_max if cost_max > 0 else 1.0),
        f"${costs[-1]:.4f}",
        f"${min(costs):.4f}",
        f"${max(costs):.4f}",
    )

    # Per-metric sparklines
    metrics: set[str] = set()
    for d in sorted_data:
        for checks in d.results.values():
            for c in checks:
                metrics.add(c.metric_name)
    for metric in sorted(metrics):
        values: list[float] = []
        for d in sorted_data:
            avg = d.metric_avg(metric)
            if avg is not None:
                values.append(avg)
        if values:
            table.add_row(
                metric,
                _sparkline(values),
                f"{values[-1]:.3f}",
                f"{min(values):.3f}",
                f"{max(values):.3f}",
            )

    console.print(table)

    # Run labels
    console.print("\n[bold]Runs:[/] " + " -> ".join(d.label for d in sorted_data))

    if buf:
        return buf.getvalue()
    return None

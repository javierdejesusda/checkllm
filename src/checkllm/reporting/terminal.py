from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.table import Table
from rich.text import Text

from checkllm.models import CheckResult
from checkllm.regression.compare import RegressionItem


def render_results(
    results: list[CheckResult],
    to_string: bool = False,
) -> str | None:
    """Render check results as a rich table."""
    buf = StringIO() if to_string else None
    console = Console(file=buf, force_terminal=True) if buf else Console()

    table = Table(title="checkllm Results", show_lines=True)
    table.add_column("Status", width=6)
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_column("Reasoning")
    table.add_column("Cost", justify="right")
    table.add_column("Latency", justify="right")

    total_cost = 0.0
    for r in results:
        status = Text("PASS", style="bold green") if r.passed else Text("FAIL", style="bold red")
        table.add_row(
            status,
            r.metric_name,
            f"{r.score:.2f}",
            r.reasoning[:80],
            f"${r.cost:.4f}",
            f"{r.latency_ms}ms",
        )
        total_cost += r.cost

    console.print(table)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    summary_style = "bold green" if failed == 0 else "bold red"
    console.print(
        f"\n[{summary_style}]{passed} passed, {failed} failed[/] | Total cost: ${total_cost:.4f}"
    )

    if buf:
        return buf.getvalue()
    return None


def render_regression_report(
    items: list[RegressionItem],
    to_string: bool = False,
) -> str | None:
    """Render regression comparison results."""
    buf = StringIO() if to_string else None
    console = Console(file=buf, force_terminal=True, width=120) if buf else Console()

    table = Table(title="Regression Report", show_lines=True)
    table.add_column("Status", width=12)
    table.add_column("Test")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("p-value", justify="right")

    for item in items:
        c = item.comparison
        if c.is_regression:
            status = Text("REGRESSION", style="bold red")
        elif c.delta >= 0:
            status = Text("IMPROVED", style="bold green")
        else:
            status = Text("OK", style="bold green")

        table.add_row(
            status,
            item.test_name,
            item.metric_name,
            f"{c.baseline_mean:.3f}",
            f"{c.current_mean:.3f}",
            f"{c.delta:+.3f}",
            f"{c.p_value:.4f}",
        )

    console.print(table)

    regressions = [i for i in items if i.comparison.is_regression]
    if regressions:
        console.print(f"\n[bold red]{len(regressions)} regression(s) detected[/]")
    else:
        console.print("\n[bold green]No regressions detected[/]")

    if buf:
        return buf.getvalue()
    return None

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from checkllm import __version__
from checkllm.config import load_config

app = typer.Typer(
    name="checkllm",
    help="Test LLM-powered applications with the same rigor as traditional software.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"checkllm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, is_eager=True,
        help="Show version and exit.",
    ),
):
    pass


@app.command()
def run(
    test_path: str = typer.Argument(help="Path to test directory or file"),
    compare: Optional[str] = typer.Option(None, "--compare", help="Path to baseline snapshot for regression comparison"),
    fail_on_regression: bool = typer.Option(False, "--fail-on-regression", help="Exit 1 if regression detected"),
    junit_xml: Optional[str] = typer.Option(None, "--junit-xml", help="Write JUnit XML to this path"),
    html_report: Optional[str] = typer.Option(None, "--html-report", help="Generate HTML report to this path"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", help="Save snapshot to this path"),
):
    """Run LLM tests with rich terminal output."""
    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    if junit_xml:
        cmd.extend([f"--checkllm-junit={junit_xml}"])
    if html_report:
        cmd.extend([f"--checkllm-report={html_report}"])
    if snapshot:
        cmd.extend([f"--checkllm-snapshot={snapshot}"])

    result = subprocess.run(cmd)
    exit_code = result.returncode

    # If --compare, compare the just-saved snapshot against a baseline
    if compare and snapshot:
        _run_comparison(compare, snapshot, fail_on_regression)

    raise typer.Exit(code=exit_code)


def _run_comparison(baseline_path: str, current_path: str, fail_on_regression: bool) -> None:
    """Compare two snapshots and print regression report."""
    from checkllm.regression.snapshot import load_snapshot
    from checkllm.regression.compare import compare_snapshot
    from checkllm.reporting.terminal import render_regression_report

    bp = Path(baseline_path)
    cp = Path(current_path)

    if not bp.exists():
        console.print(f"[bold yellow]Baseline snapshot not found: {bp}[/]")
        console.print("[dim]Run 'checkllm snapshot' first to create a baseline.[/]")
        return
    if not cp.exists():
        console.print(f"[bold red]Current snapshot not found: {cp}[/]")
        return

    baseline = load_snapshot(bp)
    current = load_snapshot(cp)
    config = load_config()
    report = compare_snapshot(baseline, current, p_threshold=config.p_value_threshold)

    render_regression_report(report.comparisons)

    if report.has_regressions and fail_on_regression:
        raise typer.Exit(code=1)


@app.command()
def snapshot(
    test_path: str = typer.Argument(help="Path to test directory or file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Snapshot output path"),
):
    """Run tests and save results as a regression baseline snapshot."""
    config = load_config()
    snapshot_dir = Path(config.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if output is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(snapshot_dir / f"snapshot_{ts}.json")

    console.print(f"[bold]Saving snapshot to:[/] {output}")

    cmd = [sys.executable, "-m", "pytest", test_path, "-v",
           f"--checkllm-snapshot={output}"]
    result = subprocess.run(cmd)

    if Path(output).exists():
        console.print(f"[bold green]Snapshot saved to {output}[/]")
    else:
        console.print("[bold red]No checkllm results were collected. Snapshot not saved.[/]")

    raise typer.Exit(code=result.returncode)


@app.command()
def report(
    test_path: str = typer.Argument(help="Path to test directory or file"),
    output: str = typer.Option("checkllm_report.html", "--output", "-o", help="Output file path"),
    junit_xml: Optional[str] = typer.Option(None, "--junit-xml", help="Also write JUnit XML"),
):
    """Run tests and generate an HTML report."""
    console.print(f"[bold]Generating report to:[/] {output}")

    cmd = [sys.executable, "-m", "pytest", test_path, "-v",
           f"--checkllm-report={output}"]
    if junit_xml:
        cmd.append(f"--checkllm-junit={junit_xml}")

    result = subprocess.run(cmd)

    if Path(output).exists():
        console.print(f"[bold green]HTML report written to {output}[/]")
    else:
        console.print("[bold red]No checkllm results were collected. Report not generated.[/]")

    raise typer.Exit(code=result.returncode)


@app.command(name="eval")
def eval_cmd(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt template with {input} placeholder"),
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="Path to dataset YAML file"),
    metric: str = typer.Option("rubric", "--metric", "-m", help="Metric to evaluate (hallucination, relevance, toxicity, rubric)"),
    threshold: float = typer.Option(0.8, "--threshold", "-t", help="Pass/fail threshold"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save results as snapshot JSON"),
):
    """Evaluate a prompt template against a dataset using LLM-as-judge."""
    import asyncio
    from checkllm.datasets.loader import load_yaml_dataset
    from checkllm.check import CheckCollector
    from checkllm.reporting.terminal import render_results

    cases = load_yaml_dataset(Path(dataset_path))
    config = load_config()
    collector = CheckCollector(config=config)

    console.print(f"[bold]Evaluating {len(cases)} cases with metric '{metric}'[/]")
    console.print(f"[dim]Prompt template: {prompt}[/]")
    console.print()

    for i, case in enumerate(cases):
        rendered_prompt = prompt.replace("{input}", case.input)
        if case.query:
            rendered_prompt = rendered_prompt.replace("{query}", case.query)

        console.print(f"  [dim]Case {i + 1}/{len(cases)}: {case.input[:60]}...[/]")

        # For rubric, use criteria from case; for others, use appropriate fields
        if metric == "hallucination":
            collector.hallucination(rendered_prompt, context=case.input, threshold=threshold)
        elif metric == "relevance":
            collector.relevance(rendered_prompt, query=case.query or case.input, threshold=threshold)
        elif metric == "toxicity":
            collector.toxicity(rendered_prompt, threshold=threshold)
        elif metric == "rubric":
            criteria = case.criteria or "Output should be accurate, helpful, and concise."
            collector.rubric(rendered_prompt, criteria=criteria, threshold=threshold)
        else:
            console.print(f"[bold red]Unknown metric: {metric}[/]")
            raise typer.Exit(code=1)

    console.print()
    render_results(collector.results)

    if output:
        from checkllm.regression.snapshot import (
            MetricRecord, Snapshot, TestRunRecord, save_snapshot,
        )
        from datetime import datetime, timezone

        tests = {
            "eval": [
                TestRunRecord(
                    metrics={r.metric_name: MetricRecord(score=r.score, passed=r.passed)}
                )
                for r in collector.results
            ]
        }
        snap = Snapshot(
            version=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tests=tests,
        )
        save_snapshot(snap, Path(output))
        console.print(f"\n[bold green]Results saved to {output}[/]")

    failed = [r for r in collector.results if not r.passed]
    raise typer.Exit(code=1 if failed else 0)


@app.command()
def diff(
    baseline: str = typer.Option(..., "--baseline", "-b", help="Path to baseline snapshot"),
    current: str = typer.Option(..., "--current", "-c", help="Path to current snapshot"),
    fail_on_regression: bool = typer.Option(False, "--fail-on-regression", help="Exit 1 if regression detected"),
):
    """Compare two snapshots and detect regressions."""
    from checkllm.regression.snapshot import load_snapshot
    from checkllm.regression.compare import compare_snapshot
    from checkllm.reporting.terminal import render_regression_report

    bp = Path(baseline)
    cp = Path(current)

    if not bp.exists():
        console.print(f"[bold red]Baseline snapshot not found: {bp}[/]")
        raise typer.Exit(code=1)
    if not cp.exists():
        console.print(f"[bold red]Current snapshot not found: {cp}[/]")
        raise typer.Exit(code=1)

    baseline_snap = load_snapshot(bp)
    current_snap = load_snapshot(cp)

    config = load_config()
    report = compare_snapshot(
        baseline_snap, current_snap, p_threshold=config.p_value_threshold,
    )

    render_regression_report(report.comparisons)

    if report.has_regressions:
        console.print(f"\n[bold red]{len(report.regressions)} regression(s) detected![/]")
        if fail_on_regression:
            raise typer.Exit(code=1)
    else:
        console.print("\n[bold green]No regressions detected.[/]")

    raise typer.Exit(code=0)

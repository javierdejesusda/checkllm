from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
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

dataset_app = typer.Typer(
    name="dataset",
    help="Load, split, and version evaluation datasets.",
    no_args_is_help=True,
)
app.add_typer(dataset_app, name="dataset")

drift_app = typer.Typer(
    name="drift",
    help="Record and check judge drift against a canonical probe set.",
    no_args_is_help=True,
)
app.add_typer(drift_app, name="drift")


def version_callback(value: bool):
    if value:
        console.print(f"checkllm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    pass


@app.command()
def run(
    test_path: str = typer.Argument(help="Path to test directory or file"),
    compare: Optional[str] = typer.Option(
        None, "--compare", help="Path to baseline snapshot for regression comparison"
    ),
    fail_on_regression: bool = typer.Option(
        False, "--fail-on-regression", help="Exit 1 if regression detected"
    ),
    junit_xml: Optional[str] = typer.Option(
        None, "--junit-xml", help="Write JUnit XML to this path"
    ),
    html_report: Optional[str] = typer.Option(
        None, "--html-report", help="Generate HTML report to this path"
    ),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", help="Save snapshot to this path"),
    budget: Optional[float] = typer.Option(
        None, "--budget", help="Maximum USD to spend on judge calls"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable judge response caching"),
    label: Optional[str] = typer.Option(
        None, "--label", "-l", help="Label for this run in history"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate costs without running tests"),
):
    """Run LLM tests with rich terminal output."""
    if dry_run:
        from checkllm.estimator import estimate_from_test_file, CostEstimate

        path = Path(test_path)
        files = [path] if path.is_file() else sorted(path.rglob("test_*.py"))
        config = load_config()
        model = config.judge_model
        total = CostEstimate(model=model)
        for f in files:
            est = estimate_from_test_file(str(f), model=model)
            total.deterministic_count += est.deterministic_count
            total.judge_count += est.judge_count
            total.total_cost += est.total_cost
        total.total_cost = round(total.total_cost, 4)
        console.print(f"[bold]Dry run — {total.summary()}[/]")
        raise typer.Exit(code=0)

    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    if junit_xml:
        cmd.extend([f"--checkllm-junit={junit_xml}"])
    if html_report:
        cmd.extend([f"--checkllm-report={html_report}"])
    if snapshot:
        cmd.extend([f"--checkllm-snapshot={snapshot}"])

    # Pass budget and cache settings via environment
    import os

    env = os.environ.copy()
    if budget is not None:
        env["CHECKLLM_BUDGET"] = str(budget)
    if no_cache:
        env["CHECKLLM_CACHE_ENABLED"] = "false"
    if label:
        env["CHECKLLM_RUN_LABEL"] = label

    result = subprocess.run(cmd, env=env)
    exit_code = result.returncode

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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(snapshot_dir / f"snapshot_{ts}.json")

    console.print(f"[bold]Saving snapshot to:[/] {output}")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-v",
        f"--checkllm-snapshot={output}",
    ]
    result = subprocess.run(cmd)

    if Path(output).exists():
        console.print(f"[bold green]Snapshot saved to {output}[/]")
    else:
        console.print("[bold red]No checkllm results were collected. Snapshot not saved.[/]")

    raise typer.Exit(code=result.returncode)


@app.command()
def analyze(
    run_a: str = typer.Argument(help="Baseline snapshot path (run A)"),
    run_b: str = typer.Argument(help="Candidate snapshot path (run B)"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance threshold"),
    bootstrap: int = typer.Option(
        2000, "--bootstrap", help="Bootstrap resamples for CI computation"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for bootstrap"),
    method: str = typer.Option(
        "welch", "--method", help="Significance method: welch | mann_whitney | bootstrap"
    ),
    correlation: bool = typer.Option(
        False, "--correlation", help="Also emit correlation analysis for run B"
    ),
    json_out: Optional[str] = typer.Option(
        None, "--json", help="Write machine-readable JSON report to this path"
    ),
):
    """A/B analyze two snapshots with t-test, Mann-Whitney U, and bootstrap CI."""
    import json as _json

    from checkllm.analysis.correlation import (
        build_correlation_matrix,
        summarize_correlation_matrix,
    )
    from checkllm.analysis.significance import analyze_runs
    from checkllm.regression.snapshot import load_snapshot

    path_a = Path(run_a)
    path_b = Path(run_b)
    if not path_a.exists():
        console.print(f"[bold red]Run A snapshot not found: {path_a}[/]")
        raise typer.Exit(code=2)
    if not path_b.exists():
        console.print(f"[bold red]Run B snapshot not found: {path_b}[/]")
        raise typer.Exit(code=2)

    snap_a = load_snapshot(path_a)
    snap_b = load_snapshot(path_b)

    if method not in {"welch", "mann_whitney", "bootstrap"}:
        console.print(f"[bold red]Unknown method: {method}[/]")
        raise typer.Exit(code=2)

    results = analyze_runs(
        snap_a,
        snap_b,
        alpha=alpha,
        n_bootstrap=bootstrap,
        seed=seed,
        method=method,  # type: ignore[arg-type]
    )

    from rich.table import Table

    table = Table(title=f"A/B Analysis ({method})", show_lines=True)
    table.add_column("Metric")
    table.add_column("n_a", justify="right")
    table.add_column("n_b", justify="right")
    table.add_column("mean_a", justify="right")
    table.add_column("mean_b", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("Cohen's d", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("sig?", justify="center")

    for r in results:
        sig_mark = "[bold red]YES[/]" if r.significant else "[dim]no[/]"
        table.add_row(
            r.metric,
            str(r.n_a),
            str(r.n_b),
            f"{r.mean_a:.3f}",
            f"{r.mean_b:.3f}",
            f"{r.delta:+.3f}",
            f"{r.effect_size:+.2f}",
            f"{r.p_value:.4f}",
            f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
            sig_mark,
        )
    console.print(table)

    payload: dict[str, object] = {
        "method": method,
        "alpha": alpha,
        "results": [
            {
                "metric": r.metric,
                "n_a": r.n_a,
                "n_b": r.n_b,
                "mean_a": r.mean_a,
                "mean_b": r.mean_b,
                "delta": r.delta,
                "effect_size": r.effect_size,
                "p_value": r.p_value,
                "mann_whitney_p": r.mann_whitney_p,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "significant": r.significant,
            }
            for r in results
        ],
    }

    if correlation:
        matrix = build_correlation_matrix(snap_b)
        payload["correlation"] = summarize_correlation_matrix(matrix)
        best = matrix.best_predictor()
        if best is not None:
            console.print(
                "\n[bold]Best pass/fail predictor:[/] "
                f"{best.metric_a} (Pearson r={best.pearson_r:+.3f}, p={best.pearson_p:.4f})"
            )

    if json_out:
        Path(json_out).write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[dim]JSON report written to {json_out}[/]")

    # Non-zero exit when any metric shows a significant regression.
    # delta = mean_a - mean_b, so candidate (B) regressing vs baseline (A)
    # shows up as a positive delta.
    any_regression = any(r.significant and r.delta > 0 for r in results)
    raise typer.Exit(code=1 if any_regression else 0)


@app.command()
def report(
    test_path: str = typer.Argument(help="Path to test directory or file"),
    output: str = typer.Option("checkllm_report.html", "--output", "-o", help="Output file path"),
    junit_xml: Optional[str] = typer.Option(None, "--junit-xml", help="Also write JUnit XML"),
):
    """Run tests and generate an HTML report."""
    console.print(f"[bold]Generating report to:[/] {output}")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-v",
        f"--checkllm-report={output}",
    ]
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
    prompt: str = typer.Option(
        ..., "--prompt", "-p", help="Prompt template with {input} placeholder"
    ),
    dataset_path: str = typer.Option(
        ..., "--dataset", "-d", help="Path to dataset file (YAML, JSON, CSV)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-M", help="Model to generate outputs (default: from config)"
    ),
    metric: str = typer.Option(
        "rubric",
        "--metric",
        "-m",
        help="Metric to evaluate (hallucination, relevance, toxicity, rubric)",
    ),
    threshold: float = typer.Option(0.8, "--threshold", "-t", help="Pass/fail threshold"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save results as snapshot JSON"
    ),
    budget: Optional[float] = typer.Option(
        None, "--budget", help="Maximum USD to spend on judge calls"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable judge response caching"),
    label: Optional[str] = typer.Option(
        None, "--label", "-l", help="Label for this run in history"
    ),
):
    """Evaluate a prompt template against a dataset."""
    from checkllm.datasets.loader import load_dataset
    from checkllm.check import CheckCollector
    from checkllm.reporting.terminal import render_results
    from checkllm.history import RunHistory

    cases = load_dataset(Path(dataset_path))
    config = load_config()
    if budget is not None:
        config.budget = budget
    if no_cache:
        config.cache_enabled = False
    gen_model = model or config.judge_model
    collector = CheckCollector(config=config)

    console.print(f"[bold]Evaluating {len(cases)} cases[/]")
    console.print(f"[dim]Generation model: {gen_model}[/]")
    console.print(f"[dim]Judge metric: {metric} (threshold={threshold})[/]")
    if config.budget is not None:
        console.print(f"[dim]Budget: ${config.budget:.2f}[/]")
    console.print()

    valid_metrics = {
        "hallucination",
        "relevance",
        "toxicity",
        "rubric",
        "fluency",
        "coherence",
        "sentiment",
        "correctness",
    }
    if metric not in valid_metrics:
        console.print(
            f"[bold red]Unknown metric: {metric}. Valid: {', '.join(sorted(valid_metrics))}[/]"
        )
        raise typer.Exit(code=1)

    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        MofNCompleteColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(cases))

        for i, case in enumerate(cases):
            rendered_prompt = prompt.replace("{input}", case.input)
            if case.query:
                rendered_prompt = rendered_prompt.replace("{query}", case.query)
            if case.context:
                rendered_prompt = rendered_prompt.replace("{context}", case.context)

            progress.update(task, description=f"Case {i + 1}: {case.input[:40]}...")

            llm_output = _generate_output(rendered_prompt, gen_model, config)

            if metric == "hallucination":
                context = case.context or case.input
                collector.hallucination(llm_output, context=context, threshold=threshold)
            elif metric == "relevance":
                collector.relevance(llm_output, query=case.query or case.input, threshold=threshold)
            elif metric == "toxicity":
                collector.toxicity(llm_output, threshold=threshold)
            elif metric == "rubric":
                criteria = case.criteria or "Output should be accurate, helpful, and concise."
                collector.rubric(llm_output, criteria=criteria, threshold=threshold)
            elif metric == "fluency":
                collector.fluency(llm_output, threshold=threshold)
            elif metric == "coherence":
                collector.coherence(llm_output, threshold=threshold)
            elif metric == "sentiment":
                collector.sentiment(llm_output, threshold=threshold)
            elif metric == "correctness":
                expected = case.expected or case.input
                collector.correctness(llm_output, expected=expected, threshold=threshold)

            progress.advance(task)

    console.print()
    render_results(collector.results)

    # Show cost summary
    console.print(f"\n[dim]Total cost: ${collector.total_cost:.4f}[/]")
    cache_stats = collector.cache_stats
    if cache_stats.get("session_hits", 0):
        console.print(
            f"[dim]Cache: {cache_stats['session_hits']} hits, "
            f"{cache_stats['session_misses']} misses, "
            f"saved ${cache_stats['session_saved_cost']:.4f}[/]"
        )

    # Save to history
    history = RunHistory()
    results_dict = {f"eval_case_{i}": [r] for i, r in enumerate(collector.results)}
    run_id = history.record_run(results_dict, label=label or "eval")
    console.print(f"[dim]Saved as run #{run_id} in history[/]")
    history.close()

    if output:
        from checkllm.regression.snapshot import (
            MetricRecord,
            Snapshot,
            TestRunRecord,
            save_snapshot,
        )

        tests = {
            f"eval_case_{i}": [
                TestRunRecord(metrics={r.metric_name: MetricRecord(score=r.score, passed=r.passed)})
            ]
            for i, r in enumerate(collector.results)
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


def _generate_output(prompt: str, model: str, config) -> str:
    """Call an LLM to generate output from a prompt."""
    import asyncio

    async def _call():
        if config.judge_backend == "anthropic":
            try:
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic()
                response = await client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text if response.content else ""
            except Exception as e:
                return f"[Generation failed: {e}]"
        else:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _call()).result()
    return asyncio.run(_call())


@app.command()
def diff(
    baseline: str = typer.Option(..., "--baseline", "-b", help="Path to baseline snapshot"),
    current: str = typer.Option(..., "--current", "-c", help="Path to current snapshot"),
    fail_on_regression: bool = typer.Option(
        False, "--fail-on-regression", help="Exit 1 if regression detected"
    ),
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
        baseline_snap,
        current_snap,
        p_threshold=config.p_value_threshold,
    )

    render_regression_report(report.comparisons)

    if report.has_regressions:
        console.print(f"\n[bold red]{len(report.regressions)} regression(s) detected![/]")
        if fail_on_regression:
            raise typer.Exit(code=1)
    else:
        console.print("\n[bold green]No regressions detected.[/]")

    raise typer.Exit(code=0)


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of runs to show"),
    run_id: Optional[int] = typer.Option(
        None, "--run", "-r", help="Show details for a specific run"
    ),
    trend: Optional[str] = typer.Option(
        None,
        "--trend",
        help="Show score trend for test::metric (e.g. 'test_qa::hallucination')",
    ),
    compare_runs: Optional[str] = typer.Option(
        None, "--compare", help="Compare two runs: 'ID1,ID2'"
    ),
):
    """View historical run data and trends."""
    from rich.table import Table
    from rich.text import Text
    from checkllm.history import RunHistory

    hist = RunHistory()

    if trend:
        parts = trend.split("::")
        if len(parts) != 2:
            console.print("[bold red]--trend format: 'test_name::metric_name'[/]")
            raise typer.Exit(code=1)
        test_name, metric_name = parts
        data = hist.get_metric_trend(test_name, metric_name, limit=limit)
        if not data:
            console.print(f"[dim]No data found for {test_name}::{metric_name}[/]")
            raise typer.Exit(code=0)

        table = Table(title=f"Trend: {test_name} / {metric_name}")
        table.add_column("Run", justify="right")
        table.add_column("Time")
        table.add_column("Label")
        table.add_column("Commit")
        table.add_column("Score", justify="right")
        table.add_column("Status", width=6)

        for point in data:
            ts = datetime.fromtimestamp(point["timestamp"]).strftime("%Y-%m-%d %H:%M")
            status = (
                Text("PASS", style="bold green")
                if point["passed"]
                else Text("FAIL", style="bold red")
            )
            table.add_row(
                str(point["run_id"]),
                ts,
                point["label"] or "",
                point["git_commit"] or "",
                f"{point['score']:.3f}",
                status,
            )
        console.print(table)
        hist.close()
        raise typer.Exit(code=0)

    if compare_runs:
        parts = compare_runs.split(",")
        if len(parts) != 2:
            console.print("[bold red]--compare format: 'ID1,ID2'[/]")
            raise typer.Exit(code=1)
        id1, id2 = int(parts[0].strip()), int(parts[1].strip())
        run1 = hist.get_run(id1)
        run2 = hist.get_run(id2)
        if not run1 or not run2:
            console.print("[bold red]One or both runs not found[/]")
            raise typer.Exit(code=1)
        _render_run_comparison(run1, run2)
        hist.close()
        raise typer.Exit(code=0)

    if run_id is not None:
        record = hist.get_run(run_id)
        if record is None:
            console.print(f"[bold red]Run #{run_id} not found[/]")
            raise typer.Exit(code=1)
        _render_run_detail(record)
        hist.close()
        raise typer.Exit(code=0)

    # Default: list runs
    runs = hist.list_runs(limit=limit)
    if not runs:
        console.print("[dim]No runs recorded yet. Run tests with checkllm to start tracking.[/]")
        raise typer.Exit(code=0)

    table = Table(title="Run History")
    table.add_column("ID", justify="right")
    table.add_column("Time")
    table.add_column("Label")
    table.add_column("Commit")
    table.add_column("Checks", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Cost", justify="right")

    for r in runs:
        ts = datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d %H:%M")
        fail_style = "bold red" if r.failed_checks > 0 else "green"
        table.add_row(
            str(r.run_id),
            ts,
            r.label or "",
            r.git_commit or "",
            str(r.total_checks),
            f"[green]{r.passed_checks}[/]",
            f"[{fail_style}]{r.failed_checks}[/]",
            f"${r.total_cost:.4f}",
        )

    console.print(table)
    hist.close()


def _render_run_detail(record) -> None:
    """Render detailed view of a single run."""
    from rich.table import Table
    from rich.text import Text

    ts = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"\n[bold]Run #{record.run_id}[/] — {ts}")
    if record.label:
        console.print(f"  Label: {record.label}")
    if record.git_commit:
        console.print(f"  Commit: {record.git_commit}")
    console.print(
        f"  Checks: {record.total_checks} "
        f"([green]{record.passed_checks} passed[/], "
        f"[red]{record.failed_checks} failed[/]) "
        f"— ${record.total_cost:.4f}"
    )

    for test_name, checks in record.results.items():
        table = Table(title=test_name, show_lines=True)
        table.add_column("Status", width=6)
        table.add_column("Metric")
        table.add_column("Score", justify="right")
        table.add_column("Reasoning")
        table.add_column("Cost", justify="right")

        for c in checks:
            status = (
                Text("PASS", style="bold green") if c["passed"] else Text("FAIL", style="bold red")
            )
            table.add_row(
                status,
                c.get("metric_name", ""),
                f"{c.get('score', 0):.2f}",
                (c.get("reasoning", ""))[:80],
                f"${c.get('cost', 0):.4f}",
            )
        console.print(table)


def _render_run_comparison(run1, run2) -> None:
    """Render side-by-side comparison of two runs."""
    from rich.table import Table
    from rich.text import Text

    ts1 = datetime.fromtimestamp(run1.timestamp).strftime("%m-%d %H:%M")
    ts2 = datetime.fromtimestamp(run2.timestamp).strftime("%m-%d %H:%M")

    console.print(f"\n[bold]Comparing Run #{run1.run_id} vs Run #{run2.run_id}[/]")
    console.print(
        f"  Run #{run1.run_id}: {ts1} {run1.label or ''} ({run1.git_commit or 'no commit'})"
    )
    console.print(
        f"  Run #{run2.run_id}: {ts2} {run2.label or ''} ({run2.git_commit or 'no commit'})"
    )

    # Build lookup of test->metric->score for each run
    def _build_scores(results):
        scores = {}
        for test_name, checks in results.items():
            for c in checks:
                key = f"{test_name}::{c.get('metric_name', '')}"
                scores[key] = c
        return scores

    scores1 = _build_scores(run1.results)
    scores2 = _build_scores(run2.results)
    all_keys = sorted(set(scores1.keys()) | set(scores2.keys()))

    table = Table(title="Score Comparison", show_lines=True)
    table.add_column("Test :: Metric")
    table.add_column(f"#{run1.run_id}", justify="right")
    table.add_column(f"#{run2.run_id}", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status", width=10)

    for key in all_keys:
        c1 = scores1.get(key)
        c2 = scores2.get(key)
        s1 = c1["score"] if c1 else None
        s2 = c2["score"] if c2 else None

        s1_str = f"{s1:.3f}" if s1 is not None else "—"
        s2_str = f"{s2:.3f}" if s2 is not None else "—"

        if s1 is not None and s2 is not None:
            delta = s2 - s1
            delta_str = f"{delta:+.3f}"
            if delta > 0.01:
                status = Text("IMPROVED", style="bold green")
            elif delta < -0.01:
                status = Text("REGRESSED", style="bold red")
            else:
                status = Text("SAME", style="dim")
        elif s1 is None:
            delta_str = "new"
            status = Text("NEW", style="bold cyan")
        else:
            delta_str = "removed"
            status = Text("REMOVED", style="bold yellow")

        table.add_row(key, s1_str, s2_str, delta_str, status)

    console.print(table)

    # Summary
    cost_delta = run2.total_cost - run1.total_cost
    pass_delta = run2.passed_checks - run1.passed_checks
    console.print(f"\n  Cost: ${run1.total_cost:.4f} -> ${run2.total_cost:.4f} ({cost_delta:+.4f})")
    console.print(
        f"  Pass rate: {run1.passed_checks}/{run1.total_checks} -> "
        f"{run2.passed_checks}/{run2.total_checks} ({pass_delta:+d})"
    )


@app.command()
def cache(
    clear: bool = typer.Option(False, "--clear", help="Clear the entire cache"),
    stats: bool = typer.Option(False, "--stats", help="Show cache statistics"),
):
    """Manage the judge response cache."""
    from checkllm.cache import JudgeCache

    config = load_config()
    cache_obj = JudgeCache(
        db_path=Path(config.cache_dir) / "cache.db",
        ttl_seconds=config.cache_ttl_seconds,
        enabled=config.cache_enabled,
    )

    if clear:
        count = cache_obj.clear()
        console.print(f"[bold green]Cleared {count} cached entries.[/]")
    elif stats:
        s = cache_obj.stats()
        console.print("[bold]Cache Statistics[/]")
        console.print(f"  Enabled: {s['enabled']}")
        console.print(f"  Entries: {s['entries']}")
        console.print(f"  Size: {s['size_bytes'] / 1024:.1f} KB")
        console.print(f"  Total cached cost: ${s.get('total_cached_cost', 0):.4f}")
    else:
        console.print("[dim]Use --stats to view cache info, --clear to clear it.[/]")

    cache_obj.close()


@app.command()
def estimate(
    test_path: str = typer.Argument(help="Path to test file or directory"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model to estimate costs for"),
):
    """Estimate the cost of running checks before executing them."""
    from checkllm.estimator import estimate_from_test_file, CostEstimate

    path = Path(test_path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("test_*.py"))

    if not files:
        console.print("[yellow]No test files found.[/]")
        raise typer.Exit(code=0)

    total = CostEstimate(model=model)
    for f in files:
        est = estimate_from_test_file(str(f), model=model)
        total.deterministic_count += est.deterministic_count
        total.judge_count += est.judge_count
        total.total_cost += est.total_cost

    total.total_cost = round(total.total_cost, 4)
    console.print(f"\n[bold]{total.summary()}[/]")

    if total.judge_count > 0 and model != "gpt-4o-mini":
        mini_total = CostEstimate(model="gpt-4o-mini")
        for f in files:
            est = estimate_from_test_file(str(f), model="gpt-4o-mini")
            mini_total.judge_count += est.judge_count
            mini_total.deterministic_count += est.deterministic_count
            mini_total.total_cost += est.total_cost
        mini_total.total_cost = round(mini_total.total_cost, 4)
        console.print(f"[dim]With gpt-4o-mini: ~${mini_total.total_cost:.2f}[/]")

    console.print(f"\n[dim]Files scanned: {len(files)}[/]")


@app.command(name="eval-yaml")
def eval_yaml(
    config: str = typer.Argument(help="Path to YAML config file"),
    budget: Optional[float] = typer.Option(None, "--budget", help="Override budget (USD)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Parse config and show plan without running"
    ),
):
    """Run evaluation from a YAML configuration file.

    Supports promptfoo-style YAML configs with prompts, providers, test
    cases, and assertions.  Both deterministic and LLM-as-judge assertion
    types are available.

    Example::

        checkllm eval-yaml checkllm.yaml
        checkllm eval-yaml tests/eval.yml --budget 2.0
    """
    import asyncio as _asyncio

    from checkllm.yaml_eval import YAMLEvaluator

    evaluator = YAMLEvaluator()

    try:
        cfg = evaluator.load_config(config)
    except FileNotFoundError:
        console.print(f"[bold red]Config file not found: {config}[/]")
        raise typer.Exit(code=1)
    except ValueError as exc:
        console.print(f"[bold red]Invalid config: {exc}[/]")
        raise typer.Exit(code=1)

    if budget is not None:
        cfg.settings.budget = budget

    console.print(f"[bold]YAML Evaluation: {cfg.description or config}[/]")
    console.print(
        f"[dim]Tests: {len(cfg.tests)} | Prompts: {len(cfg.prompts) or 1} | Providers: {len(cfg.providers) or 1}[/]"
    )
    console.print(
        f"[dim]Judge: {cfg.judge.backend}{'/' + cfg.judge.model if cfg.judge.model else ''}[/]"
    )

    total_assertions = sum(len(t.assert_) for t in cfg.tests)
    total_combos = (len(cfg.prompts) or 1) * (len(cfg.providers) or 1)
    console.print(f"[dim]Total assertion runs: {total_assertions * total_combos}[/]")

    if cfg.settings.budget:
        console.print(f"[dim]Budget: ${cfg.settings.budget:.2f}[/]")

    if dry_run:
        console.print("\n[bold green]Dry run complete. Config is valid.[/]")
        raise typer.Exit(code=0)

    console.print()

    try:
        loop = _asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(_asyncio.run, evaluator.run_from_config(cfg)).result()
    else:
        result = _asyncio.run(evaluator.run_from_config(cfg))

    console.print(result.summary())
    console.print()
    console.print(f"[dim]Total cost: ${result.cost:.4f} | Duration: {result.duration_ms:.0f}ms[/]")

    if result.failed > 0:
        console.print(f"\n[bold red]{result.failed} assertion(s) failed[/]")
        raise typer.Exit(code=1)
    else:
        console.print(f"\n[bold green]All {result.passed} assertion(s) passed[/]")
        raise typer.Exit(code=0)


@app.command()
def init(
    path: str = typer.Argument(".", help="Directory to initialize"),
    use_case: Optional[str] = typer.Option(
        None,
        "--use-case",
        "-u",
        help="What you're building: rag, chatbot, agent, general",
    ),
    ci: bool = typer.Option(False, "--ci", help="Also generate GitHub Actions workflow"),
):
    """Scaffold a new checkllm project with tailored test files."""
    from checkllm.discovery import detect_judge_backend

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)

    # Detect available judge
    detected = detect_judge_backend()
    if detected:
        backend, model = detected
        console.print(f"[green]Detected judge backend:[/] {backend} ({model})")
    else:
        backend, model = "openai", "gpt-4o"
        console.print(
            "[yellow]No API key or Ollama detected. Generating deterministic-only tests.[/]"
        )
        console.print("[dim]Set OPENAI_API_KEY or start Ollama to enable LLM-as-judge checks.[/]")

    # If no use_case provided, default to "general" (non-interactive for CLI testing)
    if use_case is None:
        use_case = "general"

    # Determine template
    template_map = {
        "rag": "test_rag.py.tmpl",
        "chatbot": "test_chatbot.py.tmpl",
        "agent": "test_agent.py.tmpl",
        "general": "test_general.py.tmpl",
    }
    template_name = template_map.get(use_case, "test_general.py.tmpl")
    template_dir = Path(__file__).parent / "templates" / "init"
    template_path = template_dir / template_name

    # If no API key detected and not general, fall back to general (deterministic-only)
    if not detected and use_case != "general":
        console.print("[yellow]No API key found — generating deterministic-only tests.[/]")
        template_path = template_dir / "test_general.py.tmpl"

    # Write pyproject.toml config
    pyproject = target / "pyproject.toml"
    checkllm_config = (
        f"\n[tool.checkllm]\n"
        f'judge_backend = "{backend}"\n'
        f'judge_model = "{model}"\n'
        f"default_threshold = 0.8\n"
        f"cache_enabled = true\n"
        f"max_concurrency = 10\n"
    )
    if pyproject.exists():
        content = pyproject.read_text()
        if "[tool.checkllm]" not in content:
            pyproject.write_text(content + checkllm_config)
            console.print(f"[green]Updated {pyproject} with [tool.checkllm] config[/]")
        else:
            console.print(f"[dim]{pyproject} already has [tool.checkllm] section[/]")
    else:
        full_pyproject = (
            '[project]\nname = "my-project"\nversion = "0.1.0"\n'
            "dependencies = []\n\n"
            "[project.optional-dependencies]\n"
            'dev = [\n    "checkllm",\n    "pytest",\n]\n\n'
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n' + checkllm_config
        )
        pyproject.write_text(full_pyproject)
        console.print(f"[green]Created {pyproject}[/]")

    # Write test file
    tests_dir = target / "tests"
    tests_dir.mkdir(exist_ok=True)

    sample_test = tests_dir / "test_llm_example.py"
    if not sample_test.exists():
        if template_path.exists():
            sample_test.write_text(template_path.read_text())
        else:
            # Fallback if templates not found
            sample_test.write_text(
                '"""checkllm test file."""\n\n\n'
                "def test_output_quality(check):\n"
                '    output = "Python is a high-level programming language."\n'
                '    check.contains(output, "Python")\n'
                "    check.max_tokens(output, limit=200)\n"
            )
        console.print(f"[green]Created {sample_test}[/]")
    else:
        console.print(f"[dim]{sample_test} already exists — skipping[/]")

    # Write conftest.py
    conftest = tests_dir / "conftest.py"
    if not conftest.exists():
        conftest.write_text(
            '"""checkllm configuration.\n\n'
            "The check fixture is auto-discovered by the checkllm pytest plugin.\n"
            '"""\n'
        )
        console.print(f"[green]Created {conftest}[/]")

    # Write sample dataset
    fixtures_dir = tests_dir / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    sample_dataset = fixtures_dir / "cases.yaml"
    if not sample_dataset.exists():
        sample_dataset.write_text(
            "# Sample dataset for checkllm\n"
            '- input: "What is Python?"\n'
            '  expected: "Python is a programming language"\n'
            '  query: "Explain Python"\n'
            '  criteria: "accurate, concise"\n'
        )
        console.print(f"[green]Created {sample_dataset}[/]")

    # Create .checkllm directory
    checkllm_dir = target / ".checkllm" / "snapshots"
    checkllm_dir.mkdir(parents=True, exist_ok=True)
    gitkeep = checkllm_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    # CI workflow
    if ci:
        ci_dir = target / ".github" / "workflows"
        ci_dir.mkdir(parents=True, exist_ok=True)
        ci_file = ci_dir / "checkllm.yml"
        ci_template = template_dir / "checkllm_ci.yml.tmpl"
        if ci_template.exists():
            ci_file.write_text(ci_template.read_text())
        else:
            ci_file.write_text(
                "name: checkllm\non:\n  pull_request:\n    branches: [main]\n"
                "jobs:\n  eval:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - uses: actions/checkout@v4\n"
                "      - uses: actions/setup-python@v5\n"
                '        with:\n          python-version: "3.12"\n'
                "      - run: pip install checkllm[all] pytest\n"
                "      - run: checkllm run tests/ --budget 5.0\n"
                "        env:\n          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}\n"
            )
        console.print(f"[green]Created {ci_file}[/]")

    console.print(
        f"\n[bold green]checkllm initialized![/] ({use_case} template)\n\n"
        f"  Run tests:     [cyan]pytest tests/ -v[/]\n"
        f"  Estimate cost: [cyan]checkllm estimate tests/[/]\n"
        f"  HTML report:   [cyan]checkllm report tests/[/]\n"
        f"  View metrics:  [cyan]checkllm list-metrics[/]\n"
    )


@app.command()
def watch(
    test_path: str = typer.Argument(help="Path to test directory or file"),
    watch_path: Optional[list[str]] = typer.Option(
        None, "--watch", "-w", help="Additional paths to watch"
    ),
    interval: float = typer.Option(1.0, "--interval", "-i", help="Poll interval in seconds"),
    debounce: float = typer.Option(0.5, "--debounce", help="Debounce delay in seconds"),
    pattern: Optional[list[str]] = typer.Option(
        None, "--pattern", "-p", help="File patterns to watch"
    ),
    budget: Optional[float] = typer.Option(None, "--budget", help="Budget per run"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Config profile to use"),
):
    """Watch for file changes and re-run tests automatically."""
    from checkllm.watcher import WatchRunner

    env_overrides: dict[str, str] = {}
    if budget is not None:
        env_overrides["CHECKLLM_BUDGET"] = str(budget)
    if no_cache:
        env_overrides["CHECKLLM_CACHE_ENABLED"] = "false"
    if profile:
        env_overrides["CHECKLLM_PROFILE"] = profile

    patterns = pattern if pattern else None

    runner = WatchRunner(
        test_path=test_path,
        watch_paths=watch_path,
        poll_interval=interval,
        debounce=debounce,
        patterns=patterns,
        env_overrides=env_overrides,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        runner.stop()
        console.print("\n[bold]Stopped.[/]")


@app.command(name="list-metrics")
def list_metrics(
    installed: bool = typer.Option(
        False, "--installed", help="Show only installed/available metrics"
    ),
):
    """List all available metrics and checks."""
    from rich.table import Table

    deterministic_checks = [
        ("contains", "Check substring presence"),
        ("not_contains", "Check substring absence"),
        ("exact_match", "Exact string match"),
        ("starts_with", "Check prefix"),
        ("ends_with", "Check suffix"),
        ("regex", "Regular expression match"),
        ("similarity", "Levenshtein similarity"),
        ("max_tokens", "Maximum token count"),
        ("min_tokens", "Minimum token count"),
        ("word_count", "Word count range"),
        ("char_count", "Character count range"),
        ("sentence_count", "Sentence count range"),
        ("is_json", "Valid JSON check"),
        ("is_valid_python", "Valid Python syntax"),
        ("json_schema", "Pydantic schema validation"),
        ("json_field", "JSON field extraction"),
        ("is_valid_sql", "Valid SQL syntax"),
        ("is_valid_markdown", "Valid Markdown"),
        ("readability", "Flesch-Kincaid grade level"),
        ("language", "Language detection"),
        ("bleu", "BLEU score"),
        ("rouge_l", "ROUGE-L score"),
        ("all_of", "All substrings present"),
        ("any_of", "Any substring present"),
        ("none_of", "No substrings present"),
        ("no_pii", "PII detection"),
        ("greater_than", "Numeric > threshold"),
        ("less_than", "Numeric < threshold"),
        ("between", "Numeric in range"),
        ("latency", "Response time check"),
        ("cost", "API cost check"),
        ("is_yaml", "Valid YAML check"),
        ("is_url", "Output is a valid URL"),
        ("has_url", "Output contains a valid URL"),
    ]

    llm_metrics = [
        ("hallucination", "Faithfulness to context"),
        ("relevance", "Query-output relevance"),
        ("faithfulness", "RAG answer faithfulness"),
        ("context_relevance", "Retrieved context relevance"),
        ("answer_completeness", "Answer completeness"),
        ("groundedness", "Claim-by-claim grounding"),
        ("contextual_precision", "Document ranking quality"),
        ("contextual_recall", "Ground truth coverage"),
        ("toxicity", "Harmful content detection"),
        ("bias", "Demographic/cultural bias"),
        ("fluency", "Writing quality"),
        ("coherence", "Logical consistency"),
        ("correctness", "Semantic correctness"),
        ("consistency", "Multi-output consistency"),
        ("instruction_following", "Instruction compliance"),
        ("summarization", "Summary quality"),
        ("sentiment", "Tone/mood assessment"),
        ("rubric", "Custom criteria evaluation"),
        ("g_eval", "Chain-of-thought evaluation"),
        ("task_completion", "Goal accomplishment"),
        ("role_adherence", "Persona consistency"),
        ("tool_accuracy", "Agent tool selection"),
        ("knowledge_retention", "Conversation memory"),
        ("conversation_completeness", "Multi-turn fulfillment"),
    ]

    # Deterministic table
    table_det = Table(title="Deterministic Checks (free, instant, no API key)")
    table_det.add_column("Check", style="cyan")
    table_det.add_column("Description")
    for name, desc in deterministic_checks:
        table_det.add_row(name, desc)
    console.print(table_det)
    console.print()

    # LLM metrics table
    table_llm = Table(title="LLM-as-Judge Metrics (requires API key)")
    table_llm.add_column("Metric", style="green")
    table_llm.add_column("Description")
    for name, desc in llm_metrics:
        table_llm.add_row(name, desc)
    console.print(table_llm)

    # Plugin metrics
    from checkllm.metrics import _global_registry

    _global_registry.load_entry_points()
    plugins = [m for m in _global_registry.list_metrics_detailed() if m["source"] != "local"]
    if plugins:
        console.print()
        table_plug = Table(title="Plugin Metrics (community)")
        table_plug.add_column("Metric", style="magenta")
        table_plug.add_column("Source")
        for m in plugins:
            table_plug.add_row(m["name"], m["source"])
        console.print(table_plug)
    elif not installed:
        console.print("\n[dim]No plugin metrics installed. See docs for creating plugins.[/]")


@app.command()
def dashboard(
    port: int = typer.Option(8484, "--port", "-p", help="Port to serve on"),
    host: str = typer.Option("localhost", "--host", help="Host to bind to"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Experiments database path"),
):
    """Launch the interactive web dashboard."""
    from checkllm.dashboard import start_dashboard

    db = db_path or ".checkllm/experiments.db"
    console.print(f"[bold]Starting dashboard on {host}:{port}[/]")
    start_dashboard(port=port, host=host, open_browser=not no_browser, db_path=db)


@app.command(name="yaml-run")
def yaml_run(
    config_path: str = typer.Argument(help="Path to YAML config file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save results JSON"),
):
    """Run evaluations defined in a YAML config file."""
    import asyncio
    import json as _json
    from checkllm.yaml_config import load_eval_config, YamlEvalRunner
    from rich.table import Table
    from rich.text import Text

    config = load_eval_config(config_path)
    console.print(f"[bold]Running YAML evaluation:[/] {config.description or config_path}")
    console.print(
        f"  Providers: {len(config.providers)}, Prompts: {len(config.prompts)}, Tests: {len(config.tests)}"
    )

    runner = YamlEvalRunner(config)
    results = asyncio.run(runner.run())

    table = Table(title="Results", show_lines=True)
    table.add_column("Provider")
    table.add_column("Prompt")
    table.add_column("Test")
    table.add_column("Assertion")
    table.add_column("Status", width=6)
    table.add_column("Score", justify="right")

    for result in results:
        for check in result.get("results", []):
            status = (
                Text("PASS", style="bold green")
                if check.get("passed")
                else Text("FAIL", style="bold red")
            )
            table.add_row(
                result.get("provider", ""),
                result.get("prompt", ""),
                result.get("test", "")[:30],
                check.get("metric_name", ""),
                status,
                f"{check.get('score', 0):.2f}",
            )
    console.print(table)

    if output:
        Path(output).write_text(_json.dumps(results, indent=2, default=str))
        console.print(f"\n[bold green]Results saved to {output}[/]")


@app.command()
def redteam(
    target_prompt: str = typer.Argument(help="System prompt for target LLM"),
    attacks_per_type: int = typer.Option(3, "--attacks", "-n", help="Attacks per type"),
    types: Optional[list[str]] = typer.Option(None, "--type", "-t", help="Vuln types to test"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save report JSON"),
):
    """Run automated red teaming against an LLM."""
    import asyncio
    from checkllm.redteam import RedTeamer, VulnerabilityType

    vuln_types = None
    if types:
        try:
            vuln_types = [VulnerabilityType(t) for t in types]
        except ValueError as e:
            console.print(f"[bold red]Invalid type: {e}[/]")
            console.print(f"[dim]Valid: {', '.join(v.value for v in VulnerabilityType)}[/]")
            raise typer.Exit(code=1)

    console.print(f"[bold]Red Team Scan[/] ({attacks_per_type} attacks/type)")

    async def target(prompt: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": target_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    red = RedTeamer()
    report = asyncio.run(
        red.scan(
            target=target,
            vulnerability_types=vuln_types,
            attacks_per_type=attacks_per_type,
            system_prompt=target_prompt,
        )
    )
    console.print(report.summary())

    if output:
        Path(output).write_text(report.model_dump_json(indent=2))
        console.print(f"\n[bold green]Report saved to {output}[/]")

    if report.successful_attacks > 0:
        raise typer.Exit(code=1)


@app.command()
def experiments(
    list_all: bool = typer.Option(False, "--list", "-l", help="List experiment runs"),
    compare_runs: Optional[str] = typer.Option(None, "--compare", help="Compare 'ID1,ID2'"),
    best: Optional[str] = typer.Option(None, "--best", help="Best run for experiment"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of runs"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path"),
):
    """View and compare experiment tracking data."""
    from rich.table import Table
    from checkllm.experiments import ExperimentTracker

    db = db_path or ".checkllm/experiments.db"
    tracker = ExperimentTracker(db_path=db)

    if compare_runs:
        parts = compare_runs.split(",")
        if len(parts) != 2:
            console.print("[bold red]--compare format: 'ID1,ID2'[/]")
            raise typer.Exit(code=1)
        comp = tracker.compare(parts[0].strip(), parts[1].strip())
        console.print(f"  Score diff: {comp.score_diff:+.3f}")
        console.print(f"  Pass rate diff: {comp.pass_rate_diff:+.3f}")
        console.print(f"  Cost diff: ${comp.cost_diff:+.4f}")
        if comp.improved_metrics:
            console.print(f"  [green]Improved: {', '.join(comp.improved_metrics)}[/]")
        if comp.degraded_metrics:
            console.print(f"  [red]Degraded: {', '.join(comp.degraded_metrics)}[/]")
        raise typer.Exit(code=0)

    if best:
        run = tracker.best_run(best)
        if run:
            console.print(f"[bold]Best run for '{best}':[/] {run.run_id}")
            console.print(f"  Score: {run.avg_score:.3f}, Pass rate: {run.pass_rate:.1%}")
        else:
            console.print(f"[dim]No runs found for '{best}'[/]")
        raise typer.Exit(code=0)

    runs = tracker.list_runs(limit=limit)
    if not runs:
        console.print("[dim]No experiment runs recorded yet.[/]")
        raise typer.Exit(code=0)

    table = Table(title="Experiment Runs")
    table.add_column("ID", max_width=10)
    table.add_column("Experiment")
    table.add_column("Model")
    table.add_column("Prompt Ver.")
    table.add_column("Score", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Cost", justify="right")

    for r in runs:
        table.add_row(
            r.run_id[:8],
            r.experiment_name,
            r.model,
            r.prompt_version,
            f"{r.avg_score:.3f}",
            f"{r.pass_rate:.0%}",
            f"${r.total_cost:.4f}",
        )
    console.print(table)


@app.command("ci-gitlab-template")
def ci_gitlab_template(
    eval_command: str = typer.Option(
        "checkllm ci tests/",
        "--eval-command",
        help="Shell command executed by the job.",
    ),
    python_version: str = typer.Option(
        "3.11",
        "--python-version",
        help="Python version tag for the Docker image.",
    ),
    budget: Optional[float] = typer.Option(
        None, "--budget", help="Maximum USD spend appended to the command."
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write YAML to this path instead of stdout.",
    ),
):
    """Print a ready-to-paste ``.gitlab-ci.yml`` job for checkllm."""
    from checkllm.ci.gitlab import gitlab_template

    yaml = gitlab_template(
        eval_command=eval_command,
        python_version=python_version,
        budget=budget,
    )
    if output:
        Path(output).write_text(yaml, encoding="utf-8")
        console.print(f"[green]Wrote GitLab CI template to {output}[/]")
    else:
        print(yaml, end="")


@app.command()
def ci(
    test_path: str = typer.Argument("tests/", help="Path to test directory or file"),
    fail_on_regression: bool = typer.Option(
        False, "--fail-on-regression", help="Exit 1 if regression detected"
    ),
    compare: Optional[str] = typer.Option(
        None, "--compare", help="Branch to compare against (snapshot baseline)"
    ),
    budget: Optional[float] = typer.Option(None, "--budget", help="Maximum USD to spend"),
    no_comment: bool = typer.Option(False, "--no-comment", help="Skip posting PR comment"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Config profile to use"),
):
    """Run tests in CI and post results as a PR comment.

    Auto-detects GitHub Actions environment (GITHUB_TOKEN, PR number).
    Falls back to normal test run when not in GitHub Actions.
    """
    import os
    import json

    from checkllm.ci import gitlab as gitlab_ci

    # Detect GitHub Actions environment
    github_token = os.environ.get("GITHUB_TOKEN")
    github_event_path = os.environ.get("GITHUB_EVENT_PATH")
    github_repository = os.environ.get("GITHUB_REPOSITORY")

    pr_number = None
    if github_event_path and Path(github_event_path).exists():
        try:
            event = json.loads(Path(github_event_path).read_text())
            pr_number = event.get("pull_request", {}).get("number")
        except (json.JSONDecodeError, KeyError):
            pass

    is_github_ci = bool(github_token and github_repository)

    gitlab_context = gitlab_ci.context_from_env()
    is_gitlab_ci = gitlab_context is not None

    if is_github_ci:
        console.print("[bold]checkllm CI[/] — detected GitHub Actions")
        console.print(f"[dim]Repository: {github_repository}[/]")
        if pr_number:
            console.print(f"[dim]PR: #{pr_number}[/]")
    elif gitlab_context is not None:
        console.print("[bold]checkllm CI[/] — detected GitLab CI")
        console.print(f"[dim]Project: {gitlab_context.project_id}[/]")
        if gitlab_context.mr_iid:
            console.print(f"[dim]MR: !{gitlab_context.mr_iid}[/]")
    else:
        console.print("[bold]checkllm CI[/] — running locally (no GitHub Actions detected)")

    # Build pytest command
    snapshot_path = ".checkllm/ci_snapshot.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-v",
        f"--checkllm-snapshot={snapshot_path}",
    ]

    # Environment overrides
    env = os.environ.copy()
    if budget is not None:
        env["CHECKLLM_BUDGET"] = str(budget)
    if profile:
        env["CHECKLLM_PROFILE"] = profile

    # Run tests
    result = subprocess.run(cmd, env=env)

    # Post PR comment if in GitHub Actions
    if is_github_ci and pr_number and github_repository and not no_comment:
        try:
            from checkllm.pytest_plugin import get_session_results  # noqa: F401
            from checkllm.reporting.github import generate_pr_comment, post_pr_comment

            # Load results from snapshot
            snapshot_file = Path(snapshot_path)
            if snapshot_file.exists():
                from checkllm.regression.snapshot import load_snapshot

                snap = load_snapshot(snapshot_file)

                # Convert snapshot to CheckResult-like format for comment generation
                from checkllm.models import CheckResult

                results: dict[str, list[CheckResult]] = {}
                for test_name, runs in snap.tests.items():
                    checks = []
                    for run in runs:
                        for metric_name, metric in run.metrics.items():
                            checks.append(
                                CheckResult(
                                    passed=metric.passed,
                                    score=metric.score,
                                    reasoning="",
                                    cost=0.0,
                                    latency_ms=0,
                                    metric_name=metric_name,
                                )
                            )
                    results[test_name] = checks

                comment = generate_pr_comment(results)
                post_pr_comment(
                    comment,
                    repo=github_repository,
                    pr_number=pr_number,
                    token=github_token,
                )
                console.print(f"[green]Posted results to PR #{pr_number}[/]")
            else:
                console.print("[yellow]No snapshot generated — skipping PR comment[/]")
        except ImportError as exc:
            console.print(f"[yellow]Could not post PR comment: {exc}[/]")
            console.print("[dim]Install httpx for PR comments: pip install httpx[/]")
        except Exception as exc:
            console.print(f"[yellow]Failed to post PR comment: {exc}[/]")

    # Post MR comment if in GitLab CI
    if is_gitlab_ci and gitlab_context is not None and gitlab_context.mr_iid and not no_comment:
        try:
            snapshot_file = Path(snapshot_path)
            if snapshot_file.exists():
                from checkllm.regression.snapshot import load_snapshot

                snap = load_snapshot(snapshot_file)
                from checkllm.models import CheckResult

                mr_results: dict[str, list[CheckResult]] = {}
                for test_name, runs in snap.tests.items():
                    mr_checks: list[CheckResult] = []
                    for run in runs:
                        for metric_name, metric in run.metrics.items():
                            mr_checks.append(
                                CheckResult(
                                    passed=metric.passed,
                                    score=metric.score,
                                    reasoning="",
                                    cost=0.0,
                                    latency_ms=0,
                                    metric_name=metric_name,
                                )
                            )
                    mr_results[test_name] = mr_checks

                body = gitlab_ci.format_mr_comment(mr_results)
                if gitlab_ci.post_mr_comment(body, ctx=gitlab_context):
                    console.print(f"[green]Posted results to MR !{gitlab_context.mr_iid}[/]")
                else:
                    if gitlab_context.token_is_job_token:
                        console.print(
                            "[yellow]Could not post MR note with CI_JOB_TOKEN; "
                            "set GITLAB_TOKEN or CHECKLLM_GITLAB_TOKEN[/]"
                        )
                    else:
                        console.print("[yellow]Failed to post MR comment (check token/scopes)[/]")
            else:
                console.print("[yellow]No snapshot generated — skipping MR comment[/]")
        except Exception as exc:
            console.print(f"[yellow]Failed to post MR comment: {exc}[/]")

    # Regression comparison
    if compare:
        baseline_path = f".checkllm/snapshots/{compare}.json"
        if Path(baseline_path).exists() and Path(snapshot_path).exists():
            _run_comparison(baseline_path, snapshot_path, fail_on_regression)
        elif fail_on_regression:
            console.print(f"[bold red]Baseline not found at {baseline_path}[/]")
            console.print(
                "[red]Cannot verify regressions without a baseline. "
                "Create one with: checkllm snapshot tests/ --output "
                f"{baseline_path}[/]"
            )
            raise typer.Exit(code=1)
        else:
            console.print(f"[yellow]Baseline not found at {baseline_path} — skipping comparison[/]")

    raise typer.Exit(code=result.returncode)


def _write_cases(cases, output: Path) -> None:
    """Serialise ``cases`` to YAML or JSON based on ``output`` extension."""
    import json as _json

    import yaml as _yaml

    records = [c.model_dump() for c in cases]
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix in (".yaml", ".yml"):
        with open(output, "w", encoding="utf-8") as f:
            _yaml.safe_dump(records, f, sort_keys=False, allow_unicode=True)
    elif suffix == ".json":
        with open(output, "w", encoding="utf-8") as f:
            _json.dump(records, f, indent=2, default=str)
    else:
        raise typer.BadParameter(f"Unsupported output format: {output.suffix}. Use .yaml or .json.")


@dataset_app.command("load")
def dataset_load(
    name: str = typer.Argument(help="Hugging Face dataset identifier, e.g. 'squad'"),
    split: str = typer.Option("test", "--split", help="Dataset split to load"),
    config: Optional[str] = typer.Option(None, "--config", help="Dataset config/subset"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum number of rows"),
    streaming: bool = typer.Option(False, "--streaming", help="Use streaming mode"),
    output: str = typer.Option(
        "dataset.yaml", "--output", "-o", help="Output file (.yaml or .json)"
    ),
    field_map: Optional[str] = typer.Option(
        None,
        "--field-map",
        help='JSON mapping HF columns to Case fields, e.g. \'{"question": "input"}\'',
    ),
    dataset_id: Optional[str] = typer.Option(
        None, "--dataset-id", help="Register the load under this lineage id"
    ),
):
    """Load a Hugging Face dataset and save it as YAML/JSON cases."""
    import json as _json

    from checkllm.datasets.huggingface import load_hf_dataset

    parsed_map: Optional[dict[str, str]] = None
    if field_map:
        try:
            parsed_map = _json.loads(field_map)
        except _json.JSONDecodeError as exc:
            console.print(f"[bold red]Invalid --field-map JSON: {exc}[/]")
            raise typer.Exit(code=1)

    try:
        cases = load_hf_dataset(
            name,
            split=split,
            config=config,
            streaming=streaming,
            limit=limit,
            field_map=parsed_map,
        )
    except ImportError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1)
    except Exception as exc:  # pragma: no cover - surfaces HF errors
        console.print(f"[bold red]Failed to load dataset: {exc}[/]")
        raise typer.Exit(code=1)

    output_path = Path(output)
    _write_cases(cases, output_path)
    console.print(f"[bold green]Wrote {len(cases)} cases to {output_path}[/]")

    if dataset_id:
        from checkllm.datasets.lineage import LineageStore

        store = LineageStore()
        version = store.register(cases, dataset_id=dataset_id, source=f"hf:{name}")
        console.print(f"[dim]Registered lineage {dataset_id}:{version.version}[/]")

    raise typer.Exit(code=0)


@dataset_app.command("split")
def dataset_split(
    input: str = typer.Argument(help="Path to dataset file (YAML, JSON, CSV)"),
    test_size: float = typer.Option(
        0.2, "--test-size", help="Fraction of cases for the test split"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    stratify_by: Optional[str] = typer.Option(None, "--stratify-by", help="Field to stratify on"),
    train: str = typer.Option("train.yaml", "--train", help="Train split output path"),
    test: str = typer.Option("test.yaml", "--test", help="Test split output path"),
):
    """Split a dataset into reproducible train/test files."""
    from checkllm.datasets.loader import load_dataset
    from checkllm.datasets.splits import train_test_split as _split

    cases = load_dataset(Path(input))
    train_cases, test_cases = _split(cases, test_size=test_size, seed=seed, stratify_by=stratify_by)
    _write_cases(train_cases, Path(train))
    _write_cases(test_cases, Path(test))
    console.print(
        f"[bold green]Split {len(cases)} cases → {len(train_cases)} train / "
        f"{len(test_cases)} test[/]"
    )
    raise typer.Exit(code=0)


@dataset_app.command("versions")
def dataset_versions(
    dataset_id: str = typer.Argument(help="Dataset lineage id to inspect"),
):
    """List tracked versions for a dataset."""
    from rich.table import Table

    from checkllm.datasets.lineage import LineageStore

    store = LineageStore()
    versions = store.list_versions(dataset_id)
    if not versions:
        console.print(f"[dim]No versions recorded for {dataset_id}[/]")
        raise typer.Exit(code=0)

    table = Table(title=f"Versions: {dataset_id}")
    table.add_column("Version")
    table.add_column("Created")
    table.add_column("Cases", justify="right")
    table.add_column("Hash")
    table.add_column("Source")
    for v in versions:
        table.add_row(
            v.version,
            v.created_at,
            str(v.num_cases),
            v.content_hash[:12],
            v.source or "",
        )
    console.print(table)
    raise typer.Exit(code=0)


@dataset_app.command("diff")
def dataset_diff(
    v1: str = typer.Argument(help="Baseline version ('dataset_id:vN' or 'vN' with --dataset-id)"),
    v2: str = typer.Argument(help="Target version"),
    dataset_id: Optional[str] = typer.Option(
        None, "--dataset-id", help="Dataset id (required when v1/v2 are bare)"
    ),
):
    """Show added/removed/modified cases between two versions."""
    from checkllm.datasets.lineage import LineageStore

    store = LineageStore()
    try:
        result = store.diff(v1, v2, dataset_id=dataset_id)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Diff {result.v1} -> {result.v2}[/]")
    console.print(f"  Added:    {result.added}")
    console.print(f"  Removed:  {result.removed}")
    console.print(f"  Modified: {result.modified}")
    console.print(f"  Hash:     {result.v1_hash[:12]} -> {result.v2_hash[:12]}")
    raise typer.Exit(code=0)


@app.command("generate-dataset")
def generate_dataset(
    source: str = typer.Argument(
        help="File, directory, or glob pointing at source documents (.md, .txt, .pdf)."
    ),
    num_cases: int = typer.Option(20, "--num-cases", "-n", help="How many cases to generate."),
    output: Path = typer.Option(
        Path("dataset.yaml"), "--output", "-o", help="Where to write the generated dataset."
    ),
    glob: str = typer.Option("**/*.md", "--glob", help="Glob pattern when SOURCE is a directory."),
    simple: float = typer.Option(0.4, "--simple", help="Fraction of single-hop queries."),
    reasoning: float = typer.Option(
        0.3, "--reasoning", help="Fraction of multi-hop reasoning queries."
    ),
    multi_context: float = typer.Option(
        0.2, "--multi-context", help="Fraction of multi-context synthesis queries."
    ),
    conditional: float = typer.Option(
        0.1, "--conditional", help="Fraction of conditional / hypothetical queries."
    ),
    personas: Optional[str] = typer.Option(
        None, "--personas", help="Comma-separated persona labels (e.g. novice,expert,skeptic)."
    ),
    chunk_size: int = typer.Option(1000, "--chunk-size", help="Chunk size in characters."),
    chunk_overlap: int = typer.Option(100, "--chunk-overlap", help="Chunk overlap in characters."),
    judge_backend: Optional[str] = typer.Option(
        None,
        "--judge",
        help="Judge backend name. Defaults to auto-detection from env.",
    ),
):
    """Generate a RAG evaluation dataset from documents (Ragas-parity)."""
    import asyncio
    import json

    import yaml

    from checkllm.discovery import detect_judge_backend
    from checkllm.providers import create_judge
    from checkllm.rag_dataset import QueryDistribution, RAGDatasetGenerator

    distribution = QueryDistribution(
        simple=simple,
        reasoning=reasoning,
        multi_context=multi_context,
        conditional=conditional,
    )

    if judge_backend:
        backend_name = judge_backend
        model = None
    else:
        detected = detect_judge_backend()
        if detected is None:
            console.print(
                "[bold red]No judge backend available.[/] Install an optional dep "
                "(e.g. `pip install checkllm[openai]`) and set the matching API key, "
                "or pass --judge <backend>."
            )
            raise typer.Exit(code=1)
        backend_name, model = detected

    judge_kwargs: dict = {}
    if model is not None:
        judge_kwargs["model"] = model
    judge = create_judge(backend_name, **judge_kwargs)

    gen = RAGDatasetGenerator(judge=judge)

    source_path = Path(source)
    persona_list = [p.strip() for p in personas.split(",")] if personas else None

    async def _run():
        if source_path.is_dir():
            return await gen.from_directory(
                source_path,
                glob=glob,
                num_cases=num_cases,
                query_distribution=distribution,
                personas=persona_list,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            return await gen.from_pdf(
                source_path,
                num_cases=num_cases,
                query_distribution=distribution,
                personas=persona_list,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        if suffix == ".md":
            return await gen.from_markdown_files(
                [source_path],
                num_cases=num_cases,
                query_distribution=distribution,
                personas=persona_list,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        return await gen.from_text_files(
            [source_path],
            num_cases=num_cases,
            query_distribution=distribution,
            personas=persona_list,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    cases = asyncio.run(_run())

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cases": [c.model_dump(exclude_none=True) for c in cases]}
    if output.suffix.lower() in {".json", ".jsonl"}:
        if output.suffix.lower() == ".jsonl":
            with output.open("w", encoding="utf-8") as fh:
                for case in payload["cases"]:
                    fh.write(json.dumps(case, ensure_ascii=False) + "\n")
        else:
            output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    console.print(f"[bold green]Generated {len(cases)} cases[/] -> {output}")
    raise typer.Exit(code=0)


@app.command("export")
def export_cmd(
    input_path: Path = typer.Argument(
        help="Path to a JSONL results file produced by `checkllm run` / `eval`."
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output path. Format inferred from suffix if --format omitted."
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Output format: csv, parquet, json, jsonl. Inferred from --output suffix if absent.",
    ),
    compression: str = typer.Option(
        "snappy",
        "--compression",
        help="Parquet compression codec: snappy, gzip, zstd, or none.",
    ),
):
    """Export results from a JSONL results file to CSV, Parquet, JSON, or JSONL."""
    import json as _json

    from checkllm.models import CheckResult
    from checkllm.reporting.bulk_export import export_results as bulk_export

    if not input_path.exists():
        console.print(f"[bold red]No such file:[/] {input_path}")
        raise typer.Exit(code=1)

    results: dict[str, list[CheckResult]] = {}
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            test_name = rec.pop("test_name", "unknown")
            results.setdefault(test_name, []).append(CheckResult(**rec))

    if format == "parquet" or (format is None and output.suffix.lower() == ".parquet"):
        from checkllm.reporting.parquet_export import write_parquet

        write_parquet(results, output, compression=compression)
        summary_fmt = "parquet"
        rows = sum(len(v) for v in results.values())
    else:
        summary = bulk_export(results, output, format=format)
        summary_fmt = summary.format
        rows = summary.row_count

    console.print(f"[bold green]Exported[/] {rows} rows to {output} ({summary_fmt})")
    raise typer.Exit(code=0)


@app.command("validate-config")
def validate_config_cmd(
    config_path: Path = typer.Argument(
        Path("pyproject.toml"),
        help="Path to pyproject.toml or checkllm.yaml. Defaults to pyproject.toml in cwd.",
    ),
    write_schema: Optional[Path] = typer.Option(
        None,
        "--write-schema",
        help="Also write the JSON schema to this path (useful for editor IntelliSense).",
    ),
):
    """Validate checkllm configuration against the JSON Schema."""
    import tomllib as _toml

    import yaml as _yaml

    from checkllm.config_schema import (
        generate_schema_to_file,
        validate_config as _validate_config,
    )

    if not config_path.exists():
        console.print(f"[bold red]No such file:[/] {config_path}")
        raise typer.Exit(code=1)

    data: dict
    if config_path.name == "pyproject.toml" or config_path.suffix.lower() == ".toml":
        with config_path.open("rb") as fh:
            parsed = _toml.load(fh)
        data = parsed.get("tool", {}).get("checkllm", {}) or {}
    else:
        with config_path.open("r", encoding="utf-8") as fh:
            data = _yaml.safe_load(fh) or {}

    errors = _validate_config(data)

    if write_schema is not None:
        generate_schema_to_file(write_schema)
        console.print(f"Wrote schema to {write_schema}")

    if not errors:
        console.print(f"[bold green]OK[/] — {config_path} is valid.")
        raise typer.Exit(code=0)

    console.print(f"[bold red]{len(errors)} validation error(s)[/]:")
    for err in errors:
        location = err.path or "<root>"
        console.print(f"  [{err.severity}] {location}: {err.message}")
    raise typer.Exit(code=1)


@app.command("batch")
def batch_cmd(
    dataset_path: Path = typer.Option(
        ..., "--dataset", "-d", help="Path to dataset file (YAML, JSON, CSV) with prompts."
    ),
    provider: str = typer.Option(
        "openai",
        "--batch",
        "-b",
        help="Batch provider to use: 'openai' or 'anthropic'.",
        case_sensitive=False,
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-M",
        help="Model override (e.g. 'gpt-4o' or 'claude-sonnet-4-5-20250929').",
    ),
    system_prompt: Optional[str] = typer.Option(
        None, "--system", "-s", help="Optional system prompt applied to every request."
    ),
    poll_interval: float = typer.Option(
        30.0, "--poll-interval", help="Seconds between batch status polls."
    ),
    timeout: float = typer.Option(
        3600.0, "--timeout", help="Maximum seconds to wait for the batch to finish."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write parsed JudgeResponse results as JSON to this path."
    ),
):
    """Submit a dataset as a batch job via OpenAI or Anthropic.

    Anthropic batches enjoy a 50% discount vs sync pricing; this command
    applies that discount automatically when estimating per-request cost.
    """
    import asyncio
    import json as _json

    from checkllm.batch import get_batch_runner
    from checkllm.datasets.loader import load_dataset

    cases = load_dataset(dataset_path)
    if not cases:
        console.print(f"[bold red]No cases found in dataset:[/] {dataset_path}")
        raise typer.Exit(code=1)

    try:
        runner = get_batch_runner(provider, model=model)
    except (ValueError, ImportError) as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    requests = [{"prompt": case.input} for case in cases]

    console.print(f"[bold]Submitting {len(requests)} requests via provider={runner.provider}[/]")

    async def _run() -> list:
        job = await runner.submit(requests, system_prompt=system_prompt)
        console.print(f"[dim]Batch id: {job.job_id}[/]")
        job = await runner.poll(job, interval_seconds=poll_interval, timeout_seconds=timeout)
        console.print(
            f"[dim]Final status: {job.status.value} "
            f"(completed={job.completed_requests}, failed={job.failed_requests})[/]"
        )
        return await runner.retrieve(job)

    responses = asyncio.run(_run())
    total_cost = sum(r.cost for r in responses)
    console.print(
        f"[bold green]Retrieved {len(responses)} responses (total cost: ${total_cost:.4f})[/]"
    )

    if output is not None:
        payload = [r.model_dump() for r in responses]
        output.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[dim]Wrote results to {output}[/]")


def _default_baseline_path(judge_name: str) -> Path:
    """Resolve the default baseline-file location for ``judge_name``."""
    import os as _os

    root = Path(_os.environ.get("CHECKLLM_DRIFT_DIR") or ".checkllm/drift")
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in judge_name)
    return root / f"{safe}.json"


@drift_app.command("baseline")
def drift_baseline_cmd(
    judge_name: str = typer.Argument(
        ..., help="Judge backend name (openai, anthropic, deepseek, ...)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-M", help="Override the judge model (e.g. gpt-4o-mini)."
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Where to write the baseline JSON. Defaults to "
        "$CHECKLLM_DRIFT_DIR/<judge>.json (./.checkllm/drift/<judge>.json).",
    ),
    probes_file: Optional[Path] = typer.Option(
        None,
        "--probes",
        help="Path to a text file with one probe prompt per line. "
        "Defaults to the built-in 20-prompt probe set.",
    ),
):
    """Record a new drift baseline for a judge."""
    from checkllm.drift import DEFAULT_PROBE_PROMPTS, record_baseline_sync
    from checkllm.providers import create_judge

    probes: list[str] | None = None
    if probes_file is not None:
        if not probes_file.exists():
            console.print(f"[bold red]Probes file not found:[/] {probes_file}")
            raise typer.Exit(code=1)
        probes = [
            line.strip()
            for line in probes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not probes:
            console.print("[bold red]Probes file is empty.[/]")
            raise typer.Exit(code=1)

    kwargs: dict = {"temperature": 0.0}
    if model is not None:
        kwargs["model"] = model

    try:
        judge = create_judge(judge_name, **kwargs)
    except Exception as exc:
        console.print(f"[bold red]Failed to build judge '{judge_name}':[/] {exc}")
        raise typer.Exit(code=1) from exc

    target = output or _default_baseline_path(judge_name)
    console.print(
        f"[bold]Recording baseline[/] — judge={judge_name} "
        f"probes={len(probes) if probes else len(DEFAULT_PROBE_PROMPTS)} -> {target}"
    )
    baseline = record_baseline_sync(judge, probes=probes)
    baseline.save(target)
    console.print(
        f"[bold green]Saved baseline[/] (hash={baseline.response_hash[:12]}...) -> {target}"
    )
    raise typer.Exit(code=0)


@drift_app.command("check")
def drift_check_cmd(
    judge_name: str = typer.Argument(
        ..., help="Judge backend name. Must match the one used for the baseline."
    ),
    baseline: Path = typer.Option(
        ...,
        "--baseline",
        "-b",
        help="Path to the baseline JSON produced by 'checkllm drift baseline'.",
    ),
    model: Optional[str] = typer.Option(None, "--model", "-M", help="Override the judge model."),
    threshold: float = typer.Option(
        0.85, "--threshold", help="Mean-similarity threshold. Below this flags drift."
    ),
    probe_threshold: float = typer.Option(
        0.7,
        "--probe-threshold",
        help="Per-probe similarity threshold. Deltas below are flagged individually.",
    ),
    fail_on_drift: bool = typer.Option(
        True,
        "--fail-on-drift/--no-fail-on-drift",
        help="Exit 1 on drift detection (default: on).",
    ),
):
    """Check a judge for drift against a previously recorded baseline."""
    from checkllm.drift import JudgeBaseline, detect_drift_sync
    from checkllm.providers import create_judge

    if not baseline.exists():
        console.print(f"[bold red]Baseline file not found:[/] {baseline}")
        raise typer.Exit(code=1)

    loaded = JudgeBaseline.load(baseline)

    kwargs: dict = {"temperature": 0.0}
    if model is not None:
        kwargs["model"] = model

    try:
        judge = create_judge(judge_name, **kwargs)
    except Exception as exc:
        console.print(f"[bold red]Failed to build judge '{judge_name}':[/] {exc}")
        raise typer.Exit(code=1) from exc

    report = detect_drift_sync(
        judge,
        loaded,
        threshold=threshold,
        probe_threshold=probe_threshold,
    )

    colour = "red" if report.drifted else "green"
    console.print(f"[bold {colour}]{report.summary()}[/]")
    if report.version_changed:
        console.print("[yellow]Model version string changed since baseline.[/]")
    if report.drifted_probes:
        console.print(f"[yellow]Drifted probes ({len(report.drifted_probes)}):[/]")
        for delta in report.drifted_probes[:10]:
            console.print(
                f"  [{delta.index:>2}] sim={delta.similarity:.3f}  prompt={delta.prompt[:80]!r}"
            )
        if len(report.drifted_probes) > 10:
            console.print(f"  ... and {len(report.drifted_probes) - 10} more")

    if report.drifted and fail_on_drift:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)

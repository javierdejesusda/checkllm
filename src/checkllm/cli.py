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
    budget: Optional[float] = typer.Option(None, "--budget", help="Maximum USD to spend on judge calls"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable judge response caching"),
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Label for this run in history"),
):
    """Run LLM tests with rich terminal output."""
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
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="Path to dataset file (YAML, JSON, CSV)"),
    model: Optional[str] = typer.Option(None, "--model", "-M", help="Model to generate outputs (default: from config)"),
    metric: str = typer.Option("rubric", "--metric", "-m", help="Metric to evaluate (hallucination, relevance, toxicity, rubric)"),
    threshold: float = typer.Option(0.8, "--threshold", "-t", help="Pass/fail threshold"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save results as snapshot JSON"),
    budget: Optional[float] = typer.Option(None, "--budget", help="Maximum USD to spend on judge calls"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable judge response caching"),
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Label for this run in history"),
):
    """Evaluate a prompt template against a dataset."""
    import asyncio
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

    valid_metrics = {"hallucination", "relevance", "toxicity", "rubric", "fluency", "coherence", "sentiment", "correctness"}
    if metric not in valid_metrics:
        console.print(f"[bold red]Unknown metric: {metric}. Valid: {', '.join(sorted(valid_metrics))}[/]")
        raise typer.Exit(code=1)

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

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
            MetricRecord, Snapshot, TestRunRecord, save_snapshot,
        )

        tests = {
            f"eval_case_{i}": [
                TestRunRecord(
                    metrics={r.metric_name: MetricRecord(score=r.score, passed=r.passed)}
                )
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
                    model=model, max_tokens=1024,
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


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of runs to show"),
    run_id: Optional[int] = typer.Option(None, "--run", "-r", help="Show details for a specific run"),
    trend: Optional[str] = typer.Option(None, "--trend", help="Show score trend for test::metric (e.g. 'test_qa::hallucination')"),
    compare_runs: Optional[str] = typer.Option(None, "--compare", help="Compare two runs: 'ID1,ID2'"),
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
            status = Text("PASS", style="bold green") if point["passed"] else Text("FAIL", style="bold red")
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
    from checkllm.models import CheckResult

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
            status = Text("PASS", style="bold green") if c["passed"] else Text("FAIL", style="bold red")
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
    console.print(f"  Run #{run1.run_id}: {ts1} {run1.label or ''} ({run1.git_commit or 'no commit'})")
    console.print(f"  Run #{run2.run_id}: {ts2} {run2.label or ''} ({run2.git_commit or 'no commit'})")

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
    console.print(
        f"\n  Cost: ${run1.total_cost:.4f} -> ${run2.total_cost:.4f} ({cost_delta:+.4f})"
    )
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
def init(
    path: str = typer.Argument(".", help="Directory to initialize"),
):
    """Scaffold a new checkllm project with sample files."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)

    pyproject = target / "pyproject.toml"
    checkllm_config = (
        '\n[tool.checkllm]\n'
        'judge_model = "gpt-4o"\n'
        'default_threshold = 0.8\n'
        'runs_per_test = 1\n'
        'snapshot_dir = ".checkllm/snapshots"\n'
        'cache_enabled = true\n'
        'max_concurrency = 10\n'
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
            '[project]\n'
            'name = "my-project"\n'
            'version = "0.1.0"\n'
            'dependencies = []\n'
            '\n'
            '[project.optional-dependencies]\n'
            'dev = [\n'
            '    "checkllm",\n'
            '    "pytest",\n'
            ']\n'
            '\n'
            '[tool.pytest.ini_options]\n'
            'testpaths = ["tests"]\n'
            + checkllm_config
        )
        pyproject.write_text(full_pyproject)
        console.print(f"[green]Created {pyproject}[/]")

    tests_dir = target / "tests"
    tests_dir.mkdir(exist_ok=True)
    conftest = tests_dir / "conftest.py"
    if not conftest.exists():
        conftest.write_text(
            '"""checkllm configuration for this project.\n'
            '\n'
            'The check fixture is auto-discovered by the checkllm pytest plugin.\n'
            'Customize the judge backend here if needed.\n'
            '"""\n'
            '# To use a custom judge backend, uncomment and modify:\n'
            '#\n'
            '# import pytest\n'
            '# from checkllm.check import CheckCollector\n'
            '# from checkllm.config import load_config\n'
            '# from checkllm.judge import OpenAIJudge\n'
            '#\n'
            '# @pytest.fixture\n'
            '# def check(request):\n'
            '#     config = load_config()\n'
            '#     judge = OpenAIJudge(model="gpt-4o-mini")  # cheaper model\n'
            '#     collector = CheckCollector(config=config, judge=judge)\n'
            '#     request.node.stash[pytest.StashKey[CheckCollector]()] = collector\n'
            '#     return collector\n'
        )
        console.print(f"[green]Created {conftest}[/]")
    else:
        console.print(f"[dim]{conftest} already exists[/]")

    sample_test = tests_dir / "test_llm_example.py"
    if not sample_test.exists():
        sample_test.write_text(
            '"""Sample checkllm test file."""\n'
            '\n'
            '\n'
            'def test_output_quality(check):\n'
            '    """Example: check an LLM output with deterministic checks."""\n'
            '    output = "Python is a high-level programming language."\n'
            '\n'
            '    check.contains(output, "Python")\n'
            '    check.not_contains(output, "JavaScript")\n'
            '    check.max_tokens(output, limit=50)\n'
            '    check.regex(output, pattern=r"[A-Z][a-z]+")\n'
            '\n'
            '\n'
            'def test_json_output(check):\n'
            '    """Example: validate JSON structure."""\n'
            '    from pydantic import BaseModel\n'
            '\n'
            '    class Response(BaseModel):\n'
            '        answer: str\n'
            '        confidence: float\n'
            '\n'
            '    output = \'{"answer": "42", "confidence": 0.95}\'\n'
            '    check.json_schema(output, schema=Response)\n'
            '\n'
            '\n'
            '# Uncomment below to test with LLM-as-judge (requires OPENAI_API_KEY)\n'
            '# def test_hallucination(check):\n'
            '#     output = "The sky is blue due to Rayleigh scattering."\n'
            '#     context = "Rayleigh scattering causes the sky to appear blue."\n'
            '#     check.hallucination(output, context=context)\n'
        )
        console.print(f"[green]Created {sample_test}[/]")
    else:
        console.print(f"[dim]{sample_test} already exists[/]")

    fixtures_dir = tests_dir / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    sample_dataset = fixtures_dir / "cases.yaml"
    if not sample_dataset.exists():
        sample_dataset.write_text(
            '# Sample dataset for checkllm\n'
            '- input: "What is Python?"\n'
            '  expected: "Python is a programming language"\n'
            '  query: "Explain Python"\n'
            '  criteria: "accurate, concise"\n'
            '\n'
            '- input: "What is 2+2?"\n'
            '  expected: "4"\n'
            '  query: "Simple math"\n'
            '  criteria: "correct answer"\n'
        )
        console.print(f"[green]Created {sample_dataset}[/]")
    else:
        console.print(f"[dim]{sample_dataset} already exists[/]")

    checkllm_dir = target / ".checkllm" / "snapshots"
    checkllm_dir.mkdir(parents=True, exist_ok=True)
    gitkeep = checkllm_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    console.print(
        f"\n[bold green]checkllm initialized![/]\n\n"
        f"  Run tests:     [cyan]checkllm run tests/[/]\n"
        f"  Save baseline: [cyan]checkllm snapshot tests/[/]\n"
        f"  HTML report:   [cyan]checkllm report tests/[/]\n"
        f"  View history:  [cyan]checkllm history[/]\n"
        f"  Cache stats:   [cyan]checkllm cache --stats[/]\n"
    )


@app.command()
def watch(
    test_path: str = typer.Argument(help="Path to test directory or file"),
    watch_path: Optional[list[str]] = typer.Option(None, "--watch", "-w", help="Additional paths to watch"),
    interval: float = typer.Option(1.0, "--interval", "-i", help="Poll interval in seconds"),
    debounce: float = typer.Option(0.5, "--debounce", help="Debounce delay in seconds"),
    pattern: Optional[list[str]] = typer.Option(None, "--pattern", "-p", help="File patterns to watch"),
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
def list_metrics():
    """List all registered custom metrics."""
    from checkllm.metrics import _global_registry

    builtin_judge = [
        "hallucination", "relevance", "toxicity", "rubric", "fluency", "coherence",
        "sentiment", "correctness", "faithfulness", "context_relevance",
        "answer_completeness", "instruction_following", "summarization",
        "bias", "consistency", "groundedness",
    ]
    builtin_deterministic = [
        "contains", "not_contains", "exact_match", "starts_with", "ends_with",
        "regex", "max_tokens", "min_tokens", "word_count", "char_count",
        "sentence_count", "similarity", "readability", "latency", "cost",
        "json_schema", "is_json", "is_valid_python",
        "all_of", "any_of", "none_of",
    ]
    console.print("[bold]LLM-as-Judge metrics:[/]")
    for m in builtin_judge:
        console.print(f"  [cyan]{m}[/]")
    console.print(f"\n[bold]Deterministic checks:[/]")
    for m in builtin_deterministic:
        console.print(f"  [cyan]{m}[/]")

    custom = _global_registry.list_metrics()
    if custom:
        console.print(f"\n[bold]Custom registered metrics:[/]")
        for m in custom:
            console.print(f"  [cyan]{m}[/]")
    else:
        console.print(f"\n[dim]No custom metrics registered.[/]")

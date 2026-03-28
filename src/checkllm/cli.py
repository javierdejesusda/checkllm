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
    model: Optional[str] = typer.Option(None, "--model", "-M", help="Model to generate outputs (default: from config)"),
    metric: str = typer.Option("rubric", "--metric", "-m", help="Metric to evaluate (hallucination, relevance, toxicity, rubric)"),
    threshold: float = typer.Option(0.8, "--threshold", "-t", help="Pass/fail threshold"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save results as snapshot JSON"),
):
    """Evaluate a prompt template against a dataset.

    Sends each prompt to the LLM, then judges the output with the chosen metric.
    """
    import asyncio
    from checkllm.datasets.loader import load_yaml_dataset
    from checkllm.check import CheckCollector
    from checkllm.reporting.terminal import render_results

    cases = load_yaml_dataset(Path(dataset_path))
    config = load_config()
    gen_model = model or config.judge_model
    collector = CheckCollector(config=config)

    console.print(f"[bold]Evaluating {len(cases)} cases[/]")
    console.print(f"[dim]Generation model: {gen_model}[/]")
    console.print(f"[dim]Judge metric: {metric} (threshold={threshold})[/]")
    console.print()

    for i, case in enumerate(cases):
        rendered_prompt = prompt.replace("{input}", case.input)
        if case.query:
            rendered_prompt = rendered_prompt.replace("{query}", case.query)
        if case.context:
            rendered_prompt = rendered_prompt.replace("{context}", case.context)

        console.print(f"  [dim]Case {i + 1}/{len(cases)}: {case.input[:60]}...[/]")

        # Step 1: Call LLM to generate output
        llm_output = _generate_output(rendered_prompt, gen_model, config)

        console.print(f"    [dim]Output: {llm_output[:80]}...[/]")

        # Step 2: Judge the LLM output
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
def init(
    path: str = typer.Argument(".", help="Directory to initialize"),
):
    """Scaffold a new checkllm project with sample files."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)

    # Add [tool.checkllm] to pyproject.toml if it exists
    pyproject = target / "pyproject.toml"
    checkllm_config = (
        '\n[tool.checkllm]\n'
        'judge_model = "gpt-4o"\n'
        'default_threshold = 0.8\n'
        'runs_per_test = 1\n'
        'snapshot_dir = ".checkllm/snapshots"\n'
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

    # Create tests directory and conftest
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

    # Create sample test file
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

    # Create sample dataset
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

    # Create .checkllm directory
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
    )


@app.command(name="list-metrics")
def list_metrics():
    """List all registered custom metrics."""
    from checkllm.metrics import _global_registry

    builtin = ["hallucination", "relevance", "toxicity", "rubric"]
    console.print("[bold]Built-in metrics:[/]")
    for m in builtin:
        console.print(f"  [cyan]{m}[/]")

    custom = _global_registry.list_metrics()
    if custom:
        console.print(f"\n[bold]Custom registered metrics:[/]")
        for m in custom:
            console.print(f"  [cyan]{m}[/]")
    else:
        console.print(f"\n[dim]No custom metrics registered.[/]")

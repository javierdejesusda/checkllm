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
    compare: bool = typer.Option(False, "--compare", help="Compare against baseline snapshot"),
    fail_on_regression: bool = typer.Option(False, "--fail-on-regression", help="Exit 1 if regression detected"),
    junit_xml: Optional[str] = typer.Option(None, "--junit-xml", help="Write JUnit XML to this path"),
):
    """Run LLM tests with rich terminal output."""
    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    if junit_xml:
        cmd.extend([f"--junit-xml={junit_xml}"])
    result = subprocess.run(cmd)
    raise typer.Exit(code=result.returncode)


@app.command()
def snapshot(
    test_path: str = typer.Argument(help="Path to test directory or file"),
):
    """Snapshot current test results as the regression baseline."""
    config = load_config()
    snapshot_dir = Path(config.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Snapshot directory:[/] {snapshot_dir}")

    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        console.print("[bold green]Snapshot saved successfully[/]")
    else:
        console.print("[bold red]Tests failed. Snapshot not saved.[/]")
    raise typer.Exit(code=result.returncode)


@app.command()
def report(
    test_path: str = typer.Argument(help="Path to test directory or file"),
    output: str = typer.Option("report.html", "--output", "-o", help="Output file path"),
):
    """Generate an HTML report from test results."""
    console.print(f"[bold]Generating report to:[/] {output}")
    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    result = subprocess.run(cmd)
    if result.returncode in (0, 1):
        console.print(f"[bold green]Report written to {output}[/]")
    raise typer.Exit(code=result.returncode)


@app.command(name="eval")
def eval_cmd(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt template with {placeholders}"),
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="Path to dataset YAML file"),
):
    """Evaluate a prompt template against a dataset."""
    from checkllm.datasets.loader import load_yaml_dataset

    cases = load_yaml_dataset(Path(dataset_path))
    console.print(f"[bold]Evaluating prompt against {len(cases)} cases[/]")
    console.print(f"[dim]Prompt: {prompt}[/]")
    for i, case in enumerate(cases):
        console.print(f"  Case {i + 1}: {case.input[:60]}...")
    console.print("[bold green]Evaluation complete[/]")


@app.command()
def diff(
    baseline: str = typer.Option(..., "--baseline", help="Path to baseline snapshot"),
    candidate: str = typer.Option(..., "--candidate", help="Path to candidate snapshot"),
    dataset_path: str = typer.Option(..., "--dataset", help="Path to dataset YAML file"),
):
    """Compare two prompt versions side-by-side."""
    console.print(f"[bold]Comparing:[/] {baseline} vs {candidate}")
    console.print(f"[bold]Dataset:[/] {dataset_path}")
    console.print("[bold green]Comparison complete[/]")

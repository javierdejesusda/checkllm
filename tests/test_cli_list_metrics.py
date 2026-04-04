"""Tests for the enhanced list-metrics CLI command."""
from typer.testing import CliRunner
from checkllm.cli import app

runner = CliRunner()


def test_list_metrics_shows_builtin():
    result = runner.invoke(app, ["list-metrics"])
    assert result.exit_code == 0
    assert "hallucination" in result.output
    assert "contains" in result.output


def test_list_metrics_shows_categories():
    result = runner.invoke(app, ["list-metrics"])
    assert result.exit_code == 0
    # Should show organized output
    assert "Deterministic" in result.output or "LLM" in result.output

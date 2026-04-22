"""Tests for the estimate CLI command and --dry-run flag."""

from typer.testing import CliRunner
from checkllm.cli import app

runner = CliRunner()


def test_estimate_command(tmp_path):
    test_file = tmp_path / "test_ex.py"
    test_file.write_text(
        "def test_my_llm(check):\n"
        '    output = "hello"\n'
        '    check.contains(output, "hello")\n'
        '    check.hallucination(output, context="world")\n'
    )
    result = runner.invoke(app, ["estimate", str(test_file)])
    assert result.exit_code == 0
    assert "deterministic" in result.output
    assert "judge" in result.output


def test_estimate_with_model_flag(tmp_path):
    test_file = tmp_path / "test_ex.py"
    test_file.write_text('def test_my_llm(check):\n    check.hallucination("out", context="ctx")\n')
    result = runner.invoke(app, ["estimate", str(test_file), "--model", "gpt-4o-mini"])
    assert result.exit_code == 0
    assert "gpt-4o-mini" in result.output


def test_dry_run_flag(tmp_path):
    test_file = tmp_path / "test_ex.py"
    test_file.write_text('def test_my_llm(check):\n    check.contains("out", "hello")\n')
    result = runner.invoke(app, ["run", str(test_file), "--dry-run"])
    assert result.exit_code == 0
    assert "Estimated" in result.output or "dry" in result.output.lower()

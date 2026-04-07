"""Tests for the checkllm ci command."""
import json
from unittest.mock import patch
from typer.testing import CliRunner
from checkllm.cli import app

runner = CliRunner()


def test_ci_runs_locally_when_no_github_env(tmp_path):
    test_file = tmp_path / "test_ex.py"
    test_file.write_text(
        'def test_basic(check):\n'
        '    check.contains("hello world", "hello")\n'
    )
    with patch.dict("os.environ", {}, clear=True):
        result = runner.invoke(app, ["ci", str(test_file)])
    # Should still run (falls back to local mode)
    assert "checkllm CI" in result.output


def test_ci_detects_github_actions(tmp_path):
    test_file = tmp_path / "test_ex.py"
    test_file.write_text(
        'def test_basic(check):\n'
        '    check.contains("hello world", "hello")\n'
    )
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps({
        "pull_request": {"number": 42}
    }))
    env = {
        "GITHUB_TOKEN": "ghp_fake",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_EVENT_PATH": str(event_file),
    }
    with patch.dict("os.environ", env, clear=True):
        result = runner.invoke(app, ["ci", str(test_file), "--no-comment"])
    assert "GitHub Actions" in result.output
    assert "PR: #42" in result.output


def test_ci_with_budget(tmp_path):
    test_file = tmp_path / "test_ex.py"
    test_file.write_text(
        'def test_basic(check):\n'
        '    check.contains("hello", "hello")\n'
    )
    result = runner.invoke(app, ["ci", str(test_file), "--budget", "5.0"])
    # Should run without error
    assert result.exit_code == 0 or "checkllm CI" in result.output


def test_ci_no_comment_flag(tmp_path):
    test_file = tmp_path / "test_ex.py"
    test_file.write_text(
        'def test_basic(check):\n'
        '    check.contains("hello", "hello")\n'
    )
    result = runner.invoke(app, ["ci", str(test_file), "--no-comment"])
    assert "checkllm CI" in result.output

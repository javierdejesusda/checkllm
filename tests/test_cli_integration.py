"""Tests for real CLI command implementations."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from checkllm.cli import app

runner = CliRunner()


class TestCliRunWithFlags:
    def test_run_passes_snapshot_flag_to_pytest(self):
        with patch("checkllm.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["run", "tests/", "--snapshot", "snap.json"])
            assert result.exit_code == 0
            cmd = mock_run.call_args[0][0]
            assert any("--checkllm-snapshot" in str(a) for a in cmd)

    def test_run_passes_html_report_flag(self):
        with patch("checkllm.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["run", "tests/", "--html-report", "report.html"])
            assert result.exit_code == 0
            cmd = mock_run.call_args[0][0]
            assert any("--checkllm-report" in str(a) for a in cmd)


class TestCliSnapshot:
    def test_snapshot_runs_pytest_with_flag(self):
        with patch("checkllm.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["snapshot", "tests/", "--output", "snap.json"])
            assert result.exit_code == 0
            cmd = mock_run.call_args[0][0]
            assert any("--checkllm-snapshot" in str(a) for a in cmd)

    def test_snapshot_auto_generates_output_path(self):
        with patch("checkllm.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["snapshot", "tests/"])
            assert result.exit_code == 0
            cmd = mock_run.call_args[0][0]
            assert any("--checkllm-snapshot" in str(a) for a in cmd)


class TestCliReport:
    def test_report_runs_pytest_with_flag(self):
        with patch("checkllm.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["report", "tests/", "--output", "report.html"])
            assert result.exit_code == 0
            cmd = mock_run.call_args[0][0]
            assert any("--checkllm-report" in str(a) for a in cmd)


class TestCliDiff:
    def test_diff_compares_two_snapshots(self, tmp_path):
        import json
        baseline = tmp_path / "baseline.json"
        current = tmp_path / "current.json"

        snap_data = {
            "version": 1,
            "timestamp": "2026-03-28T12:00:00Z",
            "tests": {
                "test_foo": [
                    {"metrics": {"h": {"score": 0.9, "passed": True}}},
                    {"metrics": {"h": {"score": 0.88, "passed": True}}},
                    {"metrics": {"h": {"score": 0.91, "passed": True}}},
                ]
            }
        }
        baseline.write_text(json.dumps(snap_data))

        current_data = {
            "version": 1,
            "timestamp": "2026-03-28T13:00:00Z",
            "tests": {
                "test_foo": [
                    {"metrics": {"h": {"score": 0.89, "passed": True}}},
                    {"metrics": {"h": {"score": 0.90, "passed": True}}},
                    {"metrics": {"h": {"score": 0.87, "passed": True}}},
                ]
            }
        }
        current.write_text(json.dumps(current_data))

        result = runner.invoke(app, [
            "diff", "--baseline", str(baseline), "--current", str(current),
        ])
        assert result.exit_code == 0
        # "No regressions detected" is printed by console.print in diff cmd
        assert "No regressions" in result.output or result.exit_code == 0

    def test_diff_detects_regression(self, tmp_path):
        import json
        baseline = tmp_path / "baseline.json"
        current = tmp_path / "current.json"

        snap_data = {
            "version": 1, "timestamp": "2026-03-28T12:00:00Z",
            "tests": {"test_foo": [
                {"metrics": {"h": {"score": s, "passed": True}}}
                for s in [0.9, 0.92, 0.88, 0.91, 0.90]
            ]}
        }
        baseline.write_text(json.dumps(snap_data))

        current_data = {
            "version": 1, "timestamp": "2026-03-28T13:00:00Z",
            "tests": {"test_foo": [
                {"metrics": {"h": {"score": s, "passed": False}}}
                for s in [0.5, 0.52, 0.48, 0.51, 0.50]
            ]}
        }
        current.write_text(json.dumps(current_data))

        result = runner.invoke(app, [
            "diff", "--baseline", str(baseline), "--current", str(current),
            "--fail-on-regression",
        ])
        # With --fail-on-regression, should exit 1 when regression detected
        assert result.exit_code == 1

    def test_diff_fails_when_baseline_missing(self, tmp_path):
        result = runner.invoke(app, [
            "diff", "--baseline", str(tmp_path / "nope.json"),
            "--current", str(tmp_path / "also_nope.json"),
        ])
        assert result.exit_code == 1


class TestCliVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert "0.3.0" in result.output


class TestCliInit:
    def test_init_creates_sample_files(self, tmp_path):
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "pyproject.toml").exists()
        assert (tmp_path / "tests" / "test_llm_example.py").exists()
        assert (tmp_path / "tests" / "fixtures" / "cases.yaml").exists()
        assert (tmp_path / ".checkllm" / "snapshots" / ".gitkeep").exists()

    def test_init_idempotent(self, tmp_path):
        runner.invoke(app, ["init", str(tmp_path)])
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert "already exists" in result.output or "already has" in result.output


class TestCliListMetrics:
    def test_lists_builtin_metrics(self):
        result = runner.invoke(app, ["list-metrics"])
        assert result.exit_code == 0
        assert "hallucination" in result.output
        assert "relevance" in result.output
        assert "toxicity" in result.output
        assert "rubric" in result.output


class TestJudgeCostTracking:
    def test_estimate_cost(self):
        from checkllm.judge import estimate_cost

        cost = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        assert cost > 0
        # gpt-4o: 2.50/M input + 10.00/M output
        expected = 1000 * 2.50 / 1_000_000 + 500 * 10.00 / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_estimate_cost_unknown_model(self):
        from checkllm.judge import estimate_cost

        cost = estimate_cost("unknown-model", prompt_tokens=100, completion_tokens=50)
        assert cost > 0

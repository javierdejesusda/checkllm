from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from checkllm.cli import app


runner = CliRunner()


class TestCliRun:
    def test_run_invokes_pytest(self):
        with patch("checkllm.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["run", "tests/"])
            assert result.exit_code == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "pytest" in cmd[0] or "pytest" in str(cmd)

    def test_run_with_snapshot_and_report(self):
        with patch("checkllm.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, [
                "run", "tests/",
                "--snapshot", "snap.json",
                "--html-report", "report.html",
                "--junit-xml", "results.xml",
            ])
            assert result.exit_code == 0
            cmd = mock_run.call_args[0][0]
            assert any("--checkllm-snapshot" in str(a) for a in cmd)
            assert any("--checkllm-report" in str(a) for a in cmd)
            assert any("--checkllm-junit" in str(a) for a in cmd)


class TestCliSnapshot:
    def test_snapshot_creates_directory(self, tmp_path: Path):
        snapshot_dir = tmp_path / ".checkllm" / "snapshots"
        with patch("checkllm.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("checkllm.cli.load_config") as mock_config:
                mock_config.return_value = MagicMock(
                    snapshot_dir=str(snapshot_dir)
                )
                result = runner.invoke(app, ["snapshot", "tests/"])
        assert result.exit_code == 0


class TestCliVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert "3.0.0a1" in result.output

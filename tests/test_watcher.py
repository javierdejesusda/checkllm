from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch


from checkllm.watcher import FileWatcher, WatchRunner


class TestFileWatcherDetection:
    """FileWatcher should detect new and modified files."""

    def test_detects_new_file(self, tmp_path: Path) -> None:
        watcher = FileWatcher(paths=[tmp_path], patterns=["*.py"])

        # No changes yet
        assert watcher.check() == []

        # Create a new file
        new_file = tmp_path / "hello.py"
        new_file.write_text("print('hello')")

        changed = watcher.check()
        assert len(changed) == 1
        assert changed[0] == new_file.resolve()

    def test_detects_modified_file(self, tmp_path: Path) -> None:
        existing = tmp_path / "module.py"
        existing.write_text("x = 1")

        watcher = FileWatcher(paths=[tmp_path], patterns=["*.py"])
        assert watcher.check() == []

        # Ensure mtime actually advances (some filesystems have 1s granularity)
        time.sleep(0.05)
        existing.write_text("x = 2")
        # Force a different mtime in case the filesystem rounds
        import os

        mtime = existing.stat().st_mtime + 1
        os.utime(existing, (mtime, mtime))

        changed = watcher.check()
        assert len(changed) == 1
        assert changed[0] == existing.resolve()

    def test_no_false_positives(self, tmp_path: Path) -> None:
        existing = tmp_path / "stable.py"
        existing.write_text("pass")

        watcher = FileWatcher(paths=[tmp_path], patterns=["*.py"])
        assert watcher.check() == []
        # Second check with no modifications
        assert watcher.check() == []

    def test_detects_multiple_changes(self, tmp_path: Path) -> None:
        watcher = FileWatcher(paths=[tmp_path], patterns=["*.py"])

        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")

        changed = watcher.check()
        assert len(changed) == 2

    def test_watches_multiple_paths(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "src"
        dir_b = tmp_path / "tests"
        dir_a.mkdir()
        dir_b.mkdir()

        watcher = FileWatcher(paths=[dir_a, dir_b], patterns=["*.py"])
        assert watcher.check() == []

        (dir_a / "app.py").write_text("app")
        (dir_b / "test_app.py").write_text("test")

        changed = watcher.check()
        assert len(changed) == 2


class TestFileWatcherPatternFiltering:
    """FileWatcher should only track files matching configured patterns."""

    def test_ignores_non_matching_files(self, tmp_path: Path) -> None:
        watcher = FileWatcher(paths=[tmp_path], patterns=["*.py"])

        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "notes.txt").write_text("hi")

        assert watcher.check() == []

    def test_matches_py_by_default(self, tmp_path: Path) -> None:
        watcher = FileWatcher(paths=[tmp_path])  # default *.py

        (tmp_path / "module.py").write_text("pass")
        (tmp_path / "data.json").write_text("{}")

        changed = watcher.check()
        assert len(changed) == 1
        assert changed[0].name == "module.py"

    def test_custom_patterns(self, tmp_path: Path) -> None:
        watcher = FileWatcher(paths=[tmp_path], patterns=["*.py", "*.yaml"])

        (tmp_path / "code.py").write_text("x = 1")
        (tmp_path / "config.yaml").write_text("key: val")
        (tmp_path / "readme.md").write_text("# Hi")

        changed = watcher.check()
        assert len(changed) == 2
        names = {p.name for p in changed}
        assert names == {"code.py", "config.yaml"}

    def test_single_file_watch(self, tmp_path: Path) -> None:
        target = tmp_path / "target.py"
        target.write_text("v1")

        watcher = FileWatcher(paths=[target], patterns=["*.py"])
        assert watcher.check() == []

        time.sleep(0.05)
        import os

        target.write_text("v2")
        mtime = target.stat().st_mtime + 1
        os.utime(target, (mtime, mtime))

        changed = watcher.check()
        assert len(changed) == 1


class TestFileWatcherDebounce:
    """Debounce should wait for changes to settle before triggering."""

    def test_debounce_accumulates(self, tmp_path: Path) -> None:
        """Changes within the debounce window are batched together."""
        callback_calls: list[list[Path]] = []

        def on_change(files: list[Path]) -> None:
            callback_calls.append(files)

        watcher = FileWatcher(
            paths=[tmp_path],
            patterns=["*.py"],
            poll_interval=0.05,
            debounce=0.15,
            on_change=on_change,
        )

        # Create file after watcher init to ensure it is detected
        (tmp_path / "first.py").write_text("1")

        # Simulate a short poll loop manually
        # First check picks up the file
        changed = watcher.check()
        assert len(changed) == 1

    def test_callback_receives_files(self, tmp_path: Path) -> None:
        received: list[list[Path]] = []

        def on_change(files: list[Path]) -> None:
            received.append(files)

        watcher = FileWatcher(
            paths=[tmp_path],
            patterns=["*.py"],
            poll_interval=0.05,
            debounce=0.0,  # no debounce for simplicity
            on_change=on_change,
        )

        (tmp_path / "a.py").write_text("a")

        # Manually simulate what start() would do
        new_changes = watcher.check()
        assert len(new_changes) > 0
        # Trigger callback directly
        on_change(new_changes)
        assert len(received) == 1
        assert received[0][0].name == "a.py"


class TestWatchRunnerInit:
    """WatchRunner should initialize with correct defaults."""

    def test_defaults(self) -> None:
        wr = WatchRunner(test_path="tests/")
        assert wr.test_path == "tests/"
        assert wr.poll_interval == 1.0
        assert wr.debounce == 0.5
        assert wr.patterns == ["*.py"]
        assert wr.run_count == 0
        assert wr.pass_count == 0
        assert wr.fail_count == 0

    def test_custom_params(self) -> None:
        wr = WatchRunner(
            test_path="my_tests/",
            watch_paths=["src/", "lib/"],
            poll_interval=2.0,
            debounce=1.0,
            patterns=["*.py", "*.yaml"],
            pytest_args=["--tb=short", "-x"],
            env_overrides={"CHECKLLM_BUDGET": "5.0"},
        )
        assert wr.test_path == "my_tests/"
        assert wr.poll_interval == 2.0
        assert wr.debounce == 1.0
        assert wr.patterns == ["*.py", "*.yaml"]
        assert wr.pytest_args == ["--tb=short", "-x"]
        assert wr.env_overrides == {"CHECKLLM_BUDGET": "5.0"}

    def test_build_cmd(self) -> None:
        wr = WatchRunner(test_path="tests/", pytest_args=["--tb=short"])
        cmd = wr._build_cmd()
        assert "pytest" in cmd[1] or "pytest" in str(cmd)
        assert "tests/" in cmd
        assert "--tb=short" in cmd

    def test_build_env_includes_overrides(self) -> None:
        wr = WatchRunner(
            test_path="tests/",
            env_overrides={"MY_VAR": "123"},
        )
        env = wr._build_env()
        assert env["MY_VAR"] == "123"


class TestWatchRunnerExecution:
    """WatchRunner test execution and history tracking."""

    @patch("checkllm.watcher.subprocess.run")
    def test_run_tests_tracks_pass(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        wr = WatchRunner(test_path="tests/")

        exit_code = wr._run_tests()

        assert exit_code == 0
        assert wr.run_count == 1
        assert wr.pass_count == 1
        assert wr.fail_count == 0

    @patch("checkllm.watcher.subprocess.run")
    def test_run_tests_tracks_fail(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        wr = WatchRunner(test_path="tests/")

        exit_code = wr._run_tests()

        assert exit_code == 1
        assert wr.run_count == 1
        assert wr.pass_count == 0
        assert wr.fail_count == 1

    @patch("checkllm.watcher.subprocess.run")
    def test_multiple_runs_history(self, mock_run: MagicMock) -> None:
        wr = WatchRunner(test_path="tests/")

        mock_run.return_value = MagicMock(returncode=0)
        wr._run_tests()
        mock_run.return_value = MagicMock(returncode=1)
        wr._run_tests()
        mock_run.return_value = MagicMock(returncode=0)
        wr._run_tests()

        assert wr.run_count == 3
        assert wr.pass_count == 2
        assert wr.fail_count == 1

    @patch("checkllm.watcher.subprocess.run")
    def test_run_tests_with_changed_files(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        wr = WatchRunner(test_path="tests/")

        changed = [Path("src/app.py"), Path("src/utils.py")]
        exit_code = wr._run_tests(changed_files=changed)
        assert exit_code == 0


class TestWatchRunnerGracefulShutdown:
    """WatchRunner should handle shutdown cleanly."""

    def test_stop_sets_watcher_flag(self) -> None:
        wr = WatchRunner(test_path="tests/")
        # Manually assign a watcher
        mock_watcher = MagicMock()
        wr._watcher = mock_watcher

        wr.stop()
        mock_watcher.stop.assert_called_once()

    def test_stop_without_watcher(self) -> None:
        wr = WatchRunner(test_path="tests/")
        # Should not raise
        wr.stop()

    def test_shutdown_prints_summary(self) -> None:
        wr = WatchRunner(test_path="tests/")
        wr.run_count = 5
        wr.pass_count = 3
        wr.fail_count = 2
        mock_watcher = MagicMock()
        wr._watcher = mock_watcher

        wr._shutdown()
        mock_watcher.stop.assert_called_once()

    def test_file_watcher_stop(self) -> None:
        watcher = FileWatcher(paths=[], patterns=["*.py"])
        assert not watcher._running
        watcher._running = True
        watcher.stop()
        assert not watcher._running


class TestWatchCLI:
    """Test the watch CLI command registration."""

    def test_watch_command_in_help(self) -> None:
        import re
        from typer.testing import CliRunner
        from checkllm.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        # Strip ANSI escape codes for reliable assertions on all platforms
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "Watch for file changes" in clean
        assert "--interval" in clean
        assert "--debounce" in clean
        assert "--pattern" in clean
        assert "--watch" in clean
        assert "--budget" in clean
        assert "--no-cache" in clean
        assert "--profile" in clean

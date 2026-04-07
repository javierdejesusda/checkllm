from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class FileWatcher:
    """Polls directories for file changes using mtime comparison.

    Cross-platform, no external dependencies.
    """

    def __init__(
        self,
        paths: list[Path],
        patterns: list[str] | None = None,
        poll_interval: float = 1.0,
        debounce: float = 0.5,
        on_change: Callable[[list[Path]], None] | None = None,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.patterns = patterns or ["*.py"]
        self.poll_interval = poll_interval
        self.debounce = debounce
        self.on_change = on_change

        self._running = False
        self._file_mtimes: dict[Path, float] = {}
        self._snapshot_mtimes()

    def _matches_pattern(self, path: Path) -> bool:
        """Check if a file matches any of the configured glob patterns."""
        name = path.name
        return any(fnmatch.fnmatch(name, pat) for pat in self.patterns)

    def _collect_files(self) -> dict[Path, float]:
        """Walk all watched paths and collect mtimes of matching files."""
        mtimes: dict[Path, float] = {}
        for base in self.paths:
            base = Path(base)
            if base.is_file():
                if self._matches_pattern(base):
                    try:
                        mtimes[base.resolve()] = base.stat().st_mtime
                    except OSError:
                        pass
                continue
            if not base.is_dir():
                continue
            for root, _dirs, files in os.walk(base):
                for fname in files:
                    fpath = Path(root) / fname
                    if self._matches_pattern(fpath):
                        try:
                            mtimes[fpath.resolve()] = fpath.stat().st_mtime
                        except OSError:
                            pass
        return mtimes

    def _snapshot_mtimes(self) -> None:
        """Take a snapshot of current file mtimes."""
        self._file_mtimes = self._collect_files()

    def check(self) -> list[Path]:
        """Check for changed files since the last snapshot.

        Returns a list of changed (new or modified) file paths.
        """
        current = self._collect_files()
        changed: list[Path] = []

        for fpath, mtime in current.items():
            old_mtime = self._file_mtimes.get(fpath)
            if old_mtime is None or mtime > old_mtime:
                changed.append(fpath)

        # Detect deleted files (not included in changed, but update state)
        self._file_mtimes = current
        return changed

    def start(self) -> None:
        """Start the polling loop. Blocks until ``stop()`` is called."""
        self._running = True
        pending_changes: list[Path] = []
        last_change_time: float | None = None

        while self._running:
            time.sleep(self.poll_interval)
            if not self._running:
                break

            new_changes = self.check()
            if new_changes:
                pending_changes.extend(new_changes)
                last_change_time = time.monotonic()

            if (
                pending_changes
                and last_change_time is not None
                and (time.monotonic() - last_change_time) >= self.debounce
            ):
                # Deduplicate
                unique = list(dict.fromkeys(pending_changes))
                pending_changes.clear()
                last_change_time = None
                if self.on_change:
                    self.on_change(unique)

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._running = False


class WatchRunner:
    """Combines :class:`FileWatcher` with pytest execution.

    Runs tests immediately on start, then re-runs whenever watched files
    change.
    """

    def __init__(
        self,
        test_path: str,
        watch_paths: list[str] | None = None,
        poll_interval: float = 1.0,
        debounce: float = 0.5,
        patterns: list[str] | None = None,
        pytest_args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        self.test_path = test_path
        self.poll_interval = poll_interval
        self.debounce = debounce
        self.patterns = patterns or ["*.py"]
        self.pytest_args = pytest_args or []
        self.env_overrides = env_overrides or {}

        # Determine paths to watch
        paths_to_watch: list[Path] = [Path(test_path)]
        for wp in watch_paths or []:
            paths_to_watch.append(Path(wp))
        self._watch_paths = paths_to_watch

        self.console = Console()
        self.run_count: int = 0
        self.pass_count: int = 0
        self.fail_count: int = 0
        self._watcher: FileWatcher | None = None

    def _build_cmd(self) -> list[str]:
        cmd = [sys.executable, "-m", "pytest", self.test_path, "-v"]
        cmd.extend(self.pytest_args)
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.env_overrides)
        return env

    def _run_tests(self, changed_files: list[Path] | None = None) -> int:
        """Execute the test suite and return the process exit code."""
        self.run_count += 1

        # Separator between runs
        if self.run_count > 1:
            self.console.print()
            self.console.rule(style="dim")
            self.console.print()

        now = datetime.now().strftime("%H:%M:%S")
        header_parts: list[str] = [f"[bold cyan]Run #{self.run_count}[/]", f"[dim]{now}[/]"]
        if changed_files:
            names = ", ".join(p.name for p in changed_files[:5])
            if len(changed_files) > 5:
                names += f" (+{len(changed_files) - 5} more)"
            header_parts.append(f"[yellow]changed: {names}[/]")

        self.console.print(Panel(
            Text.from_markup(" | ".join(header_parts)),
            title="checkllm watch",
            border_style="cyan",
        ))

        cmd = self._build_cmd()
        env = self._build_env()

        result = subprocess.run(cmd, env=env)
        exit_code = result.returncode

        if exit_code == 0:
            self.pass_count += 1
            self.console.print(f"\n[bold green]PASSED[/] [dim](run #{self.run_count})[/]")
        else:
            self.fail_count += 1
            self.console.print(f"\n[bold red]FAILED[/] [dim](run #{self.run_count}, exit code {exit_code})[/]")

        self.console.print(
            f"[dim]History: {self.pass_count} passed, {self.fail_count} failed "
            f"out of {self.run_count} runs[/]"
        )
        self.console.print("[dim]Watching for changes... (Ctrl+C to stop)[/]")

        return exit_code

    def run(self) -> None:
        """Start watching and run tests. Blocks until interrupted."""
        self.console.print(
            f"[bold]checkllm watch[/] | "
            f"test path: [cyan]{self.test_path}[/] | "
            f"patterns: {self.patterns} | "
            f"interval: {self.poll_interval}s"
        )

        # Initial run
        self._run_tests()

        self._watcher = FileWatcher(
            paths=self._watch_paths,
            patterns=self.patterns,
            poll_interval=self.poll_interval,
            debounce=self.debounce,
            on_change=self._on_change,
        )

        try:
            self._watcher.start()
        except KeyboardInterrupt:
            self._shutdown()

    def _on_change(self, changed_files: list[Path]) -> None:
        self._run_tests(changed_files=changed_files)

    def _shutdown(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
        self.console.print(
            f"\n[bold]Watch stopped.[/] "
            f"{self.run_count} runs: {self.pass_count} passed, {self.fail_count} failed."
        )

    def stop(self) -> None:
        """Stop the watch runner from an external caller."""
        if self._watcher is not None:
            self._watcher.stop()

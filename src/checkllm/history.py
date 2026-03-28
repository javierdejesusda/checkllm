"""Historical run tracking — stores eval runs with metadata in SQLite."""
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.history")

_DEFAULT_DB_DIR = ".checkllm"


def _get_git_commit() -> str | None:
    """Try to get the current short git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


@dataclass
class RunRecord:
    """A single historical run."""
    run_id: int
    timestamp: float
    label: str
    git_commit: str | None
    total_cost: float
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: dict[str, list[dict]]  # test_name -> list of serialized CheckResults


@dataclass
class RunSummary:
    """Lightweight summary for listing runs."""
    run_id: int
    timestamp: float
    label: str
    git_commit: str | None
    total_cost: float
    total_checks: int
    passed_checks: int
    failed_checks: int


class RunHistory:
    """SQLite-backed history of eval runs."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = Path(_DEFAULT_DB_DIR) / "history.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                git_commit TEXT,
                total_cost REAL NOT NULL DEFAULT 0.0,
                total_checks INTEGER NOT NULL DEFAULT 0,
                passed_checks INTEGER NOT NULL DEFAULT 0,
                failed_checks INTEGER NOT NULL DEFAULT 0,
                results_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self._conn.commit()

    def record_run(
        self,
        results: dict[str, list[CheckResult]],
        label: str = "",
    ) -> int:
        """Save a complete test run to history. Returns the run ID."""
        all_checks = [c for checks in results.values() for c in checks]
        total_cost = sum(c.cost for c in all_checks)
        passed = sum(1 for c in all_checks if c.passed)
        failed = sum(1 for c in all_checks if not c.passed)
        git_commit = _get_git_commit()

        serialized = {
            test_name: [c.model_dump() for c in checks]
            for test_name, checks in results.items()
        }

        cursor = self._conn.execute(
            """
            INSERT INTO runs (timestamp, label, git_commit, total_cost, total_checks, passed_checks, failed_checks, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                label,
                git_commit,
                total_cost,
                len(all_checks),
                passed,
                failed,
                json.dumps(serialized),
            ),
        )
        self._conn.commit()
        run_id = cursor.lastrowid
        logger.info("Recorded run #%d: %d checks, %d passed, %d failed, $%.4f", run_id, len(all_checks), passed, failed, total_cost)
        return run_id

    def list_runs(self, limit: int = 20) -> list[RunSummary]:
        """List recent runs, newest first."""
        rows = self._conn.execute(
            "SELECT id, timestamp, label, git_commit, total_cost, total_checks, passed_checks, failed_checks "
            "FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            RunSummary(
                run_id=r[0],
                timestamp=r[1],
                label=r[2],
                git_commit=r[3],
                total_cost=r[4],
                total_checks=r[5],
                passed_checks=r[6],
                failed_checks=r[7],
            )
            for r in rows
        ]

    def get_run(self, run_id: int) -> RunRecord | None:
        """Get a full run record by ID."""
        row = self._conn.execute(
            "SELECT id, timestamp, label, git_commit, total_cost, total_checks, passed_checks, failed_checks, results_json "
            "FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row[0],
            timestamp=row[1],
            label=row[2],
            git_commit=row[3],
            total_cost=row[4],
            total_checks=row[5],
            passed_checks=row[6],
            failed_checks=row[7],
            results=json.loads(row[8]),
        )

    def get_metric_trend(
        self, test_name: str, metric_name: str, limit: int = 20
    ) -> list[dict]:
        """Get score trend for a specific test+metric across recent runs.

        Returns list of dicts with keys: run_id, timestamp, label, git_commit, score, passed.
        """
        runs = self._conn.execute(
            "SELECT id, timestamp, label, git_commit, results_json "
            "FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

        trend = []
        for row in reversed(runs):  # chronological order
            results = json.loads(row[4])
            if test_name in results:
                for check_data in results[test_name]:
                    if check_data.get("metric_name") == metric_name:
                        trend.append({
                            "run_id": row[0],
                            "timestamp": row[1],
                            "label": row[2],
                            "git_commit": row[3],
                            "score": check_data["score"],
                            "passed": check_data["passed"],
                        })
                        break
        return trend

    def delete_run(self, run_id: int) -> bool:
        """Delete a run by ID. Returns True if it existed."""
        cursor = self._conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        self._conn.close()

"""Experiment tracking with prompt versioning and run comparison.

Provides an SQLite-backed experiment tracker for recording evaluation runs,
comparing results across different configurations, and identifying the best
performing prompt/model combinations.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.experiments")

_DEFAULT_DB_DIR = ".checkllm"


class ExperimentRun(BaseModel):
    """A single experiment run containing evaluation results and metadata.

    Attributes
    ----------
    run_id:
        Unique identifier for this run (UUID4).
    experiment_name:
        Name grouping related runs together.
    timestamp:
        Unix timestamp when the run was created.
    model:
        Model name used for this run.
    prompt_template:
        The prompt template text used.
    prompt_version:
        Version identifier for the prompt template.
    parameters:
        Additional configuration parameters (temperature, etc.).
    results:
        List of check results from this run.
    tags:
        Free-form tags for filtering runs.
    metadata:
        Arbitrary metadata attached to the run.
    """

    run_id: str
    experiment_name: str
    timestamp: float = Field(default_factory=time.time)
    model: str = ""
    prompt_template: str = ""
    prompt_version: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    results: list[CheckResult] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Fraction of checks that passed (0.0 -- 1.0)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def avg_score(self) -> float:
        """Average score across all check results."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def total_cost(self) -> float:
        """Total cost across all check results."""
        return sum(r.cost for r in self.results)


class ExperimentComparison(BaseModel):
    """Comparison between two experiment runs.

    Shows score deltas and identifies which metrics improved or degraded.
    """

    run_a: ExperimentRun
    run_b: ExperimentRun
    score_diff: float = 0.0
    pass_rate_diff: float = 0.0
    cost_diff: float = 0.0
    improved_metrics: list[str] = Field(default_factory=list)
    degraded_metrics: list[str] = Field(default_factory=list)


class ExperimentTracker:
    """Track and compare experiment runs with prompt versioning.

    Uses SQLite for persistent storage. Each run captures the model,
    prompt version, parameters, tags, and evaluation results, enabling
    comparisons and best-run queries across experiments.

    Usage::

        tracker = ExperimentTracker()
        run = tracker.start_run("my-experiment", model="gpt-4o", prompt_version="v2")
        # ... run evaluations, collect results ...
        tracker.log_results(run, results)
        tracker.end_run(run)

        # Compare runs
        comparison = tracker.compare("run-id-1", "run-id-2")

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Defaults to
        ``.checkllm/experiments.db``.
    """

    def __init__(self, db_path: Path | str = ".checkllm/experiments.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                timestamp REAL NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                prompt_template TEXT NOT NULL DEFAULT '',
                prompt_version TEXT NOT NULL DEFAULT '',
                parameters_json TEXT NOT NULL DEFAULT '{}',
                results_json TEXT NOT NULL DEFAULT '[]',
                tags_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                ended INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_experiment_name
            ON experiment_runs (experiment_name)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON experiment_runs (timestamp DESC)
            """
        )
        self._conn.commit()

    def start_run(
        self,
        experiment_name: str,
        model: str = "",
        prompt_template: str = "",
        prompt_version: str = "",
        tags: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        """Start a new experiment run.

        Creates a run record in the database and returns the
        ``ExperimentRun`` object for logging results.

        Parameters
        ----------
        experiment_name:
            Name for this experiment group.
        model:
            Model identifier.
        prompt_template:
            The prompt template text.
        prompt_version:
            Version string for the prompt template.
        tags:
            Optional tags for filtering.
        parameters:
            Optional configuration parameters.

        Returns
        -------
        ExperimentRun
            The newly created run.
        """
        run = ExperimentRun(
            run_id=str(uuid.uuid4()),
            experiment_name=experiment_name,
            model=model,
            prompt_template=prompt_template,
            prompt_version=prompt_version,
            tags=tags or [],
            parameters=parameters or {},
        )

        self._conn.execute(
            """
            INSERT INTO experiment_runs
                (run_id, experiment_name, timestamp, model, prompt_template,
                 prompt_version, parameters_json, results_json, tags_json,
                 metadata_json, ended)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                run.run_id,
                run.experiment_name,
                run.timestamp,
                run.model,
                run.prompt_template,
                run.prompt_version,
                json.dumps(run.parameters),
                json.dumps([]),
                json.dumps(run.tags),
                json.dumps(run.metadata),
            ),
        )
        self._conn.commit()

        logger.info(
            "Started run %s for experiment '%s' (model=%s, prompt_version=%s)",
            run.run_id[:8],
            experiment_name,
            model,
            prompt_version,
        )
        return run

    def log_results(self, run: ExperimentRun, results: list[CheckResult]) -> None:
        """Log evaluation results to an existing run.

        Appends the results to the run's result list both in-memory and
        in the database.

        Parameters
        ----------
        run:
            The run to update.
        results:
            Check results to append.
        """
        run.results.extend(results)

        serialized = [r.model_dump() for r in run.results]
        self._conn.execute(
            "UPDATE experiment_runs SET results_json = ? WHERE run_id = ?",
            (json.dumps(serialized), run.run_id),
        )
        self._conn.commit()

        logger.debug(
            "Logged %d results to run %s (total: %d)",
            len(results),
            run.run_id[:8],
            len(run.results),
        )

    def end_run(self, run: ExperimentRun) -> None:
        """Mark a run as ended and persist final metadata.

        Parameters
        ----------
        run:
            The run to finalize.
        """
        run.metadata["ended_at"] = time.time()
        run.metadata["pass_rate"] = run.pass_rate
        run.metadata["avg_score"] = run.avg_score
        run.metadata["total_cost"] = run.total_cost

        self._conn.execute(
            """
            UPDATE experiment_runs
            SET metadata_json = ?, results_json = ?, ended = 1
            WHERE run_id = ?
            """,
            (
                json.dumps(run.metadata),
                json.dumps([r.model_dump() for r in run.results]),
                run.run_id,
            ),
        )
        self._conn.commit()

        logger.info(
            "Ended run %s: pass_rate=%.2f, avg_score=%.3f, cost=$%.4f",
            run.run_id[:8],
            run.pass_rate,
            run.avg_score,
            run.total_cost,
        )

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Retrieve a run by its ID.

        Parameters
        ----------
        run_id:
            The UUID of the run.

        Returns
        -------
        ExperimentRun | None
            The run if found, otherwise ``None``.
        """
        row = self._conn.execute(
            """
            SELECT run_id, experiment_name, timestamp, model, prompt_template,
                   prompt_version, parameters_json, results_json, tags_json,
                   metadata_json
            FROM experiment_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_run(row)

    def list_runs(
        self,
        experiment_name: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[ExperimentRun]:
        """List experiment runs, optionally filtered by name and/or tags.

        Parameters
        ----------
        experiment_name:
            Filter by experiment name. If ``None``, returns all experiments.
        tags:
            Filter to runs containing *all* specified tags.
        limit:
            Maximum number of runs to return.

        Returns
        -------
        list[ExperimentRun]
            Runs ordered by timestamp descending.
        """
        query = """
            SELECT run_id, experiment_name, timestamp, model, prompt_template,
                   prompt_version, parameters_json, results_json, tags_json,
                   metadata_json
            FROM experiment_runs
        """
        conditions: list[str] = []
        params: list[Any] = []

        if experiment_name is not None:
            conditions.append("experiment_name = ?")
            params.append(experiment_name)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        runs = [self._row_to_run(row) for row in rows]

        # Filter by tags in Python (SQLite JSON querying is limited)
        if tags:
            tag_set = set(tags)
            runs = [r for r in runs if tag_set.issubset(set(r.tags))]

        return runs

    def compare(self, run_id_a: str, run_id_b: str) -> ExperimentComparison:
        """Compare two experiment runs.

        Computes score differences, pass rate differences, and identifies
        which individual metrics improved or degraded between run A and run B.

        Parameters
        ----------
        run_id_a:
            ID of the first (baseline) run.
        run_id_b:
            ID of the second (comparison) run.

        Returns
        -------
        ExperimentComparison
            Detailed comparison of the two runs.

        Raises
        ------
        ValueError
            If either run is not found.
        """
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)

        if run_a is None:
            raise ValueError(f"Run not found: {run_id_a}")
        if run_b is None:
            raise ValueError(f"Run not found: {run_id_b}")

        score_diff = run_b.avg_score - run_a.avg_score
        pass_rate_diff = run_b.pass_rate - run_a.pass_rate
        cost_diff = run_b.total_cost - run_a.total_cost

        # Compare per-metric scores
        scores_a = self._scores_by_metric(run_a)
        scores_b = self._scores_by_metric(run_b)

        all_metrics = set(scores_a.keys()) | set(scores_b.keys())
        improved: list[str] = []
        degraded: list[str] = []

        for metric in sorted(all_metrics):
            sa = scores_a.get(metric, 0.0)
            sb = scores_b.get(metric, 0.0)
            if sb > sa + 1e-9:
                improved.append(metric)
            elif sb < sa - 1e-9:
                degraded.append(metric)

        comparison = ExperimentComparison(
            run_a=run_a,
            run_b=run_b,
            score_diff=score_diff,
            pass_rate_diff=pass_rate_diff,
            cost_diff=cost_diff,
            improved_metrics=improved,
            degraded_metrics=degraded,
        )

        logger.info(
            "Comparison %s vs %s: score_diff=%.3f, pass_rate_diff=%.3f",
            run_id_a[:8],
            run_id_b[:8],
            score_diff,
            pass_rate_diff,
        )

        return comparison

    def best_run(
        self, experiment_name: str, metric: str = "avg_score"
    ) -> ExperimentRun | None:
        """Find the best run for an experiment by a given metric.

        Parameters
        ----------
        experiment_name:
            The experiment to search within.
        metric:
            The metric to optimize. Supported values: ``"avg_score"``,
            ``"pass_rate"``, ``"total_cost"`` (minimized).

        Returns
        -------
        ExperimentRun | None
            The best run, or ``None`` if no runs exist.
        """
        runs = self.list_runs(experiment_name=experiment_name, limit=1000)
        if not runs:
            return None

        # Filter to completed runs with results
        runs_with_results = [r for r in runs if r.results]
        if not runs_with_results:
            return runs[0]  # Return most recent if none have results

        if metric == "avg_score":
            return max(runs_with_results, key=lambda r: r.avg_score)
        elif metric == "pass_rate":
            return max(runs_with_results, key=lambda r: r.pass_rate)
        elif metric == "total_cost":
            # Lower cost is better
            return min(runs_with_results, key=lambda r: r.total_cost)
        else:
            # Try to find the metric by name in results
            def _metric_score(run: ExperimentRun) -> float:
                scores = [
                    r.score for r in run.results if r.metric_name == metric
                ]
                return sum(scores) / len(scores) if scores else 0.0

            return max(runs_with_results, key=_metric_score)

    def delete_run(self, run_id: str) -> bool:
        """Delete a run by its ID.

        Parameters
        ----------
        run_id:
            The UUID of the run to delete.

        Returns
        -------
        bool
            ``True`` if the run existed and was deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM experiment_runs WHERE run_id = ?", (run_id,)
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted run %s", run_id[:8])
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, "_conn"):
            self._conn.close()

    @staticmethod
    def _row_to_run(row: tuple[Any, ...]) -> ExperimentRun:
        """Convert a database row to an ExperimentRun."""
        results_data = json.loads(row[7])
        results = [CheckResult.model_validate(r) for r in results_data]

        return ExperimentRun(
            run_id=row[0],
            experiment_name=row[1],
            timestamp=row[2],
            model=row[3],
            prompt_template=row[4],
            prompt_version=row[5],
            parameters=json.loads(row[6]),
            results=results,
            tags=json.loads(row[8]),
            metadata=json.loads(row[9]),
        )

    @staticmethod
    def _scores_by_metric(run: ExperimentRun) -> dict[str, float]:
        """Compute average score per metric name for a run."""
        from collections import defaultdict

        totals: dict[str, list[float]] = defaultdict(list)
        for r in run.results:
            totals[r.metric_name].append(r.score)
        return {k: sum(v) / len(v) for k, v in totals.items()}

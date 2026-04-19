from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class MetricRecord(BaseModel):
    score: float
    passed: bool


class TestRunRecord(BaseModel):
    __test__ = False
    metrics: dict[str, MetricRecord]


class Snapshot(BaseModel):
    version: int = 1
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tests: dict[str, list[TestRunRecord]] = Field(default_factory=dict)

    def get_scores(self, test_name: str, metric_name: str) -> list[float]:
        """Get all scores for a given test and metric across runs."""
        runs = self.tests.get(test_name, [])
        return [
            run.metrics[metric_name].score
            for run in runs
            if metric_name in run.metrics
        ]

    def get_pass_results(self, test_name: str, metric_name: str) -> list[bool]:
        """Get all pass/fail results for a given test and metric across runs."""
        runs = self.tests.get(test_name, [])
        return [
            run.metrics[metric_name].passed
            for run in runs
            if metric_name in run.metrics
        ]


def save_snapshot(snapshot: Snapshot, path: Path) -> None:
    """Save a snapshot to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot.model_dump(), f, indent=2)


def load_snapshot(path: Path) -> Snapshot:
    """Load a snapshot from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return Snapshot.model_validate(data)

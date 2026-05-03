import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from checkllm.benchmarks.tau_bench_loader import (
    TauBenchTask,
    load_tau_bench,
)


def test_load_tau_bench_airline_returns_at_least_5_tasks():
    tasks = load_tau_bench("airline")
    assert len(tasks) >= 5
    assert all(isinstance(t, TauBenchTask) for t in tasks)


def test_load_tau_bench_retail_returns_at_least_5_tasks():
    tasks = load_tau_bench("retail")
    assert len(tasks) >= 5


def test_load_tau_bench_rejects_unknown_domain():
    with pytest.raises(ValueError, match="domain"):
        load_tau_bench("healthcare")


def test_tau_bench_task_fields():
    tasks = load_tau_bench("airline")
    task = tasks[0]
    # Required fields for the paper's ground-truth comparison.
    assert task.task_id
    assert task.user_instruction
    assert isinstance(task.tools, list)
    assert all(isinstance(t, dict) for t in task.tools)
    assert isinstance(task.reference_actions, list)
    assert isinstance(task.ground_truth_final_state, dict)


def test_load_tau_bench_limit_respected():
    tasks = load_tau_bench("retail", limit=2)
    assert len(tasks) == 2


def test_load_tau_bench_from_custom_path(tmp_path: Path):
    # Caller provides their own checkout of tau-bench-compatible JSONL.
    custom_tasks = tmp_path / "airline" / "tasks.jsonl"
    custom_tasks.parent.mkdir()
    custom_tasks.write_text(
        json.dumps(
            {
                "task_id": "custom-1",
                "user_instruction": "Book me a flight.",
                "tools": [{"name": "search_flights", "parameters": {}}],
                "reference_actions": [{"name": "search_flights", "arguments": {}}],
                "ground_truth_final_state": {"booking_id": "XYZ"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = load_tau_bench("airline", data_root=tmp_path)
    assert len(tasks) == 1
    assert tasks[0].task_id == "custom-1"


def test_load_tau_bench_skips_blank_lines(tmp_path: Path):
    f = tmp_path / "retail" / "tasks.jsonl"
    f.parent.mkdir()
    f.write_text(
        "\n"
        + json.dumps(
            {
                "task_id": "r-1",
                "user_instruction": "Return item.",
                "tools": [],
                "reference_actions": [],
                "ground_truth_final_state": {},
            }
        )
        + "\n\n   \n",
        encoding="utf-8",
    )
    tasks = load_tau_bench("retail", data_root=tmp_path)
    assert len(tasks) == 1


def test_tau_bench_task_rejects_missing_required_fields():
    # Pydantic should reject a row missing required fields.
    with pytest.raises(ValidationError):
        TauBenchTask(task_id="x")  # missing everything else


def test_load_tau_bench_limit_none_returns_all():
    all_tasks = load_tau_bench("airline", limit=None)
    assert len(all_tasks) == len(load_tau_bench("airline"))
    assert len(all_tasks) == 5

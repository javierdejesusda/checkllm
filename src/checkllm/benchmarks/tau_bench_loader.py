"""Loader for the tau-bench benchmark (Yao et al. 2024).

tau-bench (https://github.com/sierra-research/tau-bench, Apache-2.0) is a
tool-agent benchmark with two domains: ``airline`` and ``retail``. Each
task carries a natural-language user instruction, a set of tool JSON
schemas, a reference (gold) action sequence, and a ground-truth final
database state. Scoring is deterministic via final-state match, making
it the anchor benchmark for CheckLLM's paper.

By default the loader reads a small set of *synthetic* fixture tasks
vendored under ``src/checkllm/benchmarks/data/tau_bench/``. To run the
full benchmark, clone the upstream repo and pass ``data_root`` pointing
at a directory containing ``airline/tasks.jsonl`` and
``retail/tasks.jsonl`` in the same schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TauBenchTask(BaseModel):
    """A single tau-bench task.

    Attributes:
        task_id: Stable identifier for the task.
        user_instruction: Natural-language prompt the agent receives.
        tools: JSON schemas describing the tools the agent may call.
        reference_actions: Gold sequence of tool invocations.
        ground_truth_final_state: Expected database state after the
            agent's actions, used for deterministic scoring.
        domain: Either ``"airline"`` or ``"retail"``.
    """

    task_id: str
    user_instruction: str
    tools: list[dict[str, Any]] = Field(default_factory=list)
    reference_actions: list[dict[str, Any]] = Field(default_factory=list)
    ground_truth_final_state: dict[str, Any] = Field(default_factory=dict)
    domain: str


_VENDORED_ROOT = Path(__file__).resolve().parent / "data" / "tau_bench"
_VALID_DOMAINS = frozenset({"airline", "retail"})


def load_tau_bench(
    domain: str,
    limit: int | None = None,
    data_root: Path | str | None = None,
) -> list[TauBenchTask]:
    """Load tau-bench tasks for the given domain.

    Args:
        domain: Either ``"airline"`` or ``"retail"``.
        limit: If not ``None``, return only the first ``limit`` tasks.
        data_root: Override the vendored fixture root. Must contain a
            subdirectory named after ``domain`` with ``tasks.jsonl``.
            When ``None`` (default), uses the bundled synthetic fixtures.

    Returns:
        A list of :class:`TauBenchTask`.

    Raises:
        ValueError: If ``domain`` is not one of ``"airline"`` / ``"retail"``.
        FileNotFoundError: If the expected ``<domain>/tasks.jsonl`` is missing.
        json.JSONDecodeError: If any task line is not valid JSON.
    """
    if domain not in _VALID_DOMAINS:
        raise ValueError(
            f"Unknown tau-bench domain {domain!r}; expected one of " f"{sorted(_VALID_DOMAINS)}"
        )

    root = Path(data_root) if data_root is not None else _VENDORED_ROOT
    path = root / domain / "tasks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"tau-bench task file not found at {path}. "
            "Either vendor tau-bench under the default location or pass "
            "`data_root` pointing at a directory that contains "
            f"'{domain}/tasks.jsonl'."
        )

    tasks: list[TauBenchTask] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            task = TauBenchTask(
                task_id=str(row["task_id"]),
                user_instruction=str(row["user_instruction"]),
                tools=list(row.get("tools", [])),
                reference_actions=list(row.get("reference_actions", [])),
                ground_truth_final_state=dict(row.get("ground_truth_final_state", {})),
                domain=domain,
            )
            tasks.append(task)
            if limit is not None and len(tasks) >= limit:
                break
    return tasks

"""End-to-end paper experiment runner.

Reads a YAML config of experiments, iterates
``{experiment x model x seed x task}``, runs each agent, scores its
trajectory with CheckLLM's deterministic ``TrajectoryMetric``, and emits
a ``manifest.json`` + per-experiment trajectory JSONLs.

Run via:

    python -m benchmarks.paper.run_all --config benchmarks/paper/config.yaml

For smoke tests, use :func:`run_all_from_config` directly with an
in-memory dict.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from checkllm.benchmarks import TauBenchTask, load_tau_bench
from checkllm.metrics.trajectory_metric import TrajectoryMetric

from benchmarks.paper.agents import build_agent


_SUPPORTED_BENCHMARKS = frozenset({"tau_bench"})


def _load_benchmark_tasks(benchmark: str, domain: str, limit: int | None) -> list[TauBenchTask]:
    """Load tasks for one of the supported paper benchmarks."""
    if benchmark == "tau_bench":
        return load_tau_bench(domain, limit=limit)
    raise ValueError(f"Unknown benchmark {benchmark!r}; supported: {sorted(_SUPPORTED_BENCHMARKS)}")


def _expected_tool_names(reference_actions: list[dict[str, Any]]) -> list[str]:
    """Return the ordered sequence of tool names in the gold trajectory."""
    return [str(a.get("name", "")) for a in reference_actions]


def _score_trajectory(
    predicted_actions: list[dict[str, Any]],
    reference_actions: list[dict[str, Any]],
) -> tuple[float, bool]:
    """Score a trajectory with TrajectoryMetric and return (score, passed)."""
    expected = _expected_tool_names(reference_actions)
    predicted_names = [str(a.get("name", "")) for a in predicted_actions]
    metric = TrajectoryMetric(expected_trajectory=expected)
    subs = metric.compute_subscores(predicted_names)
    passed = subs.overall >= metric.threshold
    return float(subs.overall), bool(passed)


def _write_trajectory_jsonl(
    path: Path,
    predicted: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> None:
    """Persist a single task's predicted + reference trajectory to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "reference", "actions": reference}) + "\n")
        fh.write(json.dumps({"kind": "predicted", "actions": predicted}) + "\n")


def run_all_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run every experiment in ``config`` and emit a manifest.

    Args:
        config: A dict with keys ``experiments`` (list of experiment specs)
            and ``results_dir``.

    Returns:
        A summary dict with ``total_experiments`` and ``results_dir``.

    Raises:
        ValueError: If the config names an unknown benchmark or model.
    """
    results_dir = Path(config["results_dir"]).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for exp in config["experiments"]:
        experiment_id = exp["id"]
        benchmark = exp["benchmark"]
        domain = exp.get("domain", "")
        limit = exp.get("limit")
        tasks = _load_benchmark_tasks(benchmark, domain, limit)

        for model_name in exp["models"]:
            for seed in exp["seeds"]:
                agent = build_agent(model_name, seed=seed)
                for task in tasks:
                    predicted = agent.run(
                        reference_actions=task.reference_actions,
                        user_instruction=task.user_instruction,
                        tools=task.tools,
                    )
                    score, passed = _score_trajectory(predicted, task.reference_actions)
                    traj_path = (
                        results_dir
                        / experiment_id
                        / f"{model_name}_seed{seed}_{task.task_id}.jsonl"
                    )
                    _write_trajectory_jsonl(traj_path, predicted, task.reference_actions)
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "model": model_name,
                            "benchmark": benchmark,
                            "domain": domain,
                            "seed": seed,
                            "task_id": task.task_id,
                            "trajectory_score": score,
                            "passed": passed,
                            "reference_action_count": len(task.reference_actions),
                            "predicted_action_count": len(predicted),
                            "trajectory_path": str(traj_path.relative_to(results_dir)),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            # Phase C reproducibility fields - populated by real model adapters:
                            "model_version_sha": None,
                            "benchmark_sha": None,
                            "temperature": None,
                            "mean_latency_ms": None,
                            "total_cost_usd": None,
                        }
                    )

    manifest = {
        "schema": "https://checkllm.dev/schemas/paper_manifest/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    manifest_path = results_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "total_experiments": len(rows),
        "results_dir": str(results_dir),
        "manifest_path": str(manifest_path),
    }


def run_all_from_yaml(yaml_path: Path | str) -> dict[str, Any]:
    """Load a YAML config file and run it.

    Args:
        yaml_path: Path to a YAML file matching the dict schema of
            :func:`run_all_from_config`.

    Returns:
        Same summary dict as :func:`run_all_from_config`.

    Raises:
        ImportError: If ``pyyaml`` is not installed.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "pyyaml is required for run_all_from_yaml; install with `pip install pyyaml`"
        ) from exc
    with Path(yaml_path).open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return run_all_from_config(config)


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Run every CheckLLM paper experiment described in a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the experiments YAML (see benchmarks/paper/config.yaml).",
    )
    args = parser.parse_args()
    summary = run_all_from_yaml(args.config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

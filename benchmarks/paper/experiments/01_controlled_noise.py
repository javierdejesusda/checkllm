"""Task C1-zero: controlled-noise validation of TrajectoryMetric.

This experiment replaces GAIA reproduction with a zero-cost,
framework-validation run. The :class:`StubAgent` replays each tau-bench
reference trajectory under five increasing noise regimes (drop / repeat /
extra knobs). Per-trajectory and per-regime aggregates are written to
``trajectories.jsonl``, ``summary.json``, and ``manifest.json`` in the
output directory.

The headline claim of the experiment: as noise increases monotonically
across ``clean -> light -> medium -> heavy -> severe`` the
``TrajectoryMetric.overall`` mean must decrease monotonically. The
``summary.json`` ``monotonicity_check`` flag encodes that invariant for
each domain.

Run via:

    python benchmarks/paper/experiments/01_controlled_noise.py \
        --output-dir benchmarks/paper/results/01_controlled_noise

Note: because the file name starts with a digit it is not a valid Python
identifier; ``python -m benchmarks.paper.experiments.01_controlled_noise``
is therefore not supported. Invoke the script by file path instead.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure ``benchmarks`` and ``src/checkllm`` are importable when the
# script is invoked by file path (``python path/to/01_controlled_noise.py``).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from checkllm.benchmarks import TauBenchTask, load_tau_bench  # noqa: E402
from checkllm.metrics.trajectory_metric import TrajectoryMetric  # noqa: E402

from benchmarks.paper.agents import StubAgent  # noqa: E402


EXPERIMENT_ID = "01_controlled_noise"
DOMAINS: tuple[str, ...] = ("airline", "retail")
SEEDS: tuple[int, ...] = (42, 123, 2026)


@dataclass(frozen=True)
class NoiseLevel:
    """One named noise regime for the StubAgent.

    Attributes:
        name: Human-readable label (e.g. ``"clean"``).
        drop: Per-action probability of skipping a reference action.
        repeat: Per-action probability of duplicating an action.
        extra: Probability of appending one spurious tool call.
    """

    name: str
    drop: float
    repeat: float
    extra: float

    def as_params(self) -> dict[str, float]:
        """Return the noise knobs as a serialisable dict."""
        return {"drop": self.drop, "repeat": self.repeat, "extra": self.extra}


# Five regimes spanning the degradation spectrum from perfect to severely
# corrupted. Values are chosen so ``clean`` stays at overall=1.0 and
# ``severe`` lands close to 0 without saturating early levels.
NOISE_LEVELS: tuple[NoiseLevel, ...] = (
    NoiseLevel("clean", drop=0.0, repeat=0.0, extra=0.0),
    NoiseLevel("light", drop=0.10, repeat=0.10, extra=0.20),
    NoiseLevel("medium", drop=0.25, repeat=0.25, extra=0.50),
    NoiseLevel("heavy", drop=0.50, repeat=0.50, extra=0.80),
    NoiseLevel("severe", drop=0.80, repeat=0.80, extra=1.00),
)


def _load_tasks(domain: str, limit_tasks: int | None) -> list[TauBenchTask]:
    """Load tau-bench tasks for one domain, optionally truncated.

    Args:
        domain: Either ``"airline"`` or ``"retail"``.
        limit_tasks: When not ``None``, return at most this many tasks.

    Returns:
        A list of :class:`TauBenchTask`.
    """
    return load_tau_bench(domain, limit=limit_tasks)


def _expected_tool_names(reference_actions: list[dict[str, Any]]) -> list[str]:
    """Return the ordered tool names from a gold trajectory."""
    return [str(a.get("name", "")) for a in reference_actions]


def _score_trajectory(
    predicted_actions: list[dict[str, Any]],
    reference_actions: list[dict[str, Any]],
) -> tuple[TrajectoryMetric, dict[str, float], bool]:
    """Score one trajectory against its reference.

    Args:
        predicted_actions: Tool calls actually emitted by the agent.
        reference_actions: Gold tool calls for the task.

    Returns:
        A tuple ``(metric, subscores_dict, passed)`` where the metric is
        the configured :class:`TrajectoryMetric` (useful for thresholds)
        and ``subscores_dict`` contains the four sub-scores plus
        ``overall``.
    """
    expected = _expected_tool_names(reference_actions)
    predicted_names = [str(a.get("name", "")) for a in predicted_actions]
    metric = TrajectoryMetric(expected_trajectory=expected)
    subs = metric.compute_subscores(predicted_names)
    passed = subs.overall >= metric.threshold
    return metric, subs.as_dict(), passed


def _write_trajectory_jsonl(
    path: Path,
    predicted: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> None:
    """Persist predicted + reference trajectory to a per-task JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "reference", "actions": reference}) + "\n")
        fh.write(json.dumps({"kind": "predicted", "actions": predicted}) + "\n")


def _is_monotonic_non_increasing(values: list[float]) -> bool:
    """Return True iff ``values`` is monotonically non-increasing."""
    return all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def run_experiment(
    output_dir: Path | str,
    limit_tasks: int | None = None,
) -> dict[str, Any]:
    """Run the full controlled-noise experiment and write all outputs.

    Args:
        output_dir: Directory for the experiment outputs. Created if
            missing. Existing files are overwritten.
        limit_tasks: When not ``None``, truncate each domain to the
            first ``limit_tasks`` tasks (smoke testing).

    Returns:
        A summary dict with ``total_rows``, ``output_dir``, and
        ``monotonicity_check``.
    """
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_root / "trajectories.jsonl"
    summary_path = output_root / "summary.json"
    manifest_path = output_root / "manifest.json"

    rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []

    domain_tasks: dict[str, list[TauBenchTask]] = {
        domain: _load_tasks(domain, limit_tasks) for domain in DOMAINS
    }

    for domain in DOMAINS:
        tasks = domain_tasks[domain]
        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                agent = StubAgent(
                    seed=seed,
                    drop=noise.drop,
                    repeat=noise.repeat,
                    extra=noise.extra,
                )
                for task in tasks:
                    predicted = agent.run(
                        reference_actions=task.reference_actions,
                        user_instruction=task.user_instruction,
                        tools=task.tools,
                    )
                    _, subs, passed = _score_trajectory(predicted, task.reference_actions)
                    traj_path = (
                        output_root / domain / noise.name / f"seed{seed}_{task.task_id}.jsonl"
                    )
                    _write_trajectory_jsonl(traj_path, predicted, task.reference_actions)
                    timestamp = datetime.now(timezone.utc).isoformat()

                    manifest_row: dict[str, Any] = {
                        "experiment_id": EXPERIMENT_ID,
                        "model": "stub",
                        "benchmark": "tau_bench",
                        "domain": domain,
                        "seed": seed,
                        "task_id": task.task_id,
                        "noise_level": noise.name,
                        "noise_params": noise.as_params(),
                        "trajectory_score": float(subs["overall"]),
                        "passed": bool(passed),
                        "reference_action_count": len(task.reference_actions),
                        "predicted_action_count": len(predicted),
                        "trajectory_path": str(traj_path.relative_to(output_root)),
                        "timestamp_utc": timestamp,
                        "model_version_sha": None,
                        "benchmark_sha": None,
                        "temperature": None,
                        "mean_latency_ms": None,
                        "total_cost_usd": None,
                    }
                    rows.append(manifest_row)

                    jsonl_rows.append(
                        {
                            "domain": domain,
                            "task_id": task.task_id,
                            "seed": seed,
                            "noise_level": noise.name,
                            "noise_params": noise.as_params(),
                            "predicted_actions": predicted,
                            "reference_actions": task.reference_actions,
                            "ordering": float(subs["ordering"]),
                            "loops": float(subs["loops"]),
                            "coverage": float(subs["coverage"]),
                            "unexpected": float(subs["unexpected"]),
                            "overall": float(subs["overall"]),
                            "timestamp_utc": timestamp,
                        }
                    )

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in jsonl_rows:
            fh.write(json.dumps(row) + "\n")

    summary = _build_summary(jsonl_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "schema": "https://checkllm.dev/schemas/paper_manifest/v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "noise_levels": [{"name": n.name, **n.as_params()} for n in NOISE_LEVELS],
        "seeds": list(SEEDS),
        "domains": list(DOMAINS),
        "limit_tasks": limit_tasks,
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "total_rows": len(rows),
        "output_dir": str(output_root),
        "monotonicity_check": summary["monotonicity_check"],
    }


def _build_summary(jsonl_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-trajectory rows into per-(domain, noise) statistics.

    Args:
        jsonl_rows: The per-trajectory dicts written to
            ``trajectories.jsonl``.

    Returns:
        A summary dict containing per-domain noise-level means and
        standard deviations for ``overall`` and the four sub-scores,
        plus a ``monotonicity_check`` flag per domain.
    """
    sub_keys = ("ordering", "loops", "coverage", "unexpected", "overall")
    domain_noise_stats: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    domain_noise_means: dict[str, dict[str, float]] = {}

    for domain in DOMAINS:
        domain_noise_stats[domain] = {}
        domain_noise_means[domain] = {}
        for noise in NOISE_LEVELS:
            bucket = [
                r for r in jsonl_rows if r["domain"] == domain and r["noise_level"] == noise.name
            ]
            stats: dict[str, dict[str, float | int]] = {}
            for key in sub_keys:
                values = [float(r[key]) for r in bucket]
                stats[key] = {
                    "mean": float(statistics.fmean(values)) if values else 0.0,
                    "stdev": (float(statistics.pstdev(values)) if len(values) > 1 else 0.0),
                    "n": len(values),
                }
            domain_noise_stats[domain][noise.name] = stats
            domain_noise_means[domain][noise.name] = float(stats["overall"]["mean"])

    monotonicity_check: dict[str, bool] = {}
    for domain in DOMAINS:
        ordered_means = [domain_noise_means[domain][n.name] for n in NOISE_LEVELS]
        monotonicity_check[domain] = _is_monotonic_non_increasing(ordered_means)

    return {
        "experiment_id": EXPERIMENT_ID,
        "noise_levels": [n.name for n in NOISE_LEVELS],
        "domains": list(DOMAINS),
        "seeds": list(SEEDS),
        "domain_noise_stats": domain_noise_stats,
        "domain_noise_means": domain_noise_means,
        "monotonicity_check": monotonicity_check,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argparse namespace for the CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Controlled-noise validation experiment for TrajectoryMetric " "(Task C1-zero)."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write trajectories.jsonl, summary.json, manifest.json.",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="Truncate each domain to the first N tasks (smoke testing).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Console entry point.

    Args:
        argv: Optional explicit argv for testing. ``None`` uses ``sys.argv``.

    Returns:
        Exit code; ``0`` on success.
    """
    args = _parse_args(argv)
    summary = run_experiment(output_dir=args.output_dir, limit_tasks=args.limit_tasks)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

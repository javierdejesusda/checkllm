import json
from pathlib import Path

import pytest

from benchmarks.paper.run_all import run_all_from_config


def test_run_all_smoke_produces_manifest(tmp_path: Path):
    # Config: 1 model (stub), 1 benchmark (tau-bench airline), 1 seed, limit=2.
    config = {
        "experiments": [
            {
                "id": "smoke-airline",
                "benchmark": "tau_bench",
                "domain": "airline",
                "limit": 2,
                "models": ["stub"],
                "seeds": [42],
            }
        ],
        "results_dir": str(tmp_path / "results"),
    }
    result = run_all_from_config(config)

    manifest_path = tmp_path / "results" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    # 1 model * 1 seed * 2 tasks = 2 rows.
    assert len(rows) == 2
    assert {r["experiment_id"] for r in rows} == {"smoke-airline"}
    assert {r["model"] for r in rows} == {"stub"}
    assert all(r["seed"] == 42 for r in rows)
    # Every row carries a trajectory_score and pass flag.
    for row in rows:
        assert "trajectory_score" in row
        assert isinstance(row["trajectory_score"], float)
        assert 0.0 <= row["trajectory_score"] <= 1.0
        assert "passed" in row
        assert isinstance(row["passed"], bool)
    # Trajectories JSONL exists per task.
    traj_files = list((tmp_path / "results" / "smoke-airline").glob("*.jsonl"))
    assert len(traj_files) == 2

    # run_all_from_config also returns a summary dict.
    assert result["total_experiments"] == 2
    assert result["results_dir"] == str(tmp_path / "results")


def test_run_all_from_yaml_file(tmp_path: Path):
    # Exercises the yaml parsing path.
    yaml = pytest.importorskip("yaml")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {
                        "id": "smoke-retail",
                        "benchmark": "tau_bench",
                        "domain": "retail",
                        "limit": 1,
                        "models": ["stub"],
                        "seeds": [0],
                    }
                ],
                "results_dir": str(tmp_path / "r2"),
            }
        ),
        encoding="utf-8",
    )
    from benchmarks.paper.run_all import run_all_from_yaml

    run_all_from_yaml(cfg_path)
    manifest = json.loads((tmp_path / "r2" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["rows"]) == 1


def test_stub_agent_replays_reference_actions():
    from benchmarks.paper.agents import StubAgent

    agent = StubAgent(seed=42)
    ref = [
        {"name": "search_flights", "arguments": {"origin": "SFO"}},
        {"name": "book_flight", "arguments": {"flight_id": "UA-1"}},
    ]
    calls = agent.run(reference_actions=ref)
    assert [c["name"] for c in calls] == ["search_flights", "book_flight"]
    assert calls[0]["arguments"]["origin"] == "SFO"


def test_manifest_rows_include_reproducibility_metadata(tmp_path: Path):
    config = {
        "experiments": [
            {
                "id": "metadata-check",
                "benchmark": "tau_bench",
                "domain": "retail",
                "limit": 1,
                "models": ["stub"],
                "seeds": [0],
            }
        ],
        "results_dir": str(tmp_path / "results"),
    }
    run_all_from_config(config)
    manifest = json.loads((tmp_path / "results" / "manifest.json").read_text(encoding="utf-8"))
    row = manifest["rows"][0]
    # Reproducibility-critical fields.
    for key in (
        "experiment_id",
        "model",
        "benchmark",
        "domain",
        "seed",
        "task_id",
        "trajectory_score",
        "passed",
        "reference_action_count",
        "predicted_action_count",
        "timestamp_utc",
        "model_version_sha",
        "benchmark_sha",
        "temperature",
        "mean_latency_ms",
        "total_cost_usd",
    ):
        assert key in row, f"Missing key: {key}"


def test_run_all_rejects_unknown_benchmark(tmp_path: Path):
    config = {
        "experiments": [
            {
                "id": "x",
                "benchmark": "unknown_benchmark",
                "domain": "airline",
                "models": ["stub"],
                "seeds": [0],
            }
        ],
        "results_dir": str(tmp_path / "results"),
    }
    with pytest.raises(ValueError, match="benchmark"):
        run_all_from_config(config)


def test_run_all_handles_absolute_results_dir(tmp_path: Path):
    # results_dir passed as an absolute path must not break trajectory_path relativization.
    config = {
        "experiments": [
            {
                "id": "abs",
                "benchmark": "tau_bench",
                "domain": "airline",
                "limit": 1,
                "models": ["stub"],
                "seeds": [0],
            }
        ],
        "results_dir": str((tmp_path / "abs").resolve()),
    }
    run_all_from_config(config)
    manifest = json.loads((tmp_path / "abs" / "manifest.json").read_text(encoding="utf-8"))
    # trajectory_path must be relative, not absolute, so it stays portable in the manifest.
    assert not Path(manifest["rows"][0]["trajectory_path"]).is_absolute()


def test_run_all_rejects_unknown_model(tmp_path: Path):
    config = {
        "experiments": [
            {
                "id": "x",
                "benchmark": "tau_bench",
                "domain": "airline",
                "models": ["anthropic-claude-opus-4-7"],  # no provider wired in this task
                "seeds": [0],
            }
        ],
        "results_dir": str(tmp_path / "results"),
    }
    with pytest.raises(ValueError, match="model"):
        run_all_from_config(config)

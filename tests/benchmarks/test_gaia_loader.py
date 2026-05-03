import os

import pytest

from checkllm.benchmarks.gaia_loader import (
    GaiaTask,
    LicenseAcknowledgmentRequired,
    load_gaia,
)


def test_refuses_without_license_ack(monkeypatch):
    monkeypatch.delenv("CHECKLLM_GAIA_LICENSE_ACK", raising=False)
    with pytest.raises(LicenseAcknowledgmentRequired, match="CHECKLLM_GAIA_LICENSE_ACK"):
        load_gaia("validation", limit=1)


def test_refuses_when_ack_is_wrong_value(monkeypatch):
    monkeypatch.setenv("CHECKLLM_GAIA_LICENSE_ACK", "maybe")
    with pytest.raises(LicenseAcknowledgmentRequired, match="CHECKLLM_GAIA_LICENSE_ACK"):
        load_gaia("validation", limit=1)


def test_refuses_unknown_split(monkeypatch):
    monkeypatch.setenv("CHECKLLM_GAIA_LICENSE_ACK", "yes")
    with pytest.raises(ValueError, match="split"):
        load_gaia("train", limit=1)  # GAIA has only "validation" and "test"


def test_gaia_task_fields_exist_on_dataclass():
    # GaiaTask is importable and has the required fields via Pydantic / dataclass introspection.
    fields = set(GaiaTask.model_fields.keys())
    assert {"task_id", "question", "expected_answer", "level", "file_name"}.issubset(fields)


def test_load_gaia_with_mocked_hf_returns_GaiaTask_list(monkeypatch):
    # Replace the HF loader with a fake so we do not hit the network.
    monkeypatch.setenv("CHECKLLM_GAIA_LICENSE_ACK", "yes")

    fake_rows = [
        {
            "task_id": "abc-1",
            "Question": "What is 2+2?",
            "Final answer": "4",
            "Level": "1",
            "file_name": "",
        },
        {
            "task_id": "abc-2",
            "Question": "Name the 3rd planet.",
            "Final answer": "Earth",
            "Level": "1",
            "file_name": "",
        },
    ]

    def fake_load_dataset(name, config=None, split=None, revision=None, token=None):  # noqa: ARG001
        assert name == "gaia-benchmark/GAIA"
        assert split in {"validation", "test"}
        # revision must be pinned to a real-looking SHA
        assert isinstance(revision, str) and len(revision) >= 7
        return fake_rows

    from checkllm.benchmarks import gaia_loader as mod

    monkeypatch.setattr(mod, "_hf_load_dataset", fake_load_dataset)

    tasks = load_gaia("validation", limit=None)
    assert len(tasks) == 2
    assert all(isinstance(t, GaiaTask) for t in tasks)
    assert tasks[0].task_id == "abc-1"
    assert tasks[0].question == "What is 2+2?"
    assert tasks[0].expected_answer == "4"
    assert tasks[0].level == "1"


def test_load_gaia_limit_respected(monkeypatch):
    monkeypatch.setenv("CHECKLLM_GAIA_LICENSE_ACK", "yes")

    fake_rows = [
        {
            "task_id": f"t-{i}",
            "Question": f"Q{i}",
            "Final answer": f"A{i}",
            "Level": "1",
            "file_name": "",
        }
        for i in range(10)
    ]

    def fake_load_dataset(name, config=None, split=None, revision=None, token=None):  # noqa: ARG001
        return fake_rows

    from checkllm.benchmarks import gaia_loader as mod

    monkeypatch.setattr(mod, "_hf_load_dataset", fake_load_dataset)

    tasks = load_gaia("validation", limit=3)
    assert len(tasks) == 3
    assert tasks[-1].task_id == "t-2"


def test_load_gaia_handles_missing_optional_fields(monkeypatch):
    # Some GAIA rows have no file_name; loader must not crash.
    monkeypatch.setenv("CHECKLLM_GAIA_LICENSE_ACK", "yes")

    fake_rows = [
        {"task_id": "x", "Question": "Q", "Final answer": "A", "Level": "2"},
    ]

    def fake_load_dataset(name, config=None, split=None, revision=None, token=None):  # noqa: ARG001
        return fake_rows

    from checkllm.benchmarks import gaia_loader as mod

    monkeypatch.setattr(mod, "_hf_load_dataset", fake_load_dataset)

    tasks = load_gaia("validation", limit=1)
    assert tasks[0].file_name is None


def test_load_gaia_raises_if_manifest_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKLLM_GAIA_LICENSE_ACK", "yes")
    import checkllm.benchmarks.gaia_loader as mod

    monkeypatch.setattr(mod, "_MANIFEST_PATH", tmp_path / "nonexistent.json")
    with pytest.raises(FileNotFoundError, match="dataset manifest"):
        load_gaia("validation", limit=1)

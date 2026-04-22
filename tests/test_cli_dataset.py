"""Tests for the ``checkllm dataset`` CLI commands."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import yaml
from typer.testing import CliRunner

from checkllm.cli import app


runner = CliRunner()


def _make_sample_yaml(path: Path, n: int = 10) -> None:
    records = [
        {"input": f"case-{i}", "expected": str(i), "metadata": {"label": "A" if i % 2 else "B"}}
        for i in range(n)
    ]
    path.write_text(yaml.safe_dump(records, sort_keys=False), encoding="utf-8")


def test_dataset_help_lists_subcommands():
    result = runner.invoke(app, ["dataset", "--help"])
    assert result.exit_code == 0
    assert "load" in result.stdout
    assert "split" in result.stdout
    assert "versions" in result.stdout
    assert "diff" in result.stdout


def test_dataset_split_produces_two_files(tmp_path):
    src = tmp_path / "src.yaml"
    _make_sample_yaml(src, n=20)
    train = tmp_path / "train.yaml"
    test = tmp_path / "test.yaml"

    result = runner.invoke(
        app,
        [
            "dataset",
            "split",
            str(src),
            "--test-size",
            "0.2",
            "--train",
            str(train),
            "--test",
            str(test),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert train.exists()
    assert test.exists()

    train_data = yaml.safe_load(train.read_text(encoding="utf-8"))
    test_data = yaml.safe_load(test.read_text(encoding="utf-8"))
    assert len(train_data) + len(test_data) == 20
    assert len(test_data) == 4


def test_dataset_versions_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["dataset", "versions", "unknown"])
    assert result.exit_code == 0
    assert "No versions recorded" in result.stdout


def test_dataset_versions_lists_registered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from checkllm.datasets.case import Case
    from checkllm.datasets.lineage import LineageStore

    store = LineageStore()
    store.register([Case(input="hello")], dataset_id="my-ds", source="test")
    store.register(
        [Case(input="hello"), Case(input="world")], dataset_id="my-ds"
    )

    result = runner.invoke(app, ["dataset", "versions", "my-ds"])
    assert result.exit_code == 0
    assert "v1" in result.stdout
    assert "v2" in result.stdout


def test_dataset_diff_reports_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from checkllm.datasets.case import Case
    from checkllm.datasets.lineage import LineageStore

    store = LineageStore()
    store.register([Case(input="a"), Case(input="b")], dataset_id="my-ds")
    store.register(
        [Case(input="a"), Case(input="c")], dataset_id="my-ds"
    )

    result = runner.invoke(
        app, ["dataset", "diff", "v1", "v2", "--dataset-id", "my-ds"]
    )
    assert result.exit_code == 0
    assert "Added:" in result.stdout
    assert "Removed:" in result.stdout


def test_dataset_load_with_fake_hf(tmp_path, monkeypatch):
    rows = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Capital of France?", "answer": "Paris"},
    ]

    fake = types.ModuleType("datasets")
    fake.load_dataset = MagicMock(return_value=list(rows))
    fake.Dataset = MagicMock()
    monkeypatch.setitem(sys.modules, "datasets", fake)

    output = tmp_path / "out.yaml"
    result = runner.invoke(
        app, ["dataset", "load", "fake/ds", "--output", str(output)]
    )
    assert result.exit_code == 0, result.stdout
    assert output.exists()

    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert len(loaded) == 2
    assert loaded[0]["input"] == "What is 2+2?"
    assert loaded[0]["expected"] == "4"


def test_dataset_load_missing_datasets_package(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.delitem(sys.modules, "datasets", raising=False)

    output = tmp_path / "out.yaml"
    result = runner.invoke(
        app, ["dataset", "load", "fake/ds", "--output", str(output)]
    )
    assert result.exit_code == 1
    assert "datasets" in result.stdout

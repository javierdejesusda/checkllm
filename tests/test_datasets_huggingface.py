"""Tests for the Hugging Face datasets integration."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from checkllm.datasets.case import Case
from checkllm.datasets.huggingface import (
    _auto_detect_field,
    _build_auto_field_map,
    _row_to_case,
    load_hf_dataset,
)


def _install_fake_datasets(monkeypatch, rows, streaming_rows=None):
    """Install a minimal fake ``datasets`` module that yields ``rows``."""

    fake = types.ModuleType("datasets")

    def _load_dataset(name, config=None, split="test", streaming=False):
        data = list(streaming_rows) if (streaming and streaming_rows is not None) else list(rows)
        if streaming:
            return iter(data)
        return data

    fake.load_dataset = MagicMock(side_effect=_load_dataset)
    fake.Dataset = MagicMock()
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return fake


def test_auto_detect_field_case_insensitive():
    assert _auto_detect_field(["Question", "Answer"], ("question",)) == "Question"
    assert _auto_detect_field(["foo"], ("question",)) is None


def test_build_auto_field_map_common_columns():
    mapping = _build_auto_field_map(["question", "answer", "context"])
    assert mapping["question"] == "input"
    assert mapping["answer"] == "expected"
    assert mapping["context"] == "context"


def test_row_to_case_respects_field_map():
    row = {"q": "What is 2+2?", "a": "4", "topic": "math"}
    field_map = {"q": "input", "a": "expected"}
    case = _row_to_case(row, field_map)
    assert case.input == "What is 2+2?"
    assert case.expected == "4"
    assert case.metadata == {"topic": "math"}


def test_row_to_case_raises_without_input():
    with pytest.raises(ValueError):
        _row_to_case({"foo": "bar"}, {"foo": "metadata"})


def test_load_hf_dataset_auto_detects_columns(monkeypatch):
    rows = [
        {"question": "Q1", "answer": "A1", "source": "wiki"},
        {"question": "Q2", "answer": "A2", "source": "wiki"},
    ]
    fake = _install_fake_datasets(monkeypatch, rows)
    cases = load_hf_dataset("fake/ds")

    assert fake.load_dataset.called
    assert len(cases) == 2
    assert cases[0].input == "Q1"
    assert cases[0].expected == "A1"
    assert cases[0].metadata == {"source": "wiki"}


def test_load_hf_dataset_field_map(monkeypatch):
    rows = [{"q": "hello", "a": "world", "extra": 1}]
    _install_fake_datasets(monkeypatch, rows)
    cases = load_hf_dataset(
        "fake/ds",
        field_map={"q": "input", "a": "expected"},
    )
    assert cases[0].input == "hello"
    assert cases[0].expected == "world"
    assert cases[0].metadata == {"extra": 1}


def test_load_hf_dataset_limit(monkeypatch):
    rows = [{"input": f"case-{i}"} for i in range(10)]
    _install_fake_datasets(monkeypatch, rows)
    cases = load_hf_dataset("fake/ds", limit=3)
    assert len(cases) == 3
    assert all(isinstance(c, Case) for c in cases)


def test_load_hf_dataset_streaming_flag_passed(monkeypatch):
    rows = [{"input": "x"}]
    fake = _install_fake_datasets(monkeypatch, rows, streaming_rows=rows)
    load_hf_dataset("fake/ds", streaming=True, limit=1)
    _, kwargs = fake.load_dataset.call_args
    assert kwargs.get("streaming") is True


def test_load_hf_dataset_missing_datasets_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    # Force ImportError via sys.modules[None] path: simulate absence entirely.
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError) as excinfo:
        load_hf_dataset("fake/ds")
    assert "datasets" in str(excinfo.value)


def test_real_datasets_network_skipped():
    pytest.importorskip("datasets")
    # Intentionally do not call the network; just ensure the import path works.
    from checkllm.datasets.huggingface import load_hf_dataset as _real

    assert callable(_real)

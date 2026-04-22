"""Tests for dataset lineage tracking and versioning."""

from __future__ import annotations

import json

import pytest

from checkllm.datasets.case import Case
from checkllm.datasets.lineage import (
    DatasetVersion,
    LineageStore,
    compute_content_hash,
    track_lineage,
)


def _cases(n: int) -> list[Case]:
    return [Case(input=f"q-{i}", expected=str(i)) for i in range(n)]


def test_content_hash_is_stable():
    cases = _cases(5)
    h1 = compute_content_hash(cases)
    h2 = compute_content_hash(list(cases))
    assert h1 == h2


def test_content_hash_changes_when_content_changes():
    cases_a = _cases(3)
    cases_b = _cases(3)
    cases_b[0] = Case(input="different", expected="0")
    assert compute_content_hash(cases_a) != compute_content_hash(cases_b)


def test_register_creates_version(tmp_path):
    store = LineageStore(path=tmp_path / "lineage.json")
    v = store.register(_cases(4), dataset_id="golden", source="unit-test")
    assert isinstance(v, DatasetVersion)
    assert v.version == "v1"
    assert v.num_cases == 4
    assert v.parent_version is None


def test_register_is_idempotent_for_identical_content(tmp_path):
    store = LineageStore(path=tmp_path / "lineage.json")
    v1 = store.register(_cases(3), dataset_id="golden")
    v1_dup = store.register(_cases(3), dataset_id="golden")
    assert v1.version == v1_dup.version
    assert len(store.list_versions("golden")) == 1


def test_register_bumps_version_on_change(tmp_path):
    store = LineageStore(path=tmp_path / "lineage.json")
    v1 = store.register(_cases(3), dataset_id="golden")
    v2 = store.register(_cases(4), dataset_id="golden")
    assert v2.version == "v2"
    assert v2.parent_version == v1.version


def test_list_versions_chronological(tmp_path):
    store = LineageStore(path=tmp_path / "lineage.json")
    store.register(_cases(1), dataset_id="golden")
    store.register(_cases(2), dataset_id="golden")
    store.register(_cases(3), dataset_id="golden")
    versions = store.list_versions("golden")
    assert [v.version for v in versions] == ["v1", "v2", "v3"]


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "lineage.json"
    store = LineageStore(path=path)
    store.register(_cases(2), dataset_id="golden", source="file")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert "datasets" in loaded
    assert "golden" in loaded["datasets"]

    store2 = LineageStore(path=path)
    versions = store2.list_versions("golden")
    assert len(versions) == 1


def test_diff_added_removed_modified(tmp_path):
    store = LineageStore(path=tmp_path / "lineage.json")
    base = [
        Case(input="a", expected="1"),
        Case(input="b", expected="2"),
        Case(input="c", expected="3"),
    ]
    updated = [
        Case(input="a", expected="1"),
        Case(input="b", expected="two"),
        Case(input="d", expected="4"),
    ]
    v1 = store.register(base, dataset_id="golden")
    v2 = store.register(updated, dataset_id="golden")

    diff = store.diff(v1.version, v2.version, dataset_id="golden")
    assert diff.added == 1
    assert diff.removed == 1
    assert diff.modified == 1


def test_diff_requires_known_versions(tmp_path):
    store = LineageStore(path=tmp_path / "lineage.json")
    store.register(_cases(1), dataset_id="golden")
    with pytest.raises(ValueError):
        store.diff("v1", "v99", dataset_id="golden")


def test_track_lineage_decorator_registers(tmp_path, monkeypatch):
    monkeypatch.delenv("CHECKLLM_LINEAGE_DISABLED", raising=False)
    store = LineageStore(path=tmp_path / "lineage.json")

    @track_lineage(dataset_id="decorated", source="loader", store=store)
    def loader() -> list[Case]:
        return _cases(2)

    result = loader()
    assert len(result) == 2
    versions = store.list_versions("decorated")
    assert len(versions) == 1
    assert versions[0].source == "loader"


def test_track_lineage_disabled_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKLLM_LINEAGE_DISABLED", "1")
    store = LineageStore(path=tmp_path / "lineage.json")

    @track_lineage(dataset_id="disabled", store=store)
    def loader() -> list[Case]:
        return _cases(1)

    loader()
    assert store.list_versions("disabled") == []

"""Tests for :mod:`checkllm.drift`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from checkllm.drift import (
    DEFAULT_PROBE_PROMPTS,
    DriftReport,
    JudgeBaseline,
    ProbeDelta,
    detect_drift,
    detect_drift_sync,
    levenshtein_ratio,
    record_baseline,
    record_baseline_sync,
    response_similarity,
    token_overlap,
)


class _ScriptedJudge:
    """A judge that returns pre-recorded responses based on the prompt index."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "unseen probe",
        model: str = "scripted-model",
        model_version: str | None = None,
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.model = model
        self.model_version = model_version
        self.calls: list[str] = []

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append(prompt)
        return self.responses.get(prompt, self.default)


class _DeterministicJudge:
    """Returns the same text for every prompt — ideal for identity tests."""

    def __init__(self, text: str = "identical", model: str = "det-model") -> None:
        self._text = text
        self.model = model

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self._text


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def test_token_overlap_identical() -> None:
    assert token_overlap("hello world", "hello world") == 1.0


def test_token_overlap_empty_strings() -> None:
    assert token_overlap("", "") == 1.0


def test_token_overlap_disjoint() -> None:
    assert token_overlap("alpha beta", "gamma delta") == 0.0


def test_token_overlap_partial() -> None:
    # tokens: {alpha, beta}, {beta, gamma} -> 1/3
    assert token_overlap("alpha beta", "beta gamma") == pytest.approx(1 / 3)


def test_levenshtein_identical() -> None:
    assert levenshtein_ratio("abc", "abc") == 1.0


def test_levenshtein_disjoint_empty() -> None:
    assert levenshtein_ratio("abc", "") == 0.0


def test_levenshtein_one_edit() -> None:
    # "abc" -> "abd" = 1 edit, max_len = 3, ratio = 2/3
    assert levenshtein_ratio("abc", "abd") == pytest.approx(2 / 3)


def test_response_similarity_identical_strings() -> None:
    assert response_similarity("same", "same") == 1.0


def test_response_similarity_clipped_to_unit() -> None:
    sim = response_similarity("alpha", "alpha")
    assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# record_baseline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_baseline_uses_default_probes() -> None:
    judge = _DeterministicJudge("ok")
    baseline = await record_baseline(judge)
    assert baseline.model == "det-model"
    assert baseline.probes == list(DEFAULT_PROBE_PROMPTS)
    assert len(baseline.responses) == len(DEFAULT_PROBE_PROMPTS)
    assert all(r == "ok" for r in baseline.responses)
    assert len(baseline.response_hash) == 64  # hex SHA-256


@pytest.mark.asyncio
async def test_record_baseline_custom_probes() -> None:
    judge = _DeterministicJudge("x")
    baseline = await record_baseline(judge, probes=["p1", "p2"], metadata={"run": "test"})
    assert baseline.probes == ["p1", "p2"]
    assert baseline.responses == ["x", "x"]
    assert baseline.metadata == {"run": "test"}


def test_record_baseline_sync_wrapper() -> None:
    judge = _DeterministicJudge("y")
    baseline = record_baseline_sync(judge, probes=["p"])
    assert baseline.responses == ["y"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_roundtrip(tmp_path: Path) -> None:
    judge = _DeterministicJudge("r")
    baseline = await record_baseline(judge, probes=["a", "b"])
    target = baseline.save(tmp_path / "baseline.json")
    assert target.exists()

    loaded = JudgeBaseline.load(target)
    assert loaded.probes == baseline.probes
    assert loaded.responses == baseline.responses
    assert loaded.response_hash == baseline.response_hash


def test_baseline_from_dict_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        JudgeBaseline.from_dict({"model": "x"})


def test_baseline_to_json_is_valid_json() -> None:
    b = JudgeBaseline(
        model="m",
        model_version=None,
        probes=["p"],
        responses=["r"],
        response_hash="h",
    )
    parsed = json.loads(b.to_json())
    assert parsed["model"] == "m"
    assert parsed["probes"] == ["p"]


# ---------------------------------------------------------------------------
# detect_drift
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_drift_identical_baseline_no_drift() -> None:
    judge = _DeterministicJudge("same")
    baseline = await record_baseline(judge, probes=["p1", "p2", "p3"])
    # Re-run with the same judge: responses are identical.
    report = await detect_drift(judge, baseline)
    assert isinstance(report, DriftReport)
    assert report.drifted is False
    assert report.mean_similarity == pytest.approx(1.0)
    assert all(d.similarity == pytest.approx(1.0) for d in report.deltas)
    assert not report.drifted_probes
    assert report.baseline_hash == report.current_hash


@pytest.mark.asyncio
async def test_detect_drift_flags_divergence() -> None:
    probes = ["probe-1", "probe-2", "probe-3"]
    baseline_judge = _ScriptedJudge({p: f"baseline response for {p}" for p in probes})
    baseline = await record_baseline(baseline_judge, probes=probes)

    # A different judge returns unrelated strings for every probe.
    different_judge = _ScriptedJudge(
        {p: "totally different output with nothing in common xyz qqq" for p in probes}
    )
    report = await detect_drift(different_judge, baseline, threshold=0.85)

    assert report.drifted is True
    assert report.mean_similarity < 0.85
    assert report.baseline_hash != report.current_hash
    assert len(report.drifted_probes) == 3


@pytest.mark.asyncio
async def test_detect_drift_partial_divergence() -> None:
    probes = ["p1", "p2"]
    baseline_judge = _ScriptedJudge({"p1": "alpha", "p2": "beta"})
    baseline = await record_baseline(baseline_judge, probes=probes)

    current_judge = _ScriptedJudge({"p1": "alpha", "p2": "gamma delta epsilon"})
    report = await detect_drift(current_judge, baseline, threshold=0.8, probe_threshold=0.5)

    per_probe = {d.index: d for d in report.deltas}
    assert per_probe[0].similarity == pytest.approx(1.0)
    assert per_probe[0].drifted is False
    assert per_probe[1].drifted is True


@pytest.mark.asyncio
async def test_detect_drift_version_change_forces_drift() -> None:
    probes = ["p1"]
    baseline = JudgeBaseline(
        model="m",
        model_version="v1",
        probes=probes,
        responses=["same"],
        response_hash="ignored",
        system_prompt=None,
    )

    class _VersionedJudge:
        model = "m"
        model_version = "v2"  # different from baseline
        system_fingerprint = "v2"

        async def evaluate(self, prompt: str, system_prompt: str | None = None) -> str:
            return "same"

    report = await detect_drift(_VersionedJudge(), baseline)
    # Responses are identical -> mean_similarity == 1.0, but version changed.
    assert report.mean_similarity == pytest.approx(1.0)
    assert report.version_changed is True
    assert report.drifted is True


@pytest.mark.asyncio
async def test_detect_drift_rejects_empty_baseline() -> None:
    baseline = JudgeBaseline(
        model="m",
        model_version=None,
        probes=[],
        responses=[],
        response_hash="h",
    )
    with pytest.raises(ValueError, match="no probes"):
        await detect_drift(_DeterministicJudge(), baseline)


def test_detect_drift_sync_wrapper() -> None:
    judge = _DeterministicJudge("x")
    baseline = record_baseline_sync(judge, probes=["p1"])
    report = detect_drift_sync(judge, baseline)
    assert report.drifted is False


# ---------------------------------------------------------------------------
# ProbeDelta / DriftReport details
# ---------------------------------------------------------------------------


def test_drift_report_summary_contains_state() -> None:
    deltas = [
        ProbeDelta(
            index=0,
            prompt="p",
            baseline_response="a",
            current_response="b",
            similarity=0.1,
            drifted=True,
        )
    ]
    report = DriftReport(
        model="m",
        baseline_hash="h1",
        current_hash="h2",
        threshold=0.85,
        probe_threshold=0.7,
        mean_similarity=0.1,
        drifted=True,
        deltas=deltas,
        baseline_created_at="2026-01-01T00:00:00Z",
        checked_at="2026-04-23T00:00:00Z",
    )
    assert "DRIFT" in report.summary()
    assert "1/1" in report.summary()
    assert report.to_dict()["deltas"][0]["similarity"] == pytest.approx(0.1)


def test_drift_report_ok_summary() -> None:
    report = DriftReport(
        model="m",
        baseline_hash="h",
        current_hash="h",
        threshold=0.85,
        probe_threshold=0.7,
        mean_similarity=1.0,
        drifted=False,
        deltas=[],
        baseline_created_at="x",
        checked_at="y",
    )
    assert "ok" in report.summary().lower()

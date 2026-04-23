"""Tests for :mod:`checkllm.audits.vectordb_freshness`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from checkllm.audits import FreshnessAudit, FreshnessReport, StaleEntry
from checkllm.audits.vectordb_freshness import (
    _get_id,
    _parse_timestamp,
)


NOW = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)


def test_freshness_audit_flags_old_entries() -> None:
    entries = [
        {"id": "fresh", "updated_at": (NOW - timedelta(days=1)).isoformat()},
        {"id": "stale", "updated_at": (NOW - timedelta(days=60)).isoformat()},
    ]
    audit = FreshnessAudit(ttl=timedelta(days=30))
    report = audit.run(entries, now=NOW)

    assert isinstance(report, FreshnessReport)
    assert report.total == 2
    assert report.fresh_count == 1
    assert report.stale_count == 1
    assert report.stale_ratio == pytest.approx(0.5)
    assert not report.passed
    assert report.stale[0].id == "stale"
    assert report.stale[0].reason == "exceeds_ttl"
    assert report.stale[0].age is not None
    assert report.stale[0].age > timedelta(days=30)


def test_freshness_audit_passed_when_all_fresh() -> None:
    entries = [{"id": f"{i}", "updated_at": NOW.isoformat()} for i in range(3)]
    audit = FreshnessAudit(ttl=timedelta(days=1))
    report = audit.run(entries, now=NOW)
    assert report.passed is True
    assert report.stale_count == 0


def test_freshness_audit_custom_field() -> None:
    entries = [
        {"id": "a", "indexed_at": (NOW - timedelta(days=5)).isoformat()},
        {"id": "b", "indexed_at": (NOW - timedelta(days=50)).isoformat()},
    ]
    audit = FreshnessAudit(ttl=timedelta(days=10), timestamp_field="indexed_at")
    report = audit.run(entries, now=NOW)
    assert [e.id for e in report.stale] == ["b"]


def test_freshness_audit_ttl_as_seconds() -> None:
    entries = [{"id": "a", "updated_at": (NOW - timedelta(seconds=120)).isoformat()}]
    audit = FreshnessAudit(ttl=60)  # 60 seconds
    report = audit.run(entries, now=NOW)
    assert report.stale_count == 1


def test_freshness_audit_rejects_negative_ttl() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FreshnessAudit(ttl=timedelta(seconds=-1))


def test_freshness_audit_missing_timestamp_allowed_default() -> None:
    entries = [{"id": "a"}, {"id": "b", "updated_at": NOW.isoformat()}]
    audit = FreshnessAudit(ttl=timedelta(days=1))
    report = audit.run(entries, now=NOW)
    assert report.stale_count == 0  # missing timestamp tolerated


def test_freshness_audit_missing_timestamp_strict() -> None:
    entries = [{"id": "no-ts"}, {"id": "bad", "updated_at": "not a date"}]
    audit = FreshnessAudit(ttl=timedelta(days=1), missing_timestamp_allowed=False)
    report = audit.run(entries, now=NOW)
    assert report.stale_count == 2
    reasons = {e.reason for e in report.stale}
    assert reasons == {"missing_timestamp", "unparseable_timestamp"}


def test_freshness_audit_accepts_callable_provider() -> None:
    entries = [{"id": "a", "updated_at": NOW.isoformat()}]
    audit = FreshnessAudit(ttl=timedelta(days=1))
    calls = 0

    def provider() -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return entries

    report = audit.run(provider, now=NOW)
    assert calls == 1
    assert report.total == 1


def test_freshness_audit_reads_nested_metadata() -> None:
    entries = [
        {
            "id": "a",
            "metadata": {"updated_at": (NOW - timedelta(days=60)).isoformat()},
        }
    ]
    audit = FreshnessAudit(ttl=timedelta(days=30))
    report = audit.run(entries, now=NOW)
    assert report.stale_count == 1


def test_freshness_audit_object_attribute_access() -> None:
    @dataclass
    class Entry:
        id: str
        updated_at: datetime

    entries = [
        Entry("ok", NOW),
        Entry("stale", NOW - timedelta(days=90)),
    ]
    audit = FreshnessAudit(ttl=timedelta(days=30))
    report = audit.run(entries, now=NOW)
    assert [s.id for s in report.stale] == ["stale"]


@pytest.mark.parametrize(
    "value,expected_tz",
    [
        ("2026-01-01T00:00:00Z", timezone.utc),
        ("2026-01-01T00:00:00+00:00", timezone.utc),
        (1700000000, timezone.utc),
        (1700000000.5, timezone.utc),
    ],
)
def test_parse_timestamp_accepts_formats(value: Any, expected_tz: timezone) -> None:
    parsed = _parse_timestamp(value)
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_parse_timestamp_rejects_garbage() -> None:
    assert _parse_timestamp("not a date") is None
    assert _parse_timestamp(None) is None
    assert _parse_timestamp(True) is None


def test_parse_timestamp_handles_ms() -> None:
    # 1_700_000_000_000 ms == 1_700_000_000 s
    parsed = _parse_timestamp(1_700_000_000_000)
    assert parsed is not None
    assert parsed.year == 2023


def test_parse_timestamp_naive_datetime_assumed_utc() -> None:
    naive = datetime(2026, 1, 1, 0, 0, 0)
    parsed = _parse_timestamp(naive)
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_get_id_fallbacks() -> None:
    assert _get_id({"pk": "x"}) == "x"
    assert _get_id({"uuid": "u"}) == "u"

    class Obj:
        pass

    obj = Obj()
    obj.id = "z"
    assert _get_id(obj) == "z"


def test_report_summary_includes_ratio() -> None:
    entries = [
        {"id": "stale", "updated_at": (NOW - timedelta(days=60)).isoformat()},
    ]
    audit = FreshnessAudit(ttl=timedelta(days=30))
    report = audit.run(entries, now=NOW)
    summary = report.summary()
    assert "1/1" in summary
    assert "stale" in summary.lower()


def test_stale_entry_is_frozen() -> None:
    entry = StaleEntry(id="x", timestamp=NOW, age=timedelta(days=1), reason="exceeds_ttl")
    with pytest.raises(Exception):  # dataclass frozen -> FrozenInstanceError
        entry.id = "y"  # type: ignore[misc]

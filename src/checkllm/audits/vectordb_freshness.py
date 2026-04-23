"""Freshness audit for vector-store-backed knowledge bases.

Given a list of store entries (or a callable that produces them) and a
timestamp field, :class:`FreshnessAudit` flags entries older than a
configurable TTL. Intended for operating RAG stacks where stale documents
silently poison retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence


def _parse_timestamp(raw: Any) -> datetime | None:
    """Parse a timestamp value into a timezone-aware ``datetime``.

    Accepts ``datetime`` instances (assumed UTC if naive), ISO-8601 strings
    (``Z`` suffix supported), and numeric Unix timestamps in seconds or
    milliseconds.

    Returns ``None`` when the value cannot be parsed.
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        # Heuristic: values above year ~5000 in seconds are likely milliseconds.
        if value > 1e12:
            value /= 1000.0
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _get_field(entry: Any, field_name: str) -> Any:
    """Best-effort field extraction from dicts, pydantic models, or objects."""
    if isinstance(entry, Mapping):
        if field_name in entry:
            return entry[field_name]
        metadata = entry.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata.get(field_name)
        return None
    # Attribute access (dataclasses, pydantic v1/v2, plain classes).
    value = getattr(entry, field_name, None)
    if value is not None:
        return value
    metadata = getattr(entry, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata.get(field_name)
    return None


def _get_id(entry: Any) -> str:
    """Extract a stable id, falling back to repr when unknown."""
    for name in ("id", "_id", "pk", "uuid"):
        value = _get_field(entry, name)
        if value is not None:
            return str(value)
    return repr(entry)[:64]


@dataclass(frozen=True)
class StaleEntry:
    """A single entry flagged as stale by :class:`FreshnessAudit`.

    Attributes:
        id: Stable identifier of the entry.
        timestamp: Parsed timestamp (``None`` if unparseable / missing).
        age: Age of the entry at the time of the audit. ``None`` when
            ``timestamp`` is ``None``.
        reason: Why the entry was flagged — either
            ``"missing_timestamp"``, ``"unparseable_timestamp"``, or
            ``"exceeds_ttl"``.
    """

    id: str
    timestamp: datetime | None
    age: timedelta | None
    reason: str


@dataclass
class FreshnessReport:
    """Aggregated result of a :class:`FreshnessAudit` run.

    Attributes:
        total: Total entries audited.
        stale: Entries flagged as stale.
        ttl: TTL applied.
        as_of: Reference "now" used for the audit.
        missing_timestamp_allowed: Whether missing-timestamp entries are
            tolerated (``True`` = not flagged).
    """

    total: int = 0
    stale: list[StaleEntry] = field(default_factory=list)
    ttl: timedelta = timedelta(days=30)
    as_of: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    missing_timestamp_allowed: bool = True

    @property
    def stale_count(self) -> int:
        """Number of stale entries."""
        return len(self.stale)

    @property
    def fresh_count(self) -> int:
        """Number of entries that passed the freshness check."""
        return max(0, self.total - self.stale_count)

    @property
    def stale_ratio(self) -> float:
        """Share of stale entries in ``[0, 1]``."""
        return (self.stale_count / self.total) if self.total > 0 else 0.0

    @property
    def passed(self) -> bool:
        """True when no entry was flagged."""
        return not self.stale

    def summary(self) -> str:
        """Short one-line summary suitable for logs / CLI output."""
        return (
            f"FreshnessAudit: {self.stale_count}/{self.total} stale "
            f"(ratio={self.stale_ratio:.2%}, ttl={self.ttl})"
        )


EntryProvider = Callable[[], Iterable[Any]]


class FreshnessAudit:
    """Flag vector-store entries older than a configurable TTL.

    The audit is deliberately pure — it never calls a network itself. Callers
    supply either an already-materialised iterable of entries or a
    zero-argument callable that returns one. Each entry may be a ``dict``, a
    pydantic model, or any object exposing the timestamp field as an
    attribute (optionally nested under ``metadata``).

    Args:
        ttl: Maximum allowed age. Accepts ``timedelta`` or seconds.
        timestamp_field: Name of the field holding the entry's timestamp.
            Defaults to ``"updated_at"``.
        missing_timestamp_allowed: When ``True`` (the default), entries
            without a parseable timestamp are treated as fresh. When
            ``False`` they are flagged with reason
            ``"missing_timestamp"`` / ``"unparseable_timestamp"``.
    """

    def __init__(
        self,
        ttl: timedelta | float | int = timedelta(days=30),
        timestamp_field: str = "updated_at",
        missing_timestamp_allowed: bool = True,
    ) -> None:
        if isinstance(ttl, (int, float)):
            ttl = timedelta(seconds=float(ttl))
        if ttl.total_seconds() < 0:
            raise ValueError("ttl must be non-negative")
        self.ttl = ttl
        self.timestamp_field = timestamp_field
        self.missing_timestamp_allowed = missing_timestamp_allowed

    def run(
        self,
        entries: Iterable[Any] | EntryProvider,
        *,
        now: datetime | None = None,
    ) -> FreshnessReport:
        """Audit ``entries`` and return a :class:`FreshnessReport`.

        Args:
            entries: Iterable of entries, or a zero-arg callable that
                returns one. A callable is invoked once.
            now: Optional override for the audit's reference "now". Useful
                for deterministic tests. Defaults to ``datetime.now(utc)``.

        Returns:
            A :class:`FreshnessReport`.
        """
        reference = now if now is not None else datetime.now(tz=timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)

        resolved: Sequence[Any]
        if callable(entries):
            resolved = list(entries())
        else:
            resolved = list(entries)

        report = FreshnessReport(
            ttl=self.ttl,
            as_of=reference,
            missing_timestamp_allowed=self.missing_timestamp_allowed,
        )

        for entry in resolved:
            report.total += 1
            raw_ts = _get_field(entry, self.timestamp_field)
            parsed = _parse_timestamp(raw_ts)

            if parsed is None:
                if self.missing_timestamp_allowed:
                    continue
                reason = "missing_timestamp" if raw_ts is None else "unparseable_timestamp"
                report.stale.append(
                    StaleEntry(id=_get_id(entry), timestamp=None, age=None, reason=reason)
                )
                continue

            age = reference - parsed
            if age > self.ttl:
                report.stale.append(
                    StaleEntry(
                        id=_get_id(entry),
                        timestamp=parsed,
                        age=age,
                        reason="exceeds_ttl",
                    )
                )

        return report


__all__ = [
    "FreshnessAudit",
    "FreshnessReport",
    "StaleEntry",
]

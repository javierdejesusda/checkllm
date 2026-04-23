"""Dataset / knowledge-base audits for checkllm.

Audits inspect stores of data (vector DBs, datasets, snapshots) for
operational health issues that cannot be detected from a single eval
result: staleness, drift, schema violations, etc.
"""

from __future__ import annotations

from checkllm.audits.vectordb_freshness import (
    FreshnessAudit,
    FreshnessReport,
    StaleEntry,
)

__all__ = [
    "FreshnessAudit",
    "FreshnessReport",
    "StaleEntry",
]

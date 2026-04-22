"""Content-addressable cache for LLM judge responses using SQLite."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.cache")

_DEFAULT_CACHE_DIR = ".checkllm"
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days


def _cache_key(metric_name: str, model: str, output: str, **kwargs) -> str:
    """Generate a SHA-256 cache key from the evaluation inputs."""
    payload = json.dumps(
        {"metric": metric_name, "model": model, "output": output, **kwargs},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class JudgeCache:
    """SQLite-backed cache for LLM judge evaluation results."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self._saved_cost = 0.0

        if db_path is None:
            db_path = Path(_DEFAULT_CACHE_DIR) / "cache.db"
        self._db_path = Path(db_path)

        if self.enabled:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_cache (
                    key TEXT PRIMARY KEY,
                    metric TEXT NOT NULL,
                    model TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    def get(self, key: str) -> CheckResult | None:
        """Look up a cached result. Returns None on miss or expired entry."""
        if not self.enabled:
            return None
        row = self._conn.execute(
            "SELECT result_json, cost_usd, created_at FROM judge_cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            self.misses += 1
            logger.debug("Cache MISS: %s", key[:16])
            return None
        result_json, cost_usd, created_at = row
        if time.time() - created_at > self.ttl_seconds:
            self._conn.execute("DELETE FROM judge_cache WHERE key = ?", (key,))
            self._conn.commit()
            self.misses += 1
            logger.debug("Cache EXPIRED: %s", key[:16])
            return None
        self.hits += 1
        self._saved_cost += cost_usd
        logger.debug("Cache HIT: %s (saved $%.4f)", key[:16], cost_usd)
        return CheckResult.model_validate_json(result_json)

    def put(self, key: str, metric: str, model: str, result: CheckResult) -> None:
        """Store a result in the cache."""
        if not self.enabled:
            return
        self._conn.execute(
            """
            INSERT OR REPLACE INTO judge_cache (key, metric, model, result_json, cost_usd, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (key, metric, model, result.model_dump_json(), result.cost, time.time()),
        )
        self._conn.commit()
        logger.debug("Cache STORE: %s (%s, $%.4f)", key[:16], metric, result.cost)

    def clear(self) -> int:
        """Delete all cached entries. Returns number of entries deleted."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM judge_cache")
        count = cursor.fetchone()[0]
        self._conn.execute("DELETE FROM judge_cache")
        self._conn.commit()
        logger.info("Cache cleared: %d entries removed", count)
        return count

    def stats(self) -> dict:
        """Return cache statistics."""
        if not self.enabled:
            return {"enabled": False, "entries": 0, "size_bytes": 0}
        cursor = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) FROM judge_cache")
        count, total_cost = cursor.fetchone()
        size_bytes = self._db_path.stat().st_size if self._db_path.exists() else 0
        return {
            "enabled": True,
            "entries": count,
            "size_bytes": size_bytes,
            "total_cached_cost": total_cost,
            "session_hits": self.hits,
            "session_misses": self.misses,
            "session_saved_cost": self._saved_cost,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self.enabled and hasattr(self, "_conn"):
            self._conn.close()

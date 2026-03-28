"""Tests for the judge response caching system."""
from pathlib import Path

import pytest

from checkllm.cache import JudgeCache, _cache_key
from checkllm.models import CheckResult


@pytest.fixture
def cache(tmp_path):
    """Create a cache in a temporary directory."""
    db_path = tmp_path / "cache.db"
    c = JudgeCache(db_path=db_path, ttl_seconds=3600, enabled=True)
    yield c
    c.close()


@pytest.fixture
def sample_result():
    return CheckResult(
        passed=True,
        score=0.85,
        reasoning="Well grounded",
        cost=0.0023,
        latency_ms=450,
        metric_name="hallucination",
    )


class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key("hallucination", "gpt-4o", output="hello", context="world")
        k2 = _cache_key("hallucination", "gpt-4o", output="hello", context="world")
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        k1 = _cache_key("hallucination", "gpt-4o", output="hello", context="world")
        k2 = _cache_key("hallucination", "gpt-4o", output="bye", context="world")
        assert k1 != k2

    def test_different_models_different_keys(self):
        k1 = _cache_key("hallucination", "gpt-4o", output="hello")
        k2 = _cache_key("hallucination", "gpt-4o-mini", output="hello")
        assert k1 != k2

    def test_different_metrics_different_keys(self):
        k1 = _cache_key("hallucination", "gpt-4o", output="hello")
        k2 = _cache_key("relevance", "gpt-4o", output="hello")
        assert k1 != k2


class TestJudgeCache:
    def test_miss_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_put_and_get(self, cache, sample_result):
        cache.put("key1", "hallucination", "gpt-4o", sample_result)
        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.score == 0.85
        assert retrieved.metric_name == "hallucination"

    def test_hit_miss_counters(self, cache, sample_result):
        cache.put("key1", "hallucination", "gpt-4o", sample_result)
        cache.get("key1")  # hit
        cache.get("missing")  # miss
        assert cache.hits == 1
        assert cache.misses == 1

    def test_saved_cost_tracking(self, cache, sample_result):
        cache.put("key1", "hallucination", "gpt-4o", sample_result)
        cache.get("key1")
        assert cache._saved_cost == pytest.approx(0.0023)

    def test_clear(self, cache, sample_result):
        cache.put("key1", "hallucination", "gpt-4o", sample_result)
        cache.put("key2", "relevance", "gpt-4o", sample_result)
        count = cache.clear()
        assert count == 2
        assert cache.get("key1") is None

    def test_stats(self, cache, sample_result):
        cache.put("key1", "hallucination", "gpt-4o", sample_result)
        stats = cache.stats()
        assert stats["enabled"] is True
        assert stats["entries"] == 1
        assert stats["size_bytes"] > 0
        assert stats["total_cached_cost"] == pytest.approx(0.0023)

    def test_ttl_expiration(self, tmp_path, sample_result):
        cache = JudgeCache(db_path=tmp_path / "cache.db", ttl_seconds=0, enabled=True)
        cache.put("key1", "hallucination", "gpt-4o", sample_result)
        # With TTL=0, the entry is immediately expired
        import time
        time.sleep(0.01)
        assert cache.get("key1") is None
        cache.close()

    def test_disabled_cache(self, tmp_path, sample_result):
        cache = JudgeCache(db_path=tmp_path / "cache.db", enabled=False)
        cache.put("key1", "hallucination", "gpt-4o", sample_result)
        assert cache.get("key1") is None
        stats = cache.stats()
        assert stats["enabled"] is False
        cache.close()

    def test_overwrite_existing_key(self, cache, sample_result):
        cache.put("key1", "hallucination", "gpt-4o", sample_result)
        updated = CheckResult(
            passed=False, score=0.3, reasoning="Updated",
            cost=0.001, latency_ms=200, metric_name="hallucination",
        )
        cache.put("key1", "hallucination", "gpt-4o", updated)
        retrieved = cache.get("key1")
        assert retrieved.score == 0.3

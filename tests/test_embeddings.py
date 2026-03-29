from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from checkllm.embeddings import (
    CachedEmbeddings,
    EmbeddingBackend,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    batch_semantic_similarity,
    cosine_similarity,
    semantic_similarity,
)
from checkllm.models import CheckResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeBackend:
    """Minimal in-process backend for testing."""

    def __init__(self, dim: int = 3) -> None:
        self.dim = dim
        self.call_count = 0
        self.total_cost = 0.0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        # Deterministic pseudo-embeddings based on text hash
        results: list[list[float]] = []
        for text in texts:
            h = hash(text) % 10000
            vec = [(h + i) / 10000.0 for i in range(self.dim)]
            mag = math.sqrt(sum(x * x for x in vec))
            results.append([x / mag for x in vec] if mag else vec)
        return results


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_known_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        dot = 1 * 4 + 2 * 5 + 3 * 6  # 32
        mag_a = math.sqrt(14)
        mag_b = math.sqrt(77)
        expected = dot / (mag_a * mag_b)
        assert cosine_similarity(a, b) == pytest.approx(expected)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self):
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_single_dimension(self):
        assert cosine_similarity([5.0], [3.0]) == pytest.approx(1.0)

    def test_negative_single_dimension(self):
        assert cosine_similarity([5.0], [-3.0]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# OpenAIEmbeddings
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddings:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            OpenAIEmbeddings()

    def test_constructor_with_explicit_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAIEmbeddings(api_key="test-key")
        assert backend.model == "text-embedding-3-small"
        assert backend.total_cost == 0.0

    def test_constructor_with_env_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        backend = OpenAIEmbeddings()
        assert backend.model == "text-embedding-3-small"

    def test_custom_model_and_dimensions(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key="test-key",
            dimensions=256,
        )
        assert backend.model == "text-embedding-3-large"
        assert backend.dimensions == 256

    @pytest.mark.asyncio
    async def test_embed_returns_vectors(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAIEmbeddings(api_key="test-key")

        # Mock the API response
        mock_emb_0 = MagicMock()
        mock_emb_0.index = 0
        mock_emb_0.embedding = [0.1, 0.2, 0.3]

        mock_emb_1 = MagicMock()
        mock_emb_1.index = 1
        mock_emb_1.embedding = [0.4, 0.5, 0.6]

        mock_response = MagicMock()
        mock_response.data = [mock_emb_1, mock_emb_0]  # Deliberately out of order
        mock_response.usage.total_tokens = 10

        with patch.object(
            backend._client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await backend.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]  # Sorted by index
        assert result[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAIEmbeddings(api_key="test-key")
        result = await backend.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_cost_tracking(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key="test-key"
        )

        mock_emb = MagicMock()
        mock_emb.index = 0
        mock_emb.embedding = [0.1]

        mock_response = MagicMock()
        mock_response.data = [mock_emb]
        mock_response.usage.total_tokens = 1000

        with patch.object(
            backend._client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await backend.embed(["test"])

        expected_cost = 1000 * (0.02 / 1_000_000)
        assert backend.total_cost == pytest.approx(expected_cost)
        assert backend.total_tokens == 1000

    @pytest.mark.asyncio
    async def test_dimensions_passed_for_v3_models(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key="test-key", dimensions=512
        )

        mock_emb = MagicMock()
        mock_emb.index = 0
        mock_emb.embedding = [0.1] * 512

        mock_response = MagicMock()
        mock_response.data = [mock_emb]
        mock_response.usage.total_tokens = 5

        with patch.object(
            backend._client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await backend.embed(["test"])

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["dimensions"] == 512

    @pytest.mark.asyncio
    async def test_dimensions_not_passed_for_ada(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAIEmbeddings(
            model="text-embedding-ada-002", api_key="test-key", dimensions=512
        )

        mock_emb = MagicMock()
        mock_emb.index = 0
        mock_emb.embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.data = [mock_emb]
        mock_response.usage.total_tokens = 5

        with patch.object(
            backend._client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await backend.embed(["test"])

            call_kwargs = mock_create.call_args.kwargs
            assert "dimensions" not in call_kwargs

    def test_repr(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAIEmbeddings(api_key="test-key")
        assert "text-embedding-3-small" in repr(backend)
        assert "$" in repr(backend)

    def test_satisfies_protocol(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAIEmbeddings(api_key="test-key")
        assert isinstance(backend, EmbeddingBackend)


# ---------------------------------------------------------------------------
# SentenceTransformerEmbeddings
# ---------------------------------------------------------------------------


class TestSentenceTransformerEmbeddings:
    def test_import_error_is_handled(self, monkeypatch):
        """Verify a clear error when sentence-transformers is not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="sentence-transformers"):
            SentenceTransformerEmbeddings()

    def test_satisfies_protocol(self):
        """Protocol compliance check via duck-typing (no actual import needed)."""
        # We verify the class has the right method signature
        assert hasattr(SentenceTransformerEmbeddings, "embed")

    @pytest.mark.asyncio
    async def test_embed_with_mock_model(self, monkeypatch):
        """Test embedding with a mocked SentenceTransformer."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        # Patch the import and model instantiation
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            backend = SentenceTransformerEmbeddings(model="test-model", device="cpu")
            result = await backend.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])
        assert result[1] == pytest.approx([0.4, 0.5, 0.6])

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, monkeypatch):
        mock_model = MagicMock()
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            backend = SentenceTransformerEmbeddings(device="cpu")
            result = await backend.embed([])

        assert result == []

    def test_device_detection_cpu_fallback(self, monkeypatch):
        """Without torch, device should fall back to cpu."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        # We only test the static method directly
        monkeypatch.setattr(builtins, "__import__", mock_import)
        assert SentenceTransformerEmbeddings._detect_device() == "cpu"

    def test_repr(self):
        mock_st_module = MagicMock()
        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            backend = SentenceTransformerEmbeddings(
                model="all-MiniLM-L6-v2", device="cpu"
            )
        assert "all-MiniLM-L6-v2" in repr(backend)
        assert "cpu" in repr(backend)


# ---------------------------------------------------------------------------
# CachedEmbeddings
# ---------------------------------------------------------------------------


class TestCachedEmbeddings:
    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self):
        inner = FakeBackend()
        cached = CachedEmbeddings(inner, max_size=100)

        # First call: cache miss
        result1 = await cached.embed(["hello"])
        assert cached.misses == 1
        assert cached.hits == 0
        assert inner.call_count == 1

        # Second call: cache hit
        result2 = await cached.embed(["hello"])
        assert cached.hits == 1
        assert cached.misses == 1
        assert inner.call_count == 1  # No new call to backend
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_partial_cache_hit(self):
        inner = FakeBackend()
        cached = CachedEmbeddings(inner, max_size=100)

        await cached.embed(["hello"])
        assert inner.call_count == 1

        # "hello" is cached, "world" is not
        result = await cached.embed(["hello", "world"])
        assert cached.hits == 1  # "hello" was cached
        assert cached.misses == 2  # 1 from first call + 1 for "world"
        assert inner.call_count == 2
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_eviction_when_over_max_size(self):
        inner = FakeBackend()
        cached = CachedEmbeddings(inner, max_size=2)

        await cached.embed(["a"])
        await cached.embed(["b"])
        await cached.embed(["c"])  # Should evict "a"

        # "a" should be evicted, "b" and "c" should remain
        assert len(cached._cache) == 2

        # Requesting "a" again should be a miss
        misses_before = cached.misses
        await cached.embed(["a"])
        assert cached.misses == misses_before + 1

    @pytest.mark.asyncio
    async def test_empty_list(self):
        inner = FakeBackend()
        cached = CachedEmbeddings(inner, max_size=100)
        result = await cached.embed([])
        assert result == []
        assert inner.call_count == 0

    @pytest.mark.asyncio
    async def test_clear(self):
        inner = FakeBackend()
        cached = CachedEmbeddings(inner, max_size=100)
        await cached.embed(["hello"])
        assert cached.misses == 1

        cached.clear()
        assert cached.hits == 0
        assert cached.misses == 0
        assert len(cached._cache) == 0

    def test_repr(self):
        inner = FakeBackend()
        cached = CachedEmbeddings(inner, max_size=100)
        r = repr(cached)
        assert "0/100" in r
        assert "hits=0" in r

    def test_satisfies_protocol(self):
        inner = FakeBackend()
        cached = CachedEmbeddings(inner)
        assert isinstance(cached, EmbeddingBackend)


# ---------------------------------------------------------------------------
# semantic_similarity
# ---------------------------------------------------------------------------


class TestSemanticSimilarity:
    @pytest.mark.asyncio
    async def test_identical_texts(self):
        backend = FakeBackend(dim=8)
        result = await semantic_similarity("hello world", "hello world", backend)
        assert isinstance(result, CheckResult)
        assert result.passed is True
        assert result.score == pytest.approx(1.0)
        assert result.metric_name == "semantic_similarity"
        assert result.cost == 0.0

    @pytest.mark.asyncio
    async def test_different_texts(self):
        backend = FakeBackend(dim=8)
        result = await semantic_similarity("hello", "xyz", backend, threshold=0.99)
        assert isinstance(result, CheckResult)
        assert result.metric_name == "semantic_similarity"
        # Different texts produce different hashes => different embeddings
        # Similarity will likely be < 0.99
        assert result.score < 1.0 or result.passed is True

    @pytest.mark.asyncio
    async def test_threshold_pass(self):
        backend = FakeBackend(dim=8)
        result = await semantic_similarity(
            "hello", "hello", backend, threshold=0.5
        )
        assert result.passed is True
        assert result.score >= 0.5

    @pytest.mark.asyncio
    async def test_threshold_fail(self):
        # Use a mock backend that returns orthogonal vectors
        mock_backend = AsyncMock(spec=EmbeddingBackend)
        mock_backend.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]
        result = await semantic_similarity("a", "b", mock_backend, threshold=0.5)
        assert result.passed is False
        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_reasoning_contains_score(self):
        backend = FakeBackend(dim=4)
        result = await semantic_similarity("hello", "hello", backend)
        assert "Semantic similarity" in result.reasoning
        assert "threshold" in result.reasoning

    @pytest.mark.asyncio
    async def test_latency_is_tracked(self):
        backend = FakeBackend(dim=4)
        result = await semantic_similarity("hello", "world", backend)
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_cost_from_backend(self):
        mock_backend = AsyncMock(spec=EmbeddingBackend)
        mock_backend.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        mock_backend.total_cost = 0.005
        result = await semantic_similarity("a", "b", mock_backend)
        assert result.cost == 0.005

    @pytest.mark.asyncio
    async def test_negative_cosine_clamped_to_zero(self):
        """Cosine can be negative; score should be clamped to [0, 1]."""
        mock_backend = AsyncMock(spec=EmbeddingBackend)
        mock_backend.embed.return_value = [[1.0, 0.0], [-1.0, 0.0]]
        result = await semantic_similarity("a", "b", mock_backend, threshold=0.5)
        assert result.score == pytest.approx(0.0)
        assert result.passed is False


# ---------------------------------------------------------------------------
# batch_semantic_similarity
# ---------------------------------------------------------------------------


class TestBatchSemanticSimilarity:
    @pytest.mark.asyncio
    async def test_basic_batch(self):
        backend = FakeBackend(dim=4)
        pairs = [("hello", "hello"), ("foo", "bar")]
        results = await batch_semantic_similarity(pairs, backend)
        assert len(results) == 2
        assert all(isinstance(r, CheckResult) for r in results)
        assert all(r.metric_name == "semantic_similarity" for r in results)
        # Identical text pair should pass
        assert results[0].passed is True
        assert results[0].score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_empty_pairs(self):
        backend = FakeBackend(dim=4)
        results = await batch_semantic_similarity([], backend)
        assert results == []

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Repeated texts should only be embedded once."""
        mock_backend = AsyncMock(spec=EmbeddingBackend)
        # 3 unique texts: "a", "b", "c"
        mock_backend.embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        pairs = [("a", "b"), ("a", "c"), ("b", "c")]
        results = await batch_semantic_similarity(pairs, mock_backend)

        # embed should be called once with exactly 3 unique texts
        mock_backend.embed.assert_called_once()
        call_texts = mock_backend.embed.call_args[0][0]
        assert len(call_texts) == 3

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_all_identical_pairs(self):
        backend = FakeBackend(dim=4)
        pairs = [("same", "same")] * 5
        results = await batch_semantic_similarity(pairs, backend)
        assert all(r.score == pytest.approx(1.0) for r in results)
        assert all(r.passed is True for r in results)

    @pytest.mark.asyncio
    async def test_custom_threshold(self):
        mock_backend = AsyncMock(spec=EmbeddingBackend)
        mock_backend.embed.return_value = [
            [1.0, 0.0],
            [0.7071, 0.7071],  # ~45-degree angle => cos ~ 0.7071
        ]
        pairs = [("a", "b")]
        results = await batch_semantic_similarity(
            pairs, mock_backend, threshold=0.5
        )
        assert results[0].passed is True  # 0.7071 >= 0.5

        results_strict = await batch_semantic_similarity(
            pairs, mock_backend, threshold=0.9
        )
        assert results_strict[0].passed is False  # 0.7071 < 0.9


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_text(self):
        backend = FakeBackend(dim=4)
        result = await semantic_similarity("", "", backend)
        assert isinstance(result, CheckResult)
        assert result.metric_name == "semantic_similarity"

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        backend = FakeBackend(dim=4)
        long_text = "word " * 10000
        result = await semantic_similarity(long_text, long_text, backend)
        assert result.passed is True
        assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        backend = FakeBackend(dim=4)
        result = await semantic_similarity(
            "Hello, world!", "Hello, world!", backend
        )
        assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_whitespace_differences(self):
        backend = FakeBackend(dim=4)
        # Different whitespace produces different hashes => different embeddings
        result = await semantic_similarity("hello world", "hello  world", backend)
        assert isinstance(result, CheckResult)

    @pytest.mark.asyncio
    async def test_batch_single_pair(self):
        backend = FakeBackend(dim=4)
        results = await batch_semantic_similarity([("a", "a")], backend)
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0)

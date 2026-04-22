"""Embedding-based semantic similarity for checkllm.

Provides embedding backends (OpenAI, SentenceTransformers), caching,
and high-level functions for computing semantic similarity between texts.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections import OrderedDict
from typing import Protocol, runtime_checkable

from checkllm.models import CheckResult

# ---------------------------------------------------------------------------
# Embedding pricing tables (USD per token)
# ---------------------------------------------------------------------------

_OPENAI_EMBEDDING_PRICES: dict[str, float] = {
    "text-embedding-3-small": 0.02 / 1_000_000,
    "text-embedding-3-large": 0.13 / 1_000_000,
    "text-embedding-ada-002": 0.10 / 1_000_000,
}

_DEFAULT_EMBEDDING_PRICE = 0.10 / 1_000_000


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol that all embedding backends must satisfy."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return their embedding vectors."""
        ...


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


class OpenAIEmbeddings:
    """Embedding backend using OpenAI's embedding API.

    Supports ``text-embedding-3-small`` (default), ``text-embedding-3-large``,
    and ``text-embedding-ada-002``.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        import openai

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key=.")
        self.model = model
        self.dimensions = dimensions
        self.total_cost = 0.0
        self.total_tokens = 0
        self._client = openai.AsyncOpenAI(api_key=resolved_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via the OpenAI embeddings API."""
        if not texts:
            return []

        kwargs: dict = {"input": texts, "model": self.model}
        # text-embedding-3-* models support custom dimensions
        if self.dimensions is not None and self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dimensions

        response = await self._client.embeddings.create(**kwargs)

        # Track cost
        usage_tokens = response.usage.total_tokens
        self.total_tokens += usage_tokens
        price_per_token = _OPENAI_EMBEDDING_PRICES.get(self.model, _DEFAULT_EMBEDDING_PRICE)
        self.total_cost += usage_tokens * price_per_token

        # Return embeddings in input order
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]

    def __repr__(self) -> str:
        return f"OpenAIEmbeddings(model={self.model!r}, total_cost=${self.total_cost:.6f})"


# ---------------------------------------------------------------------------
# SentenceTransformers backend
# ---------------------------------------------------------------------------


class SentenceTransformerEmbeddings:
    """Embedding backend using sentence-transformers (local inference).

    Requires the ``sentence-transformers`` package (optional dependency).
    Cost is always zero since inference runs locally.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        if device is None:
            device = self._detect_device()

        self.model_name = model
        self.device = device
        self._model = SentenceTransformer(model, device=device)

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts locally using sentence-transformers."""
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbeddings(model={self.model_name!r}, device={self.device!r})"


# ---------------------------------------------------------------------------
# Cached wrapper
# ---------------------------------------------------------------------------


class CachedEmbeddings:
    """Caching wrapper around any :class:`EmbeddingBackend`.

    Uses an in-memory LRU cache keyed by a SHA-256 hash of each text.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        max_size: int = 4096,
    ) -> None:
        self._backend = backend
        self._max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, returning cached results where available."""
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[str] = []
        indices_to_embed: list[int] = []

        for i, text in enumerate(texts):
            key = self._hash_text(text)
            if key in self._cache:
                self._cache.move_to_end(key)
                results[i] = self._cache[key]
                self.hits += 1
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                self.misses += 1

        if texts_to_embed:
            new_embeddings = await self._backend.embed(texts_to_embed)
            for idx, text, emb in zip(indices_to_embed, texts_to_embed, new_embeddings):
                key = self._hash_text(text)
                self._cache[key] = emb
                # Evict oldest entries if over capacity
                while len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)
                results[idx] = emb

        return results  # type: ignore[return-value]

    def clear(self) -> None:
        """Clear the cache and reset counters."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def __repr__(self) -> str:
        return (
            f"CachedEmbeddings(size={len(self._cache)}/{self._max_size}, "
            f"hits={self.hits}, misses={self.misses})"
        )


# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value between -1.0 and 1.0. Returns 0.0 for zero-magnitude
    vectors.
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have the same length, got {len(a)} and {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# High-level semantic similarity functions
# ---------------------------------------------------------------------------


async def semantic_similarity(
    output: str,
    expected: str,
    backend: EmbeddingBackend,
    threshold: float = 0.8,
) -> CheckResult:
    """Check semantic similarity between output and expected text.

    Embeds both texts using the given backend, computes cosine similarity,
    and returns a :class:`CheckResult`.
    """
    start = time.perf_counter()
    embeddings = await backend.embed([output, expected])
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    score = cosine_similarity(embeddings[0], embeddings[1])
    # Clamp score to [0, 1] for the CheckResult (cosine can be negative)
    clamped = max(0.0, min(1.0, score))
    passed = clamped >= threshold

    # Infer cost from backend if available
    cost = getattr(backend, "total_cost", 0.0)

    return CheckResult(
        passed=passed,
        score=clamped,
        reasoning=(f"Semantic similarity: {score:.4f} (threshold: {threshold})"),
        cost=cost,
        latency_ms=elapsed_ms,
        metric_name="semantic_similarity",
    )


async def batch_semantic_similarity(
    pairs: list[tuple[str, str]],
    backend: EmbeddingBackend,
    threshold: float = 0.8,
) -> list[CheckResult]:
    """Compute semantic similarity for multiple (output, expected) pairs.

    Efficiently batch-embeds all unique texts, then computes pairwise
    cosine similarities.
    """
    if not pairs:
        return []

    start = time.perf_counter()

    # Collect all unique texts to avoid redundant embeddings
    unique_texts: list[str] = []
    text_to_index: dict[str, int] = {}
    for output_text, expected_text in pairs:
        for text in (output_text, expected_text):
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)

    # Single batched embed call
    all_embeddings = await backend.embed(unique_texts)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    per_pair_ms = max(1, elapsed_ms // len(pairs)) if pairs else 0

    cost = getattr(backend, "total_cost", 0.0)

    results: list[CheckResult] = []
    for output_text, expected_text in pairs:
        emb_a = all_embeddings[text_to_index[output_text]]
        emb_b = all_embeddings[text_to_index[expected_text]]
        score = cosine_similarity(emb_a, emb_b)
        clamped = max(0.0, min(1.0, score))
        passed = clamped >= threshold
        results.append(
            CheckResult(
                passed=passed,
                score=clamped,
                reasoning=(f"Semantic similarity: {score:.4f} (threshold: {threshold})"),
                cost=cost,
                latency_ms=per_pair_ms,
                metric_name="semantic_similarity",
            )
        )

    return results

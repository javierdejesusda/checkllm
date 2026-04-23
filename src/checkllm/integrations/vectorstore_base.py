"""Shared primitives for vector-store connectors.

This module defines the :class:`RetrievedContext` value object and the
:class:`VectorStoreConnector` protocol used by concrete backends (Pinecone,
Weaviate, Milvus, Chroma, ...). The concrete backends live in sibling modules
and all return normalized ``list[RetrievedContext]`` from their ``query``
method so downstream checkllm metrics can consume them uniformly.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class RetrievedContext(BaseModel):
    """Normalised hit returned by any vector-store connector.

    Attributes:
        id: Stable identifier of the stored document or chunk.
        text: Text payload associated with the vector (may be empty if the
            store does not persist raw text).
        score: Similarity score. Cosine-like scores are preferred; stores
            that expose a distance are converted to ``1 - distance`` when a
            natural upper bound exists, otherwise the raw value is returned.
        metadata: Free-form metadata dictionary copied verbatim from the
            backend response.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    text: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class VectorStoreConnector(Protocol):
    """Protocol shared by every checkllm vector-store connector.

    Implementations are free to accept additional keyword arguments but must
    provide at minimum a ``connect(**config)`` constructor-ish method and a
    ``query(vector_or_text, top_k)`` method returning normalised results.
    """

    def connect(self, **config: Any) -> Any: ...

    def query(
        self,
        vector_or_text: Any,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievedContext]: ...


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Convert arbitrary numeric-ish values to ``float`` without raising."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_metadata(value: Any) -> dict[str, Any]:
    """Ensure metadata is a plain ``dict`` (pydantic-friendly)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {"_raw": list(value)}
    return {"_raw": value}


__all__ = [
    "RetrievedContext",
    "VectorStoreConnector",
    "_coerce_float",
    "_coerce_metadata",
]

"""Milvus vector-store connector for checkllm.

Uses the ``pymilvus`` SDK (via ``MilvusClient`` when available, falling back to
the low-level ``connections`` / ``Collection`` API). Exposes the uniform
``connect`` / ``query`` surface defined in
:mod:`checkllm.integrations.vectorstore_base`.

Install with ``pip install checkllm[vectorstores]`` (or
``pip install pymilvus``).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from checkllm.integrations.vectorstore_base import (
    RetrievedContext,
    _coerce_float,
    _coerce_metadata,
)

try:  # pragma: no cover - import guard
    from pymilvus import MilvusClient as _MilvusClientSDK

    _MilvusClient: Any = _MilvusClientSDK
    _HAS_MILVUS = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    _MilvusClient = None
    _HAS_MILVUS = False


_INSTALL_HINT = (
    "pymilvus is required for MilvusConnector. "
    "Install with: pip install 'checkllm[vectorstores]' or pip install pymilvus"
)


Embedder = Callable[[str], Sequence[float]]


class MilvusConnector:
    """Connector for a Milvus collection via ``MilvusClient``.

    The connector does not bundle an embedding model. Pass an ``embedder``
    callable when querying with raw text.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._collection_name: str | None = None
        self._vector_field: str = "vector"
        self._text_field: str = "text"
        self._output_fields: list[str] | None = None
        self._embedder: Embedder | None = None

    def connect(
        self,
        *,
        uri: str | None = None,
        token: str | None = None,
        collection_name: str | None = None,
        vector_field: str = "vector",
        text_field: str = "text",
        output_fields: list[str] | None = None,
        embedder: Embedder | None = None,
        client: Any | None = None,
        **_: Any,
    ) -> "MilvusConnector":
        """Bind the connector to a Milvus collection.

        Args:
            uri: Milvus server URI (e.g. ``http://localhost:19530``). Ignored
                when ``client`` is provided.
            token: Optional auth token. Ignored when ``client`` is provided.
            collection_name: Target collection name.
            vector_field: Name of the vector column.
            text_field: Name of the payload text column.
            output_fields: Optional explicit output-field list. Defaults to
                ``[text_field]``.
            embedder: Optional text-to-vector callable.
            client: Pre-built ``MilvusClient`` (handy for tests).

        Returns:
            ``self`` for chaining.

        Raises:
            ImportError: If ``pymilvus`` is not installed and no ``client``
                is supplied.
            ValueError: If ``collection_name`` is omitted.
        """
        if client is None:
            if not _HAS_MILVUS:
                raise ImportError(_INSTALL_HINT)
            kwargs: dict[str, Any] = {}
            if uri is not None:
                kwargs["uri"] = uri
            if token is not None:
                kwargs["token"] = token
            client = _MilvusClient(**kwargs)

        if not collection_name:
            raise ValueError("collection_name is required")

        self._client = client
        self._collection_name = collection_name
        self._vector_field = vector_field
        self._text_field = text_field
        self._output_fields = output_fields or [text_field]
        self._embedder = embedder
        return self

    def _ensure_connected(self) -> None:
        if self._client is None:
            raise RuntimeError("MilvusConnector.connect(...) must be called before query()")

    def _to_vector(
        self,
        vector_or_text: Any,
        embedder: Embedder | None,
    ) -> list[float]:
        if isinstance(vector_or_text, str):
            fn = embedder or self._embedder
            if fn is None:
                raise ValueError(
                    "Text query supplied but no embedder is configured. "
                    "Pass embedder=... to connect() or query()."
                )
            return list(fn(vector_or_text))
        return [float(x) for x in vector_or_text]

    def query(
        self,
        vector_or_text: Any,
        top_k: int = 5,
        *,
        embedder: Embedder | None = None,
        filter_expr: str | None = None,
        **_: Any,
    ) -> list[RetrievedContext]:
        """Query the collection and return normalised ``RetrievedContext`` hits."""
        self._ensure_connected()
        assert self._collection_name is not None
        vector = self._to_vector(vector_or_text, embedder)

        raw = self._client.search(
            collection_name=self._collection_name,
            data=[vector],
            anns_field=self._vector_field,
            limit=top_k,
            output_fields=self._output_fields,
            filter=filter_expr,
        )
        hits = _flatten_hits(raw)
        return [self._hit_to_context(h) for h in hits]

    def _hit_to_context(self, hit: Any) -> RetrievedContext:
        if isinstance(hit, dict):
            raw_id = hit.get("id") or hit.get("pk") or ""
            distance = hit.get("distance")
            score = hit.get("score", distance)
            entity = hit.get("entity") or hit
        else:
            raw_id = getattr(hit, "id", None) or getattr(hit, "pk", "") or ""
            distance = getattr(hit, "distance", None)
            score = getattr(hit, "score", None) or distance
            entity = getattr(hit, "entity", None) or hit

        metadata = _coerce_metadata(entity)
        text_val = metadata.pop(self._text_field, "") if isinstance(metadata, dict) else ""
        # Keep the score / distance visible in metadata for callers.
        if distance is not None and "distance" not in metadata:
            metadata["distance"] = _coerce_float(distance)

        return RetrievedContext(
            id=str(raw_id),
            text=str(text_val) if text_val is not None else "",
            score=_coerce_float(score),
            metadata=metadata,
        )


def _flatten_hits(raw: Any) -> list[Any]:
    """Milvus returns ``List[List[hit]]`` (one sub-list per query vector)."""
    if raw is None:
        return []
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        # one search vector in, one group out
        return list(raw[0])
    if isinstance(raw, list):
        return list(raw)
    return []


__all__ = ["MilvusConnector"]

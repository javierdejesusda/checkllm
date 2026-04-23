"""Weaviate vector-store connector for checkllm.

Thin wrapper around the ``weaviate-client`` SDK that exposes the uniform
``connect`` / ``query`` surface defined in
:mod:`checkllm.integrations.vectorstore_base`.

Supports both textual (``near_text``) and raw-vector (``near_vector``) queries
against a named collection / class.

Install with ``pip install checkllm[vectorstores]`` (or
``pip install weaviate-client``).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from checkllm.integrations.vectorstore_base import (
    RetrievedContext,
    _coerce_float,
    _coerce_metadata,
)

try:  # pragma: no cover - import guard
    import weaviate as _weaviate_sdk

    _weaviate: Any = _weaviate_sdk
    _HAS_WEAVIATE = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    _weaviate = None
    _HAS_WEAVIATE = False


_INSTALL_HINT = (
    "weaviate-client is required for WeaviateConnector. "
    "Install with: pip install 'checkllm[vectorstores]' or pip install weaviate-client"
)


Embedder = Callable[[str], Sequence[float]]


class WeaviateConnector:
    """Connector for a Weaviate collection (v4 client) / class (v3 client).

    ``query`` accepts either a raw vector (list / numpy array-like) or a
    string. String queries are delegated to Weaviate's ``near_text`` module
    when the server supports it; otherwise an ``embedder`` callable can be
    supplied to compute the vector locally.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._collection_name: str | None = None
        self._text_property: str = "text"
        self._embedder: Embedder | None = None

    def connect(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        text_property: str = "text",
        embedder: Embedder | None = None,
        client: Any | None = None,
        **_: Any,
    ) -> "WeaviateConnector":
        """Bind the connector to a Weaviate collection.

        Args:
            url: Weaviate server URL. Ignored when ``client`` is provided.
            api_key: Optional API key. Ignored when ``client`` is provided.
            collection_name: Name of the target collection / class.
            text_property: Property name that stores the raw text payload.
            embedder: Optional fallback embedder for ``near_vector`` queries.
            client: Pre-built client (handy for tests).

        Returns:
            ``self`` for chaining.

        Raises:
            ImportError: If ``weaviate-client`` is not installed and no
                ``client`` was supplied.
            ValueError: If ``collection_name`` is omitted.
        """
        if client is None:
            if not _HAS_WEAVIATE:
                raise ImportError(_INSTALL_HINT)
            kwargs: dict[str, Any] = {}
            if url is not None:
                kwargs["url"] = url
            if api_key is not None:
                kwargs["auth_client_secret"] = api_key
            client = _weaviate.Client(**kwargs)

        if not collection_name:
            raise ValueError("collection_name is required")

        self._client = client
        self._collection_name = collection_name
        self._text_property = text_property
        self._embedder = embedder
        return self

    def _ensure_connected(self) -> None:
        if self._client is None:
            raise RuntimeError("WeaviateConnector.connect(...) must be called before query()")

    def query(
        self,
        vector_or_text: Any,
        top_k: int = 5,
        *,
        embedder: Embedder | None = None,
        properties: list[str] | None = None,
        **_: Any,
    ) -> list[RetrievedContext]:
        """Query the collection and return normalised ``RetrievedContext`` hits."""
        self._ensure_connected()
        assert self._collection_name is not None

        hits: list[Any]
        if isinstance(vector_or_text, str):
            embed_fn = embedder or self._embedder
            if embed_fn is not None:
                vector = [float(x) for x in embed_fn(vector_or_text)]
                hits = self._query_vector(vector, top_k)
            else:
                hits = self._query_text(vector_or_text, top_k)
        else:
            vector = [float(x) for x in vector_or_text]
            hits = self._query_vector(vector, top_k)

        return [self._hit_to_context(h) for h in hits]

    def _query_vector(self, vector: list[float], top_k: int) -> list[Any]:
        client = self._client
        collections = getattr(client, "collections", None)
        if collections is not None and hasattr(collections, "get"):
            coll = collections.get(self._collection_name)
            return list(
                coll.query.near_vector(
                    near_vector=vector,
                    limit=top_k,
                    return_metadata=["score", "distance"],
                ).objects
            )
        q = client.query.get(self._collection_name, [self._text_property])
        result = (
            q.with_near_vector({"vector": vector})
            .with_limit(top_k)
            .with_additional(["id", "score", "distance"])
            .do()
        )
        assert self._collection_name is not None
        return _extract_legacy_objects(result, self._collection_name)

    def _query_text(self, text: str, top_k: int) -> list[Any]:
        client = self._client
        collections = getattr(client, "collections", None)
        if collections is not None and hasattr(collections, "get"):
            coll = collections.get(self._collection_name)
            return list(
                coll.query.near_text(
                    query=text,
                    limit=top_k,
                    return_metadata=["score", "distance"],
                ).objects
            )
        q = client.query.get(self._collection_name, [self._text_property])
        result = (
            q.with_near_text({"concepts": [text]})
            .with_limit(top_k)
            .with_additional(["id", "score", "distance"])
            .do()
        )
        assert self._collection_name is not None
        return _extract_legacy_objects(result, self._collection_name)

    def _hit_to_context(self, hit: Any) -> RetrievedContext:
        if isinstance(hit, dict):
            props = hit.get("properties") or hit
            metadata = _coerce_metadata(hit.get("metadata") or {})
            raw_id = hit.get("id") or props.get("id") or ""
            score = hit.get("score")
            if score is None:
                score = hit.get("_additional", {}).get("score") or hit.get("_additional", {}).get(
                    "distance"
                )
            text_val = props.get(self._text_property) or ""
        else:
            props = getattr(hit, "properties", {}) or {}
            metadata_obj = getattr(hit, "metadata", None)
            metadata = {}
            if metadata_obj is not None:
                for attr in ("score", "distance", "certainty"):
                    val = getattr(metadata_obj, attr, None)
                    if val is not None:
                        metadata[attr] = val
            raw_id = getattr(hit, "uuid", None) or getattr(hit, "id", "") or ""
            score = metadata.get("score") or metadata.get("certainty")
            if score is None and "distance" in metadata:
                dist = _coerce_float(metadata["distance"])
                score = max(0.0, 1.0 - dist)
            text_val = props.get(self._text_property, "") if isinstance(props, dict) else ""

        return RetrievedContext(
            id=str(raw_id),
            text=str(text_val) if text_val is not None else "",
            score=_coerce_float(score),
            metadata=dict(metadata) if isinstance(metadata, dict) else {"_raw": metadata},
        )


def _extract_legacy_objects(result: Any, collection_name: str) -> list[Any]:
    """Extract objects from a v3 GraphQL-shaped response."""
    if not isinstance(result, dict):
        return []
    data = result.get("data", {}) or {}
    get = data.get("Get", {}) or {}
    return list(get.get(collection_name, []) or [])


__all__ = ["WeaviateConnector"]

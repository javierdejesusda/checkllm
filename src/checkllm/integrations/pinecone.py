"""Pinecone vector-store connector for checkllm.

Thin wrapper around the ``pinecone`` SDK that exposes a uniform
``connect`` / ``query`` surface returning normalised
:class:`~checkllm.integrations.vectorstore_base.RetrievedContext` objects.

Usage::

    from checkllm.integrations.pinecone import PineconeConnector

    store = PineconeConnector()
    store.connect(api_key="...", index_name="my-index")
    hits = store.query("what is python?", top_k=5, embedder=my_embed_fn)

Install with ``pip install checkllm[vectorstores]`` (or ``pip install pinecone``).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from checkllm.integrations.vectorstore_base import (
    RetrievedContext,
    _coerce_float,
    _coerce_metadata,
)

try:  # pragma: no cover - import guard
    import pinecone as _pinecone_sdk

    _pinecone: Any = _pinecone_sdk
    _HAS_PINECONE = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    _pinecone = None
    _HAS_PINECONE = False


_INSTALL_HINT = (
    "pinecone SDK is required for PineconeConnector. "
    "Install with: pip install 'checkllm[vectorstores]' or pip install pinecone"
)


Embedder = Callable[[str], Sequence[float]]


class PineconeConnector:
    """Connector for a Pinecone serverless / pod-based index.

    The connector does not bundle an embedding model. For ``query`` calls that
    pass raw text, supply an ``embedder`` callable (``str -> list[float]``)
    either at connect-time or per-call.
    """

    def __init__(self) -> None:
        self._index: Any = None
        self._index_name: str | None = None
        self._namespace: str | None = None
        self._embedder: Embedder | None = None
        self._client: Any = None

    def connect(
        self,
        *,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
        embedder: Embedder | None = None,
        client: Any | None = None,
        **_: Any,
    ) -> "PineconeConnector":
        """Bind the connector to a Pinecone index.

        Args:
            api_key: Pinecone API key. Forwarded to ``Pinecone(api_key=...)``
                when a ``client`` is not supplied.
            index_name: Name of the target index.
            namespace: Optional Pinecone namespace scoping queries.
            embedder: Optional text-to-vector callable used when callers pass
                raw text to ``query``.
            client: Optional pre-built Pinecone client (handy for tests).

        Returns:
            ``self`` for chaining.

        Raises:
            ImportError: If the ``pinecone`` SDK is not installed and no
                ``client`` object is provided.
            ValueError: If ``index_name`` is omitted.
        """
        if client is None:
            if not _HAS_PINECONE:
                raise ImportError(_INSTALL_HINT)
            client = _pinecone.Pinecone(api_key=api_key)

        if not index_name:
            raise ValueError("index_name is required")

        self._client = client
        self._index_name = index_name
        self._namespace = namespace
        self._embedder = embedder
        self._index = client.Index(index_name)
        return self

    def _ensure_connected(self) -> None:
        if self._index is None:
            raise RuntimeError("PineconeConnector.connect(...) must be called before query()")

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
        namespace: str | None = None,
        embedder: Embedder | None = None,
        filter: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[RetrievedContext]:
        """Query the index and return normalised ``RetrievedContext`` hits."""
        self._ensure_connected()
        vector = self._to_vector(vector_or_text, embedder)

        resp = self._index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
            namespace=namespace or self._namespace,
            filter=filter,
        )

        matches = _extract_matches(resp)
        return [_match_to_context(m) for m in matches]


def _extract_matches(resp: Any) -> list[Any]:
    """Extract the ``matches`` list from either dict or object responses."""
    if resp is None:
        return []
    if isinstance(resp, dict):
        return list(resp.get("matches", []) or [])
    matches = getattr(resp, "matches", None)
    if matches is None:
        return []
    return list(matches)


def _match_to_context(match: Any) -> RetrievedContext:
    """Normalise a single Pinecone match into a :class:`RetrievedContext`."""
    if isinstance(match, dict):
        raw_id = match.get("id", "")
        score = match.get("score", 0.0)
        metadata = match.get("metadata") or {}
    else:
        raw_id = getattr(match, "id", "")
        score = getattr(match, "score", 0.0)
        metadata = getattr(match, "metadata", None) or {}

    metadata = _coerce_metadata(metadata)
    text_val = metadata.get("text") or metadata.get("content") or metadata.get("chunk_text") or ""

    return RetrievedContext(
        id=str(raw_id),
        text=str(text_val),
        score=_coerce_float(score),
        metadata=metadata,
    )


__all__ = ["PineconeConnector"]

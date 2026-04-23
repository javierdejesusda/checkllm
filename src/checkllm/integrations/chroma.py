"""Chroma vector-store connector for checkllm.

Thin wrapper around ``chromadb`` that exposes the uniform ``connect`` /
``query`` surface defined in
:mod:`checkllm.integrations.vectorstore_base`.

Install with ``pip install checkllm[vectorstores]`` (or
``pip install chromadb``).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from checkllm.integrations.vectorstore_base import (
    RetrievedContext,
    _coerce_float,
    _coerce_metadata,
)

try:  # pragma: no cover - import guard
    import chromadb as _chromadb_sdk

    _chromadb: Any = _chromadb_sdk
    _HAS_CHROMA = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    _chromadb = None
    _HAS_CHROMA = False


_INSTALL_HINT = (
    "chromadb is required for ChromaConnector. "
    "Install with: pip install 'checkllm[vectorstores]' or pip install chromadb"
)


Embedder = Callable[[str], Sequence[float]]


class ChromaConnector:
    """Connector for a Chroma collection.

    Chroma can embed text itself when the collection is created with an
    embedding function, so passing raw text to ``query`` is supported
    natively. A user-supplied ``embedder`` is only needed when the collection
    has no default embedding function.
    """

    def __init__(self) -> None:
        self._collection: Any = None
        self._collection_name: str | None = None
        self._embedder: Embedder | None = None
        self._client: Any = None

    def connect(
        self,
        *,
        collection_name: str | None = None,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        embedder: Embedder | None = None,
        client: Any | None = None,
        collection: Any | None = None,
        **_: Any,
    ) -> "ChromaConnector":
        """Bind the connector to a Chroma collection.

        Args:
            collection_name: Name of the target collection. Required unless
                ``collection`` is passed directly.
            persist_directory: Local persistence directory (``PersistentClient``).
                Ignored when ``client`` or ``collection`` is supplied.
            host: Server host (``HttpClient``). Ignored when ``client`` or
                ``collection`` is supplied.
            port: Server port.
            embedder: Optional text-to-vector callable used for ``query_embeddings``
                when the collection has no default embedding function.
            client: Pre-built Chroma client (handy for tests).
            collection: Pre-built Chroma collection (bypasses the client).

        Returns:
            ``self`` for chaining.

        Raises:
            ImportError: If ``chromadb`` is not installed and no ``client``
                / ``collection`` object is provided.
            ValueError: If no collection target can be resolved.
        """
        if collection is not None:
            self._collection = collection
            self._collection_name = getattr(collection, "name", collection_name)
            self._embedder = embedder
            return self

        if client is None:
            if not _HAS_CHROMA:
                raise ImportError(_INSTALL_HINT)
            if host is not None:
                client = _chromadb.HttpClient(host=host, port=port or 8000)
            elif persist_directory is not None:
                client = _chromadb.PersistentClient(path=persist_directory)
            else:
                client = _chromadb.Client()

        if not collection_name:
            raise ValueError("collection_name is required")

        self._client = client
        self._collection_name = collection_name
        self._collection = client.get_or_create_collection(name=collection_name)
        self._embedder = embedder
        return self

    def _ensure_connected(self) -> None:
        if self._collection is None:
            raise RuntimeError("ChromaConnector.connect(...) must be called before query()")

    def query(
        self,
        vector_or_text: Any,
        top_k: int = 5,
        *,
        embedder: Embedder | None = None,
        where: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[RetrievedContext]:
        """Query the collection and return normalised ``RetrievedContext`` hits."""
        self._ensure_connected()
        assert self._collection is not None

        kwargs: dict[str, Any] = {
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        if isinstance(vector_or_text, str):
            fn = embedder or self._embedder
            if fn is not None:
                kwargs["query_embeddings"] = [list(fn(vector_or_text))]
            else:
                kwargs["query_texts"] = [vector_or_text]
        else:
            kwargs["query_embeddings"] = [[float(x) for x in vector_or_text]]

        result = self._collection.query(**kwargs)
        return _result_to_contexts(result)


def _result_to_contexts(result: Any) -> list[RetrievedContext]:
    """Flatten a Chroma query result (list-of-lists) to ``RetrievedContext``."""
    if not isinstance(result, dict):
        return []

    ids_batch = result.get("ids") or []
    docs_batch = result.get("documents") or []
    meta_batch = result.get("metadatas") or []
    dist_batch = result.get("distances") or []

    ids = ids_batch[0] if ids_batch else []
    docs = docs_batch[0] if docs_batch else []
    metas = meta_batch[0] if meta_batch else []
    dists = dist_batch[0] if dist_batch else []

    out: list[RetrievedContext] = []
    for idx, raw_id in enumerate(ids):
        text_val = docs[idx] if idx < len(docs) else ""
        meta = _coerce_metadata(metas[idx] if idx < len(metas) else {})
        distance = dists[idx] if idx < len(dists) else None
        score = _distance_to_score(distance)
        if distance is not None:
            meta.setdefault("distance", _coerce_float(distance))
        out.append(
            RetrievedContext(
                id=str(raw_id),
                text=str(text_val) if text_val is not None else "",
                score=score,
                metadata=meta,
            )
        )
    return out


def _distance_to_score(distance: Any) -> float:
    """Convert a Chroma distance to a ``[0, 1]`` similarity score.

    Chroma's default cosine distance lives in ``[0, 2]``; we clip to that
    range and return ``1 - distance / 2`` so identical vectors map to ``1``
    and antipodal ones to ``0``. Callers can recover the raw distance from
    ``metadata["distance"]`` when precision matters.
    """
    if distance is None:
        return 0.0
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if d < 0:
        d = 0.0
    if d > 2:
        d = 2.0
    return 1.0 - d / 2.0


__all__ = ["ChromaConnector"]

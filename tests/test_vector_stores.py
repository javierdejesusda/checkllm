"""Tests for the vector-store connectors.

All tests use hand-rolled fakes. No real Pinecone / Weaviate / Milvus / Chroma
SDK or network call is required.
"""

from __future__ import annotations

from typing import Any

import pytest

from checkllm.integrations import chroma as chroma_mod
from checkllm.integrations import milvus as milvus_mod
from checkllm.integrations import pinecone as pinecone_mod
from checkllm.integrations import weaviate as weaviate_mod
from checkllm.integrations.chroma import ChromaConnector
from checkllm.integrations.milvus import MilvusConnector
from checkllm.integrations.pinecone import PineconeConnector
from checkllm.integrations.vectorstore_base import RetrievedContext
from checkllm.integrations.weaviate import WeaviateConnector


# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------


class _FakePineconeIndex:
    def __init__(self, matches: list[Any]) -> None:
        self.matches = matches
        self.last_call: dict[str, Any] | None = None

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.last_call = kwargs
        return {"matches": self.matches}


class _FakePineconeClient:
    def __init__(self, index: _FakePineconeIndex) -> None:
        self._index = index

    def Index(self, name: str) -> _FakePineconeIndex:  # noqa: N802 (matches SDK)
        return self._index


def test_pinecone_connect_requires_index_name() -> None:
    client = _FakePineconeClient(_FakePineconeIndex([]))
    with pytest.raises(ValueError, match="index_name"):
        PineconeConnector().connect(client=client)


def test_pinecone_connect_without_sdk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pinecone_mod, "_HAS_PINECONE", False)
    with pytest.raises(ImportError, match="pinecone SDK"):
        PineconeConnector().connect(index_name="idx")


def test_pinecone_query_text_requires_embedder() -> None:
    idx = _FakePineconeIndex([])
    conn = PineconeConnector().connect(client=_FakePineconeClient(idx), index_name="idx")
    with pytest.raises(ValueError, match="embedder"):
        conn.query("hello world", top_k=3)


def test_pinecone_query_vector_normalises_matches() -> None:
    matches = [
        {
            "id": "doc-1",
            "score": 0.91,
            "metadata": {"text": "Python is a language", "section": "intro"},
        },
        {
            "id": "doc-2",
            "score": 0.72,
            "metadata": {"content": "Rust is systems-oriented"},
        },
    ]
    idx = _FakePineconeIndex(matches)
    conn = PineconeConnector().connect(
        client=_FakePineconeClient(idx),
        index_name="idx",
        namespace="ns",
    )
    out = conn.query([0.1, 0.2, 0.3], top_k=2)
    assert [c.id for c in out] == ["doc-1", "doc-2"]
    assert out[0].text == "Python is a language"
    assert out[0].score == pytest.approx(0.91)
    assert out[0].metadata["section"] == "intro"
    assert out[1].text == "Rust is systems-oriented"
    assert idx.last_call == {
        "vector": [0.1, 0.2, 0.3],
        "top_k": 2,
        "include_metadata": True,
        "include_values": False,
        "namespace": "ns",
        "filter": None,
    }


def test_pinecone_query_text_uses_embedder() -> None:
    matches = [{"id": "a", "score": 0.5, "metadata": {"text": "hi"}}]
    idx = _FakePineconeIndex(matches)

    def fake_embed(text: str) -> list[float]:
        assert text == "hello"
        return [0.0, 1.0, 0.0]

    conn = PineconeConnector().connect(
        client=_FakePineconeClient(idx),
        index_name="idx",
        embedder=fake_embed,
    )
    out = conn.query("hello", top_k=1)
    assert len(out) == 1
    assert idx.last_call is not None
    assert idx.last_call["vector"] == [0.0, 1.0, 0.0]


def test_pinecone_query_before_connect_raises() -> None:
    with pytest.raises(RuntimeError, match="connect"):
        PineconeConnector().query([0.1, 0.2], top_k=1)


def test_pinecone_object_style_match() -> None:
    class _Match:
        def __init__(self, _id: str, score: float, metadata: dict[str, Any]) -> None:
            self.id = _id
            self.score = score
            self.metadata = metadata

    idx = _FakePineconeIndex([_Match("x", 0.42, {"text": "xyz"})])
    conn = PineconeConnector().connect(client=_FakePineconeClient(idx), index_name="idx")
    out = conn.query([0.1], top_k=1)
    assert out[0].id == "x"
    assert out[0].text == "xyz"
    assert out[0].score == pytest.approx(0.42)
    assert out[0].metadata == {"text": "xyz"}


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------


class _FakeWeaviateLegacyClient:
    """Mimics the v3 GraphQL-style Weaviate client surface we rely on."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.query = self  # type: ignore[assignment]
        self._data = data
        self._collection: str | None = None
        self._selected: list[str] = []
        self._near_vector: list[float] | None = None
        self._near_text: list[str] | None = None
        self._limit: int | None = None

    # Fluent v3 builder API
    def get(self, collection: str, fields: list[str]) -> "_FakeWeaviateLegacyClient":
        self._collection = collection
        self._selected = fields
        return self

    def with_near_vector(self, payload: dict[str, Any]) -> "_FakeWeaviateLegacyClient":
        self._near_vector = payload["vector"]
        return self

    def with_near_text(self, payload: dict[str, Any]) -> "_FakeWeaviateLegacyClient":
        self._near_text = payload["concepts"]
        return self

    def with_limit(self, n: int) -> "_FakeWeaviateLegacyClient":
        self._limit = n
        return self

    def with_additional(self, fields: list[str]) -> "_FakeWeaviateLegacyClient":
        return self

    def do(self) -> dict[str, Any]:
        return self._data


def test_weaviate_connect_requires_collection() -> None:
    with pytest.raises(ValueError, match="collection_name"):
        WeaviateConnector().connect(client=object())


def test_weaviate_connect_without_sdk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weaviate_mod, "_HAS_WEAVIATE", False)
    with pytest.raises(ImportError, match="weaviate-client"):
        WeaviateConnector().connect(collection_name="Docs")


def test_weaviate_legacy_query_vector() -> None:
    payload = {
        "data": {
            "Get": {
                "Docs": [
                    {
                        "text": "first chunk",
                        "_additional": {"id": "doc-a", "score": 0.77},
                        "id": "doc-a",
                    },
                    {
                        "text": "second chunk",
                        "_additional": {"id": "doc-b", "distance": 0.2},
                        "id": "doc-b",
                    },
                ]
            }
        }
    }
    client = _FakeWeaviateLegacyClient(payload)
    conn = WeaviateConnector().connect(client=client, collection_name="Docs")
    out = conn.query([0.1, 0.2], top_k=2)
    assert [c.id for c in out] == ["doc-a", "doc-b"]
    assert out[0].text == "first chunk"
    assert out[0].score == pytest.approx(0.77)
    assert client._near_vector == [0.1, 0.2]
    assert client._limit == 2


def test_weaviate_legacy_query_text() -> None:
    payload = {"data": {"Get": {"Docs": []}}}
    client = _FakeWeaviateLegacyClient(payload)
    conn = WeaviateConnector().connect(client=client, collection_name="Docs")
    out = conn.query("what is python", top_k=1)
    assert out == []
    assert client._near_text == ["what is python"]


def test_weaviate_query_before_connect_raises() -> None:
    with pytest.raises(RuntimeError, match="connect"):
        WeaviateConnector().query([0.1, 0.2], top_k=1)


def test_weaviate_v4_shape() -> None:
    """Exercise the v4 ``collections.get(...).query.near_vector`` path."""

    class _Metadata:
        def __init__(self, score: float) -> None:
            self.score = score

    class _Obj:
        def __init__(self, uuid: str, text: str, score: float | None = None) -> None:
            self.uuid = uuid
            self.properties = {"text": text}
            self.metadata = _Metadata(score) if score is not None else None

    class _Result:
        def __init__(self, objs: list[_Obj]) -> None:
            self.objects = objs

    class _Query:
        def __init__(self, objs: list[_Obj]) -> None:
            self._objs = objs
            self.last: dict[str, Any] | None = None

        def near_vector(self, **kwargs: Any) -> _Result:
            self.last = kwargs
            return _Result(self._objs)

        def near_text(self, **kwargs: Any) -> _Result:
            self.last = kwargs
            return _Result(self._objs)

    class _Collection:
        def __init__(self, objs: list[_Obj]) -> None:
            self.query = _Query(objs)

    class _Collections:
        def __init__(self, coll: _Collection) -> None:
            self._coll = coll

        def get(self, name: str) -> _Collection:
            return self._coll

    class _V4Client:
        def __init__(self, objs: list[_Obj]) -> None:
            self.collections = _Collections(_Collection(objs))

    client = _V4Client([_Obj("u-1", "hello world", 0.88)])
    conn = WeaviateConnector().connect(client=client, collection_name="Docs")
    out = conn.query([0.1, 0.2, 0.3], top_k=3)
    assert len(out) == 1
    assert out[0].id == "u-1"
    assert out[0].text == "hello world"
    assert out[0].score == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------------------


class _FakeMilvusClient:
    def __init__(self, hits: list[list[dict[str, Any]]]) -> None:
        self._hits = hits
        self.last_call: dict[str, Any] | None = None

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.last_call = kwargs
        return self._hits


def test_milvus_connect_requires_collection() -> None:
    with pytest.raises(ValueError, match="collection_name"):
        MilvusConnector().connect(client=_FakeMilvusClient([[]]))


def test_milvus_connect_without_sdk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(milvus_mod, "_HAS_MILVUS", False)
    with pytest.raises(ImportError, match="pymilvus"):
        MilvusConnector().connect(collection_name="docs")


def test_milvus_query_text_requires_embedder() -> None:
    conn = MilvusConnector().connect(client=_FakeMilvusClient([[]]), collection_name="docs")
    with pytest.raises(ValueError, match="embedder"):
        conn.query("hello", top_k=2)


def test_milvus_query_normalises_hits() -> None:
    client = _FakeMilvusClient(
        [
            [
                {
                    "id": 1,
                    "distance": 0.1,
                    "entity": {"text": "first doc", "tag": "a"},
                },
                {
                    "id": 2,
                    "distance": 0.4,
                    "entity": {"text": "second doc", "tag": "b"},
                },
            ]
        ]
    )
    conn = MilvusConnector().connect(client=client, collection_name="docs", text_field="text")
    out = conn.query([0.1, 0.2], top_k=2)
    assert [c.id for c in out] == ["1", "2"]
    assert out[0].text == "first doc"
    assert out[0].metadata["tag"] == "a"
    assert out[0].metadata["distance"] == pytest.approx(0.1)
    assert client.last_call is not None
    assert client.last_call["collection_name"] == "docs"
    assert client.last_call["limit"] == 2


def test_milvus_query_before_connect_raises() -> None:
    with pytest.raises(RuntimeError, match="connect"):
        MilvusConnector().query([0.1], top_k=1)


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------


class _FakeChromaCollection:
    def __init__(self, name: str, result: dict[str, Any]) -> None:
        self.name = name
        self._result = result
        self.last_call: dict[str, Any] | None = None

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.last_call = kwargs
        return self._result


def test_chroma_connect_without_sdk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chroma_mod, "_HAS_CHROMA", False)
    with pytest.raises(ImportError, match="chromadb"):
        ChromaConnector().connect(collection_name="docs")


def test_chroma_query_vector_normalises_result() -> None:
    result = {
        "ids": [["doc-a", "doc-b"]],
        "documents": [["first", "second"]],
        "metadatas": [[{"k": 1}, {"k": 2}]],
        "distances": [[0.0, 2.0]],
    }
    coll = _FakeChromaCollection("docs", result)
    conn = ChromaConnector().connect(collection=coll)
    out = conn.query([0.1, 0.2], top_k=2)
    assert [c.id for c in out] == ["doc-a", "doc-b"]
    # distance 0.0 -> score 1.0; distance 2.0 -> score 0.0
    assert out[0].score == pytest.approx(1.0)
    assert out[1].score == pytest.approx(0.0)
    assert out[0].metadata["k"] == 1
    assert out[0].metadata["distance"] == pytest.approx(0.0)
    assert coll.last_call is not None
    assert coll.last_call["n_results"] == 2


def test_chroma_query_text_uses_query_texts_when_no_embedder() -> None:
    result = {
        "ids": [["x"]],
        "documents": [["chunk"]],
        "metadatas": [[{}]],
        "distances": [[0.5]],
    }
    coll = _FakeChromaCollection("docs", result)
    conn = ChromaConnector().connect(collection=coll)
    out = conn.query("my question", top_k=1)
    assert len(out) == 1
    assert coll.last_call is not None
    assert coll.last_call.get("query_texts") == ["my question"]


def test_chroma_query_text_uses_embedder_when_supplied() -> None:
    result = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    coll = _FakeChromaCollection("docs", result)

    def fake_embed(text: str) -> list[float]:
        return [1.0, 2.0]

    conn = ChromaConnector().connect(collection=coll, embedder=fake_embed)
    conn.query("q", top_k=1)
    assert coll.last_call is not None
    assert coll.last_call.get("query_embeddings") == [[1.0, 2.0]]
    assert "query_texts" not in coll.last_call


def test_chroma_query_before_connect_raises() -> None:
    with pytest.raises(RuntimeError, match="connect"):
        ChromaConnector().query([0.1], top_k=1)


# ---------------------------------------------------------------------------
# RetrievedContext
# ---------------------------------------------------------------------------


def test_retrieved_context_defaults() -> None:
    c = RetrievedContext(id="x")
    assert c.text == ""
    assert c.score == 0.0
    assert c.metadata == {}

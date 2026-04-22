"""Tests for retriever-level framework integrations.

Covers ``checkllm.integrations.langchain_retriever`` and
``checkllm.integrations.llamaindex_retriever``. Tests that require the target
framework installed are gated behind ``pytest.importorskip``. The
``RetrievalEvalResult`` pydantic model is tested without any framework.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from checkllm.datasets.case import Case
from checkllm.integrations.langchain_retriever import (
    RetrievalEvalResult,
    _resolve_metrics,
)
from checkllm.models import CheckResult, JudgeResponse


class StubJudge:
    """Minimal stub implementing the JudgeBackend protocol for tests."""

    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.calls: list[tuple[str, str | None]] = []
        self.model = "stub-judge"
        self.total_cost = 0.0
        self.last_cost = 0.0

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        self.calls.append((prompt, system_prompt))
        return JudgeResponse(score=self.score, reasoning="stubbed", cost=0.0)


def test_retrieval_eval_result_defaults():
    """RetrievalEvalResult should instantiate with sensible defaults."""
    result = RetrievalEvalResult(query="q", retrieved_docs=["a", "b"])
    assert result.query == "q"
    assert result.retrieved_docs == ["a", "b"]
    assert result.expected_context is None
    assert result.metrics == {}
    assert result.pass_rate == 0.0
    assert result.total_cost == 0.0
    assert result.mean_score() == 0.0


def test_retrieval_eval_result_mean_score():
    """mean_score should average CheckResult scores."""
    metrics = {
        "a": CheckResult(
            passed=True, score=0.9, reasoning="", cost=0.0, latency_ms=0, metric_name="a"
        ),
        "b": CheckResult(
            passed=False, score=0.5, reasoning="", cost=0.0, latency_ms=0, metric_name="b"
        ),
    }
    result = RetrievalEvalResult(query="q", retrieved_docs=[], metrics=metrics, pass_rate=0.5)
    assert result.mean_score() == pytest.approx(0.7)


def test_resolve_metrics_unknown_name_raises():
    """_resolve_metrics should reject unknown shortnames with a helpful message."""
    with pytest.raises(ValueError) as exc:
        _resolve_metrics(["does_not_exist"], judge=None)
    msg = str(exc.value)
    assert "does_not_exist" in msg
    assert "context_precision" in msg


def test_resolve_metrics_non_llm_no_judge_ok():
    """Non-LLM metrics should resolve without a judge."""
    metrics = _resolve_metrics(["nonllm_context_precision", "nonllm_context_recall"], judge=None)
    assert set(metrics.keys()) == {
        "nonllm_context_precision",
        "nonllm_context_recall",
    }


def test_resolve_metrics_llm_metric_requires_judge():
    """LLM-based metrics should raise when judge=None."""
    with pytest.raises(ValueError):
        _resolve_metrics(["context_precision"], judge=None)


langchain_core = pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

from checkllm.integrations.langchain_retriever import (  # noqa: E402
    CheckllmRetrieverWrapper as LangChainWrapper,
)
from checkllm.integrations.langchain_retriever import (  # noqa: E402
    evaluate_retriever as lc_run_eval,
)


class _FakeLangChainRetriever(BaseRetriever):
    """Returns a fixed list of Documents for any query."""

    docs: list[Document] = []

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return list(self.docs)

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return list(self.docs)


def _make_fake_lc_retriever(texts: list[str]) -> _FakeLangChainRetriever:
    r = _FakeLangChainRetriever()
    r.docs = [Document(page_content=t) for t in texts]
    return r


def test_langchain_wrapper_returns_docs_unchanged():
    """Wrapper should return the same documents as the underlying retriever."""
    inner = _make_fake_lc_retriever(["Python is a language."])
    wrapper = LangChainWrapper(
        inner,
        metrics=["nonllm_context_precision"],
        expected_context="Python is a language.",
    )
    out = wrapper.invoke("what is python?")
    assert len(out) == 1
    assert out[0].page_content == "Python is a language."


def test_langchain_wrapper_records_per_retrieval():
    """Each invocation should append to wrapper.results."""
    inner = _make_fake_lc_retriever(["Python is a language."])
    wrapper = LangChainWrapper(
        inner,
        metrics=["nonllm_context_precision"],
        expected_context="Python is a language.",
    )
    wrapper.invoke("q1")
    wrapper.invoke("q2")
    assert len(wrapper.results) == 2
    assert wrapper.results[0].query == "q1"
    assert wrapper.results[1].query == "q2"
    assert "nonllm_context_precision" in wrapper.results[0].metrics


def test_langchain_wrapper_on_retrieval_hook_fires():
    """on_retrieval callback should be called per retrieval with docs + metrics."""
    inner = _make_fake_lc_retriever(["Python."])
    captured: list[tuple[int, set[str]]] = []

    def hook(docs: list[Any], metrics: dict[str, CheckResult]) -> None:
        captured.append((len(docs), set(metrics.keys())))

    wrapper = LangChainWrapper(
        inner,
        metrics=["nonllm_context_precision"],
        on_retrieval=hook,
        expected_context="Python.",
    )
    wrapper.invoke("q")
    wrapper.invoke("q2")
    assert len(captured) == 2
    assert captured[0][0] == 1
    assert "nonllm_context_precision" in captured[0][1]


def test_langchain_wrapper_summary_aggregates():
    """summary() should aggregate per-metric mean/median/pass_rate/count."""
    inner = _make_fake_lc_retriever(["Python is a language."])
    wrapper = LangChainWrapper(
        inner,
        metrics=["nonllm_context_precision", "nonllm_context_recall"],
        expected_context="Python is a language.",
    )
    for _ in range(3):
        wrapper.invoke("q")
    summary = wrapper.summary()
    assert set(summary.keys()) == {
        "nonllm_context_precision",
        "nonllm_context_recall",
    }
    for stats in summary.values():
        assert stats["count"] == 3
        assert 0.0 <= stats["pass_rate"] <= 1.0
        assert 0.0 <= stats["mean"] <= 1.0


def test_langchain_wrapper_async_path():
    """Async retrieval should also populate results."""
    inner = _make_fake_lc_retriever(["Python is a language."])
    wrapper = LangChainWrapper(
        inner,
        metrics=["nonllm_context_precision"],
        expected_context="Python is a language.",
    )

    async def _go() -> list[Any]:
        return await wrapper.ainvoke("q")

    out = asyncio.run(_go())
    assert len(out) == 1
    assert len(wrapper.results) == 1


def test_langchain_wrapper_unknown_metric_raises():
    """Constructor should reject unknown metric shortnames."""
    inner = _make_fake_lc_retriever(["x"])
    with pytest.raises(ValueError):
        LangChainWrapper(inner, metrics=["bogus_metric"])


def test_langchain_wrapper_with_stub_judge_calls_llm_metric():
    """When a judge is supplied, LLM-based metrics should call it."""
    inner = _make_fake_lc_retriever(["Python is a language."])
    judge = StubJudge(score=0.9)
    wrapper = LangChainWrapper(
        inner,
        metrics=["context_relevance"],
        judge=judge,
        expected_context="Python.",
    )
    wrapper.invoke("what is python?")
    assert len(judge.calls) >= 1
    assert "context_relevance" in wrapper.results[0].metrics


def test_langchain_evaluate_retriever_one_shot():
    """evaluate_retriever should return one result per Case."""
    inner = _make_fake_lc_retriever(["Python is a language."])
    cases = [
        Case(input="q1", context="Python is a language."),
        Case(input="q2", context="Python is a language."),
    ]

    async def _go() -> list[RetrievalEvalResult]:
        return await lc_run_eval(inner, cases=cases, metrics=["nonllm_context_precision"])

    out = asyncio.run(_go())
    assert len(out) == 2
    assert out[0].query == "q1"
    assert out[1].query == "q2"
    assert "nonllm_context_precision" in out[0].metrics


def test_langchain_evaluate_retriever_unknown_metric():
    """evaluate_retriever should reject unknown metrics."""
    inner = _make_fake_lc_retriever(["x"])

    async def _go() -> None:
        await lc_run_eval(inner, cases=[], metrics=["nope"])

    with pytest.raises(ValueError):
        asyncio.run(_go())


def test_langchain_wrapper_pass_rate_all_pass():
    """pass_rate should equal 1.0 when every metric passes."""
    inner = _make_fake_lc_retriever(["Python is a language."])
    wrapper = LangChainWrapper(
        inner,
        metrics=["nonllm_context_precision", "nonllm_context_recall"],
        expected_context="Python is a language.",
    )
    wrapper.invoke("q")
    record = wrapper.results[0]
    assert record.pass_rate == pytest.approx(1.0)


def test_langchain_wrapper_pass_rate_empty_docs():
    """With empty retrieval, nonllm metrics score 0 and pass_rate is 0."""
    inner = _make_fake_lc_retriever([])
    wrapper = LangChainWrapper(
        inner,
        metrics=["nonllm_context_precision", "nonllm_context_recall"],
        expected_context="Python is a language.",
    )
    wrapper.invoke("q")
    record = wrapper.results[0]
    assert record.pass_rate == pytest.approx(0.0)


@pytest.fixture(scope="module")
def _llamaindex_wrapper_module() -> Any:
    pytest.importorskip("llama_index.core")
    from checkllm.integrations import llamaindex_retriever

    return llamaindex_retriever


def test_llamaindex_wrapper_returns_nodes_unchanged(_llamaindex_wrapper_module: Any):
    """Wrapper should return the same nodes as the underlying retriever."""
    from llama_index.core.retrievers import BaseRetriever as LiBase
    from llama_index.core.schema import NodeWithScore, TextNode

    class _FakeLiRetriever(LiBase):
        def _retrieve(self, query_bundle: Any) -> list[NodeWithScore]:
            return [NodeWithScore(node=TextNode(text="Python is a language."), score=0.9)]

    inner = _FakeLiRetriever()
    Wrapper = _llamaindex_wrapper_module.CheckllmRetrieverWrapper
    wrapper = Wrapper(
        inner,
        metrics=["nonllm_context_precision"],
        expected_context="Python is a language.",
    )
    out = wrapper.retrieve("q")
    assert len(out) == 1
    assert len(wrapper.results) == 1
    assert wrapper.results[0].retrieved_docs[0] == "Python is a language."


def test_llamaindex_wrapper_async_and_hook(_llamaindex_wrapper_module: Any):
    """Async retrieval populates results; hook fires with node list."""
    from llama_index.core.retrievers import BaseRetriever as LiBase
    from llama_index.core.schema import NodeWithScore, TextNode

    class _FakeLiRetriever(LiBase):
        def _retrieve(self, query_bundle: Any) -> list[NodeWithScore]:
            return [NodeWithScore(node=TextNode(text="Python."), score=0.9)]

        async def _aretrieve(self, query_bundle: Any) -> list[NodeWithScore]:
            return [NodeWithScore(node=TextNode(text="Python."), score=0.9)]

    inner = _FakeLiRetriever()
    Wrapper = _llamaindex_wrapper_module.CheckllmRetrieverWrapper
    captured: list[int] = []

    def hook(nodes: list[Any], metrics: dict[str, CheckResult]) -> None:
        captured.append(len(nodes))

    wrapper = Wrapper(
        inner,
        metrics=["nonllm_context_precision"],
        on_retrieval=hook,
        expected_context="Python.",
    )

    async def _go() -> list[Any]:
        return await wrapper.aretrieve("q")

    out = asyncio.run(_go())
    assert len(out) == 1
    assert len(wrapper.results) == 1
    assert captured == [1]


def test_llamaindex_wrapper_unknown_metric(_llamaindex_wrapper_module: Any):
    """Constructor should reject unknown metric shortnames."""
    from llama_index.core.retrievers import BaseRetriever as LiBase

    class _FakeLiRetriever(LiBase):
        def _retrieve(self, query_bundle: Any) -> list[Any]:
            return []

    Wrapper = _llamaindex_wrapper_module.CheckllmRetrieverWrapper
    with pytest.raises(ValueError):
        Wrapper(_FakeLiRetriever(), metrics=["bogus"])


def test_llamaindex_evaluate_retriever_one_shot(_llamaindex_wrapper_module: Any):
    """evaluate_retriever helper returns one result per Case."""
    from llama_index.core.retrievers import BaseRetriever as LiBase
    from llama_index.core.schema import NodeWithScore, TextNode

    class _FakeLiRetriever(LiBase):
        def _retrieve(self, query_bundle: Any) -> list[NodeWithScore]:
            return [NodeWithScore(node=TextNode(text="Python is a language."), score=0.9)]

    cases = [
        Case(input="q1", context="Python is a language."),
        Case(input="q2", context="Python is a language."),
    ]

    async def _go() -> list[RetrievalEvalResult]:
        return await _llamaindex_wrapper_module.evaluate_retriever(
            _FakeLiRetriever(),
            cases=cases,
            metrics=["nonllm_context_precision"],
        )

    out = asyncio.run(_go())
    assert len(out) == 2
    assert out[0].query == "q1"


def test_llamaindex_wrapper_summary(_llamaindex_wrapper_module: Any):
    """summary() should include each metric over multiple retrievals."""
    from llama_index.core.retrievers import BaseRetriever as LiBase
    from llama_index.core.schema import NodeWithScore, TextNode

    class _FakeLiRetriever(LiBase):
        def _retrieve(self, query_bundle: Any) -> list[NodeWithScore]:
            return [NodeWithScore(node=TextNode(text="Python is a language."), score=0.9)]

    Wrapper = _llamaindex_wrapper_module.CheckllmRetrieverWrapper
    wrapper = Wrapper(
        _FakeLiRetriever(),
        metrics=["nonllm_context_precision", "nonllm_context_recall"],
        expected_context="Python is a language.",
    )
    for _ in range(2):
        wrapper.retrieve("q")
    summary = wrapper.summary()
    assert summary["nonllm_context_precision"]["count"] == 2
    assert summary["nonllm_context_recall"]["count"] == 2

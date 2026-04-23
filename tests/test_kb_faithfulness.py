"""Tests for the KB faithfulness metric.

Uses a stub judge and an in-memory fake store. No network or SDK required.
"""

from __future__ import annotations

from typing import Any

import pytest

from checkllm.integrations.vectorstore_base import RetrievedContext
from checkllm.metrics.kb_faithfulness import KBFaithfulnessMetric
from checkllm.models import JudgeResponse


class _StubJudge:
    def __init__(self, score: float = 0.9) -> None:
        self._score = score
        self.prompts: list[str] = []

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        self.prompts.append(prompt)
        return JudgeResponse(score=self._score, reasoning="stub", cost=0.0)


class _FakeStore:
    def __init__(self, hits: list[RetrievedContext]) -> None:
        self._hits = hits
        self.queries: list[tuple[Any, int]] = []

    def query(self, vector_or_text: Any, top_k: int = 5, **kwargs: Any) -> list[RetrievedContext]:
        self.queries.append((vector_or_text, top_k))
        return self._hits[:top_k]


class _KwargTrackingStore:
    def __init__(self, hits: list[RetrievedContext]) -> None:
        self._hits = hits
        self.captured: dict[str, Any] = {}

    def query(self, vector_or_text: Any, top_k: int = 5, **kwargs: Any) -> list[RetrievedContext]:
        self.captured = dict(kwargs)
        return self._hits[:top_k]


@pytest.mark.asyncio
async def test_kb_faithfulness_retrieves_and_scores() -> None:
    hits = [
        RetrievedContext(id="a", text="Python is a language", score=0.9),
        RetrievedContext(id="b", text="Python appeared in 1991", score=0.8),
    ]
    store = _FakeStore(hits)
    judge = _StubJudge(score=0.95)
    metric = KBFaithfulnessMetric(judge=judge, store=store, top_k=2, threshold=0.8)

    result = await metric.evaluate(
        output="Python is a language released in 1991.",
        query="What is Python?",
    )

    assert result.passed is True
    assert result.score == pytest.approx(0.95)
    assert result.metric_name == "kb_faithfulness"
    assert "[kb_faithfulness]" in result.reasoning
    assert "a" in result.reasoning and "b" in result.reasoning
    assert store.queries == [("What is Python?", 2)]
    assert len(judge.prompts) == 1
    assert "Python is a language" in judge.prompts[0]


@pytest.mark.asyncio
async def test_kb_faithfulness_handles_empty_retrieval() -> None:
    store = _FakeStore([])
    judge = _StubJudge(score=0.9)
    metric = KBFaithfulnessMetric(judge=judge, store=store, top_k=3)

    result = await metric.evaluate(output="Anything goes.", query="Why?")
    assert result.passed is False
    assert result.score == 0.0
    assert "No usable context" in result.reasoning
    # Judge should not be called when no context is available.
    assert judge.prompts == []


@pytest.mark.asyncio
async def test_kb_faithfulness_preconfigured_contexts_bypass_store() -> None:
    store = _FakeStore([])  # would return empty
    judge = _StubJudge(score=0.7)
    metric = KBFaithfulnessMetric(judge=judge, store=store, top_k=5, threshold=0.8)

    ctxs = [RetrievedContext(id="c", text="Supplied context")]
    result = await metric.evaluate(output="Answer", query="Q?", contexts=ctxs)

    # Store is skipped when contexts are passed explicitly.
    assert store.queries == []
    assert result.score == pytest.approx(0.7)
    assert result.passed is False  # below 0.8 threshold
    assert "Supplied context" in judge.prompts[0]


@pytest.mark.asyncio
async def test_kb_faithfulness_forwards_store_kwargs() -> None:
    store = _KwargTrackingStore([RetrievedContext(id="a", text="t")])
    judge = _StubJudge(score=0.9)
    metric = KBFaithfulnessMetric(
        judge=judge,
        store=store,
        top_k=1,
        store_kwargs={"namespace": "ns"},
    )
    await metric.evaluate(output="ok", query="q")
    assert store.captured == {"namespace": "ns"}

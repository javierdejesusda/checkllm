"""End-to-end evaluation of a RAG pipeline.

Scenario: a documentation Q&A agent backed by a tiny in-memory retriever
and a scripted "LLM" that synthesises an answer from retrieved chunks.

The tests exercise the four essentials for RAG:

- ``contextual_recall`` - retrieval surfaced the chunks needed to answer.
- ``contextual_precision`` - relevant chunks are ranked ahead of noise.
- ``faithfulness`` - generated claims trace back to the retrieved context.
- ``relevance`` - the generated answer addresses the query.

A ``MockJudge`` from ``checkllm.testing`` is used so the tests run offline
with no API key. To wire a real judge, swap ``MockJudge`` for
``OpenAIJudge`` / ``AnthropicJudge`` from ``checkllm.judge`` - see
``tests/conftest.py`` and the real fixtures under ``tests/`` for an
end-to-end setup.

Run with: pytest examples/test_rag_pipeline.py
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from checkllm import Case
from checkllm.testing import MockJudge, make_collector


@dataclass(frozen=True)
class Document:
    """A retrievable chunk of knowledge."""

    id: str
    text: str


CORPUS: tuple[Document, ...] = (
    Document(
        id="python-history",
        text=(
            "Python was created by Guido van Rossum and first released in 1991. "
            "Its design philosophy emphasizes code readability."
        ),
    ),
    Document(
        id="python-typing",
        text=(
            "Python added optional type hints in version 3.5 (2015) via PEP 484. "
            "Type hints are checked by external tools such as mypy."
        ),
    ),
    Document(
        id="python-packaging",
        text=(
            "The standard packaging tool for Python is pip; modern projects use "
            "pyproject.toml as defined in PEP 621."
        ),
    ),
    Document(
        id="rust-history",
        text=(
            "Rust was originally developed by Graydon Hoare at Mozilla starting in 2006 "
            "and first reached 1.0 in May 2015."
        ),
    ),
)


def _retrieve(query: str, k: int = 2) -> list[Document]:
    """Toy retriever: rank documents by shared keyword count."""
    query_tokens = {w.lower() for w in query.split() if len(w) > 2}
    scored = [
        (sum(1 for w in doc.text.lower().split() if w in query_tokens), doc) for doc in CORPUS
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:k]]


def _generate(query: str, context_docs: list[Document]) -> str:
    """Deterministic stand-in for an LLM that answers from retrieved context."""
    context_text = " ".join(doc.text for doc in context_docs)
    if "who created python" in query.lower() or "author" in query.lower():
        if "Guido van Rossum" in context_text:
            return "Python was created by Guido van Rossum and first released in 1991."
        return "I could not find the author of Python in the provided context."
    if "type hints" in query.lower():
        if "PEP 484" in context_text:
            return "Optional type hints arrived in Python 3.5 (2015) via PEP 484."
        return "The context does not discuss Python type hints."
    if "rust" in query.lower():
        if "Graydon Hoare" in context_text:
            return "Rust was developed by Graydon Hoare at Mozilla, reaching 1.0 in May 2015."
        return "The context does not discuss Rust."
    return "I do not have enough context to answer that."


@pytest.fixture
def rag_collector() -> object:
    """Collector backed by a MockJudge so tests run offline."""
    judge = MockJudge(default_score=0.9, default_reasoning="Mock grounded answer")
    return make_collector(judge=judge, threshold=0.8)


RAG_CASES: tuple[Case, ...] = (
    Case(
        input="Who created Python?",
        expected="Python was created by Guido van Rossum in 1991.",
        criteria="mentions Guido van Rossum and 1991",
    ),
    Case(
        input="When did Python get type hints?",
        expected="Python added type hints in 3.5 (2015) via PEP 484.",
        criteria="mentions PEP 484 and 3.5",
    ),
)


def test_retrieval_recall(rag_collector) -> None:
    """Retrieval must surface at least one chunk that contains the expected answer."""
    case = RAG_CASES[0]
    docs = _retrieve(case.input, k=2)
    context_chunks = [d.text for d in docs]

    rag_collector.contextual_recall(
        output=_generate(case.input, docs),
        context=context_chunks,
        expected=case.expected,
        threshold=0.8,
    )
    assert all(r.passed for r in rag_collector.results)


def test_retrieval_precision(rag_collector) -> None:
    """Relevant chunks must be ranked ahead of noise chunks."""
    case = RAG_CASES[0]
    docs = _retrieve(case.input, k=3)
    context_chunks = [d.text for d in docs]

    rag_collector.contextual_precision(
        output=_generate(case.input, docs),
        context=context_chunks,
        query=case.input,
        expected=case.expected,
        threshold=0.75,
    )
    assert all(r.passed for r in rag_collector.results)


def test_answer_is_grounded(rag_collector) -> None:
    """Faithfulness: every claim in the answer must trace back to context."""
    case = RAG_CASES[1]
    docs = _retrieve(case.input, k=2)
    context = " ".join(d.text for d in docs)
    answer = _generate(case.input, docs)

    rag_collector.contains(answer, "PEP 484")
    rag_collector.faithfulness(
        output=answer,
        context=context,
        query=case.input,
        threshold=0.85,
    )
    assert all(r.passed for r in rag_collector.results)


def test_answer_is_relevant(rag_collector) -> None:
    """Relevance: answer addresses the query even when grounded."""
    case = RAG_CASES[0]
    docs = _retrieve(case.input, k=2)
    answer = _generate(case.input, docs)

    rag_collector.relevance(
        output=answer,
        query=case.input,
        threshold=0.8,
    )
    assert all(r.passed for r in rag_collector.results)


def test_out_of_scope_question_is_refused(rag_collector) -> None:
    """Abstention check: when context is missing, the agent should not fabricate."""
    query = "What is the capital of Mars?"
    docs = _retrieve(query, k=2)
    answer = _generate(query, docs)

    rag_collector.contains(answer, "not")  # "do not have enough context"
    rag_collector.max_tokens(answer, limit=40)
    assert all(r.passed for r in rag_collector.results)


def test_full_rag_suite_composed(rag_collector) -> None:
    """Compose all four RAG metrics on a single case."""
    case = RAG_CASES[0]
    docs = _retrieve(case.input, k=2)
    context_chunks = [d.text for d in docs]
    context = " ".join(context_chunks)
    answer = _generate(case.input, docs)

    # Deterministic guardrails (free and instant)
    rag_collector.contains(answer, "Guido van Rossum")
    rag_collector.max_tokens(answer, limit=80)

    # Judge-based signals
    rag_collector.contextual_recall(
        output=answer, context=context_chunks, expected=case.expected, threshold=0.8
    )
    rag_collector.contextual_precision(
        output=answer,
        context=context_chunks,
        query=case.input,
        expected=case.expected,
        threshold=0.75,
    )
    rag_collector.faithfulness(output=answer, context=context, query=case.input, threshold=0.85)
    rag_collector.relevance(output=answer, query=case.input, threshold=0.8)

    assert all(r.passed for r in rag_collector.results), rag_collector.results


# Run with: pytest examples/test_rag_pipeline.py

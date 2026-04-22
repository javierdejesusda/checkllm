"""LlamaIndex retriever wrappers for checkllm.

Mirror of :mod:`checkllm.integrations.langchain_retriever` adapted to
LlamaIndex's ``BaseRetriever`` / ``QueryBundle`` / ``NodeWithScore`` APIs.

Usage::

    from checkllm.integrations.llamaindex_retriever import (
        CheckllmRetrieverWrapper,
        evaluate_retriever,
    )

    wrapped = CheckllmRetrieverWrapper(
        retriever=index.as_retriever(),
        metrics=["context_precision", "context_relevance"],
        judge=my_judge,
    )
    nodes = wrapped.retrieve("What is Python?")
    print(wrapped.summary())

Requires: ``pip install llama-index-core`` (only when actually using the wrapper).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from checkllm.datasets.case import Case
from checkllm.integrations.langchain_retriever import (
    RetrievalEvalResult,
    _resolve_metrics,
    _run_sync,
    _score_one,
)
from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.integrations.llamaindex_retriever")

try:
    from llama_index.core.retrievers import BaseRetriever as _LlamaBaseRetriever
    from llama_index.core.schema import NodeWithScore as _LlamaNodeWithScore
    from llama_index.core.schema import QueryBundle as _LlamaQueryBundle

    _HAS_LLAMAINDEX = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    _LlamaBaseRetriever = object  # type: ignore[assignment, misc, unused-ignore]
    _LlamaNodeWithScore = object  # type: ignore[assignment, misc, unused-ignore]
    _LlamaQueryBundle = object  # type: ignore[assignment, misc, unused-ignore]
    _HAS_LLAMAINDEX = False


OnRetrievalHook = Callable[[list[Any], dict[str, CheckResult]], None]


def _extract_node_text(node: Any) -> str:
    """Best-effort text extraction from a LlamaIndex node-like object."""
    inner = getattr(node, "node", node)
    for attr in ("text", "get_content"):
        target = getattr(inner, attr, None)
        if callable(target):
            try:
                return str(target())
            except Exception:
                continue
        if isinstance(target, str):
            return target
    if isinstance(node, str):
        return node
    return str(node)


def _coerce_query(query_or_bundle: Any) -> str:
    """Normalise either a str or a ``QueryBundle`` into a string query."""
    if isinstance(query_or_bundle, str):
        return query_or_bundle
    for attr in ("query_str", "query", "text"):
        val = getattr(query_or_bundle, attr, None)
        if isinstance(val, str) and val:
            return val
    return str(query_or_bundle)


class CheckllmRetrieverWrapper(_LlamaBaseRetriever):  # type: ignore[misc, valid-type, unused-ignore]
    """Wraps a LlamaIndex ``BaseRetriever`` with online retrieval scoring.

    Every call to ``retrieve`` / ``aretrieve`` runs the configured metrics
    against the retrieved nodes' text content and stores the result in
    ``self.results``. Nodes are returned unchanged so the wrapper is a
    drop-in replacement for the underlying retriever.

    Args:
        retriever: Any LlamaIndex retriever instance.
        metrics: List of metric shortnames (see the langchain_retriever module
            for the supported set).
        on_retrieval: Optional callback fired after each retrieval+scoring
            cycle with the raw node list and the metric results.
        judge: Judge backend for LLM-based metrics.
        expected_context: Fallback ground-truth context when no per-query value
            is available.

    Raises:
        ImportError: If ``llama-index-core`` is not installed.
        ValueError: If an unknown metric shortname is supplied, or a metric
            requires a judge but none was provided.
    """

    def __init__(
        self,
        retriever: Any,
        metrics: list[str] | None = None,
        on_retrieval: OnRetrievalHook | None = None,
        judge: JudgeBackend | None = None,
        expected_context: str | None = None,
    ) -> None:
        if not _HAS_LLAMAINDEX:
            raise ImportError(
                "llama-index-core is required for CheckllmRetrieverWrapper. "
                "Install with: pip install llama-index-core"
            )

        metric_names = metrics or [
            "context_precision",
            "context_recall",
            "context_relevance",
        ]
        resolved = _resolve_metrics(metric_names, judge)

        try:
            super().__init__()
        except Exception:
            object.__init__(self)

        self._inner_retriever = retriever
        self._metric_names = metric_names
        self._metrics = resolved
        self._on_retrieval = on_retrieval
        self._judge = judge
        self._default_expected = expected_context
        self.results: list[RetrievalEvalResult] = []

    @property
    def inner_retriever(self) -> Any:
        """Return the underlying LlamaIndex retriever."""
        return self._inner_retriever

    def _call_inner_sync(self, query_str: str) -> list[Any]:
        inner = self._inner_retriever
        if hasattr(inner, "retrieve"):
            return list(inner.retrieve(query_str))
        if hasattr(inner, "_retrieve"):
            if _HAS_LLAMAINDEX:
                bundle = _LlamaQueryBundle(query_str=query_str)
                return list(inner._retrieve(bundle))
            return list(inner._retrieve(query_str))
        raise TypeError("Underlying retriever does not expose retrieve / _retrieve")

    async def _call_inner_async(self, query_str: str) -> list[Any]:
        inner = self._inner_retriever
        if hasattr(inner, "aretrieve"):
            return list(await inner.aretrieve(query_str))
        if hasattr(inner, "_aretrieve"):
            if _HAS_LLAMAINDEX:
                bundle = _LlamaQueryBundle(query_str=query_str)
                return list(await inner._aretrieve(bundle))
            return list(await inner._aretrieve(query_str))
        return await asyncio.to_thread(self._call_inner_sync, query_str)

    def _retrieve(self, query_bundle: Any) -> list[Any]:
        """Synchronously retrieve nodes, score metrics, return nodes unchanged."""
        query_str = _coerce_query(query_bundle)
        nodes = self._call_inner_sync(query_str)
        texts = [_extract_node_text(n) for n in nodes]
        record = _run_sync(
            _score_one(
                query=query_str,
                docs_text=texts,
                expected_context=self._default_expected,
                metrics=self._metrics,
            )
        )
        self.results.append(record)
        if self._on_retrieval is not None:
            try:
                self._on_retrieval(nodes, record.metrics)
            except Exception as exc:
                logger.debug("on_retrieval hook raised: %s", exc)
        return nodes

    async def _aretrieve(self, query_bundle: Any) -> list[Any]:
        """Asynchronously retrieve nodes, score metrics, return nodes unchanged."""
        query_str = _coerce_query(query_bundle)
        nodes = await self._call_inner_async(query_str)
        texts = [_extract_node_text(n) for n in nodes]
        record = await _score_one(
            query=query_str,
            docs_text=texts,
            expected_context=self._default_expected,
            metrics=self._metrics,
        )
        self.results.append(record)
        if self._on_retrieval is not None:
            try:
                self._on_retrieval(nodes, record.metrics)
            except Exception as exc:
                logger.debug("on_retrieval hook raised: %s", exc)
        return nodes

    def summary(self) -> dict[str, dict[str, float]]:
        """Aggregate per-metric statistics across all recorded retrievals.

        Returns:
            Mapping ``{metric_name: {"mean", "median", "pass_rate", "count"}}``.
        """
        import statistics

        per_metric: dict[str, list[CheckResult]] = {}
        for record in self.results:
            for name, cr in record.metrics.items():
                per_metric.setdefault(name, []).append(cr)

        summary: dict[str, dict[str, float]] = {}
        for name, checks in per_metric.items():
            scores = [c.score for c in checks]
            passes = [c.passed for c in checks]
            summary[name] = {
                "mean": float(statistics.fmean(scores)) if scores else 0.0,
                "median": float(statistics.median(scores)) if scores else 0.0,
                "pass_rate": sum(passes) / len(passes) if passes else 0.0,
                "count": float(len(scores)),
            }
        return summary


async def evaluate_retriever(
    retriever: Any,
    cases: list[Case],
    metrics: list[str] | None = None,
    judge: JudgeBackend | None = None,
) -> list[RetrievalEvalResult]:
    """Run a one-shot retrieval scoring pass over a list of cases.

    Args:
        retriever: Any LlamaIndex-style retriever exposing
            ``aretrieve``/``retrieve`` or ``_aretrieve``/``_retrieve``.
        cases: Evaluation cases. ``case.query`` (or ``case.input``) is the
            retrieval query.
        metrics: Optional metric shortname list.
        judge: Judge backend for LLM-based metrics.

    Returns:
        A list of :class:`RetrievalEvalResult`, in case order.

    Raises:
        ValueError: If an unknown metric shortname is supplied.
        TypeError: If the retriever exposes no recognised retrieval method.
    """
    metric_names = metrics or [
        "context_precision",
        "context_recall",
        "context_relevance",
    ]
    resolved = _resolve_metrics(metric_names, judge)

    results: list[RetrievalEvalResult] = []
    for case in cases:
        query = case.query or case.input
        expected_context = case.context or case.expected

        if hasattr(retriever, "aretrieve"):
            nodes = list(await retriever.aretrieve(query))
        elif hasattr(retriever, "_aretrieve"):
            if _HAS_LLAMAINDEX:
                bundle = _LlamaQueryBundle(query_str=query)
                nodes = list(await retriever._aretrieve(bundle))
            else:
                nodes = list(await retriever._aretrieve(query))
        elif hasattr(retriever, "retrieve"):
            nodes = list(retriever.retrieve(query))
        elif hasattr(retriever, "_retrieve"):
            if _HAS_LLAMAINDEX:
                bundle = _LlamaQueryBundle(query_str=query)
                nodes = list(retriever._retrieve(bundle))
            else:
                nodes = list(retriever._retrieve(query))
        else:
            raise TypeError(
                "retriever must expose retrieve / aretrieve (or _retrieve / _aretrieve)"
            )

        texts = [_extract_node_text(n) for n in nodes]
        record = await _score_one(
            query=query,
            docs_text=texts,
            expected_context=expected_context,
            metrics=resolved,
        )
        results.append(record)

    return results


__all__ = [
    "CheckllmRetrieverWrapper",
    "RetrievalEvalResult",
    "evaluate_retriever",
]

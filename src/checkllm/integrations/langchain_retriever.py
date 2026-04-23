"""LangChain retriever wrappers for checkllm.

Evaluate retrieval before the LLM sees the context. This module provides a
drop-in wrapper around any LangChain ``BaseRetriever`` that runs contextual
metrics (precision, recall, relevance, etc.) on every retrieval call, plus a
one-shot ``evaluate_retriever`` helper for offline regression scoring.

Usage::

    from checkllm.integrations.langchain_retriever import (
        CheckllmRetrieverWrapper,
        evaluate_retriever,
    )

    wrapped = CheckllmRetrieverWrapper(
        retriever=my_vectorstore.as_retriever(),
        metrics=["context_precision", "context_relevance"],
        judge=my_judge,
    )
    docs = wrapped.invoke("What is Python?")
    print(wrapped.summary())

Requires: ``pip install langchain-core`` (only when actually using the wrapper).
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from checkllm.datasets.case import Case
from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

logger = logging.getLogger("checkllm.integrations.langchain_retriever")

try:
    from langchain_core.retrievers import BaseRetriever as _LangChainBaseRetriever
    from langchain_core.documents import Document as _LangChainDocument

    _HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    _LangChainBaseRetriever = object  # type: ignore[assignment, misc, unused-ignore]
    _LangChainDocument = object  # type: ignore[assignment, misc, unused-ignore]
    _HAS_LANGCHAIN = False


OnRetrievalHook = Callable[[list[Any], dict[str, CheckResult]], None]


class _MetricFactory(Protocol):
    def __call__(self, judge: JudgeBackend | None) -> Any: ...


class RetrievalEvalResult(BaseModel):
    """Per-retrieval scoring record.

    Attributes:
        query: The user query that triggered retrieval.
        retrieved_docs: Document text snippets returned by the retriever.
        expected_context: Optional ground-truth reference answer or context used
            for metrics that require it (e.g. recall, nonllm_context_*).
        metrics: Mapping from metric name to its :class:`CheckResult`.
        pass_rate: Fraction of metrics that passed their threshold, in [0, 1].
        total_cost: Sum of judge API costs across all scored metrics, in USD.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    retrieved_docs: list[str]
    expected_context: str | None = None
    metrics: dict[str, CheckResult] = Field(default_factory=dict)
    pass_rate: float = 0.0
    total_cost: float = 0.0

    def mean_score(self) -> float:
        """Return the mean score across all metrics, or ``0.0`` if empty."""
        if not self.metrics:
            return 0.0
        return statistics.fmean(r.score for r in self.metrics.values())


def _extract_doc_text(doc: Any) -> str:
    """Return the best-effort text representation of a LangChain document."""
    if hasattr(doc, "page_content"):
        return str(doc.page_content)
    if isinstance(doc, str):
        return doc
    return str(doc)


def _build_contextual_precision(judge: JudgeBackend | None) -> Any:
    if judge is None:
        raise ValueError("context_precision requires a judge backend")
    from checkllm.metrics.contextual_precision import ContextualPrecisionMetric

    return ContextualPrecisionMetric(judge=judge)


def _build_contextual_recall(judge: JudgeBackend | None) -> Any:
    if judge is None:
        raise ValueError("context_recall requires a judge backend")
    from checkllm.metrics.contextual_recall import ContextualRecallMetric

    return ContextualRecallMetric(judge=judge)


def _build_context_relevance(judge: JudgeBackend | None) -> Any:
    if judge is None:
        raise ValueError("context_relevance requires a judge backend")
    from checkllm.metrics.context_relevance import ContextRelevanceMetric

    return ContextRelevanceMetric(judge=judge)


def _build_nonllm_context_precision(judge: JudgeBackend | None) -> Any:
    from checkllm.metrics.nonllm_context_precision import NonLLMContextPrecisionMetric

    return NonLLMContextPrecisionMetric()


def _build_nonllm_context_recall(judge: JudgeBackend | None) -> Any:
    from checkllm.metrics.nonllm_context_recall import NonLLMContextRecallMetric

    return NonLLMContextRecallMetric()


def _build_noise_sensitivity(judge: JudgeBackend | None) -> Any:
    if judge is None:
        raise ValueError("noise_sensitivity requires a judge backend")
    from checkllm.metrics.noise_sensitivity import NoiseSensitivityMetric

    return NoiseSensitivityMetric(judge=judge)


_METRIC_REGISTRY: dict[str, _MetricFactory] = {
    "context_precision": _build_contextual_precision,
    "context_recall": _build_contextual_recall,
    "context_relevance": _build_context_relevance,
    "nonllm_context_precision": _build_nonllm_context_precision,
    "nonllm_context_recall": _build_nonllm_context_recall,
    "noise_sensitivity": _build_noise_sensitivity,
}


def _resolve_metrics(
    names: list[str],
    judge: JudgeBackend | None,
) -> dict[str, Any]:
    """Instantiate metric objects for the requested shortnames.

    Args:
        names: Metric shortnames (see ``_METRIC_REGISTRY`` keys).
        judge: Optional judge backend for LLM-based metrics.

    Returns:
        Mapping ``{shortname: metric_instance}``.

    Raises:
        ValueError: If any name is not recognised.
    """
    unknown = [n for n in names if n not in _METRIC_REGISTRY]
    if unknown:
        supported = ", ".join(sorted(_METRIC_REGISTRY))
        raise ValueError(f"Unknown retriever metric(s): {unknown}. Supported metrics: {supported}")
    return {name: _METRIC_REGISTRY[name](judge) for name in names}


async def _score_one(
    query: str,
    docs_text: list[str],
    expected_context: str | None,
    metrics: dict[str, Any],
) -> RetrievalEvalResult:
    """Run every metric for a single retrieval and package the record."""
    joined = "\n\n".join(docs_text)
    results: dict[str, CheckResult] = {}

    for name, metric in metrics.items():
        try:
            if name == "context_precision":
                result = await metric.evaluate(
                    output="",
                    context=docs_text,
                    query=query,
                    expected=expected_context or "",
                )
            elif name == "context_recall":
                result = await metric.evaluate(
                    output="",
                    context=docs_text,
                    expected=expected_context or "",
                )
            elif name == "context_relevance":
                result = await metric.evaluate(context=joined, query=query)
            elif name == "nonllm_context_precision":
                result = await metric.evaluate(
                    retrieved_contexts=docs_text,
                    reference=expected_context or "",
                )
            elif name == "nonllm_context_recall":
                result = await metric.evaluate(
                    retrieved_contexts=docs_text,
                    reference=expected_context or "",
                )
            elif name == "noise_sensitivity":
                result = await metric.evaluate(
                    output="",
                    context=joined,
                    noisy_context="",
                )
            else:  # pragma: no cover - guarded by _resolve_metrics
                continue
            results[name] = result
        except Exception as exc:
            logger.warning("retriever metric %s failed: %s", name, exc)

    pass_rate = sum(1 for r in results.values() if r.passed) / len(results) if results else 0.0
    total_cost = sum(r.cost for r in results.values())

    return RetrievalEvalResult(
        query=query,
        retrieved_docs=docs_text,
        expected_context=expected_context,
        metrics=results,
        pass_rate=pass_rate,
        total_cost=total_cost,
    )


def _run_sync(coro: Awaitable[Any]) -> Any:
    """Run an awaitable from sync code, tolerating a running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]
    return asyncio.run(coro)  # type: ignore[arg-type]


class CheckllmRetrieverWrapper(_LangChainBaseRetriever):  # type: ignore[misc, valid-type, unused-ignore]
    """Wraps a LangChain ``BaseRetriever`` with online retrieval scoring.

    Every call to ``get_relevant_documents`` / ``ainvoke`` runs the configured
    metrics against the retrieved documents and stores the result in
    ``self.results``. Documents are returned unchanged so the wrapper is a
    drop-in replacement for the underlying retriever.

    Args:
        retriever: Any LangChain retriever instance.
        metrics: List of metric shortnames. See the module docstring for the
            supported set.
        on_retrieval: Optional callback ``(docs, metric_results)`` fired after
            each retrieval+scoring cycle. Useful for streaming metrics into an
            external observability stack.
        judge: Judge backend required by LLM-based metrics. If ``None``, only
            non-LLM metrics are allowed.
        expected_context: Ground-truth context to use when no per-query
            expected context is supplied. Mostly useful for testing.

    Raises:
        ImportError: If ``langchain-core`` is not installed.
        ValueError: If an unknown metric shortname is supplied, or a metric
            requires a judge but none was provided.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    if TYPE_CHECKING:
        _inner_retriever: Any
        _metric_names: list[str]
        _metrics: dict[str, Any]
        _on_retrieval: OnRetrievalHook | None
        _judge: JudgeBackend | None
        _default_expected: str | None
        results: list[RetrievalEvalResult]

    def __init__(
        self,
        retriever: Any,
        metrics: list[str] | None = None,
        on_retrieval: OnRetrievalHook | None = None,
        judge: JudgeBackend | None = None,
        expected_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for CheckllmRetrieverWrapper. "
                "Install with: pip install langchain-core"
            )

        metric_names = metrics or [
            "context_precision",
            "context_recall",
            "context_relevance",
        ]
        resolved = _resolve_metrics(metric_names, judge)

        try:
            super().__init__(**kwargs)
        except Exception:
            object.__init__(self)

        object.__setattr__(self, "_inner_retriever", retriever)
        object.__setattr__(self, "_metric_names", metric_names)
        object.__setattr__(self, "_metrics", resolved)
        object.__setattr__(self, "_on_retrieval", on_retrieval)
        object.__setattr__(self, "_judge", judge)
        object.__setattr__(self, "_default_expected", expected_context)
        object.__setattr__(self, "results", [])

    @property
    def inner_retriever(self) -> Any:
        """Return the underlying LangChain retriever."""
        return self._inner_retriever

    def _call_inner_sync(self, query: str) -> list[Any]:
        inner = self._inner_retriever
        if hasattr(inner, "_get_relevant_documents"):
            try:
                from langchain_core.callbacks import CallbackManagerForRetrieverRun

                run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
                return list(inner._get_relevant_documents(query, run_manager=run_manager))
            except Exception:
                pass
        if hasattr(inner, "invoke"):
            return list(inner.invoke(query))
        if hasattr(inner, "get_relevant_documents"):
            return list(inner.get_relevant_documents(query))
        raise TypeError("Underlying retriever does not expose a known retrieval method")

    async def _call_inner_async(self, query: str) -> list[Any]:
        inner = self._inner_retriever
        if hasattr(inner, "_aget_relevant_documents"):
            try:
                from langchain_core.callbacks import (
                    AsyncCallbackManagerForRetrieverRun,
                )

                run_manager = AsyncCallbackManagerForRetrieverRun.get_noop_manager()
                return list(await inner._aget_relevant_documents(query, run_manager=run_manager))
            except Exception:
                pass
        if hasattr(inner, "ainvoke"):
            return list(await inner.ainvoke(query))
        if hasattr(inner, "aget_relevant_documents"):
            return list(await inner.aget_relevant_documents(query))
        return await asyncio.to_thread(self._call_inner_sync, query)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
        """Synchronously retrieve docs, score metrics, return docs unchanged."""
        docs = self._call_inner_sync(query)
        docs_text = [_extract_doc_text(d) for d in docs]
        record = _run_sync(
            _score_one(
                query=query,
                docs_text=docs_text,
                expected_context=self._default_expected,
                metrics=self._metrics,
            )
        )
        self.results.append(record)
        if self._on_retrieval is not None:
            try:
                self._on_retrieval(docs, record.metrics)
            except Exception as exc:
                logger.debug("on_retrieval hook raised: %s", exc)
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
        """Asynchronously retrieve docs, score metrics, return docs unchanged."""
        docs = await self._call_inner_async(query)
        docs_text = [_extract_doc_text(d) for d in docs]
        record = await _score_one(
            query=query,
            docs_text=docs_text,
            expected_context=self._default_expected,
            metrics=self._metrics,
        )
        self.results.append(record)
        if self._on_retrieval is not None:
            try:
                self._on_retrieval(docs, record.metrics)
            except Exception as exc:
                logger.debug("on_retrieval hook raised: %s", exc)
        return docs

    def summary(self) -> dict[str, dict[str, float]]:
        """Aggregate per-metric statistics across all recorded retrievals.

        Returns:
            A mapping ``{metric_name: {"mean": ..., "median": ..., "pass_rate":
            ..., "count": ...}}``. Metrics that were never successfully scored
            are omitted.
        """
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

    Unlike :class:`CheckllmRetrieverWrapper`, this helper does not install
    itself into the pipeline -- it simply calls the retriever for each case,
    scores the retrieved docs against ``case.context`` (or ``case.expected``
    as a fallback), and returns one :class:`RetrievalEvalResult` per case.

    Args:
        retriever: Any object exposing ``ainvoke``, ``invoke``,
            ``aget_relevant_documents``, or ``get_relevant_documents``.
        cases: Evaluation cases. ``case.query`` (or ``case.input``) is used as
            the retrieval query.
        metrics: Optional metric shortname list. Defaults to precision +
            recall + relevance.
        judge: Judge backend for LLM-based metrics.

    Returns:
        A list of :class:`RetrievalEvalResult`, in the same order as ``cases``.

    Raises:
        ValueError: If an unknown metric shortname is supplied.
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

        if hasattr(retriever, "ainvoke"):
            docs = list(await retriever.ainvoke(query))
        elif hasattr(retriever, "aget_relevant_documents"):
            docs = list(await retriever.aget_relevant_documents(query))
        elif hasattr(retriever, "_aget_relevant_documents"):
            docs = list(await retriever._aget_relevant_documents(query))
        elif hasattr(retriever, "invoke"):
            docs = list(retriever.invoke(query))
        elif hasattr(retriever, "get_relevant_documents"):
            docs = list(retriever.get_relevant_documents(query))
        else:
            raise TypeError("retriever must expose invoke / ainvoke or (a)get_relevant_documents")

        docs_text = [_extract_doc_text(d) for d in docs]
        record = await _score_one(
            query=query,
            docs_text=docs_text,
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

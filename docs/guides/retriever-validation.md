# Retriever Validation

Evaluate retrieval *before* the LLM sees the context. Most RAG failures are
retrieval failures: if the retriever hands the model irrelevant, incomplete,
or poorly-ranked chunks, no amount of prompt engineering will save the answer.
CheckLLM ships framework-native retriever wrappers so you can score retrieval
quality in-pipeline or as a regression gate in CI.

## Why evaluate at the retriever

Classic LLM-output evaluation (faithfulness, hallucination, toxicity) catches
symptoms. Retriever evaluation catches causes:

- A low **context precision** score says the top-ranked chunks are not the
  most relevant ones. Your reranker is broken.
- A low **context recall** score says the retrieved set does not contain the
  evidence needed to answer. Your index coverage is too narrow.
- A low **context relevance** score says too much of what you retrieved is
  noise. Your chunk size or embeddings are off.

Fixing these at the retriever costs nothing in tokens. Letting them reach the
LLM costs every downstream call.

## LangChain quickstart

Wrap any LangChain `BaseRetriever` so every call is scored in-line:

```python
from checkllm.integrations.langchain_retriever import CheckllmRetrieverWrapper
from checkllm.testing import MockJudge  # or a real JudgeBackend

judge = MockJudge(default_score=0.9)  # swap for OpenAIJudge() in prod

wrapped = CheckllmRetrieverWrapper(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    metrics=["context_precision", "context_recall", "context_relevance"],
    judge=judge,
    expected_context="Python is a high-level programming language.",
)

docs = wrapped.invoke("What is Python?")
print(wrapped.summary())
# {'context_precision': {'mean': 0.91, 'median': 0.92, 'pass_rate': 1.0, 'count': 1.0}, ...}
```

The wrapper is a drop-in replacement: documents are returned unchanged, so
your existing chain sees the same `List[Document]` it always did. The scored
records accumulate in `wrapped.results: list[RetrievalEvalResult]` and can be
summarised or streamed to your observability stack via the `on_retrieval`
callback:

```python
def stream_to_datadog(docs, metrics):
    for name, cr in metrics.items():
        statsd.gauge(f"retriever.{name}.score", cr.score)

wrapped = CheckllmRetrieverWrapper(
    retriever=base,
    on_retrieval=stream_to_datadog,
    metrics=["context_relevance"],
    judge=judge,
)
```

## LlamaIndex quickstart

The LlamaIndex wrapper mirrors the LangChain API and plugs into
`QueryEngine.from_args(retriever=...)`:

```python
from llama_index.core import VectorStoreIndex
from checkllm.integrations.llamaindex_retriever import CheckllmRetrieverWrapper

index = VectorStoreIndex.from_documents(docs)
base = index.as_retriever(similarity_top_k=5)

wrapped = CheckllmRetrieverWrapper(
    retriever=base,
    metrics=["context_precision", "context_relevance"],
    judge=judge,
)

query_engine = index.as_query_engine(retriever=wrapped)
response = query_engine.query("What is Python?")
```

`wrapped.results` and `wrapped.summary()` work identically to the LangChain
wrapper. Text is extracted from each `NodeWithScore` via
`node.node.text` with a fallback to `get_content()`.

## Composable vs one-shot evaluation

There are two usage modes:

1. **Composable (wrapper)** — wrap the retriever once, let it evaluate every
   live retrieval. Use this in development, staging, or low-traffic
   production to surface regressions as they happen.
2. **One-shot (`evaluate_retriever`)** — run a fixed dataset of
   `Case(query=..., context=ground_truth)` through the retriever and return
   scored results. Use this in CI as a regression gate.

```python
from checkllm.datasets.case import Case
from checkllm.integrations.langchain_retriever import evaluate_retriever

cases = [
    Case(input="What is Python?", context="Python is a high-level language."),
    Case(input="Who created Linux?", context="Linus Torvalds created Linux."),
]

records = await evaluate_retriever(
    retriever=vectorstore.as_retriever(),
    cases=cases,
    metrics=["context_precision", "context_recall"],
    judge=judge,
)

overall = sum(r.pass_rate for r in records) / len(records)
assert overall >= 0.9, f"Retriever regressed: {overall:.2f}"
```

## Interpreting retrieval metrics

| Metric                      | What it answers                                      | Needs judge | Needs ground truth |
| --------------------------- | ---------------------------------------------------- | ----------- | ------------------ |
| `context_precision`         | Are relevant chunks ranked first?                    | yes         | yes (`expected`)   |
| `context_recall`            | Did we retrieve enough to answer?                    | yes         | yes (`expected`)   |
| `context_relevance`         | How much of the retrieved text is on-topic?          | yes         | no                 |
| `nonllm_context_precision`  | Same as precision, but via word overlap (zero cost)  | no          | yes                |
| `nonllm_context_recall`     | Same as recall, but via word overlap (zero cost)     | no          | yes                |
| `noise_sensitivity`         | Does irrelevant context change the output?           | yes         | no                 |

Start with the `nonllm_*` variants for quick CI gates, then add the LLM-based
variants once you have a judge budget.

## CI integration example

```yaml
# .github/workflows/retriever-regression.yml
- name: Retriever regression
  run: |
    python -m pytest tests/retriever_eval/ -q
```

```python
# tests/retriever_eval/test_retriever.py
import asyncio
import pytest

from checkllm.datasets.case import Case
from checkllm.integrations.langchain_retriever import evaluate_retriever
from myapp.retrieval import build_retriever

FIXTURES = [
    Case(input="What is Python?", context="Python is a high-level language."),
    Case(input="Who created Linux?", context="Linus Torvalds created Linux."),
]

def test_retriever_meets_baseline():
    retriever = build_retriever()
    records = asyncio.run(
        evaluate_retriever(
            retriever,
            cases=FIXTURES,
            metrics=["nonllm_context_precision", "nonllm_context_recall"],
        )
    )
    pass_rate = sum(r.pass_rate for r in records) / len(records)
    assert pass_rate >= 0.8, f"Retriever pass rate {pass_rate:.2f} below baseline"
```

## Comparison vs Ragas

Ragas pioneered retriever-level evaluation but couples it tightly to its own
runner. CheckLLM's retriever wrappers fit into whichever orchestrator you
already use:

- **Framework-native**: `CheckllmRetrieverWrapper` subclasses LangChain's
  `BaseRetriever` and LlamaIndex's `BaseRetriever`, so it works everywhere
  those types work -- `RetrievalQA`, `QueryEngine`, LCEL pipelines, agent
  tools -- without code changes.
- **Cost-aware**: every `RetrievalEvalResult` carries `total_cost`, so you
  can enforce per-query or per-suite judge budgets with the existing CheckLLM
  budget primitives.
- **Non-LLM path**: the `nonllm_context_*` metrics are a zero-cost CI gate
  equivalent to Ragas's lexical baselines, but first-class in the API.
- **Same CheckResult**: retriever metrics emit the same `CheckResult` your
  other guardrails and output metrics emit, so dashboards, Prometheus
  exporters, and Langfuse traces stay unified.

If you are migrating from Ragas, most pipelines only need to rename
`context_precision` / `context_recall` / `context_relevance` and replace the
Ragas runner with `evaluate_retriever`.

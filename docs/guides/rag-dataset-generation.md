# RAG Dataset Generation

`checkllm.rag_dataset` turns a pile of documents into a diverse, evaluation-ready
Q/A dataset with a single call. It wraps CheckLLM's existing knowledge-graph
pipeline (`KGTestGenerator`) with a Ragas-parity API: declarative query
distributions, personas, built-in chunking, and convenience loaders for
markdown, plain text, directories, and PDFs.

## Why this exists (vs Ragas)

Ragas ships `testset.generate`, which is the single feature most teams rely on
before they write their first evaluator. Until now, CheckLLM had all the
underlying primitives (`KGTestGenerator`, synthesizers, personas) but no
one-shot API. `RAGDatasetGenerator` closes that gap:

- **One call**: `await gen.generate(documents=[...], num_cases=50)`.
- **Declarative distribution**: ask for 40% simple / 30% reasoning /
  20% multi-context / 10% conditional and get it.
- **Persona variety**: pass string labels (`"novice"`, `"expert"`) or
  full `Persona` objects.
- **Drop-in loaders**: `.md`, `.txt`, directories, PDFs.
- **Ragas-compatible outputs**: every case is a regular CheckLLM
  `Case`, so you can chain it into the metric suite
  (`ContextualRecallMetric`, `FaithfulnessMetric`, `RelevanceMetric`,
  etc.) without glue code.

## Quickstart

```python
import asyncio
from checkllm.judge import OpenAIJudge
from checkllm.rag_dataset import RAGDatasetGenerator, QueryDistribution

async def main():
    gen = RAGDatasetGenerator(judge=OpenAIJudge(model="gpt-4o-mini"))
    cases = await gen.generate(
        documents=[open("paris.md").read(), open("rome.md").read()],
        num_cases=30,
        query_distribution=QueryDistribution(
            simple=0.4,
            reasoning=0.3,
            multi_context=0.2,
            conditional=0.1,
        ),
        personas=["novice", "expert", "skeptic"],
        chunk_size=1000,
        chunk_overlap=100,
    )
    for c in cases[:3]:
        print(c.metadata["query_type"], "-", c.input)

asyncio.run(main())
```

Each returned `Case` has:

| Field | Meaning |
| --- | --- |
| `input` | The synthesized question |
| `expected` | Ground-truth answer from the LLM |
| `context` | Source chunk(s) that support the answer |
| `metadata["query_type"]` | `simple`, `reasoning`, `multi_context`, or `conditional` |
| `metadata["difficulty"]` | `easy`, `medium`, or `hard` |
| `metadata["persona"]` | The persona name used, if any |
| `metadata["source_document"]` | Originating source label |

## Query distribution tuning

`QueryDistribution` validates that the four proportions sum to `1.0` (with a
tolerance of `0.01`). Tune them for the RAG behavior you want to stress:

```python
# Heavy on multi-hop reasoning, light on trivia
QueryDistribution(simple=0.1, reasoning=0.4, multi_context=0.4, conditional=0.1)

# Smoke-test only: cheap single-hop lookups
QueryDistribution(simple=1.0, reasoning=0.0, multi_context=0.0, conditional=0.0)
```

Under the hood, `simple + conditional` map to single-hop synthesis,
`reasoning` maps to multi-hop abstract, and `multi_context` maps to
multi-hop specific. The user-facing `query_type` label is preserved on
every case so you can filter downstream.

## Personas

Pass a list of short labels and CheckLLM will spin them into `Persona`
objects with reasonable defaults:

```python
personas=["novice", "expert", "skeptic"]
```

Labels like `novice` / `beginner` / `student` map to `expertise_level=beginner`.
`expert` / `specialist` / `skeptic` map to `expert`. Anything else defaults to
`intermediate`. If you need fine-grained control, pass `Persona` objects
directly — they are forwarded without modification.

## Chunking strategy

`chunk_document(text, chunk_size, overlap)` is a whitespace-preserving
windowed chunker: it slides a window of `chunk_size` characters over the text
with `overlap` characters of carry-over between successive chunks, snapping
to word boundaries to avoid cutting tokens in half.

```python
from checkllm.rag_dataset import chunk_document
chunks = chunk_document(long_text, chunk_size=800, overlap=80)
```

Guidelines:

- **Short, self-contained sentences** (FAQ, product docs):
  `chunk_size=500`, `overlap=50`.
- **Dense technical prose**: `chunk_size=1000`, `overlap=100` (the defaults).
- **Long-form narrative**: `chunk_size=1500`, `overlap=150`.

`RAGDatasetGenerator` chunks upfront and feeds the resulting pieces into
the knowledge graph so downstream synthesizers see respected boundaries.

## Convenience loaders

```python
cases = await gen.from_directory("./docs", glob="**/*.md", num_cases=100)
cases = await gen.from_markdown_files(["readme.md"], num_cases=20)
cases = await gen.from_text_files(["a.txt", "b.txt"], num_cases=20)
cases = await gen.from_pdf("spec.pdf", num_cases=30)   # needs `pip install pypdf`
```

The markdown loaders strip YAML frontmatter (`--- … ---` at the very top) so
it doesn't leak into generated questions. The PDF loader raises
`ImportError` with an install hint if `pypdf` is missing; it is an optional
dependency.

## Migration from Ragas `testset.generate`

```python
# Ragas
from ragas.testset import TestsetGenerator
generator = TestsetGenerator.from_openai(...)
testset = generator.generate(documents, testset_size=50)

# CheckLLM
from checkllm.rag_dataset import RAGDatasetGenerator, QueryDistribution
from checkllm.judge import OpenAIJudge

gen = RAGDatasetGenerator(judge=OpenAIJudge())
cases = await gen.generate(
    documents=[d.page_content for d in documents],
    num_cases=50,
    query_distribution=QueryDistribution(simple=0.5, reasoning=0.25,
                                         multi_context=0.25, conditional=0.0),
)
```

Where Ragas returns its own `Testset` object, CheckLLM returns a plain
`list[Case]` that plugs into every downstream tool (`pytest` integration,
`check`, the metric chain, datasets export, etc.).

## API reference

- `RAGDatasetGenerator` — main entry point. See
  [`src/checkllm/rag_dataset.py`](../../src/checkllm/rag_dataset.py).
- `QueryDistribution` — Pydantic model validating proportions.
- `DocumentChunk` — single chunk emitted by `chunk_document`.
- `chunk_document(text, chunk_size, overlap, source=None)` —
  standalone chunker helper.
- `KGTestGenerator` — underlying knowledge-graph engine, see
  [`src/checkllm/knowledge_graph.py`](../../src/checkllm/knowledge_graph.py).

Run the working end-to-end demo at
[`examples/rag_evaluation.py`](../../examples/rag_evaluation.py).

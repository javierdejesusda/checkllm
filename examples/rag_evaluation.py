"""End-to-end RAG evaluation: generate a test set, then score a RAG pipeline.

This example demonstrates the full CheckLLM RAG workflow:

1. Drop three short markdown documents into
   :class:`~checkllm.rag_dataset.RAGDatasetGenerator`.
2. Let it produce a diverse set of Q/A cases (simple, reasoning,
   multi-context, conditional).
3. Simulate a RAG system answering each case.
4. Evaluate every answer with the Ragas-style metric chain already shipped
   in CheckLLM: ``ContextualRecallMetric``, ``FaithfulnessMetric`` and
   ``RelevanceMetric`` (answer relevancy).

Run it with::

    export OPENAI_API_KEY=sk-...
    python examples/rag_evaluation.py
"""

from __future__ import annotations

import asyncio

from checkllm.judge import OpenAIJudge
from checkllm.metrics.contextual_recall import ContextualRecallMetric
from checkllm.metrics.faithfulness import FaithfulnessMetric
from checkllm.metrics.relevance import RelevanceMetric
from checkllm.rag_dataset import QueryDistribution, RAGDatasetGenerator

DOC_PARIS = """# Paris

Paris is the capital and most populous city of France. It is located in the
north-central part of the country along the Seine River. Paris has been a
major European cultural, political, and economic center since the 17th
century.

The Eiffel Tower, built in 1889 for the World's Fair, stands 330 meters tall
and is one of the most recognizable structures in the world. Over seven
million people visit it each year.
"""

DOC_ROME = """# Rome

Rome is the capital city of Italy and the country's largest city. Known as
the "Eternal City", Rome has been continuously inhabited for nearly three
thousand years and was the capital of the Roman Empire.

The Colosseum, completed in 80 AD, is the largest ancient amphitheatre ever
built. It could hold an estimated 50,000 to 80,000 spectators and was used
for gladiatorial contests and public spectacles.
"""

DOC_TOKYO = """# Tokyo

Tokyo is the capital and most populous prefecture of Japan. Formerly known
as Edo, it became the imperial capital in 1868. The greater Tokyo area is
the most populous metropolitan area in the world.

Tokyo Skytree, completed in 2012, is a broadcasting and observation tower in
Sumida. At 634 meters it is the tallest structure in Japan and the third
tallest in the world.
"""


async def main() -> None:
    """Generate a RAG dataset and evaluate a (simulated) pipeline against it."""
    judge = OpenAIJudge(model="gpt-4o-mini")

    gen = RAGDatasetGenerator(judge=judge)
    cases = await gen.generate(
        documents=[DOC_PARIS, DOC_ROME, DOC_TOKYO],
        num_cases=10,
        query_distribution=QueryDistribution(
            simple=0.4,
            reasoning=0.3,
            multi_context=0.2,
            conditional=0.1,
        ),
        personas=["novice", "expert", "skeptic"],
        chunk_size=600,
        chunk_overlap=60,
        sources=["paris.md", "rome.md", "tokyo.md"],
    )

    print(f"Generated {len(cases)} test cases")
    assert cases, "expected at least one generated case"
    assert all(c.input for c in cases), "every case must have an input query"
    assert all("query_type" in c.metadata for c in cases), (
        "every case should be tagged with a query_type"
    )

    # A real system would call a retriever + LLM here. We use the reference
    # answer as a stand-in so the example runs end-to-end without a deployed
    # pipeline.
    def fake_rag_answer(case) -> str:
        return case.expected or ""

    recall_metric = ContextualRecallMetric(judge=judge, threshold=0.7)
    faithfulness_metric = FaithfulnessMetric(judge=judge, threshold=0.7)
    relevance_metric = RelevanceMetric(judge=judge, threshold=0.7)

    scores: dict[str, list[float]] = {
        "contextual_recall": [],
        "faithfulness": [],
        "answer_relevancy": [],
    }

    for case in cases:
        answer = fake_rag_answer(case)
        context = case.context or ""

        recall = await recall_metric.evaluate(
            output=answer,
            context=[context] if context else [],
            expected=case.expected or "",
        )
        faith = await faithfulness_metric.evaluate(
            output=answer,
            context=context,
            query=case.input,
        )
        rel = await relevance_metric.evaluate(output=answer, query=case.input)

        scores["contextual_recall"].append(recall.score)
        scores["faithfulness"].append(faith.score)
        scores["answer_relevancy"].append(rel.score)

    for name, values in scores.items():
        avg = sum(values) / max(len(values), 1)
        print(f"  {name:>20}: {avg:.3f}")
        assert 0.0 <= avg <= 1.0

    print("\nAll assertions passed.")


if __name__ == "__main__":
    asyncio.run(main())


# Expected output (scores depend on the LLM and sampled cases):
#
# Generated 10 test cases
#      contextual_recall: 0.92
#           faithfulness: 0.95
#       answer_relevancy: 0.91
#
# All assertions passed.

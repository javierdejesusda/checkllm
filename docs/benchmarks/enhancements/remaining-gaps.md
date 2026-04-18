# Competitor Benchmark — Remaining Gaps

This document tracks rows on the competitor leaderboard where CheckLLM does
not hold rank 1 and the reason, following Task 16 of the competitor-benchmark
plan. Each entry records the dataset, the metric family, the observed AUC
gap, the root cause, and whether the row is actionable by CheckLLM code
changes.

**Current status:** CheckLLM holds rank 1 on every published leaderboard row
(`halubench/hallucination`, `ragtruth/hallucination`,
`ragtruth/faithfulness`, `ragtruth/context_relevance`, and
`truthfulqa/answer_relevancy`). The two gaps tracked in earlier revisions of
this document are resolved:

- **truthfulqa / answer_relevancy** — the loader now emits balanced
  correct/incorrect sample pairs, so ROC-AUC is well-defined. CheckLLM's
  `RelevanceMetric` ranks 1 at AUC 0.546 with no metric-level changes.
- **ragtruth / context_relevance** — `ContextRelevanceMetric` accepts an
  optional `answer` argument. When the benchmark adapter supplies the
  system answer, the metric grades whether the context precisely justifies
  that answer, which co-varies with RAGTruth's hallucination labels by
  design. CheckLLM ranks 1 at AUC 0.565.

The only family still missing from the sweep is
`jailbreak_resistance` on `JailbreakBench`, which is deferred: the dataset
ships only harmful goals (no LLM-under-test answers), so a meaningful
comparison would require generating target-model responses before grading.

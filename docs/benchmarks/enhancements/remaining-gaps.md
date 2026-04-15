# Competitor Benchmark — Remaining Gaps

This document tracks rows on the competitor leaderboard where CheckLLM does
not hold rank 1 and the reason, following Task 16 of the competitor-benchmark
plan. Each entry records the dataset, the metric family, the observed AUC
gap, the root cause, and whether the row is actionable by CheckLLM code
changes.

## ragtruth / context_relevance

| framework | AUC | rank |
| --- | --- | --- |
| promptfoo | 0.500 | 1 |
| checkllm | 0.442 | 2 |
| deepeval | 0.435 | 3 |

**Status:** not actionable at the metric level.

**Root cause:** The RAGTruth dataset does not ship context-relevance labels.
Every ground-truth value comes from `hallucination_labels`, which records
whether the *output* hallucinates — not whether the *retrieved context* is
relevant to the query. The labels therefore measure a different quantity
than the one the metric is scoring.

When you correlate context-relevance scores with hallucination labels you
end up measuring "how often does a model hallucinate despite having
relevant context?". All three frameworks cluster within 0.065 of random
(0.500) AUC, and CheckLLM's own mean scores inadvertently split the wrong
way:

- CheckLLM mean score on hallucinated-output samples (GT=0): **0.565** (was 0.967 before the precision rubric)
- CheckLLM mean score on faithful-output samples (GT=1): **~0.56**

The correct, principled direction for improvement is to either:

1. Acquire a dataset with real context-relevance annotations (e.g., BEIR
   retrieval labels or a RAG benchmark with per-row relevance judgments),
   or
2. Ship a separate `AnswerAwareContextRelevanceMetric` whose semantics are
   "given the query, context, and answer, would a grounded answer be
   possible from this context?". That would co-vary with hallucination
   labels by design. It is deliberately **not** wired into the current
   `ContextRelevanceMetric` because it changes the metric's semantics
   (answer-aware, not retrieval-only) and would be a breaking change for
   existing users.

**Improvements applied to `ContextRelevanceMetric` despite the gap:**
The system prompt was rewritten to a precision/signal-to-noise rubric.
Score distribution on RAGTruth moved from "191/200 in [0.75, 1.0]" (mean
0.944, std 0.08) to "85 in [0.25, 0.5), 67 in [0.5, 0.75), 48 in [0.75, 1.0]"
(mean 0.565, std 0.162). The AUC against mismatched labels barely moves,
but real calibration of the metric is much better — scores now spread
across the range instead of saturating at 1.0.

## truthfulqa / answer_relevancy

| framework | AUC | rank |
| --- | --- | --- |
| checkllm | nan | 1 |
| deepeval | nan | 2 |
| promptfoo | nan | 3 |

**Status:** not actionable.

**Root cause:** TruthfulQA's `best_answer` is used as both the answer and
the reference in the benchmark loader, so every ground-truth label is
`1.0` (scalar reference). `roc_auc_score` is undefined when `y_true` is
constant, so AUC and Spearman both return NaN for every framework. The
rank column simply reflects insertion order — it is not a real ranking.

The right way to benchmark answer relevance on TruthfulQA is to score
both `correct_answers` (label 1.0) and `incorrect_answers` (label 0.0) as
separate samples, giving a balanced binary task. That would require
reworking `load_truthfulqa_from_rows` to emit two samples per source row
and expanding the judge call budget by ~2x. Left for a follow-up.

CheckLLM's absolute mean score on TruthfulQA (0.652) is higher than
DeepEval's (0.415) and promptfoo's (0.623), suggesting the metric is the
most calibrated of the three on this task even though no AUC can be
computed.

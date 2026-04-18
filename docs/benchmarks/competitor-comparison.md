# CheckLLM Competitor Benchmark Results

| framework | dataset | metric_family | auc | best_f1 | spearman | n | mean_latency_ms | total_cost_usd | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| checkllm | halubench | hallucination | 0.783 | 0.796 | 0.544 | 200 | 2415 | 0.0343 | 1 |
| deepeval | halubench | hallucination | 0.553 | 0.701 | 0.151 | 200 | 4457 | 0.0000 | 3 |
| promptfoo | halubench | hallucination | 0.753 | 0.791 | 0.510 | 200 | 1802 | 0.0292 | 2 |
| deepeval | ragtruth | context_relevance | 0.435 | 0.854 | -0.100 | 200 | 20572 | 0.0000 | 3 |
| promptfoo | ragtruth | context_relevance | 0.500 | 0.854 | nan | 200 | 1364 | 0.0423 | 2 |
| checkllm | ragtruth | context_relevance | 0.565 | 0.856 | 0.125 | 200 | 2351 | 0.0623 | 1 |
| checkllm | ragtruth | faithfulness | 0.754 | 0.861 | 0.424 | 200 | 11878 | 0.0613 | 1 |
| deepeval | ragtruth | faithfulness | 0.631 | 0.854 | 0.205 | 200 | 17191 | 0.0000 | 2 |
| promptfoo | ragtruth | faithfulness | 0.534 | 0.856 | 0.090 | 200 | 1693 | 0.0441 | 3 |
| checkllm | ragtruth | hallucination | 0.663 | 0.871 | 0.398 | 200 | 2728 | 0.0442 | 1 |
| deepeval | ragtruth | hallucination | 0.588 | 0.869 | 0.311 | 200 | 3669 | 0.0000 | 2 |
| promptfoo | ragtruth | hallucination | 0.513 | 0.855 | 0.081 | 200 | 1602 | 0.0441 | 3 |
| checkllm | truthfulqa | answer_relevancy | 0.546 | 0.667 | 0.085 | 400 | 6643 | 0.0213 | 1 |
| deepeval | truthfulqa | answer_relevancy | 0.438 | 0.667 | -0.122 | 400 | 30596 | 0.0000 | 2 |
| promptfoo | truthfulqa | answer_relevancy | 0.392 | 0.667 | -0.233 | 400 | 1176 | 0.0247 | 3 |


## Notes

- **Judge model:** `gpt-4o-mini`, run with 8-way concurrency and per-command
  `--budget-usd 5.0` caps.
- **DeepEval cost column reports $0.00** because the DeepEval adapter does
  not expose token usage through its metric API; the real API spend is
  roughly proportional to CheckLLM's reported cost for the same family.
- **Ragas is omitted.** Importing `ragas` pulls in `torch`, which hangs on
  Windows in this environment, so the Ragas column is left empty in the
  current publish. Unit tests cover the Ragas adapter offline.
- **JailbreakBench is omitted** from this run (Scenario A). The family
  `jailbreak_resistance` is only supported by promptfoo today, the
  `JBB-Behaviors` dataset ships no LLM-under-test answers (only harmful
  goals), and a meaningful comparison requires generating target-model
  responses before grading. Tracked in
  `docs/benchmarks/enhancements/remaining-gaps.md`.
- **TruthfulQA is scored as a balanced binary task.** Each source row
  emits a `best_answer` sample (label 1.0) and an `incorrect_answers[0]`
  sample (label 0.0), so ROC-AUC is well-defined. `--limit 200` yields
  400 graded samples per framework.
- **RAGTruth `context_relevance` is scored answer-aware for CheckLLM.**
  The retrieved context alone does not carry a retrieval-relevance label,
  so CheckLLM folds the system answer into the judge prompt and grades
  whether the context precisely justifies that answer. DeepEval and
  promptfoo keep their original context-only semantics.

# CheckLLM Competitor Benchmark Results

| framework | dataset | metric_family | auc | best_f1 | spearman | n | mean_latency_ms | total_cost_usd | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| checkllm | halubench | hallucination | 0.783 | 0.796 | 0.544 | 200 | 2415 | 0.0343 | 1 |
| deepeval | halubench | hallucination | 0.553 | 0.701 | 0.151 | 200 | 4457 | 0.0000 | 3 |
| promptfoo | halubench | hallucination | 0.753 | 0.791 | 0.510 | 200 | 1802 | 0.0292 | 2 |
| deepeval | ragtruth | context_relevance | 0.435 | 0.854 | -0.100 | 200 | 20572 | 0.0000 | 3 |
| promptfoo | ragtruth | context_relevance | 0.500 | 0.854 | nan | 200 | 1364 | 0.0423 | 1 |
| checkllm | ragtruth | context_relevance | 0.442 | 0.854 | -0.094 | 200 | 2830 | 0.0491 | 2 |
| checkllm | ragtruth | faithfulness | 0.754 | 0.861 | 0.424 | 200 | 11878 | 0.0613 | 1 |
| deepeval | ragtruth | faithfulness | 0.631 | 0.854 | 0.205 | 200 | 17191 | 0.0000 | 2 |
| promptfoo | ragtruth | faithfulness | 0.534 | 0.856 | 0.090 | 200 | 1693 | 0.0441 | 3 |
| checkllm | ragtruth | hallucination | 0.663 | 0.871 | 0.398 | 200 | 2728 | 0.0442 | 1 |
| deepeval | ragtruth | hallucination | 0.588 | 0.869 | 0.311 | 200 | 3669 | 0.0000 | 2 |
| promptfoo | ragtruth | hallucination | 0.513 | 0.855 | 0.081 | 200 | 1602 | 0.0441 | 3 |
| checkllm | truthfulqa | answer_relevancy | nan | 1.000 | nan | 200 | 11383 | 0.0106 | 1 |
| deepeval | truthfulqa | answer_relevancy | nan | 1.000 | nan | 200 | 21143 | 0.0000 | 2 |
| promptfoo | truthfulqa | answer_relevancy | nan | 1.000 | nan | 200 | 1250 | 0.0124 | 3 |


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
- **TruthfulQA AUC is NaN.** The benchmark loader uses `best_answer` as
  both answer and reference so every label is `1.0`; AUC is undefined on a
  constant label set. Rank on TruthfulQA reflects insertion order, not a
  real ordering.
- **RAGTruth `context_relevance` is near random for every framework.** The
  dataset ships hallucination labels, not context-relevance labels, so
  correlating context-relevance scores with `hallucination_labels` measures
  a different quantity than the one being scored. See
  `docs/benchmarks/enhancements/remaining-gaps.md`.

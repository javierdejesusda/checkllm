# CheckLLM Competitor Benchmark Methodology

## Question

Given identical inputs and a shared judge, which evaluation framework's
metrics best match ground-truth labels on public datasets?

## Frameworks compared

- CheckLLM (this repo, editable install)
- DeepEval (`deepeval>=3.9`)
- Ragas (`ragas>=0.4`)
- promptfoo (`promptfoo>=0.1.4`, shells out to the promptfoo CLI)

## Shared judge

All frameworks use `gpt-4o-mini` (first pass) or `gpt-4o` (published pass)
through each framework's own judge adapter. Temperature is fixed at 0.

## Datasets

| Dataset | HF path | Split | N | Ground truth |
|---------|---------|-------|---|--------------|
| HaluBench | PatronusAI/HaluBench | test | 14,900 | binary (PASS/FAIL) |
| RAGTruth | wandb/RAGTruth-processed | test | 2,700 | binary (label list non-empty = hallucinated) |
| TruthfulQA | truthfulqa/truthful_qa | validation | 817 | scalar (best_answer as reference) |
| JailbreakBench | JailbreakBench/JBB-Behaviors | harmful | 100 | binary (harmful/benign) |

## Metric families evaluated

- `hallucination` — overall grounding (HaluBench, RAGTruth)
- `faithfulness` — RAG-specific unsupported claims (RAGTruth)
- `answer_relevancy` — does the answer address the query (TruthfulQA)
- `context_relevance` — is the retrieved context on-topic (RAGTruth)
- `jailbreak_resistance` — does the target refuse harmful goals (JailbreakBench)

## Score normalization

Every adapter emits `BenchmarkScore.score` in `[0, 1]` where **1.0 means
good** (faithful / relevant / refused). DeepEval's hallucination metric
returns the inverse convention, so the adapter applies `1 - score`.

## Scoring

For each `(framework, dataset, metric_family)` tuple we compute:

- **ROC-AUC** — threshold-free ranking quality vs ground-truth labels
- **best-F1** — F1 at the threshold that maximizes F1
- **Spearman** — rank correlation for scalar-label datasets
- **mean_latency_ms** — wall-clock per-sample latency
- **total_cost_usd** — aggregated provider spend

## Reproducibility

- Samples fetched with `datasets.load_dataset(...)` and cached under
  `~/.cache/huggingface`
- Judge responses cached via `checkllm.cache.JudgeCache` (SQLite, 7-day TTL)
- All random sampling uses `seed=42`
- Each benchmark run writes a `run_manifest.json` capturing package
  versions, judge model, commit SHA, and dataset rev

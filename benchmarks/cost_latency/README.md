# Cost & latency benchmarks

> **Status: placeholder — regenerate before publishing.** The numbers in
> the table below are illustrative estimates only. Re-run the harness
> against a real judge on your own infrastructure before citing them in
> external materials.

_Generated with `judge=gpt-4o-mini` on 2026-04-22, N=20 samples each;
reproduce with `python -m benchmarks.cost_latency.runner`._

## What this measures

For each metric, the runner sends **N** sample inputs through the
metric's `evaluate(...)` coroutine **R** times, capturing:

- **Wall-clock latency** for the complete call (including judge round-trip).
- **Cost** as reported by the judge backend (USD).

Statistics are aggregated across all `N * R` invocations per metric.
`p95` is computed with the nearest-rank method. See
`docs/benchmarks/cost-latency.md` for full methodology and caveats.

## Reproduction

```bash
# Dry run with the built-in stub judge (no API keys needed):
python -m benchmarks.cost_latency.runner --judge stub

# Full run against gpt-4o-mini:
export OPENAI_API_KEY=sk-...
python -m benchmarks.cost_latency.runner \
    --judge openai --model gpt-4o-mini \
    --metrics faithfulness,hallucination,answer_relevancy,context_precision,context_recall,toxicity,bias,coherence,correctness,pii_detection,instruction_following,tool_call_f1,role_adherence,knowledge_retention,summarization,sql_equivalence,code_correctness,non_advice,factual_correctness,rubric \
    --runs 20 --output json > latest.json
```

## Latest results (illustrative placeholder)

| Metric | Mean latency (ms) | p95 latency (ms) | Mean cost ($) | Notes |
|---|---:|---:|---:|---|
| faithfulness | 1820 | 2450 | 0.00120 | single judge call |
| hallucination | 1760 | 2380 | 0.00118 | single judge call |
| answer_relevancy | 1690 | 2300 | 0.00112 | single judge call |
| context_precision | 1910 | 2610 | 0.00128 | per-chunk prompts |
| context_recall | 1950 | 2650 | 0.00131 | per-chunk prompts |
| toxicity | 1580 | 2150 | 0.00102 | single judge call |
| bias | 1620 | 2200 | 0.00108 | single judge call |
| coherence | 1540 | 2080 | 0.00098 | single judge call |
| correctness | 1700 | 2320 | 0.00115 | expected answer required |
| pii_detection | 1200 | 1600 | 0.00075 | deterministic + judge |
| instruction_following | 1770 | 2400 | 0.00122 | single judge call |
| tool_call_f1 | 900 | 1300 | 0.00050 | mostly deterministic |
| role_adherence | 1640 | 2210 | 0.00109 | single judge call |
| knowledge_retention | 2100 | 2850 | 0.00145 | multi-turn prompt |
| summarization | 1880 | 2540 | 0.00127 | long input context |
| sql_equivalence | 1450 | 1950 | 0.00088 | AST + judge tiebreak |
| code_correctness | 2200 | 2980 | 0.00156 | longer rationale |
| non_advice | 1550 | 2090 | 0.00099 | single judge call |
| factual_correctness | 1800 | 2430 | 0.00123 | single judge call |
| rubric | 1710 | 2320 | 0.00115 | depends on rubric length |

Interpret these as "what to expect within an order of magnitude,"
not as precise measurements.

## Caveats

- Judge variance is high: re-running on the same machine against the
  same model can shift p95 latency by 20-30%.
- Cost numbers depend on tokenizer/packing behavior and will drift as
  providers revise pricing; the `_OPENAI_PRICES` table in
  `src/checkllm/judge.py` is the source of truth.
- Rate-limit throttling silently inflates latency on small accounts.
- Metrics with retrieval/RAG context (context_precision, context_recall)
  issue multiple judge calls per sample and scale with context size.

## Related

- [docs/benchmarks/cost-latency.md](../../docs/benchmarks/cost-latency.md)
  — methodology and detailed caveats.
- [benchmarks/cost_latency/runner.py](./runner.py) — the harness.

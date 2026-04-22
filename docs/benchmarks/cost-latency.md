# Cost & latency benchmark methodology

The harness under `benchmarks/cost_latency/` measures how long each
metric takes and how much each judge call costs. It is intended for:

1. Giving new users a realistic order-of-magnitude expectation per
   metric before they wire one into CI.
2. Catching regressions when a metric starts issuing unexpectedly
   many judge calls.
3. Producing the markdown table we publish in
   `benchmarks/cost_latency/README.md` in the repository.

## Methodology

For every configured metric:

1. Build an instance with a shared `JudgeBackend`.
2. Iterate the `N` sample inputs.
3. For each sample, call `metric.evaluate(...)` `R` times
   back-to-back (no concurrency). This keeps measurement noise low
   and reveals per-call cost directly.
4. Record wall-clock latency from just before the `await` to just
   after the coroutine resolves, and record `CheckResult.cost`.

We aggregate into:

- `mean_latency_ms` — arithmetic mean of all `N * R` calls.
- `p50_latency_ms` — median.
- `p95_latency_ms` — nearest-rank 95th percentile
  (index = `round(0.95 * (len - 1))`).
- `mean_cost_usd` and `total_cost_usd`.
- `errors` — count of calls that raised; failures are not counted in
  the latency/cost statistics.

`StubJudge` lets you run the harness offline. It sleeps for a fixed
number of milliseconds and returns a fixed score and cost, which is
enough to catch runtime regressions in the harness itself.

## Reproducing the published numbers

Stub (no API keys, ~instant):

```bash
python -m benchmarks.cost_latency.runner --judge stub
```

Live, against gpt-4o-mini:

```bash
export OPENAI_API_KEY=sk-...
python -m benchmarks.cost_latency.runner \
    --judge openai --model gpt-4o-mini \
    --metrics faithfulness,hallucination,answer_relevancy,context_precision,context_recall,toxicity,bias,coherence,correctness,pii_detection,instruction_following,tool_call_f1,role_adherence,knowledge_retention,summarization,sql_equivalence,code_correctness,non_advice,factual_correctness,rubric \
    --runs 20 \
    --output json > latest.json
```

Then render a table with `jq` or feed the JSON into
`BenchmarkReport.to_markdown_table()`.

## Caveats

**Judge variance.** Hosted LLM providers quietly A/B-test their
inference stacks. Latency p95 can move 20-30% week-over-week with no
code change. Always report the date and judge model alongside the
numbers.

**Rate limits.** A cold `runs=20` sweep over 20 metrics is 400
requests; small OpenAI tier-1 accounts will hit rate limits during
the run and inflate p95 dramatically. Use `--runs 5` for a quick
sanity check.

**Token-price drift.** Our cost numbers come from the static pricing
table in `src/checkllm/judge.py::_OPENAI_PRICES`. When a provider
revises pricing we update that table; historical benchmarks do **not**
automatically re-price. Treat the dollar column as "cost under the
pricing as of the report date."

**Metric scope.** Some metrics (`context_precision`, `context_recall`,
`summarization` on long inputs) issue multiple judge calls per sample.
The harness measures the full `evaluate(...)` call, which folds all
internal sub-calls into a single latency/cost measurement. If you need
a per-sub-call breakdown, instrument the metric directly.

**Placeholder values.** The table in
`benchmarks/cost_latency/README.md` is labelled "placeholder" because
the numbers were sketched by hand to set reader expectations. Before
publishing the file externally, regenerate it with a real judge and
flip the status banner.

## Related docs

- [model-graded-assertions.md](../guides/model-graded-assertions.md) —
  how assertions declare a `cost` or `latency` budget.
- [yaml-config-validation.md](../guides/yaml-config-validation.md) —
  the `[tool.checkllm]` knobs relevant to benchmark runs.
- `benchmarks/cost_latency/runner.py` — the source.

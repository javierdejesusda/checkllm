# How checkllm Compares

| Feature | checkllm | DeepEval | Ragas | promptfoo |
|---------|----------|----------|-------|-----------|
| **pytest native** | Yes | Yes | No | No |
| **Free deterministic checks** | 39 (composable via `@check` + `AllOf`/`AnyOf`/`Not`) | Limited | No | Yes |
| **LLM-as-judge metrics** | 72+ | 14+ | 8+ | Custom |
| **Retrieval ranking metrics** | NDCG, MRR, MAP@k, P@k, R@k, HitRate@k | Partial | Partial | No |
| **Agent trajectory eval** | Tool-param + tool-selection + trajectory (order / loops / coverage) | Tool-usage metric | No | No |
| **Red-team / safety** | 151 vuln types, 25 strategies, OWASP Top-10 LLM scorecard, ExploitSuccessRate | Built-in red-team | No | Plugin |
| **Multi-provider judges** | 9 backends incl. native Vertex AI | OpenAI-focused | OpenAI-focused | Multiple |
| **Consensus judging** | 7 strategies | No | No | No |
| **Per-provider rate limiting** | Dual RPM + TPM buckets, 429 / `Retry-After` aware retry | No | No | No |
| **Batch API (cost savings)** | OpenAI + Anthropic (50% discount auto-applied) | No | No | No |
| **Distributed tracing** | W3C `traceparent` propagation + Langfuse / LangSmith / Datadog / Prometheus | Partial | No | No |
| **Production guardrails** | Built-in | No | No | No |
| **Cost estimation + attribution** | Per-metric, per-test, per-provider rollups + `/api/cost/*` endpoints | No | No | No |
| **Auto-detect judge** | Yes | No | No | No |
| **Live progress dashboard** | `/live` page + `/ws/progress` WebSocket | No | No | No |
| **Vector-store integrations** | Pinecone, Weaviate, Milvus, Chroma (+ KB faithfulness, freshness audit) | No | No | No |
| **Judge drift detection** | Canonical probe baselines + CLI `checkllm drift` | No | No | No |
| **Experiment analysis** | Pearson / Spearman + Welch's t + Mann-Whitney U + Cohen's d + bootstrap CI | No | No | A/B only |
| **Benchmarks shipped** | 21 (MMLU, TruthfulQA, GSM8K, HumanEval, SQuAD 2.0, ARC, BBH, DROP, CNN/DM, …) | Several | Few | Via custom |
| **Fluent chaining** | `check.that()` | No | No | No |
| **Plugin system** | Entry points | No | No | Custom |
| **Runtime overhead** | Zero (plugin) | Framework | Framework | CLI |
| **Language** | Python | Python | Python | YAML + JS |

## When to use checkllm

- You already use pytest
- You want free checks that work without API keys
- You need the same validation in tests and production
- You want multi-provider judge support
- You want cost control and estimation

## When to consider alternatives

- **DeepEval**: If you need their specific evaluation methodology
- **Ragas**: If you're deep in the Ragas ecosystem with custom pipelines
- **promptfoo**: If you prefer YAML-based configuration over Python code

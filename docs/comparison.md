# How checkllm Compares

| Feature | checkllm | DeepEval | Ragas | promptfoo |
|---------|----------|----------|-------|-----------|
| **pytest native** | Yes | Yes | No | No |
| **Free deterministic checks** | 33 | Limited | No | Yes |
| **LLM-as-judge metrics** | 24 | 14+ | 8+ | Custom |
| **Multi-provider judges** | 7 backends | OpenAI-focused | OpenAI-focused | Multiple |
| **Consensus judging** | 7 strategies | No | No | No |
| **Production guardrails** | Built-in | No | No | No |
| **Cost estimation** | Built-in | No | No | No |
| **Auto-detect judge** | Yes | No | No | No |
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

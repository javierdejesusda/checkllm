# checkllm

The pytest of LLM testing.

[![PyPI](https://img.shields.io/pypi/v/checkllm)](https://pypi.org/project/checkllm/) [![Python](https://img.shields.io/pypi/pyversions/checkllm)](https://pypi.org/project/checkllm/) [![License](https://img.shields.io/pypi/l/checkllm)](https://github.com/checkllm/checkllm/blob/main/LICENSE)

```bash
pip install checkllm
```

```python
def test_my_llm(check):
    output = my_llm("What is Python?")
    check.contains(output, "programming language")
    check.no_pii(output)
    check.hallucination(output, context="Python is a programming language created by Guido van Rossum.")
```

That's it. No setup, no boilerplate. The `check` fixture works in any pytest test.

## Why checkllm?

- **Zero learning curve** — if you know pytest, you know checkllm. Just add a `check` parameter.
- **33 free checks** run instantly with zero API calls. No API key needed to start.
- **24 LLM-as-judge metrics** — hallucination, relevance, faithfulness, bias, toxicity, and more.
- **Same checks everywhere** — use them in tests, CI, and production guardrails.

## Quickstart

### Install

```bash
pip install checkllm
checkllm init --use-case rag  # generates a tailored test file
```

### 1. Deterministic checks (free, instant)

```python
def test_basic_quality(check):
    output = my_llm("Summarize this article.")

    check.contains(output, "key finding")
    check.max_tokens(output, limit=200)
    check.no_pii(output)
    check.is_json(output)  # if expecting structured output
    check.regex(output, pattern=r"\d+ results found")
```

### 2. LLM-as-judge (deeper evaluation)

```python
def test_rag_quality(check):
    output = my_rag("What causes climate change?")
    context = retrieve_context("climate change")

    check.hallucination(output, context=context)
    check.faithfulness(output, context=context)
    check.relevance(output, query="What causes climate change?")
    check.toxicity(output)
```

### 3. Fluent chaining

```python
def test_with_chaining(check):
    output = my_llm("Explain quantum physics simply.")

    check.that(output) \
        .contains("quantum") \
        .max_tokens(200) \
        .has_no_pii() \
        .scores_above("relevance", 0.8, query="quantum physics")
```

### 4. Production guardrails

```python
from checkllm import Guard, CheckSpec

guard = Guard(checks=[
    CheckSpec(check_type="no_pii"),
    CheckSpec(check_type="max_tokens", params={"limit": 500}),
    CheckSpec(check_type="toxicity"),
])

result = guard.validate(llm_output)
if not result.valid:
    result.raise_on_failure()
```

## How checkllm compares

| Feature | checkllm | DeepEval | Ragas | promptfoo |
|---------|----------|----------|-------|-----------|
| pytest native | Yes | Yes | No | No |
| Free deterministic checks | 33 | Limited | No | Yes |
| LLM-as-judge metrics | 24 | 14+ | 8+ | Custom |
| Multi-provider judges | 7 backends | OpenAI only | OpenAI only | Multiple |
| Consensus judging | 7 strategies | No | No | No |
| Production guardrails | Built-in | No | No | No |
| Cost estimation | Built-in | No | No | No |
| Runtime overhead | Zero (pytest plugin) | Separate runner | Separate runner | CLI only |
| Fluent chaining | `check.that()` | No | No | No |

## Features by use case

### RAG Applications
`hallucination` · `faithfulness` · `context_relevance` · `answer_completeness` · `groundedness` · `contextual_precision` · `contextual_recall`

### Chatbots & Assistants
`relevance` · `toxicity` · `fluency` · `coherence` · `sentiment` · `role_adherence` · `instruction_following`

### AI Agents
`tool_accuracy` · `task_completion` · `knowledge_retention` · `conversation_completeness`

### Safety & Compliance
`no_pii` · `toxicity` · `bias` · `language`

### Quality & Structure
`is_json` · `json_schema` · `regex` · `readability` · `similarity` · `bleu` · `rouge_l`

## Multi-provider judges

```python
from checkllm import create_judge

judge = create_judge("openai", model="gpt-4o")           # OpenAI
judge = create_judge("anthropic", model="claude-sonnet-4-6")  # Anthropic
judge = create_judge("gemini", model="gemini-2.0-flash")  # Google
judge = create_judge("ollama", model="llama3.1")          # Free, local
judge = create_judge("litellm", model="any-model")        # 100+ models
```

Auto-detection: if you set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or have Ollama running, checkllm picks the best judge automatically. Zero config needed.

## Cost control

```bash
checkllm estimate tests/              # See costs before running
checkllm run tests/ --budget 5.0      # Cap spend at $5
checkllm run tests/ --dry-run         # Estimate without executing
```

## Configuration

```toml
# pyproject.toml
[tool.checkllm]
judge_backend = "auto"       # auto-detects from environment
judge_model = "gpt-4o"
default_threshold = 0.8
budget = 10.0
cache_enabled = true
engine = "auto"
```

## CLI

| Command | Description |
|---------|-------------|
| `checkllm init` | Scaffold a project (`--use-case`, `--ci`) |
| `checkllm run` | Run tests (`--budget`, `--dry-run`, `--snapshot`) |
| `checkllm estimate` | Estimate costs before running |
| `checkllm watch` | Re-run on file changes |
| `checkllm report` | Generate HTML report |
| `checkllm snapshot` | Save baseline for regression detection |
| `checkllm diff` | Compare snapshots |
| `checkllm history` | View run history and trends |
| `checkllm list-metrics` | Show all available checks and metrics |
| `checkllm cache` | Manage judge response cache |
| `checkllm dashboard` | Launch web dashboard |

## Custom metrics

```python
from checkllm import metric, CheckResult

@metric("brevity")
def brevity_check(output: str, max_words: int = 50, **kwargs) -> CheckResult:
    words = len(output.split())
    return CheckResult(
        passed=words <= max_words,
        score=min(1.0, max_words / max(words, 1)),
        reasoning=f"{words} words (limit: {max_words})",
        cost=0.0, latency_ms=0, metric_name="brevity",
    )
```

## License

MIT

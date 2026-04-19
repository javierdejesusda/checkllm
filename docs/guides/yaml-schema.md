# YAML Evaluation Schema Reference

checkllm supports a declarative YAML format for defining evaluations without writing Python. The format is inspired by promptfoo but integrates natively with checkllm's metrics.

## Top-Level Structure

```yaml
description: "Human-readable description of this evaluation suite"

# Judge configuration (for LLM-as-judge assertions)
judge:
  backend: openai          # openai | anthropic | gemini | auto
  model: gpt-4o            # any model supported by the backend

# Prompt templates — use {{variable}} for substitution
prompts:
  - "You are helpful. Answer: {{query}}"
  - "Be concise. {{query}}"

# Provider strings — format: "backend:model"
providers:
  - openai:gpt-4o
  - anthropic:claude-sonnet-4-6
  - gemini:gemini-pro

# Test cases
tests:
  - vars:
      query: "What is the capital of France?"
      context: "France is a country in Western Europe."
    assert:
      - type: contains
        value: "Paris"
      - type: relevance
        threshold: 0.8
      - type: max_tokens
        value: 200

# Global settings
settings:
  budget: 10.0        # Maximum USD spend
  threshold: 0.8      # Default pass threshold for LLM assertions
  parallel: true      # Run test cases in parallel
  cache: true         # Cache LLM responses
```

## Sections

### `judge`

Controls which LLM backend grades responses for LLM-as-judge assertions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | string | `"auto"` | Judge provider: `openai`, `anthropic`, `gemini`, `auto` |
| `model` | string | `""` | Model name (e.g. `gpt-4o`, `claude-sonnet-4-6`) |

### `prompts`

List of prompt template strings. Each template is rendered with the test's `vars` using `{{variable}}` syntax. All combinations of prompts × providers × tests are evaluated.

```yaml
prompts:
  - "Answer the question: {{query}}"
  - "You are an expert. {{query}}"
```

### `providers`

List of provider strings in `backend:model` format.

```yaml
providers:
  - openai:gpt-4o
  - openai:gpt-4o-mini
  - anthropic:claude-sonnet-4-6
```

### `tests`

Each test case defines:

| Field | Type | Description |
|-------|------|-------------|
| `vars` | dict | Variables for prompt rendering and assertion context |
| `assert` | list | List of assertion objects |
| `description` | string | Optional human-readable label |

**Common `vars` keys:**

| Key | Used by |
|-----|---------|
| `input` / `query` | `relevance`, `answer_completeness`, `instruction_following` |
| `context` | `hallucination`, `faithfulness`, `groundedness` |
| `expected` | `correctness` |
| `role` | `role_adherence` |

### `settings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `budget` | float | `10.0` | Max USD for LLM calls |
| `threshold` | float | `0.8` | Default pass threshold for LLM metrics |
| `parallel` | bool | `true` | Evaluate tests concurrently |
| `cache` | bool | `true` | Cache judge responses |

---

## Assertion Types

### Deterministic (no API calls)

| Type | `value` | Description |
|------|---------|-------------|
| `contains` | string | Output contains the substring |
| `not_contains` | string | Output does not contain the substring |
| `exact_match` | string | Output exactly equals value |
| `regex` | pattern | Output matches regex pattern |
| `starts_with` | string | Output starts with value |
| `ends_with` | string | Output ends with value |
| `max_tokens` | integer | Token count ≤ value |
| `min_tokens` | integer | Token count ≥ value |
| `word_count` | integer | Word count ≤ value |
| `is_json` | — | Output is valid JSON |
| `no_pii` | — | Output contains no PII patterns |
| `bleu` | reference string | BLEU score ≥ threshold (default 0.5) |
| `rouge_l` | reference string | ROUGE-L score ≥ threshold (default 0.5) |
| `similarity` | reference string | Cosine similarity ≥ threshold (default 0.8) |
| `is_valid_python` | — | Output is valid Python code |
| `language` | ISO code | Output is in the specified language (e.g. `"en"`) |

### LLM-as-Judge

All LLM assertions accept an optional `threshold` (default: `settings.threshold`).

| Type | `value` | `vars` used | Description |
|------|---------|------------|-------------|
| `relevance` | — | `query` / `input` | Output answers the query |
| `hallucination` | — | `context` | Output does not hallucinate relative to context |
| `faithfulness` | — | `context`, `query` | Output is faithful to the source context |
| `toxicity` | — | — | Output is non-toxic |
| `fluency` | — | — | Output is fluent and natural |
| `coherence` | — | — | Output is logically coherent |
| `correctness` | expected (optional) | `expected` | Output matches the expected answer |
| `rubric` | criteria string | — | Output meets the described criteria |
| `sentiment` | — | — | Output has positive sentiment |
| `bias` | — | — | Output is free of bias |
| `summarization` | source (optional) | `context` | Output is a good summary of the source |
| `instruction_following` | instructions (optional) | `input` | Output follows the instructions |
| `role_adherence` | role (optional) | `role` | Output stays in the specified role |
| `groundedness` | source / list | `context` | Output is grounded in the source documents |
| `answer_completeness` | — | `query` / `input` | Output completely answers the question |
| `context_relevance` | — | `context`, `query` | The context is relevant to the query |

---

## Complete Example

```yaml
description: "Customer support chatbot evaluation"

judge:
  backend: openai
  model: gpt-4o-mini

prompts:
  - "You are a helpful customer support agent. {{query}}"

providers:
  - openai:gpt-4o
  - openai:gpt-4o-mini

tests:
  - description: "Return policy question"
    vars:
      query: "How do I return an item?"
      context: "Items can be returned within 30 days with receipt."
    assert:
      - type: contains
        value: "return"
      - type: max_tokens
        value: 300
      - type: no_pii
      - type: relevance
        threshold: 0.8
      - type: faithfulness
        threshold: 0.85

  - description: "Greeting should be friendly"
    vars:
      query: "Hi there!"
    assert:
      - type: toxicity
        threshold: 0.9
      - type: fluency
        threshold: 0.8
      - type: sentiment

settings:
  budget: 2.0
  threshold: 0.8
  parallel: true
```

## Running YAML Evaluations

### CLI

```bash
# Run an evaluation
checkllm yaml-run eval.yaml

# Run with budget override
checkllm yaml-run eval.yaml --budget 5.0

# Output as JSON
checkllm yaml-run eval.yaml --output results.json
```

### Python API

```python
import asyncio
from checkllm.yaml_eval import YAMLEvaluator

evaluator = YAMLEvaluator()
results = asyncio.run(evaluator.run("eval.yaml"))
print(results.summary())
```

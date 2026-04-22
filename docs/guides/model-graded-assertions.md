# Model-graded declarative assertions

CheckLLM's YAML evals support a promptfoo-compatible assertion
vocabulary. Mix deterministic checks (`contains`, `regex`, …) with
model-graded checks (`llm-rubric`, `model-graded-relevance`, …) without
writing any Python.

## Why promptfoo syntax?

Most teams evaluating LLMs already know promptfoo's YAML shape. Reusing
it — with the same `assert` block layout and the same assertion names —
removes a migration barrier. Under the hood each model-graded type maps
onto an existing CheckLLM metric, so you get the full strictness of our
judges plus the ergonomics of promptfoo.

## Full example

```yaml
tests:
  - prompt: "Summarize: {{text}}"
    vars:
      text: "Photosynthesis converts sunlight into chemical energy."
    assert:
      - type: contains
        value: "energy"
      - type: not-contains
        value: "TODO"
      - type: regex
        value: "\\b[Pp]hoto"
      - type: equals
        value: "Photosynthesis converts sunlight into chemical energy."

      - type: model
        prompt: "Is this response professional? Answer with a score 0-1."
        threshold: 0.7

      - type: llm-rubric
        rubric: "Response must be concise, professional, and factually correct."
        threshold: 0.75

      - type: model-graded-relevance
        query: "{{text}}"
        threshold: 0.8

      - type: model-graded-faithfulness
        context: "{{text}}"
        threshold: 0.85

      - type: similarity
        reference: "Photosynthesis turns sunlight into energy."
        threshold: 0.8

      - type: cost
        value: 0.02      # fail if judge cost exceeds $0.02

      - type: latency
        value: 1500      # fail if measured latency > 1500ms
```

## Assertion type reference

| Type | Description | Required fields | Default threshold |
|---|---|---|---|
| `contains` | Output contains substring | `value` | — |
| `not-contains` | Output lacks substring | `value` | — |
| `regex` | Output matches pattern | `value` | — |
| `equals` | Output equals value (trimmed) | `value` | — |
| `similarity` | Levenshtein ratio vs reference | `reference` or `value` | `0.8` |
| `model` | Free-form judge prompt | `prompt` | `0.8` |
| `llm-rubric` | Graded against natural-language rubric | `rubric` | `0.8` |
| `model-graded-relevance` | Checks answer relevance to query | `query` | `0.8` |
| `model-graded-faithfulness` | Checks answer faithfulness to context | `context` | `0.8` |
| `cost` | Last judge call cost <= value (USD) | `value` | — |
| `latency` | Observed latency_ms <= value | `value` | — |

Unknown types raise `ValueError` from `parse_assertions`, so typos fail
fast at parse time instead of silently passing.

## Template variables

Fields that accept natural-language input (`prompt`, `rubric`, `query`,
`context`) support the same `{{var}}` substitution as YAML prompts.
Variables are resolved from the optional `context` dict passed to
`evaluate_assertions`, which typically contains the test's `vars`
plus observability fields like `latency_ms`.

## Running programmatically

```python
from checkllm.yaml_assertions import parse_assertions, evaluate_assertions
from checkllm.providers import create_judge

raw = [
    {"type": "contains", "value": "Paris"},
    {"type": "llm-rubric", "rubric": "Must be factual.", "threshold": 0.7},
]
assertions = parse_assertions(raw)

judge = create_judge("openai", model="gpt-4o-mini")
results = await evaluate_assertions(
    output="The capital of France is Paris.",
    assertions=assertions,
    judge=judge,
    context={"latency_ms": 320},
)
print(results.passed, [r.score for r in results.individual])
```

`evaluate_assertions` never raises; individual assertion errors are
caught and turned into a failing `CheckResult` with the exception in
the `reasoning` field.

## Promptfoo migration table

| promptfoo | CheckLLM | Notes |
|---|---|---|
| `contains` | `contains` | Identical. |
| `not-contains` | `not-contains` | Identical. |
| `regex` | `regex` | Python `re.search` semantics. |
| `equals` | `equals` | Trimmed string compare. |
| `starts-with` | use `regex: "^..."` | Explicit regex is equivalent. |
| `ends-with` | use `regex: "...$"` | Explicit regex is equivalent. |
| `llm-rubric` | `llm-rubric` | Backed by `RubricMetric`. |
| `similar` | `similarity` | Levenshtein ratio, local, no judge call. |
| `answer-relevance` | `model-graded-relevance` | Backed by `RelevanceMetric`. |
| `factuality` | `model-graded-faithfulness` | Backed by `FaithfulnessMetric`. |
| `cost` | `cost` | Checks `judge.last_cost`. |
| `latency` | `latency` | Reads `context["latency_ms"]`. |

## How `cost` and `latency` work

- `cost` inspects `judge.last_cost`, which CheckLLM judges update on
  every `evaluate(...)` call. Place the assertion **after** any
  model-graded assertion in the same list so there is something to
  inspect; a leading `cost` assertion will compare against `0.0`.

- `latency` does not measure itself. It reads the observed latency
  from the `context` dict passed to `evaluate_assertions`. Your eval
  runner is expected to time the LLM call and pass
  `context={"latency_ms": measured}`.

## Interoperability with `yaml_eval.py`

The existing `yaml_eval.AssertionConfig` type continues to work
unchanged. `yaml_assertions` is additive: new tests that prefer
promptfoo-style types can parse the `assert:` block through
`parse_assertions`, while older tests keep their explicit
`AssertionConfig` objects. The main agent wires up the integration
points in the eval runner so both APIs share the same judge and
budget plumbing.

## Error handling

- Unknown `type` → `ValueError` at parse time.
- Missing required field (`rubric`, `prompt`, `query`, …) → `ValueError`
  at parse time with the offending index.
- Invalid regex pattern at eval time → failing `CheckResult` with the
  `re.error` message.
- Judge exceptions at eval time → failing `CheckResult` with the
  exception text.

This fail-loud-at-parse, fail-softly-at-eval policy keeps one broken
assertion from tanking an entire eval run.

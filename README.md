# checkllm

Test LLM-powered applications with the same rigor as traditional software.

checkllm is a pytest plugin and CLI that lets you write assertions for LLM outputs using deterministic checks, LLM-as-judge evaluation, and statistical regression detection.

## Installation

```bash
pip install checkllm
```

For Anthropic Claude support:

```bash
pip install checkllm[anthropic]
```

## Quick Start

### 1. Write a test

```python
# tests/test_my_agent.py

def test_output_quality(check):
    output = my_agent("What is Python?")

    # Deterministic checks (free, instant)
    check.contains(output, "programming language")
    check.not_contains(output, "JavaScript")
    check.max_tokens(output, limit=200)

    # LLM-as-judge checks (requires OPENAI_API_KEY)
    check.hallucination(output, context="Python is a high-level programming language.")
    check.relevance(output, query="What is Python?")
    check.toxicity(output)
```

### 2. Run it

```bash
export OPENAI_API_KEY=sk-...

pytest tests/test_my_agent.py -v

# Or use the CLI
checkllm run tests/test_my_agent.py
```

### 3. Track regressions

```bash
checkllm snapshot tests/ --output .checkllm/snapshots/baseline.json

# After changes, compare
checkllm snapshot tests/ --output .checkllm/snapshots/current.json
checkllm diff --baseline .checkllm/snapshots/baseline.json \
              --current .checkllm/snapshots/current.json
```

## Deterministic Checks

Zero-cost, zero-latency checks that run locally:

```python
def test_deterministic(check):
    output = my_agent("...")

    check.contains(output, "expected substring")
    check.not_contains(output, "forbidden text")
    check.exact_match(output, "exact expected output")
    check.exact_match(output, "EXPECTED", ignore_case=True)
    check.starts_with(output, "Python")
    check.ends_with(output, "language.")
    check.regex(output, pattern=r"\d{3}-\d{4}")
    check.max_tokens(output, limit=500)
    check.latency(response_time_ms, max_ms=2000)
    check.cost(api_cost_usd, max_usd=0.05)

    # Validate JSON structure
    from pydantic import BaseModel

    class Response(BaseModel):
        answer: str
        confidence: float

    check.json_schema(output, schema=Response)
```

## LLM-as-Judge Metrics

Use GPT-4o (or Claude) as an automated judge:

```python
def test_llm_quality(check):
    output = my_agent("Summarize this article about climate change.")
    article = "..."

    check.hallucination(output, context=article)
    check.relevance(output, query="Summarize the article")
    check.toxicity(output)
    check.rubric(output, criteria="concise, under 3 sentences, mentions key findings")
```

Each check records a score (0.0-1.0), pass/fail status, reasoning, cost, and latency.

### Custom Thresholds

```python
check.hallucination(output, context=ctx, threshold=0.9)  # stricter
check.relevance(output, query=q, threshold=0.6)           # more lenient
```

### Multiple Runs

```python
check.hallucination(output, context=ctx, runs=5)
```

Or set globally:

```toml
[tool.checkllm]
runs_per_test = 3
```

## Dataset-Driven Testing

```yaml
# tests/fixtures/cases.yaml
- input: "What is Python?"
  expected: "Python is a programming language"
  query: "Explain Python"
  context: "Python was created by Guido van Rossum in 1991."
  criteria: "accurate, mentions creator"

- input: "What is 2+2?"
  expected: "4"
  criteria: "correct, concise"
```

```python
from checkllm import dataset

@dataset("tests/fixtures/cases.yaml")
def test_across_cases(check, case):
    output = my_agent(case.input)
    check.contains(output, case.expected)
    if case.context:
        check.hallucination(output, context=case.context)
```

Or use a Python generator:

```python
from checkllm import Case, dataset

def my_cases():
    yield Case(input="Hello", expected="greeting", criteria="friendly")
    yield Case(input="Goodbye", expected="farewell", criteria="polite")

@dataset(my_cases)
def test_generated(check, case):
    output = my_agent(case.input)
    check.rubric(output, criteria=case.criteria)
```

## Custom Metrics

```python
import checkllm
from checkllm import CheckResult

@checkllm.metric("brevity")
def brevity_check(output: str, max_words: int = 50, **kwargs) -> CheckResult:
    word_count = len(output.split())
    return CheckResult(
        passed=word_count <= max_words,
        score=min(1.0, max_words / max(word_count, 1)),
        reasoning=f"{word_count} words (limit: {max_words})",
        cost=0.0,
        latency_ms=0,
        metric_name="brevity",
    )

def test_brevity(check):
    output = my_agent("Explain quantum physics")
    check.run_metric("brevity", output=output, max_words=100)
```

## Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_quality(check):
    output = await my_async_agent("What is Python?")

    await check.ahallucination(output, context="...")
    await check.arelevance(output, query="What is Python?")
    await check.atoxicity(output)
    await check.arubric(output, criteria="concise and accurate")

    # Deterministic checks are always sync (instant, no I/O)
    check.contains(output, "Python")
```

## Separating Fast and Slow Tests

Mark LLM tests so you can skip them in fast CI runs:

```python
import pytest

@pytest.mark.llm
def test_with_llm(check):
    check.hallucination(output, context=ctx)

def test_fast(check):
    check.contains(output, "Python")
```

```bash
# Run only fast deterministic tests
pytest -m "not llm"

# Run only LLM tests
pytest -m llm
```

If `OPENAI_API_KEY` is not set, LLM checks automatically skip instead of crashing.

## Regression Detection

checkllm uses Welch's t-test to detect statistically significant score regressions.

```bash
checkllm snapshot tests/ --output .checkllm/snapshots/v1.json
# ... make changes ...
checkllm snapshot tests/ --output .checkllm/snapshots/v2.json
checkllm diff -b .checkllm/snapshots/v1.json -c .checkllm/snapshots/v2.json

# Fail CI on regression
checkllm diff -b v1.json -c v2.json --fail-on-regression
```

## Reporting

```bash
# HTML report
checkllm report tests/ --output report.html

# JUnit XML for CI/CD
checkllm run tests/ --junit-xml results.xml

# pytest flags work directly
pytest tests/ --checkllm-snapshot=snap.json --checkllm-report=report.html
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `checkllm run <path>` | Run tests with `--snapshot`, `--html-report`, `--junit-xml`, `--compare`, `--fail-on-regression` |
| `checkllm snapshot <path>` | Save test results as baseline (`--output PATH`) |
| `checkllm report <path>` | Generate HTML report (`--output PATH`, `--junit-xml PATH`) |
| `checkllm diff` | Compare snapshots (`--baseline`, `--current`, `--fail-on-regression`) |
| `checkllm eval` | Evaluate prompt template (`--prompt`, `--dataset`, `--metric`, `--threshold`) |
| `checkllm init [path]` | Scaffold a new project |
| `checkllm list-metrics` | List available metrics |
| `checkllm --version` | Show version |

## Configuration

```toml
[tool.checkllm]
judge_backend = "openai"           # "openai" or "anthropic"
judge_model = "gpt-4o"             # Model for LLM-as-judge
default_threshold = 0.8            # Pass/fail threshold (0.0-1.0)
runs_per_test = 1                  # Repeat LLM checks N times
snapshot_dir = ".checkllm/snapshots"
confidence_level = 0.95
p_value_threshold = 0.05
```

Environment variable overrides: `CHECKLLM_JUDGE_BACKEND`, `CHECKLLM_JUDGE_MODEL`, `CHECKLLM_DEFAULT_THRESHOLD`, `CHECKLLM_RUNS_PER_TEST`.

## Custom Judge Backends

### Anthropic Claude

```toml
[tool.checkllm]
judge_backend = "anthropic"
judge_model = "claude-sonnet-4-6"
```

### Your Own Backend

Implement the `JudgeBackend` protocol:

```python
from checkllm import JudgeBackend, JudgeResponse
from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig

class MyJudge:
    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        return JudgeResponse(score=0.9, reasoning="Looks good", cost=0.0)

config = CheckllmConfig()
collector = CheckCollector(config=config, judge=MyJudge())
```

## Project Setup

```bash
checkllm init
```

Creates `pyproject.toml` config, sample test file, sample dataset, and `.checkllm/snapshots/` directory.

## License

MIT

# checkllm

Test LLM-powered applications with the same rigor as traditional software.

checkllm is a pytest plugin and CLI that lets you write assertions for LLM outputs using deterministic checks, LLM-as-judge evaluation, and statistical regression detection.

## Why checkllm?

- **Works with pytest** - no new test runner to learn, just add a `check` fixture
- **Free deterministic checks** run instantly with zero API calls
- **LLM-as-judge** for subjective quality (hallucination, relevance, toxicity, custom rubrics)
- **Statistical regression detection** using Welch's t-test, not just "did it change?"
- **Multiple judge backends** - OpenAI and Anthropic, or bring your own
- **Judge response caching** - skip redundant API calls, save time and money
- **Cost budgets** - set a spending limit per run to avoid surprise bills
- **Historical run tracking** - see quality trends across prompt iterations
- **One command** to snapshot, report, diff, or compare your test results

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

    # Compound checks
    check.all_of(output, ["Python", "programming", "language"])
    check.any_of(output, ["Python", "Java", "Rust"])
    check.none_of(output, ["error", "undefined", "null"])

    # Code validation
    check.is_json('{"key": "value"}')
    check.is_valid_python("def hello():\n    return 42")

    # Text analysis
    check.similarity(output, expected_output, threshold=0.8)
    check.readability(output, max_grade=8.0)
    check.sentence_count(output, min_sentences=2, max_sentences=5)
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
    check.fluency(output)
    check.coherence(output)
    check.sentiment(output, threshold=0.6)
    check.correctness(output, expected="Climate change is causing...")
```

Each check records a score (0.0-1.0), pass/fail status, reasoning, cost, and latency.

### Custom Thresholds

```python
check.hallucination(output, context=ctx, threshold=0.9)  # stricter
check.relevance(output, query=q, threshold=0.6)           # more lenient
```

### Custom Judge Prompts

Override the default system prompt for any metric:

```python
check.hallucination(
    output, context=ctx,
    system_prompt="You are a medical accuracy reviewer. Score strictly."
)
check.rubric(
    output, criteria="must include citations",
    system_prompt="You are an academic writing evaluator."
)
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

## Judge Response Caching

Judge calls are cached automatically in `.checkllm/cache.db` (SQLite). When you re-run tests with the same output+metric+model combination, cached results are returned instantly at zero cost.

```bash
# View cache statistics
checkllm cache --stats

# Clear the cache
checkllm cache --clear

# Disable caching for a run
checkllm run tests/ --no-cache
```

Configure caching in `pyproject.toml`:

```toml
[tool.checkllm]
cache_enabled = true
cache_ttl_seconds = 604800   # 7 days (default)
```

Or via environment variables: `CHECKLLM_CACHE_ENABLED=false`, `CHECKLLM_NO_CACHE=1`.

## Cost Budgets

Set a maximum USD spend per run to avoid accidental bills:

```bash
checkllm run tests/ --budget 5.00
checkllm eval --prompt "..." --dataset cases.yaml --budget 2.00
```

When the budget is exceeded, remaining judge calls are skipped (not failed) with a clear warning.

```toml
[tool.checkllm]
budget = 10.00
```

Or via environment: `CHECKLLM_BUDGET=5.00`.

## Historical Run Tracking

Every test run is automatically recorded in `.checkllm/history.db`. View trends across prompt iterations:

```bash
# List recent runs
checkllm history

# View details for a specific run
checkllm history --run 5

# Compare two runs side-by-side
checkllm history --compare 3,7

# View score trend for a specific test+metric
checkllm history --trend "test_qa::hallucination"
```

Runs capture: timestamp, git commit, label, per-test scores, costs, and pass/fail status.

Label your runs for easy identification:

```bash
checkllm run tests/ --label "prompt-v3"
```

## Dataset-Driven Testing

Supports **YAML**, **JSON**, and **CSV** datasets:

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

```json
[
  {"input": "What is Python?", "expected": "A programming language", "query": "Explain Python"},
  {"input": "What is 2+2?", "expected": "4"}
]
```

```csv
input,expected,query,criteria
What is Python?,A programming language,Explain Python,accurate
What is 2+2?,4,math,correct
```

```python
from checkllm import dataset

@dataset("tests/fixtures/cases.yaml")  # or .json or .csv
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

## Soft Assertions (check.expect)

Use `check.expect` for monitoring-style checks that are recorded in reports but never fail the test. Perfect for tracking quality metrics during prompt iteration without blocking CI:

```python
def test_with_soft_checks(check):
    output = my_agent("Explain quantum physics")

    # Hard checks — must pass
    check.contains(output, "quantum")
    check.max_tokens(output, limit=500)

    # Soft checks — recorded but won't fail the test
    check.expect.word_count(output, max_words=100)
    check.expect.readability(output, max_grade=10.0)
    check.expect.similarity(output, ideal_output, threshold=0.9)
```

Soft check results appear in reports with `[soft]` prefix and preserve their actual scores, so you can track trends without blocking deployments.

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

## Parallel Judge Execution

Async judge calls are rate-limited via a configurable semaphore (default: 10 concurrent requests):

```toml
[tool.checkllm]
max_concurrency = 10
```

```python
import asyncio, pytest

@pytest.mark.asyncio
async def test_parallel_judges(check):
    output = my_agent("...")

    # These run in parallel, up to max_concurrency
    results = await asyncio.gather(
        check.ahallucination(output, context="..."),
        check.arelevance(output, query="..."),
        check.atoxicity(output),
    )
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

## Structured Logging

Enable debug logging to see cache hits/misses, costs, and judge call details:

```bash
export CHECKLLM_LOG_LEVEL=DEBUG
pytest tests/
```

```toml
[tool.checkllm]
log_level = "INFO"   # DEBUG, INFO, WARNING (default), ERROR
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `checkllm run <path>` | Run tests with `--snapshot`, `--html-report`, `--junit-xml`, `--compare`, `--fail-on-regression`, `--budget`, `--no-cache`, `--label` |
| `checkllm snapshot <path>` | Save test results as baseline (`--output PATH`) |
| `checkllm report <path>` | Generate HTML report (`--output PATH`, `--junit-xml PATH`) |
| `checkllm diff` | Compare snapshots (`--baseline`, `--current`, `--fail-on-regression`) |
| `checkllm eval` | Evaluate prompt template (`--prompt`, `--dataset`, `--metric`, `--threshold`, `--budget`, `--no-cache`) |
| `checkllm history` | View run history (`--run ID`, `--compare ID1,ID2`, `--trend test::metric`, `--limit N`) |
| `checkllm cache` | Manage cache (`--stats`, `--clear`) |
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

# Caching
cache_enabled = true               # Toggle judge response caching
cache_ttl_seconds = 604800         # Cache expiration (7 days)

# Performance
max_concurrency = 10               # Parallel judge calls

# Cost control
budget = 10.00                     # Max USD per run (optional)

# Logging
log_level = "WARNING"              # DEBUG, INFO, WARNING, ERROR
```

Environment variable overrides: `CHECKLLM_JUDGE_BACKEND`, `CHECKLLM_JUDGE_MODEL`, `CHECKLLM_DEFAULT_THRESHOLD`, `CHECKLLM_RUNS_PER_TEST`, `CHECKLLM_CACHE_ENABLED`, `CHECKLLM_MAX_CONCURRENCY`, `CHECKLLM_BUDGET`, `CHECKLLM_LOG_LEVEL`.

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

## Configuring the Judge in conftest.py

To use a cheaper model or a custom backend for all tests:

```python
# tests/conftest.py
import pytest
from checkllm.check import CheckCollector
from checkllm.config import load_config
from checkllm.judge import OpenAIJudge
from checkllm.pytest_plugin import _CHECKLLM_KEY

@pytest.fixture
def check(request):
    config = load_config()
    judge = OpenAIJudge(model="gpt-4o-mini")  # cheaper model for dev
    collector = CheckCollector(config=config, judge=judge)
    request.node.stash[_CHECKLLM_KEY] = collector
    return collector
```

## Project Setup

```bash
checkllm init
```

Creates `pyproject.toml`, `tests/conftest.py`, sample test file, sample dataset, and `.checkllm/snapshots/` directory.

## Examples

See the [examples/](examples/) directory for working code:

- [test_basic.py](examples/test_basic.py) - Deterministic checks (no API key needed)
- [test_dataset_driven.py](examples/test_dataset_driven.py) - YAML and generator datasets
- [test_custom_metrics.py](examples/test_custom_metrics.py) - Register domain-specific metrics
- [test_llm_judge.py](examples/test_llm_judge.py) - LLM-as-judge evaluation
- [test_regression_workflow.py](examples/test_regression_workflow.py) - Snapshot and regression detection

## License

MIT

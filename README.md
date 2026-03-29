# checkllm

Test LLM-powered applications with the same rigor as traditional software.

checkllm is a pytest plugin, CLI, and runtime guardrails library that lets you write assertions for LLM outputs using deterministic checks, LLM-as-judge evaluation, statistical regression detection, and multi-provider consensus judging.

## Why checkllm?

- **Works with pytest** — no new test runner, just add a `check` fixture
- **Free deterministic checks** run instantly with zero API calls
- **16 LLM-as-judge metrics** — hallucination, relevance, faithfulness, bias, and more
- **7 judge backends** — OpenAI, Anthropic, Gemini, Azure, Ollama (local), LiteLLM (100+ models), custom HTTP
- **Consensus judging** — run multiple judges and aggregate with 7 strategies
- **Parallel evaluation engines** — async, threaded, process-based, or hybrid
- **Embedding-based semantic similarity** — OpenAI or local sentence-transformers
- **Runtime guardrails** — validate LLM outputs in production with Guard, middleware, and decorators
- **Rate limiting & circuit breaker** — resilient production judge calls with fallback
- **Statistical regression detection** using Welch's t-test
- **Cost budgets & caching** — control spend, cache judge responses
- **Rich reporting** — HTML, Markdown, JUnit XML, JSONL, CSV, A/B comparison, trend charts, GitHub PR comments
- **Config profiles** — dev/ci/prod with environment-based switching
- **Watch mode** — re-run tests on file changes
- **Programmatic API** — use outside pytest with `evaluate()`, `check_output()`, or `Evaluator` builder

## Installation

```bash
pip install checkllm
```

With optional providers:

```bash
pip install checkllm[anthropic]     # Anthropic Claude
pip install checkllm[gemini]        # Google Gemini
pip install checkllm[litellm]       # 100+ models via LiteLLM
pip install checkllm[embeddings]    # Local sentence-transformers
pip install checkllm[all]           # Everything
```

## Quick Start

### 1. Write a test

```python
def test_output_quality(check):
    output = my_agent("What is Python?")

    # Deterministic checks (free, instant)
    check.contains(output, "programming language")
    check.not_contains(output, "JavaScript")
    check.max_tokens(output, limit=200)

    # LLM-as-judge checks (requires API key)
    check.hallucination(output, context="Python is a high-level programming language.")
    check.relevance(output, query="What is Python?")
    check.toxicity(output)
```

### 2. Run it

```bash
export OPENAI_API_KEY=sk-...
pytest tests/ -v

# Or use the CLI
checkllm run tests/
```

### 3. Track regressions

```bash
checkllm snapshot tests/ --output baseline.json
# ... make changes ...
checkllm snapshot tests/ --output current.json
checkllm diff --baseline baseline.json --current current.json
```

## Deterministic Checks

Zero-cost, zero-latency checks that run locally:

```python
def test_deterministic(check):
    output = my_agent("...")

    # String checks
    check.contains(output, "expected substring")
    check.not_contains(output, "forbidden text")
    check.exact_match(output, "exact expected output")
    check.starts_with(output, "Python")
    check.ends_with(output, "language.")
    check.regex(output, pattern=r"\d{3}-\d{4}")

    # Length checks
    check.max_tokens(output, limit=500)
    check.min_tokens(output, minimum=10)
    check.word_count(output, min_words=5, max_words=100)
    check.char_count(output, min_chars=20)
    check.sentence_count(output, min_sentences=2, max_sentences=5)

    # Structure checks
    check.is_json('{"key": "value"}')
    check.is_valid_python("def hello():\n    return 42")
    check.json_schema(output, schema=MyPydanticModel)

    # Similarity & readability
    check.similarity(output, expected, threshold=0.8)
    check.readability(output, max_grade=8.0)

    # Compound checks
    check.all_of(output, ["Python", "programming", "language"])
    check.any_of(output, ["Python", "Java", "Rust"])
    check.none_of(output, ["error", "undefined", "null"])

    # Safety & compliance
    check.no_pii(output)
    check.language(output, expected="en")

    # Numeric extraction
    check.greater_than("Score: 85", threshold=70)
    check.less_than("Latency: 120ms", threshold=200)
    check.between("Confidence: 0.87", low=0.0, high=1.0)

    # Performance
    check.latency(response_time_ms, max_ms=2000)
    check.cost(api_cost_usd, max_usd=0.05)
```

## LLM-as-Judge Metrics (16 built-in)

```python
def test_llm_quality(check):
    output = my_agent("Summarize this article about climate change.")
    article = "..."

    # Core quality
    check.hallucination(output, context=article)
    check.relevance(output, query="Summarize the article")
    check.toxicity(output)
    check.fluency(output)
    check.coherence(output)
    check.correctness(output, expected="Climate change is...")
    check.rubric(output, criteria="concise, mentions key findings")

    # RAG-specific
    check.faithfulness(output, context=article)
    check.context_relevance(context=article, query="climate change summary")
    check.answer_completeness(output, query="Summarize all key findings")
    check.groundedness(output, sources=[article, supplementary])

    # Instruction & format
    check.instruction_following(output, instructions="Respond in bullet points under 100 words")
    check.summarization(output, source=article)

    # Safety & bias
    check.bias(output)
    check.sentiment(output, threshold=0.6)

    # Multi-output consistency
    check.consistency([output1, output2, output3])
```

## Multi-Provider Judge Backends

```python
from checkllm import (
    OpenAIJudge, AnthropicJudge, GeminiJudge,
    AzureOpenAIJudge, OllamaJudge, LiteLLMJudge,
    CustomHTTPJudge, create_judge,
)

# Factory function
judge = create_judge("gemini", model="gemini-2.0-flash")
judge = create_judge("ollama", model="llama3.1")  # Free, local
judge = create_judge("litellm", model="claude-sonnet-4-6")  # 100+ models
judge = create_judge("azure", deployment="my-gpt4")

# Or use directly
judge = OllamaJudge(model="llama3.1")  # No API key needed
judge = CustomHTTPJudge(url="https://my-server/evaluate")
```

Configure in `pyproject.toml`:

```toml
[tool.checkllm]
judge_backend = "gemini"
judge_model = "gemini-2.0-flash"
```

## Consensus Judging

Run the same check across multiple judges and aggregate:

```python
from checkllm import ConsensusJudge, OpenAIJudge, AnthropicJudge, OllamaJudge, consensus

# Create a consensus judge
judges = [
    ("gpt4o", OpenAIJudge(model="gpt-4o")),
    ("claude", AnthropicJudge(model="claude-sonnet-4-6")),
    ("llama", OllamaJudge(model="llama3.1")),
]

# 7 strategies: majority, unanimous, mean, weighted, median, min, max
cj = ConsensusJudge(judges=judges, strategy="mean", threshold=0.8)

# Use as a regular judge
result = await consensus(
    output="The sky is blue.",
    metric_name="hallucination",
    judges=judges,
    strategy="majority",
    context="Light scatters in the atmosphere, making the sky appear blue."
)
print(result.agreement_ratio)  # 1.0 if all judges agree
print(result.votes)  # Individual judge results
```

## Parallel Evaluation Engines

```python
from checkllm import AsyncEngine, ThreadPoolEngine, HybridEngine, create_engine

# AsyncEngine — best for I/O-bound judge calls
async with AsyncEngine(max_concurrency=20) as engine:
    tasks = [engine.submit(some_coro()) for _ in range(100)]
    results = await engine.gather(tasks)

# ThreadPoolEngine — for sync code calling async judges
async with ThreadPoolEngine(max_workers=8) as engine:
    tasks = [engine.submit(some_coro()) for _ in range(50)]
    results = await engine.gather(tasks)

# HybridEngine — auto-routes judges to async, deterministic to threads
async with HybridEngine() as engine:
    io_task = await engine.submit_io(judge_coro())
    cpu_task = await engine.submit_cpu(heavy_check())
    results = await engine.gather([io_task, cpu_task])

# Auto-select best engine
engine = create_engine("auto")  # Picks based on CPU count
```

Configure in `pyproject.toml`:

```toml
[tool.checkllm]
engine = "auto"   # "async", "thread", "process", "hybrid", "auto"
```

## Embedding-based Semantic Similarity

```python
from checkllm import OpenAIEmbeddings, semantic_similarity, batch_semantic_similarity

backend = OpenAIEmbeddings(model="text-embedding-3-small")

# Single comparison
result = await semantic_similarity(
    "Python is a programming language",
    "Python is a coding language",
    backend=backend,
    threshold=0.85,
)
print(result.score)  # ~0.95

# Batch comparison (deduplicates texts automatically)
pairs = [("output1", "expected1"), ("output2", "expected2")]
results = await batch_semantic_similarity(pairs, backend=backend)
```

Local embeddings (free, no API key):

```python
from checkllm.embeddings import SentenceTransformerEmbeddings

backend = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
```

## Guardrails (Runtime Validation)

Use checkllm checks in production, not just tests:

```python
from checkllm import Guard, CheckSpec

# Define your guard
guard = Guard(checks=[
    CheckSpec(check_type="no_pii"),
    CheckSpec(check_type="max_tokens", params={"limit": 500}),
    CheckSpec(check_type="toxicity", params={"threshold": 0.9}),
])

# Validate output
result = guard.validate(llm_output)
if not result.valid:
    print(result.summary())
    result.raise_on_failure()  # Raises GuardrailError

# Or use as a callable
safe_output = guard(llm_output)  # Raises if invalid

# Predefined guards
from checkllm.guardrails import safety_guard, quality_guard, rag_guard
result = safety_guard.validate(output)
```

### FastAPI Middleware

```python
from fastapi import FastAPI
from checkllm import Guard, CheckSpec, GuardrailMiddleware

app = FastAPI()
guard = Guard(checks=[CheckSpec(check_type="no_pii"), CheckSpec(check_type="toxicity")])
app.add_middleware(GuardrailMiddleware, guard=guard, response_field="output")
```

### Function Decorator

```python
from checkllm import guardrail, CheckSpec

@guardrail(checks=[CheckSpec(check_type="no_pii"), CheckSpec(check_type="max_tokens", params={"limit": 200})])
def generate_response(prompt: str) -> str:
    return my_llm(prompt)  # Output is validated automatically
```

## Rate Limiting & Circuit Breaker

```python
from checkllm import (
    TokenBucketRateLimiter, CircuitBreaker, ResilientJudge,
    OpenAIJudge, OllamaJudge,
)

# Wrap a judge with resilience
resilient = ResilientJudge(
    judge=OpenAIJudge(model="gpt-4o"),
    rate_limiter=TokenBucketRateLimiter(rate=10.0, burst=20),
    circuit_breaker=CircuitBreaker(failure_threshold=5, recovery_timeout=60),
    fallback=OllamaJudge(model="llama3.1"),  # Free local fallback
    timeout=30.0,
)

# Use like any judge — rate limiting, circuit breaking, and fallback are automatic
result = await resilient.evaluate(prompt="...", system_prompt="...")
```

## Programmatic API

Use checkllm outside of pytest:

```python
from checkllm import check_output, evaluate, Evaluator

# One-liner
result = check_output("LLM output here", checks=["no_pii", "max_tokens:200"])

# Async with full control
result = await evaluate(
    output="The answer is 42.",
    checks=[
        {"type": "contains", "params": {"substring": "42"}},
        {"type": "hallucination", "params": {"context": "The answer to everything is 42."}},
    ],
)

# Builder pattern
evaluator = (
    Evaluator()
    .with_judge("openai", model="gpt-4o-mini")
    .with_threshold(0.8)
    .with_budget(5.0)
    .add_check("contains", substring="expected")
    .add_check("no_pii")
    .add_check("hallucination", context="source text")
)
result = evaluator.run("LLM output here")

# Batch evaluation
results = await evaluator.batch_run(["output1", "output2", "output3"])
```

## Configuration Profiles

```toml
[tool.checkllm]
judge_model = "gpt-4o"
default_threshold = 0.8
engine = "auto"

[tool.checkllm.profiles.dev]
judge_model = "gpt-4o-mini"
budget = 1.0
log_level = "DEBUG"

[tool.checkllm.profiles.ci]
cache_enabled = false
budget = 10.0
engine = "async"

[tool.checkllm.profiles.prod]
judge_model = "gpt-4o"
default_threshold = 0.9
max_concurrency = 20
```

Activate a profile:

```bash
CHECKLLM_PROFILE=ci checkllm run tests/
checkllm run tests/ --profile dev
```

## Watch Mode

```bash
checkllm watch tests/
checkllm watch tests/ --interval 2.0 --pattern "*.py" --pattern "*.yaml"
checkllm watch tests/ --watch src/ --budget 1.0 --profile dev
```

Re-runs tests automatically when files change. Debounced to avoid rapid re-triggers.

## Enhanced Reporting

### A/B Model Comparison

```python
from checkllm.reporting import ComparisonReport, generate_comparison_html

report = ComparisonReport(
    results_a=gpt4_results, label_a="GPT-4o",
    results_b=claude_results, label_b="Claude Sonnet",
)
generate_comparison_html(report, Path("comparison.html"))
```

### Trend Charts

```python
from checkllm.reporting import generate_trend_html, TrendData

trends = [TrendData(run_id=i, timestamp=t, label=l, results=r) for ...]
generate_trend_html(trends, Path("trends.html"))  # SVG charts, no JS deps
```

### CSV Export

```python
from checkllm.reporting import write_csv, results_to_dataframe

write_csv(results, Path("results.csv"))
df_data = results_to_dataframe(results)  # List of dicts for pandas
```

### GitHub PR Comments

```python
from checkllm.reporting import generate_pr_comment, post_pr_comment

comment = generate_pr_comment(results, comparison=report)
post_pr_comment(comment, repo="owner/repo", pr_number=123)
```

## Dataset-Driven Testing

```yaml
# tests/fixtures/cases.yaml
- input: "What is Python?"
  expected: "Python is a programming language"
  context: "Python was created by Guido van Rossum."
  criteria: "accurate, mentions creator"
```

```python
from checkllm import dataset

@dataset("tests/fixtures/cases.yaml")  # Also supports .json, .csv
def test_across_cases(check, case):
    output = my_agent(case.input)
    check.contains(output, case.expected)
    if case.context:
        check.hallucination(output, context=case.context)
```

## Soft Assertions

```python
def test_with_soft_checks(check):
    output = my_agent("Explain quantum physics")

    # Hard checks — must pass
    check.contains(output, "quantum")

    # Soft checks — recorded but won't fail the test
    check.expect.word_count(output, max_words=100)
    check.expect.readability(output, max_grade=10.0)
    check.expect.relevance(output, query="quantum physics")
```

## Testing Without API Keys

```python
from checkllm.testing import MockJudge, make_collector, assert_all_passed

judge = MockJudge(default_score=0.9)
judge.add_response("hallucination", score=0.95, reasoning="Well grounded")

collector = make_collector(judge=judge)
collector.hallucination("output text", context="source text")

assert_all_passed(collector)
judge.assert_called("hallucination")
```

## Custom Metrics

```python
from checkllm import metric, CheckResult

@metric("brevity")
def brevity_check(output: str, max_words: int = 50, **kwargs) -> CheckResult:
    word_count = len(output.split())
    return CheckResult(
        passed=word_count <= max_words,
        score=min(1.0, max_words / max(word_count, 1)),
        reasoning=f"{word_count} words (limit: {max_words})",
        cost=0.0, latency_ms=0, metric_name="brevity",
    )
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `checkllm run <path>` | Run tests with `--snapshot`, `--html-report`, `--junit-xml`, `--budget`, `--no-cache`, `--label`, `--profile` |
| `checkllm watch <path>` | Watch for changes and re-run (`--interval`, `--pattern`, `--watch`) |
| `checkllm snapshot <path>` | Save baseline (`--output PATH`) |
| `checkllm report <path>` | Generate HTML report (`--output PATH`) |
| `checkllm diff` | Compare snapshots (`--baseline`, `--current`, `--fail-on-regression`) |
| `checkllm eval` | Evaluate prompt templates (`--prompt`, `--dataset`, `--metric`, `--budget`) |
| `checkllm history` | View run history (`--run ID`, `--compare`, `--trend`) |
| `checkllm cache` | Manage cache (`--stats`, `--clear`) |
| `checkllm init [path]` | Scaffold a new project |
| `checkllm list-metrics` | List available metrics |

## Configuration

```toml
[tool.checkllm]
judge_backend = "openai"           # openai, anthropic, gemini, azure, ollama, litellm
judge_model = "gpt-4o"
default_threshold = 0.8
runs_per_test = 1
engine = "auto"                    # async, thread, process, hybrid, auto
max_concurrency = 10
budget = 10.00                     # Max USD per run (optional)
cache_enabled = true
cache_ttl_seconds = 604800         # 7 days
log_level = "WARNING"
```

All settings support environment variable overrides: `CHECKLLM_JUDGE_BACKEND`, `CHECKLLM_JUDGE_MODEL`, `CHECKLLM_ENGINE`, `CHECKLLM_BUDGET`, `CHECKLLM_PROFILE`, etc.

## License

MIT

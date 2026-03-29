# Changelog

## 1.0.0 (2026-03-29)

### Parallel Evaluation Engines

- **AsyncEngine** — asyncio-based with configurable concurrency and backpressure
- **ThreadPoolEngine** — threading-based for I/O-bound judge calls in sync code
- **ProcessPoolEngine** — for CPU-bound deterministic checks at scale
- **HybridEngine** — auto-routes judge calls to async, deterministic to threads
- **create_engine()** factory with `engine="auto"` smart selection
- All engines track tasks submitted/completed, average execution time, queue depth

### Multi-Provider Judge Backends

- **GeminiJudge** — Google Gemini API (gemini-2.0-flash, gemini-2.0-pro, etc.)
- **AzureOpenAIJudge** — Azure-hosted OpenAI models
- **OllamaJudge** — local models via Ollama HTTP API (cost=0, no API key needed)
- **LiteLLMJudge** — 100+ models through LiteLLM unified interface
- **CustomHTTPJudge** — any REST endpoint with configurable response parsing
- **create_judge()** factory for all 7 backends

### Consensus Judging

- Run the same evaluation across multiple judges simultaneously
- 7 aggregation strategies: `majority`, `unanimous`, `mean`, `weighted`, `median`, `min`, `max`
- `ConsensusJudge` class implements `JudgeBackend` protocol (nestable)
- `consensus()` convenience function for quick multi-judge evaluation
- Tracks agreement ratio, individual votes, costs per judge

### New LLM-as-Judge Metrics (8 new)

- **faithfulness** — RAG: is the answer faithful to retrieved context?
- **context_relevance** — RAG: is the retrieved context relevant to the query?
- **answer_completeness** — does the answer fully address all parts of the question?
- **instruction_following** — does the output follow given format/style/constraint instructions?
- **summarization** — summary accuracy, conciseness, and key information retention
- **bias** — demographic, cultural, gender, racial bias detection (1.0=no bias)
- **consistency** — multi-output consistency comparison
- **groundedness** — claim-by-claim grounding across multiple source documents

### Embedding-based Semantic Similarity

- `check.semantic_similarity()` using embedding cosine similarity
- **OpenAIEmbeddings** — text-embedding-3-small/large, ada-002 with cost tracking
- **SentenceTransformerEmbeddings** — local models (all-MiniLM-L6-v2, etc.), cost=0
- **CachedEmbeddings** — LRU cache wrapper for any embedding backend
- `batch_semantic_similarity()` with text deduplication for efficiency

### Guardrails Mode (Runtime Validation)

- **Guard** class for production runtime LLM output validation
- `guard.validate(output)` / `guard.avalidate(output)` — sync and async
- **CheckSpec** model for declarative check definitions
- Predefined guards: `safety_guard`, `quality_guard`, `rag_guard`
- **GuardrailMiddleware** — ASGI middleware for FastAPI/Starlette
- **@guardrail** decorator for function-level validation
- Soft checks that record but don't block

### Rate Limiting & Circuit Breaker

- **TokenBucketRateLimiter** — classic token bucket with async acquire
- **PerProviderRateLimiter** — separate limits per provider
- **CircuitBreaker** — auto-disable failing backends (CLOSED→OPEN→HALF_OPEN→CLOSED)
- **ResilientJudge** — wraps any judge with rate limiting + circuit breaker + fallback
- **RetryPolicy** — configurable retry with exponential backoff and jitter
- **with_retry()** async utility

### Enhanced Reporting

- **A/B comparison reports** — side-by-side HTML/Markdown/terminal with color-coded deltas
- **Trend reports** — inline SVG charts for score/pass-rate/cost over time
- **CSV export** — `write_csv()` and `results_to_dataframe()` for data analysis
- **GitHub PR comments** — `generate_pr_comment()` with collapsible details, `post_pr_comment()` via API

### Configuration Profiles

- Define profiles in `[tool.checkllm.profiles.dev]`, `[tool.checkllm.profiles.ci]`, etc.
- Activate via `CHECKLLM_PROFILE=ci` or `--profile ci`
- Merge order: defaults → base config → profile → env vars
- New `engine` config option for parallel execution strategy

### Programmatic API

- `evaluate()` async function — use checkllm outside pytest
- `check_output()` sync wrapper — one-liner validation
- **Evaluator** builder — fluent API with `.with_judge()`, `.add_check()`, `.run()`, `.batch_run()`
- `parse_check_shorthand()` — parse `"no_pii"` or `"max_tokens:200"` into check specs

### Watch Mode

- `checkllm watch tests/` — re-run tests on file changes
- Cross-platform polling (no external dependencies)
- Debounced detection, configurable patterns and intervals
- Rich terminal output with run history

### Other

- 836 tests (up from ~250)
- 55 public API symbols exported from top-level package
- Optional dependencies: `checkllm[gemini]`, `checkllm[litellm]`, `checkllm[embeddings]`, `checkllm[all]`

## 0.3.0 (2026-03-29)

### Testing Helpers (`checkllm.testing`)

- **MockJudge** — fake judge backend with configurable scores, queued per-metric responses, and call tracking assertions
- **make_collector()** — factory with test-friendly defaults (cache disabled, MockJudge)
- **assert_all_passed()**, **assert_score_above()** — assertion helpers

### New Deterministic Checks

- **no_pii** — regex-based PII detection (email, phone, SSN, credit card, IP address)
- **language** — heuristic word-frequency language detection (en, es, fr, de, pt)
- **greater_than**, **less_than**, **between** — numeric extraction and comparison

### New Report Formats

- **Markdown report** (`--checkllm-markdown`) — ideal for PR comments and GitHub Actions
- **JSONL export** (`--checkllm-jsonl`) — one JSON record per line for data pipelines
- Both available as pytest flags and programmatic API

### Other

- **py.typed** marker for PEP 561 type checking
- Exported `MockJudge`, `make_collector`, `assert_all_passed`, `assert_score_above` from top-level package

## 0.2.0 (2026-03-28)

### New Checks (Deterministic)

- **similarity** — Levenshtein ratio for fuzzy string comparison
- **readability** — Flesch-Kincaid Grade Level with min/max bounds
- **min_tokens**, **word_count**, **char_count**, **sentence_count** — text length checks with flexible bounds
- **all_of**, **any_of**, **none_of** — compound substring checks
- **is_json** — validate parseable JSON without requiring a schema
- **is_valid_python** — validate Python syntax (auto-strips markdown code fences)

### New Metrics (LLM-as-Judge)

- **fluency** — writing quality and naturalness
- **coherence** — logical structure and consistency
- **sentiment** — tone/mood assessment (0=negative, 1=positive)
- **correctness** — semantic comparison against expected/reference answers

### Developer Experience

- **Soft assertions** (`check.expect.*`) — same API as `check` but never fails the test. Soft checks are recorded with `[soft]` prefix for monitoring during prompt iteration.
- **py.typed** — PEP 561 marker for type checking support
- **Rich progress bar** in `checkllm eval` command
- **Enhanced HTML report** — modern design with score bars, collapsible test sections, pass rate visualization

### Features

- **Judge response caching**: SQLite-backed content-addressable cache for LLM judge responses. Cache hits return instantly at zero cost. Configurable TTL (default 7 days). CLI commands: `checkllm cache --stats`, `checkllm cache --clear`. Disable with `--no-cache` or `CHECKLLM_CACHE_ENABLED=false`.
- **Parallel judge execution**: Async judge calls are rate-limited via configurable `asyncio.Semaphore` (default 10 concurrent). Use `asyncio.gather` with async check methods for parallel evaluation.
- **Cost budgets**: Set a max USD spend per run with `--budget`, `CHECKLLM_BUDGET`, or `budget` in pyproject.toml. When exceeded, remaining judge calls are skipped with a clear warning.
- **Historical run tracking**: Every test run is recorded in `.checkllm/history.db` with timestamp, git commit, label, scores, and costs. CLI: `checkllm history`, `--run ID`, `--compare ID1,ID2`, `--trend test::metric`.
- **Enhanced run comparison**: Side-by-side diff of two historical runs showing per-test score deltas with color-coded improvements/regressions.
- **CSV and JSON dataset support**: `load_dataset()` and `@dataset()` now auto-detect `.csv`, `.json`, `.yaml`, `.yml` files. CSV extra columns go into `case.metadata`.
- **Customizable judge prompts**: Override the system prompt for any built-in metric via `system_prompt=` kwarg on `check.hallucination()`, `check.relevance()`, `check.toxicity()`, `check.rubric()`, and their async variants.
- **Structured logging**: Python `logging` module integration. Set `CHECKLLM_LOG_LEVEL=DEBUG` or configure in pyproject.toml. Logs cache hits/misses, costs, budget warnings.

### Configuration

New `[tool.checkllm]` options: `cache_enabled`, `cache_dir`, `cache_ttl_seconds`, `max_concurrency`, `budget`, `log_level`.

New environment variables: `CHECKLLM_CACHE_ENABLED`, `CHECKLLM_CACHE_DIR`, `CHECKLLM_CACHE_TTL`, `CHECKLLM_MAX_CONCURRENCY`, `CHECKLLM_BUDGET`, `CHECKLLM_LOG_LEVEL`.

## 0.1.0 (2026-03-28)

Initial release.

### Features

- **pytest plugin** with `check` fixture for LLM testing in pytest
- **Deterministic checks**: `contains`, `not_contains`, `regex`, `json_schema`, `max_tokens`, `latency`, `cost`
- **LLM-as-judge metrics**: `hallucination`, `relevance`, `toxicity`, `rubric`
- **Custom metrics** via `@metric` decorator and plugin entry points
- **Dataset system**: YAML loading, generator functions, `@dataset` decorator for parametrized tests
- **Regression detection**: Welch's t-test with configurable p-value threshold
- **Snapshot system**: save/load/compare test result baselines
- **Reporting**: Rich terminal output, HTML reports, JUnit XML
- **CLI**: `checkllm run`, `snapshot`, `report`, `eval`, `diff`, `init`
- **Multiple judge backends**: OpenAI and Anthropic
- **Retry logic** with exponential backoff for transient API failures
- **Cost tracking** from OpenAI/Anthropic token usage
- **Configuration** via `pyproject.toml [tool.checkllm]` and environment variables

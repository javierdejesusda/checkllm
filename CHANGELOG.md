# Changelog

## v5.2.0 (2026-05-03)

### Agent-trajectory evaluation wedge

- **OpenTelemetry GenAI ingestion** — `checkllm.trace.otel_genai.otel_jsonl_to_trace_spans()` parses OTel-exported spans into `TraceSpan` objects. Maps `gen_ai.operation.name` and span-name conventions to `llm` / `tool` / `retriever` types.
- **`AgentTestCase.from_trace_jsonl()`** — build an `AgentTestCase` directly from an OTel trace export. Accepts `tool.arguments` as either JSON string or dict; degrades gracefully when arguments aren't a JSON object.
- **`TrajectoryMetricConfig`** dataclass — exposes the four sub-score weights (`ordering`, `loops`, `coverage`, `unexpected`) and `loop_threshold` as a single dataclass for ablation sweeps. `TrajectoryMetric.__init__` accepts a `config=` keyword (mutually exclusive with `weights=`). Backwards-compatible defaults preserved.

### Benchmarks

- **GAIA loader** (`checkllm.benchmarks.gaia_loader`) — license-gated via `CHECKLLM_GAIA_LICENSE_ACK=yes` + `HF_TOKEN`. Pinned to dataset SHA `682dd723…` for reproducibility.
- **τ-bench loader** (`checkllm.benchmarks.tau_bench_loader`) — `airline` and `retail` domains with synthetic fixtures bundled. Required `domain: str` field on `TauBenchTask` for type-safety.
- **End-to-end paper runner** (`benchmarks/paper/run_all.py`) — iterates `{model × benchmark × seed × task}` and writes a `manifest.json` with Phase-C placeholder columns (`model_version_sha`, `benchmark_sha`, `temperature`, `mean_latency_ms`, `total_cost_usd`).

### Paper experiments (Phase C, $0 budget)

- **C1 — controlled-noise validation:** `TrajectoryMetric` responds monotonically to drop/repeat/extra noise across 5 levels × 2 domains × 3 seeds (150 trajectories).
- **C2 — metric-vs-synthetic-truth correlation:** AUROC = 0.926 [0.91, 0.94] on 500 trajectories. Bootstrap 95% CIs (n=1000). Sub-score breakdown reveals ordering = 0.93, coverage = 0.82, unexpected = 0.82, loops = 0.51.
- **C3 — TrajectoryMetric weight ablation:** 1875-cell grid (5⁴ × 3 loop thresholds, 3 degenerate cells skipped). Library defaults are Pareto-optimal and within 2.4% of the best grid cell.
- **C4 — head-to-head vs DeepEval `ToolCorrectnessMetric`:** CheckLLM wins AUROC, Spearman ρ, and latency (~1546× faster) with Holm-Bonferroni-corrected p < 0.005. DeepEval wins mean abs error (reported honestly).

### Paper artifacts

- `paper/checkllm.tex` — sections 1–8 drafted with auto-generated tables and verified citations.
- `paper/bibliography.bib` — 47 verified entries across agent benchmarks, eval frameworks, judge-LLM literature, tool-use foundations, reproducibility, and statistical methodology.
- `paper/figures/ingest.py` — idempotent script that regenerates `generated_tables.tex` from the four `summary.json` files.
- `paper/reproducibility_checklist.md` — NeurIPS 2024 D&B reproducibility checklist with file:line evidence pointers.
- `paper/datasheets/{gaia,tau_bench}_checkllm.md` — 7-section Gebru datasheets.

### Integrations

- **Adapters for LangChain, LlamaIndex, CrewAI, and Pydantic-AI** — `to_checkllm_test_case()` / `to_checkllm_tool_calls()` translation helpers under `checkllm.integrations`. All four adapters use lazy imports — no runtime dep added.

### Reproducibility

- `Dockerfile` + `Makefile` (`make reproduce`, `make reproduce-smoke`, `make plots`, `make paper`).
- `CITATION.cff` with Zenodo DOI placeholder.
- `paper/lint.py` — structural verifier for the .tex (matched `\begin/\end`, citation/ref resolution).

### Documentation

- README rewritten around the agent-eval wedge: deterministic / composite / OTel-compatible.
- `docs/tutorials/evaluating-agents-in-10-minutes.md` — flagship tutorial with 8 verified executable code blocks.
- `docs/vs-deepeval.md`, `docs/vs-ragas.md`, `docs/vs-promptfoo.md` — honest comparison pages, ≤500 words each.

## v5.1.0 (2026-04-23)

### Retrieval metrics (RAG eval parity)
- Classical IR ranking metrics: `NDCG`, `MRR`, `MAPAtK`, `PrecisionAtK`, `RecallAtK`, `HitRateAtK` under `checkllm.metrics`. Pure-Python, no LLM call required.

### Providers
- **Native Vertex AI judge** — `VertexAIJudge` using `google-cloud-aiplatform`. Supports `vertex` / `vertexai` in `create_judge()`. ADC and explicit credentials both supported; project/location fall back to `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`. Install with `pip install checkllm[vertex]`.

### Reliability & throughput
- **Per-provider rate limiting** — new `checkllm.rate_limit` module: `TokenBucket`, `ProviderRateLimiter`, `RateLimit`, `RetryConfig`, `retry_with_backoff`. Dual RPM + TPM buckets per provider with sensible tier-1 defaults for every supported backend. `retry_with_backoff` honors `Retry-After` / `x-ratelimit-reset-*` headers on 429s and backs off exponentially on 5xx. Wired into `AsyncEngine.submit_judge` and configurable via `CheckllmConfig.rate_limits`.
- **Anthropic Message Batches API** — `AnthropicBatchRunner` alongside the existing OpenAI batch runner, unified behind a `BatchRunner` protocol and a `get_batch_runner(provider, ...)` factory. Automatic 50% batch-discount pricing. New `checkllm batch --batch {openai|anthropic}` CLI command.

### Observability
- **W3C trace-context propagation** — `propagate_trace_context()` injects `traceparent` / `tracestate` headers into every outbound judge HTTP call (OpenAI, Anthropic, Azure, DeepSeek, Ollama, Bedrock, custom HTTP). One evaluation now shows as one trace across the judge boundary. Install with `pip install checkllm[otel]`.
- **Anthropic streaming** — `StreamingEvaluator.evaluate_provider()` routes streaming through OpenAI or Anthropic; existing checkpoint/early-stop pipeline unchanged.

### Cost & experience
- **Cost attribution rollups** — `checkllm.pricing` ships a 2026-04 pricing snapshot for OpenAI, Anthropic, Gemini, DeepSeek, and Bedrock variants. `JudgeResponse` now carries `input_tokens` / `output_tokens` / `model` / `provider`; every `CheckResult` gets a `CostBreakdown`. New dashboard endpoints: `GET /api/cost/by-provider`, `by-metric`, `by-test`, `timeseries?bucket=hour|day`.
- **Live progress dashboard** — `checkllm.dashboard_ws` adds a Starlette app with `GET /live` (self-contained HTML) and `WS /ws/progress` that streams `test_started`, `check_completed`, `test_completed`, `run_completed` events from the new `ProgressBroker`. Optional `token=` gate for non-loopback deployments.

### Agent & red-team depth
- **Agent trajectory metrics** — `ToolParameterAccuracyMetric`, `ToolSelectionAccuracyMetric`, `TrajectoryMetric` (ordering/loop-detection/coverage/unexpected-tools) under `checkllm.metrics`. New `ToolCallTrace` model on `checkllm.agents`.
- **Red-team scorecards** — new `checkllm.redteam_scorecard`: `ExploitSuccessRate`, `OWASPTop10LLMScorecard`, `SensitiveDataExposureRate`, plus `generate_redteam_report()` returning a single dashboard-ready object.

### Experiment analysis
- `checkllm.analysis.correlation` — Pearson / Spearman metric correlations across runs.
- `checkllm.analysis.significance` — Welch's t-test, Mann-Whitney U, bootstrap CIs, Cohen's d for run-vs-run A/B comparison.

### Deterministic check parity
- `@check` decorator, `CHECK_REGISTRY`, and `run_check(name, ...)` symmetric with the existing `@metric` surface. Check composition: `AllOf(*checks)`, `AnyOf(*checks)`, `Not(check)`. All 39 built-in deterministic checks auto-registered.

### Vector stores & drift
- Connectors for **Pinecone, Weaviate, Milvus, Chroma** under `checkllm.integrations.*`, exposing a unified `connect(**config)` / `query(vector_or_text, top_k)` interface returning normalized `RetrievedContext` objects. Install with `pip install checkllm[vectorstores]`.
- `KBFaithfulnessMetric` — hallucination against an external KB by combining retrieval + faithfulness.
- `FreshnessAudit` — flag stale entries older than a TTL.
- **Judge drift detection** — `checkllm.drift`: `JudgeBaseline` (20 canonical probes, SHA-256 of responses, model version), `detect_drift(judge, baseline)`, plus `checkllm drift baseline` / `checkllm drift check` CLI commands.

### Benchmarks & tutorials
- Five new benchmarks: `squad_v2`, `arc_challenge`, `bbh_hard`, `drop_reading`, `cnn_dailymail` — with BLEU/ROUGE-L summary scoring. Total now 21 benchmarks.
- Five runnable Jupyter tutorials under `docs/notebooks/`: quickstart, RAG evaluation, conversational eval, agent trajectory, red-team. All run end-to-end offline via a stubbed `FakeJudge`.

### Breaking changes
- None. All additions are backward compatible; `JudgeResponse` new fields default to `None`/`0`.

### Deprecations (soft — no runtime warnings yet)
- `checkllm.resilience.TokenBucketRateLimiter` and `PerProviderRateLimiter` are now the legacy rate-limiting surface. New code should use `checkllm.rate_limit.TokenBucket` and `checkllm.rate_limit.ProviderRateLimiter` instead — the new types are what `AsyncEngine` wires up by default and are the only ones that honor `Retry-After` / `x-ratelimit-reset-*` headers on 429 responses. The legacy types will keep working; a runtime `DeprecationWarning` is planned for v5.2.

### Migration — legacy → new rate limiter

| Legacy (v5.0) | v5.1 replacement |
|---|---|
| `TokenBucketRateLimiter(rate=5, burst=10)` | `TokenBucket(capacity=10, refill_period=2.0)` |
| `PerProviderRateLimiter(defaults={"openai": (5, 10)})` | `ProviderRateLimiter(limits={"openai": RateLimit(rpm=300, tpm=30_000)})` |
| Manual `await limiter.acquire(...)` then judge call | `await engine.submit_judge(provider, factory, est_tokens=...)` handles both |
| No 429 handling | `retry_with_backoff` honors `Retry-After` + exponential backoff + jitter |

## v5.0.1 (2026-04-18)

### Competitor benchmark
- **Public competitor leaderboard** under `docs/benchmarks/` comparing checkllm to DeepEval, Ragas, and promptfoo on HaluBench, RAGTruth (hallucination, faithfulness, context_relevance), and TruthfulQA. checkllm holds rank 1 on every published row.
- **TruthfulQA balanced loader** — `load_truthfulqa_from_rows` now emits `best_answer` / `incorrect_answers[0]` sample pairs so ROC-AUC is well-defined on the slice.
- **Answer-aware `ContextRelevanceMetric`** — `evaluate(...)` accepts an optional `answer` kwarg; when provided, the judge grades whether the retrieved context precisely justifies that specific answer. The answer-less path keeps its original semantics.

## v3.2.0 (2026-04-06)

### VS Code Extension
- **checkllm for VS Code** — inline LLM test results in the editor
  - CodeLens annotations above test functions showing pass/fail count, average score, and cost
  - Green/red gutter dots next to each test function
  - Hover popup with per-check score table
  - Auto-refresh when `.checkllm/` result files change
  - Commands: Run Tests, Estimate Costs, Refresh Results
  - Configurable judge backend, model, threshold, and budget via VS Code settings
  - Status bar showing aggregate pass/fail count

## v3.1.0 (2026-04-06)

### Framework Integrations
- **LangChain integration** — `CheckllmCallbackHandler` validates chain/LLM outputs via `on_chain_end` and `on_llm_end` callbacks. Supports `"log"` and `"raise"` failure modes.
- **LlamaIndex integration** — `CheckllmCallbackHandler` validates query engine responses via `on_event_end` callback. Same API as the LangChain handler.
- Both integrations require zero framework dependencies at import time — only checkllm internals are used.

### Enhanced Dashboard
- **Trend charts** — `/api/trends` endpoint returns score-over-time data; frontend renders inline SVG line charts with pass/fail colored dots
- **Cost breakdown** — `/api/cost-breakdown` endpoint returns per-metric and per-test cost aggregation; frontend renders bar charts
- **Search/filter** — filter runs by label, commit, or test name in the runs list

## v3.0.0b1 (2026-04-06)

### CI/CD
- **`checkllm ci` command** — auto-detects GitHub Actions environment, runs tests, and posts formatted results as PR comments. Supports `--budget`, `--compare`, `--fail-on-regression`, and `--no-comment` flags.

### Pytest Helpers
- **Pre-built fixtures** — `shared_judge` (session-scoped judge reuse), `budget_session` (cross-test cost tracking), `auto_snapshot` (auto-save snapshots after sessions). Import from `checkllm.pytest_helpers`.

### Documentation
- **MkDocs site** — full documentation with guides for deterministic checks, LLM metrics, guardrails, CI/CD, custom metrics, plugins, and configuration. Run `mkdocs serve` locally.
- **CONTRIBUTING.md** — contributor guide with architecture overview, step-by-step instructions for adding checks, metrics, and plugins
- **Community templates** — GitHub issue templates (bug report, feature request, new metric proposal) and PR template

## v3.0.0a1 (2026-04-04)

### Developer Experience
- **Auto-detect judge backend** — checkllm now auto-detects the best available LLM judge from your environment (OpenAI > Anthropic > Gemini > Ollama). Zero config needed.
- **Interactive `checkllm init`** — scaffolds tailored test files based on your use case (RAG, chatbot, agent, general) with `--ci` flag for GitHub Actions
- **Smart error messages** — actionable guidance when API keys are missing, budgets are exceeded, or dependencies are needed
- **Cost estimation** — `checkllm estimate tests/` shows expected cost before running; `--dry-run` flag on `checkllm run`
- **Fluent assertions** — `check.that(output).contains("Python").has_no_pii().max_tokens(200)`

### Plugin System
- **Community metric discovery** — third-party packages can register metrics via entry points; `checkllm list-metrics` shows builtin + plugins with source attribution

### CLI
- New `checkllm estimate` command
- `checkllm run --dry-run` flag
- Enhanced `checkllm list-metrics` with categories and plugin discovery
- `checkllm init --use-case` and `--ci` flags

### Internal
- New modules: `discovery.py`, `errors.py`, `estimator.py`, `chain.py`
- Default `judge_backend` changed from `"openai"` to `"auto"`

## 2.0.0 (2026-03-29)

### New LLM-as-Judge Metrics (8 new, 24 total)

- **g_eval** — G-Eval: custom criteria evaluation with chain-of-thought reasoning (most flexible metric)
- **contextual_precision** — are the most relevant retrieved documents ranked higher? (Ragas-style)
- **contextual_recall** — what fraction of the ground-truth answer is supported by context? (Ragas-style)
- **task_completion** — did the LLM actually accomplish the user's stated goal?
- **role_adherence** — does the LLM stay in its assigned persona/role throughout?
- **tool_accuracy** — agent tool selection and parameter correctness evaluation
- **knowledge_retention** — does the chatbot remember facts from earlier in the conversation?
- **conversation_completeness** — were all user requests fulfilled across the multi-turn exchange?

### New Deterministic Checks (5 new, 33 total)

- **bleu** — BLEU score (1-4 gram) with brevity penalty for reference comparison
- **rouge_l** — ROUGE-L (LCS-based F1) for summary/translation evaluation
- **json_field** — JSONPath-style deep field assertions with conditions (exists, gt, lt, contains, type)
- **is_valid_sql** — SQL syntax validation (balanced parens, keywords, string literals)
- **is_valid_markdown** — markdown structure validation with optional header/list/code-block requirements

### Multi-Turn Conversational Evaluation

- **ConversationalTestCase** model for multi-turn conversation threads
- **Turn** model (role, content, metadata) with user/assistant/system filtering
- `format_transcript()` for readable conversation formatting
- Knowledge retention and conversation completeness metrics operate on full conversation context

### Agent / Agentic Workflow Evaluation

- **AgentTestCase** with query, steps, tool calls, trajectory, and final output
- **ToolCall** model (name, parameters, result, timestamp)
- **AgentStep** model (thought, action, tool_call, observation)
- `validate_tool_calls()` — compare actual vs expected tool usage
- `validate_trajectory_length()` — ensure agent efficiency
- `validate_tool_order()` — verify tool call sequencing via LCS
- `validate_no_repeated_tools()` — detect redundant tool usage

### Synthetic Test Data Generation

- **Synthesizer** class — generate test cases from documents or descriptions using an LLM
- `from_documents()` — ground test cases in provided document corpus
- `from_description()` — generate from free-text system description
- `evolve()` — make existing test cases more challenging
- 6 evolution strategies: simple, reasoning, multi_context, conditional, adversarial, comparative
- Parallel batch generation with cost tracking

### Red Teaming / Adversarial Testing

- **RedTeamer** — automated vulnerability scanning for LLM applications
- 10 vulnerability types: prompt injection, jailbreak, PII leakage, harmful content, bias exploitation, context manipulation, instruction override, role escape, data extraction, encoding attack
- 8 attack enhancement strategies: direct, roleplay, leetspeak, ROT-13, base64, multi-turn, logic trap, authority
- 70+ built-in attack templates
- Heuristic + LLM-judge evaluation of attack success
- **VulnerabilityReport** with severity breakdown and summary

### Batch API Support

- **BatchEvaluator** — submit evaluation jobs via OpenAI batch API for 50% cost savings
- JSONL request building, file upload, polling, and result retrieval
- Automatic cost estimation at batch discount rate

### Streaming Evaluation

- **StreamingEvaluator** — evaluate LLM outputs as they stream token by token
- Configurable check intervals with sync and async check support
- Early stop conditions for immediate termination
- **StreamingCheckpoint** model for progressive evaluation status

### OpenTelemetry Tracing

- **Tracer** with `span()` context manager and `trace()` decorator
- Integrates with OpenTelemetry when available, falls back to local-only
- Record CheckResults as span events
- Nested span support, JSON export, sync and async function tracing

### Experiment Tracking

- **ExperimentTracker** with SQLite persistence
- `start_run()`, `log_results()`, `end_run()` lifecycle
- Run comparison with score/pass-rate/cost diffs and per-metric analysis
- `best_run()` by avg_score, pass_rate, cost, or custom metric
- Tag-based filtering and prompt version tracking

### YAML Declarative Evaluation Configs

- Define evaluations in YAML (promptfoo-style): providers, prompts, tests, assertions
- **YamlEvalRunner** — orchestrate provider x prompt x test matrix
- Jinja2 template rendering for prompt variables
- 30+ assertion types mapped to deterministic and LLM checks
- `checkllm yaml-run config.yaml` CLI command

### Interactive Web Dashboard

- `checkllm dashboard` — browser-based UI for exploring evaluation results
- Dark theme with modern design, no external JS/CSS dependencies
- Experiment run list with search/filter
- Run detail view with score distribution charts
- Side-by-side run comparison with delta badges

### New CLI Commands

- `checkllm dashboard` — launch interactive web dashboard
- `checkllm yaml-run` — run YAML-defined evaluations
- `checkllm redteam` — automated red teaming against an LLM
- `checkllm experiments` — view and compare experiment tracking data

### GitHub Action Template

- Ready-to-use `.github/workflows/checkllm-action.yml` template
- Deterministic + LLM evaluation with budget controls
- Automatic PR comment with results summary
- Artifact upload for full reports

### Stats

- 1,112 tests (up from 836)
- 87 public API symbols (up from 55)
- 74 source files, 16,350 LOC
- 84 test files, 13,031 LOC
- 24 LLM-as-judge metrics + 33 deterministic checks
- 14 CLI commands

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

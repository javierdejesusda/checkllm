# CheckLLM: Reproducible Agent-Trajectory Evaluation

A deterministic, judge-free metric for scoring agent tool-call trajectories -- with AUROC 0.93 against synthetic ground truth and ~1500x faster than DeepEval's `ToolCorrectnessMetric`.

[![PyPI](https://img.shields.io/pypi/v/checkllm)](https://pypi.org/project/checkllm/) [![Python](https://img.shields.io/pypi/pyversions/checkllm)](https://pypi.org/project/checkllm/) [![License](https://img.shields.io/pypi/l/checkllm)](https://github.com/javierdejesusda/checkllm/blob/main/LICENSE) [![CI](https://github.com/javierdejesusda/checkllm/actions/workflows/ci.yml/badge.svg)](https://github.com/javierdejesusda/checkllm/actions/workflows/ci.yml) [![arXiv](https://img.shields.io/badge/arXiv-XXXX.YYYYY-b31b1b.svg)](https://arxiv.org/abs/XXXX.YYYYY) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.PLACEHOLDER-blue)](https://doi.org/10.5281/zenodo.PLACEHOLDER) [![Benchmark](https://img.shields.io/badge/leaderboard-rank%201-brightgreen)](docs/benchmarks/competitor-comparison.md)

```bash
pip install checkllm
```

```python
from checkllm.metrics.trajectory_metric import TrajectoryMetric

# Expected plan vs. what the agent actually did
expected = ["search", "fetch", "parse", "respond"]
actual = ["search", "fetch", "parse", "fetch", "respond"]

metric = TrajectoryMetric(expected_trajectory=expected)
sub = metric.compute_subscores(actual)

print(f"ordering   {sub.ordering:.2f}")   # 0.80
print(f"loops      {sub.loops:.2f}")      # 1.00
print(f"coverage   {sub.coverage:.2f}")   # 1.00
print(f"unexpected {sub.unexpected:.2f}") # 1.00
print(f"overall    {sub.overall:.2f}")    # 0.92
```

No judge LLM. No API key. Bit-identical scores across runs. See [the 10-minute tutorial](docs/tutorials/evaluating-agents-in-10-minutes.md) for the full walkthrough.

## Why CheckLLM?

- **Deterministic** -- no judge LLM, no API cost, bit-identical scores across runs and machines.
- **Composite** -- 4-axis trajectory scoring (ordering, loops, coverage, unexpected), AUROC 0.93 [0.91, 0.94] on 500 trajectories.
- **OTel-compatible** -- ingest traces from any agent framework via OpenTelemetry GenAI semantic conventions.

Beyond trajectory evaluation, CheckLLM also ships the broader testing suite the project has always provided:

- **Zero learning curve** -- if you know pytest, you know checkllm. Just add a `check` parameter.
- **39 free deterministic checks** run instantly with zero API calls. No API key needed to start.
- **72 LLM-as-judge metrics** -- hallucination, faithfulness, trajectory, per-turn, dual-judge, and more.
- **151 red team vulnerability types** with 25 attack strategies -- the most comprehensive adversarial testing suite available.
- **17 compliance frameworks** -- OWASP LLM/API/Agentic Top 10, MITRE ATLAS, EU AI Act, ISO 42001, HIPAA, GDPR, and more.
- **Same checks everywhere** -- use them in tests, CI, and production guardrails.

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
    check.is_json(output)
    check.gleu(output, reference="Expected summary text.", threshold=0.5)
    check.chrf(output, reference="Expected summary text.", threshold=0.4)
    check.latency_check(start_time, end_time, max_ms=3000)
    check.cost_check(input_tokens=500, output_tokens=200, model="gpt-4o", max_cost=0.05)
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

### 5. YAML-based evaluation

```yaml
# checkllm.yaml
description: "Customer support chatbot evaluation"
judge:
  backend: openai
  model: gpt-4o

prompts:
  - "You are a helpful support agent. Answer: {{query}}"

tests:
  - vars:
      query: "How do I return an item?"
    assert:
      - type: contains
        value: "return policy"
      - type: relevance
        threshold: 0.8
      - type: no_pii
      - type: max_tokens
        value: 500

settings:
  budget: 5.0
```

```bash
checkllm eval-yaml checkllm.yaml
```

## How checkllm compares

> **Independent benchmark, not just feature counts.** On the public competitor leaderboard
> ([docs/benchmarks/competitor-comparison.md](docs/benchmarks/competitor-comparison.md))
> checkllm holds **rank 1 on every published row** against DeepEval and promptfoo:
> halubench/hallucination 0.783, ragtruth/hallucination 0.663,
> ragtruth/faithfulness 0.754, ragtruth/context_relevance 0.565, and
> truthfulqa/answer_relevancy 0.546 (ROC-AUC, gpt-4o-mini judge,
> 200 source rows per slice). Methodology is in
> [docs/benchmarks/methodology.md](docs/benchmarks/methodology.md);
> raw scores ship in `benchmarks/competitor_comparison/`.

### Feature comparison

| Feature | checkllm | DeepEval | Ragas | promptfoo |
|---------|----------|----------|-------|-----------|
| pytest native | Yes | Wrapper | No | No |
| Free deterministic checks | **39** | Limited | Limited | Yes |
| LLM-as-judge metrics | **72** | ~50 | ~40 | Custom |
| Red team vulnerability types | **151** | 40+ | 0 | 100+ |
| Attack strategies | **25** | 10+ | 0 | 30+ |
| Compliance frameworks | **17** | 3 | 0 | 10+ |
| Multi-provider judges | **15+ backends** | 13+ | ~6 | 50+ |
| Consensus judging | **7 strategies** | No | Dual-judge | No |
| Production guardrails | **Built-in** | No | No | API |
| Cost control & budgets | **Built-in** | No | No | Caching |
| Knowledge Graph synthesis | **Full pipeline** | No | Yes | No |
| Multilingual prompts | **20 languages** | No | Yes | No |
| Prompt optimization | **4 algorithms** | 4 | 2 | No |
| YAML config evaluation | **Yes** | No | No | Yes |
| Streaming evaluation | **Token-by-token** | No | No | No |
| Regression detection | **Statistical (p-values)** | No | No | No |
| DPO export | **Yes** | No | No | No |
| Telemetry / phoning home | **None** | PostHog + Sentry | None | Telemetry |
| Independence | **Fully independent** | YC-backed | YC-backed | OpenAI-owned |

## All metrics by category

### RAG Evaluation (14 metrics)
`hallucination` `faithfulness` `faithfulness_hhem` `context_relevance` `context_entity_recall` `contextual_precision` `contextual_recall` `answer_completeness` `groundedness` `nonllm_context_precision` `nonllm_context_recall` `quoted_spans_alignment` `nv_context_relevance` `nv_response_groundedness`

### General Quality (12 metrics)
`relevance` `coherence` `fluency` `consistency` `correctness` `factual_correctness` `sentiment` `toxicity` `bias` `summarization` `nv_answer_accuracy` `prompt_alignment`

### Completeness & Instruction Following (5 metrics)
`response_completeness` `instruction_following` `instruction_completeness` `conversation_completeness` `topic_adherence`

### Agent & Tool Evaluation (12 metrics)
`task_completion` `tool_accuracy` `tool_call_f1` `plan_adherence` `plan_quality` `step_efficiency` `knowledge_retention` `goal_accuracy` `trajectory_goal_success` `trajectory_tool_sequence` `trajectory_step_count` `trajectory_tool_args_match`

### Per-Turn Conversation (3 metrics)
`turn_relevancy` `turn_faithfulness` `turn_coherence`

### Multimodal (6 metrics)
`image_relevance` `image_helpfulness` `image_coherence` `text_to_image` `image_editing` `image_reference`

### Structured Output (4 metrics)
`code_correctness` `sql_equivalence` `comparative_quality` `datacompy_score`

### Role & Safety (3 metrics)
`role_adherence` `role_violation` `non_advice`

### MCP & Tool-Specific (3 metrics)
`mcp_use` `mcp_task_completion` `multi_turn_mcp_use`

### Specialized (3 metrics)
`g_eval` `noise_sensitivity` `rubric`

### Deterministic Checks (39, zero API cost)
`contains` `not_contains` `starts_with` `ends_with` `regex` `exact_match` `exact_match_strict` `min_tokens` `max_tokens` `min_words` `max_words` `min_chars` `max_chars` `min_sentences` `max_sentences` `is_json` `json_schema` `is_xml` `is_yaml` `is_html` `no_pii` `language` `readability` `similarity` `bleu` `rouge_l` `meteor` `gleu` `chrf` `latency_check` `cost_check` `string_distance` `perplexity` `is_valid_python` `is_url` `has_url` `word_count` `char_count` `sentence_count`

## Red teaming & adversarial testing

```python
from checkllm.redteam import RedTeamer, VulnerabilityType
from checkllm.redteam_strategies import StrategyType

red = RedTeamer()
report = await red.scan(
    target=my_llm_function,
    vulnerability_types=[
        VulnerabilityType.PROMPT_INJECTION,
        VulnerabilityType.JAILBREAK,
        VulnerabilityType.PII_LEAKAGE,
        VulnerabilityType.DATA_EXFILTRATION,
    ],
    strategies=[StrategyType.BASE64, StrategyType.CRESCENDO, StrategyType.PERSONA],
    attacks_per_type=5,
)
print(report.summary())
print(report.risk_summary())  # CVSS severity breakdown
```

**151 vulnerability types** across 12 categories: prompt injection, jailbreak, PII leakage, harmful content, encoding attacks, privilege escalation, agentic AI attacks, brand & reputation, industry compliance, and more.

**25 attack strategies**: BASE64, ROT13, HEX, LEETSPEAK, MORSE, HOMOGLYPH, CRESCENDO (multi-turn escalation), JAILBREAK_TREE, JAILBREAK_META, JAILBREAK_COMPOSITE, BEST_OF_N, PERSONA, HYPOTHETICAL, ROLEPLAY, LAYER (composable chaining), and more.

### Coding agent security

```python
from checkllm.redteam_coding_agents import CodingAgentScanner

scanner = CodingAgentScanner(judge=judge)
report = await scanner.scan(target=my_coding_agent)
# Tests: repo prompt injection, sandbox escape, secret leakage, verifier sabotage
```

## Compliance frameworks

```python
from checkllm.compliance_frameworks import ComplianceScanner, ComplianceFramework

scanner = ComplianceScanner(judge=judge)
report = await scanner.scan(
    target=my_llm,
    frameworks=[
        ComplianceFramework.OWASP_LLM_TOP10,
        ComplianceFramework.OWASP_AGENTIC_TOP10,
        ComplianceFramework.EU_AI_ACT,
        ComplianceFramework.HIPAA,
    ],
)
print(report.summary())
```

**17 frameworks**: OWASP LLM Top 10, OWASP API Top 10, OWASP Agentic Top 10, MITRE ATLAS, EU AI Act, ISO 42001, NIST AI RMF, NIST CSF, HIPAA, GDPR, PCI-DSS, SOC2, ISO 27001, COPPA, FERPA, CCPA, DoD AI Ethics.

## Knowledge Graph test generation

```python
from checkllm.knowledge_graph import KGTestGenerator, EntityExtractor, SimilarityBuilder

gen = KGTestGenerator(judge=judge)
samples = await gen.generate(
    documents=["doc1 text...", "doc2 text..."],
    num_samples=50,
    synthesizers={"single_hop": 0.4, "multi_hop_abstract": 0.3, "multi_hop_specific": 0.3},
    personas=5,
)
cases = gen.to_cases(samples)
```

Build a knowledge graph from your documents, then generate diverse test cases with single-hop, multi-hop abstract, and multi-hop specific queries. Supports persona variation, query styles (web search, misspelled, conversational), and configurable complexity.

## Multilingual evaluation

```python
from checkllm.multilingual import PromptAdapter, detect_language

adapter = PromptAdapter(judge=judge)
translated = await adapter.adapt(template=my_prompt, target_language="ja")
adapter.save_translations("translations/ja.json")

lang = detect_language("Esto es un texto en espanol.")  # "es"
```

Supports 20+ languages with automatic prompt adaptation. Language detection uses Unicode character-range analysis with LLM fallback.

## Prompt optimization

```python
from checkllm.optimize import create_optimizer

optimizer = create_optimizer("miprov2", judge=judge)  # or "genetic", "copro", "simba"
result = await optimizer.optimize(
    prompt="Summarize this document.",
    test_cases=my_test_cases,
    metric_fn=my_metric,
    num_candidates=10,
)
print(f"Improved from {result.initial_score:.2f} to {result.best_score:.2f}")
```

Four optimization algorithms: Genetic (evolutionary), MIPROv2 (instruction + demonstration), COPRO (failure-driven iterative), SIMBA (similarity-based adaptation).

## Multi-provider judges

```python
from checkllm import create_judge

judge = create_judge("openai", model="gpt-4o")
judge = create_judge("anthropic", model="claude-sonnet-4-6")
judge = create_judge("gemini", model="gemini-2.0-flash")
judge = create_judge("ollama", model="llama3.1")       # Free, local
judge = create_judge("litellm", model="any-model")     # 100+ models
judge = create_judge("deepseek")
judge = create_judge("groq")
judge = create_judge("fireworks")
```

Auto-detection: set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or have Ollama running -- checkllm picks the best judge automatically.

## Consensus judging

```python
from checkllm import ConsensusJudge

judges = [("gpt4", gpt4_judge), ("claude", claude_judge), ("gemini", gemini_judge)]
consensus = ConsensusJudge(judges, strategy="majority")  # or mean, median, unanimous, min, max, weighted
```

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
judge_backend = "auto"
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
| `checkllm eval-yaml` | Run YAML-based evaluation |
| `checkllm estimate` | Estimate costs before running |
| `checkllm watch` | Re-run on file changes |
| `checkllm report` | Generate HTML report |
| `checkllm snapshot` | Save baseline for regression detection |
| `checkllm diff` | Compare snapshots |
| `checkllm history` | View run history and trends |
| `checkllm list-metrics` | Show all available checks and metrics |
| `checkllm cache` | Manage judge response cache |
| `checkllm dashboard` | Launch web dashboard |

## Framework integrations

```python
# LangChain
from checkllm.integrations.langchain import CheckllmCallbackHandler
chain.invoke(input, config={"callbacks": [CheckllmCallbackHandler(checks=["no_pii"])]})

# CrewAI
from checkllm.integrations.crewai import CheckllmCrewCallback

# OpenAI Agents SDK
from checkllm.integrations.openai_agents import CheckllmRunHandler

# Claude Agent SDK
from checkllm.integrations.claude_agents import CheckllmAgentHandler

# PydanticAI
from checkllm.integrations.pydantic_ai import CheckllmResultValidator

# LlamaIndex
from checkllm.integrations.llama_index import CheckllmCallbackHandler
```

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

## Citing CheckLLM

If you use CheckLLM's trajectory metric in academic work, please cite the companion paper:

```bibtex
@article{dejesus2026checkllm,
  title        = {{CheckLLM}: Reproducible Agent-Trajectory Evaluation at Scale},
  author       = {de Jesus, Javier},
  journal      = {arXiv preprint arXiv:XXXX.YYYYY},
  year         = {2026},
  doi          = {10.5281/zenodo.PLACEHOLDER},
  url          = {https://github.com/javierdejesusda/checkllm}
}
```

The arXiv ID and Zenodo DOI placeholders will be replaced once the paper-v1 tag is cut. See [`CITATION.cff`](CITATION.cff) for the canonical citation metadata.

## License

MIT

# Advanced Patterns

## Consensus Judging Strategies

A single judge can be unreliable. checkllm supports 7 consensus strategies that
combine multiple judge backends into a single result.

### When to use each strategy

| Strategy | When to Use |
|----------|-------------|
| `majority_vote` | Pass/fail decisions; use an odd number of judges |
| `average` | Continuous scores; balanced ensemble |
| `weighted_average` | Judges with different reliability or cost profiles |
| `unanimous` | Safety-critical — all judges must agree |
| `any_pass` | High-recall — pass if at least one judge approves |
| `trimmed_mean` | Automatically removes outlier scores |
| `borda_count` | Rank-based consensus for comparing multiple outputs |

### Setup

```python
from checkllm import ConsensusJudge, OpenAIJudge, AnthropicJudge, GeminiJudge

judges = [
    OpenAIJudge(model="gpt-4o"),
    AnthropicJudge(model="claude-3-5-sonnet-20241022"),
    GeminiJudge(model="gemini-1.5-pro"),
]
```

### majority_vote

```python
consensus = ConsensusJudge(judges=judges, strategy="majority_vote")
result = consensus.score(metric="hallucination", output="Paris is in France.")
# Passes if >= 2/3 judges score >= threshold
```

### weighted_average

```python
consensus = ConsensusJudge(
    judges=judges,
    strategy="weighted_average",
    weights=[0.5, 0.35, 0.15],   # OpenAI carries 50% of the weight
)
```

### unanimous (safety-critical)

```python
consensus = ConsensusJudge(judges=judges, strategy="unanimous", threshold=0.9)
# ALL judges must score >= 0.9; a single disagreement fails the test
```

### any_pass (high-recall)

```python
consensus = ConsensusJudge(judges=judges, strategy="any_pass", threshold=0.7)
# Passes if at least one judge scores >= 0.7
```

### trimmed_mean (outlier-resistant)

```python
consensus = ConsensusJudge(
    judges=judges,
    strategy="trimmed_mean",
    trim_percent=0.2,   # Drop top and bottom 20% of scores
)
```

### borda_count (ranking multiple outputs)

```python
from checkllm import ConsensusJudge, OpenAIJudge

consensus = ConsensusJudge(
    judges=[OpenAIJudge(model="gpt-4o"), OpenAIJudge(model="gpt-4o-mini")],
    strategy="borda_count",
)

outputs = ["Summary A", "Summary B", "Summary C"]
ranked = consensus.rank(metric="summarization", outputs=outputs)
print(ranked[0])   # Highest-ranked output
```

---

## Prompt Optimization Algorithms

checkllm includes 4 optimization algorithms that improve prompts automatically.

### When to use each

| Algorithm | Best For | Cost |
|-----------|----------|------|
| `genetic` | Large search spaces, no labelled data needed | Low |
| `copro` | Few-shot example selection | Medium |
| `mipro` | Joint instruction + few-shot optimisation | High |
| `simba` | Multi-step chain / agent prompt optimisation | High |

### Genetic Algorithm

```python
from checkllm import optimize, GeneticOptimizer

@optimize(
    optimizer=GeneticOptimizer(
        population_size=20,
        generations=10,
        mutation_rate=0.2,
    ),
    metric="answer_relevance",
    threshold=0.85,
)
def summarize(text: str) -> str:
    return f"Summarize this: {text}"

best_prompt = summarize.optimize(training_data=my_examples)
print(best_prompt)
```

### COPRO (few-shot selection)

```python
from checkllm import optimize, COPROOptimizer

@optimize(
    optimizer=COPROOptimizer(
        breadth=10,             # Candidate prompts per round
        depth=3,                # Refinement rounds
        examples=my_examples,   # list[{"input": ..., "output": ...}]
    ),
    metric="answer_correctness",
)
def answer(question: str) -> str:
    return f"Answer: {question}"
```

### MIPROv2 (instruction + few-shot)

```python
from checkllm import optimize, MIPROOptimizer

@optimize(
    optimizer=MIPROOptimizer(
        num_candidates=10,
        num_trials=20,
        max_bootstrapped_demos=4,
    ),
    metric="hallucination",
)
def rag_answer(question: str, context: list[str]) -> str:
    ...
```

### SIMBA (multi-step chains)

```python
from checkllm import optimize, SIMBAOptimizer

@optimize(
    optimizer=SIMBAOptimizer(
        num_candidates=16,
        max_steps=4,
    ),
    metric="trajectory_goal_completion",
)
def multi_step_agent(task: str) -> str:
    ...
```

---

## Cost Workflows

### Hard budget caps

```toml
[tool.checkllm]
budget = 5.0   # Stop after $5.00 spent

[tool.checkllm.profiles.ci]
budget = 10.0
```

Or per pytest run:

```bash
pytest tests/ --checkllm-budget 5.0
```

### Estimate before running

```bash
checkllm estimate tests/

# Estimated cost: $2.34  (47 tests x avg $0.05/test)
# Breakdown:
#   hallucination x 20      $1.20
#   answer_relevance x 15   $0.75
#   summarization x 12      $0.39
```

### Response caching

```toml
[tool.checkllm]
cache_enabled = true
cache_dir     = ".checkllm/cache"
cache_ttl     = 86400   # 24 h
```

```bash
checkllm cache clear                  # Wipe entire cache
checkllm cache clear --older-than 7d  # Remove stale entries only
```

### Use cheaper models in development

```toml
[tool.checkllm.profiles.dev]
judge_model = "gpt-4o-mini"          # ~15x cheaper than gpt-4o
budget      = 1.0

[tool.checkllm.profiles.prod]
judge_model = "gpt-4o-2024-11-20"   # Pinned version
budget      = 50.0
```

```bash
CHECKLLM_PROFILE=dev  pytest tests/ -v   # development
CHECKLLM_PROFILE=prod pytest tests/ -v   # production
```

### Cost report in CI

```yaml
- name: Run evaluations
  run: checkllm ci --budget 10.0 --fail-on-regression
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

- name: Cost summary
  if: always()
  run: checkllm report --format markdown >> $GITHUB_STEP_SUMMARY
```

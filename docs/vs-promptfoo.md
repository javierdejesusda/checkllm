# CheckLLM vs promptfoo

## TL;DR

promptfoo (\citep{promptfoo2024}) is a YAML-driven CLI for prompt and
model comparison, with a web UI and a strong red-teaming offering.
CheckLLM is a Python library focused on **agent-trajectory** evaluation.
The two solve different problems: promptfoo is about comparing prompts
and models across a test grid; CheckLLM is about scoring multi-step
agent traces.

## When to use promptfoo

- YAML configs are easy to commit, diff, and review — non-engineers can
  edit them without learning a Python API.
- Built-in web UI for browsing eval runs and side-by-side diffs across
  models or prompt variants.
- Red-team testing is a first-class feature, with a curated plugin set
  for jailbreaks, prompt injection, and policy violations.
- Strong matrix-evaluation ergonomics: try N prompts × M models × K
  test cases with a single command.
- Hosted SaaS option (promptfoo Cloud / Enterprise) for teams that want
  shared dashboards without self-hosting.

## When to use CheckLLM

- Deterministic 4-axis trajectory scoring (selection, parameters, order,
  efficiency) at **0.034 ms/trajectory**, with no judge-LLM cost.
- **AUROC = 0.938 [0.909, 0.965]** vs synthetic ground truth on 150
  trajectories — measured against the closest agent-eval competitor
  (DeepEval `ToolCorrectnessMetric`, AUROC 0.850).
- First OSS framework with native MCP (Model Context Protocol) metric
  coverage.
- OTel-GenAI ingestion: traces from LangChain, LlamaIndex, CrewAI, and
  PydanticAI all map into one evaluable trajectory format.
- Composite metric weights are Pareto-optimal vs a 1875-cell ablation grid.

## Where they don't overlap

promptfoo's unit of evaluation is a **(prompt, model, test case)**
triple producing a single output to be graded by an assertion or a
judge. It is excellent at "which of these 6 prompts works best on
GPT-4 vs Claude vs Llama?"

CheckLLM's unit of evaluation is a **trajectory** — an ordered sequence
of tool calls, observations, and reasoning steps produced by an agent.
The metrics ask whether the agent took a sensible path, not whether a
prompt produced a good string.

If you are choosing between prompts, use promptfoo. If you are scoring
a tool-using agent's behaviour over time, use CheckLLM. If you are
doing red-teaming on raw prompt I/O, promptfoo is more mature today.

## If you're already using promptfoo

Drive both from the same CI job. promptfoo handles the prompt grid;
CheckLLM scores the agent traces collected during those runs.

```yaml
# promptfoo.yaml
providers: [openai:gpt-4o, anthropic:claude-3-5-sonnet]
tests:
  - vars: { input: "book a flight to Tokyo" }
    assert: [{ type: python, value: file://check_trajectory.py }]
```

```python
# check_trajectory.py
from checkllm.agents import TrajectoryMetric
def get_assert(output, ctx):
    return TrajectoryMetric().score(ctx["trace"]) >= 0.8
```

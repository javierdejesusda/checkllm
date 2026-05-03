<!-- TODO: replace `<placeholder>` in the Colab badge URL with the canonical
     org/repo path once the companion notebook is generated and committed. -->
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<placeholder>/checkllm/blob/main/docs/tutorials/evaluating-agents-in-10-minutes.ipynb)

# Evaluating Agents in 10 Minutes

This tutorial takes you from `pip install` to "I just found a real bug in my agent's trajectory" in roughly ten minutes. We will build a small agent trajectory by hand, score it with CheckLLM's deterministic `TrajectoryMetric`, mutate it three different ways to see how each sub-score reacts, and finish by ingesting an OpenTelemetry trace where the metric pinpoints a 3-step loop.

Everything in this tutorial is runnable, deterministic, and free. No judge LLM, no API key, no hidden randomness. The numbers printed by the code are the numbers in the prose -- if your run produces different output, that is a bug in CheckLLM, not a flaky test.

## Setup

CheckLLM is a single `pip install` away. Use whatever environment manager you like; a fresh virtualenv is fine.

```bash
pip install checkllm
```

That is the entire setup. The `TrajectoryMetric` is part of the core install, with no optional extras required. Nothing else in this tutorial needs network access.

## Build a 5-step trajectory by hand

Let's pretend we are evaluating an agent that answers research questions. The expected plan is straightforward: search for sources, fetch the most promising URL, parse the content, and respond. Our hypothetical agent did almost the right thing, but fetched twice -- once for the main article, once for its references section.

```python
from checkllm.agents import AgentStep, AgentTestCase, ToolCall

def step(name, **params):
    return AgentStep(action="call_tool", tool_call=ToolCall(name=name, parameters=params))

baseline = AgentTestCase(
    query="What causes climate change?",
    steps=[
        step("search", query="climate change causes"),
        step("fetch",  url="https://example.org/climate"),
        step("parse",  selector="main"),
        step("fetch",  url="https://example.org/climate#refs"),
        step("respond", text="Greenhouse gases ..."),
    ],
)
```

`AgentTestCase` is the canonical container for an agent run in CheckLLM. Each `AgentStep` records the action, the tool call, and (in production) the observation. Here we keep things minimal: just the tool name and parameters, which is all the trajectory metric inspects.

## Score it with `TrajectoryMetric`

The metric needs an `expected_trajectory` -- the ordered list of tool names that should have been called -- and a list of `ToolCall` objects to score. The four sub-scores are returned as a single `TrajectorySubScores` object.

```python
from checkllm.metrics.trajectory_metric import TrajectoryMetric

expected = ["search", "fetch", "parse", "respond"]
metric = TrajectoryMetric(expected_trajectory=expected)
sub = metric.compute_subscores(baseline.tool_calls)

print(sub.as_dict())
# {'ordering': 0.8, 'loops': 1.0, 'coverage': 1.0, 'unexpected': 1.0, 'overall': 0.92}
```

The `overall` score is a weighted average of the four sub-scores, with default weights `ordering=0.4`, `loops=0.2`, `coverage=0.25`, `unexpected=0.15`. Those defaults were chosen by an ablation sweep against synthetic ground truth and sit within 2.4% of the empirically best configuration on a 1875-cell grid.

A 0.92 overall on the baseline is exactly what we want: the agent did *almost* the right thing. The `ordering` sub-score is 0.80 because the actual sequence has one extra `fetch` call that displaces the rest of the plan; the other three sub-scores are perfect because no tool was looped, every expected tool appeared, and there were no unexpected tools. We will now break this baseline three different ways and watch each sub-score react.

## Mutation 1: inject a 3-step loop

Real agents loop. They retry the same tool with slightly different arguments, get stuck in retrieval rabbit-holes, or fail to converge. The `loops` sub-score is the metric's defense against this failure mode.

```python
loop_case = AgentTestCase(
    query=baseline.query,
    steps=[
        step("search", q="x"),
        step("fetch",  url="a"),
        step("fetch",  url="b"),
        step("fetch",  url="c"),
        step("parse",  s="main"),
        step("respond", t="..."),
    ],
)

print(TrajectoryMetric(expected, loop_threshold=2).compute_subscores(loop_case.tool_calls).as_dict())
# {'ordering': 0.6667, 'loops': 0.6667, 'coverage': 1.0, 'unexpected': 1.0, 'overall': 0.8}
```

The agent now calls `fetch` three times in a row. With the default `loop_threshold=2`, runs of two consecutive identical tools are tolerated, and only the third call counts as overshoot. The `loops` sub-score drops to 0.67. Notice that `ordering` also drops, because the longer sequence pushes the Levenshtein distance up; loops hurt twice.

The threshold is adjustable. If your agent legitimately hits the same tool three times in a row -- many retrieval pipelines do exactly that to refine results -- you can raise the threshold and re-score:

```python
print(TrajectoryMetric(expected, loop_threshold=3).compute_subscores(loop_case.tool_calls).as_dict())
# {'ordering': 0.6667, 'loops': 1.0, 'coverage': 1.0, 'unexpected': 1.0, 'overall': 0.8667}
```

`loops` jumps back to 1.0 and the overall climbs to 0.87. Same trajectory, same metric, different policy: the threshold lets you encode what counts as a benign repetition for *your* agent without changing any code.

## Mutation 2: drop a tool

What if the agent skipped a step entirely? The `coverage` sub-score reports the fraction of expected tools that actually appeared, which is the cleanest signal for "the agent took a shortcut".

```python
drop_case = AgentTestCase(
    query=baseline.query,
    steps=[step("search", q="x"), step("fetch", url="a"), step("respond", t="...")],
)

print(TrajectoryMetric(expected).compute_subscores(drop_case.tool_calls).as_dict())
# {'ordering': 0.75, 'loops': 1.0, 'coverage': 0.75, 'unexpected': 1.0, 'overall': 0.8375}
```

`parse` was never called, so `coverage` falls to 0.75 (3 of 4 expected tools appeared). `ordering` mirrors the drop because Levenshtein still penalises the missing element. The overall lands at 0.84 -- still passing the default 0.80 threshold, but visibly worse than the baseline. In a CI run, a regression like this would show up as a clear, monotonic drop, not a noisy LLM-judge wobble.

## Mutation 3: inject an unexpected tool

The fourth sub-score is the safety net for unauthorized tool use. If your agent calls a tool you never expected -- say, an exfiltration endpoint, a different LLM provider, or a deprecated API -- the `unexpected` sub-score flags it.

```python
unx_case = AgentTestCase(
    query=baseline.query,
    steps=[
        step("search", q="x"),
        step("fetch",  url="a"),
        step("parse",  s="main"),
        step("exfiltrate", payload="secrets"),
        step("respond", t="..."),
    ],
)

print(TrajectoryMetric(expected).compute_subscores(unx_case.tool_calls).as_dict())
# {'ordering': 0.8, 'loops': 1.0, 'coverage': 1.0, 'unexpected': 0.8, 'overall': 0.89}
```

Every expected tool was still called in roughly the right order, so `coverage` and `loops` are perfect and `ordering` only takes a small hit. But the agent also reached for an `exfiltrate` tool that has no business being there. `unexpected` reports 0.80 -- exactly `1 - 1/5`, the fraction of unexpected calls in the actual sequence -- and the overall drops accordingly. The signal is small but distinct, which is what you want from a defense-in-depth metric: it complements your red-teaming and guardrails rather than replacing them.

## The finale: score a real OpenTelemetry trace

Building trajectories by hand is fine for tutorials, but production agents emit OpenTelemetry traces. CheckLLM ingests OTel JSONL exports directly via `from_trace_jsonl`, which uses the GenAI semantic conventions to identify tool spans automatically. The repository ships a small fixture you can use right now.

```python
from checkllm.agents import AgentTestCase
from checkllm.metrics.trajectory_metric import TrajectoryMetric

case = AgentTestCase.from_trace_jsonl(
    "docs/tutorials/fixtures/example_trace.jsonl",
    query="What causes climate change?",
)
print([c.name for c in case.tool_calls])
# ['search', 'search', 'search', 'respond']
```

The fixture contains five OTel spans: one LLM `chat` span and four tool spans. CheckLLM correctly classifies the LLM span and skips it, returning only the tool calls. The agent here got into trouble: it called `search` three times in a row with slightly different queries before giving up and responding.

Now we score it against the same expected plan we have used throughout the tutorial.

```python
expected = ["search", "fetch", "parse", "respond"]
result = TrajectoryMetric(expected_trajectory=expected, loop_threshold=2).evaluate(case.tool_calls)

print(result.score)      # 0.6083
print(result.passed)     # False
print(result.reasoning)
# Overall 0.61 (threshold 0.80). Ordering: 0.50 ... Loops: 0.67 (max consecutive run=3 of 'search', threshold=2). ...
```

The `reasoning` field on the `CheckResult` is a deterministic, human-readable breakdown. It tells you the maximum consecutive run length, which tool was looping, and the threshold that was tripped. No judge LLM, no API call, no hidden state -- just arithmetic on the tool-name sequence.

**You just found a 3-step loop in your agent.**

That finding is reproducible: rerun the same script on the same fixture on any machine and you will get bit-identical sub-scores, the same reasoning string, and the same `passed=False` verdict. That is the property that makes CheckLLM's trajectory metric useful in CI: regressions are real regressions, not judge variance.

## Where to go next

The full benchmark backing the AUROC 0.93 number lives in `benchmarks/paper/results/02_metric_vs_truth/summary.json`, with a 1875-cell ablation in `03_trajectory_ablation/` and a head-to-head against DeepEval's `ToolCorrectnessMetric` in `04_vs_deepeval/`. The companion paper (arXiv ID pending) writes up the methodology and statistical claims in detail.

If you want to push further inside CheckLLM itself, the natural next steps are: tune the four sub-score weights against your own labelled trajectories using `TrajectoryMetricConfig`; ingest your real production traces via OTel exporters; and wire the metric into your pytest suite as a regression gate. The metric is deterministic, cheap, and contractually simple -- exactly the properties you want for an agent-eval signal you can rely on every commit.

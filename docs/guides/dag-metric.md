# DAG Metric

CheckLLM's `DAGMetric` evaluates an LLM output by running it through a
directed acyclic graph of judge nodes. Every node is a single
LLM-as-judge call, and edges choose the next node based on the current
node's verdict. This is the tool of choice when a single rubric is not
enough — when downstream checks depend on upstream outcomes.

## When to use the DAG metric (vs simple metrics)

Reach for a plain metric (`FaithfulnessMetric`, `CorrectnessMetric`,
`GEval`, ...) when you want a single scalar answer to a single
question. Reach for `DAGMetric` when any of the following is true:

- Different checks need to run based on what the previous check found
  (for example, only run "remediation analysis" when correctness is
  low).
- Later nodes should see earlier results (parent score / reasoning).
- You want a single `CheckResult` that reflects a multi-step reasoning
  policy, not an unweighted average of independent metrics.
- You want to short-circuit expensive judges when cheaper gates fail.

## Quickstart

```python
import asyncio

from checkllm.judge import OpenAIJudge
from checkllm.metrics.dag import DAGMetric, DAGNode


async def main() -> None:
    judge = OpenAIJudge(model="gpt-4o-mini")
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="safety",
                prompt_template="Is this output safe? {output}",
                threshold=0.8,
                children_on_pass=["quality"],
            ),
            DAGNode(
                name="quality",
                prompt_template="Rate quality 0-1: {output}",
                threshold=0.7,
                is_leaf=True,
            ),
        ],
        root="safety",
    )
    result = await dag.evaluate(output="Hello, world!")
    print(result.score, result.passed)


asyncio.run(main())
```

The metric validates the graph at construction time. You'll get a
`ValueError` immediately if the root is missing, a child name is
unknown, a threshold is outside `[0, 1]`, or the graph has a cycle.

## Conditional branching

There are two ways to branch from a node.

**Pass / fail** — the simplest. Set `threshold`, `children_on_pass`,
`children_on_fail`. Children referenced in `children_on_pass` run when
`score >= threshold`; otherwise `children_on_fail` run.

**Score ranges** — use `children_on_score_ranges` when you need more
than a binary split. Keys are `(lo, hi)` tuples (inclusive on `lo`,
exclusive on `hi`, except the bucket containing `1.0` includes the
upper bound). When both `children_on_pass`/`children_on_fail` and
`children_on_score_ranges` are set, the ranges win.

```python
DAGNode(
    name="correctness",
    prompt_template="Is this correct? {output}",
    children_on_score_ranges={
        (0.0, 0.5): ["remediate"],
        (0.5, 0.8): ["improve"],
        (0.8, 1.0): ["polish"],
    },
)
```

## Context passing between nodes

Prompts support four placeholders:

| Placeholder | Resolves to |
| --- | --- |
| `{output}` | The string passed to `evaluate()` |
| `{parent_score}` | Previous node's score (formatted to four decimals) |
| `{parent_reasoning}` | Previous node's free-text reasoning |
| `{context.<key>}` | A lookup in the `context` dict |
| `{context}` | The whole context dict as a JSON blob |

```python
result = await dag.evaluate(
    output=llm_output,
    context={"spec": "Write a pure add(a, b)", "language": "python"},
)
```

## Leaf nodes and final verdicts

By default, the DAG's score is the weighted average of every visited
node. Set `is_leaf=True` on any node to mark it as a terminal verdict
— when the traversal reaches a leaf, that node's score becomes the
final DAG score on its own (the weighted average is ignored). This is
useful for flows where a specialised final judge should have the last
word.

Sibling children of a non-leaf node run concurrently via
`asyncio.gather`, so wide graphs stay fast. Use
`DAGMetric.aevaluate_batch(outputs)` to evaluate many outputs in
parallel.

After any `evaluate()` call, inspect `dag.get_last_path()` for the
ordered list of `DAGEvalResult` nodes that were visited. The same
trace is serialised into `result.reasoning` under a `[dag-trace] {...}`
prefix, so it survives JSON logs and CI artefact dumps.

## Visualizing your graph (Mermaid)

Every `DAGMetric` can print itself as a Mermaid flowchart via
`dag.to_mermaid()`. Drop the result into a Markdown document or use it
to review complex graphs in code review.

```mermaid
flowchart TD
    safety["safety"]
    correctness["correctness"]
    remediation_analysis["remediation_analysis (leaf)"]
    style["style (leaf)"]
    safety_reject["safety_reject (leaf)"]
    safety -- pass --> correctness
    safety -- fail --> safety_reject
    correctness -- "[0.00,0.50)" --> remediation_analysis
    correctness -- "[0.50,1.00)" --> style
    classDef leaf fill:#e0f7e9,stroke:#2e7d32;
    class remediation_analysis,style,safety_reject leaf;
```

## Comparison vs DeepEval DAG

| Feature | DeepEval `DAGMetric` | CheckLLM `DAGMetric` |
| --- | --- | --- |
| Judge nodes with thresholds | Yes | Yes |
| Pass/fail branching | Yes | Yes |
| Score-range branching | Limited (binary verdict nodes) | Native `children_on_score_ranges` |
| Parent score / reasoning in child prompt | Manual plumbing | `{parent_score}` / `{parent_reasoning}` placeholders |
| User-supplied context | Manual | `context={...}` kwarg + `{context.*}` placeholders |
| Parallel sibling execution | Sequential | `asyncio.gather` under the hood |
| Construction-time validation | Runtime errors | Immediate `ValueError` (cycles, bad refs, bad thresholds) |
| Graph visualization | Not built in | `to_mermaid()` |
| Batch evaluation | Loop yourself | `aevaluate_batch(outputs)` |
| Path trace on result | Implicit | `get_last_path()` + `[dag-trace]` JSON prefix in reasoning |

The intent is full parity on the DAG mental model, with better
ergonomics for the things teams actually do in production: inject
rubric-level context, visualise the graph, and batch evaluate across a
dataset.

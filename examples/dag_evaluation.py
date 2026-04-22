"""End-to-end example: multi-step code-generation evaluation as a DAG.

The pipeline evaluates a snippet of code in three stages:

1. ``safety`` — reject anything with obvious unsafe patterns.
2. ``correctness`` — judge whether the code meets its spec.
3. ``style`` — judge readability and idiom use.

The ``correctness`` node branches with score ranges: a low score routes
to a dedicated ``remediation_analysis`` node that diagnoses why the
code is broken, while a healthy score continues into ``style`` as the
final leaf verdict.

Run with ``python examples/dag_evaluation.py``. A ``FakeJudge`` is used
so the example is hermetic — swap in :class:`checkllm.judge.OpenAIJudge`
(or any :class:`checkllm.judge.JudgeBackend`) for real evaluation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from checkllm.metrics.dag import DAGMetric, DAGNode
from checkllm.models import JudgeResponse


class FakeJudge:
    """Scripted judge used to make the example deterministic."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    async def evaluate(self, prompt: str, **_: Any) -> JudgeResponse:
        for name, score in self.scores.items():
            if f"[{name}]" in prompt:
                return JudgeResponse(
                    score=score,
                    reasoning=f"judged {name}={score:.2f}",
                    raw_output=None,
                )
        return JudgeResponse(score=0.5, reasoning="default", raw_output=None)


def build_dag(judge: FakeJudge) -> DAGMetric:
    """Build a three-step code-evaluation DAG with score-range branching."""
    return DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="safety",
                prompt_template=(
                    "[safety] Is this code safe (no shell injection, no eval)?\n" "Code:\n{output}"
                ),
                threshold=0.8,
                children_on_pass=["correctness"],
                children_on_fail=["safety_reject"],
                weight=2.0,
            ),
            DAGNode(
                name="correctness",
                prompt_template=(
                    "[correctness] Given spec={context.spec}, does the code "
                    "meet the requirements?\nCode:\n{output}"
                ),
                children_on_score_ranges={
                    (0.0, 0.5): ["remediation_analysis"],
                    (0.5, 1.0): ["style"],
                },
            ),
            DAGNode(
                name="remediation_analysis",
                prompt_template=(
                    "[remediation_analysis] The correctness judge gave "
                    "{parent_score}. Diagnose why. Previous reasoning: "
                    "{parent_reasoning}\nCode:\n{output}"
                ),
                is_leaf=True,
            ),
            DAGNode(
                name="style",
                prompt_template=(
                    "[style] Rate the readability and idiomatic style of " "this code:\n{output}"
                ),
                threshold=0.7,
                is_leaf=True,
            ),
            DAGNode(
                name="safety_reject",
                prompt_template=(
                    "[safety_reject] Summarise the safety issue for logs.\n" "Code:\n{output}"
                ),
                is_leaf=True,
            ),
        ],
        root="safety",
        threshold=0.6,
    )


async def main() -> None:
    spec = "Write a pure function add(a, b) -> int"
    code_ok = "def add(a, b):\n    return a + b\n"
    code_buggy = "def add(a, b):\n    return a - b\n"

    happy_judge = FakeJudge({"safety": 0.95, "correctness": 0.85, "style": 0.9})
    dag = build_dag(happy_judge)
    happy = await dag.evaluate(output=code_ok, context={"spec": spec})
    print("Happy path:")
    print(f"  score={happy.score:.2f} passed={happy.passed}")
    print(f"  trace={[r.node_name for r in dag.get_last_path()]}")

    buggy_judge = FakeJudge({"safety": 0.9, "correctness": 0.3, "remediation_analysis": 0.4})
    dag = build_dag(buggy_judge)
    buggy = await dag.evaluate(output=code_buggy, context={"spec": spec})
    print("Buggy path:")
    print(f"  score={buggy.score:.2f} passed={buggy.passed}")
    print(f"  trace={[r.node_name for r in dag.get_last_path()]}")

    print("\nMermaid diagram:")
    print(dag.to_mermaid())


if __name__ == "__main__":
    asyncio.run(main())

# Expected output:
# Happy path:
#   score=0.90 passed=True
#   trace=['safety', 'correctness', 'style']
# Buggy path:
#   score=0.40 passed=False
#   trace=['safety', 'correctness', 'remediation_analysis']
#
# Mermaid diagram:
# flowchart TD
#     safety["safety"]
#     correctness["correctness"]
#     remediation_analysis["remediation_analysis (leaf)"]
#     style["style (leaf)"]
#     safety_reject["safety_reject (leaf)"]
#     safety -- pass --> correctness
#     safety -- fail --> safety_reject
#     correctness -- "[0.00,0.50)" --> remediation_analysis
#     correctness -- "[0.50,1.00)" --> style
#     classDef leaf fill:#e0f7e9,stroke:#2e7d32;
#     class remediation_analysis,style,safety_reject leaf;

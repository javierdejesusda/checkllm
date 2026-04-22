"""Tests for the enhanced DAG metric features.

Exercises score-range branching, context injection, leaf-node verdicts,
Mermaid export, graph validation, parallel child execution, and batch
evaluation. A deterministic stub judge keeps every test hermetic.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from checkllm.metrics.dag import DAGEvalResult, DAGMetric, DAGNode
from checkllm.models import JudgeResponse


class StubJudge:
    """Deterministic judge that returns scores based on node name.

    The node name is looked up in the prompt (every generated prompt
    includes the template, which includes the node name via the test
    fixtures). When the node name is not found, the default score is
    returned.
    """

    def __init__(
        self,
        scores: dict[str, float],
        default: float = 0.0,
        reasons: dict[str, str] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.scores = scores
        self.default = default
        self.reasons = reasons or {}
        self.delay = delay
        self.calls: list[str] = []
        self.call_times: list[float] = []

    async def evaluate(self, prompt: str, **_: Any) -> JudgeResponse:
        self.calls.append(prompt)
        self.call_times.append(time.perf_counter())
        if self.delay:
            await asyncio.sleep(self.delay)
        score = self.default
        reasoning = "stub"
        for key, value in self.scores.items():
            if f"[node:{key}]" in prompt:
                score = value
                reasoning = self.reasons.get(key, f"score for {key}")
                break
        return JudgeResponse(score=score, reasoning=reasoning, raw_output=None)


def _tagged(node: str, body: str = "Evaluate {output}") -> str:
    """Build a prompt template that embeds a unique node tag."""
    return f"[node:{node}] {body}"


async def test_score_range_low_bucket_routes_to_remediate():
    judge = StubJudge({"root": 0.2, "remediate": 0.9})
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="root",
                prompt_template=_tagged("root"),
                children_on_score_ranges={
                    (0.0, 0.5): ["remediate"],
                    (0.5, 0.8): ["improve"],
                    (0.8, 1.0): ["polish"],
                },
            ),
            DAGNode(name="remediate", prompt_template=_tagged("remediate")),
            DAGNode(name="improve", prompt_template=_tagged("improve")),
            DAGNode(name="polish", prompt_template=_tagged("polish")),
        ],
        root="root",
    )
    result = await dag.evaluate(output="hi")
    path_names = [r.node_name for r in dag.get_last_path()]
    assert path_names == ["root", "remediate"]
    assert result.metric_name == "dag"


async def test_score_range_mid_bucket_routes_to_improve():
    judge = StubJudge({"root": 0.65, "improve": 0.8})
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="root",
                prompt_template=_tagged("root"),
                children_on_score_ranges={
                    (0.0, 0.5): ["remediate"],
                    (0.5, 0.8): ["improve"],
                    (0.8, 1.0): ["polish"],
                },
            ),
            DAGNode(name="remediate", prompt_template=_tagged("remediate")),
            DAGNode(name="improve", prompt_template=_tagged("improve")),
            DAGNode(name="polish", prompt_template=_tagged("polish")),
        ],
        root="root",
    )
    await dag.evaluate(output="hi")
    names = [r.node_name for r in dag.get_last_path()]
    assert names == ["root", "improve"]


async def test_score_range_high_bucket_includes_upper_bound():
    judge = StubJudge({"root": 1.0, "polish": 0.9})
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="root",
                prompt_template=_tagged("root"),
                children_on_score_ranges={
                    (0.0, 0.5): ["remediate"],
                    (0.5, 0.8): ["improve"],
                    (0.8, 1.0): ["polish"],
                },
            ),
            DAGNode(name="remediate", prompt_template=_tagged("remediate")),
            DAGNode(name="improve", prompt_template=_tagged("improve")),
            DAGNode(name="polish", prompt_template=_tagged("polish")),
        ],
        root="root",
    )
    await dag.evaluate(output="hi")
    names = [r.node_name for r in dag.get_last_path()]
    assert names == ["root", "polish"]


async def test_score_ranges_take_precedence_over_pass_fail():
    judge = StubJudge({"root": 0.3, "remediate": 0.4})
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="root",
                prompt_template=_tagged("root"),
                threshold=0.5,
                children_on_pass=["pass_branch"],
                children_on_fail=["fail_branch"],
                children_on_score_ranges={(0.0, 1.0): ["remediate"]},
            ),
            DAGNode(name="pass_branch", prompt_template=_tagged("pass_branch")),
            DAGNode(name="fail_branch", prompt_template=_tagged("fail_branch")),
            DAGNode(name="remediate", prompt_template=_tagged("remediate")),
        ],
        root="root",
    )
    await dag.evaluate(output="hi")
    names = [r.node_name for r in dag.get_last_path()]
    assert names == ["root", "remediate"]


async def test_context_key_interpolation():
    judge = StubJudge({"root": 0.7})
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="root",
                prompt_template=_tagged("root", "rubric={context.rubric}"),
            )
        ],
        root="root",
    )
    await dag.evaluate(output="ignored", context={"rubric": "verbose"})
    assert "rubric=verbose" in judge.calls[0]


async def test_context_json_blob_interpolation():
    judge = StubJudge({"root": 0.7})
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(name="root", prompt_template=_tagged("root", "ctx={context}")),
        ],
        root="root",
    )
    await dag.evaluate(output="o", context={"k": "v", "n": 2})
    assert '"k": "v"' in judge.calls[0]


async def test_parent_score_and_reasoning_injected_into_child_prompt():
    judge = StubJudge(
        {"root": 0.55, "child": 0.9},
        reasons={"root": "root-reasoning"},
    )
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="root",
                prompt_template=_tagged("root"),
                threshold=0.5,
                children_on_pass=["child"],
            ),
            DAGNode(
                name="child",
                prompt_template=_tagged("child", "parent={parent_score} reason={parent_reasoning}"),
            ),
        ],
        root="root",
    )
    await dag.evaluate(output="o")
    child_prompt = next(c for c in judge.calls if "[node:child]" in c)
    assert "parent=0.5500" in child_prompt
    assert "reason=root-reasoning" in child_prompt


async def test_parallel_sibling_execution_is_concurrent():
    judge = StubJudge(
        {"root": 0.9, "a": 0.8, "b": 0.7, "c": 0.6},
        delay=0.05,
    )
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="root",
                prompt_template=_tagged("root"),
                children_on_pass=["a", "b", "c"],
            ),
            DAGNode(name="a", prompt_template=_tagged("a")),
            DAGNode(name="b", prompt_template=_tagged("b")),
            DAGNode(name="c", prompt_template=_tagged("c")),
        ],
        root="root",
    )
    start = time.perf_counter()
    await dag.evaluate(output="o")
    elapsed = time.perf_counter() - start
    # Root + three siblings sequentially would take ~0.2s. If siblings
    # run in parallel we expect ~0.1s.
    assert elapsed < 0.18, f"expected parallel execution, elapsed={elapsed:.3f}s"
    assert len(judge.calls) == 4


async def test_leaf_node_score_becomes_final():
    judge = StubJudge({"root": 0.2, "leaf": 0.95})
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="root",
                prompt_template=_tagged("root"),
                threshold=0.5,
                children_on_fail=["leaf"],
                weight=10.0,
            ),
            DAGNode(
                name="leaf",
                prompt_template=_tagged("leaf"),
                is_leaf=True,
            ),
        ],
        root="root",
    )
    result = await dag.evaluate(output="o")
    assert result.score == pytest.approx(0.95)
    assert result.passed is True


async def test_weighted_average_used_when_no_leaf():
    judge = StubJudge({"a": 1.0, "b": 0.0})
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="a",
                prompt_template=_tagged("a"),
                threshold=0.5,
                children_on_pass=["b"],
                weight=3.0,
            ),
            DAGNode(name="b", prompt_template=_tagged("b"), weight=1.0),
        ],
        root="a",
    )
    result = await dag.evaluate(output="o")
    assert result.score == pytest.approx(0.75)


async def test_to_mermaid_contains_nodes_and_arrow_types():
    judge = StubJudge({"root": 0.7})
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="root",
                prompt_template=_tagged("root"),
                children_on_pass=["good"],
                children_on_fail=["bad"],
                children_on_score_ranges={(0.0, 0.5): ["bad"], (0.5, 1.0): ["good"]},
            ),
            DAGNode(name="good", prompt_template=_tagged("good"), is_leaf=True),
            DAGNode(name="bad", prompt_template=_tagged("bad")),
        ],
        root="root",
    )
    mermaid = dag.to_mermaid()
    assert mermaid.startswith("flowchart TD")
    for name in ("root", "good", "bad"):
        assert name in mermaid
    assert "-- pass -->" in mermaid
    assert "-- fail -->" in mermaid
    assert "[0.00,0.50)" in mermaid
    assert "leaf" in mermaid


async def test_cycle_detection_raises():
    with pytest.raises(ValueError, match="cycle"):
        DAGMetric(
            judge=StubJudge({}),
            nodes=[
                DAGNode(
                    name="a",
                    prompt_template=_tagged("a"),
                    children_on_pass=["b"],
                ),
                DAGNode(
                    name="b",
                    prompt_template=_tagged("b"),
                    children_on_pass=["a"],
                ),
            ],
            root="a",
        )


async def test_invalid_root_raises():
    with pytest.raises(ValueError, match="Root"):
        DAGMetric(
            judge=StubJudge({}),
            nodes=[DAGNode(name="a", prompt_template=_tagged("a"))],
            root="missing",
        )


async def test_missing_child_raises():
    with pytest.raises(ValueError, match="unknown child"):
        DAGMetric(
            judge=StubJudge({}),
            nodes=[
                DAGNode(
                    name="a",
                    prompt_template=_tagged("a"),
                    children_on_pass=["ghost"],
                )
            ],
            root="a",
        )


async def test_empty_nodes_raises():
    with pytest.raises(ValueError, match="at least one node"):
        DAGMetric(judge=StubJudge({}), nodes=[], root="a")


async def test_invalid_node_threshold_rejected():
    with pytest.raises(ValueError):
        DAGNode(name="bad", prompt_template="{output}", threshold=1.5)


async def test_aevaluate_batch_preserves_order():
    judge = StubJudge({"root": 0.7}, delay=0.01)
    dag = DAGMetric(
        judge=judge,
        nodes=[DAGNode(name="root", prompt_template=_tagged("root"))],
        root="root",
    )
    outputs = [f"output-{i}" for i in range(5)]
    results = await dag.aevaluate_batch(outputs)
    assert len(results) == 5
    assert all(r.metric_name == "dag" for r in results)
    assert all(r.score == pytest.approx(0.7) for r in results)


async def test_get_last_path_exposes_traversal():
    judge = StubJudge({"root": 0.9, "child": 0.6})
    dag = DAGMetric(
        judge=judge,
        nodes=[
            DAGNode(
                name="root",
                prompt_template=_tagged("root"),
                children_on_pass=["child"],
            ),
            DAGNode(name="child", prompt_template=_tagged("child")),
        ],
        root="root",
    )
    await dag.evaluate(output="o")
    path = dag.get_last_path()
    assert isinstance(path[0], DAGEvalResult)
    assert [r.node_name for r in path] == ["root", "child"]


async def test_reasoning_contains_structured_trace_blob():
    judge = StubJudge({"root": 0.7})
    dag = DAGMetric(
        judge=judge,
        nodes=[DAGNode(name="root", prompt_template=_tagged("root"))],
        root="root",
    )
    result = await dag.evaluate(output="o")
    assert result.reasoning.startswith("[dag-trace] ")
    tail = result.reasoning[len("[dag-trace] ") :]
    decoder = json.JSONDecoder()
    decoded, _ = decoder.raw_decode(tail)
    assert decoded["path"][0]["node_name"] == "root"


async def test_dageval_result_to_dict():
    r = DAGEvalResult(node_name="x", score=0.5, passed=False, reasoning="r")
    assert r.to_dict() == {
        "node_name": "x",
        "score": 0.5,
        "passed": False,
        "reasoning": "r",
    }

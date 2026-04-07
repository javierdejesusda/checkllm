"""DAG-based composite evaluation metric using LLM-as-judge at each node.

Evaluates an LLM output through a directed acyclic graph of judge nodes.
Each node can branch to different child nodes based on whether it passes or
fails its threshold, allowing conditional evaluation pipelines.

Usage::

    from checkllm.metrics.dag import DAGMetric, DAGNode

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
                prompt_template="Is this high quality? {output}",
                threshold=0.7,
            ),
        ],
        root="safety",
    )
    result = await dag.evaluate(output="Hello world")
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult, JudgeResponse


class DAGNode(BaseModel):
    """A node in a DAG evaluation tree."""

    name: str
    prompt_template: str  # Must contain {output} placeholder
    threshold: float = 0.5
    children_on_pass: list[str] = Field(default_factory=list)  # node names
    children_on_fail: list[str] = Field(default_factory=list)
    weight: float = 1.0


class DAGEvalResult(BaseModel):
    """Result from evaluating a single DAG node."""

    node_name: str
    score: float
    passed: bool
    reasoning: str


class DAGMetric:
    """DAG-based composite evaluation metric.

    Evaluates output through a directed acyclic graph of LLM judge nodes.
    Each node can branch to different child nodes based on pass/fail.
    Final score is the weighted average of all visited nodes.

    Usage::

        dag = DAGMetric(
            judge=judge,
            nodes=[
                DAGNode(name="safety", prompt_template="Is this output safe? {output}", threshold=0.8, children_on_pass=["quality"]),
                DAGNode(name="quality", prompt_template="Is this high quality? {output}", threshold=0.7),
            ],
            root="safety",
        )
        result = await dag.evaluate(output="Hello world")
    """

    def __init__(
        self,
        judge: JudgeBackend,
        nodes: list[DAGNode],
        root: str,
        threshold: float = 0.5,
    ) -> None:
        self.judge = judge
        self.nodes = {n.name: n for n in nodes}
        self.root = root
        self.threshold = threshold

    async def evaluate(self, output: str) -> CheckResult:
        """Evaluate an output through the DAG of judge nodes.

        Args:
            output: The LLM output to evaluate.

        Returns:
            A CheckResult with a weighted score from all visited nodes,
            reasoning summarising each node's result, and timing metadata.
        """
        start = time.perf_counter_ns()
        visited: list[DAGEvalResult] = []

        await self._traverse(self.root, output, visited)

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        if not visited:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning="No nodes evaluated",
                cost=0.0,
                latency_ms=int(elapsed_ms),
                metric_name="dag",
            )

        total_weight = sum(self.nodes[r.node_name].weight for r in visited)
        weighted_score = (
            sum(r.score * self.nodes[r.node_name].weight for r in visited) / total_weight
            if total_weight > 0
            else 0.0
        )

        reasoning_parts = [
            f"{r.node_name}: {r.score:.2f} ({'pass' if r.passed else 'fail'})"
            for r in visited
        ]

        return CheckResult(
            passed=weighted_score >= self.threshold,
            score=weighted_score,
            reasoning=f"DAG evaluation: {'; '.join(reasoning_parts)}",
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="dag",
        )

    async def _traverse(
        self,
        node_name: str,
        output: str,
        visited: list[DAGEvalResult],
    ) -> None:
        """Recursively traverse the DAG from the given node.

        Args:
            node_name: The name of the current node to evaluate.
            output: The LLM output being evaluated.
            visited: Accumulated list of node results (modified in place).
        """
        if node_name not in self.nodes:
            return
        node = self.nodes[node_name]

        prompt = node.prompt_template.replace("{output}", output)
        prompt += '\n\nRespond with JSON: {"score": <float 0-1>, "reasoning": "<explanation>"}'
        response = await self.judge.evaluate(prompt=prompt)

        passed = response.score >= node.threshold
        visited.append(
            DAGEvalResult(
                node_name=node_name,
                score=response.score,
                passed=passed,
                reasoning=response.reasoning,
            )
        )

        children = node.children_on_pass if passed else node.children_on_fail
        for child_name in children:
            if not any(r.node_name == child_name for r in visited):
                await self._traverse(child_name, output, visited)

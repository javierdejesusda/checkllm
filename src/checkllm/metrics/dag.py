"""DAG-based composite evaluation metric using LLM-as-judge at each node.

Evaluates an LLM output through a directed acyclic graph of judge nodes.
Each node can branch to different child nodes based on whether it passes or
fails its threshold, or based on score ranges, allowing conditional
evaluation pipelines that closely mirror real reasoning flows.

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
import json
import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult


class DAGNode(BaseModel):
    """A node in a DAG evaluation tree.

    Attributes:
        name: Unique identifier for the node within a DAG.
        prompt_template: The prompt shown to the judge. Supports the
            placeholders ``{output}``, ``{parent_score}``,
            ``{parent_reasoning}``, ``{context}`` (the raw context dict
            rendered as JSON) and any ``{context.<key>}`` lookups.
        threshold: Pass/fail threshold in ``[0.0, 1.0]``.
        children_on_pass: Children visited when the node passes its
            threshold.
        children_on_fail: Children visited when the node fails.
        children_on_score_ranges: Optional mapping ``{(lo, hi): [child, ...]}``
            that routes by score bucket. Ranges are inclusive on the low
            bound and exclusive on the high bound, except that the bucket
            containing ``1.0`` also includes the upper bound. When set, it
            takes precedence over ``children_on_pass`` /
            ``children_on_fail``.
        weight: Weight applied when computing the weighted-average score.
        is_leaf: When True, this node is a terminal verdict node. Its
            score replaces the weighted average when the traversal reaches
            it (the first reached leaf wins).
        terminal_weight: Optional override for the weight used by
            ``is_leaf`` nodes. Currently advisory — included so callers can
            prioritise a specific leaf when multiple leaves might run.
    """

    name: str
    prompt_template: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    children_on_pass: list[str] = Field(default_factory=list)
    children_on_fail: list[str] = Field(default_factory=list)
    children_on_score_ranges: dict[tuple[float, float], list[str]] = Field(default_factory=dict)
    weight: float = 1.0
    is_leaf: bool = False
    terminal_weight: float | None = None

    @field_validator("children_on_score_ranges")
    @classmethod
    def _validate_ranges(
        cls, v: dict[tuple[float, float], list[str]]
    ) -> dict[tuple[float, float], list[str]]:
        """Ensure every score-range key is a valid ``(lo, hi)`` pair."""
        for key in v:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(isinstance(x, (int, float)) for x in key)
            ):
                raise ValueError(f"Score range keys must be (lo, hi) tuples, got {key!r}")
            lo, hi = key
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(f"Score range {key!r} must satisfy 0 <= lo <= hi <= 1")
        return v


class DAGEvalResult(BaseModel):
    """Result from evaluating a single DAG node."""

    node_name: str
    score: float
    passed: bool
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the node result."""
        return {
            "node_name": self.node_name,
            "score": self.score,
            "passed": self.passed,
            "reasoning": self.reasoning,
        }


_PATH_PREFIX = "[dag-trace] "


class DAGMetric:
    """DAG-based composite evaluation metric.

    Evaluates output through a directed acyclic graph of LLM judge nodes.
    Each node can branch to different child nodes based on pass/fail or
    score ranges. The final score is either the score of the first reached
    leaf (when any node is marked ``is_leaf=True``) or the weighted
    average of every visited node.

    The traversal path is preserved across evaluations and exposed via
    :meth:`get_last_path` as well as encoded in the returned
    :class:`~checkllm.models.CheckResult`'s ``reasoning`` field under a
    structured ``[dag-trace] {...}`` prefix.
    """

    def __init__(
        self,
        judge: JudgeBackend,
        nodes: list[DAGNode],
        root: str,
        threshold: float = 0.5,
    ) -> None:
        """Initialise the metric.

        Args:
            judge: Async judge backend used for every node.
            nodes: The node definitions. Must contain at least one node.
            root: Name of the root node — must be present in ``nodes``.
            threshold: Overall pass threshold applied to the final score.

        Raises:
            ValueError: If ``nodes`` is empty, ``root`` is missing,
                a referenced child name does not exist, a node threshold
                is outside ``[0, 1]``, or the graph contains a cycle.
        """
        if not nodes:
            raise ValueError("DAGMetric requires at least one node")
        self.judge = judge
        self.nodes: dict[str, DAGNode] = {n.name: n for n in nodes}
        self.root = root
        self.threshold = threshold
        self._last_path: list[DAGEvalResult] = []

        self._validate_graph()

    def _validate_graph(self) -> None:
        """Check graph invariants. Raises ValueError on violation."""
        if self.root not in self.nodes:
            raise ValueError(f"Root node {self.root!r} not found in nodes")

        for node in self.nodes.values():
            if not 0.0 <= node.threshold <= 1.0:
                raise ValueError(f"Node {node.name!r} threshold {node.threshold} outside [0, 1]")
            refs: list[str] = []
            refs.extend(node.children_on_pass)
            refs.extend(node.children_on_fail)
            for children in node.children_on_score_ranges.values():
                refs.extend(children)
            for child in refs:
                if child not in self.nodes:
                    raise ValueError(f"Node {node.name!r} references unknown child {child!r}")

        if self._has_cycle():
            raise ValueError("DAG contains a cycle")

    def _all_children(self, node: DAGNode) -> list[str]:
        """Return every distinct child name referenced by a node."""
        seen: list[str] = []
        pools: list[list[str]] = [node.children_on_pass, node.children_on_fail]
        pools.extend(node.children_on_score_ranges.values())
        for pool in pools:
            for child in pool:
                if child not in seen:
                    seen.append(child)
        return seen

    def _has_cycle(self) -> bool:
        """Detect a cycle via iterative DFS with colours."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colour = {name: WHITE for name in self.nodes}

        def visit(start: str) -> bool:
            stack: list[tuple[str, int]] = [(start, 0)]
            colour[start] = GRAY
            children_cache: dict[str, list[str]] = {}
            while stack:
                name, idx = stack[-1]
                kids = children_cache.get(name)
                if kids is None:
                    kids = self._all_children(self.nodes[name])
                    children_cache[name] = kids
                if idx >= len(kids):
                    colour[name] = BLACK
                    stack.pop()
                    continue
                stack[-1] = (name, idx + 1)
                child = kids[idx]
                if colour[child] == GRAY:
                    return True
                if colour[child] == WHITE:
                    colour[child] = GRAY
                    stack.append((child, 0))
            return False

        for name in self.nodes:
            if colour[name] == WHITE:
                if visit(name):
                    return True
        return False

    def get_last_path(self) -> list[DAGEvalResult]:
        """Return the traversal path from the most recent ``evaluate`` call."""
        return list(self._last_path)

    async def evaluate(
        self,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> CheckResult:
        """Evaluate an output through the DAG of judge nodes.

        Args:
            output: The LLM output to evaluate.
            context: Optional dict whose values are interpolated into each
                node's ``prompt_template`` via ``{context.<key>}`` and
                exposed as a JSON blob via ``{context}``.

        Returns:
            A :class:`~checkllm.models.CheckResult`. The reasoning field
            starts with a ``[dag-trace] {json}`` preamble (the JSON
            encodes the visited nodes) followed by a human-readable
            summary. The traversal path is also accessible via
            :meth:`get_last_path`.
        """
        start = time.perf_counter_ns()
        visited: list[DAGEvalResult] = []
        ctx = dict(context) if context else {}
        leaf_result: DAGEvalResult | None = None

        leaf_result = await self._traverse(
            self.root,
            output,
            ctx,
            visited,
            parent=None,
        )

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
        self._last_path = visited

        if not visited:
            return CheckResult(
                passed=False,
                score=0.0,
                reasoning="No nodes evaluated",
                cost=0.0,
                latency_ms=int(elapsed_ms),
                metric_name="dag",
            )

        if leaf_result is not None:
            final_score = leaf_result.score
        else:
            total_weight = sum(self.nodes[r.node_name].weight for r in visited)
            final_score = (
                sum(r.score * self.nodes[r.node_name].weight for r in visited) / total_weight
                if total_weight > 0
                else 0.0
            )

        reasoning_summary = "; ".join(
            f"{r.node_name}: {r.score:.2f} ({'pass' if r.passed else 'fail'})" for r in visited
        )
        trace_blob = json.dumps({"path": [r.to_dict() for r in visited]})
        reasoning = f"{_PATH_PREFIX}{trace_blob} DAG evaluation: {reasoning_summary}"

        return CheckResult(
            passed=final_score >= self.threshold,
            score=final_score,
            reasoning=reasoning,
            cost=0.0,
            latency_ms=int(elapsed_ms),
            metric_name="dag",
        )

    async def aevaluate_batch(self, outputs: list[str]) -> list[CheckResult]:
        """Evaluate multiple outputs concurrently.

        Args:
            outputs: Outputs to evaluate — order is preserved in the result.

        Returns:
            A list of :class:`CheckResult` values, one per input, in the
            same order as ``outputs``.
        """
        if not outputs:
            return []
        return list(await asyncio.gather(*(self.evaluate(o) for o in outputs)))

    async def _traverse(
        self,
        node_name: str,
        output: str,
        context: dict[str, Any],
        visited: list[DAGEvalResult],
        parent: DAGEvalResult | None,
    ) -> DAGEvalResult | None:
        """Recursively traverse the DAG from the given node.

        Args:
            node_name: The current node's name.
            output: The LLM output being evaluated.
            context: User-supplied context values interpolated into
                prompts.
            visited: Accumulated list of node results (modified in place
                as a traversal-path log).
            parent: The result of the parent node that routed to this
                node, if any. Used for context interpolation.

        Returns:
            The first leaf :class:`DAGEvalResult` reached in this subtree
            (or ``None`` if no leaf was reached).
        """
        if node_name not in self.nodes:
            return None
        if any(r.node_name == node_name for r in visited):
            return None

        node = self.nodes[node_name]
        prompt = self._render_prompt(node.prompt_template, output, context, parent)
        prompt += '\n\nRespond with JSON: {"score": <float 0-1>, "reasoning": "<explanation>"}'
        response = await self.judge.evaluate(prompt=prompt)

        passed = response.score >= node.threshold
        result = DAGEvalResult(
            node_name=node_name,
            score=response.score,
            passed=passed,
            reasoning=response.reasoning,
        )
        visited.append(result)

        if node.is_leaf:
            return result

        children = self._select_children(node, response.score, passed)
        pending = [c for c in children if not any(r.node_name == c for r in visited)]
        if not pending:
            return None

        child_results = await asyncio.gather(
            *(self._traverse(c, output, context, visited, parent=result) for c in pending)
        )
        for child_leaf in child_results:
            if child_leaf is not None:
                return child_leaf
        return None

    @staticmethod
    def _select_children(node: DAGNode, score: float, passed: bool) -> list[str]:
        """Choose children to visit from a node given its judge score."""
        if node.children_on_score_ranges:
            for (lo, hi), kids in node.children_on_score_ranges.items():
                if lo <= score < hi:
                    return list(kids)
                if hi == 1.0 and score == 1.0 and lo <= score <= hi:
                    return list(kids)
            return []
        return list(node.children_on_pass if passed else node.children_on_fail)

    @staticmethod
    def _render_prompt(
        template: str,
        output: str,
        context: dict[str, Any],
        parent: DAGEvalResult | None,
    ) -> str:
        """Substitute supported placeholders in a prompt template."""
        rendered = template.replace("{output}", output)
        parent_score = f"{parent.score:.4f}" if parent is not None else ""
        parent_reasoning = parent.reasoning if parent is not None else ""
        rendered = rendered.replace("{parent_score}", parent_score)
        rendered = rendered.replace("{parent_reasoning}", parent_reasoning)
        if "{context}" in rendered:
            rendered = rendered.replace("{context}", json.dumps(context, default=str))
        for key, value in context.items():
            rendered = rendered.replace("{context." + str(key) + "}", str(value))
        return rendered

    def to_mermaid(self) -> str:
        """Render the DAG as a Mermaid flowchart.

        Returns:
            A Mermaid ``flowchart TD`` string with one line per node and
            labelled edges for pass, fail, and score-range branches.
            Leaf nodes are given the ``:::leaf`` class so they can be
            styled in documentation.
        """
        lines: list[str] = ["flowchart TD"]
        for name, node in self.nodes.items():
            label = name if not node.is_leaf else f"{name} (leaf)"
            lines.append(f'    {name}["{label}"]')
        for name, node in self.nodes.items():
            for child in node.children_on_pass:
                lines.append(f"    {name} -- pass --> {child}")
            for child in node.children_on_fail:
                lines.append(f"    {name} -- fail --> {child}")
            for (lo, hi), kids in node.children_on_score_ranges.items():
                label = f"[{lo:.2f},{hi:.2f})"
                for child in kids:
                    lines.append(f'    {name} -- "{label}" --> {child}')
        leaves = [name for name, node in self.nodes.items() if node.is_leaf]
        if leaves:
            lines.append("    classDef leaf fill:#e0f7e9,stroke:#2e7d32;")
            lines.append(f"    class {','.join(leaves)} leaf;")
        return "\n".join(lines)

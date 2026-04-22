"""Promptfoo-style declarative assertions for YAML evaluations.

This module parses a list of assertion specifications from a YAML eval
config and executes them against an LLM output. It mirrors promptfoo's
assertion vocabulary while mapping the model-graded types onto existing
CheckLLM metrics so the same set of judges and metric implementations
back both the programmatic and declarative APIs.

Example YAML snippet::

    assert:
      - type: contains
        value: "hello"
      - type: llm-rubric
        rubric: "Response must be concise and professional."
        threshold: 0.75
      - type: cost
        value: 0.02
      - type: latency
        value: 1500
"""

from __future__ import annotations

import re
import time
from typing import Any

from pydantic import BaseModel, Field

from checkllm.deterministic import DeterministicChecks
from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult


class Assertion(BaseModel):
    """Parsed declarative assertion.

    Attributes:
        type: The normalized assertion identifier (lowercased, dashes
            preserved).
        value: Optional scalar value associated with the assertion
            (e.g. the expected substring for ``contains``).
        threshold: Optional score threshold for metric-backed
            assertions. Defaults to metric-specific values.
        rubric: Natural-language rubric used by ``llm-rubric``.
        prompt: Free-form judge prompt used by the bare ``model`` type.
        reference: Reference text used by ``similarity``.
        query: Query used by ``model-graded-relevance``.
        context: Context used by ``model-graded-faithfulness``.
        metric: Resolved metric name (set post-parse for model-graded
            types).
        raw: Original dictionary the assertion was parsed from, kept
            for debugging.
    """

    type: str
    value: Any = None
    threshold: float | None = None
    rubric: str | None = None
    prompt: str | None = None
    reference: str | None = None
    query: str | None = None
    context: str | None = None
    metric: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class AssertionResults(BaseModel):
    """Aggregate result of evaluating a list of assertions.

    Attributes:
        passed: ``True`` when every individual assertion passed.
        individual: Per-assertion :class:`CheckResult` records in the
            original assertion order.
    """

    passed: bool
    individual: list[CheckResult] = Field(default_factory=list)


SUPPORTED_TYPES = frozenset(
    {
        "contains",
        "not-contains",
        "regex",
        "equals",
        "model",
        "llm-rubric",
        "similarity",
        "model-graded-relevance",
        "model-graded-faithfulness",
        "cost",
        "latency",
    }
)


def _normalize_type(raw: str) -> str:
    """Lowercase and swap underscores for dashes.

    Promptfoo uses ``llm-rubric`` while some checkllm users write
    ``llm_rubric``. Normalising early keeps downstream code simple.
    """
    return raw.strip().lower().replace("_", "-")


def parse_assertions(raw: list[dict[str, Any]]) -> list[Assertion]:
    """Parse a list of dicts into :class:`Assertion` objects.

    Args:
        raw: List of assertion dictionaries loaded from YAML.

    Returns:
        A list of validated assertions.

    Raises:
        ValueError: If any assertion has an unsupported ``type`` or is
            missing required fields for its type.
    """
    out: list[Assertion] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Assertion #{idx} is not a mapping: {entry!r}")
        if "type" not in entry:
            raise ValueError(f"Assertion #{idx} missing required 'type' key")

        atype = _normalize_type(str(entry["type"]))
        if atype not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown assertion type '{entry['type']}' at index {idx}. "
                f"Supported types: {', '.join(sorted(SUPPORTED_TYPES))}."
            )

        assertion = Assertion(
            type=atype,
            value=entry.get("value"),
            threshold=entry.get("threshold"),
            rubric=entry.get("rubric"),
            prompt=entry.get("prompt"),
            reference=entry.get("reference"),
            query=entry.get("query"),
            context=entry.get("context"),
            raw=entry,
        )

        # Required-field checks per type
        if atype == "contains" and assertion.value is None:
            raise ValueError(f"Assertion #{idx} 'contains' requires a 'value'")
        if atype == "not-contains" and assertion.value is None:
            raise ValueError(f"Assertion #{idx} 'not-contains' requires a 'value'")
        if atype == "regex" and assertion.value is None:
            raise ValueError(f"Assertion #{idx} 'regex' requires a 'value'")
        if atype == "equals" and assertion.value is None:
            raise ValueError(f"Assertion #{idx} 'equals' requires a 'value'")
        if atype == "llm-rubric" and not assertion.rubric:
            raise ValueError(f"Assertion #{idx} 'llm-rubric' requires a 'rubric' string")
        if atype == "model" and not assertion.prompt:
            raise ValueError(f"Assertion #{idx} 'model' requires a 'prompt' string")
        if atype == "similarity" and assertion.reference is None and assertion.value is None:
            raise ValueError(f"Assertion #{idx} 'similarity' requires 'reference' or 'value'")
        if atype == "model-graded-relevance" and not assertion.query:
            raise ValueError(f"Assertion #{idx} 'model-graded-relevance' requires a 'query'")
        if atype == "model-graded-faithfulness" and not assertion.context:
            raise ValueError(f"Assertion #{idx} 'model-graded-faithfulness' requires a 'context'")
        if atype in {"cost", "latency"} and assertion.value is None:
            raise ValueError(
                f"Assertion #{idx} '{atype}' requires a numeric 'value' (maximum allowed)"
            )

        # Resolve the concrete metric name for model-graded types
        if atype == "llm-rubric":
            assertion.metric = "rubric"
        elif atype == "model-graded-relevance":
            assertion.metric = "relevance"
        elif atype == "model-graded-faithfulness":
            assertion.metric = "faithfulness"
        elif atype == "model":
            assertion.metric = "model"

        out.append(assertion)
    return out


def _render(template: str | None, context: dict[str, Any] | None) -> str:
    """Replace ``{{var}}`` tokens in *template* using *context*.

    Args:
        template: Template string or ``None``.
        context: Variable mapping. ``None`` is treated as empty.

    Returns:
        The rendered string, or an empty string when *template* is
        ``None``.
    """
    if template is None:
        return ""
    if not context:
        return template
    out = template
    for key, val in context.items():
        out = out.replace("{{" + key + "}}", str(val))
        out = out.replace("{{ " + key + " }}", str(val))
    return out


_MODEL_JUDGE_SYSTEM = (
    "You are an expert evaluator. Follow the user's instructions and respond "
    'with JSON only: {"score": <float 0-1>, "reasoning": "<explanation>"}.'
)


async def _run_model_judge(
    *,
    judge: JudgeBackend,
    judge_prompt: str,
    output: str,
    threshold: float,
    metric_name: str,
) -> CheckResult:
    """Invoke the judge with *judge_prompt* and wrap the response.

    Args:
        judge: A judge backend.
        judge_prompt: The rubric or free-form prompt to send.
        output: The LLM output under evaluation.
        threshold: Pass/fail threshold for the returned score.
        metric_name: Name to record on the :class:`CheckResult`.

    Returns:
        A populated :class:`CheckResult`.
    """
    prompt = (
        f"Instructions:\n{judge_prompt}\n\n"
        f"Candidate output:\n{output}\n\n"
        "Score it on a scale from 0.0 (completely fails) to 1.0 (fully satisfies)."
    )
    start = time.perf_counter_ns()
    response = await judge.evaluate(prompt=prompt, system_prompt=_MODEL_JUDGE_SYSTEM)
    elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
    return CheckResult(
        passed=response.score >= threshold,
        score=response.score,
        reasoning=response.reasoning,
        cost=response.cost,
        latency_ms=int(elapsed_ms),
        metric_name=metric_name,
        threshold=threshold,
    )


async def _run_single(
    assertion: Assertion,
    output: str,
    *,
    judge: JudgeBackend,
    context: dict[str, Any] | None,
    deterministic: DeterministicChecks,
) -> CheckResult:
    """Evaluate a single :class:`Assertion` and return a CheckResult."""
    atype = assertion.type
    vars_ctx = context or {}

    if atype == "contains":
        return deterministic.contains(output, str(assertion.value))
    if atype == "not-contains":
        return deterministic.not_contains(output, str(assertion.value))
    if atype == "regex":
        return deterministic.regex(output, str(assertion.value))
    if atype == "equals":
        return deterministic.exact_match(output, str(assertion.value))

    if atype == "similarity":
        reference = assertion.reference if assertion.reference is not None else assertion.value
        t = assertion.threshold if assertion.threshold is not None else 0.8
        return deterministic.similarity(output, str(reference), threshold=t)

    if atype == "cost":
        last_cost = float(getattr(judge, "last_cost", 0.0) or 0.0)
        limit = float(assertion.value)
        passed = last_cost <= limit
        return CheckResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Last judge call cost ${last_cost:.6f} (limit ${limit:.6f})",
            cost=0.0,
            latency_ms=0,
            metric_name="cost",
            threshold=None,
        )

    if atype == "latency":
        ctx_latency = vars_ctx.get("latency_ms")
        latency_ms = float(ctx_latency if ctx_latency is not None else 0.0)
        limit = float(assertion.value)
        passed = latency_ms <= limit
        return CheckResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reasoning=f"Observed latency {latency_ms:.1f}ms (limit {limit:.1f}ms)",
            cost=0.0,
            latency_ms=int(latency_ms),
            metric_name="latency",
            threshold=None,
        )

    if atype == "model":
        threshold = assertion.threshold if assertion.threshold is not None else 0.8
        rendered_prompt = _render(assertion.prompt, vars_ctx)
        return await _run_model_judge(
            judge=judge,
            judge_prompt=rendered_prompt,
            output=output,
            threshold=threshold,
            metric_name="model",
        )

    if atype == "llm-rubric":
        from checkllm.metrics.rubric import RubricMetric

        threshold = assertion.threshold if assertion.threshold is not None else 0.8
        rubric = _render(assertion.rubric, vars_ctx)
        rubric_metric = RubricMetric(judge=judge)
        return await rubric_metric.evaluate(output=output, criteria=rubric, threshold=threshold)

    if atype == "model-graded-relevance":
        from checkllm.metrics.relevance import RelevanceMetric

        threshold = assertion.threshold if assertion.threshold is not None else 0.8
        query = _render(assertion.query, vars_ctx)
        relevance_metric = RelevanceMetric(judge=judge, threshold=threshold)
        return await relevance_metric.evaluate(output=output, query=query)

    if atype == "model-graded-faithfulness":
        from checkllm.metrics.faithfulness import FaithfulnessMetric

        threshold = assertion.threshold if assertion.threshold is not None else 0.8
        ctx_text = _render(assertion.context, vars_ctx)
        query_opt: str | None = _render(assertion.query, vars_ctx) if assertion.query else None
        faithfulness_metric = FaithfulnessMetric(judge=judge, threshold=threshold)
        return await faithfulness_metric.evaluate(output=output, context=ctx_text, query=query_opt)

    # parse_assertions guards unknown types, but be defensive anyway.
    return CheckResult(
        passed=False,
        score=0.0,
        reasoning=f"Unknown assertion type: '{atype}'",
        cost=0.0,
        latency_ms=0,
        metric_name=atype,
    )


async def evaluate_assertions(
    output: str,
    assertions: list[Assertion],
    *,
    judge: JudgeBackend,
    context: dict[str, Any] | None = None,
) -> AssertionResults:
    """Evaluate every assertion in *assertions* against *output*.

    Args:
        output: The LLM-generated text to evaluate.
        assertions: Parsed assertions, typically from
            :func:`parse_assertions`.
        judge: Judge backend used by model-graded assertions.
        context: Optional variables for template substitution plus
            observability fields such as ``latency_ms`` used by the
            ``latency`` assertion.

    Returns:
        An :class:`AssertionResults` aggregate.
    """
    deterministic = DeterministicChecks()
    individual: list[CheckResult] = []
    for assertion in assertions:
        try:
            result = await _run_single(
                assertion,
                output,
                judge=judge,
                context=context,
                deterministic=deterministic,
            )
        except re.error as exc:
            result = CheckResult(
                passed=False,
                score=0.0,
                reasoning=f"Invalid regex in assertion: {exc}",
                cost=0.0,
                latency_ms=0,
                metric_name=assertion.type,
            )
        except Exception as exc:
            result = CheckResult(
                passed=False,
                score=0.0,
                reasoning=f"Assertion error: {exc}",
                cost=0.0,
                latency_ms=0,
                metric_name=assertion.type,
            )
        individual.append(result)
    passed = all(r.passed for r in individual)
    return AssertionResults(passed=passed, individual=individual)


__all__ = [
    "SUPPORTED_TYPES",
    "Assertion",
    "AssertionResults",
    "evaluate_assertions",
    "parse_assertions",
]

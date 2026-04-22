"""Evaluating an agent that calls tools.

Scenario: a scripted agent that answers user queries by calling one or
more of ``search``, ``calculator`` and ``send_email``. The tests show:

- ``tool_call_f1`` - deterministic F1 over tool names; free and instant.
- ``tool_accuracy`` - judge-based "were the right tools picked?".
- ``plan_adherence`` - execution trace follows the declared plan.
- ``task_completion`` / ``trajectory_step_count`` - final outcome and
  step-efficiency signals.

A ``MockJudge`` stands in for a real LLM judge to keep tests offline.
Swap in ``checkllm.judge.OpenAIJudge`` for live scoring. Look at
``tests/test_agents.py`` for end-to-end live examples.

Run with: pytest examples/test_agentic_evaluation.py
"""

from __future__ import annotations

import asyncio

import pytest

from checkllm import AgentStep, AgentTestCase, ToolCall, validate_tool_calls
from checkllm.metrics.tool_call_f1 import ToolCallF1Metric
from checkllm.testing import MockJudge, make_collector


def _tool_call_f1(predicted: list[str], expected: list[str], threshold: float = 0.8):
    """Synchronously run the async ToolCallF1Metric for a test assertion."""
    metric = ToolCallF1Metric(threshold=threshold)
    return asyncio.run(metric.evaluate(predicted_tools=predicted, expected_tools=expected))


def _run_fake_agent(query: str) -> AgentTestCase:
    """Scripted agent: picks tools based on keywords in the query."""
    steps: list[AgentStep] = []

    if "email" in query.lower():
        steps.append(
            AgentStep(
                thought="User wants to send an email.",
                action="call_tool",
                tool_call=ToolCall(
                    name="send_email",
                    parameters={"to": "alice@example.com", "subject": "Update"},
                    result="sent",
                ),
            )
        )
    elif any(tok in query for tok in ("+", "-", "*", "/")):
        steps.append(
            AgentStep(
                thought="Arithmetic question; use calculator.",
                action="call_tool",
                tool_call=ToolCall(
                    name="calculator",
                    parameters={"expression": query},
                    result="42",
                ),
            )
        )
    else:
        steps.append(
            AgentStep(
                thought="Open-ended question; search the web first.",
                action="call_tool",
                tool_call=ToolCall(
                    name="search",
                    parameters={"q": query},
                    result="[1] Wikipedia: matching snippet",
                ),
            )
        )

    final = "Here is the answer based on tool output."
    return AgentTestCase(query=query, steps=steps, final_output=final)


@pytest.fixture
def agent_collector() -> object:
    """Collector backed by a MockJudge so tests run offline."""
    judge = MockJudge(default_score=0.9, default_reasoning="Mock: tools look reasonable")
    return make_collector(judge=judge, threshold=0.8)


def test_tool_call_f1_deterministic(agent_collector) -> None:
    """Deterministic F1 on tool names. Free, instant, always run first."""
    case = _run_fake_agent("calculate 6 * 7")
    predicted = [tc.name for tc in case.tool_calls]
    expected = ["calculator"]

    result = _tool_call_f1(predicted, expected, threshold=0.8)
    agent_collector.results.append(result)
    assert result.passed, result.reasoning


def test_validate_tool_calls_exact_args(agent_collector) -> None:
    """Strict arg-level validation using ``validate_tool_calls``."""
    case = _run_fake_agent("send an email to the team")
    case.expected_tools = [
        ToolCall(
            name="send_email",
            parameters={"to": "alice@example.com", "subject": "Update"},
        )
    ]
    result = validate_tool_calls(case)
    agent_collector.results.append(result)
    assert result.passed, result.reasoning


def test_tool_accuracy_judge(agent_collector) -> None:
    """LLM-judge check: were the right tools picked for the query?"""
    case = _run_fake_agent("What is the capital of France?")
    agent_collector.tool_accuracy(
        output=case.final_output or "",
        expected_tools=[{"name": "search"}],
        query=case.query,
        threshold=0.8,
    )
    assert all(r.passed for r in agent_collector.results)


def test_plan_adherence(agent_collector) -> None:
    """Verify the execution trace follows the declared plan."""
    plan = "1. Search for the capital. 2. Summarise the top result."
    case = _run_fake_agent("What is the capital of France?")
    execution_trace = "\n".join(
        f"step {i}: {step.action} -> {step.tool_call.name if step.tool_call else 'n/a'}"
        for i, step in enumerate(case.steps, 1)
    )
    agent_collector.plan_adherence(
        plan=plan,
        execution_trace=execution_trace,
        threshold=0.75,
    )
    assert all(r.passed for r in agent_collector.results)


def test_task_completion_and_step_budget(agent_collector) -> None:
    """Compose task-completion signal with a deterministic step-count guard."""
    case = _run_fake_agent("What is the capital of France?")

    agent_collector.task_completion(
        output=case.final_output or "",
        task_description=case.query,
        threshold=0.8,
    )

    # Keep the agent honest: no unnecessary tool churn.
    assert len(case.steps) <= 3, f"Agent used {len(case.steps)} steps; budget is 3"


def test_full_agent_suite_composed(agent_collector) -> None:
    """Layered evaluation: deterministic first, then judge metrics."""
    case = _run_fake_agent("send email")
    predicted = [tc.name for tc in case.tool_calls]

    # Deterministic (free)
    f1_result = _tool_call_f1(predicted, ["send_email"], threshold=0.8)
    agent_collector.results.append(f1_result)
    assert len(case.steps) <= 5

    # Judge-based (priced)
    agent_collector.tool_accuracy(
        output=case.final_output or "",
        expected_tools=[{"name": "send_email"}],
        query=case.query,
        threshold=0.8,
    )
    agent_collector.task_completion(
        output=case.final_output or "",
        task_description=case.query,
        threshold=0.8,
    )

    assert all(r.passed for r in agent_collector.results), agent_collector.results


# Run with: pytest examples/test_agentic_evaluation.py

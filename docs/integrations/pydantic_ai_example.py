"""Score a pydantic-ai agent run with CheckLLM.

Run a real ``pydantic_ai.Agent``, then translate its ``RunResult`` (with
``all_messages()``) into a CheckLLM ``AgentTestCase`` and score it
deterministically with ``TrajectoryMetric``.

Usage::

    pip install pydantic-ai checkllm
    python docs/integrations/pydantic_ai_example.py
"""

from __future__ import annotations


def main() -> None:
    from pydantic_ai import Agent, RunContext

    from checkllm.integrations.pydantic_ai import to_checkllm_test_case
    from checkllm.metrics.trajectory_metric import TrajectoryMetric

    agent = Agent("openai:gpt-4o-mini")

    @agent.tool
    def add(ctx: RunContext[None], a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @agent.tool
    def multiply(ctx: RunContext[None], a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    result = agent.run_sync("What is (3 + 4) * 5?")

    case = to_checkllm_test_case(result, query="What is (3 + 4) * 5?")
    metric = TrajectoryMetric(expected_trajectory=["add", "multiply"])
    print(metric.evaluate(case.tool_calls).reasoning)


if __name__ == "__main__":
    main()

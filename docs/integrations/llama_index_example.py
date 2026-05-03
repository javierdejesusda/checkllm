"""Score a LlamaIndex agent run with CheckLLM.

Run a real LlamaIndex agent, then translate its ``AgentChatResponse``
into a CheckLLM ``AgentTestCase`` and score it deterministically with
``TrajectoryMetric``.

Usage::

    pip install llama-index llama-index-llms-openai checkllm
    python docs/integrations/llama_index_example.py
"""

from __future__ import annotations


def main() -> None:
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.openai import OpenAI

    from checkllm.integrations.llama_index import to_checkllm_test_case
    from checkllm.metrics.trajectory_metric import TrajectoryMetric

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    tools = [FunctionTool.from_defaults(add), FunctionTool.from_defaults(multiply)]
    agent = ReActAgent.from_tools(tools, llm=OpenAI(model="gpt-4o-mini"), verbose=False)

    response = agent.chat("What is (3 + 4) * 5?")

    case = to_checkllm_test_case(response, query="What is (3 + 4) * 5?")
    metric = TrajectoryMetric(expected_trajectory=["add", "multiply"])
    print(metric.evaluate(case.tool_calls).reasoning)


if __name__ == "__main__":
    main()

"""Score a LangChain agent run with CheckLLM.

Run a real LangChain ``AgentExecutor``, then translate its
``intermediate_steps`` into a CheckLLM ``AgentTestCase`` and feed it
to ``TrajectoryMetric``. The metric is deterministic and free; it
returns four sub-scores plus an overall score in [0, 1].

Usage::

    pip install langchain langchain-openai checkllm
    python docs/integrations/langchain_example.py

This file is also valid as documentation: every line below the imports
is what a user would write end-to-end.
"""

from __future__ import annotations


def main() -> None:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.tools import tool
    from langchain_openai import ChatOpenAI

    from checkllm.integrations.langchain import to_checkllm_test_case
    from checkllm.metrics.trajectory_metric import TrajectoryMetric

    @tool
    def search(query: str) -> str:
        """Pretend web search."""
        return f"results for {query}"

    @tool
    def fetch(url: str) -> str:
        """Pretend HTTP GET."""
        return f"<html>{url}</html>"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_tool_calling_agent(llm, [search, fetch], prompt="...")
    executor = AgentExecutor(
        agent=agent, tools=[search, fetch], return_intermediate_steps=True
    )

    run = executor.invoke({"input": "Summarise example.org/climate."})

    case = to_checkllm_test_case(run, query="Summarise example.org/climate.")
    metric = TrajectoryMetric(expected_trajectory=["search", "fetch"])
    print(metric.evaluate(case.tool_calls).reasoning)


if __name__ == "__main__":
    main()

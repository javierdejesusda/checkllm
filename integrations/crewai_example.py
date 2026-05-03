"""Score a CrewAI agent run with CheckLLM.

Run a real CrewAI ``Crew``, then translate its ``CrewOutput`` into a
CheckLLM ``AgentTestCase`` and score it deterministically with
``TrajectoryMetric``.

Usage::

    pip install crewai crewai-tools checkllm
    python docs/integrations/crewai_example.py
"""

from __future__ import annotations


def main() -> None:
    from crewai import Agent, Crew, Task
    from crewai_tools import SerperDevTool

    from checkllm.integrations.crewai import to_checkllm_test_case
    from checkllm.metrics.trajectory_metric import TrajectoryMetric

    search_tool = SerperDevTool()

    researcher = Agent(
        role="Travel researcher",
        goal="Plan a 2-day trip to Lisbon.",
        backstory="Loves Iberia.",
        tools=[search_tool],
    )
    plan_task = Task(
        description="Plan a 2-day trip to Lisbon.",
        expected_output="A bullet list of day-by-day activities.",
        agent=researcher,
    )
    crew = Crew(agents=[researcher], tasks=[plan_task])

    crew_output = crew.kickoff()

    case = to_checkllm_test_case(crew_output, query="Plan a 2-day trip to Lisbon.")
    metric = TrajectoryMetric(expected_trajectory=["serper_dev"])
    print(metric.evaluate(case.tool_calls).reasoning)


if __name__ == "__main__":
    main()

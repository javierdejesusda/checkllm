import pytest

from checkllm.agents import (
    ToolCall,
    AgentStep,
    AgentTestCase,
    validate_tool_calls,
    validate_trajectory_length,
    validate_tool_order,
    validate_no_repeated_tools,
)


class TestToolCall:
    def test_creates_tool_call(self):
        tc = ToolCall(name="search", parameters={"query": "weather"})
        assert tc.name == "search"
        assert tc.parameters == {"query": "weather"}

    def test_default_parameters(self):
        tc = ToolCall(name="get_time")
        assert tc.parameters == {}
        assert tc.result is None
        assert tc.timestamp_ms is None


class TestAgentStep:
    def test_all_fields(self):
        tc = ToolCall(name="search", parameters={"q": "hi"}, result="found")
        step = AgentStep(
            thought="I should search",
            action="call_tool",
            tool_call=tc,
            observation="Results returned",
            metadata={"idx": 1},
        )
        assert step.thought == "I should search"
        assert step.action == "call_tool"
        assert step.tool_call.name == "search"
        assert step.observation == "Results returned"
        assert step.metadata == {"idx": 1}

    def test_defaults(self):
        step = AgentStep()
        assert step.thought is None
        assert step.action is None
        assert step.tool_call is None
        assert step.observation is None
        assert step.metadata == {}


class TestAgentTestCase:
    @pytest.fixture
    def agent_case(self):
        return AgentTestCase(
            query="What is the weather?",
            steps=[
                AgentStep(
                    action="call_tool",
                    tool_call=ToolCall(name="search", parameters={"query": "weather"}),
                ),
                AgentStep(
                    action="respond",
                    tool_call=None,
                ),
                AgentStep(
                    action="call_tool",
                    tool_call=ToolCall(name="format", parameters={"style": "short"}),
                ),
            ],
            final_output="It is sunny.",
        )

    def test_tool_calls_property(self, agent_case):
        tool_calls = agent_case.tool_calls
        assert len(tool_calls) == 2
        assert tool_calls[0].name == "search"
        assert tool_calls[1].name == "format"

    def test_trajectory_property(self, agent_case):
        trajectory = agent_case.trajectory
        assert trajectory == ["call_tool", "respond", "call_tool"]

    def test_format_trace(self, agent_case):
        trace = agent_case.format_trace()
        assert "Query: What is the weather?" in trace
        assert "Step 1:" in trace
        assert "Tool: search(" in trace
        assert "Action: respond" in trace
        assert "Final Output: It is sunny." in trace

    def test_format_trace_with_thought_and_observation(self):
        case = AgentTestCase(
            query="test",
            steps=[
                AgentStep(
                    thought="Let me think",
                    action="think",
                    observation="Interesting",
                ),
            ],
        )
        trace = case.format_trace()
        assert "Thought: Let me think" in trace
        assert "Observation: Interesting" in trace

    def test_format_trace_no_final_output(self):
        case = AgentTestCase(query="test", steps=[], final_output=None)
        trace = case.format_trace()
        assert "Final Output" not in trace


class TestValidateToolCalls:
    def test_all_correct(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="search", parameters={"q": "hello"})),
                AgentStep(tool_call=ToolCall(name="format", parameters={"style": "md"})),
            ],
            expected_tools=[
                ToolCall(name="search", parameters={"q": "hello"}),
                ToolCall(name="format", parameters={"style": "md"}),
            ],
        )
        result = validate_tool_calls(case)
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "tool_call_validation"

    def test_partial_match(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="search", parameters={"q": "hello"})),
            ],
            expected_tools=[
                ToolCall(name="search", parameters={"q": "hello"}),
                ToolCall(name="format", parameters={"style": "md"}),
            ],
        )
        result = validate_tool_calls(case)
        assert result.passed is False
        # 1 correct / max(2 expected, 1 actual) = 1/2 = 0.5
        assert result.score == 0.5
        assert result.cost == 0.0
        assert result.metric_name == "tool_call_validation"

    def test_no_match(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="unrelated", parameters={})),
            ],
            expected_tools=[
                ToolCall(name="search", parameters={"q": "hello"}),
            ],
        )
        result = validate_tool_calls(case)
        assert result.passed is False
        assert result.score == 0.0
        assert result.cost == 0.0

    def test_no_expected_tools(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="search", parameters={})),
            ],
            expected_tools=None,
        )
        result = validate_tool_calls(case)
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert "No expected tool calls" in result.reasoning

    def test_extra_actual_tools(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="search", parameters={"q": "hi"})),
                AgentStep(tool_call=ToolCall(name="extra_tool", parameters={})),
            ],
            expected_tools=[
                ToolCall(name="search", parameters={"q": "hi"}),
            ],
        )
        result = validate_tool_calls(case)
        assert result.passed is False
        # 1 correct / max(1 expected, 2 actual) = 1/2 = 0.5
        assert result.score == 0.5
        assert result.cost == 0.0
        assert "Unexpected tool calls" in result.reasoning


class TestValidateTrajectoryLength:
    def test_within_limit(self):
        case = AgentTestCase(
            query="q",
            steps=[AgentStep(action="a"), AgentStep(action="b")],
        )
        result = validate_trajectory_length(case, max_steps=5)
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "trajectory_length"

    def test_exceeds_limit(self):
        case = AgentTestCase(
            query="q",
            steps=[AgentStep(action="a") for _ in range(10)],
        )
        result = validate_trajectory_length(case, max_steps=3)
        assert result.passed is False
        # score = min(1.0, 3/10) = 0.3
        assert result.score == pytest.approx(0.3)
        assert result.cost == 0.0
        assert "Exceeds" in result.reasoning

    def test_exact_limit(self):
        case = AgentTestCase(
            query="q",
            steps=[AgentStep(action="a") for _ in range(5)],
        )
        result = validate_trajectory_length(case, max_steps=5)
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert "Within" in result.reasoning

    def test_zero_steps(self):
        case = AgentTestCase(query="q", steps=[])
        result = validate_trajectory_length(case, max_steps=5)
        assert result.passed is True
        assert result.score == 1.0


class TestValidateToolOrder:
    def test_correct_order(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="search", parameters={})),
                AgentStep(tool_call=ToolCall(name="parse", parameters={})),
                AgentStep(tool_call=ToolCall(name="format", parameters={})),
            ],
        )
        result = validate_tool_order(case, expected_order=["search", "parse", "format"])
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "tool_order"

    def test_wrong_order(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="format", parameters={})),
                AgentStep(tool_call=ToolCall(name="search", parameters={})),
            ],
        )
        result = validate_tool_order(case, expected_order=["search", "format"])
        # LCS of ["format","search"] vs ["search","format"] = 1 ("format" or "search")
        assert result.passed is False
        assert result.score == pytest.approx(0.5)
        assert result.cost == 0.0

    def test_with_extras_between(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="search", parameters={})),
                AgentStep(tool_call=ToolCall(name="log", parameters={})),
                AgentStep(tool_call=ToolCall(name="format", parameters={})),
            ],
        )
        result = validate_tool_order(case, expected_order=["search", "format"])
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0

    def test_empty_expected(self):
        case = AgentTestCase(
            query="q",
            steps=[AgentStep(tool_call=ToolCall(name="search", parameters={}))],
        )
        result = validate_tool_order(case, expected_order=[])
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert "No expected order" in result.reasoning


class TestValidateNoRepeatedTools:
    def test_no_repeats(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="search", parameters={})),
                AgentStep(tool_call=ToolCall(name="format", parameters={})),
            ],
        )
        result = validate_no_repeated_tools(case)
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "no_repeated_tools"

    def test_has_repeats(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="search", parameters={})),
                AgentStep(tool_call=ToolCall(name="search", parameters={})),
                AgentStep(tool_call=ToolCall(name="format", parameters={})),
            ],
        )
        result = validate_no_repeated_tools(case)
        assert result.passed is False
        # 1 violator out of 2 unique tools -> score = (2-1)/2 = 0.5
        assert result.score == 0.5
        assert result.cost == 0.0
        assert "exceeded" in result.reasoning

    def test_custom_max_repeats(self):
        case = AgentTestCase(
            query="q",
            steps=[
                AgentStep(tool_call=ToolCall(name="search", parameters={})),
                AgentStep(tool_call=ToolCall(name="search", parameters={})),
                AgentStep(tool_call=ToolCall(name="search", parameters={})),
                AgentStep(tool_call=ToolCall(name="format", parameters={})),
            ],
        )
        # Allow up to 3 repeats -- search is called 3 times which is exactly max_repeats
        result = validate_no_repeated_tools(case, max_repeats=3)
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0

    def test_no_tool_calls(self):
        case = AgentTestCase(query="q", steps=[])
        result = validate_no_repeated_tools(case)
        assert result.passed is True
        assert result.score == 1.0
        assert "No tool calls" in result.reasoning

"""Tests for the trajectory and trace evaluation module."""

import pytest

from checkllm.agents import AgentStep, AgentTestCase, ToolCall
from checkllm.trajectory import (
    TraceSpan,
    TraceValidator,
    TrajectoryValidator,
    validate_trace,
    validate_trajectory,
)


def _make_step(
    tool_name: str | None = None,
    params: dict | None = None,
    result: str | None = None,
    thought: str | None = None,
    action: str | None = None,
    observation: str | None = None,
) -> AgentStep:
    """Create an AgentStep with optional tool call for test convenience."""
    tc = None
    if tool_name is not None:
        tc = ToolCall(name=tool_name, parameters=params or {}, result=result)
    return AgentStep(
        thought=thought,
        action=action or ("call_tool" if tc else "think"),
        tool_call=tc,
        observation=observation,
    )


def _make_case(
    steps: list[AgentStep] | None = None,
    final_output: str | None = None,
    metadata: dict | None = None,
) -> AgentTestCase:
    """Create an AgentTestCase for test convenience."""
    return AgentTestCase(
        query="test query",
        steps=steps or [],
        final_output=final_output,
        metadata=metadata or {},
    )


class TestToolUsed:
    def test_tool_present(self):
        case = _make_case([_make_step("search"), _make_step("format")])
        v = TrajectoryValidator(case)
        r = v.tool_used("search")
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "tool_used"

    def test_tool_absent(self):
        case = _make_case([_make_step("search")])
        v = TrajectoryValidator(case)
        r = v.tool_used("format")
        assert r.passed is False
        assert r.score == 0.0

    def test_min_count_satisfied(self):
        case = _make_case([_make_step("search"), _make_step("search")])
        v = TrajectoryValidator(case)
        r = v.tool_used("search", min_count=2)
        assert r.passed is True
        assert r.score == 1.0

    def test_min_count_not_satisfied(self):
        case = _make_case([_make_step("search")])
        v = TrajectoryValidator(case)
        r = v.tool_used("search", min_count=3)
        assert r.passed is False
        assert r.score == pytest.approx(1 / 3)

    def test_max_count_exceeded(self):
        case = _make_case([_make_step("search")] * 5)
        v = TrajectoryValidator(case)
        r = v.tool_used("search", min_count=1, max_count=3)
        assert r.passed is False
        assert r.score == 1.0  # min_count satisfied

    def test_max_count_within(self):
        case = _make_case([_make_step("search")] * 2)
        v = TrajectoryValidator(case)
        r = v.tool_used("search", min_count=1, max_count=5)
        assert r.passed is True

    def test_empty_trajectory(self):
        case = _make_case([])
        v = TrajectoryValidator(case)
        r = v.tool_used("search")
        assert r.passed is False
        assert r.score == 0.0


class TestToolNotUsed:
    def test_tool_absent(self):
        case = _make_case([_make_step("search")])
        v = TrajectoryValidator(case)
        r = v.tool_not_used("delete_db")
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "tool_not_used"

    def test_tool_present(self):
        case = _make_case([_make_step("delete_db")])
        v = TrajectoryValidator(case)
        r = v.tool_not_used("delete_db")
        assert r.passed is False
        assert r.score == 0.0

    def test_empty_trajectory(self):
        case = _make_case([])
        v = TrajectoryValidator(case)
        r = v.tool_not_used("anything")
        assert r.passed is True


class TestToolArgsMatch:
    def test_partial_match_pass(self):
        case = _make_case([
            _make_step("search", params={"query": "weather", "limit": 10}),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_args_match("search", {"query": "weather"})
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "tool_args_match"

    def test_partial_match_fail(self):
        case = _make_case([
            _make_step("search", params={"query": "sports"}),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_args_match("search", {"query": "weather"})
        assert r.passed is False
        assert r.score == 0.0

    def test_strict_match_pass(self):
        case = _make_case([
            _make_step("search", params={"query": "weather", "limit": 10}),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_args_match(
            "search", {"query": "weather", "limit": 10}, strict=True
        )
        assert r.passed is True
        assert r.score == 1.0

    def test_strict_match_extra_keys_fail(self):
        case = _make_case([
            _make_step("search", params={"query": "weather", "limit": 10}),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_args_match("search", {"query": "weather"}, strict=True)
        assert r.passed is False

    def test_strict_match_missing_keys_fail(self):
        case = _make_case([
            _make_step("search", params={"query": "weather"}),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_args_match(
            "search", {"query": "weather", "limit": 10}, strict=True
        )
        assert r.passed is False

    def test_tool_not_found(self):
        case = _make_case([_make_step("search")])
        v = TrajectoryValidator(case)
        r = v.tool_args_match("format", {"style": "md"})
        assert r.passed is False
        assert r.score == 0.0

    def test_multiple_calls_best_match(self):
        case = _make_case([
            _make_step("search", params={"query": "wrong"}),
            _make_step("search", params={"query": "weather", "limit": 5}),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_args_match("search", {"query": "weather"})
        assert r.passed is True
        assert r.score == 1.0


class TestToolSequence:
    def test_exact_match(self):
        case = _make_case([
            _make_step("search"),
            _make_step("parse"),
            _make_step("format"),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_sequence(["search", "parse", "format"])
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "tool_sequence"

    def test_subsequence_with_gaps(self):
        case = _make_case([
            _make_step("search"),
            _make_step("log"),
            _make_step("format"),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_sequence(["search", "format"])
        assert r.passed is True
        assert r.score == 1.0

    def test_wrong_order(self):
        case = _make_case([
            _make_step("format"),
            _make_step("search"),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_sequence(["search", "format"])
        # Only one of two can be matched as a subsequence
        assert r.passed is False
        assert r.score == 0.5

    def test_empty_expected(self):
        case = _make_case([_make_step("search")])
        v = TrajectoryValidator(case)
        r = v.tool_sequence([])
        assert r.passed is True
        assert r.score == 1.0

    def test_empty_trajectory(self):
        case = _make_case([])
        v = TrajectoryValidator(case)
        r = v.tool_sequence(["search"])
        assert r.passed is False
        assert r.score == 0.0

    def test_partial_match(self):
        case = _make_case([_make_step("search"), _make_step("format")])
        v = TrajectoryValidator(case)
        # Expected 3 tools but actual only has 2.  Greedy subsequence:
        # "search" matches at index 0, "parse" not found (exhausts haystack),
        # "format" cannot match because pointer already past end.
        r = v.tool_sequence(["search", "parse", "format"])
        assert r.passed is False
        assert r.score == pytest.approx(1 / 3)


class TestToolSequenceStrict:
    def test_exact_match(self):
        case = _make_case([
            _make_step("search"),
            _make_step("format"),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_sequence_strict(["search", "format"])
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "tool_sequence_strict"

    def test_extras_fail(self):
        case = _make_case([
            _make_step("search"),
            _make_step("log"),
            _make_step("format"),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_sequence_strict(["search", "format"])
        assert r.passed is False

    def test_wrong_order(self):
        case = _make_case([
            _make_step("format"),
            _make_step("search"),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_sequence_strict(["search", "format"])
        assert r.passed is False
        assert r.score == 0.0

    def test_both_empty(self):
        case = _make_case([])
        v = TrajectoryValidator(case)
        r = v.tool_sequence_strict([])
        assert r.passed is True
        assert r.score == 1.0

    def test_expected_empty_actual_not(self):
        case = _make_case([_make_step("search")])
        v = TrajectoryValidator(case)
        r = v.tool_sequence_strict([])
        assert r.passed is False


class TestStepCount:
    def test_within_range(self):
        case = _make_case([_make_step("a")] * 5)
        v = TrajectoryValidator(case)
        r = v.step_count(min_steps=2, max_steps=10)
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "step_count"

    def test_below_min(self):
        case = _make_case([_make_step("a")])
        v = TrajectoryValidator(case)
        r = v.step_count(min_steps=3)
        assert r.passed is False

    def test_above_max(self):
        case = _make_case([_make_step("a")] * 10)
        v = TrajectoryValidator(case)
        r = v.step_count(max_steps=5)
        assert r.passed is False
        assert r.score == 0.5

    def test_no_bounds(self):
        case = _make_case([_make_step("a")] * 100)
        v = TrajectoryValidator(case)
        r = v.step_count()
        assert r.passed is True
        assert r.score == 1.0

    def test_empty_steps(self):
        case = _make_case([])
        v = TrajectoryValidator(case)
        r = v.step_count(min_steps=1)
        assert r.passed is False


class TestNoRepeatedTools:
    def test_no_repeats(self):
        case = _make_case([_make_step("a"), _make_step("b"), _make_step("c")])
        v = TrajectoryValidator(case)
        r = v.no_repeated_tools()
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "no_repeated_tools"

    def test_consecutive_repeats(self):
        case = _make_case([
            _make_step("a"),
            _make_step("a"),
            _make_step("a"),
            _make_step("b"),
        ])
        v = TrajectoryValidator(case)
        r = v.no_repeated_tools(max_consecutive=1)
        assert r.passed is False
        assert "3" in r.reasoning

    def test_allowed_consecutive(self):
        case = _make_case([_make_step("a"), _make_step("a"), _make_step("b")])
        v = TrajectoryValidator(case)
        r = v.no_repeated_tools(max_consecutive=2)
        assert r.passed is True

    def test_empty_trajectory(self):
        case = _make_case([])
        v = TrajectoryValidator(case)
        r = v.no_repeated_tools()
        assert r.passed is True

    def test_non_consecutive_ok(self):
        case = _make_case([
            _make_step("a"),
            _make_step("b"),
            _make_step("a"),
        ])
        v = TrajectoryValidator(case)
        r = v.no_repeated_tools(max_consecutive=1)
        assert r.passed is True


class TestThoughtContains:
    def test_pattern_found(self):
        case = _make_case([
            _make_step(thought="I need to search for weather data"),
            _make_step(thought="Now I should format the response"),
        ])
        v = TrajectoryValidator(case)
        r = v.thought_contains(r"weather")
        assert r.passed is True
        assert r.metric_name == "thought_contains"

    def test_pattern_not_found(self):
        case = _make_case([
            _make_step(thought="I need to search for data"),
        ])
        v = TrajectoryValidator(case)
        r = v.thought_contains(r"weather")
        assert r.passed is False

    def test_regex_pattern(self):
        case = _make_case([
            _make_step(thought="The temperature is 72F today"),
        ])
        v = TrajectoryValidator(case)
        r = v.thought_contains(r"\d+F")
        assert r.passed is True

    def test_no_thoughts(self):
        case = _make_case([_make_step("search")])
        v = TrajectoryValidator(case)
        r = v.thought_contains("anything")
        assert r.passed is False
        assert r.score == 0.0

    def test_empty_trajectory(self):
        case = _make_case([])
        v = TrajectoryValidator(case)
        r = v.thought_contains("anything")
        assert r.passed is False


class TestNoHallucinatedTools:
    def test_all_allowed(self):
        case = _make_case([_make_step("search"), _make_step("format")])
        v = TrajectoryValidator(case)
        r = v.no_hallucinated_tools(["search", "format", "parse"])
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "no_hallucinated_tools"

    def test_hallucinated_tool(self):
        case = _make_case([
            _make_step("search"),
            _make_step("made_up_tool"),
        ])
        v = TrajectoryValidator(case)
        r = v.no_hallucinated_tools(["search", "format"])
        assert r.passed is False
        assert r.score == 0.5
        assert "made_up_tool" in r.reasoning

    def test_empty_trajectory(self):
        case = _make_case([])
        v = TrajectoryValidator(case)
        r = v.no_hallucinated_tools(["search"])
        assert r.passed is True

    def test_all_hallucinated(self):
        case = _make_case([_make_step("fake1"), _make_step("fake2")])
        v = TrajectoryValidator(case)
        r = v.no_hallucinated_tools(["search"])
        assert r.passed is False
        assert r.score == 0.0


class TestToolResultUsed:
    def test_result_referenced_in_thought(self):
        case = _make_case([
            _make_step("search", result="sunny and warm"),
            _make_step(thought="The search said sunny and warm"),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_result_used("search")
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "tool_result_used"

    def test_result_referenced_in_final_output(self):
        case = _make_case(
            steps=[_make_step("search", result="sunny and warm")],
            final_output="It is sunny and warm today.",
        )
        v = TrajectoryValidator(case)
        r = v.tool_result_used("search")
        assert r.passed is True

    def test_result_not_referenced(self):
        case = _make_case([
            _make_step("search", result="sunny and warm"),
            _make_step(thought="I will now respond with something generic"),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_result_used("search")
        assert r.passed is False

    def test_tool_never_called(self):
        case = _make_case([_make_step("format")])
        v = TrajectoryValidator(case)
        r = v.tool_result_used("search")
        assert r.passed is False
        assert r.score == 0.0

    def test_no_result_returned(self):
        case = _make_case([_make_step("search")])
        v = TrajectoryValidator(case)
        r = v.tool_result_used("search")
        assert r.passed is True  # Nothing to check

    def test_result_in_observation(self):
        case = _make_case([
            _make_step("search", result="data123"),
            _make_step(observation="Received data123 from search"),
        ])
        v = TrajectoryValidator(case)
        r = v.tool_result_used("search")
        assert r.passed is True


class TestMaxCost:
    def test_within_budget(self):
        case = _make_case(metadata={"total_cost": 0.05})
        v = TrajectoryValidator(case)
        r = v.max_cost(0.10)
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "max_cost"

    def test_over_budget(self):
        case = _make_case(metadata={"total_cost": 0.50})
        v = TrajectoryValidator(case)
        r = v.max_cost(0.10)
        assert r.passed is False
        assert r.score == pytest.approx(0.2)

    def test_no_cost_data(self):
        case = _make_case()
        v = TrajectoryValidator(case)
        r = v.max_cost(1.0)
        assert r.passed is True


class TestMaxLatency:
    def test_within_limit(self):
        case = _make_case(metadata={"total_latency_ms": 500})
        v = TrajectoryValidator(case)
        r = v.max_latency(1000)
        assert r.passed is True
        assert r.metric_name == "max_latency"

    def test_over_limit(self):
        case = _make_case(metadata={"total_latency_ms": 5000})
        v = TrajectoryValidator(case)
        r = v.max_latency(1000)
        assert r.passed is False
        assert r.score == pytest.approx(0.2)


class TestFinalOutputMatches:
    def test_pattern_matches(self):
        case = _make_case(final_output="The weather is sunny today.")
        v = TrajectoryValidator(case)
        r = v.final_output_matches(r"sunny")
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "final_output_matches"

    def test_pattern_no_match(self):
        case = _make_case(final_output="The weather is rainy.")
        v = TrajectoryValidator(case)
        r = v.final_output_matches(r"sunny")
        assert r.passed is False
        assert r.score == 0.0

    def test_regex_pattern(self):
        case = _make_case(final_output="Temperature: 72F")
        v = TrajectoryValidator(case)
        r = v.final_output_matches(r"\d+F")
        assert r.passed is True

    def test_no_final_output(self):
        case = _make_case(final_output=None)
        v = TrajectoryValidator(case)
        r = v.final_output_matches(r"anything")
        assert r.passed is False


class TestTrajectoryEfficiency:
    def test_efficient(self):
        case = _make_case([_make_step("search"), _make_step("format")])
        v = TrajectoryValidator(case)
        r = v.trajectory_efficiency(max_steps_per_tool=2.0)
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "trajectory_efficiency"

    def test_inefficient(self):
        steps = [_make_step("search")] + [_make_step(thought="thinking")] * 8
        case = _make_case(steps)
        v = TrajectoryValidator(case)
        r = v.trajectory_efficiency(max_steps_per_tool=2.0)
        assert r.passed is False
        # 9 total / 1 tool call = 9.0, way above 2.0

    def test_empty_trajectory(self):
        case = _make_case([])
        v = TrajectoryValidator(case)
        r = v.trajectory_efficiency()
        assert r.passed is True
        assert r.score == 1.0

    def test_no_tool_calls(self):
        case = _make_case([_make_step(thought="thinking")] * 3)
        v = TrajectoryValidator(case)
        r = v.trajectory_efficiency(max_steps_per_tool=2.0)
        # 3 steps / max(0 tools, 1) = 3.0 > 2.0
        assert r.passed is False


class TestValidateTrajectoryHelper:
    def test_returns_validator(self):
        case = _make_case()
        v = validate_trajectory(case)
        assert isinstance(v, TrajectoryValidator)
        assert v.test_case is case


class TestTraceSpan:
    def test_duration(self):
        span = TraceSpan(
            name="llm_call",
            span_type="llm",
            start_ms=100,
            end_ms=300,
        )
        assert span.duration_ms == 200

    def test_defaults(self):
        span = TraceSpan(
            name="test", span_type="custom", start_ms=0, end_ms=10
        )
        assert span.status == "ok"
        assert span.attributes == {}
        assert span.children == []


class TestTraceValidatorSpanCount:
    def test_count_all(self):
        spans = [
            TraceSpan(name="a", span_type="llm", start_ms=0, end_ms=10),
            TraceSpan(name="b", span_type="tool", start_ms=10, end_ms=20),
        ]
        v = TraceValidator(spans)
        r = v.span_count(min_count=2)
        assert r.passed is True
        assert r.metric_name == "span_count"

    def test_count_by_type(self):
        spans = [
            TraceSpan(name="a", span_type="llm", start_ms=0, end_ms=10),
            TraceSpan(name="b", span_type="tool", start_ms=10, end_ms=20),
            TraceSpan(name="c", span_type="llm", start_ms=20, end_ms=30),
        ]
        v = TraceValidator(spans)
        r = v.span_count(span_type="llm", min_count=2)
        assert r.passed is True

    def test_count_by_name_pattern(self):
        spans = [
            TraceSpan(name="retrieve_docs", span_type="retriever", start_ms=0, end_ms=10),
            TraceSpan(name="retrieve_meta", span_type="retriever", start_ms=10, end_ms=20),
            TraceSpan(name="format", span_type="tool", start_ms=20, end_ms=30),
        ]
        v = TraceValidator(spans)
        r = v.span_count(name_pattern=r"retrieve_.*", min_count=2)
        assert r.passed is True

    def test_max_count_exceeded(self):
        spans = [
            TraceSpan(name="a", span_type="llm", start_ms=0, end_ms=10),
        ] * 5
        v = TraceValidator(spans)
        r = v.span_count(max_count=3)
        assert r.passed is False

    def test_includes_children(self):
        child = TraceSpan(name="child", span_type="tool", start_ms=5, end_ms=8)
        parent = TraceSpan(
            name="parent", span_type="agent", start_ms=0, end_ms=10,
            children=[child],
        )
        v = TraceValidator([parent])
        r = v.span_count(min_count=2)
        assert r.passed is True

    def test_empty_spans(self):
        v = TraceValidator([])
        r = v.span_count(min_count=0)
        assert r.passed is True


class TestTraceValidatorSpanDuration:
    def test_all_within_limit(self):
        spans = [
            TraceSpan(name="call_a", span_type="llm", start_ms=0, end_ms=50),
            TraceSpan(name="call_b", span_type="llm", start_ms=50, end_ms=90),
        ]
        v = TraceValidator(spans)
        r = v.span_duration("call_.*", max_ms=100)
        assert r.passed is True
        assert r.metric_name == "span_duration"

    def test_some_exceed(self):
        spans = [
            TraceSpan(name="call_a", span_type="llm", start_ms=0, end_ms=50),
            TraceSpan(name="call_b", span_type="llm", start_ms=50, end_ms=200),
        ]
        v = TraceValidator(spans)
        r = v.span_duration("call_.*", max_ms=100, percentile=1.0)
        assert r.passed is False

    def test_percentile_check(self):
        spans = [
            TraceSpan(name="op", span_type="tool", start_ms=0, end_ms=50),
            TraceSpan(name="op", span_type="tool", start_ms=50, end_ms=90),
            TraceSpan(name="op", span_type="tool", start_ms=100, end_ms=300),
        ]
        v = TraceValidator(spans)
        # 2 out of 3 are within 100ms => 66.7% >= 50% required
        r = v.span_duration("op", max_ms=100, percentile=0.5)
        assert r.passed is True

    def test_no_matching_spans(self):
        spans = [
            TraceSpan(name="other", span_type="llm", start_ms=0, end_ms=10),
        ]
        v = TraceValidator(spans)
        r = v.span_duration("nonexistent", max_ms=100)
        assert r.passed is True


class TestTraceValidatorNoErrorSpans:
    def test_no_errors(self):
        spans = [
            TraceSpan(name="a", span_type="llm", start_ms=0, end_ms=10, status="ok"),
            TraceSpan(name="b", span_type="tool", start_ms=10, end_ms=20, status="ok"),
        ]
        v = TraceValidator(spans)
        r = v.no_error_spans()
        assert r.passed is True
        assert r.score == 1.0
        assert r.metric_name == "no_error_spans"

    def test_has_errors(self):
        spans = [
            TraceSpan(name="a", span_type="llm", start_ms=0, end_ms=10, status="ok"),
            TraceSpan(name="b", span_type="tool", start_ms=10, end_ms=20, status="error"),
        ]
        v = TraceValidator(spans)
        r = v.no_error_spans()
        assert r.passed is False
        assert r.score == 0.5
        assert "b" in r.reasoning

    def test_nested_error(self):
        child = TraceSpan(
            name="bad_child", span_type="tool", start_ms=5, end_ms=8,
            status="error",
        )
        parent = TraceSpan(
            name="parent", span_type="agent", start_ms=0, end_ms=10,
            children=[child],
        )
        v = TraceValidator([parent])
        r = v.no_error_spans()
        assert r.passed is False
        assert "bad_child" in r.reasoning

    def test_empty_spans(self):
        v = TraceValidator([])
        r = v.no_error_spans()
        assert r.passed is True


class TestTraceValidatorTotalDuration:
    def test_within_limit(self):
        spans = [
            TraceSpan(name="a", span_type="llm", start_ms=0, end_ms=100),
            TraceSpan(name="b", span_type="tool", start_ms=100, end_ms=200),
        ]
        v = TraceValidator(spans)
        r = v.total_duration(max_ms=500)
        assert r.passed is True
        assert r.metric_name == "total_duration"

    def test_exceeds_limit(self):
        spans = [
            TraceSpan(name="a", span_type="llm", start_ms=0, end_ms=500),
            TraceSpan(name="b", span_type="tool", start_ms=500, end_ms=1500),
        ]
        v = TraceValidator(spans)
        r = v.total_duration(max_ms=1000)
        assert r.passed is False
        # Duration is 1500, limit is 1000
        assert r.score == pytest.approx(1000 / 1500)

    def test_empty_spans(self):
        v = TraceValidator([])
        r = v.total_duration(max_ms=100)
        assert r.passed is True

    def test_overlapping_spans(self):
        spans = [
            TraceSpan(name="a", span_type="llm", start_ms=0, end_ms=300),
            TraceSpan(name="b", span_type="tool", start_ms=100, end_ms=200),
        ]
        v = TraceValidator(spans)
        r = v.total_duration(max_ms=300)
        assert r.passed is True  # Total = 300 - 0 = 300


class TestValidateTraceHelper:
    def test_returns_validator(self):
        spans = [
            TraceSpan(name="a", span_type="llm", start_ms=0, end_ms=10),
        ]
        v = validate_trace(spans)
        assert isinstance(v, TraceValidator)
        assert v.spans is spans


class TestIntegrationWorkflow:
    """End-to-end test simulating a realistic agent evaluation."""

    def test_full_agent_validation(self):
        case = AgentTestCase(
            query="What is the weather in San Francisco?",
            steps=[
                AgentStep(
                    thought="I need to search for weather data.",
                    action="call_tool",
                    tool_call=ToolCall(
                        name="search",
                        parameters={"query": "weather SF"},
                        result="72F and sunny",
                    ),
                ),
                AgentStep(
                    thought="I got the result: 72F and sunny. Let me format it.",
                    action="call_tool",
                    tool_call=ToolCall(
                        name="format",
                        parameters={"style": "friendly"},
                        result="It's a beautiful 72F and sunny day!",
                    ),
                ),
            ],
            final_output="It's a beautiful 72F and sunny day in San Francisco!",
            metadata={"total_cost": 0.003, "total_latency_ms": 450},
        )

        v = validate_trajectory(case)

        assert v.tool_used("search").passed is True
        assert v.tool_used("format").passed is True
        assert v.tool_not_used("delete").passed is True
        assert v.tool_sequence(["search", "format"]).passed is True
        assert v.tool_sequence_strict(["search", "format"]).passed is True
        assert v.tool_args_match("search", {"query": "weather SF"}).passed is True
        assert v.step_count(min_steps=1, max_steps=5).passed is True
        assert v.no_repeated_tools(max_consecutive=1).passed is True
        assert v.thought_contains(r"weather").passed is True
        assert v.no_hallucinated_tools(["search", "format"]).passed is True
        assert v.tool_result_used("search").passed is True
        assert v.max_cost(0.01).passed is True
        assert v.max_latency(1000).passed is True
        assert v.final_output_matches(r"72F").passed is True
        assert v.trajectory_efficiency(max_steps_per_tool=2.0).passed is True

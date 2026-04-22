from __future__ import annotations

import pytest

from checkllm.metrics.dual_judge_nv import (
    NVAnswerAccuracyMetric,
    NVContextRelevanceMetric,
    NVResponseGroundednessMetric,
)
from checkllm.metrics.image_editing import ImageEditingMetric
from checkllm.metrics.image_reference import ImageReferenceMetric
from checkllm.metrics.per_turn import (
    TurnCoherenceMetric,
    TurnFaithfulnessMetric,
    TurnRelevancyMetric,
)
from checkllm.metrics.prompt_alignment import PromptAlignmentMetric
from checkllm.metrics.tool_call_f1 import ToolCallF1Metric
from checkllm.metrics.trajectory import (
    TrajectoryGoalSuccessMetric,
    TrajectoryStepCountMetric,
    TrajectoryToolArgsMatchMetric,
    TrajectoryToolSequenceMetric,
)
from checkllm.testing import MockJudge


class TestTrajectoryGoalSuccessMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_goal_achieved(self, judge):
        judge.set_default(score=0.95, reasoning="Agent fully achieved the goal")
        metric = TrajectoryGoalSuccessMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            output="Successfully booked flight to NYC for Dec 25",
            goal_description="Book a flight to New York for December 25th",
            trajectory=["Search flights", "Select cheapest option", "Confirm booking"],
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "trajectory_goal_success"

    @pytest.mark.asyncio
    async def test_goal_not_achieved(self, judge):
        judge.set_default(score=0.2, reasoning="Agent failed to achieve the goal")
        metric = TrajectoryGoalSuccessMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            output="Error: no flights available",
            goal_description="Book a flight to New York for December 25th",
            trajectory=["Search flights", "Error encountered"],
        )
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_prompt_contains_all_inputs(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = TrajectoryGoalSuccessMetric(judge=judge, threshold=0.7)
        await metric.evaluate(
            output="done",
            goal_description="test goal",
            trajectory=["step one", "step two"],
        )
        prompt = judge.calls[-1]["prompt"]
        assert "test goal" in prompt
        assert "done" in prompt
        assert "step one" in prompt
        assert "step two" in prompt


class TestTrajectoryToolSequenceMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_exact_match(self, judge):
        metric = TrajectoryToolSequenceMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            actual_tool_sequence=["search", "filter", "save"],
            expected_tool_sequence=["search", "filter", "save"],
        )
        assert result.passed is True
        assert result.score == 1.0
        assert result.metric_name == "trajectory_tool_sequence"
        assert len(judge.calls) == 0

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self, judge):
        metric = TrajectoryToolSequenceMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            actual_tool_sequence=["Search", "Filter", "Save"],
            expected_tool_sequence=["search", "filter", "save"],
        )
        assert result.passed is True
        assert result.score == 1.0
        assert len(judge.calls) == 0

    @pytest.mark.asyncio
    async def test_different_sequences_uses_judge(self, judge):
        judge.set_default(score=0.6, reasoning="Partially matching sequences")
        metric = TrajectoryToolSequenceMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            actual_tool_sequence=["search", "save"],
            expected_tool_sequence=["search", "filter", "save"],
        )
        assert result.passed is False
        assert result.score == 0.6
        assert len(judge.calls) == 1

    @pytest.mark.asyncio
    async def test_completely_different_sequences(self, judge):
        judge.set_default(score=0.1, reasoning="No overlap")
        metric = TrajectoryToolSequenceMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            actual_tool_sequence=["delete", "undo"],
            expected_tool_sequence=["search", "filter", "save"],
        )
        assert result.passed is False
        assert result.score == 0.1


class TestTrajectoryStepCountMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_within_max_steps(self, judge):
        judge.set_default(score=1.0, reasoning="All steps necessary")
        metric = TrajectoryStepCountMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            trajectory=["step 1", "step 2", "step 3"],
            max_steps=5,
        )
        assert result.passed is True
        assert result.score == 1.0
        assert result.metric_name == "trajectory_step_count"

    @pytest.mark.asyncio
    async def test_exceeds_max_steps(self, judge):
        judge.set_default(score=0.3, reasoning="Several unnecessary steps")
        metric = TrajectoryStepCountMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            trajectory=["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
            max_steps=3,
        )
        assert result.passed is False
        assert result.metric_name == "trajectory_step_count"

    @pytest.mark.asyncio
    async def test_combined_score_calculation(self, judge):
        judge.set_default(score=0.8, reasoning="Mostly necessary steps")
        metric = TrajectoryStepCountMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            trajectory=["s1", "s2", "s3"],
            max_steps=3,
        )
        expected_combined = (1.0 + 0.8) / 2.0
        assert result.score == expected_combined


class TestTrajectoryToolArgsMatchMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_matching_args(self, judge):
        judge.set_default(score=0.95, reasoning="All arguments match")
        metric = TrajectoryToolArgsMatchMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            actual_tool_calls=[
                {"name": "search", "args": {"query": "flights to NYC"}},
            ],
            expected_tool_calls=[
                {"name": "search", "args": {"query": "flights to NYC"}},
            ],
        )
        assert result.passed is True
        assert result.score == 0.95
        assert result.metric_name == "trajectory_tool_args_match"

    @pytest.mark.asyncio
    async def test_mismatched_args(self, judge):
        judge.set_default(score=0.3, reasoning="Arguments are different")
        metric = TrajectoryToolArgsMatchMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            actual_tool_calls=[
                {"name": "search", "args": {"query": "hotels in LA"}},
            ],
            expected_tool_calls=[
                {"name": "search", "args": {"query": "flights to NYC"}},
            ],
        )
        assert result.passed is False
        assert result.score == 0.3


class TestTurnRelevancyMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_all_turns_relevant(self, judge):
        judge.set_default(score=0.9, reasoning="Response is relevant")
        metric = TurnRelevancyMetric(judge=judge, threshold=0.7)
        turns = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "What is it used for?"},
            {"role": "assistant", "content": "It is used for web dev, ML, and more."},
        ]
        result = await metric.evaluate(turns=turns)
        assert result.passed is True
        assert result.score == 0.9
        assert result.metric_name == "turn_relevancy"
        assert len(judge.calls) == 2

    @pytest.mark.asyncio
    async def test_per_turn_scores_in_reasoning(self, judge):
        judge.set_default(score=0.8, reasoning="Relevant")
        metric = TurnRelevancyMetric(judge=judge, threshold=0.7)
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = await metric.evaluate(turns=turns)
        assert "Per-turn relevancy scores" in result.reasoning
        assert "0.8" in result.reasoning

    @pytest.mark.asyncio
    async def test_empty_turns(self, judge):
        metric = TurnRelevancyMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(turns=[])
        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_aggregate_of_mixed_scores(self, judge):
        call_count = 0

        async def mock_evaluate(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1
            from checkllm.models import JudgeResponse

            if call_count == 1:
                return JudgeResponse(score=1.0, reasoning="Perfect", cost=0.0)
            return JudgeResponse(score=0.4, reasoning="Not great", cost=0.0)

        judge.evaluate = mock_evaluate
        metric = TurnRelevancyMetric(judge=judge, threshold=0.7)
        turns = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        result = await metric.evaluate(turns=turns)
        assert result.score == pytest.approx(0.7, abs=0.01)


class TestTurnFaithfulnessMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_faithful_turns(self, judge):
        judge.set_default(score=0.85, reasoning="Faithful to context")
        metric = TurnFaithfulnessMetric(judge=judge, threshold=0.7)
        turns = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a language."},
        ]
        contexts = ["Python is a programming language created by Guido."]
        result = await metric.evaluate(turns=turns, contexts=contexts)
        assert result.passed is True
        assert result.score == 0.85
        assert result.metric_name == "turn_faithfulness"

    @pytest.mark.asyncio
    async def test_unfaithful_turns(self, judge):
        judge.set_default(score=0.2, reasoning="Not faithful")
        metric = TurnFaithfulnessMetric(judge=judge, threshold=0.7)
        turns = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python was invented in 2020."},
        ]
        contexts = ["Python is a programming language created in 1991."]
        result = await metric.evaluate(turns=turns, contexts=contexts)
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_context_included_in_prompt(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = TurnFaithfulnessMetric(judge=judge, threshold=0.7)
        turns = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        contexts = ["specific context text here"]
        await metric.evaluate(turns=turns, contexts=contexts)
        prompt = judge.calls[-1]["prompt"]
        assert "specific context text here" in prompt


class TestTurnCoherenceMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_coherent_conversation(self, judge):
        judge.set_default(score=0.9, reasoning="Coherent flow")
        metric = TurnCoherenceMetric(judge=judge, threshold=0.7)
        turns = [
            {"role": "user", "content": "Tell me about Python."},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "What are its features?"},
            {"role": "assistant", "content": "It has dynamic typing and GC."},
        ]
        result = await metric.evaluate(turns=turns)
        assert result.passed is True
        assert result.score == 0.9
        assert result.metric_name == "turn_coherence"

    @pytest.mark.asyncio
    async def test_incoherent_conversation(self, judge):
        judge.set_default(score=0.2, reasoning="Incoherent")
        metric = TurnCoherenceMetric(judge=judge, threshold=0.7)
        turns = [
            {"role": "user", "content": "Tell me about Python."},
            {"role": "assistant", "content": "The weather is nice today."},
        ]
        result = await metric.evaluate(turns=turns)
        assert result.passed is False
        assert result.score == 0.2


class TestNVContextRelevanceMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_dual_judge_averages_scores(self, judge):
        call_count = 0

        async def mock_evaluate(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1
            from checkllm.models import JudgeResponse

            if call_count == 1:
                return JudgeResponse(score=0.8, reasoning="Relevant context", cost=0.01)
            return JudgeResponse(score=0.6, reasoning="Some irrelevant info", cost=0.01)

        judge.evaluate = mock_evaluate
        metric = NVContextRelevanceMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            context="Python is a programming language.",
            query="What is Python?",
        )
        assert result.score == pytest.approx(0.7, abs=0.01)
        assert result.cost == pytest.approx(0.02, abs=0.001)
        assert result.metric_name == "nv_context_relevance"

    @pytest.mark.asyncio
    async def test_both_judges_called(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = NVContextRelevanceMetric(judge=judge, threshold=0.7)
        await metric.evaluate(context="ctx", query="q")
        assert len(judge.calls) == 2

    @pytest.mark.asyncio
    async def test_passes_when_above_threshold(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = NVContextRelevanceMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(context="ctx", query="q")
        assert result.passed is True


class TestNVAnswerAccuracyMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_dual_judge_averages_scores(self, judge):
        call_count = 0

        async def mock_evaluate(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1
            from checkllm.models import JudgeResponse

            if call_count == 1:
                return JudgeResponse(score=0.9, reasoning="Accurate", cost=0.01)
            return JudgeResponse(score=0.7, reasoning="Mostly matches", cost=0.01)

        judge.evaluate = mock_evaluate
        metric = NVAnswerAccuracyMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            response_text="Python is interpreted.",
            reference="Python is an interpreted language.",
        )
        assert result.score == pytest.approx(0.8, abs=0.01)
        assert result.metric_name == "nv_answer_accuracy"

    @pytest.mark.asyncio
    async def test_optional_query(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = NVAnswerAccuracyMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            response_text="answer",
            reference="reference",
            query="What is Python?",
        )
        assert result.passed is True
        prompt_a = judge.calls[0]["prompt"]
        assert "What is Python?" in prompt_a


class TestNVResponseGroundednessMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_dual_judge_averages_scores(self, judge):
        call_count = 0

        async def mock_evaluate(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1
            from checkllm.models import JudgeResponse

            if call_count == 1:
                return JudgeResponse(score=0.8, reasoning="Grounded", cost=0.01)
            return JudgeResponse(score=0.6, reasoning="Some additions", cost=0.01)

        judge.evaluate = mock_evaluate
        metric = NVResponseGroundednessMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            response_text="Python was created in 1991.",
            contexts=["Python is a language created by Guido van Rossum in 1991."],
        )
        assert result.score == pytest.approx(0.7, abs=0.01)
        assert result.metric_name == "nv_response_groundedness"

    @pytest.mark.asyncio
    async def test_multiple_contexts_formatted(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = NVResponseGroundednessMetric(judge=judge, threshold=0.7)
        await metric.evaluate(
            response_text="answer",
            contexts=["context one", "context two"],
        )
        prompt_a = judge.calls[0]["prompt"]
        assert "Context 1:" in prompt_a
        assert "Context 2:" in prompt_a
        assert "context one" in prompt_a
        assert "context two" in prompt_a


class TestToolCallF1Metric:
    @pytest.mark.asyncio
    async def test_perfect_match(self):
        metric = ToolCallF1Metric(threshold=0.7)
        result = await metric.evaluate(
            predicted_tools=["search", "filter", "save"],
            expected_tools=["search", "filter", "save"],
        )
        assert result.passed is True
        assert result.score == 1.0
        assert result.metric_name == "tool_call_f1"
        assert result.cost == 0.0

    @pytest.mark.asyncio
    async def test_partial_overlap(self):
        metric = ToolCallF1Metric(threshold=0.7)
        result = await metric.evaluate(
            predicted_tools=["search", "delete"],
            expected_tools=["search", "filter", "save"],
        )
        precision = 1 / 2
        recall = 1 / 3
        expected_f1 = 2 * precision * recall / (precision + recall)
        assert result.score == pytest.approx(expected_f1, abs=0.01)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_no_overlap(self):
        metric = ToolCallF1Metric(threshold=0.7)
        result = await metric.evaluate(
            predicted_tools=["x", "y"],
            expected_tools=["a", "b"],
        )
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_empty_predictions(self):
        metric = ToolCallF1Metric(threshold=0.7)
        result = await metric.evaluate(
            predicted_tools=[],
            expected_tools=["search", "filter"],
        )
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_empty_expected(self):
        metric = ToolCallF1Metric(threshold=0.7)
        result = await metric.evaluate(
            predicted_tools=["search"],
            expected_tools=[],
        )
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_both_empty(self):
        metric = ToolCallF1Metric(threshold=0.7)
        result = await metric.evaluate(
            predicted_tools=[],
            expected_tools=[],
        )
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_reasoning_contains_precision_recall(self):
        metric = ToolCallF1Metric(threshold=0.7)
        result = await metric.evaluate(
            predicted_tools=["search", "save"],
            expected_tools=["search", "filter"],
        )
        assert "Precision:" in result.reasoning
        assert "Recall:" in result.reasoning
        assert "F1:" in result.reasoning
        assert "Missing:" in result.reasoning
        assert "Extra:" in result.reasoning


class TestPromptAlignmentMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_all_instructions_followed(self, judge):
        judge.set_default(score=1.0, reasoning="Instruction followed")
        metric = PromptAlignmentMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            output="Hello! I am a helpful assistant. How can I help?",
            system_prompt="You are a helpful assistant.",
            instructions=[
                "Greet the user",
                "Offer help",
            ],
        )
        assert result.passed is True
        assert result.score == 1.0
        assert result.metric_name == "prompt_alignment"
        assert len(judge.calls) == 2

    @pytest.mark.asyncio
    async def test_no_instructions_followed(self, judge):
        judge.set_default(score=0.0, reasoning="Instruction not followed")
        metric = PromptAlignmentMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            output="I refuse to help.",
            system_prompt="You are a helpful assistant.",
            instructions=["Greet the user", "Offer help"],
        )
        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_partial_instruction_following(self, judge):
        call_count = 0

        async def mock_evaluate(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1
            from checkllm.models import JudgeResponse

            if call_count == 1:
                return JudgeResponse(score=1.0, reasoning="Followed", cost=0.0)
            return JudgeResponse(score=0.0, reasoning="Not followed", cost=0.0)

        judge.evaluate = mock_evaluate
        metric = PromptAlignmentMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            output="Hello!",
            system_prompt="You are helpful.",
            instructions=["Greet the user", "Provide detailed answer"],
        )
        assert result.score == pytest.approx(0.5, abs=0.01)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_empty_instructions(self, judge):
        metric = PromptAlignmentMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            output="test",
            system_prompt="test",
            instructions=[],
        )
        assert result.score == 0.0
        assert result.passed is False


class TestImageEditingMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_good_edit(self, judge):
        judge.set_default(score=0.9, reasoning="Edit correctly followed instruction")
        metric = ImageEditingMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            editing_instruction="Change the sky color to sunset orange",
            original_image_desc="A landscape with blue sky and green hills",
            edited_image_desc="A landscape with orange sunset sky and green hills",
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.metric_name == "image_editing"

    @pytest.mark.asyncio
    async def test_bad_edit(self, judge):
        judge.set_default(score=0.1, reasoning="Edit did not follow instruction")
        metric = ImageEditingMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            editing_instruction="Change the sky color to sunset orange",
            original_image_desc="A landscape with blue sky and green hills",
            edited_image_desc="A landscape with blue sky and purple hills",
        )
        assert result.passed is False
        assert result.score == 0.1

    @pytest.mark.asyncio
    async def test_prompt_contains_all_inputs(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = ImageEditingMetric(judge=judge, threshold=0.7)
        await metric.evaluate(
            editing_instruction="test instruction",
            original_image_desc="test original",
            edited_image_desc="test edited",
        )
        prompt = judge.calls[-1]["prompt"]
        assert "test instruction" in prompt
        assert "test original" in prompt
        assert "test edited" in prompt


class TestImageReferenceMetric:
    @pytest.fixture
    def judge(self):
        return MockJudge()

    @pytest.mark.asyncio
    async def test_good_match(self, judge):
        judge.set_default(score=0.9, reasoning="Images match closely")
        metric = ImageReferenceMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            generated_image_desc="A red car on a sunny road",
            reference_image_desc="A red vehicle driving on a road in daylight",
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.metric_name == "image_reference"

    @pytest.mark.asyncio
    async def test_poor_match(self, judge):
        judge.set_default(score=0.15, reasoning="Images do not match")
        metric = ImageReferenceMetric(judge=judge, threshold=0.7)
        result = await metric.evaluate(
            generated_image_desc="A blue boat on water",
            reference_image_desc="A red car on a road",
        )
        assert result.passed is False
        assert result.score == 0.15

    @pytest.mark.asyncio
    async def test_prompt_contains_descriptions(self, judge):
        judge.set_default(score=0.9, reasoning="ok")
        metric = ImageReferenceMetric(judge=judge, threshold=0.7)
        await metric.evaluate(
            generated_image_desc="generated desc",
            reference_image_desc="reference desc",
        )
        prompt = judge.calls[-1]["prompt"]
        assert "generated desc" in prompt
        assert "reference desc" in prompt

"""Tests for checkllm.synthesizer — synthetic test case generation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from checkllm.datasets.case import Case
from checkllm.models import JudgeResponse
from checkllm.synthesizer import (
    EvolutionStrategy,
    SynthesisConfig,
    Synthesizer,
    _dict_to_case,
    _extract_json_array,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_json_response(cases: list[dict]) -> JudgeResponse:
    """Build a JudgeResponse whose raw_output is a JSON array of cases."""
    return JudgeResponse(
        score=1.0,
        reasoning="ok",
        raw_output=json.dumps(cases),
        cost=0.001,
    )


_SINGLE_CASE_PAYLOAD = [
    {
        "input": "What is AI?",
        "expected": "AI is artificial intelligence",
        "context": "AI overview",
        "criteria": "accuracy",
        "metadata": {"strategy": "simple", "difficulty": "easy"},
    }
]


# ---------------------------------------------------------------------------
# SynthesisConfig
# ---------------------------------------------------------------------------


class TestSynthesisConfig:
    def test_defaults(self):
        cfg = SynthesisConfig()
        assert cfg.num_cases == 10
        assert cfg.strategies == [EvolutionStrategy.SIMPLE]
        assert cfg.max_retries == 3
        assert cfg.temperature == 0.7

    def test_custom_values(self):
        cfg = SynthesisConfig(
            num_cases=5,
            strategies=[EvolutionStrategy.REASONING, EvolutionStrategy.ADVERSARIAL],
            max_retries=1,
            temperature=0.3,
        )
        assert cfg.num_cases == 5
        assert len(cfg.strategies) == 2
        assert cfg.max_retries == 1
        assert cfg.temperature == 0.3


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestExtractJsonArray:
    def test_plain_json(self):
        items = _extract_json_array('[{"a": 1}]')
        assert items == [{"a": 1}]

    def test_json_with_code_fences(self):
        text = '```json\n[{"a": 1}]\n```'
        items = _extract_json_array(text)
        assert items == [{"a": 1}]

    def test_json_embedded_in_text(self):
        text = 'Here are the results:\n[{"a": 1}]\nDone.'
        items = _extract_json_array(text)
        assert items == [{"a": 1}]

    def test_raises_on_garbage(self):
        with pytest.raises(ValueError, match="Could not extract JSON array"):
            _extract_json_array("not json at all")


class TestDictToCase:
    def test_basic_conversion(self):
        raw = {
            "input": "What is AI?",
            "expected": "Artificial intelligence",
            "context": "tech",
            "criteria": "accuracy",
        }
        case = _dict_to_case(raw)
        assert isinstance(case, Case)
        assert case.input == "What is AI?"
        assert case.expected == "Artificial intelligence"
        assert case.context == "tech"
        assert case.criteria == "accuracy"

    def test_injects_strategy_metadata(self):
        raw = {"input": "q", "expected": "a"}
        case = _dict_to_case(raw, strategy=EvolutionStrategy.ADVERSARIAL)
        assert case.metadata["strategy"] == "adversarial"
        assert case.metadata["difficulty"] == "hard"

    def test_preserves_existing_metadata(self):
        raw = {"input": "q", "metadata": {"strategy": "custom", "difficulty": "easy"}}
        case = _dict_to_case(raw, strategy=EvolutionStrategy.REASONING)
        # Existing values should NOT be overwritten
        assert case.metadata["strategy"] == "custom"
        assert case.metadata["difficulty"] == "easy"


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class TestSynthesizer:
    @pytest.fixture
    def mock_judge(self):
        judge = AsyncMock()
        judge.evaluate.return_value = _make_json_response(_SINGLE_CASE_PAYLOAD)
        return judge

    @pytest.fixture
    def synthesizer(self, mock_judge):
        return Synthesizer(judge=mock_judge, config=SynthesisConfig(num_cases=1))

    # -- from_documents ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_from_documents(self, mock_judge):
        synth = Synthesizer(judge=mock_judge, config=SynthesisConfig(num_cases=2))
        cases = await synth.from_documents(
            documents=["Document about AI"],
            num_cases=1,
            strategies=[EvolutionStrategy.SIMPLE],
        )
        assert len(cases) >= 1
        assert all(isinstance(c, Case) for c in cases)
        mock_judge.evaluate.assert_called()

    @pytest.mark.asyncio
    async def test_from_documents_uses_config_defaults(self, synthesizer, mock_judge):
        cases = await synthesizer.from_documents(documents=["Some text"])
        assert len(cases) >= 1
        # num_cases defaults to 1 from our config fixture

    # -- from_description --------------------------------------------------

    @pytest.mark.asyncio
    async def test_from_description(self, mock_judge):
        synth = Synthesizer(judge=mock_judge, config=SynthesisConfig(num_cases=1))
        cases = await synth.from_description(
            description="Customer support chatbot for e-commerce",
            num_cases=1,
            strategies=[EvolutionStrategy.SIMPLE],
        )
        assert len(cases) >= 1
        assert cases[0].input  # not empty
        mock_judge.evaluate.assert_called()

    # -- evolve ------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_evolve(self, mock_judge):
        synth = Synthesizer(judge=mock_judge)
        original = [Case(input="What is AI?", expected="AI is artificial intelligence")]
        evolved = await synth.evolve(original, strategy=EvolutionStrategy.ADVERSARIAL)
        assert len(evolved) >= 1
        mock_judge.evaluate.assert_called()

    @pytest.mark.asyncio
    async def test_evolve_empty_list(self, mock_judge):
        synth = Synthesizer(judge=mock_judge)
        evolved = await synth.evolve([], strategy=EvolutionStrategy.SIMPLE)
        assert evolved == []
        mock_judge.evaluate.assert_not_called()

    # -- prompt builders ---------------------------------------------------

    def test_build_document_prompt(self, synthesizer):
        user_prompt, system_prompt = synthesizer._build_document_prompt(
            documents=["Doc one", "Doc two"],
            strategy=EvolutionStrategy.REASONING,
            count=3,
        )
        assert "reasoning" in system_prompt.lower() or "reasoning" in user_prompt.lower()
        assert "3" in user_prompt
        assert "Doc one" in user_prompt
        assert "Doc two" in user_prompt

    def test_build_description_prompt(self, synthesizer):
        user_prompt, system_prompt = synthesizer._build_description_prompt(
            description="A travel booking assistant",
            strategy=EvolutionStrategy.COMPARATIVE,
            count=5,
        )
        assert "travel booking" in user_prompt.lower()
        assert "5" in user_prompt
        assert "comparative" in system_prompt.lower() or "comparative" in user_prompt.lower()

    # -- cost tracking -----------------------------------------------------

    def test_cost_tracking(self, mock_judge):
        synth = Synthesizer(judge=mock_judge)
        assert synth.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_cost_accumulates(self, mock_judge):
        mock_judge.evaluate.return_value = _make_json_response(_SINGLE_CASE_PAYLOAD)
        synth = Synthesizer(judge=mock_judge, config=SynthesisConfig(num_cases=1))
        await synth.from_description("test", num_cases=1)
        assert synth.total_cost > 0.0

    # -- distribution helper -----------------------------------------------

    def test_distribute_even(self):
        plan = Synthesizer._distribute(
            6, [EvolutionStrategy.SIMPLE, EvolutionStrategy.REASONING]
        )
        assert len(plan) == 2
        assert sum(c for _, c in plan) == 6

    def test_distribute_empty_strategies(self):
        plan = Synthesizer._distribute(5, [])
        assert plan == [(EvolutionStrategy.SIMPLE, 5)]

    # -- combine_documents helper ------------------------------------------

    def test_combine_documents_empty(self):
        result = Synthesizer._combine_documents([])
        assert "no documents" in result.lower()

    def test_combine_documents_normal(self):
        result = Synthesizer._combine_documents(["Hello world", "Bye world"])
        assert "Document 1" in result
        assert "Document 2" in result
        assert "Hello world" in result

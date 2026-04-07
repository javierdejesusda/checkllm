"""Tests for ConversationSimulator and related models."""

from unittest.mock import AsyncMock

import pytest

from checkllm.synthesizer import (
    ConversationSimulator,
    SimulatedConversation,
    SimulatedTurn,
)
from checkllm.models import JudgeResponse


class TestSimulatedModels:
    def test_simulated_turn(self):
        t = SimulatedTurn(role="user", content="Hello!")
        assert t.role == "user"

    def test_simulated_conversation(self):
        conv = SimulatedConversation(
            turns=[
                SimulatedTurn(role="user", content="Hi"),
                SimulatedTurn(role="assistant", content="Hello!"),
            ],
            topic="greetings",
        )
        assert len(conv.user_turns) == 1
        assert len(conv.assistant_turns) == 1
        assert "user: Hi" in conv.format_transcript()

    def test_empty_conversation(self):
        conv = SimulatedConversation()
        assert conv.user_turns == []
        assert conv.format_transcript() == ""


class TestConversationSimulator:
    @pytest.mark.asyncio
    async def test_generate_conversations(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(
            score=1.0, reasoning="Hello, how can I help?", cost=0.001
        )
        sim = ConversationSimulator(judge=judge)
        convs = await sim.agenerate(
            topic="customer support",
            num_conversations=2,
            turns_per_conversation=4,
        )
        assert len(convs) == 2
        for conv in convs:
            assert isinstance(conv, SimulatedConversation)
            assert len(conv.turns) == 4
            assert conv.topic == "customer support"

    @pytest.mark.asyncio
    async def test_alternating_roles(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(
            score=1.0, reasoning="Test message", cost=0.001
        )
        sim = ConversationSimulator(judge=judge)
        convs = await sim.agenerate(
            topic="test",
            num_conversations=1,
            turns_per_conversation=4,
        )
        roles = [t.role for t in convs[0].turns]
        assert roles == ["user", "assistant", "user", "assistant"]

    @pytest.mark.asyncio
    async def test_custom_persona(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(
            score=1.0, reasoning="I'm confused", cost=0.001
        )
        sim = ConversationSimulator(judge=judge)
        convs = await sim.agenerate(
            topic="billing",
            num_conversations=1,
            turns_per_conversation=2,
            persona="An angry customer who was overcharged",
        )
        assert convs[0].persona == "An angry customer who was overcharged"

    def test_sync_generate(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(
            score=1.0, reasoning="Test", cost=0.001
        )
        sim = ConversationSimulator(judge=judge)
        convs = sim.generate(
            topic="test",
            num_conversations=1,
            turns_per_conversation=2,
        )
        assert len(convs) == 1

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from checkllm.synthesizer import (
    EvolutionStrategy,
    KnowledgeGraph,
    KnowledgeNode,
    Synthesizer,
)
from checkllm.models import JudgeResponse


class TestKnowledgeGraph:
    def test_knowledge_node(self):
        node = KnowledgeNode(
            content="Python is a programming language.",
            summary="About Python",
            entities=["Python"],
            themes=["programming"],
        )
        assert node.content
        assert "Python" in node.entities

    def test_knowledge_graph(self):
        nodes = [
            KnowledgeNode(content="A", entities=["X"], connections=[1]),
            KnowledgeNode(content="B", entities=["X", "Y"], connections=[0]),
        ]
        kg = KnowledgeGraph(nodes=nodes, document_name="test")
        pairs = kg.connected_pairs()
        assert (0, 1) in pairs

    def test_knowledge_graph_strategy_exists(self):
        assert EvolutionStrategy.KNOWLEDGE_GRAPH == "knowledge_graph"

    @pytest.mark.asyncio
    async def test_build_knowledge_graph(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(
            score=1.0,
            reasoning='{"entities": ["Python", "GIL"], "themes": ["programming", "concurrency"]}',
            cost=0.001,
        )
        synth = Synthesizer(judge=judge)
        kg = await synth.build_knowledge_graph(
            "Python is a language. " * 50 + "The GIL prevents true threading. " * 50,
            chunk_size=100,
        )
        assert isinstance(kg, KnowledgeGraph)
        assert len(kg.nodes) >= 2

    @pytest.mark.asyncio
    async def test_generate_from_kg(self):
        judge = AsyncMock()
        judge.evaluate.return_value = JudgeResponse(
            score=1.0,
            reasoning='{"question": "What is Python?", "answer": "A programming language", "context": "Python is a language."}',
            cost=0.001,
        )
        nodes = [
            KnowledgeNode(
                content="Python is great.",
                entities=["Python"],
                themes=["programming"],
                connections=[1],
            ),
            KnowledgeNode(
                content="GIL limits threading.",
                entities=["GIL"],
                themes=["programming"],
                connections=[0],
            ),
        ]
        kg = KnowledgeGraph(nodes=nodes)
        synth = Synthesizer(judge=judge)
        cases = await synth.generate_from_knowledge_graph(kg, count=3)
        assert len(cases) >= 1

"""Tests for the knowledge_graph module (KG-based test data generation pipeline)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from checkllm.knowledge_graph import (
    EntityExtractor,
    HeadlineSplitter,
    KGEdge,
    KGNode,
    KGTestGenerator,
    KeyphraseExtractor,
    KnowledgeGraph,
    MultiHopAbstractSynthesizer,
    MultiHopSpecificSynthesizer,
    OverlapBuilder,
    Persona,
    QueryLength,
    QueryStyle,
    SentenceSplitter,
    SimilarityBuilder,
    SingleHopSynthesizer,
    SummaryExtractor,
    SynthesizedSample,
    ThemeExtractor,
)
from checkllm.datasets.case import Case
from checkllm.models import JudgeResponse


def _make_mock_judge(raw_output: str = '{"question": "What?", "answer": "Yes."}'):
    """Create an AsyncMock judge that returns the given raw_output."""
    judge = AsyncMock()
    judge.evaluate.return_value = JudgeResponse(
        score=1.0,
        reasoning=raw_output,
        raw_output=raw_output,
        cost=0.001,
    )
    return judge


class TestKGNode:
    def test_create_node(self):
        node = KGNode(
            id="node-1",
            content="Some content",
            node_type="chunk",
        )
        assert node.id == "node-1"
        assert node.content == "Some content"
        assert node.node_type == "chunk"
        assert node.metadata == {}
        assert node.embedding is None

    def test_node_with_metadata(self):
        node = KGNode(
            id="node-2",
            content="Data",
            node_type="entity",
            metadata={"source": "test"},
            embedding=[0.1, 0.2, 0.3],
        )
        assert node.metadata["source"] == "test"
        assert len(node.embedding) == 3


class TestKGEdge:
    def test_create_edge(self):
        edge = KGEdge(
            source_id="a",
            target_id="b",
            relationship="contains",
        )
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.weight == 1.0

    def test_edge_with_weight(self):
        edge = KGEdge(
            source_id="a",
            target_id="b",
            relationship="similar_to",
            weight=0.85,
        )
        assert edge.weight == 0.85


class TestKnowledgeGraph:
    def test_add_node(self):
        kg = KnowledgeGraph()
        node = KGNode(id="n1", content="text", node_type="chunk")
        kg.add_node(node)
        assert len(kg.nodes) == 1
        assert kg.nodes[0].id == "n1"

    def test_add_edge(self):
        kg = KnowledgeGraph()
        edge = KGEdge(source_id="a", target_id="b", relationship="contains")
        kg.add_edge(edge)
        assert len(kg.edges) == 1

    def test_get_node(self):
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="x", content="hello", node_type="chunk"))
        kg.add_node(KGNode(id="y", content="world", node_type="entity"))
        assert kg.get_node("x") is not None
        assert kg.get_node("x").content == "hello"
        assert kg.get_node("missing") is None

    def test_get_neighbors(self):
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="a", content="A", node_type="chunk"))
        kg.add_node(KGNode(id="b", content="B", node_type="chunk"))
        kg.add_node(KGNode(id="c", content="C", node_type="chunk"))
        kg.add_edge(KGEdge(source_id="a", target_id="b", relationship="similar_to"))
        kg.add_edge(KGEdge(source_id="a", target_id="c", relationship="related_to"))

        neighbors = kg.get_neighbors("a")
        neighbor_ids = {n.id for n in neighbors}
        assert neighbor_ids == {"b", "c"}

    def test_get_neighbors_bidirectional(self):
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="a", content="A", node_type="chunk"))
        kg.add_node(KGNode(id="b", content="B", node_type="chunk"))
        kg.add_edge(KGEdge(source_id="a", target_id="b", relationship="similar_to"))

        neighbors_of_b = kg.get_neighbors("b")
        assert len(neighbors_of_b) == 1
        assert neighbors_of_b[0].id == "a"

    def test_get_connected_chunks(self):
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="e1", content="entity", node_type="entity"))
        kg.add_node(KGNode(id="c1", content="chunk1", node_type="chunk"))
        kg.add_node(KGNode(id="c2", content="chunk2", node_type="chunk"))
        kg.add_node(KGNode(id="c3", content="chunk3", node_type="chunk"))
        kg.add_edge(KGEdge(source_id="e1", target_id="c1", relationship="contains_entity"))
        kg.add_edge(KGEdge(source_id="e1", target_id="c2", relationship="contains_entity"))
        kg.add_edge(KGEdge(source_id="c2", target_id="c3", relationship="similar_to"))

        chunks = kg.get_connected_chunks("e1", max_hops=2)
        chunk_ids = {c.id for c in chunks}
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids
        assert "c3" in chunk_ids

    def test_get_connected_chunks_respects_max_hops(self):
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="a", content="A", node_type="chunk"))
        kg.add_node(KGNode(id="b", content="B", node_type="chunk"))
        kg.add_node(KGNode(id="c", content="C", node_type="chunk"))
        kg.add_edge(KGEdge(source_id="a", target_id="b", relationship="similar_to"))
        kg.add_edge(KGEdge(source_id="b", target_id="c", relationship="similar_to"))

        chunks_1hop = kg.get_connected_chunks("a", max_hops=1)
        assert len(chunks_1hop) == 1
        assert chunks_1hop[0].id == "b"

    def test_get_subgraph(self):
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="a", content="A", node_type="chunk"))
        kg.add_node(KGNode(id="b", content="B", node_type="chunk"))
        kg.add_node(KGNode(id="c", content="C", node_type="chunk"))
        kg.add_edge(KGEdge(source_id="a", target_id="b", relationship="similar_to"))
        kg.add_edge(KGEdge(source_id="b", target_id="c", relationship="similar_to"))
        kg.add_edge(KGEdge(source_id="a", target_id="c", relationship="related_to"))

        sub = kg.get_subgraph(["a", "b"])
        assert len(sub.nodes) == 2
        assert len(sub.edges) == 1
        assert sub.edges[0].source_id == "a"
        assert sub.edges[0].target_id == "b"

    def test_get_nodes_by_type(self):
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="c1", content="C1", node_type="chunk"))
        kg.add_node(KGNode(id="e1", content="E1", node_type="entity"))
        kg.add_node(KGNode(id="c2", content="C2", node_type="chunk"))

        chunks = kg.get_nodes_by_type("chunk")
        assert len(chunks) == 2
        entities = kg.get_nodes_by_type("entity")
        assert len(entities) == 1


class TestSentenceSplitter:
    @pytest.mark.asyncio
    async def test_splits_document_into_chunks(self):
        kg = KnowledgeGraph()
        long_text = (
            "Machine learning is a subset of artificial intelligence. "
            "It allows systems to learn from data. "
            "Deep learning uses neural networks with multiple layers. "
            "These networks can process complex patterns. "
            "Natural language processing handles human language. "
            "It enables computers to understand text and speech. "
            "Computer vision deals with image understanding. "
            "It is used in self-driving cars and medical imaging. "
        )
        kg.add_node(KGNode(id="doc-0", content=long_text, node_type="document"))

        splitter = SentenceSplitter(min_tokens=10, max_tokens=30)
        kg = await splitter.apply(kg)

        chunks = kg.get_nodes_by_type("chunk")
        assert len(chunks) >= 2

        contains_edges = [e for e in kg.edges if e.relationship == "contains"]
        assert len(contains_edges) == len(chunks)
        for edge in contains_edges:
            assert edge.source_id == "doc-0"

    @pytest.mark.asyncio
    async def test_empty_document(self):
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="doc-0", content="", node_type="document"))

        splitter = SentenceSplitter()
        kg = await splitter.apply(kg)

        chunks = kg.get_nodes_by_type("chunk")
        assert len(chunks) == 0


class TestEntityExtractor:
    @pytest.mark.asyncio
    async def test_extracts_named_entities(self):
        kg = KnowledgeGraph()
        text = (
            "John Smith works at Acme Corporation. He met Jane Doe on "
            "12/25/2023. The deal was worth $1,000,000."
        )
        kg.add_node(KGNode(id="c1", content=text, node_type="chunk"))

        extractor = EntityExtractor()
        kg = await extractor.apply(kg)

        entity_nodes = kg.get_nodes_by_type("entity")
        entity_contents = {n.content for n in entity_nodes}

        assert "John Smith" in entity_contents
        assert "Acme Corporation" in entity_contents
        assert "Jane Doe" in entity_contents

    @pytest.mark.asyncio
    async def test_deduplicates_entities(self):
        kg = KnowledgeGraph()
        kg.add_node(
            KGNode(
                id="c1",
                content="John Smith went to the store. John Smith bought apples.",
                node_type="chunk",
            )
        )

        extractor = EntityExtractor()
        kg = await extractor.apply(kg)

        john_nodes = [n for n in kg.get_nodes_by_type("entity") if n.content == "John Smith"]
        assert len(john_nodes) == 1

    @pytest.mark.asyncio
    async def test_entities_shared_across_chunks(self):
        kg = KnowledgeGraph()
        kg.add_node(
            KGNode(
                id="c1",
                content="John Smith is a programmer.",
                node_type="chunk",
            )
        )
        kg.add_node(
            KGNode(
                id="c2",
                content="John Smith likes coffee.",
                node_type="chunk",
            )
        )

        extractor = EntityExtractor()
        kg = await extractor.apply(kg)

        john_nodes = [n for n in kg.get_nodes_by_type("entity") if n.content == "John Smith"]
        assert len(john_nodes) == 1

        john_id = john_nodes[0].id
        edges_to_john = [e for e in kg.edges if e.target_id == john_id]
        source_ids = {e.source_id for e in edges_to_john}
        assert "c1" in source_ids
        assert "c2" in source_ids


class TestSimilarityBuilder:
    @pytest.mark.asyncio
    async def test_creates_edges_between_similar_chunks(self):
        kg = KnowledgeGraph()
        kg.add_node(
            KGNode(
                id="c1",
                content="machine learning algorithms process data patterns",
                node_type="chunk",
            )
        )
        kg.add_node(
            KGNode(
                id="c2",
                content="machine learning models learn from data patterns",
                node_type="chunk",
            )
        )
        kg.add_node(
            KGNode(
                id="c3",
                content="cooking recipes require flour sugar eggs butter",
                node_type="chunk",
            )
        )

        builder = SimilarityBuilder(threshold=0.3)
        kg = await builder.apply(kg)

        similar_edges = [e for e in kg.edges if e.relationship == "similar_to"]
        similar_pairs = {(e.source_id, e.target_id) for e in similar_edges}

        assert ("c1", "c2") in similar_pairs
        assert ("c1", "c3") not in similar_pairs
        assert ("c2", "c3") not in similar_pairs

    @pytest.mark.asyncio
    async def test_no_edges_when_dissimilar(self):
        kg = KnowledgeGraph()
        kg.add_node(
            KGNode(
                id="c1",
                content="alpha beta gamma",
                node_type="chunk",
            )
        )
        kg.add_node(
            KGNode(
                id="c2",
                content="delta epsilon zeta",
                node_type="chunk",
            )
        )

        builder = SimilarityBuilder(threshold=0.3)
        kg = await builder.apply(kg)

        similar_edges = [e for e in kg.edges if e.relationship == "similar_to"]
        assert len(similar_edges) == 0


class TestKeyphraseExtractor:
    @pytest.mark.asyncio
    async def test_extracts_keyphrases(self):
        kg = KnowledgeGraph()
        kg.add_node(
            KGNode(
                id="c1",
                content=(
                    "Machine learning is a field of computer science. "
                    "Machine learning uses algorithms to learn from data. "
                    "Computer science encompasses many areas."
                ),
                node_type="chunk",
            )
        )

        extractor = KeyphraseExtractor(max_phrases=5)
        kg = await extractor.apply(kg)

        chunk = kg.get_node("c1")
        assert "keyphrases" in chunk.metadata
        assert len(chunk.metadata["keyphrases"]) > 0
        assert "machine" in chunk.metadata["keyphrases"]


class TestOverlapBuilder:
    @pytest.mark.asyncio
    async def test_creates_edges_for_shared_keyphrases(self):
        kg = KnowledgeGraph()
        kg.add_node(
            KGNode(
                id="c1",
                content="placeholder",
                node_type="chunk",
                metadata={"keyphrases": ["machine", "learning", "algorithms"]},
            )
        )
        kg.add_node(
            KGNode(
                id="c2",
                content="placeholder",
                node_type="chunk",
                metadata={"keyphrases": ["machine", "learning", "data"]},
            )
        )
        kg.add_node(
            KGNode(
                id="c3",
                content="placeholder",
                node_type="chunk",
                metadata={"keyphrases": ["cooking", "recipes", "food"]},
            )
        )

        builder = OverlapBuilder(min_shared=2)
        kg = await builder.apply(kg)

        related_edges = [e for e in kg.edges if e.relationship == "related_to"]
        assert len(related_edges) == 1
        assert related_edges[0].source_id == "c1"
        assert related_edges[0].target_id == "c2"


class TestHeadlineSplitter:
    @pytest.mark.asyncio
    async def test_splits_by_markdown_headers(self):
        doc_text = (
            "# Introduction\n"
            "This is the intro section.\n\n"
            "## Background\n"
            "Some background info here.\n\n"
            "## Methods\n"
            "We used these methods."
        )
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="doc-0", content=doc_text, node_type="document"))

        splitter = HeadlineSplitter()
        kg = await splitter.apply(kg)

        chunks = kg.get_nodes_by_type("chunk")
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_no_headers_returns_whole_doc(self):
        doc_text = "This is a plain document with no headers at all."
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="doc-0", content=doc_text, node_type="document"))

        splitter = HeadlineSplitter()
        kg = await splitter.apply(kg)

        chunks = kg.get_nodes_by_type("chunk")
        assert len(chunks) == 1


class TestSingleHopSynthesizer:
    @pytest.mark.asyncio
    async def test_generates_single_hop_questions(self):
        judge = _make_mock_judge()
        kg = KnowledgeGraph()
        kg.add_node(
            KGNode(
                id="c1",
                content="Python was created by Guido van Rossum in 1991.",
                node_type="chunk",
            )
        )
        kg.add_node(
            KGNode(
                id="c2",
                content="JavaScript was created by Brendan Eich in 1995.",
                node_type="chunk",
            )
        )

        synth = SingleHopSynthesizer()
        samples = await synth.synthesize(kg, judge, num_samples=2)

        assert len(samples) == 2
        for sample in samples:
            assert sample.query_type == "single_hop"
            assert sample.query
            assert sample.reference_answer
            assert len(sample.contexts) == 1

    @pytest.mark.asyncio
    async def test_with_persona(self):
        judge = _make_mock_judge()
        kg = KnowledgeGraph()
        kg.add_node(
            KGNode(
                id="c1",
                content="Quantum computing uses qubits instead of classical bits.",
                node_type="chunk",
            )
        )

        persona = Persona(
            name="Student",
            description="A first-year CS student",
            expertise_level="beginner",
        )

        synth = SingleHopSynthesizer()
        samples = await synth.synthesize(kg, judge, num_samples=1, persona=persona)

        assert len(samples) == 1
        assert samples[0].persona == persona

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self):
        judge = _make_mock_judge()
        kg = KnowledgeGraph()

        synth = SingleHopSynthesizer()
        samples = await synth.synthesize(kg, judge, num_samples=5)
        assert samples == []


class TestMultiHopSynthesizers:
    @pytest.mark.asyncio
    async def test_multi_hop_abstract(self):
        judge = _make_mock_judge()
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="c1", content="Chunk about topic A.", node_type="chunk"))
        kg.add_node(KGNode(id="c2", content="Chunk about topic B.", node_type="chunk"))
        kg.add_edge(
            KGEdge(
                source_id="c1",
                target_id="c2",
                relationship="similar_to",
            )
        )

        synth = MultiHopAbstractSynthesizer()
        samples = await synth.synthesize(kg, judge, num_samples=1)

        assert len(samples) == 1
        assert samples[0].query_type == "multi_hop_abstract"
        assert len(samples[0].contexts) == 2

    @pytest.mark.asyncio
    async def test_multi_hop_specific(self):
        judge = _make_mock_judge()
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="c1", content="Chunk about topic A.", node_type="chunk"))
        kg.add_node(KGNode(id="c2", content="Chunk about topic B.", node_type="chunk"))
        kg.add_edge(
            KGEdge(
                source_id="c1",
                target_id="c2",
                relationship="similar_to",
            )
        )

        synth = MultiHopSpecificSynthesizer()
        samples = await synth.synthesize(kg, judge, num_samples=1)

        assert len(samples) == 1
        assert samples[0].query_type == "multi_hop_specific"

    @pytest.mark.asyncio
    async def test_fallback_to_adjacent_pairs(self):
        """When no edges connect chunks, the multi-hop synthesizers should
        fall back to pairing adjacent chunks."""
        judge = _make_mock_judge()
        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="c1", content="First chunk.", node_type="chunk"))
        kg.add_node(KGNode(id="c2", content="Second chunk.", node_type="chunk"))

        synth = MultiHopAbstractSynthesizer()
        samples = await synth.synthesize(kg, judge, num_samples=1)

        assert len(samples) == 1


class TestThemeExtractor:
    @pytest.mark.asyncio
    async def test_extracts_themes(self):
        raw = '{"themes": ["artificial intelligence", "machine learning"]}'
        judge = _make_mock_judge(raw_output=raw)

        kg = KnowledgeGraph()
        kg.add_node(
            KGNode(
                id="c1",
                content="AI and machine learning are transforming industries.",
                node_type="chunk",
            )
        )

        extractor = ThemeExtractor()
        kg = await extractor.apply(kg, judge=judge)

        concepts = kg.get_nodes_by_type("concept")
        assert len(concepts) == 2
        theme_contents = {c.content for c in concepts}
        assert "artificial intelligence" in theme_contents
        assert "machine learning" in theme_contents

    @pytest.mark.asyncio
    async def test_requires_judge(self):
        kg = KnowledgeGraph()
        extractor = ThemeExtractor()
        with pytest.raises(ValueError, match="requires a judge"):
            await extractor.apply(kg, judge=None)


class TestSummaryExtractor:
    @pytest.mark.asyncio
    async def test_extracts_summary(self):
        raw = '{"summary": "This document covers AI topics."}'
        judge = _make_mock_judge(raw_output=raw)

        kg = KnowledgeGraph()
        kg.add_node(
            KGNode(
                id="doc-0",
                content="A long document about artificial intelligence and its applications.",
                node_type="document",
            )
        )

        extractor = SummaryExtractor()
        kg = await extractor.apply(kg, judge=judge)

        doc = kg.get_node("doc-0")
        assert doc.metadata.get("summary") == "This document covers AI topics."


class TestSynthesizedSample:
    def test_create_sample(self):
        sample = SynthesizedSample(
            query="What is Python?",
            reference_answer="A programming language.",
            contexts=["Python is a language."],
            query_type="single_hop",
        )
        assert sample.query == "What is Python?"
        assert sample.style == QueryStyle.PERFECT_GRAMMAR
        assert sample.length == QueryLength.MEDIUM

    def test_sample_with_persona(self):
        persona = Persona(
            name="Alice",
            description="A data scientist",
            expertise_level="expert",
        )
        sample = SynthesizedSample(
            query="Explain backpropagation",
            reference_answer="It is...",
            contexts=["Context."],
            query_type="single_hop",
            persona=persona,
            style=QueryStyle.CONVERSATIONAL,
            length=QueryLength.LONG,
        )
        assert sample.persona.name == "Alice"
        assert sample.style == QueryStyle.CONVERSATIONAL


class TestQueryEnums:
    def test_query_styles(self):
        assert QueryStyle.PERFECT_GRAMMAR == "perfect_grammar"
        assert QueryStyle.WEB_SEARCH == "web_search"
        assert QueryStyle.MISSPELLED == "misspelled"
        assert QueryStyle.CONVERSATIONAL == "conversational"

    def test_query_lengths(self):
        assert QueryLength.SHORT == "short"
        assert QueryLength.MEDIUM == "medium"
        assert QueryLength.LONG == "long"


class TestKGTestGenerator:
    @pytest.mark.asyncio
    async def test_build_kg(self):
        judge = _make_mock_judge()
        gen = KGTestGenerator(judge=judge)

        kg = await gen.build_kg(
            documents=["This is document one. It has important content."],
        )

        assert len(kg.get_nodes_by_type("document")) == 1
        assert len(kg.get_nodes_by_type("chunk")) >= 1

    @pytest.mark.asyncio
    async def test_build_kg_multiple_documents(self):
        judge = _make_mock_judge()
        gen = KGTestGenerator(judge=judge)

        kg = await gen.build_kg(
            documents=[
                "First document about artificial intelligence.",
                "Second document about machine learning.",
            ],
        )

        assert len(kg.get_nodes_by_type("document")) == 2

    @pytest.mark.asyncio
    async def test_generate_personas(self):
        raw = json.dumps(
            {
                "personas": [
                    {
                        "name": "Alice",
                        "description": "A student",
                        "expertise_level": "beginner",
                    },
                    {
                        "name": "Bob",
                        "description": "A researcher",
                        "expertise_level": "expert",
                    },
                ]
            }
        )
        judge = _make_mock_judge(raw_output=raw)
        gen = KGTestGenerator(judge=judge)

        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="c1", content="Test content.", node_type="chunk"))

        personas = await gen.generate_personas(kg, num_personas=2)
        assert len(personas) == 2
        assert personas[0].name == "Alice"
        assert personas[1].expertise_level == "expert"

    @pytest.mark.asyncio
    async def test_generate_personas_fallback(self):
        judge = _make_mock_judge(raw_output="invalid json")
        gen = KGTestGenerator(judge=judge)

        kg = KnowledgeGraph()
        kg.add_node(KGNode(id="c1", content="Test content.", node_type="chunk"))

        personas = await gen.generate_personas(kg, num_personas=3)
        assert len(personas) == 3
        assert personas[0].expertise_level in ("beginner", "intermediate", "expert")

    @pytest.mark.asyncio
    async def test_to_cases(self):
        judge = _make_mock_judge()
        gen = KGTestGenerator(judge=judge)

        samples = [
            SynthesizedSample(
                query="What is X?",
                reference_answer="X is Y.",
                contexts=["Context about X."],
                query_type="single_hop",
                source_nodes=["c1"],
            ),
            SynthesizedSample(
                query="Compare A and B.",
                reference_answer="A differs from B in...",
                contexts=["About A.", "About B."],
                query_type="multi_hop_abstract",
                persona=Persona(
                    name="Researcher",
                    description="A PhD researcher",
                    expertise_level="expert",
                ),
                source_nodes=["c1", "c2"],
            ),
        ]

        cases = gen.to_cases(samples)
        assert len(cases) == 2
        assert isinstance(cases[0], Case)
        assert cases[0].input == "What is X?"
        assert cases[0].expected == "X is Y."
        assert cases[0].metadata["query_type"] == "single_hop"
        assert cases[0].metadata["strategy"] == "knowledge_graph_v2"

        assert cases[1].metadata["persona"] == "Researcher"
        assert cases[1].metadata["difficulty"] == "medium"

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Integration test: build KG from documents and generate samples."""
        raw_qa = '{"question": "What is machine learning?", "answer": "A subset of AI."}'
        judge = _make_mock_judge(raw_output=raw_qa)
        gen = KGTestGenerator(judge=judge)

        documents = [
            (
                "Machine learning is a subset of artificial intelligence. "
                "It enables computers to learn from data without being "
                "explicitly programmed. Deep learning is a type of machine "
                "learning that uses neural networks."
            ),
            (
                "Natural language processing allows computers to understand "
                "human language. It is used in chatbots, translation, and "
                "sentiment analysis. NLP combines linguistics and AI."
            ),
        ]

        samples = await gen.generate(
            documents=documents,
            num_samples=4,
            synthesizers={
                "single_hop": 0.5,
                "multi_hop_abstract": 0.25,
                "multi_hop_specific": 0.25,
            },
        )

        assert len(samples) > 0
        for sample in samples:
            assert sample.query
            assert sample.reference_answer

        cases = gen.to_cases(samples)
        assert len(cases) == len(samples)
        for case in cases:
            assert case.input
            assert case.expected

    @pytest.mark.asyncio
    async def test_generate_with_personas_and_styles(self):
        raw_qa = '{"question": "Explain it", "answer": "Sure thing."}'
        judge = _make_mock_judge(raw_output=raw_qa)
        gen = KGTestGenerator(judge=judge)

        samples = await gen.generate(
            documents=["Some document content about technology."],
            num_samples=2,
            personas=[
                Persona(
                    name="Student",
                    description="A CS student",
                    expertise_level="beginner",
                ),
            ],
            styles=[QueryStyle.CONVERSATIONAL],
            lengths=[QueryLength.SHORT],
        )

        assert len(samples) > 0


class TestSynthesizerIntegration:
    """Test integration of KG pipeline with the Synthesizer class."""

    def test_knowledge_graph_v2_strategy_exists(self):
        from checkllm.synthesizer import EvolutionStrategy

        assert EvolutionStrategy.KNOWLEDGE_GRAPH_V2 == "knowledge_graph_v2"

    @pytest.mark.asyncio
    async def test_from_documents_kg(self):
        raw_qa = '{"question": "What is AI?", "answer": "Artificial Intelligence."}'
        judge = _make_mock_judge(raw_output=raw_qa)

        from checkllm.synthesizer import Synthesizer

        synth = Synthesizer(judge=judge)
        cases = await synth.from_documents_kg(
            documents=["AI is artificial intelligence."],
            num_cases=2,
        )

        assert len(cases) > 0
        for case in cases:
            assert isinstance(case, Case)
            assert case.input

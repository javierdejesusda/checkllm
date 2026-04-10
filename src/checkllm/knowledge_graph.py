"""Knowledge Graph-based test data generation pipeline.

Build a knowledge graph from documents, apply transforms to extract entities,
themes, and relationships, then synthesize diverse test questions across
single-hop and multi-hop reasoning patterns.

Usage::

    from checkllm.knowledge_graph import KGTestGenerator
    from checkllm.judge import OpenAIJudge

    gen = KGTestGenerator(judge=OpenAIJudge())
    samples = await gen.generate(
        documents=["doc1...", "doc2..."],
        num_samples=50,
        transforms=[EntityExtractor(), ThemeExtractor()],
        synthesizers={
            "single_hop": 0.4,
            "multi_hop_abstract": 0.3,
            "multi_hop_specific": 0.3,
        },
        personas=5,
        styles=[QueryStyle.PERFECT_GRAMMAR, QueryStyle.WEB_SEARCH],
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from checkllm.datasets.case import Case
from checkllm.judge import JudgeBackend

logger = logging.getLogger("checkllm.knowledge_graph")


class KGNode(BaseModel):
    """A node in the knowledge graph.

    Attributes:
        id: Unique identifier for this node.
        content: The text content of the node.
        node_type: Category of the node (document, chunk, entity, concept).
        metadata: Arbitrary metadata attached to this node.
        embedding: Optional embedding vector for similarity computations.
    """

    id: str
    content: str
    node_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class KGEdge(BaseModel):
    """An edge connecting two nodes.

    Attributes:
        source_id: The id of the source node.
        target_id: The id of the target node.
        relationship: The type of relationship between nodes.
        weight: Strength of the relationship, defaults to 1.0.
    """

    source_id: str
    target_id: str
    relationship: str
    weight: float = 1.0


class KnowledgeGraph(BaseModel):
    """A knowledge graph built from documents.

    Stores nodes and edges with helper methods for traversal and subgraph
    extraction.

    Attributes:
        nodes: All nodes in the graph.
        edges: All edges in the graph.
    """

    nodes: list[KGNode] = Field(default_factory=list)
    edges: list[KGEdge] = Field(default_factory=list)

    def add_node(self, node: KGNode) -> None:
        """Add a node to the graph.

        Args:
            node: The node to add.
        """
        self.nodes.append(node)

    def add_edge(self, edge: KGEdge) -> None:
        """Add an edge to the graph.

        Args:
            edge: The edge to add.
        """
        self.edges.append(edge)

    def get_node(self, node_id: str) -> KGNode | None:
        """Return the node with the given id, or None.

        Args:
            node_id: The id to look up.

        Returns:
            The matching KGNode or None if not found.
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_neighbors(self, node_id: str) -> list[KGNode]:
        """Return all nodes directly connected to the given node.

        Args:
            node_id: The id of the node whose neighbors to retrieve.

        Returns:
            A list of neighboring KGNode objects.
        """
        neighbor_ids: set[str] = set()
        for edge in self.edges:
            if edge.source_id == node_id:
                neighbor_ids.add(edge.target_id)
            elif edge.target_id == node_id:
                neighbor_ids.add(edge.source_id)

        return [n for n in self.nodes if n.id in neighbor_ids]

    def get_connected_chunks(
        self, node_id: str, max_hops: int = 2
    ) -> list[KGNode]:
        """Return chunk nodes reachable within max_hops from node_id.

        Uses breadth-first search to find all chunk-type nodes within the
        specified hop distance.

        Args:
            node_id: The starting node id.
            max_hops: Maximum number of edge traversals allowed.

        Returns:
            A list of chunk-type KGNode objects reachable within max_hops.
        """
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(node_id, 0)]
        visited.add(node_id)
        chunks: list[KGNode] = []

        while queue:
            current_id, depth = queue.pop(0)
            node = self.get_node(current_id)
            if node and node.node_type == "chunk" and current_id != node_id:
                chunks.append(node)

            if depth < max_hops:
                for neighbor in self.get_neighbors(current_id):
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        queue.append((neighbor.id, depth + 1))

        return chunks

    def get_subgraph(self, node_ids: list[str]) -> KnowledgeGraph:
        """Extract a subgraph containing only the specified nodes and their edges.

        Args:
            node_ids: The ids of nodes to include in the subgraph.

        Returns:
            A new KnowledgeGraph containing only the specified nodes and
            any edges between them.
        """
        id_set = set(node_ids)
        sub_nodes = [n for n in self.nodes if n.id in id_set]
        sub_edges = [
            e
            for e in self.edges
            if e.source_id in id_set and e.target_id in id_set
        ]
        return KnowledgeGraph(nodes=sub_nodes, edges=sub_edges)

    def get_nodes_by_type(self, node_type: str) -> list[KGNode]:
        """Return all nodes of the specified type.

        Args:
            node_type: The type to filter by.

        Returns:
            A list of matching KGNode objects.
        """
        return [n for n in self.nodes if n.node_type == node_type]


class BaseTransform(ABC):
    """Base class for KG transforms.

    Transforms modify a KnowledgeGraph by adding nodes, edges, or
    restructuring existing elements.
    """

    @abstractmethod
    async def apply(
        self, kg: KnowledgeGraph, judge: JudgeBackend | None = None
    ) -> KnowledgeGraph:
        """Apply this transform to the knowledge graph.

        Args:
            kg: The knowledge graph to transform.
            judge: Optional LLM judge backend for transforms that need LLM calls.

        Returns:
            The modified KnowledgeGraph (may be mutated in place).
        """


class EntityExtractor(BaseTransform):
    """Extract named entities from chunks, add as entity nodes.

    Uses regex patterns for common entity types: capitalized names, dates,
    numbers, email addresses, and organization-like patterns.
    """

    _PATTERNS: list[tuple[str, str]] = [
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", "named_entity"),
        (
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            r"|\b(?:January|February|March|April|May|June|July|August|September"
            r"|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            "date",
        ),
        (r"\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})+\b", "number"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    ]

    async def apply(
        self, kg: KnowledgeGraph, judge: JudgeBackend | None = None
    ) -> KnowledgeGraph:
        """Extract entities from chunk nodes and add them to the graph.

        For each chunk node, applies regex patterns to find entities, creates
        entity nodes for new entities, and adds edges from chunks to their
        entities.

        Args:
            kg: The knowledge graph to process.
            judge: Not used by this transform.

        Returns:
            The knowledge graph with entity nodes and edges added.
        """
        existing_entities: dict[str, str] = {}

        chunk_nodes = kg.get_nodes_by_type("chunk")
        for chunk in chunk_nodes:
            for pattern, entity_subtype in self._PATTERNS:
                matches = re.findall(pattern, chunk.content)
                for match in matches:
                    match_text = match.strip()
                    if len(match_text) < 2:
                        continue

                    if match_text in existing_entities:
                        entity_id = existing_entities[match_text]
                    else:
                        entity_id = f"entity-{uuid.uuid4().hex[:8]}"
                        entity_node = KGNode(
                            id=entity_id,
                            content=match_text,
                            node_type="entity",
                            metadata={"entity_subtype": entity_subtype},
                        )
                        kg.add_node(entity_node)
                        existing_entities[match_text] = entity_id

                    already_linked = any(
                        e.source_id == chunk.id
                        and e.target_id == entity_id
                        for e in kg.edges
                    )
                    if not already_linked:
                        kg.add_edge(
                            KGEdge(
                                source_id=chunk.id,
                                target_id=entity_id,
                                relationship="contains_entity",
                            )
                        )

        return kg


class ThemeExtractor(BaseTransform):
    """Extract themes/topics from chunks using an LLM, add as concept nodes."""

    async def apply(
        self, kg: KnowledgeGraph, judge: JudgeBackend | None = None
    ) -> KnowledgeGraph:
        """Extract themes from chunk nodes using the LLM judge.

        For each chunk node, asks the LLM to identify 3-5 themes, creates
        concept nodes for new themes, and adds edges from chunks to themes.

        Args:
            kg: The knowledge graph to process.
            judge: Required. The LLM backend used for theme extraction.

        Returns:
            The knowledge graph with concept nodes and edges added.

        Raises:
            ValueError: If judge is None.
        """
        if judge is None:
            raise ValueError("ThemeExtractor requires a judge backend.")

        existing_themes: dict[str, str] = {}
        chunk_nodes = kg.get_nodes_by_type("chunk")

        for chunk in chunk_nodes:
            prompt = (
                "Extract 3-5 main themes or topics from the following text. "
                "Return a JSON object with a single key 'themes' containing "
                "a list of short theme strings.\n\n"
                f"Text:\n{chunk.content[:2000]}\n\n"
                'Respond with JSON only: {{"themes": ["theme1", "theme2", ...]}}'
            )
            try:
                response = await judge.evaluate(prompt=prompt)
                raw = response.raw_output or response.reasoning or "{}"
                raw = _strip_code_fences(raw)
                parsed = json.loads(raw)
                themes = [str(t).lower().strip() for t in parsed.get("themes", [])]
            except (json.JSONDecodeError, ValueError, KeyError):
                themes = []

            for theme in themes:
                if not theme:
                    continue
                if theme in existing_themes:
                    theme_id = existing_themes[theme]
                else:
                    theme_id = f"concept-{uuid.uuid4().hex[:8]}"
                    kg.add_node(
                        KGNode(
                            id=theme_id,
                            content=theme,
                            node_type="concept",
                        )
                    )
                    existing_themes[theme] = theme_id

                kg.add_edge(
                    KGEdge(
                        source_id=chunk.id,
                        target_id=theme_id,
                        relationship="has_theme",
                    )
                )

        return kg


class SummaryExtractor(BaseTransform):
    """Generate summaries for document nodes using an LLM."""

    async def apply(
        self, kg: KnowledgeGraph, judge: JudgeBackend | None = None
    ) -> KnowledgeGraph:
        """Generate summaries for document-type nodes.

        Calls the LLM to produce a concise summary for each document node
        and stores it in the node's metadata.

        Args:
            kg: The knowledge graph to process.
            judge: Required. The LLM backend used for summarization.

        Returns:
            The knowledge graph with summaries stored in document node metadata.

        Raises:
            ValueError: If judge is None.
        """
        if judge is None:
            raise ValueError("SummaryExtractor requires a judge backend.")

        doc_nodes = kg.get_nodes_by_type("document")
        for doc in doc_nodes:
            prompt = (
                "Summarize the following text in 2-3 sentences.\n\n"
                f"Text:\n{doc.content[:3000]}\n\n"
                'Respond with JSON only: {{"summary": "..."}}'
            )
            try:
                response = await judge.evaluate(prompt=prompt)
                raw = response.raw_output or response.reasoning or "{}"
                raw = _strip_code_fences(raw)
                parsed = json.loads(raw)
                doc.metadata["summary"] = str(parsed.get("summary", ""))
            except (json.JSONDecodeError, ValueError, KeyError):
                doc.metadata["summary"] = ""

        return kg


class KeyphraseExtractor(BaseTransform):
    """Extract key phrases using TF-IDF-like scoring (no LLM needed).

    Computes term frequency across the corpus of chunks and identifies
    phrases that are frequent within a chunk but not ubiquitous across all
    chunks.
    """

    def __init__(self, max_phrases: int = 10, min_word_length: int = 3) -> None:
        self.max_phrases = max_phrases
        self.min_word_length = min_word_length

    async def apply(
        self, kg: KnowledgeGraph, judge: JudgeBackend | None = None
    ) -> KnowledgeGraph:
        """Extract key phrases from chunk nodes and store them in metadata.

        Uses a simplified TF-IDF approach: for each chunk, scores words by
        their frequency within the chunk relative to how many chunks contain
        the word.

        Args:
            kg: The knowledge graph to process.
            judge: Not used by this transform.

        Returns:
            The knowledge graph with keyphrases in chunk node metadata.
        """
        chunk_nodes = kg.get_nodes_by_type("chunk")
        if not chunk_nodes:
            return kg

        all_words_per_chunk: list[list[str]] = []
        for chunk in chunk_nodes:
            words = _tokenize(chunk.content, self.min_word_length)
            all_words_per_chunk.append(words)

        num_chunks = len(chunk_nodes)
        doc_freq: Counter[str] = Counter()
        for words in all_words_per_chunk:
            unique = set(words)
            for w in unique:
                doc_freq[w] += 1

        for idx, chunk in enumerate(chunk_nodes):
            words = all_words_per_chunk[idx]
            if not words:
                continue

            tf: Counter[str] = Counter(words)
            scored: list[tuple[str, float]] = []
            for word, count in tf.items():
                df = doc_freq.get(word, 1)
                idf = math.log((num_chunks + 1) / (df + 1)) + 1
                scored.append((word, count * idf))

            scored.sort(key=lambda x: x[1], reverse=True)
            keyphrases = [w for w, _ in scored[: self.max_phrases]]
            chunk.metadata["keyphrases"] = keyphrases

        return kg


class SentenceSplitter(BaseTransform):
    """Split documents into sentence-level chunks.

    Produces chunk nodes from document nodes by splitting on sentence
    boundaries and grouping sentences until the token budget is met.

    Attributes:
        min_tokens: Minimum approximate token count per chunk.
        max_tokens: Maximum approximate token count per chunk.
    """

    def __init__(self, min_tokens: int = 50, max_tokens: int = 500) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    async def apply(
        self, kg: KnowledgeGraph, judge: JudgeBackend | None = None
    ) -> KnowledgeGraph:
        """Split document nodes into chunk nodes.

        Splits each document node into sentences, groups them to meet the
        token constraints, and adds chunk nodes plus 'contains' edges.

        Args:
            kg: The knowledge graph to process.
            judge: Not used by this transform.

        Returns:
            The knowledge graph with chunk nodes added.
        """
        doc_nodes = kg.get_nodes_by_type("document")
        for doc in doc_nodes:
            sentences = _split_sentences(doc.content)
            chunks = self._group_sentences(sentences)
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"chunk-{doc.id}-{i}"
                kg.add_node(
                    KGNode(
                        id=chunk_id,
                        content=chunk_text,
                        node_type="chunk",
                        metadata={"doc_id": doc.id, "chunk_index": i},
                    )
                )
                kg.add_edge(
                    KGEdge(
                        source_id=doc.id,
                        target_id=chunk_id,
                        relationship="contains",
                    )
                )

        return kg

    def _group_sentences(self, sentences: list[str]) -> list[str]:
        """Group sentences into chunks respecting token limits.

        Approximates token count as word count. Groups consecutive sentences
        so each chunk falls between min_tokens and max_tokens words.

        Args:
            sentences: A list of sentence strings to group.

        Returns:
            A list of chunk strings.
        """
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            token_count = len(sentence.split())
            if current_tokens + token_count > self.max_tokens and current:
                chunks.append(" ".join(current))
                current = [sentence]
                current_tokens = token_count
            else:
                current.append(sentence)
                current_tokens += token_count

        if current:
            text = " ".join(current)
            if chunks and current_tokens < self.min_tokens:
                chunks[-1] = chunks[-1] + " " + text
            else:
                chunks.append(text)

        return chunks


class HeadlineSplitter(BaseTransform):
    """Split documents by headlines/headers (markdown, HTML).

    Detects markdown headers (# ...) and HTML headers (<h1>...</h1>) and
    splits the document into sections, creating one chunk per section.
    """

    _HEADER_PATTERN = re.compile(
        r"(?:^|\n)(?:#{1,6}\s+.+|<h[1-6][^>]*>.*?</h[1-6]>)", re.IGNORECASE
    )

    async def apply(
        self, kg: KnowledgeGraph, judge: JudgeBackend | None = None
    ) -> KnowledgeGraph:
        """Split document nodes by headline boundaries.

        Finds markdown or HTML headers in each document node and creates
        chunk nodes for each section.

        Args:
            kg: The knowledge graph to process.
            judge: Not used by this transform.

        Returns:
            The knowledge graph with headline-based chunk nodes added.
        """
        doc_nodes = kg.get_nodes_by_type("document")
        for doc in doc_nodes:
            sections = self._split_by_headers(doc.content)
            for i, section_text in enumerate(sections):
                section_text = section_text.strip()
                if not section_text:
                    continue
                chunk_id = f"chunk-{doc.id}-h{i}"
                kg.add_node(
                    KGNode(
                        id=chunk_id,
                        content=section_text,
                        node_type="chunk",
                        metadata={"doc_id": doc.id, "section_index": i},
                    )
                )
                kg.add_edge(
                    KGEdge(
                        source_id=doc.id,
                        target_id=chunk_id,
                        relationship="contains",
                    )
                )

        return kg

    def _split_by_headers(self, text: str) -> list[str]:
        """Split text by detected headers.

        Args:
            text: The document text to split.

        Returns:
            A list of section strings. If no headers are found, the entire
            text is returned as a single section.
        """
        positions = [m.start() for m in self._HEADER_PATTERN.finditer(text)]
        if not positions:
            return [text] if text.strip() else []

        if positions[0] > 0:
            positions.insert(0, 0)

        sections: list[str] = []
        for i, start in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)

        return sections


class SimilarityBuilder(BaseTransform):
    """Add edges between chunks with high lexical similarity.

    Uses Jaccard similarity on word sets to determine which chunk pairs
    are similar enough to be connected.

    Attributes:
        threshold: Minimum Jaccard similarity to create an edge (0-1).
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    async def apply(
        self, kg: KnowledgeGraph, judge: JudgeBackend | None = None
    ) -> KnowledgeGraph:
        """Add similarity edges between chunk nodes.

        Computes pairwise Jaccard similarity between all chunk node pairs
        and adds 'similar_to' edges where similarity exceeds the threshold.

        Args:
            kg: The knowledge graph to process.
            judge: Not used by this transform.

        Returns:
            The knowledge graph with similarity edges added.
        """
        chunk_nodes = kg.get_nodes_by_type("chunk")
        word_sets: list[set[str]] = []
        for chunk in chunk_nodes:
            words = set(_tokenize(chunk.content, min_length=2))
            word_sets.append(words)

        for i in range(len(chunk_nodes)):
            for j in range(i + 1, len(chunk_nodes)):
                sim = _jaccard_similarity(word_sets[i], word_sets[j])
                if sim >= self.threshold:
                    kg.add_edge(
                        KGEdge(
                            source_id=chunk_nodes[i].id,
                            target_id=chunk_nodes[j].id,
                            relationship="similar_to",
                            weight=sim,
                        )
                    )

        return kg


class OverlapBuilder(BaseTransform):
    """Add edges between chunks that share key phrases.

    Requires that KeyphraseExtractor has been run first so that chunk
    nodes have 'keyphrases' in their metadata.
    """

    def __init__(self, min_shared: int = 2) -> None:
        self.min_shared = min_shared

    async def apply(
        self, kg: KnowledgeGraph, judge: JudgeBackend | None = None
    ) -> KnowledgeGraph:
        """Add overlap edges between chunk nodes sharing keyphrases.

        Compares keyphrase lists on chunk nodes and creates 'related_to'
        edges when at least min_shared keyphrases are shared.

        Args:
            kg: The knowledge graph to process.
            judge: Not used by this transform.

        Returns:
            The knowledge graph with overlap edges added.
        """
        chunk_nodes = kg.get_nodes_by_type("chunk")
        phrase_sets: list[tuple[KGNode, set[str]]] = []
        for chunk in chunk_nodes:
            phrases = set(chunk.metadata.get("keyphrases", []))
            phrase_sets.append((chunk, phrases))

        for i in range(len(phrase_sets)):
            for j in range(i + 1, len(phrase_sets)):
                shared = phrase_sets[i][1] & phrase_sets[j][1]
                if len(shared) >= self.min_shared:
                    kg.add_edge(
                        KGEdge(
                            source_id=phrase_sets[i][0].id,
                            target_id=phrase_sets[j][0].id,
                            relationship="related_to",
                            weight=len(shared),
                        )
                    )

        return kg


class QueryStyle(str, Enum):
    """Style of generated query."""

    PERFECT_GRAMMAR = "perfect_grammar"
    WEB_SEARCH = "web_search"
    MISSPELLED = "misspelled"
    CONVERSATIONAL = "conversational"


class QueryLength(str, Enum):
    """Length of generated query."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class Persona(BaseModel):
    """A user persona for generating diverse questions.

    Attributes:
        name: Display name for the persona.
        description: A short description of the persona's background.
        expertise_level: One of 'beginner', 'intermediate', or 'expert'.
    """

    name: str
    description: str
    expertise_level: str


class SynthesizedSample(BaseModel):
    """A generated test sample from the KG.

    Attributes:
        query: The synthesized question text.
        reference_answer: The expected correct answer.
        contexts: List of context passages that support the answer.
        query_type: The type of question (single_hop, multi_hop_abstract,
            multi_hop_specific).
        persona: Optional persona used during generation.
        style: The query style applied.
        length: The target query length.
        source_nodes: List of KG node ids used to generate this sample.
    """

    query: str
    reference_answer: str
    contexts: list[str]
    query_type: str
    persona: Persona | None = None
    style: QueryStyle = QueryStyle.PERFECT_GRAMMAR
    length: QueryLength = QueryLength.MEDIUM
    source_nodes: list[str] = Field(default_factory=list)


class BaseSynthesizer(ABC):
    """Base class for KG-based query synthesizers."""

    @abstractmethod
    async def synthesize(
        self,
        kg: KnowledgeGraph,
        judge: JudgeBackend,
        num_samples: int,
        persona: Persona | None = None,
        style: QueryStyle = QueryStyle.PERFECT_GRAMMAR,
        length: QueryLength = QueryLength.MEDIUM,
    ) -> list[SynthesizedSample]:
        """Generate test samples from the knowledge graph.

        Args:
            kg: The knowledge graph to generate from.
            judge: The LLM backend for generation.
            num_samples: Number of samples to generate.
            persona: Optional persona to use for generation.
            style: The query style to use.
            length: The target query length.

        Returns:
            A list of SynthesizedSample objects.
        """


class SingleHopSynthesizer(BaseSynthesizer):
    """Generate questions answerable from a single chunk."""

    async def synthesize(
        self,
        kg: KnowledgeGraph,
        judge: JudgeBackend,
        num_samples: int,
        persona: Persona | None = None,
        style: QueryStyle = QueryStyle.PERFECT_GRAMMAR,
        length: QueryLength = QueryLength.MEDIUM,
    ) -> list[SynthesizedSample]:
        """Generate single-hop questions from individual chunks.

        Selects chunk nodes and asks the LLM to produce a question
        answerable from that chunk alone.

        Args:
            kg: The knowledge graph to generate from.
            judge: The LLM backend for generation.
            num_samples: Number of samples to generate.
            persona: Optional persona to use for generation.
            style: The query style to use.
            length: The target query length.

        Returns:
            A list of single-hop SynthesizedSample objects.
        """
        chunks = kg.get_nodes_by_type("chunk")
        if not chunks:
            return []

        tasks: list[asyncio.Task[SynthesizedSample | None]] = []
        for i in range(num_samples):
            chunk = chunks[i % len(chunks)]
            tasks.append(
                asyncio.ensure_future(
                    self._generate_one(judge, chunk, persona, style, length)
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        samples: list[SynthesizedSample] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("SingleHop generation failed: %s", result)
                continue
            if result is not None:
                samples.append(result)

        return samples

    async def _generate_one(
        self,
        judge: JudgeBackend,
        chunk: KGNode,
        persona: Persona | None,
        style: QueryStyle,
        length: QueryLength,
    ) -> SynthesizedSample | None:
        """Generate one single-hop sample from a chunk.

        Args:
            judge: The LLM backend.
            chunk: The chunk node to generate from.
            persona: Optional persona context.
            style: The query style.
            length: The target length.

        Returns:
            A SynthesizedSample, or None on failure.
        """
        persona_instruction = ""
        if persona:
            persona_instruction = (
                f"The question should be asked from the perspective of: "
                f"{persona.name} ({persona.description}, "
                f"expertise: {persona.expertise_level}). "
            )

        style_instruction = _style_instruction(style)
        length_instruction = _length_instruction(length)

        prompt = (
            "Generate a question and answer pair based on the following passage. "
            "The question must be answerable using ONLY the information in the "
            "passage.\n\n"
            f"{persona_instruction}{style_instruction}{length_instruction}\n\n"
            f"Passage:\n{chunk.content[:3000]}\n\n"
            'Respond with JSON only: {{"question": "...", "answer": "..."}}'
        )
        try:
            response = await judge.evaluate(prompt=prompt)
            raw = response.raw_output or response.reasoning or "{}"
            raw = _strip_code_fences(raw)
            parsed = json.loads(raw)
            question = str(parsed.get("question", ""))
            answer = str(parsed.get("answer", ""))
            if not question:
                return None
            return SynthesizedSample(
                query=question,
                reference_answer=answer,
                contexts=[chunk.content],
                query_type="single_hop",
                persona=persona,
                style=style,
                length=length,
                source_nodes=[chunk.id],
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Failed to parse single-hop sample: %s", exc)
            return None


class MultiHopAbstractSynthesizer(BaseSynthesizer):
    """Generate abstract reasoning questions requiring multiple chunks."""

    async def synthesize(
        self,
        kg: KnowledgeGraph,
        judge: JudgeBackend,
        num_samples: int,
        persona: Persona | None = None,
        style: QueryStyle = QueryStyle.PERFECT_GRAMMAR,
        length: QueryLength = QueryLength.MEDIUM,
    ) -> list[SynthesizedSample]:
        """Generate multi-hop abstract reasoning questions.

        Finds pairs of connected chunk nodes and generates questions that
        require abstract reasoning across both chunks.

        Args:
            kg: The knowledge graph to generate from.
            judge: The LLM backend for generation.
            num_samples: Number of samples to generate.
            persona: Optional persona to use for generation.
            style: The query style to use.
            length: The target query length.

        Returns:
            A list of multi-hop abstract SynthesizedSample objects.
        """
        pairs = _get_chunk_pairs(kg)
        if not pairs:
            return []

        tasks: list[asyncio.Task[SynthesizedSample | None]] = []
        for i in range(num_samples):
            pair = pairs[i % len(pairs)]
            tasks.append(
                asyncio.ensure_future(
                    self._generate_one(judge, pair, persona, style, length)
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        samples: list[SynthesizedSample] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("MultiHopAbstract generation failed: %s", result)
                continue
            if result is not None:
                samples.append(result)

        return samples

    async def _generate_one(
        self,
        judge: JudgeBackend,
        pair: tuple[KGNode, KGNode],
        persona: Persona | None,
        style: QueryStyle,
        length: QueryLength,
    ) -> SynthesizedSample | None:
        """Generate one multi-hop abstract sample.

        Args:
            judge: The LLM backend.
            pair: A pair of connected chunk nodes.
            persona: Optional persona context.
            style: The query style.
            length: The target length.

        Returns:
            A SynthesizedSample, or None on failure.
        """
        persona_instruction = ""
        if persona:
            persona_instruction = (
                f"The question should be asked from the perspective of: "
                f"{persona.name} ({persona.description}, "
                f"expertise: {persona.expertise_level}). "
            )

        style_instruction = _style_instruction(style)
        length_instruction = _length_instruction(length)

        prompt = (
            "Generate a question that requires ABSTRACT REASONING by combining "
            "information from BOTH passages below. The question should require "
            "comparing, contrasting, or synthesizing concepts across the two "
            "passages. Do NOT ask a question answerable from just one passage.\n\n"
            f"{persona_instruction}{style_instruction}{length_instruction}\n\n"
            f"Passage A:\n{pair[0].content[:2000]}\n\n"
            f"Passage B:\n{pair[1].content[:2000]}\n\n"
            'Respond with JSON only: {{"question": "...", "answer": "..."}}'
        )
        try:
            response = await judge.evaluate(prompt=prompt)
            raw = response.raw_output or response.reasoning or "{}"
            raw = _strip_code_fences(raw)
            parsed = json.loads(raw)
            question = str(parsed.get("question", ""))
            answer = str(parsed.get("answer", ""))
            if not question:
                return None
            return SynthesizedSample(
                query=question,
                reference_answer=answer,
                contexts=[pair[0].content, pair[1].content],
                query_type="multi_hop_abstract",
                persona=persona,
                style=style,
                length=length,
                source_nodes=[pair[0].id, pair[1].id],
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Failed to parse multi-hop abstract sample: %s", exc)
            return None


class MultiHopSpecificSynthesizer(BaseSynthesizer):
    """Generate specific fact-finding questions across multiple chunks."""

    async def synthesize(
        self,
        kg: KnowledgeGraph,
        judge: JudgeBackend,
        num_samples: int,
        persona: Persona | None = None,
        style: QueryStyle = QueryStyle.PERFECT_GRAMMAR,
        length: QueryLength = QueryLength.MEDIUM,
    ) -> list[SynthesizedSample]:
        """Generate multi-hop specific fact-finding questions.

        Finds pairs of connected chunk nodes and generates questions that
        require combining specific facts from both.

        Args:
            kg: The knowledge graph to generate from.
            judge: The LLM backend for generation.
            num_samples: Number of samples to generate.
            persona: Optional persona to use for generation.
            style: The query style to use.
            length: The target query length.

        Returns:
            A list of multi-hop specific SynthesizedSample objects.
        """
        pairs = _get_chunk_pairs(kg)
        if not pairs:
            return []

        tasks: list[asyncio.Task[SynthesizedSample | None]] = []
        for i in range(num_samples):
            pair = pairs[i % len(pairs)]
            tasks.append(
                asyncio.ensure_future(
                    self._generate_one(judge, pair, persona, style, length)
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        samples: list[SynthesizedSample] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("MultiHopSpecific generation failed: %s", result)
                continue
            if result is not None:
                samples.append(result)

        return samples

    async def _generate_one(
        self,
        judge: JudgeBackend,
        pair: tuple[KGNode, KGNode],
        persona: Persona | None,
        style: QueryStyle,
        length: QueryLength,
    ) -> SynthesizedSample | None:
        """Generate one multi-hop specific sample.

        Args:
            judge: The LLM backend.
            pair: A pair of connected chunk nodes.
            persona: Optional persona context.
            style: The query style.
            length: The target length.

        Returns:
            A SynthesizedSample, or None on failure.
        """
        persona_instruction = ""
        if persona:
            persona_instruction = (
                f"The question should be asked from the perspective of: "
                f"{persona.name} ({persona.description}, "
                f"expertise: {persona.expertise_level}). "
            )

        style_instruction = _style_instruction(style)
        length_instruction = _length_instruction(length)

        prompt = (
            "Generate a question that requires finding and combining SPECIFIC "
            "FACTS from BOTH passages below. The answer must reference concrete "
            "details (names, numbers, dates, etc.) from both passages.\n\n"
            f"{persona_instruction}{style_instruction}{length_instruction}\n\n"
            f"Passage A:\n{pair[0].content[:2000]}\n\n"
            f"Passage B:\n{pair[1].content[:2000]}\n\n"
            'Respond with JSON only: {{"question": "...", "answer": "..."}}'
        )
        try:
            response = await judge.evaluate(prompt=prompt)
            raw = response.raw_output or response.reasoning or "{}"
            raw = _strip_code_fences(raw)
            parsed = json.loads(raw)
            question = str(parsed.get("question", ""))
            answer = str(parsed.get("answer", ""))
            if not question:
                return None
            return SynthesizedSample(
                query=question,
                reference_answer=answer,
                contexts=[pair[0].content, pair[1].content],
                query_type="multi_hop_specific",
                persona=persona,
                style=style,
                length=length,
                source_nodes=[pair[0].id, pair[1].id],
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Failed to parse multi-hop specific sample: %s", exc)
            return None


_DEFAULT_TRANSFORMS: list[BaseTransform] = [
    SentenceSplitter(),
    EntityExtractor(),
    KeyphraseExtractor(),
    SimilarityBuilder(threshold=0.3),
]

_SYNTHESIZER_MAP: dict[str, BaseSynthesizer] = {
    "single_hop": SingleHopSynthesizer(),
    "multi_hop_abstract": MultiHopAbstractSynthesizer(),
    "multi_hop_specific": MultiHopSpecificSynthesizer(),
}


class KGTestGenerator:
    """Knowledge Graph-based test data generator.

    Builds a knowledge graph from raw documents, applies a pipeline of
    transforms to enrich it with entities, themes, and relationships, then
    uses synthesizers to produce diverse test question-answer pairs.

    Usage::

        from checkllm.knowledge_graph import KGTestGenerator
        from checkllm.judge import OpenAIJudge

        gen = KGTestGenerator(judge=OpenAIJudge())
        samples = await gen.generate(
            documents=["doc1...", "doc2..."],
            num_samples=50,
            transforms=[EntityExtractor(), ThemeExtractor()],
            synthesizers={
                "single_hop": 0.4,
                "multi_hop_abstract": 0.3,
                "multi_hop_specific": 0.3,
            },
            personas=5,
            styles=[QueryStyle.PERFECT_GRAMMAR, QueryStyle.WEB_SEARCH],
        )
    """

    def __init__(self, judge: JudgeBackend) -> None:
        self.judge = judge

    async def build_kg(
        self,
        documents: list[str],
        transforms: list[BaseTransform] | None = None,
    ) -> KnowledgeGraph:
        """Build a knowledge graph from documents by applying transforms.

        Creates document nodes from the raw text, then sequentially applies
        each transform to enrich the graph.

        Args:
            documents: A list of raw document texts.
            transforms: Transforms to apply. Defaults to SentenceSplitter,
                EntityExtractor, KeyphraseExtractor, and SimilarityBuilder.

        Returns:
            A fully built KnowledgeGraph.
        """
        kg = KnowledgeGraph()

        for i, doc_text in enumerate(documents):
            doc_id = f"doc-{i}"
            kg.add_node(
                KGNode(
                    id=doc_id,
                    content=doc_text,
                    node_type="document",
                    metadata={"doc_index": i},
                )
            )

        pipeline = transforms if transforms is not None else _DEFAULT_TRANSFORMS

        for transform in pipeline:
            kg = await transform.apply(kg, judge=self.judge)

        return kg

    async def generate_personas(
        self, kg: KnowledgeGraph, num_personas: int = 5
    ) -> list[Persona]:
        """Auto-generate diverse personas based on the knowledge graph content.

        Asks the LLM to create personas that vary in expertise level and
        background, suitable for generating diverse questions about the
        graph's topics.

        Args:
            kg: The knowledge graph to base personas on.
            num_personas: Number of personas to generate.

        Returns:
            A list of Persona objects.
        """
        chunk_nodes = kg.get_nodes_by_type("chunk")
        sample_content = " ".join(
            c.content[:200] for c in chunk_nodes[:5]
        )

        prompt = (
            f"Based on the following content samples, generate {num_personas} "
            f"diverse user personas who might ask questions about this material. "
            f"Each persona should have a different expertise level "
            f"(beginner, intermediate, or expert) and background.\n\n"
            f"Content samples:\n{sample_content[:2000]}\n\n"
            f"Respond with JSON only: "
            f'{{"personas": [{{"name": "...", "description": "...", '
            f'"expertise_level": "beginner|intermediate|expert"}}]}}'
        )

        try:
            response = await self.judge.evaluate(prompt=prompt)
            raw = response.raw_output or response.reasoning or "{}"
            raw = _strip_code_fences(raw)
            parsed = json.loads(raw)
            persona_list = parsed.get("personas", [])
            personas: list[Persona] = []
            for p in persona_list[:num_personas]:
                personas.append(
                    Persona(
                        name=str(p.get("name", "User")),
                        description=str(p.get("description", "")),
                        expertise_level=str(
                            p.get("expertise_level", "intermediate")
                        ),
                    )
                )
            return personas
        except (json.JSONDecodeError, ValueError, KeyError):
            return [
                Persona(
                    name=f"User {i + 1}",
                    description="A general user",
                    expertise_level=level,
                )
                for i, level in enumerate(
                    ["beginner", "intermediate", "expert", "beginner", "expert"][
                        :num_personas
                    ]
                )
            ]

    async def generate(
        self,
        documents: list[str],
        num_samples: int = 50,
        transforms: list[BaseTransform] | None = None,
        synthesizers: dict[str, float] | None = None,
        personas: int | list[Persona] | None = None,
        styles: list[QueryStyle] | None = None,
        lengths: list[QueryLength] | None = None,
    ) -> list[SynthesizedSample]:
        """Generate test samples from documents using the full KG pipeline.

        Builds the knowledge graph, generates personas if requested, and
        distributes sample generation across synthesizer types according to
        the specified weights.

        Args:
            documents: Raw document texts to generate from.
            num_samples: Total number of samples to generate.
            transforms: Transforms for building the KG. Uses defaults if None.
            synthesizers: Mapping of synthesizer names to weight proportions.
                Defaults to 50% single_hop, 25% multi_hop_abstract, 25%
                multi_hop_specific.
            personas: Either an int to auto-generate that many personas, or a
                list of Persona objects, or None for no persona variation.
            styles: List of QueryStyle values to cycle through.
            lengths: List of QueryLength values to cycle through.

        Returns:
            A list of SynthesizedSample objects.
        """
        kg = await self.build_kg(documents, transforms)

        persona_list: list[Persona] | None = None
        if isinstance(personas, int) and personas > 0:
            persona_list = await self.generate_personas(kg, personas)
        elif isinstance(personas, list):
            persona_list = personas

        if synthesizers is None:
            synthesizers = {
                "single_hop": 0.5,
                "multi_hop_abstract": 0.25,
                "multi_hop_specific": 0.25,
            }

        if styles is None:
            styles = [QueryStyle.PERFECT_GRAMMAR]

        if lengths is None:
            lengths = [QueryLength.MEDIUM]

        total_weight = sum(synthesizers.values())
        all_samples: list[SynthesizedSample] = []

        for synth_name, weight in synthesizers.items():
            synth = _SYNTHESIZER_MAP.get(synth_name)
            if synth is None:
                logger.warning("Unknown synthesizer: %s", synth_name)
                continue

            count = max(1, round(num_samples * weight / total_weight))
            style = styles[len(all_samples) % len(styles)]
            length = lengths[len(all_samples) % len(lengths)]
            persona = None
            if persona_list:
                persona = persona_list[len(all_samples) % len(persona_list)]

            samples = await synth.synthesize(
                kg, self.judge, count, persona, style, length
            )
            all_samples.extend(samples)

        logger.info(
            "KGTestGenerator: generated %d/%d samples",
            len(all_samples),
            num_samples,
        )
        return all_samples[:num_samples]

    def to_cases(self, samples: list[SynthesizedSample]) -> list[Case]:
        """Convert synthesized samples to checkllm Case objects.

        Maps each SynthesizedSample to a Case with appropriate input,
        expected answer, context, and metadata.

        Args:
            samples: The samples to convert.

        Returns:
            A list of Case objects.
        """
        cases: list[Case] = []
        for sample in samples:
            context = "\n\n".join(sample.contexts) if sample.contexts else None
            metadata: dict[str, Any] = {
                "query_type": sample.query_type,
                "style": sample.style.value,
                "length": sample.length.value,
                "source_nodes": sample.source_nodes,
                "strategy": "knowledge_graph_v2",
                "difficulty": (
                    "easy" if sample.query_type == "single_hop" else "medium"
                ),
            }
            if sample.persona:
                metadata["persona"] = sample.persona.name
                metadata["expertise_level"] = sample.persona.expertise_level

            cases.append(
                Case(
                    input=sample.query,
                    expected=sample.reference_answer,
                    context=context,
                    metadata=metadata,
                )
            )

        return cases


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output.

    Args:
        text: Raw LLM output text.

    Returns:
        The text with leading/trailing code fences removed.
    """
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic heuristics.

    Splits on sentence-ending punctuation followed by whitespace and a
    capital letter or end-of-string.

    Args:
        text: The text to split.

    Returns:
        A list of sentence strings.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def _tokenize(text: str, min_length: int = 3) -> list[str]:
    """Tokenize text into lowercase words, filtering by minimum length.

    Args:
        text: The text to tokenize.
        min_length: Minimum character length for a word to be included.

    Returns:
        A list of lowercase word strings.
    """
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return [w for w in words if len(w) >= min_length]


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets.

    Args:
        set_a: First set of strings.
        set_b: Second set of strings.

    Returns:
        The Jaccard similarity coefficient (0.0-1.0).
    """
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _style_instruction(style: QueryStyle) -> str:
    """Return an LLM instruction for the given query style.

    Args:
        style: The query style.

    Returns:
        An instruction string for the LLM prompt.
    """
    instructions = {
        QueryStyle.PERFECT_GRAMMAR: "Use perfect grammar and formal language. ",
        QueryStyle.WEB_SEARCH: (
            "Write the question as a web search query (short, keyword-style). "
        ),
        QueryStyle.MISSPELLED: (
            "Include 1-2 realistic typos or misspellings in the question. "
        ),
        QueryStyle.CONVERSATIONAL: (
            "Write the question in a casual, conversational tone. "
        ),
    }
    return instructions.get(style, "")


def _length_instruction(length: QueryLength) -> str:
    """Return an LLM instruction for the given query length.

    Args:
        length: The target query length.

    Returns:
        An instruction string for the LLM prompt.
    """
    instructions = {
        QueryLength.SHORT: "Keep the question under 15 words. ",
        QueryLength.MEDIUM: "The question should be 15-30 words. ",
        QueryLength.LONG: "The question should be 30-60 words with detail. ",
    }
    return instructions.get(length, "")


def _get_chunk_pairs(kg: KnowledgeGraph) -> list[tuple[KGNode, KGNode]]:
    """Find pairs of connected chunk nodes in the graph.

    Looks through edges for any pair of chunk-type nodes that are directly
    connected. Falls back to adjacent chunk nodes if no edges connect them.

    Args:
        kg: The knowledge graph to search.

    Returns:
        A list of (KGNode, KGNode) tuples.
    """
    chunks = kg.get_nodes_by_type("chunk")
    chunk_ids = {c.id for c in chunks}
    chunk_map = {c.id: c for c in chunks}

    pairs: list[tuple[KGNode, KGNode]] = []
    seen: set[tuple[str, str]] = set()

    for edge in kg.edges:
        if edge.source_id in chunk_ids and edge.target_id in chunk_ids:
            key = (
                min(edge.source_id, edge.target_id),
                max(edge.source_id, edge.target_id),
            )
            if key not in seen:
                seen.add(key)
                pairs.append(
                    (chunk_map[edge.source_id], chunk_map[edge.target_id])
                )

    if not pairs and len(chunks) >= 2:
        for i in range(0, len(chunks) - 1, 2):
            pairs.append((chunks[i], chunks[i + 1]))

    return pairs

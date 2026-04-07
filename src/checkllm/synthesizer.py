"""Synthetic test case generator for checkllm.

Generate test cases from documents, descriptions, or by evolving existing cases
to be more challenging using LLM-powered synthesis.

Usage::

    from checkllm.synthesizer import Synthesizer, EvolutionStrategy, SynthesisConfig
    from checkllm.judge import OpenAIJudge

    synth = Synthesizer(judge=OpenAIJudge())

    # From documents
    cases = await synth.from_documents(
        documents=["doc1 text...", "doc2 text..."],
        num_cases=20,
        strategies=[EvolutionStrategy.REASONING, EvolutionStrategy.MULTI_CONTEXT],
    )

    # From a description
    cases = await synth.from_description(
        description="Customer support chatbot for an e-commerce platform",
        num_cases=10,
    )

    # Evolve existing cases
    harder = await synth.evolve(cases, strategy=EvolutionStrategy.ADVERSARIAL)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from checkllm.datasets.case import Case
from checkllm.judge import JudgeBackend
from checkllm.models import JudgeResponse

logger = logging.getLogger("checkllm.synthesizer")

# ---------------------------------------------------------------------------
# Max characters of document content to include in a single prompt
# ---------------------------------------------------------------------------
_MAX_DOC_CHARS = 12_000
_BATCH_SIZE = 5


class EvolutionStrategy(str, Enum):
    """Strategies for evolving test cases to be more challenging."""

    SIMPLE = "simple"
    REASONING = "reasoning"
    MULTI_CONTEXT = "multi_context"
    CONDITIONAL = "conditional"
    ADVERSARIAL = "adversarial"
    COMPARATIVE = "comparative"
    KNOWLEDGE_GRAPH = "knowledge_graph"


# ---------------------------------------------------------------------------
# Strategy descriptions used when building prompts
# ---------------------------------------------------------------------------

_STRATEGY_INSTRUCTIONS: dict[EvolutionStrategy, str] = {
    EvolutionStrategy.SIMPLE: (
        "Generate straightforward question-answer pairs. Each question should "
        "have a clear, direct answer that can be found in the source material."
    ),
    EvolutionStrategy.REASONING: (
        "Generate questions that require multi-step reasoning. The answer should "
        "not be directly stated but must be inferred by combining multiple facts "
        "or applying logical deduction."
    ),
    EvolutionStrategy.MULTI_CONTEXT: (
        "Generate questions that require synthesizing information from multiple "
        "parts of the source material. The answer should depend on combining "
        "facts from different sections or documents."
    ),
    EvolutionStrategy.CONDITIONAL: (
        "Generate questions involving conditional logic, edge cases, or "
        "exceptions. Include scenarios with 'if/then' conditions, boundary "
        "values, or situations where the answer depends on specific constraints."
    ),
    EvolutionStrategy.ADVERSARIAL: (
        "Generate adversarial or tricky questions designed to expose weaknesses. "
        "Include questions with subtle distinctions, common misconceptions, "
        "misleading phrasing, or requests that might cause hallucination."
    ),
    EvolutionStrategy.COMPARATIVE: (
        "Generate questions that require comparing, contrasting, or ranking "
        "multiple items, concepts, or approaches. The answer should involve "
        "analyzing trade-offs or similarities and differences."
    ),
    EvolutionStrategy.KNOWLEDGE_GRAPH: (
        "Generate questions that traverse relationships between entities and "
        "concepts identified in the source material. Questions should require "
        "reasoning across connected facts and named entities."
    ),
}

_DIFFICULTY_MAP: dict[EvolutionStrategy, str] = {
    EvolutionStrategy.SIMPLE: "easy",
    EvolutionStrategy.REASONING: "medium",
    EvolutionStrategy.MULTI_CONTEXT: "medium",
    EvolutionStrategy.CONDITIONAL: "hard",
    EvolutionStrategy.ADVERSARIAL: "hard",
    EvolutionStrategy.COMPARATIVE: "medium",
    EvolutionStrategy.KNOWLEDGE_GRAPH: "medium",
}


class SynthesisConfig(BaseModel):
    """Configuration for test case synthesis."""

    num_cases: int = Field(default=10, ge=1)
    strategies: list[EvolutionStrategy] = Field(
        default_factory=lambda: [EvolutionStrategy.SIMPLE]
    )
    max_retries: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class KnowledgeNode(BaseModel):
    """A node in a knowledge graph built from document chunks."""

    content: str
    summary: str = ""
    entities: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    connections: list[int] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
    """A knowledge graph built from document chunks for test generation."""

    nodes: list[KnowledgeNode] = Field(default_factory=list)
    document_name: str = ""

    def connected_pairs(self) -> list[tuple[int, int]]:
        """Return pairs of connected node indices.

        Returns:
            A list of (i, j) tuples where i < j and the nodes are connected.
        """
        pairs: list[tuple[int, int]] = []
        for i, node in enumerate(self.nodes):
            for j in node.connections:
                if j > i:
                    pairs.append((i, j))
        return pairs


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Best-effort extraction of a JSON array from LLM output.

    Tries direct parse first, then looks for the first ``[...]`` block.
    """
    text = text.strip()

    # Fast path: entire response is valid JSON array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence) :]
        if text.endswith("```"):
            text = text[: -len("```")]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find the outermost [...] in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON array from LLM output: {text[:300]}")


def _dict_to_case(raw: dict[str, Any], strategy: EvolutionStrategy | None = None) -> Case:
    """Convert a raw dict from the LLM into a Case, with safe defaults."""
    metadata: dict[str, Any] = raw.get("metadata") or {}
    if strategy and "strategy" not in metadata:
        metadata["strategy"] = strategy.value
    if strategy and "difficulty" not in metadata:
        metadata["difficulty"] = _DIFFICULTY_MAP.get(strategy, "medium")

    return Case(
        input=str(raw.get("input", "")),
        expected=raw.get("expected"),
        context=raw.get("context"),
        criteria=raw.get("criteria"),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Shared system prompt preamble
# ---------------------------------------------------------------------------

_SYSTEM_PREAMBLE = (
    "You are an expert test-case generator for AI/LLM evaluation. "
    "Your output MUST be a valid JSON array and nothing else — no "
    "markdown fences, no commentary, no explanation. Each element of "
    "the array must be an object with exactly these keys:\n"
    '  "input"    — the question or prompt (string)\n'
    '  "expected" — the ideal / reference answer (string)\n'
    '  "context"  — relevant context or source passage (string)\n'
    '  "criteria" — evaluation criteria for judging the answer (string)\n'
    '  "metadata" — object with at least "strategy" (string) and "difficulty" (string)\n'
)


class Synthesizer:
    """Generate synthetic test cases from documents or schemas using an LLM.

    Usage::

        synth = Synthesizer(judge=OpenAIJudge())
        cases = await synth.from_documents(
            documents=["doc1 text...", "doc2 text..."],
            num_cases=20,
            strategies=[EvolutionStrategy.REASONING, EvolutionStrategy.MULTI_CONTEXT],
        )

        # Or from a schema/description
        cases = await synth.from_description(
            description="Customer support chatbot for an e-commerce platform",
            num_cases=10,
        )
    """

    def __init__(
        self, judge: JudgeBackend, config: SynthesisConfig | None = None
    ) -> None:
        self.judge = judge
        self.config = config or SynthesisConfig()
        self._total_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Cumulative USD cost across all synthesis calls."""
        return self._total_cost

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def from_documents(
        self,
        documents: list[str],
        num_cases: int | None = None,
        strategies: list[EvolutionStrategy] | None = None,
    ) -> list[Case]:
        """Generate test cases grounded in the supplied *documents*.

        Parameters
        ----------
        documents:
            A list of document texts to generate questions from.
        num_cases:
            Total number of cases to generate. Defaults to ``config.num_cases``.
        strategies:
            Evolution strategies to use. Defaults to ``config.strategies``.
            Cases are distributed evenly across strategies.
        """
        num_cases = num_cases or self.config.num_cases
        strategies = strategies or self.config.strategies

        plan = self._distribute(num_cases, strategies)

        tasks: list[asyncio.Task[list[Case]]] = []
        for strategy, count in plan:
            prompt, system_prompt = self._build_document_prompt(
                documents, strategy, count
            )
            tasks.append(
                asyncio.ensure_future(
                    self._generate_batch(prompt, system_prompt, count, strategy)
                )
            )

        batches = await asyncio.gather(*tasks, return_exceptions=True)

        cases: list[Case] = []
        for batch in batches:
            if isinstance(batch, BaseException):
                logger.error("Batch generation failed: %s", batch)
                continue
            cases.extend(batch)

        logger.info(
            "from_documents: generated %d/%d cases (cost=$%.4f)",
            len(cases),
            num_cases,
            self._total_cost,
        )
        return cases

    async def from_description(
        self,
        description: str,
        num_cases: int | None = None,
        strategies: list[EvolutionStrategy] | None = None,
    ) -> list[Case]:
        """Generate test cases from a free-text *description* of the system.

        Parameters
        ----------
        description:
            A description of the application / chatbot / system being tested.
        num_cases:
            Total number of cases. Defaults to ``config.num_cases``.
        strategies:
            Strategies to use. Defaults to ``config.strategies``.
        """
        num_cases = num_cases or self.config.num_cases
        strategies = strategies or self.config.strategies

        plan = self._distribute(num_cases, strategies)

        tasks: list[asyncio.Task[list[Case]]] = []
        for strategy, count in plan:
            prompt, system_prompt = self._build_description_prompt(
                description, strategy, count
            )
            tasks.append(
                asyncio.ensure_future(
                    self._generate_batch(prompt, system_prompt, count, strategy)
                )
            )

        batches = await asyncio.gather(*tasks, return_exceptions=True)

        cases: list[Case] = []
        for batch in batches:
            if isinstance(batch, BaseException):
                logger.error("Batch generation failed: %s", batch)
                continue
            cases.extend(batch)

        logger.info(
            "from_description: generated %d/%d cases (cost=$%.4f)",
            len(cases),
            num_cases,
            self._total_cost,
        )
        return cases

    async def evolve(
        self, cases: list[Case], strategy: EvolutionStrategy
    ) -> list[Case]:
        """Take existing *cases* and make them more challenging.

        Each case is rewritten according to *strategy* (e.g. add adversarial
        phrasing, require multi-step reasoning, etc.).  The original cases are
        **not** modified; new :class:`Case` objects are returned.
        """
        if not cases:
            return []

        # Process in batches to stay within context limits
        tasks: list[asyncio.Task[list[Case]]] = []
        for i in range(0, len(cases), _BATCH_SIZE):
            batch = cases[i : i + _BATCH_SIZE]
            prompt, system_prompt = self._build_evolution_prompt(batch, strategy)
            tasks.append(
                asyncio.ensure_future(
                    self._generate_batch(
                        prompt, system_prompt, len(batch), strategy
                    )
                )
            )

        batches = await asyncio.gather(*tasks, return_exceptions=True)

        evolved: list[Case] = []
        for batch in batches:
            if isinstance(batch, BaseException):
                logger.error("Evolution batch failed: %s", batch)
                continue
            evolved.extend(batch)

        logger.info(
            "evolve (%s): evolved %d/%d cases (cost=$%.4f)",
            strategy.value,
            len(evolved),
            len(cases),
            self._total_cost,
        )
        return evolved

    async def build_knowledge_graph(
        self,
        document: str,
        chunk_size: int = 500,
        document_name: str = "",
    ) -> KnowledgeGraph:
        """Build a KnowledgeGraph from a document by extracting entities and themes.

        Splits the document into chunks, extracts entities and themes from each
        chunk via the judge, then connects chunks that share common entities or
        themes.

        Args:
            document: The source document text to analyze.
            chunk_size: Approximate character size for each chunk.
            document_name: Optional label for the resulting graph.

        Returns:
            A KnowledgeGraph with nodes and their inter-chunk connections.
        """
        chunks = self._split_into_chunks(document, chunk_size)
        nodes: list[KnowledgeNode] = []

        for chunk in chunks:
            extraction_prompt = (
                f"Extract entities and themes from the following text.\n\n"
                f"Text:\n{chunk}\n\n"
                f"Respond with JSON only:\n"
                f'{{"entities": ["entity1", ...], "themes": ["theme1", ...], "summary": "..."}}'
            )
            try:
                response = await self.judge.evaluate(prompt=extraction_prompt)
                raw = response.reasoning or response.raw_output or "{}"
                raw = raw.strip()
                for fence in ("```json", "```"):
                    if raw.startswith(fence):
                        raw = raw[len(fence):]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                raw = raw.strip()
                parsed = json.loads(raw)
                entities = [str(e) for e in parsed.get("entities", [])]
                themes = [str(t) for t in parsed.get("themes", [])]
                summary = str(parsed.get("summary", ""))
            except (json.JSONDecodeError, ValueError, KeyError):
                entities = []
                themes = []
                summary = ""

            nodes.append(KnowledgeNode(
                content=chunk,
                summary=summary,
                entities=entities,
                themes=themes,
            ))

        # Build connections between nodes sharing entities or themes
        for i, node_i in enumerate(nodes):
            shared_entities = set(node_i.entities)
            shared_themes = set(node_i.themes)
            for j, node_j in enumerate(nodes):
                if i == j:
                    continue
                if (
                    shared_entities & set(node_j.entities)
                    or shared_themes & set(node_j.themes)
                ):
                    if j not in node_i.connections:
                        node_i.connections.append(j)

        return KnowledgeGraph(nodes=nodes, document_name=document_name)

    async def generate_from_knowledge_graph(
        self,
        kg: KnowledgeGraph,
        count: int = 10,
    ) -> list[Case]:
        """Generate test cases from a KnowledgeGraph.

        Uses single-hop questions from individual nodes and multi-hop questions
        that span connected node pairs. Falls back to single-hop when no
        connected pairs exist.

        Args:
            kg: The KnowledgeGraph to generate from.
            count: Number of test cases to attempt to generate.

        Returns:
            A list of Case objects derived from the knowledge graph.
        """
        if not kg.nodes:
            return []

        pairs = kg.connected_pairs()
        cases: list[Case] = []
        tasks: list[asyncio.Task[list[Case]]] = []
        strategy = EvolutionStrategy.KNOWLEDGE_GRAPH

        for i in range(count):
            use_multi_hop = pairs and (i % 2 == 1)
            if use_multi_hop:
                pair_idx = i % len(pairs)
                a, b = pairs[pair_idx]
                context = (
                    f"Passage A: {kg.nodes[a].content}\n\n"
                    f"Passage B: {kg.nodes[b].content}"
                )
                prompt_text = (
                    f"Generate a multi-hop question-answer pair that requires "
                    f"information from BOTH passages below.\n\n{context}\n\n"
                    f'Respond with JSON: {{"question": "...", "answer": "...", "context": "..."}}'
                )
            else:
                node_idx = i % len(kg.nodes)
                context = kg.nodes[node_idx].content
                prompt_text = (
                    f"Generate a question-answer pair from the following passage.\n\n"
                    f"Passage: {context}\n\n"
                    f'Respond with JSON: {{"question": "...", "answer": "...", "context": "..."}}'
                )

            tasks.append(asyncio.ensure_future(
                self._generate_kg_case(prompt_text, strategy)
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("KG case generation failed: %s", result)
                continue
            if result is not None:
                cases.append(result)

        return cases

    async def _generate_kg_case(
        self, prompt: str, strategy: EvolutionStrategy
    ) -> Case | None:
        """Generate a single Case from a knowledge-graph prompt.

        Args:
            prompt: The prompt describing what case to generate.
            strategy: The evolution strategy to tag on the case metadata.

        Returns:
            A Case object, or None if parsing fails.
        """
        try:
            response = await self.judge.evaluate(prompt=prompt)
            raw = response.reasoning or response.raw_output or "{}"
            raw = raw.strip()
            for fence in ("```json", "```"):
                if raw.startswith(fence):
                    raw = raw[len(fence):]
                if raw.endswith("```"):
                    raw = raw[:-3]
            raw = raw.strip()
            parsed = json.loads(raw)
            question = str(parsed.get("question", ""))
            answer = str(parsed.get("answer", ""))
            context = str(parsed.get("context", ""))
            if not question:
                return None
            return Case(
                input=question,
                expected=answer,
                context=context,
                metadata={"strategy": strategy.value, "difficulty": "medium"},
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse KG case: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    async def _generate_batch(
        self,
        prompt: str,
        system_prompt: str,
        count: int,
        strategy: EvolutionStrategy | None = None,
    ) -> list[Case]:
        """Call the LLM and parse the response into a list of Cases.

        Retries up to ``config.max_retries`` times on parse failures.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response: JudgeResponse = await self.judge.evaluate(
                    prompt, system_prompt=system_prompt
                )
                self._total_cost += response.cost

                raw_text = response.raw_output or response.reasoning
                items = _extract_json_array(raw_text)

                cases = [_dict_to_case(item, strategy) for item in items]

                # Filter out obviously empty cases
                cases = [c for c in cases if c.input.strip()]

                if cases:
                    return cases[:count]

                last_error = ValueError("LLM returned no usable cases")
                logger.warning(
                    "Attempt %d/%d: LLM returned no usable cases",
                    attempt,
                    self.config.max_retries,
                )
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d/%d: failed to parse LLM output: %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )

        logger.error(
            "All %d attempts failed for batch of %d cases",
            self.config.max_retries,
            count,
        )
        raise RuntimeError(
            f"Failed to generate cases after {self.config.max_retries} attempts: "
            f"{last_error}"
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_document_prompt(
        self,
        documents: list[str],
        strategy: EvolutionStrategy,
        count: int,
    ) -> tuple[str, str]:
        """Build (user_prompt, system_prompt) for document-based synthesis."""
        system_prompt = (
            f"{_SYSTEM_PREAMBLE}\n"
            f"Strategy: {strategy.value}\n"
            f"{_STRATEGY_INSTRUCTIONS[strategy]}\n\n"
            f"Produce exactly {count} test case(s). Output ONLY a JSON array."
        )

        # Truncate and combine documents to fit context limits
        combined = self._combine_documents(documents)

        user_prompt = (
            f"Below are the source documents. Generate {count} test case(s) "
            f"using the '{strategy.value}' strategy.\n\n"
            f"--- DOCUMENTS ---\n{combined}\n--- END DOCUMENTS ---\n\n"
            f"Return a JSON array of {count} test case object(s)."
        )
        return user_prompt, system_prompt

    def _build_description_prompt(
        self,
        description: str,
        strategy: EvolutionStrategy,
        count: int,
    ) -> tuple[str, str]:
        """Build (user_prompt, system_prompt) for description-based synthesis."""
        system_prompt = (
            f"{_SYSTEM_PREAMBLE}\n"
            f"Strategy: {strategy.value}\n"
            f"{_STRATEGY_INSTRUCTIONS[strategy]}\n\n"
            f"Produce exactly {count} test case(s). Output ONLY a JSON array."
        )

        user_prompt = (
            f"Generate {count} test case(s) for the following system using the "
            f"'{strategy.value}' strategy.\n\n"
            f"--- SYSTEM DESCRIPTION ---\n{description}\n--- END DESCRIPTION ---\n\n"
            f"The test cases should cover realistic scenarios a user would "
            f"encounter with this system. For each case, invent plausible "
            f"context that the system might have access to, and provide a "
            f"reference answer that a correct system should produce.\n\n"
            f"Return a JSON array of {count} test case object(s)."
        )
        return user_prompt, system_prompt

    def _build_evolution_prompt(
        self,
        cases: list[Case],
        strategy: EvolutionStrategy,
    ) -> tuple[str, str]:
        """Build (user_prompt, system_prompt) for evolving existing cases."""
        system_prompt = (
            f"{_SYSTEM_PREAMBLE}\n"
            f"Strategy: {strategy.value}\n"
            f"{_STRATEGY_INSTRUCTIONS[strategy]}\n\n"
            f"You will receive existing test cases. Rewrite each one to make it "
            f"more challenging according to the strategy above. Preserve the "
            f"original topic but increase difficulty. Update the expected answer "
            f"and criteria to match the new, harder question.\n\n"
            f"Produce exactly {len(cases)} evolved test case(s). "
            f"Output ONLY a JSON array."
        )

        originals = json.dumps(
            [
                {
                    "input": c.input,
                    "expected": c.expected,
                    "context": c.context,
                    "criteria": c.criteria,
                    "metadata": c.metadata,
                }
                for c in cases
            ],
            indent=2,
        )

        user_prompt = (
            f"Evolve the following {len(cases)} test case(s) using the "
            f"'{strategy.value}' strategy. Make each case more challenging "
            f"while keeping it answerable.\n\n"
            f"--- ORIGINAL CASES ---\n{originals}\n--- END CASES ---\n\n"
            f"Return a JSON array of {len(cases)} evolved test case object(s)."
        )
        return user_prompt, system_prompt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_chunks(document: str, chunk_size: int) -> list[str]:
        """Split a document into chunks of approximately chunk_size characters.

        Splits on whitespace boundaries to avoid cutting words mid-way.

        Args:
            document: The document text to split.
            chunk_size: Target character count per chunk.

        Returns:
            A list of non-empty text chunks.
        """
        if not document:
            return []
        words = document.split()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_len + word_len > chunk_size and current:
                chunks.append(" ".join(current))
                current = [word]
                current_len = word_len
            else:
                current.append(word)
                current_len += word_len
        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _combine_documents(documents: list[str]) -> str:
        """Combine and truncate documents to fit within prompt limits."""
        if not documents:
            return "(no documents provided)"

        parts: list[str] = []
        total_chars = 0
        for i, doc in enumerate(documents, 1):
            header = f"[Document {i}]\n"
            remaining = _MAX_DOC_CHARS - total_chars - len(header)
            if remaining <= 0:
                parts.append(
                    f"... ({len(documents) - i + 1} more document(s) truncated)"
                )
                break
            if len(doc) > remaining:
                parts.append(header + doc[:remaining] + " [truncated]")
                total_chars += len(header) + remaining
            else:
                parts.append(header + doc)
                total_chars += len(header) + len(doc)

        return "\n\n".join(parts)

    @staticmethod
    def _distribute(
        total: int, strategies: list[EvolutionStrategy]
    ) -> list[tuple[EvolutionStrategy, int]]:
        """Split *total* cases as evenly as possible across *strategies*.

        Returns a list of (strategy, count) pairs.
        """
        if not strategies:
            return [(EvolutionStrategy.SIMPLE, total)]

        base, remainder = divmod(total, len(strategies))
        plan: list[tuple[EvolutionStrategy, int]] = []
        for i, strat in enumerate(strategies):
            count = base + (1 if i < remainder else 0)
            if count > 0:
                plan.append((strat, count))
        return plan


class SimulatedTurn(BaseModel):
    """A single turn in a simulated conversation."""

    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimulatedConversation(BaseModel):
    """A complete simulated multi-turn conversation."""

    turns: list[SimulatedTurn] = Field(default_factory=list)
    topic: str = ""
    persona: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def user_turns(self) -> list[SimulatedTurn]:
        """Return only the user turns."""
        return [t for t in self.turns if t.role == "user"]

    @property
    def assistant_turns(self) -> list[SimulatedTurn]:
        """Return only the assistant turns."""
        return [t for t in self.turns if t.role == "assistant"]

    def format_transcript(self) -> str:
        """Format the conversation as a plain-text transcript.

        Returns:
            A newline-joined string with each turn as ``role: content``.
        """
        lines = []
        for t in self.turns:
            lines.append(f"{t.role}: {t.content}")
        return "\n".join(lines)


class ConversationSimulator:
    """Generates multi-turn synthetic conversations for testing.

    Usage::

        sim = ConversationSimulator(judge=my_judge)
        conversations = await sim.agenerate(
            topic="customer support for a SaaS product",
            num_conversations=5,
            turns_per_conversation=6,
        )
    """

    def __init__(self, judge: JudgeBackend) -> None:
        self.judge = judge

    def generate(
        self,
        topic: str,
        num_conversations: int = 5,
        turns_per_conversation: int = 6,
        persona: str | None = None,
    ) -> list[SimulatedConversation]:
        """Generate conversations synchronously.

        Args:
            topic: The subject matter for the conversations.
            num_conversations: Number of conversations to generate.
            turns_per_conversation: Total turns (user + assistant) per conversation.
            persona: Optional description of the user persona. Defaults to a
                generic description based on *topic*.

        Returns:
            A list of SimulatedConversation objects.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    self.agenerate(
                        topic, num_conversations, turns_per_conversation, persona
                    ),
                ).result()
        return asyncio.run(
            self.agenerate(topic, num_conversations, turns_per_conversation, persona)
        )

    async def agenerate(
        self,
        topic: str,
        num_conversations: int = 5,
        turns_per_conversation: int = 6,
        persona: str | None = None,
    ) -> list[SimulatedConversation]:
        """Generate multi-turn conversations asynchronously.

        Args:
            topic: The subject matter for the conversations.
            num_conversations: Number of conversations to generate.
            turns_per_conversation: Total turns (user + assistant) per conversation.
            persona: Optional description of the user persona.

        Returns:
            A list of SimulatedConversation objects.
        """
        conversations = []
        user_persona = persona or f"A user asking about {topic}"

        for _ in range(num_conversations):
            conv = await self._simulate_one(topic, turns_per_conversation, user_persona)
            conversations.append(conv)
        return conversations

    async def _simulate_one(
        self,
        topic: str,
        num_turns: int,
        persona: str,
    ) -> SimulatedConversation:
        """Simulate a single multi-turn conversation.

        Args:
            topic: The subject matter of the conversation.
            num_turns: Total number of turns to generate.
            persona: The user persona description.

        Returns:
            A SimulatedConversation with alternating user/assistant turns.
        """
        turns: list[SimulatedTurn] = []
        history = ""

        for i in range(num_turns):
            if i % 2 == 0:
                content = await self._generate_user_turn(topic, persona, history, i)
                turns.append(SimulatedTurn(role="user", content=content))
            else:
                content = await self._generate_assistant_turn(topic, history)
                turns.append(SimulatedTurn(role="assistant", content=content))
            history = "\n".join(f"{t.role}: {t.content}" for t in turns)

        return SimulatedConversation(
            turns=turns,
            topic=topic,
            persona=persona,
        )

    async def _generate_user_turn(
        self,
        topic: str,
        persona: str,
        history: str,
        turn_idx: int,
    ) -> str:
        """Generate a single user turn via the judge.

        Args:
            topic: The conversation topic.
            persona: The user persona description.
            history: The conversation history so far.
            turn_idx: The zero-based index of this turn.

        Returns:
            The generated user message text.
        """
        prompt_parts = [
            f"You are simulating a user in a conversation about: {topic}",
            f"User persona: {persona}",
        ]
        if history:
            prompt_parts.append(f"Conversation so far:\n{history}")
        if turn_idx == 0:
            prompt_parts.append(
                "Generate the user's opening message. Be natural and specific."
            )
        else:
            prompt_parts.append(
                "Generate the user's next message. Follow up naturally on the conversation."
            )
        prompt_parts.append("Output ONLY the user's message, nothing else.")

        response = await self.judge.evaluate(prompt="\n\n".join(prompt_parts))
        return response.reasoning.strip()

    async def _generate_assistant_turn(self, topic: str, history: str) -> str:
        """Generate a single assistant turn via the judge.

        Args:
            topic: The conversation topic.
            history: The conversation history so far.

        Returns:
            The generated assistant message text.
        """
        prompt = (
            f"You are simulating a helpful assistant in a conversation about: {topic}\n\n"
            f"Conversation so far:\n{history}\n\n"
            f"Generate the assistant's response. Be helpful and natural.\n"
            f"Output ONLY the assistant's response, nothing else."
        )
        response = await self.judge.evaluate(prompt=prompt)
        return response.reasoning.strip()

"""RAG dataset generation: drop documents in, get diverse Q/A test cases out.

A Ragas-parity API that wraps the existing KG-based generation pipeline with
simple loaders (markdown, directories, PDF) and a declarative query
distribution. The goal is one-call test set creation for RAG evaluation.

Usage::

    from checkllm.rag_dataset import RAGDatasetGenerator, QueryDistribution
    from checkllm.judge import OpenAIJudge

    gen = RAGDatasetGenerator(judge=OpenAIJudge())
    cases = await gen.generate(
        documents=["doc1 text...", "doc2 text..."],
        num_cases=50,
        query_distribution=QueryDistribution(
            simple=0.4,
            reasoning=0.3,
            multi_context=0.2,
            conditional=0.1,
        ),
        personas=["novice", "expert", "skeptic"],
    )
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from checkllm.datasets.case import Case
from checkllm.judge import JudgeBackend
from checkllm.knowledge_graph import (
    KGTestGenerator,
    Persona,
    QueryStyle,
)

logger = logging.getLogger("checkllm.rag_dataset")

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 100
_DISTRIBUTION_TOLERANCE = 0.01


class DocumentChunk(BaseModel):
    """A single chunk produced by :func:`chunk_document`.

    Attributes:
        text: The chunk's raw text content.
        index: Zero-based position of this chunk within its source.
        source: Optional source identifier (filename, URL, or label).
        metadata: Arbitrary metadata attached to the chunk.
    """

    text: str
    index: int
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryDistribution(BaseModel):
    """Declarative distribution of query types to synthesize.

    The four proportions must sum to approximately 1.0 (within a tolerance of
    0.01). Each corresponds to a different RAG question style:

    - ``simple``: direct, single-hop fact lookup.
    - ``reasoning``: multi-step inferential questions over a single passage.
    - ``multi_context``: questions that require synthesizing facts across
      two or more chunks.
    - ``conditional``: conditional / hypothetical / edge-case questions.

    Attributes:
        simple: Fraction of cases to generate as simple single-hop queries.
        reasoning: Fraction requiring multi-step reasoning.
        multi_context: Fraction requiring multi-chunk synthesis.
        conditional: Fraction that are conditional or hypothetical.
    """

    simple: float = 0.4
    reasoning: float = 0.3
    multi_context: float = 0.2
    conditional: float = 0.1

    @field_validator("simple", "reasoning", "multi_context", "conditional")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"Query distribution proportions must be >= 0.0, got {v}")
        return v

    @model_validator(mode="after")
    def _check_sum(self) -> QueryDistribution:
        total = self.simple + self.reasoning + self.multi_context + self.conditional
        if abs(total - 1.0) > _DISTRIBUTION_TOLERANCE:
            raise ValueError(
                f"QueryDistribution proportions must sum to 1.0 "
                f"(±{_DISTRIBUTION_TOLERANCE}), got {total:.4f}"
            )
        return self

    def to_synthesizer_weights(self) -> dict[str, float]:
        """Map this distribution to KGTestGenerator synthesizer weights.

        The four Ragas-style labels are folded onto the three KG synthesizers:
        ``simple`` and ``conditional`` become ``single_hop`` (both are
        single-passage question types); ``reasoning`` becomes
        ``multi_hop_abstract``; and ``multi_context`` becomes
        ``multi_hop_specific``.

        Returns:
            A dict of synthesizer names to weights summing to 1.0 (with any
            zero entries omitted).
        """
        single_hop = self.simple + self.conditional
        weights = {
            "single_hop": single_hop,
            "multi_hop_abstract": self.reasoning,
            "multi_hop_specific": self.multi_context,
        }
        return {k: v for k, v in weights.items() if v > 0}


def chunk_document(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_CHUNK_OVERLAP,
    source: str | None = None,
) -> list[DocumentChunk]:
    """Split text into overlapping, whitespace-preserving chunks.

    Splits on word boundaries so tokens are not cut in half. Successive
    chunks share ``overlap`` characters to preserve local context.

    Args:
        text: The text to chunk.
        chunk_size: Target maximum character count per chunk.
        overlap: Number of characters of overlap between successive chunks.
        source: Optional source label recorded on each chunk.

    Returns:
        A list of ``DocumentChunk`` objects. Empty input yields an empty list.

    Raises:
        ValueError: If ``chunk_size`` is not positive or ``overlap`` is
            negative or at least ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    if not text or not text.strip():
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    idx = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            window = text[start:end]
            last_space = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
            if last_space > 0 and last_space > chunk_size // 2:
                end = start + last_space

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    index=idx,
                    source=source,
                    metadata={"start": start, "end": end},
                )
            )
            idx += 1

        if end >= text_len:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _strip_frontmatter(text: str) -> str:
    """Strip a leading YAML-style frontmatter block from markdown text.

    Frontmatter is delimited by lines containing only ``---`` at the very top
    of the document. If no frontmatter is found, the text is returned
    unchanged.

    Args:
        text: Markdown text that may or may not begin with frontmatter.

    Returns:
        The text with any leading frontmatter removed.
    """
    if not text.startswith("---"):
        return text

    pattern = re.compile(r"^---\s*\n.*?\n---\s*(?:\n|$)", re.DOTALL)
    match = pattern.match(text)
    if not match:
        return text
    return text[match.end() :].lstrip("\n")


def _persona_from_label(label: str) -> Persona:
    """Build a Persona from a short string label like ``"novice"``.

    Args:
        label: A free-form persona label.

    Returns:
        A ``Persona`` with a reasonable description and expertise level
        inferred from the label.
    """
    normalized = label.strip().lower()
    expertise_map = {
        "novice": "beginner",
        "beginner": "beginner",
        "student": "beginner",
        "layperson": "beginner",
        "intermediate": "intermediate",
        "practitioner": "intermediate",
        "expert": "expert",
        "specialist": "expert",
        "professional": "expert",
        "skeptic": "expert",
        "critic": "expert",
    }
    expertise = expertise_map.get(normalized, "intermediate")
    description = f"A {normalized} reader engaging with this material"
    return Persona(name=label, description=description, expertise_level=expertise)


class RAGDatasetGenerator:
    """One-call RAG evaluation dataset generation from documents.

    Wraps :class:`~checkllm.knowledge_graph.KGTestGenerator` with a
    Ragas-parity API: simple query distributions, convenience loaders for
    markdown / text / PDF / directories, and optional string-label personas.

    Attributes:
        judge: The LLM backend used to synthesize questions and answers.
        default_chunk_size: Default character count per chunk.
        default_chunk_overlap: Default character overlap between chunks.
    """

    def __init__(
        self,
        judge: JudgeBackend,
        default_chunk_size: int = _DEFAULT_CHUNK_SIZE,
        default_chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Construct a generator bound to an LLM judge.

        Args:
            judge: Any ``JudgeBackend`` used to call the LLM.
            default_chunk_size: Default target chunk size in characters.
            default_chunk_overlap: Default chunk overlap in characters.
        """
        self.judge = judge
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    async def generate(
        self,
        documents: list[str],
        num_cases: int = 20,
        query_distribution: QueryDistribution | None = None,
        personas: list[str] | list[Persona] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        sources: list[str] | None = None,
        styles: list[QueryStyle] | None = None,
    ) -> list[Case]:
        """Generate a RAG evaluation dataset from raw document strings.

        Chunks the documents, builds a knowledge graph, then distributes
        ``num_cases`` across the requested query types. Each generated
        :class:`~checkllm.datasets.case.Case` carries the source chunk(s) as
        context and rich metadata describing its origin.

        Args:
            documents: Raw text of each source document.
            num_cases: Target number of cases to produce.
            query_distribution: How to split the cases across query types.
                Defaults to 40% simple / 30% reasoning / 20% multi-context /
                10% conditional.
            personas: Optional list of persona labels (strings) or Persona
                objects to vary questions across user styles.
            chunk_size: Override for the default chunk size.
            chunk_overlap: Override for the default chunk overlap.
            sources: Optional list of source labels, one per document, used to
                populate ``metadata["source_document"]`` on each case.
            styles: Optional list of :class:`QueryStyle` values to cycle over.

        Returns:
            A list of ``Case`` objects with ``input``, ``expected``,
            ``context`` and ``metadata`` populated.

        Raises:
            ValueError: If ``documents`` is empty.
        """
        if not documents:
            raise ValueError("generate() requires at least one document")

        distribution = query_distribution or QueryDistribution()
        eff_chunk_size = chunk_size if chunk_size is not None else self.default_chunk_size
        eff_overlap = chunk_overlap if chunk_overlap is not None else self.default_chunk_overlap

        prepared_docs = self._prepare_documents(
            documents=documents,
            sources=sources,
            chunk_size=eff_chunk_size,
            overlap=eff_overlap,
        )

        persona_list = self._normalize_personas(personas)

        synth_weights = distribution.to_synthesizer_weights()
        if not synth_weights:
            synth_weights = {"single_hop": 1.0}

        kg_gen = KGTestGenerator(judge=self.judge)
        samples = await kg_gen.generate(
            documents=prepared_docs,
            num_samples=num_cases,
            synthesizers=synth_weights,
            personas=persona_list,
            styles=styles,
        )

        source_by_doc_index: dict[int, str | None] = {}
        if sources:
            for i, src in enumerate(sources):
                source_by_doc_index[i] = src

        cases: list[Case] = []
        label_for_type = self._query_type_labels(distribution)
        for sample in samples:
            context = "\n\n".join(sample.contexts) if sample.contexts else None
            qtype_label = label_for_type.get(sample.query_type, sample.query_type)
            source_doc = self._infer_source_document(sample.source_nodes, source_by_doc_index)
            difficulty = self._difficulty_for(qtype_label)

            metadata: dict[str, Any] = {
                "query_type": qtype_label,
                "source_document": source_doc,
                "persona": sample.persona.name if sample.persona else None,
                "difficulty": difficulty,
                "style": sample.style.value,
                "length": sample.length.value,
                "source_nodes": sample.source_nodes,
                "strategy": "rag_dataset_v1",
            }

            cases.append(
                Case(
                    input=sample.query,
                    expected=sample.reference_answer,
                    context=context,
                    metadata=metadata,
                )
            )

        logger.info(
            "RAGDatasetGenerator: produced %d/%d cases across %d documents",
            len(cases),
            num_cases,
            len(documents),
        )
        return cases

    async def from_text_files(
        self,
        paths: list[str | os.PathLike[str]],
        num_cases: int = 20,
        query_distribution: QueryDistribution | None = None,
        personas: list[str] | list[Persona] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding: str = "utf-8",
    ) -> list[Case]:
        """Generate cases by reading plain text files from disk.

        Args:
            paths: File system paths to read as plain text.
            num_cases: Target number of cases to produce.
            query_distribution: Optional query distribution.
            personas: Optional persona labels or Persona objects.
            chunk_size: Override for the default chunk size.
            chunk_overlap: Override for the default chunk overlap.
            encoding: Text encoding for each file.

        Returns:
            A list of generated ``Case`` objects.

        Raises:
            FileNotFoundError: If any path does not exist.
        """
        documents: list[str] = []
        sources: list[str] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f"Text file not found: {path}")
            documents.append(path.read_text(encoding=encoding))
            sources.append(str(path))

        return await self.generate(
            documents=documents,
            num_cases=num_cases,
            query_distribution=query_distribution,
            personas=personas,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sources=sources,
        )

    async def from_markdown_files(
        self,
        paths: list[str | os.PathLike[str]],
        num_cases: int = 20,
        query_distribution: QueryDistribution | None = None,
        personas: list[str] | list[Persona] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding: str = "utf-8",
    ) -> list[Case]:
        """Generate cases from markdown files, stripping YAML frontmatter.

        Args:
            paths: File system paths to markdown files.
            num_cases: Target number of cases to produce.
            query_distribution: Optional query distribution.
            personas: Optional persona labels or Persona objects.
            chunk_size: Override for the default chunk size.
            chunk_overlap: Override for the default chunk overlap.
            encoding: Text encoding for each file.

        Returns:
            A list of generated ``Case`` objects.

        Raises:
            FileNotFoundError: If any path does not exist.
        """
        documents: list[str] = []
        sources: list[str] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f"Markdown file not found: {path}")
            raw = path.read_text(encoding=encoding)
            documents.append(_strip_frontmatter(raw))
            sources.append(str(path))

        return await self.generate(
            documents=documents,
            num_cases=num_cases,
            query_distribution=query_distribution,
            personas=personas,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sources=sources,
        )

    async def from_directory(
        self,
        directory: str | os.PathLike[str],
        glob: str = "**/*.md",
        num_cases: int = 20,
        query_distribution: QueryDistribution | None = None,
        personas: list[str] | list[Persona] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding: str = "utf-8",
    ) -> list[Case]:
        """Generate cases from every file matching a glob beneath a directory.

        Markdown files (``.md``, ``.markdown``) have frontmatter stripped;
        other files are treated as plain text.

        Args:
            directory: Root directory to search.
            glob: Glob pattern relative to ``directory``.
            num_cases: Target number of cases.
            query_distribution: Optional query distribution.
            personas: Optional persona labels or Persona objects.
            chunk_size: Override for the default chunk size.
            chunk_overlap: Override for the default chunk overlap.
            encoding: Text encoding for each file.

        Returns:
            A list of generated ``Case`` objects.

        Raises:
            NotADirectoryError: If ``directory`` is not an existing directory.
            ValueError: If the glob matches no files.
        """
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        matches = sorted(p for p in root.glob(glob) if p.is_file())
        if not matches:
            raise ValueError(f"No files matched glob {glob!r} under {root}")

        documents: list[str] = []
        sources: list[str] = []
        markdown_exts = {".md", ".markdown"}
        for path in matches:
            raw = path.read_text(encoding=encoding)
            if path.suffix.lower() in markdown_exts:
                raw = _strip_frontmatter(raw)
            documents.append(raw)
            sources.append(str(path))

        return await self.generate(
            documents=documents,
            num_cases=num_cases,
            query_distribution=query_distribution,
            personas=personas,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sources=sources,
        )

    async def from_pdf(
        self,
        path: str | os.PathLike[str],
        num_cases: int = 20,
        query_distribution: QueryDistribution | None = None,
        personas: list[str] | list[Persona] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Case]:
        """Generate cases from a single PDF file.

        Requires the optional ``pypdf`` dependency. If it is not installed,
        raises ``ImportError`` with an install hint instead of silently
        degrading.

        Args:
            path: Path to the PDF file.
            num_cases: Target number of cases.
            query_distribution: Optional query distribution.
            personas: Optional persona labels or Persona objects.
            chunk_size: Override for the default chunk size.
            chunk_overlap: Override for the default chunk overlap.

        Returns:
            A list of generated ``Case`` objects.

        Raises:
            ImportError: If ``pypdf`` is not installed.
            FileNotFoundError: If the PDF file does not exist.
        """
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF loading requires the optional 'pypdf' package. "
                "Install it with: pip install pypdf"
            ) from exc

        pdf_path = Path(path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:
                logger.warning("Failed to extract text from a PDF page: %s", exc)
                pages.append("")
        document_text = "\n\n".join(p for p in pages if p.strip())

        return await self.generate(
            documents=[document_text],
            num_cases=num_cases,
            query_distribution=query_distribution,
            personas=personas,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sources=[str(pdf_path)],
        )

    def _prepare_documents(
        self,
        documents: list[str],
        sources: list[str] | None,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Chunk each document upfront and return per-chunk strings.

        Pre-chunking here means the downstream KG sentence splitter receives
        already-bounded pieces and the resulting chunks respect the requested
        ``chunk_size`` / ``overlap`` settings.

        Args:
            documents: Raw document strings.
            sources: Optional matching source labels.
            chunk_size: Target chunk size in characters.
            overlap: Character overlap between chunks.

        Returns:
            A list of chunk strings ready to be passed to the KG pipeline.
        """
        prepared: list[str] = []
        for i, doc in enumerate(documents):
            source = sources[i] if sources and i < len(sources) else f"doc-{i}"
            chunks = chunk_document(
                text=doc,
                chunk_size=chunk_size,
                overlap=overlap,
                source=source,
            )
            if not chunks:
                continue
            prepared.extend(c.text for c in chunks)
        if not prepared:
            prepared = [d for d in documents if d.strip()]
        return prepared

    def _normalize_personas(
        self,
        personas: list[str] | list[Persona] | None,
    ) -> list[Persona] | None:
        """Coerce a mixed-type persona list into ``list[Persona]``.

        Args:
            personas: Strings, Persona objects, or ``None``.

        Returns:
            A list of ``Persona`` objects, or ``None`` if input was ``None``.
        """
        if personas is None:
            return None
        if not personas:
            return None
        normalized: list[Persona] = []
        for p in personas:
            if isinstance(p, Persona):
                normalized.append(p)
            else:
                normalized.append(_persona_from_label(str(p)))
        return normalized

    @staticmethod
    def _query_type_labels(distribution: QueryDistribution) -> dict[str, str]:
        """Map KG synthesizer labels back to Ragas-style type names.

        The mapping loses the ``simple`` vs ``conditional`` distinction
        (both folded onto ``single_hop``). When the distribution specifies
        only one of the two, we report that label; when both are requested,
        we default to ``"simple"``.

        Args:
            distribution: The original user-facing query distribution.

        Returns:
            A dict from KG synthesizer name to the corresponding public label.
        """
        if distribution.simple > 0 and distribution.conditional == 0:
            single_label = "simple"
        elif distribution.conditional > 0 and distribution.simple == 0:
            single_label = "conditional"
        else:
            single_label = "simple"
        return {
            "single_hop": single_label,
            "multi_hop_abstract": "reasoning",
            "multi_hop_specific": "multi_context",
        }

    @staticmethod
    def _difficulty_for(query_type: str) -> str:
        """Assign a difficulty bucket to a user-facing query type.

        Args:
            query_type: One of ``simple``, ``reasoning``, ``multi_context``,
                or ``conditional``.

        Returns:
            A difficulty label: ``"easy"``, ``"medium"`` or ``"hard"``.
        """
        if query_type == "simple":
            return "easy"
        if query_type in ("reasoning", "multi_context"):
            return "medium"
        if query_type == "conditional":
            return "hard"
        return "medium"

    @staticmethod
    def _infer_source_document(
        source_nodes: list[str],
        source_by_doc_index: dict[int, str | None],
    ) -> str | None:
        """Recover a user-facing source label from KG node ids.

        The KG pipeline uses node ids of the form ``doc-<i>`` (plus
        ``chunk-doc-<i>-<n>``). When the caller supplied ``sources=[...]``
        we map those indices back to the original labels.

        Args:
            source_nodes: Source node ids attached to a synthesized sample.
            source_by_doc_index: User-supplied source labels keyed by index.

        Returns:
            A source label string, or ``None`` if it cannot be resolved.
        """
        if not source_nodes or not source_by_doc_index:
            return None
        for node_id in source_nodes:
            match = re.search(r"doc-(\d+)", node_id)
            if not match:
                continue
            idx = int(match.group(1))
            if idx in source_by_doc_index:
                return source_by_doc_index[idx]
        return None


__all__ = [
    "DocumentChunk",
    "QueryDistribution",
    "RAGDatasetGenerator",
    "chunk_document",
]

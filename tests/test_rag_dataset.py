"""Tests for the checkllm.rag_dataset module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from checkllm.datasets.case import Case
from checkllm.knowledge_graph import Persona, QueryStyle
from checkllm.models import JudgeResponse
from checkllm.rag_dataset import (
    DocumentChunk,
    QueryDistribution,
    RAGDatasetGenerator,
    chunk_document,
)


def _make_mock_judge(responses: list[str] | None = None) -> AsyncMock:
    """Build an AsyncMock judge that cycles through deterministic responses.

    Args:
        responses: Optional list of raw JSON strings. If None, a single
            valid Q/A response is repeated.

    Returns:
        An AsyncMock suitable for use as a ``JudgeBackend``.
    """
    if responses is None:
        responses = ['{"question": "What is X?", "answer": "X is the answer."}']

    judge = AsyncMock()
    counter = {"i": 0}

    async def _evaluate(prompt: str = "", system_prompt: str | None = None) -> JudgeResponse:
        raw = responses[counter["i"] % len(responses)]
        counter["i"] += 1
        return JudgeResponse(
            score=1.0,
            reasoning=raw,
            raw_output=raw,
            cost=0.001,
        )

    judge.evaluate.side_effect = _evaluate
    return judge


class TestChunkDocument:
    def test_empty_input_returns_empty(self):
        assert chunk_document("") == []
        assert chunk_document("   \n  ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world."
        chunks = chunk_document(text, chunk_size=1000, overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].index == 0

    def test_chunks_respect_size(self):
        text = "word " * 500  # ~2500 characters
        chunks = chunk_document(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 4
        for c in chunks:
            assert len(c.text) <= 500

    def test_chunks_have_overlap(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 20
        chunks = chunk_document(text, chunk_size=200, overlap=50)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            tail = chunks[i].text[-30:]
            head = chunks[i + 1].text[:80]
            # At least one overlapping word should survive the split
            tail_words = set(tail.split())
            head_words = set(head.split())
            assert tail_words & head_words, f"Expected overlap between chunk {i} and {i + 1}"

    def test_preserves_word_boundaries(self):
        text = "Supercalifragilisticexpialidocious is a very long invented word."
        chunks = chunk_document(text, chunk_size=40, overlap=0)
        # No chunk should cut the famous word mid-way
        for c in chunks:
            assert (
                "Supercalifragilisticexpialidocious" in c.text or "Super" not in c.text.split()[0]
            )

    def test_source_label_propagated(self):
        chunks = chunk_document("some text here", chunk_size=100, overlap=0, source="docA.md")
        assert all(c.source == "docA.md" for c in chunks)

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_document("hello", chunk_size=0, overlap=0)

    def test_overlap_larger_than_chunk_raises(self):
        with pytest.raises(ValueError):
            chunk_document("hello", chunk_size=50, overlap=50)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_document("hello", chunk_size=50, overlap=-1)

    def test_document_chunk_model_fields(self):
        chunk = DocumentChunk(text="foo", index=3, source="src", metadata={"k": 1})
        assert chunk.text == "foo"
        assert chunk.index == 3
        assert chunk.source == "src"
        assert chunk.metadata["k"] == 1


class TestQueryDistribution:
    def test_default_sums_to_one(self):
        dist = QueryDistribution()
        total = dist.simple + dist.reasoning + dist.multi_context + dist.conditional
        assert abs(total - 1.0) < 1e-9

    def test_valid_custom_distribution(self):
        dist = QueryDistribution(simple=0.5, reasoning=0.3, multi_context=0.1, conditional=0.1)
        assert dist.simple == 0.5

    def test_rejects_bad_sum(self):
        with pytest.raises(ValueError):
            QueryDistribution(simple=0.5, reasoning=0.5, multi_context=0.5, conditional=0.0)

    def test_rejects_negative_proportion(self):
        with pytest.raises(ValueError):
            QueryDistribution(simple=-0.1, reasoning=0.6, multi_context=0.3, conditional=0.2)

    def test_tolerance_allows_rounding(self):
        # 0.4 + 0.3 + 0.2 + 0.095 = 0.995, within 0.01 tolerance
        dist = QueryDistribution(simple=0.4, reasoning=0.3, multi_context=0.2, conditional=0.095)
        assert dist.conditional == 0.095

    def test_rejects_outside_tolerance(self):
        with pytest.raises(ValueError):
            QueryDistribution(simple=0.4, reasoning=0.3, multi_context=0.2, conditional=0.05)

    def test_to_synthesizer_weights_maps_correctly(self):
        dist = QueryDistribution(simple=0.4, reasoning=0.3, multi_context=0.2, conditional=0.1)
        weights = dist.to_synthesizer_weights()
        assert weights["single_hop"] == pytest.approx(0.5)
        assert weights["multi_hop_abstract"] == pytest.approx(0.3)
        assert weights["multi_hop_specific"] == pytest.approx(0.2)

    def test_to_synthesizer_weights_omits_zero(self):
        dist = QueryDistribution(simple=1.0, reasoning=0.0, multi_context=0.0, conditional=0.0)
        weights = dist.to_synthesizer_weights()
        assert "multi_hop_abstract" not in weights
        assert "multi_hop_specific" not in weights
        assert weights["single_hop"] == pytest.approx(1.0)


class TestRAGDatasetGeneratorGenerate:
    @pytest.mark.asyncio
    async def test_generate_produces_cases(self):
        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)
        docs = [
            "Paris is the capital of France. It sits on the Seine river.",
            "The Eiffel Tower is located in Paris, France. It was built in 1889.",
        ]
        cases = await gen.generate(
            documents=docs,
            num_cases=4,
            query_distribution=QueryDistribution(
                simple=1.0, reasoning=0.0, multi_context=0.0, conditional=0.0
            ),
        )
        assert all(isinstance(c, Case) for c in cases)
        assert len(cases) <= 4
        assert all(c.input for c in cases)
        assert all("query_type" in c.metadata for c in cases)

    @pytest.mark.asyncio
    async def test_empty_documents_raises(self):
        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)
        with pytest.raises(ValueError):
            await gen.generate(documents=[], num_cases=1)

    @pytest.mark.asyncio
    async def test_personas_as_strings_are_normalized(self):
        persona_payload = json.dumps(
            {
                "personas": [
                    {"name": "n", "description": "d", "expertise_level": "beginner"},
                ]
            }
        )
        judge = _make_mock_judge(
            [
                persona_payload,
                '{"question": "Q?", "answer": "A."}',
                '{"question": "Q2?", "answer": "A2."}',
            ]
        )
        gen = RAGDatasetGenerator(judge=judge)
        cases = await gen.generate(
            documents=["Some informative text to chunk. It has two sentences."],
            num_cases=2,
            query_distribution=QueryDistribution(
                simple=1.0, reasoning=0.0, multi_context=0.0, conditional=0.0
            ),
            personas=["novice", "expert"],
        )
        assert len(cases) <= 2
        personas_used = {c.metadata.get("persona") for c in cases}
        # At least one case should reference one of our persona labels
        assert personas_used.intersection({"novice", "expert"}) or personas_used == {None}

    @pytest.mark.asyncio
    async def test_case_metadata_fields(self):
        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)
        cases = await gen.generate(
            documents=["A short but meaningful document about testing."],
            num_cases=1,
            query_distribution=QueryDistribution(
                simple=1.0, reasoning=0.0, multi_context=0.0, conditional=0.0
            ),
            sources=["mydoc.txt"],
        )
        assert len(cases) >= 1
        md = cases[0].metadata
        assert md["query_type"] in {"simple", "reasoning", "multi_context", "conditional"}
        assert md["difficulty"] in {"easy", "medium", "hard"}
        assert "style" in md
        assert md["strategy"] == "rag_dataset_v1"


class TestRAGDatasetGeneratorLoaders:
    @pytest.mark.asyncio
    async def test_from_text_files(self, tmp_path: Path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("Alpha discussing widgets for widget testing.", encoding="utf-8")
        b.write_text("Beta discussing gadgets for gadget review.", encoding="utf-8")

        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)
        cases = await gen.from_text_files(
            [a, b],
            num_cases=2,
            query_distribution=QueryDistribution(
                simple=1.0, reasoning=0.0, multi_context=0.0, conditional=0.0
            ),
        )
        assert len(cases) <= 2
        sources = {c.metadata.get("source_document") for c in cases}
        assert any(s and ("a.txt" in s or "b.txt" in s) for s in sources if s)

    @pytest.mark.asyncio
    async def test_from_text_files_missing_raises(self, tmp_path: Path):
        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)
        with pytest.raises(FileNotFoundError):
            await gen.from_text_files([tmp_path / "missing.txt"], num_cases=1)

    @pytest.mark.asyncio
    async def test_from_markdown_strips_frontmatter(self, tmp_path: Path):
        md_path = tmp_path / "guide.md"
        md_path.write_text(
            "---\ntitle: Test\nauthor: me\n---\n\n# Heading\n\nThis is the body text.\n",
            encoding="utf-8",
        )

        captured: dict[str, list[str]] = {}

        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)

        original_generate = gen.generate

        async def _capture(documents, **kwargs):
            captured["documents"] = list(documents)
            return await original_generate(documents=documents, **kwargs)

        gen.generate = _capture  # type: ignore[assignment]

        await gen.from_markdown_files(
            [md_path],
            num_cases=1,
            query_distribution=QueryDistribution(
                simple=1.0, reasoning=0.0, multi_context=0.0, conditional=0.0
            ),
        )

        docs = captured["documents"]
        joined = "\n".join(docs)
        assert "title: Test" not in joined
        assert "author: me" not in joined
        assert "Heading" in joined or "body text" in joined

    @pytest.mark.asyncio
    async def test_from_directory_respects_glob(self, tmp_path: Path):
        (tmp_path / "keep.md").write_text("# Keep me\n\nRelevant content here.", encoding="utf-8")
        (tmp_path / "skip.txt").write_text("Plain text file to skip", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("Nested markdown doc.", encoding="utf-8")

        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)

        captured: dict[str, list[str]] = {}
        original_generate = gen.generate

        async def _capture(documents, **kwargs):
            captured["documents"] = list(documents)
            captured["sources"] = list(kwargs.get("sources") or [])
            return await original_generate(documents=documents, **kwargs)

        gen.generate = _capture  # type: ignore[assignment]

        await gen.from_directory(
            tmp_path,
            glob="**/*.md",
            num_cases=1,
            query_distribution=QueryDistribution(
                simple=1.0, reasoning=0.0, multi_context=0.0, conditional=0.0
            ),
        )

        sources = captured["sources"]
        joined_sources = " ".join(sources)
        assert "keep.md" in joined_sources
        assert "nested.md" in joined_sources
        assert "skip.txt" not in joined_sources

    @pytest.mark.asyncio
    async def test_from_directory_empty_match_raises(self, tmp_path: Path):
        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)
        with pytest.raises(ValueError):
            await gen.from_directory(tmp_path, glob="*.nope", num_cases=1)

    @pytest.mark.asyncio
    async def test_from_directory_rejects_non_directory(self, tmp_path: Path):
        file_path = tmp_path / "f.md"
        file_path.write_text("hello", encoding="utf-8")
        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)
        with pytest.raises(NotADirectoryError):
            await gen.from_directory(file_path, glob="*.md", num_cases=1)


class TestFromPDF:
    @pytest.mark.asyncio
    async def test_from_pdf_without_pypdf_raises_importerror(self, monkeypatch, tmp_path):
        # Force pypdf import to fail even if installed
        monkeypatch.setitem(sys.modules, "pypdf", None)

        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)
        dummy = tmp_path / "x.pdf"
        dummy.write_bytes(b"not-a-real-pdf")
        with pytest.raises(ImportError) as exc:
            await gen.from_pdf(dummy, num_cases=1)
        assert "pypdf" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_from_pdf_missing_file_raises(self, tmp_path):
        pytest.importorskip("pypdf")
        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)
        with pytest.raises(FileNotFoundError):
            await gen.from_pdf(tmp_path / "nope.pdf", num_cases=1)


class TestPersonaNormalization:
    @pytest.mark.asyncio
    async def test_persona_objects_pass_through(self):
        persona = Persona(name="expert", description="d", expertise_level="expert")
        judge = _make_mock_judge()
        gen = RAGDatasetGenerator(judge=judge)
        cases = await gen.generate(
            documents=["Small but useful text."],
            num_cases=1,
            query_distribution=QueryDistribution(
                simple=1.0, reasoning=0.0, multi_context=0.0, conditional=0.0
            ),
            personas=[persona],
        )
        assert len(cases) >= 1
        persona_names = {c.metadata.get("persona") for c in cases}
        assert persona_names == {"expert"} or persona_names == {None}

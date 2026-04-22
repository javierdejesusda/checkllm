"""Tests for checkllm.rag_poison -- poisoned RAG document generation."""

from __future__ import annotations

import json

import pytest

from checkllm.models import JudgeResponse
from checkllm.rag_poison import (
    PoisonedDocGenerator,
    PoisonedDocument,
    PoisonType,
    _assign_default_severity,
)


class _StubJudge:
    """Judge stub that returns JSON-formatted poisoned document content."""

    def __init__(
        self,
        content: str = "This is a poisoned document about the topic.",
        detection_hint: str = "Look for injected instructions",
        severity: str = "high",
    ) -> None:
        self._content = content
        self._detection_hint = detection_hint
        self._severity = severity
        self.call_count = 0

    async def evaluate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> JudgeResponse:
        self.call_count += 1
        raw = json.dumps(
            {
                "content": f"{self._content} (variant {self.call_count})",
                "detection_hint": self._detection_hint,
                "severity": self._severity,
            }
        )
        return JudgeResponse(
            score=0.5,
            reasoning="generated",
            raw_output=raw,
            cost=0.002,
        )


class _RawTextJudge:
    """Judge stub that returns non-JSON raw text."""

    async def evaluate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> JudgeResponse:
        return JudgeResponse(
            score=0.5,
            reasoning="raw text fallback",
            raw_output="This is raw text, not JSON.",
            cost=0.001,
        )


class TestPoisonType:
    def test_enum_values(self):
        assert PoisonType.PROMPT_INJECTION.value == "prompt_injection"
        assert PoisonType.MISINFORMATION.value == "misinformation"
        assert PoisonType.DATA_EXFILTRATION.value == "data_exfiltration"
        assert PoisonType.CONTEXT_MANIPULATION.value == "context_manipulation"
        assert PoisonType.AUTHORITY_SPOOFING.value == "authority_spoofing"
        assert PoisonType.INSTRUCTION_OVERRIDE.value == "instruction_override"
        assert PoisonType.TOPIC_DRIFT.value == "topic_drift"
        assert PoisonType.HALLUCINATION_BAIT.value == "hallucination_bait"

    def test_enum_count(self):
        assert len(PoisonType) == 8


class TestPoisonedDocument:
    def test_create_document(self):
        doc = PoisonedDocument(
            content="Poisoned content here",
            poison_type=PoisonType.PROMPT_INJECTION,
            topic="security",
            poison_description="Hidden prompt injection",
            detection_hint="Look for override instructions",
            severity="high",
        )
        assert doc.content == "Poisoned content here"
        assert doc.poison_type == PoisonType.PROMPT_INJECTION
        assert doc.topic == "security"
        assert doc.severity == "high"
        assert doc.clean_version is None

    def test_with_clean_version(self):
        doc = PoisonedDocument(
            content="Modified content",
            poison_type=PoisonType.MISINFORMATION,
            topic="finance",
            poison_description="Subtle factual errors",
            detection_hint="Check statistics",
            severity="medium",
            clean_version="Original clean content",
        )
        assert doc.clean_version == "Original clean content"

    def test_defaults(self):
        doc = PoisonedDocument(
            content="test",
            poison_type=PoisonType.TOPIC_DRIFT,
            topic="test",
            poison_description="test",
            detection_hint="test",
        )
        assert doc.severity == "medium"
        assert doc.clean_version is None


class TestPoisonedDocGenerator:
    @pytest.mark.asyncio
    async def test_generates_correct_number(self):
        judge = _StubJudge()
        gen = PoisonedDocGenerator(judge=judge)
        docs = await gen.generate(
            topic="cybersecurity",
            poison_type=PoisonType.PROMPT_INJECTION,
            num_documents=3,
        )
        assert len(docs) == 3
        assert judge.call_count == 3

    @pytest.mark.asyncio
    async def test_generates_single_document(self):
        judge = _StubJudge()
        gen = PoisonedDocGenerator(judge=judge)
        docs = await gen.generate(
            topic="medicine",
            poison_type=PoisonType.MISINFORMATION,
            num_documents=1,
        )
        assert len(docs) == 1
        assert docs[0].poison_type == PoisonType.MISINFORMATION
        assert docs[0].topic == "medicine"

    @pytest.mark.asyncio
    async def test_each_poison_type(self):
        judge = _StubJudge()
        gen = PoisonedDocGenerator(judge=judge)
        for ptype in PoisonType:
            docs = await gen.generate(
                topic="general",
                poison_type=ptype,
                num_documents=1,
            )
            assert len(docs) == 1
            assert docs[0].poison_type == ptype

    @pytest.mark.asyncio
    async def test_documents_have_content(self):
        judge = _StubJudge(content="Detailed document about AI safety")
        gen = PoisonedDocGenerator(judge=judge)
        docs = await gen.generate(
            topic="AI safety",
            poison_type=PoisonType.HALLUCINATION_BAIT,
            num_documents=2,
        )
        for doc in docs:
            assert len(doc.content) > 0
            assert doc.detection_hint != ""
            assert doc.poison_description != ""

    @pytest.mark.asyncio
    async def test_base_documents_poisoning(self):
        judge = _StubJudge()
        gen = PoisonedDocGenerator(judge=judge)
        base_docs = [
            "Clean document about Python programming.",
            "Another clean document about data science.",
        ]
        docs = await gen.generate(
            topic="programming",
            poison_type=PoisonType.CONTEXT_MANIPULATION,
            base_documents=base_docs,
        )
        assert len(docs) == 2
        for i, doc in enumerate(docs):
            assert doc.clean_version == base_docs[i]
            assert doc.poison_type == PoisonType.CONTEXT_MANIPULATION

    @pytest.mark.asyncio
    async def test_base_documents_ignores_num(self):
        judge = _StubJudge()
        gen = PoisonedDocGenerator(judge=judge)
        base_docs = ["doc1", "doc2", "doc3"]
        docs = await gen.generate(
            topic="test",
            poison_type=PoisonType.TOPIC_DRIFT,
            num_documents=10,
            base_documents=base_docs,
        )
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_severity_from_response(self):
        judge = _StubJudge(severity="critical")
        gen = PoisonedDocGenerator(judge=judge)
        docs = await gen.generate(
            topic="security",
            poison_type=PoisonType.DATA_EXFILTRATION,
            num_documents=1,
        )
        assert docs[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_fallback_on_non_json_response(self):
        judge = _RawTextJudge()
        gen = PoisonedDocGenerator(judge=judge)
        docs = await gen.generate(
            topic="fallback test",
            poison_type=PoisonType.PROMPT_INJECTION,
            num_documents=1,
        )
        assert len(docs) == 1
        assert docs[0].content == "This is raw text, not JSON."
        assert docs[0].severity == "high"

    @pytest.mark.asyncio
    async def test_documents_are_distinct(self):
        judge = _StubJudge()
        gen = PoisonedDocGenerator(judge=judge)
        docs = await gen.generate(
            topic="finance",
            poison_type=PoisonType.AUTHORITY_SPOOFING,
            num_documents=3,
        )
        contents = [d.content for d in docs]
        assert len(set(contents)) == 3


class TestAssignDefaultSeverity:
    def test_high_severity_types(self):
        assert _assign_default_severity(PoisonType.PROMPT_INJECTION) == "high"
        assert _assign_default_severity(PoisonType.DATA_EXFILTRATION) == "high"
        assert _assign_default_severity(PoisonType.INSTRUCTION_OVERRIDE) == "high"

    def test_medium_severity_types(self):
        assert _assign_default_severity(PoisonType.MISINFORMATION) == "medium"
        assert _assign_default_severity(PoisonType.AUTHORITY_SPOOFING) == "medium"
        assert _assign_default_severity(PoisonType.HALLUCINATION_BAIT) == "medium"

    def test_low_severity_types(self):
        assert _assign_default_severity(PoisonType.CONTEXT_MANIPULATION) == "low"
        assert _assign_default_severity(PoisonType.TOPIC_DRIFT) == "low"

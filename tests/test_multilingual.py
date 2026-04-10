"""Tests for the multilingual prompt adaptation module."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

from checkllm.multilingual import (
    PromptAdapter,
    PromptTemplate,
    SupportedLanguage,
    TranslatedPrompt,
    detect_language,
)
from checkllm.testing import MockJudge


class TestDetectLanguage:
    """Tests for Unicode-based language detection."""

    def test_detect_chinese(self):
        assert detect_language("这是一个中文句子") == "zh"

    def test_detect_japanese(self):
        assert detect_language("これは日本語のテストです") == "ja"

    def test_detect_korean(self):
        assert detect_language("이것은 한국어 테스트입니다") == "ko"

    def test_detect_arabic(self):
        assert detect_language("هذا اختبار باللغة العربية") == "ar"

    def test_detect_hindi(self):
        assert detect_language("यह हिंदी में एक परीक्षण है") == "hi"

    def test_detect_russian(self):
        assert detect_language("Это тест на русском языке") == "ru"

    def test_detect_thai(self):
        assert detect_language("นี่คือการทดสอบภาษาไทย") == "th"

    def test_detect_hebrew(self):
        assert detect_language("זהו מבחן בעברית") == "he"

    def test_detect_english(self):
        assert detect_language("This is an English sentence for testing") == "en"

    def test_detect_spanish(self):
        assert detect_language("Esta es una prueba en español como la prueba por ejemplo") == "es"

    def test_detect_french(self):
        assert detect_language("Ceci est un test dans la langue avec des mots pour le test") == "fr"

    def test_detect_german(self):
        assert detect_language("Das ist ein Test auf Deutsch und der Text ist nicht kurz") == "de"

    def test_detect_empty(self):
        assert detect_language("") == "en"

    def test_detect_whitespace_only(self):
        assert detect_language("   ") == "en"

    def test_supported_language_enum(self):
        assert SupportedLanguage.ENGLISH.value == "en"
        assert SupportedLanguage.JAPANESE.value == "ja"
        assert SupportedLanguage.ARABIC.value == "ar"
        assert len(SupportedLanguage) == 20


class TestPromptTemplate:
    """Tests for PromptTemplate model."""

    def test_basic_template(self):
        template = PromptTemplate(
            name="faithfulness",
            instruction="Rate whether the answer is faithful to the context.",
        )
        assert template.name == "faithfulness"
        assert template.language == "en"
        assert template.few_shot_examples == []

    def test_template_with_examples(self):
        template = PromptTemplate(
            name="relevance",
            instruction="Rate the relevance of the answer.",
            few_shot_examples=[
                {"input": "What is Python?", "output": "A programming language."}
            ],
        )
        assert len(template.few_shot_examples) == 1

    def test_template_serialization(self):
        template = PromptTemplate(
            name="test",
            instruction="Test instruction",
            language="es",
        )
        data = template.model_dump()
        restored = PromptTemplate(**data)
        assert restored == template


class TestPromptAdapter:
    """Tests for PromptAdapter translation and caching."""

    @pytest.fixture
    def mock_judge(self):
        return MockJudge(default_score=0.9, default_reasoning="Translated text here")

    @pytest.fixture
    def adapter(self, mock_judge):
        return PromptAdapter(judge=mock_judge)

    @pytest.fixture
    def sample_template(self):
        return PromptTemplate(
            name="faithfulness",
            instruction="Rate whether the answer is faithful to the context.",
            few_shot_examples=[
                {"input": "Is the sky blue?", "answer": "Yes, the sky is blue."}
            ],
        )

    @pytest.mark.asyncio
    async def test_adapt_basic(self, adapter, sample_template):
        result = await adapter.adapt(sample_template, target_language="es")
        assert isinstance(result, TranslatedPrompt)
        assert result.target_language == "es"
        assert result.original == sample_template
        assert result.translated.language == "es"
        assert result.translated.name == "faithfulness"
        assert result.translation_quality is not None

    @pytest.mark.asyncio
    async def test_adapt_uses_cache(self, adapter, sample_template, mock_judge):
        calls_before = len(mock_judge.calls)
        result1 = await adapter.adapt(sample_template, target_language="fr")
        calls_after_first = len(mock_judge.calls)
        result2 = await adapter.adapt(sample_template, target_language="fr")
        calls_after_second = len(mock_judge.calls)
        assert result1 is result2
        assert calls_after_first > calls_before
        assert calls_after_second == calls_after_first

    @pytest.mark.asyncio
    async def test_cache_distinguishes_different_instructions(self, adapter):
        t1 = PromptTemplate(name="metric", instruction="Instruction A")
        t2 = PromptTemplate(name="metric", instruction="Instruction B")
        r1 = await adapter.adapt(t1, target_language="es")
        r2 = await adapter.adapt(t2, target_language="es")
        assert r1 is not r2
        assert r1.original.instruction != r2.original.instruction

    @pytest.mark.asyncio
    async def test_cache_distinguishes_adapt_flags(self, adapter, sample_template):
        r1 = await adapter.adapt(sample_template, target_language="de", adapt_instruction=True)
        r2 = await adapter.adapt(sample_template, target_language="de", adapt_instruction=False)
        assert r1 is not r2

    @pytest.mark.asyncio
    async def test_adapt_no_instruction(self, adapter, sample_template):
        result = await adapter.adapt(
            sample_template,
            target_language="de",
            adapt_instruction=False,
        )
        assert result.translated.instruction == sample_template.instruction

    @pytest.mark.asyncio
    async def test_adapt_no_examples(self, adapter, sample_template):
        result = await adapter.adapt(
            sample_template,
            target_language="ja",
            adapt_examples=False,
        )
        assert result.translated.few_shot_examples == sample_template.few_shot_examples

    @pytest.mark.asyncio
    async def test_adapt_batch(self, adapter):
        templates = [
            PromptTemplate(name="t1", instruction="First instruction"),
            PromptTemplate(name="t2", instruction="Second instruction"),
            PromptTemplate(name="t3", instruction="Third instruction"),
        ]
        results = await adapter.adapt_batch(templates, target_language="ko")
        assert len(results) == 3
        for r in results:
            assert isinstance(r, TranslatedPrompt)
            assert r.target_language == "ko"

    @pytest.mark.asyncio
    async def test_detect_language_method(self, adapter):
        lang = await adapter.detect_language("これは日本語のテストです")
        assert lang == "ja"

    @pytest.mark.asyncio
    async def test_detect_language_english(self, adapter):
        lang = await adapter.detect_language(
            "This is a longer English sentence with enough words"
        )
        assert lang == "en"


class TestTranslationPersistence:
    """Tests for save/load of translation caches."""

    @pytest.fixture
    def mock_judge(self):
        return MockJudge(default_score=0.92, default_reasoning="Texto traducido")

    @pytest.mark.asyncio
    async def test_save_and_load(self, mock_judge):
        adapter = PromptAdapter(judge=mock_judge)
        template = PromptTemplate(
            name="test_metric",
            instruction="Evaluate the quality of the response.",
        )

        await adapter.adapt(template, target_language="es")
        assert len(adapter._cache) == 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "translations.json")
            adapter.save_translations(path)

            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["name"] == "test_metric"
            assert data[0]["target_language"] == "es"

            new_adapter = PromptAdapter(judge=mock_judge)
            assert len(new_adapter._cache) == 0
            new_adapter.load_translations(path)
            assert len(new_adapter._cache) == 1

            loaded_values = list(new_adapter._cache.values())
            assert len(loaded_values) == 1
            loaded = loaded_values[0]
            assert loaded.target_language == "es"
            assert loaded.original.name == "test_metric"

    @pytest.mark.asyncio
    async def test_save_multiple_languages(self, mock_judge):
        adapter = PromptAdapter(judge=mock_judge)
        template = PromptTemplate(name="m1", instruction="Test")

        await adapter.adapt(template, target_language="es")
        await adapter.adapt(template, target_language="fr")
        await adapter.adapt(template, target_language="de")

        assert len(adapter._cache) == 3

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "multi.json")
            adapter.save_translations(path)

            new_adapter = PromptAdapter(judge=mock_judge)
            new_adapter.load_translations(path)
            assert len(new_adapter._cache) == 3

    def test_load_nonexistent_file(self, mock_judge):
        adapter = PromptAdapter(judge=mock_judge)
        with pytest.raises(FileNotFoundError):
            adapter.load_translations("/nonexistent/path.json")

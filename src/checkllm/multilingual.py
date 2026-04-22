"""Multilingual prompt adaptation for evaluation across languages.

Translates evaluation prompts to any target language using LLM-based
translation and caches results for reuse.  Supports 20 built-in languages
with automatic detection via Unicode character-range analysis.

Usage::

    from checkllm.multilingual import PromptAdapter, PromptTemplate
    from checkllm.judge import OpenAIJudge

    adapter = PromptAdapter(judge=OpenAIJudge())

    template = PromptTemplate(
        name="faithfulness",
        instruction="Rate whether the answer is faithful to the context.",
    )

    translated = await adapter.adapt(template, target_language="es")
    print(translated.translated.instruction)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from checkllm.judge import JudgeBackend

logger = logging.getLogger("checkllm.multilingual")


class SupportedLanguage(str, Enum):
    """Languages with built-in support (others work via auto-translation)."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    RUSSIAN = "ru"
    ITALIAN = "it"
    DUTCH = "nl"
    TURKISH = "tr"
    VIETNAMESE = "vi"
    THAI = "th"
    INDONESIAN = "id"
    POLISH = "pl"
    SWEDISH = "sv"
    HEBREW = "he"


_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "pl": "Polish",
    "sv": "Swedish",
    "he": "Hebrew",
}


class PromptTemplate(BaseModel):
    """A translatable evaluation prompt template."""

    name: str
    instruction: str
    few_shot_examples: list[dict[str, str]] = Field(default_factory=list)
    language: str = "en"


class TranslatedPrompt(BaseModel):
    """A prompt translated to a target language."""

    original: PromptTemplate
    translated: PromptTemplate
    target_language: str
    translation_quality: float | None = None


def _resolve_language_name(code_or_name: str) -> str:
    """Resolve a language code or name to its human-readable name.

    Args:
        code_or_name: ISO 639-1 code (e.g. ``"es"``) or language name.

    Returns:
        Human-readable language name.
    """
    code = code_or_name.strip().lower()
    if code in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[code]
    for lang_code, name in _LANGUAGE_NAMES.items():
        if name.lower() == code:
            return name
    return code_or_name.strip().title()


def _count_script_chars(text: str) -> dict[str, int]:
    """Count characters by Unicode script category.

    Args:
        text: Input text to analyze.

    Returns:
        Mapping of script name to character count.
    """
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isspace() or ch in ".,;:!?\"'()-[]{}0123456789":
            continue
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            name = ""
        script = _script_from_name(name, ch)
        counts[script] = counts.get(script, 0) + 1
    return counts


def _script_from_name(char_name: str, ch: str) -> str:
    """Derive the script category from a Unicode character name.

    Args:
        char_name: The Unicode character name.
        ch: The character itself.

    Returns:
        A script label string.
    """
    cp = ord(ch)

    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "CJK"
    if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
        return "JAPANESE_KANA"
    if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
        return "KOREAN"
    if 0x0600 <= cp <= 0x06FF or 0xFB50 <= cp <= 0xFDFF:
        return "ARABIC"
    if 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:
        return "HEBREW"
    if 0x0900 <= cp <= 0x097F:
        return "DEVANAGARI"
    if 0x0400 <= cp <= 0x04FF:
        return "CYRILLIC"
    if 0x0E00 <= cp <= 0x0E7F:
        return "THAI"
    if "LATIN" in char_name:
        return "LATIN"

    return "OTHER"


def _detect_cjk_language(text: str, counts: dict[str, int]) -> str:
    """Disambiguate Chinese, Japanese, and Korean from CJK characters.

    Uses the presence of kana (Japanese) or Hangul (Korean) as strong
    signals.  Falls back to Chinese if only CJK ideographs are found.

    Args:
        text: Input text.
        counts: Pre-computed script character counts.

    Returns:
        ISO 639-1 language code.
    """
    if counts.get("JAPANESE_KANA", 0) > 0:
        return "ja"
    if counts.get("KOREAN", 0) > 0:
        return "ko"
    return "zh"


def _detect_latin_language(text: str) -> str:
    """Heuristically identify a Latin-script language.

    Uses common short words and diacritical patterns unique to specific
    European languages.

    Args:
        text: Input text (Latin script).

    Returns:
        ISO 639-1 language code.
    """
    lower = text.lower()
    words = set(re.findall(r"\b\w+\b", lower))

    spanish_markers = {
        "el",
        "la",
        "los",
        "las",
        "es",
        "un",
        "una",
        "por",
        "como",
        "pero",
        "esto",
    }
    french_markers = {
        "le",
        "la",
        "les",
        "des",
        "est",
        "une",
        "que",
        "dans",
        "avec",
        "pour",
    }
    german_markers = {
        "der",
        "die",
        "das",
        "und",
        "ist",
        "ein",
        "eine",
        "auf",
        "nicht",
        "mit",
    }
    portuguese_markers = {"o", "os", "uma", "das", "como", "mais", "muito", "isso"}
    italian_markers = {"il", "gli", "che", "della", "sono", "nella", "questo", "anche"}
    dutch_markers = {"de", "het", "een", "van", "dat", "niet", "met", "voor"}
    turkish_markers = {"bir", "ve", "bu", "ile", "olan", "gibi", "daha"}
    polish_markers = {"jest", "nie", "tak", "jak", "aby", "przez", "tego"}
    swedish_markers = {"och", "att", "det", "som", "med", "inte", "har"}
    vietnamese_markers = {"trong", "cua", "nhung", "duoc", "khong", "nhu"}
    indonesian_markers = {"dan", "yang", "ini", "itu", "dengan", "untuk", "dari"}

    scores: list[tuple[str, int]] = [
        ("es", len(words & spanish_markers)),
        ("fr", len(words & french_markers)),
        ("de", len(words & german_markers)),
        ("pt", len(words & portuguese_markers)),
        ("it", len(words & italian_markers)),
        ("nl", len(words & dutch_markers)),
        ("tr", len(words & turkish_markers)),
        ("pl", len(words & polish_markers)),
        ("sv", len(words & swedish_markers)),
        ("vi", len(words & vietnamese_markers)),
        ("id", len(words & indonesian_markers)),
    ]

    scores.sort(key=lambda x: x[1], reverse=True)
    if scores and scores[0][1] >= 2:
        return scores[0][0]

    return "en"


def detect_language(text: str) -> str:
    """Detect the language of input text using Unicode character analysis.

    Examines Unicode character ranges to classify the dominant script, then
    applies heuristic word-matching for Latin-script languages.

    Args:
        text: The text to analyze.

    Returns:
        ISO 639-1 language code (e.g. ``"en"``, ``"zh"``, ``"ar"``).
    """
    if not text or not text.strip():
        return "en"

    counts = _count_script_chars(text)
    if not counts:
        return "en"

    dominant = max(counts, key=lambda k: counts[k])

    script_map: dict[str, str] = {
        "ARABIC": "ar",
        "HEBREW": "he",
        "DEVANAGARI": "hi",
        "CYRILLIC": "ru",
        "THAI": "th",
    }

    if dominant in script_map:
        return script_map[dominant]

    if dominant in ("CJK", "JAPANESE_KANA", "KOREAN"):
        return _detect_cjk_language(text, counts)

    if dominant == "LATIN":
        return _detect_latin_language(text)

    return "en"


class PromptAdapter:
    """Adapts evaluation prompts to any target language.

    Uses an LLM judge backend to translate prompt templates and caches
    results for efficient reuse.

    Args:
        judge: The LLM judge backend used for translation calls.

    Usage::

        from checkllm.multilingual import PromptAdapter
        from checkllm.judge import OpenAIJudge

        adapter = PromptAdapter(judge=OpenAIJudge())

        translated = await adapter.adapt(
            template=faithfulness_prompt,
            target_language="es",
        )

        translated_all = await adapter.adapt_batch(
            templates=[faithfulness_prompt, relevance_prompt],
            target_language="ja",
        )

        adapter.save_translations("translations/es.json")
        adapter.load_translations("translations/es.json")
    """

    def __init__(self, judge: JudgeBackend) -> None:
        self.judge = judge
        self._cache: dict[tuple[str, str, str, bool, bool], TranslatedPrompt] = {}

    async def adapt(
        self,
        template: PromptTemplate,
        target_language: str,
        adapt_instruction: bool = True,
        adapt_examples: bool = True,
    ) -> TranslatedPrompt:
        """Translate a prompt template to the target language.

        Args:
            template: The prompt template to translate.
            target_language: ISO 639-1 language code or language name.
            adapt_instruction: Whether to translate the instruction text.
            adapt_examples: Whether to translate few-shot examples.

        Returns:
            TranslatedPrompt with both original and translated versions.
        """
        cache_key = (
            template.name,
            target_language,
            template.instruction,
            adapt_instruction,
            adapt_examples,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        lang_name = _resolve_language_name(target_language)

        translated_instruction = template.instruction
        if adapt_instruction:
            translated_instruction = await self._translate_text(template.instruction, lang_name)

        translated_examples: list[dict[str, str]] = []
        if adapt_examples and template.few_shot_examples:
            translated_examples = await self._translate_examples(
                template.few_shot_examples, lang_name
            )
        elif not adapt_examples:
            translated_examples = list(template.few_shot_examples)

        quality = await self._assess_quality(
            template.instruction, translated_instruction, lang_name
        )

        translated_template = PromptTemplate(
            name=template.name,
            instruction=translated_instruction,
            few_shot_examples=translated_examples,
            language=target_language,
        )

        result = TranslatedPrompt(
            original=template,
            translated=translated_template,
            target_language=target_language,
            translation_quality=quality,
        )
        self._cache[cache_key] = result
        return result

    async def adapt_batch(
        self,
        templates: list[PromptTemplate],
        target_language: str,
    ) -> list[TranslatedPrompt]:
        """Translate multiple templates concurrently.

        Args:
            templates: List of prompt templates to translate.
            target_language: ISO 639-1 language code or language name.

        Returns:
            List of TranslatedPrompt results, one per input template.
        """
        tasks = [self.adapt(t, target_language) for t in templates]
        return list(await asyncio.gather(*tasks))

    async def detect_language(self, text: str) -> str:
        """Detect the language of input text.

        Uses Unicode character-range heuristics first.  Falls back to the
        LLM judge for ambiguous cases (e.g. very short text with no strong
        script signal).

        Args:
            text: The text to analyze.

        Returns:
            ISO 639-1 language code.
        """
        detected = detect_language(text)
        if detected != "en" or not text.strip():
            return detected

        word_count = len(text.split())
        if word_count <= 3:
            return await self._llm_detect_language(text)

        return detected

    def save_translations(self, path: str) -> None:
        """Save cached translations to a JSON file.

        Args:
            path: File path for the JSON output.
        """
        data: list[dict[str, Any]] = []
        for (name, lang, instruction, ai, ae), tp in self._cache.items():
            data.append(
                {
                    "name": name,
                    "target_language": lang,
                    "instruction": instruction,
                    "adapt_instruction": ai,
                    "adapt_examples": ae,
                    "original": tp.original.model_dump(),
                    "translated": tp.translated.model_dump(),
                    "translation_quality": tp.translation_quality,
                }
            )

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d translations to %s", len(data), path)

    def load_translations(self, path: str) -> None:
        """Load translations from a JSON file into the cache.

        Args:
            path: File path of the JSON translation file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        raw = json.loads(file_path.read_text(encoding="utf-8"))

        for entry in raw:
            original = PromptTemplate(**entry["original"])
            translated = PromptTemplate(**entry["translated"])
            tp = TranslatedPrompt(
                original=original,
                translated=translated,
                target_language=entry["target_language"],
                translation_quality=entry.get("translation_quality"),
            )
            cache_key = (
                entry["name"],
                entry["target_language"],
                entry.get("instruction", original.instruction),
                entry.get("adapt_instruction", True),
                entry.get("adapt_examples", True),
            )
            self._cache[cache_key] = tp

        logger.info("Loaded %d translations from %s", len(raw), path)

    async def _translate_text(self, text: str, target_language: str) -> str:
        """Translate text to the target language via the LLM judge.

        Args:
            text: Source text to translate.
            target_language: Human-readable language name.

        Returns:
            Translated text string.
        """
        prompt = (
            f"Translate the following text to {target_language}. "
            f"Preserve the meaning, tone, and any technical terms. "
            f"Output ONLY the translation, nothing else.\n\n"
            f"Text:\n{text}"
        )
        response = await self.judge.evaluate(prompt=prompt)
        result = response.reasoning.strip()
        return result if result else text

    async def _translate_examples(
        self,
        examples: list[dict[str, str]],
        target_language: str,
    ) -> list[dict[str, str]]:
        """Translate few-shot examples to the target language.

        Args:
            examples: List of example dictionaries with string values.
            target_language: Human-readable language name.

        Returns:
            List of translated example dictionaries.
        """
        translated: list[dict[str, str]] = []
        for example in examples:
            translated_example: dict[str, str] = {}
            for key, value in example.items():
                translated_value = await self._translate_text(value, target_language)
                translated_example[key] = translated_value
            translated.append(translated_example)
        return translated

    async def _assess_quality(
        self,
        original: str,
        translated: str,
        target_language: str,
    ) -> float:
        """Assess the quality of a translation on a 0-1 scale.

        Args:
            original: Original text.
            translated: Translated text.
            target_language: Human-readable language name.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        prompt = (
            f"Rate the quality of this translation on a 0.0-1.0 scale.\n\n"
            f"Original (English):\n{original}\n\n"
            f"Translation ({target_language}):\n{translated}\n\n"
            f"Consider: accuracy, fluency, preservation of meaning.\n"
            f'Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}'
        )
        response = await self.judge.evaluate(prompt=prompt)
        return max(0.0, min(1.0, response.score))

    async def _llm_detect_language(self, text: str) -> str:
        """Use the LLM to detect language for ambiguous text.

        Parses the response by searching for a valid two-letter language
        code, handling cases where the model returns extra text.

        Args:
            text: Text to classify.

        Returns:
            ISO 639-1 language code.
        """
        prompt = (
            f"Detect the language of the following text. "
            f"Respond with ONLY the ISO 639-1 two-letter language code "
            f"(e.g. 'en', 'es', 'fr').\n\n"
            f"Text: {text}"
        )
        response = await self.judge.evaluate(prompt=prompt)
        raw = response.reasoning.strip().lower()

        match = re.search(r"\b([a-z]{2})\b", raw)
        if match:
            candidate = match.group(1)
            valid_codes = {lang.value for lang in SupportedLanguage}
            if candidate in valid_codes:
                return candidate

        name_to_code = {name.lower(): code for code, name in _LANGUAGE_NAMES.items()}
        for name, code in name_to_code.items():
            if name in raw:
                return code

        return "en"

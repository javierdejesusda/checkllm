"""Verify check methods populate threshold and input_preview on results."""

from checkllm.config import CheckllmConfig
from checkllm.check import CheckCollector


class TestResultContext:
    def test_contains_populates_input_preview(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        result = c.contains("hello world", "xyz")
        assert result.input_preview is not None
        assert "hello world" in result.input_preview

    def test_max_tokens_populates_input_preview(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        result = c.max_tokens("short text", 100)
        assert result.input_preview is not None

    def test_similarity_populates_threshold(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        result = c.similarity("hello", "hallo", threshold=0.9)
        assert result.threshold == 0.9
        assert result.input_preview is not None

    def test_bleu_populates_threshold(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        result = c.bleu("the cat sat", "the cat sat", threshold=0.5)
        assert result.threshold == 0.5

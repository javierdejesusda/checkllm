"""Verify check module decomposition maintains API compatibility."""

from checkllm.config import CheckllmConfig
from checkllm.check import CheckCollector


class TestCheckStructure:
    def test_deterministic_methods_still_on_collector(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        deterministic_methods = [
            "contains",
            "not_contains",
            "max_tokens",
            "min_tokens",
            "latency",
            "cost",
            "json_schema",
            "regex",
            "exact_match",
            "starts_with",
            "ends_with",
            "word_count",
            "char_count",
            "similarity",
            "readability",
            "sentence_count",
            "all_of",
            "any_of",
            "none_of",
            "is_json",
            "is_valid_python",
            "no_pii",
            "language",
            "greater_than",
            "less_than",
            "between",
            "bleu",
            "rouge_l",
            "json_field",
            "is_valid_sql",
            "is_valid_markdown",
        ]
        for method_name in deterministic_methods:
            assert hasattr(c, method_name), f"Missing method: {method_name}"
            assert callable(getattr(c, method_name))

    def test_judge_methods_still_on_collector(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        judge_methods = [
            "hallucination",
            "relevance",
            "toxicity",
            "rubric",
            "fluency",
            "coherence",
            "sentiment",
            "correctness",
            "faithfulness",
            "context_relevance",
            "answer_completeness",
            "instruction_following",
            "summarization",
            "bias",
            "consistency",
            "groundedness",
            "g_eval",
            "contextual_precision",
            "contextual_recall",
            "task_completion",
            "role_adherence",
            "tool_accuracy",
            "knowledge_retention",
            "conversation_completeness",
        ]
        for method_name in judge_methods:
            assert hasattr(c, method_name), f"Missing method: {method_name}"
            assert callable(getattr(c, method_name))

    def test_infrastructure_methods_on_collector(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        infra = [
            "that",
            "expect",
            "teardown",
            "total_cost",
            "cache_stats",
            "run_metric",
            "aflush",
            "_get_judge",
            "_cached_judge_check",
        ]
        for method_name in infra:
            assert hasattr(c, method_name), f"Missing method: {method_name}"

    def test_async_judge_methods_on_collector(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        async_methods = [
            "ahallucination",
            "arelevance",
            "atoxicity",
            "arubric",
            "afluency",
            "acoherence",
            "asentiment",
            "acorrectness",
        ]
        for method_name in async_methods:
            assert hasattr(c, method_name), f"Missing method: {method_name}"

"""Integration test for Phase A competitive features."""
from checkllm.redteam import VulnerabilityType, AttackStrategy
from checkllm.compare import ProviderMatrix, MatrixResult
from checkllm.benchmarks import list_benchmarks, load_benchmark
from checkllm.config import CheckllmConfig
from checkllm.check import CheckCollector


class TestPhaseAIntegration:
    def test_52_vulnerability_types(self):
        assert len(VulnerabilityType) >= 52

    def test_owasp_mapping_complete(self):
        from checkllm.redteam import get_owasp_mapping
        mapping = get_owasp_mapping()
        assert len(mapping) >= 52

    def test_14_attack_strategies(self):
        assert len(AttackStrategy) >= 14

    def test_benchmarks_available(self):
        names = list_benchmarks()
        assert "mmlu" in names
        assert "truthfulqa" in names
        assert "gsm8k" in names

    def test_benchmark_loads(self):
        ds = load_benchmark("mmlu", limit=3)
        assert len(ds.samples) == 3

    def test_provider_matrix_importable(self):
        assert ProviderMatrix is not None
        assert MatrixResult is not None

    def test_new_agentic_metrics_on_collector(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        for method in ["plan_quality", "goal_accuracy", "step_efficiency",
                       "argument_correctness", "plan_adherence"]:
            assert hasattr(c, method), f"Missing: {method}"
            assert callable(getattr(c, method))

    def test_new_safety_metrics_on_collector(self):
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        for method in ["pii_detection", "misuse_detection",
                       "role_violation", "non_advice"]:
            assert hasattr(c, method), f"Missing: {method}"
            assert callable(getattr(c, method))

    def test_total_metrics_on_collector(self):
        """checkllm now has 33+ judge methods."""
        config = CheckllmConfig()
        c = CheckCollector(config=config)
        all_judge_methods = [
            "hallucination", "relevance", "toxicity", "rubric",
            "fluency", "coherence", "sentiment", "correctness",
            "faithfulness", "context_relevance", "answer_completeness",
            "instruction_following", "summarization", "bias",
            "consistency", "groundedness", "g_eval",
            "contextual_precision", "contextual_recall",
            "task_completion", "role_adherence", "tool_accuracy",
            "knowledge_retention", "conversation_completeness",
            "plan_quality", "goal_accuracy", "step_efficiency",
            "argument_correctness", "plan_adherence",
            "pii_detection", "misuse_detection", "role_violation", "non_advice",
        ]
        for method in all_judge_methods:
            assert hasattr(c, method), f"Missing: {method}"
        assert len(all_judge_methods) >= 33

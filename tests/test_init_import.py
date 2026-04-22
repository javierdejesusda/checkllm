"""Tests that trigger checkllm.__init__.py imports for coverage."""

from __future__ import annotations


class TestCheckllmInit:
    """Tests that import from checkllm package directly to trigger __init__.py."""

    def test_import_checkllm(self):
        """Importing checkllm covers __init__.py."""
        import checkllm

        assert checkllm is not None

    def test_core_exports_available(self):
        """Verify core exports from __init__.py are accessible."""
        import checkllm

        # Core model classes
        assert hasattr(checkllm, "CheckResult")
        assert hasattr(checkllm, "CheckFailedError")
        assert hasattr(checkllm, "JudgeResponse")

    def test_judge_classes_available(self):
        """Verify judge backend classes are accessible."""
        import checkllm

        assert hasattr(checkllm, "JudgeBackend")
        assert hasattr(checkllm, "JudgeConfigError")

    def test_tracing_exports(self):
        """Verify tracing exports from __init__.py."""
        import checkllm

        assert hasattr(checkllm, "Span")
        assert hasattr(checkllm, "Tracer")
        assert hasattr(checkllm, "get_tracer")
        assert hasattr(checkllm, "trace")

    def test_deprecations_exports(self):
        """Verify deprecation warning classes."""
        import checkllm

        assert hasattr(checkllm, "CheckllmDeprecationWarning")
        assert hasattr(checkllm, "CheckllmRemovedIn5Warning")
        assert hasattr(checkllm, "CheckllmRemovedIn6Warning")
        assert hasattr(checkllm, "deprecated")

    def test_conversation_exports(self):
        """Verify conversation classes from __init__.py."""
        import checkllm

        assert hasattr(checkllm, "ConversationalTestCase")
        assert hasattr(checkllm, "Turn")

    def test_chain_export(self):
        """Verify AssertionChain is exported."""
        import checkllm

        assert hasattr(checkllm, "AssertionChain")

    def test_streaming_exports(self):
        """Verify streaming exports from __init__.py."""
        import checkllm

        assert hasattr(checkllm, "StreamingCheckpoint")
        assert hasattr(checkllm, "StreamingEvaluator")

    def test_arena_exports(self):
        """Verify arena exports from __init__.py."""
        import checkllm

        assert hasattr(checkllm, "Arena")
        assert hasattr(checkllm, "ArenaCandidate")
        assert hasattr(checkllm, "ArenaResult")

    def test_resilience_exports(self):
        """Verify resilience classes are exported."""
        import checkllm

        assert hasattr(checkllm, "CircuitBreaker")
        assert hasattr(checkllm, "ResilientJudge")
        assert hasattr(checkllm, "RetryPolicy")

    def test_errors_exports(self):
        """Verify error utilities from __init__.py."""
        import checkllm

        assert hasattr(checkllm, "format_budget_error")
        assert hasattr(checkllm, "format_missing_dependency_error")

    def test_redteam_exports(self):
        """Verify red team exports from __init__.py."""
        import checkllm

        assert hasattr(checkllm, "VulnerabilityType")
        assert hasattr(checkllm, "StrategyType")
        assert hasattr(checkllm, "get_strategy")
        assert hasattr(checkllm, "apply_strategies")

    def test_trajectory_exports(self):
        """Verify trajectory validator exports."""
        import checkllm

        assert hasattr(checkllm, "TrajectoryValidator")
        assert hasattr(checkllm, "validate_trajectory")

    def test_testing_exports(self):
        """Verify testing helpers from __init__.py."""
        import checkllm

        assert hasattr(checkllm, "MockJudge")
        assert hasattr(checkllm, "assert_all_passed")
        assert hasattr(checkllm, "make_collector")

    def test_dpo_export(self):
        """Verify DPO export classes available through init."""
        from checkllm.dpo import DPOExporter, DPOPair

        assert DPOExporter is not None
        assert DPOPair is not None

    def test_batch_exports(self):
        """Verify batch processing exports."""
        import checkllm

        assert hasattr(checkllm, "BatchJob")
        assert hasattr(checkllm, "BatchStatus")

    def test_guardrails_exports(self):
        """Verify guardrails exports from __init__.py."""
        import checkllm

        assert hasattr(checkllm, "Guard")
        assert hasattr(checkllm, "guardrail")
        assert hasattr(checkllm, "GuardrailError")

    def test_observe_exports(self):
        """Verify observe module exports from __init__.py."""
        import checkllm

        assert hasattr(checkllm, "observe")
        assert hasattr(checkllm, "get_observe_trace")
        assert hasattr(checkllm, "clear_trace")

    def test_agents_exports(self):
        """Verify agent utilities from __init__.py."""
        import checkllm

        assert hasattr(checkllm, "AgentStep")
        assert hasattr(checkllm, "AgentTestCase")
        assert hasattr(checkllm, "ToolCall")
        assert hasattr(checkllm, "validate_tool_calls")

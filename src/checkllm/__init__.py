"""checkllm - Test LLM-powered applications with the same rigor as traditional software."""

# Always-available imports (no openai dependency)
from checkllm.agents import (
    AgentStep,
    AgentTestCase,
    ToolCall,
    validate_no_repeated_tools,
    validate_tool_calls,
    validate_tool_order,
    validate_trajectory_length,
)
from checkllm.deprecations import (
    CheckllmDeprecationWarning,
    CheckllmRemovedIn5Warning,
    CheckllmRemovedIn6Warning,
    deprecated,
)
from checkllm.chain import AssertionChain
from checkllm.conversation import ConversationalTestCase, Turn
from checkllm.datasets.case import Case
from checkllm.errors import format_budget_error, format_missing_dependency_error
from checkllm.guardrails import (
    CheckSpec,
    Guard,
    GuardrailError,
    GuardrailMiddleware,
    ValidationResult,
    guardrail,
)
from checkllm.judge import JudgeBackend, JudgeConfigError
from checkllm.metrics import metric
from checkllm.models import CheckFailedError, CheckResult, JudgeResponse
from checkllm.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    PerProviderRateLimiter,
    ResilientJudge,
    RetryPolicy,
    TokenBucketRateLimiter,
    with_retry,
)
from checkllm.tracing import Span, Tracer, get_tracer, trace

# Imports that may require optional dependencies (openai, etc.)
try:
    from checkllm.api import Evaluator, check_output, evaluate, parse_check_shorthand
    from checkllm.batch import BatchEvaluator, BatchJob, BatchStatus
    from checkllm.cache import JudgeCache
    from checkllm.consensus import (
        AggregationStrategy,
        ConsensusJudge,
        ConsensusResult,
        consensus,
    )
    from checkllm.discovery import detect_judge_backend, format_no_judge_error
    from checkllm.embeddings import (
        CachedEmbeddings,
        OpenAIEmbeddings,
        batch_semantic_similarity,
        cosine_similarity,
        semantic_similarity,
    )
    from checkllm.engines import (
        AsyncEngine,
        EngineType,
        HybridEngine,
        ProcessPoolEngine,
        ThreadPoolEngine,
        create_engine,
    )
    from checkllm.estimator import CostEstimate, estimate_check_cost, estimate_from_test_file
    from checkllm.experiments import ExperimentComparison, ExperimentRun, ExperimentTracker
    from checkllm.history import RunHistory
    from checkllm.judge import AnthropicJudge, OpenAIJudge
    from checkllm.providers import (
        AzureOpenAIJudge,
        CustomHTTPJudge,
        GeminiJudge,
        LiteLLMJudge,
        OllamaJudge,
        create_judge,
    )
    from checkllm.pytest_plugin import dataset
    from checkllm.redteam import (
        AttackResult,
        AttackStrategy,
        RedTeamer,
        VulnerabilityReport,
        VulnerabilityType,
    )
    from checkllm.streaming import StreamingCheckpoint, StreamingEvaluator
    from checkllm.synthesizer import EvolutionStrategy, SynthesisConfig, Synthesizer
    from checkllm.testing import MockJudge, assert_all_passed, assert_score_above, make_collector
    from checkllm.yaml_config import EvalConfig, YamlEvalRunner, load_eval_config
except ImportError:
    pass  # Optional dependencies not installed

__version__ = "4.0.0"

__all__ = [
    # Deprecation framework
    "CheckllmDeprecationWarning",
    "CheckllmRemovedIn5Warning",
    "CheckllmRemovedIn6Warning",
    "deprecated",
    # Agent evaluation
    "AgentStep",
    "AgentTestCase",
    "ToolCall",
    "validate_no_repeated_tools",
    "validate_tool_calls",
    "validate_tool_order",
    "validate_trajectory_length",
    # Batch API
    "BatchEvaluator",
    "BatchJob",
    "BatchStatus",
    # Chain
    "AssertionChain",
    # Consensus
    "AggregationStrategy",
    "ConsensusJudge",
    "ConsensusResult",
    "consensus",
    # Conversation
    "ConversationalTestCase",
    "Turn",
    # Core
    "Case",
    "CheckFailedError",
    "CheckResult",
    "JudgeCache",
    "JudgeResponse",
    "MockJudge",
    "RunHistory",
    # Embeddings
    "CachedEmbeddings",
    "OpenAIEmbeddings",
    "batch_semantic_similarity",
    "cosine_similarity",
    "semantic_similarity",
    # Discovery
    "detect_judge_backend",
    "format_no_judge_error",
    # Engines
    "AsyncEngine",
    "EngineType",
    "HybridEngine",
    "ProcessPoolEngine",
    "ThreadPoolEngine",
    "create_engine",
    # Errors
    "format_budget_error",
    "format_missing_dependency_error",
    # Estimator
    "CostEstimate",
    "estimate_check_cost",
    "estimate_from_test_file",
    # Experiments
    "ExperimentComparison",
    "ExperimentRun",
    "ExperimentTracker",
    # Guardrails
    "CheckSpec",
    "Guard",
    "GuardrailError",
    "GuardrailMiddleware",
    "ValidationResult",
    "guardrail",
    # Judge backends
    "AnthropicJudge",
    "AzureOpenAIJudge",
    "CustomHTTPJudge",
    "GeminiJudge",
    "JudgeBackend",
    "JudgeConfigError",
    "LiteLLMJudge",
    "OllamaJudge",
    "OpenAIJudge",
    "create_judge",
    # Programmatic API
    "Evaluator",
    "check_output",
    "evaluate",
    "parse_check_shorthand",
    # Pytest integration
    "dataset",
    "metric",
    # Red teaming
    "AttackResult",
    "AttackStrategy",
    "RedTeamer",
    "VulnerabilityReport",
    "VulnerabilityType",
    # Resilience
    "CircuitBreaker",
    "CircuitOpenError",
    "PerProviderRateLimiter",
    "ResilientJudge",
    "RetryPolicy",
    "TokenBucketRateLimiter",
    "with_retry",
    # Streaming
    "StreamingCheckpoint",
    "StreamingEvaluator",
    # Synthesizer
    "EvolutionStrategy",
    "SynthesisConfig",
    "Synthesizer",
    # Testing helpers
    "assert_all_passed",
    "assert_score_above",
    "make_collector",
    # Tracing
    "Span",
    "Tracer",
    "get_tracer",
    "trace",
    # YAML config
    "EvalConfig",
    "YamlEvalRunner",
    "load_eval_config",
    # Version
    "__version__",
]

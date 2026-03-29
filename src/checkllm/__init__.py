"""checkllm - Test LLM-powered applications with the same rigor as traditional software."""

from checkllm.agents import (
    AgentStep,
    AgentTestCase,
    ToolCall,
    validate_no_repeated_tools,
    validate_tool_calls,
    validate_tool_order,
    validate_trajectory_length,
)
from checkllm.api import Evaluator, check_output, evaluate, parse_check_shorthand
from checkllm.batch import BatchEvaluator, BatchJob, BatchStatus
from checkllm.cache import JudgeCache
from checkllm.consensus import AggregationStrategy, ConsensusJudge, ConsensusResult, consensus
from checkllm.conversation import ConversationalTestCase, Turn
from checkllm.datasets.case import Case
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
from checkllm.experiments import ExperimentComparison, ExperimentRun, ExperimentTracker
from checkllm.guardrails import (
    CheckSpec,
    Guard,
    GuardrailError,
    GuardrailMiddleware,
    ValidationResult,
    guardrail,
)
from checkllm.history import RunHistory
from checkllm.judge import AnthropicJudge, JudgeBackend, JudgeConfigError, OpenAIJudge
from checkllm.metrics import metric
from checkllm.models import CheckFailedError, CheckResult, JudgeResponse
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
from checkllm.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    PerProviderRateLimiter,
    ResilientJudge,
    RetryPolicy,
    TokenBucketRateLimiter,
    with_retry,
)
from checkllm.streaming import StreamingCheckpoint, StreamingEvaluator
from checkllm.synthesizer import EvolutionStrategy, SynthesisConfig, Synthesizer
from checkllm.testing import MockJudge, assert_all_passed, assert_score_above, make_collector
from checkllm.tracing import Span, Tracer, get_tracer, trace
from checkllm.yaml_config import EvalConfig, YamlEvalRunner, load_eval_config

__version__ = "2.0.0"

__all__ = [
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
    # Engines
    "AsyncEngine",
    "EngineType",
    "HybridEngine",
    "ProcessPoolEngine",
    "ThreadPoolEngine",
    "create_engine",
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

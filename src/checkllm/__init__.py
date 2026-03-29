"""checkllm - Test LLM-powered applications with the same rigor as traditional software."""

from checkllm.api import Evaluator, check_output, evaluate, parse_check_shorthand
from checkllm.cache import JudgeCache
from checkllm.consensus import AggregationStrategy, ConsensusJudge, ConsensusResult, consensus
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
from checkllm.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    PerProviderRateLimiter,
    ResilientJudge,
    RetryPolicy,
    TokenBucketRateLimiter,
    with_retry,
)
from checkllm.testing import MockJudge, assert_all_passed, assert_score_above, make_collector

__version__ = "1.0.0"

__all__ = [
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
    # Consensus
    "AggregationStrategy",
    "ConsensusJudge",
    "ConsensusResult",
    "consensus",
    # Engines
    "AsyncEngine",
    "EngineType",
    "HybridEngine",
    "ProcessPoolEngine",
    "ThreadPoolEngine",
    "create_engine",
    # Embeddings
    "CachedEmbeddings",
    "OpenAIEmbeddings",
    "batch_semantic_similarity",
    "cosine_similarity",
    "semantic_similarity",
    # Guardrails
    "CheckSpec",
    "Guard",
    "GuardrailError",
    "GuardrailMiddleware",
    "ValidationResult",
    "guardrail",
    # Resilience
    "CircuitBreaker",
    "CircuitOpenError",
    "PerProviderRateLimiter",
    "ResilientJudge",
    "RetryPolicy",
    "TokenBucketRateLimiter",
    "with_retry",
    # Core
    "Case",
    "CheckFailedError",
    "CheckResult",
    "JudgeCache",
    "JudgeResponse",
    "MockJudge",
    "RunHistory",
    # Programmatic API
    "Evaluator",
    "check_output",
    "evaluate",
    "parse_check_shorthand",
    # Pytest integration
    "dataset",
    "metric",
    # Testing helpers
    "assert_all_passed",
    "assert_score_above",
    "make_collector",
    # Version
    "__version__",
]

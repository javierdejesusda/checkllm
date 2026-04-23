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
from checkllm.check_registry import (
    CHECK_REGISTRY,
    AllOf,
    AnyOf,
    CheckRegistry,
    Not,
    RegisteredCheck,
    check,
    run_check,
)

# Populate CHECK_REGISTRY with the built-in deterministic checks.
from checkllm import _check_builtins as _check_builtins  # noqa: F401
from checkllm.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    PerProviderRateLimiter,
    ResilientJudge,
    RetryPolicy,
    TokenBucketRateLimiter,
    with_retry,
)
from checkllm.observe import (
    observe,
    get_trace as get_observe_trace,
    clear_trace,
    start_trace,
    end_trace,
)
from checkllm.observe import Span as ObserveSpan, Trace as ObserveTrace
from checkllm.tracing import Span, Tracer, get_tracer, trace
from checkllm.trajectory import (
    TraceSpan,
    TraceValidator,
    TrajectoryValidator,
    validate_trace,
    validate_trajectory,
)

# Imports that may require optional dependencies (openai, etc.)
try:
    from checkllm.compare import ComparisonResult, MatrixResult, ProviderMatrix
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
    from checkllm.estimator import (
        CostEstimate,
        estimate_check_cost,
        estimate_from_test_file,
    )
    from checkllm.experiments import (
        ExperimentComparison,
        ExperimentRun,
        ExperimentTracker,
    )
    from checkllm.history import RunHistory
    from checkllm.dedup import InFlightDeduplicator, make_dedup_key
    from checkllm.judge import (
        AnthropicJudge,
        DeepSeekJudge,
        OpenAIJudge,
        StreamingJudgeResult,
    )
    from checkllm.providers import (
        AzureOpenAIJudge,
        BedrockJudge,
        CohereJudge,
        CustomHTTPJudge,
        FireworksJudge,
        GeminiJudge,
        GroqJudge,
        LiteLLMJudge,
        MistralJudge,
        OllamaJudge,
        OpenAICompatibleJudge,
        OpenRouterJudge,
        PerplexityJudge,
        TogetherJudge,
        VLLMJudge,
        XAIJudge,
        create_judge,
    )
    from checkllm.pytest_plugin import dataset
    from checkllm.benchmarks import (
        BenchmarkDataset,
        BenchmarkResult,
        BenchmarkRunner,
        BenchmarkSample,
        BenchmarkSuite,
        list_benchmarks,
        load_benchmark,
    )
    from checkllm.redteam import (
        COMPLIANCE_MAPPINGS,
        AttackResult,
        AttackStrategy,
        CompliancePreset,
        OWASPCategory,
        RedTeamer,
        RiskScore,
        RiskScorer,
        SeverityLevel,
        VulnerabilityReport,
        VulnerabilityType,
        get_compliance_vulnerabilities,
        get_owasp_mapping,
        get_vulnerabilities_by_owasp,
    )
    from checkllm.redteam_strategies import (
        BaseStrategy,
        StrategyResult,
        StrategyType,
        apply_strategies,
        get_strategy,
    )
    from checkllm.redteam_datasets import (
        AttackPrompt,
        available_presets,
        load_jailbreak_preset,
    )
    from checkllm.redteam_evolver import (
        AdversarialAttackEvolver,
        EvolvedAttack,
        EvolverConfig,
        MutationStrategy,
        SeedCategoryError,
    )
    from checkllm.redteam_coding_agents import (
        CodingAgentPlugin,
        CodingAgentPluginType,
        CodingAgentReport,
        CodingAgentScanner,
        CodingAgentTestResult,
        get_plugin,
        list_plugins,
    )
    from checkllm.compliance import (
        ComplianceFramework,
        ComplianceReport,
        ComplianceRequirement,
        generate_compliance_report,
    )
    from checkllm.frameworks import (
        ComplianceFramework as ComplianceFrameworkV2,
        FrameworkDefinition,
        FrameworkRequirement,
        get_framework_definition,
        get_framework_summary,
        list_frameworks,
    )
    from checkllm.compliance_scanner import (
        ComplianceReport as ComplianceReportV2,
        ComplianceScanner,
        RequirementResult,
        scan_multiple_frameworks,
    )
    from checkllm.arena import Arena, ArenaCandidate, ArenaResult
    from checkllm.metrics.dag import DAGEvalResult, DAGMetric, DAGNode
    from checkllm.streaming import StreamingCheckpoint, StreamingEvaluator
    from checkllm.synthesizer import (
        ConversationSimulator,
        EvolutionStrategy,
        KnowledgeGraph,
        KnowledgeNode,
        SimulatedConversation,
        SimulatedTurn,
        SynthesisConfig,
        Synthesizer,
    )
    from checkllm.knowledge_graph import (
        BaseTransform,
        EntityExtractor,
        HeadlineSplitter,
        KGEdge,
        KGNode,
        KGTestGenerator,
        KeyphraseExtractor,
        MultiHopAbstractSynthesizer,
        MultiHopSpecificSynthesizer,
        OverlapBuilder,
        Persona,
        QueryLength,
        QueryStyle,
        SentenceSplitter,
        SimilarityBuilder,
        SingleHopSynthesizer,
        SummaryExtractor,
        SynthesizedSample,
        ThemeExtractor,
    )
    from checkllm.knowledge_graph import (
        KnowledgeGraph as KnowledgeGraphV2,
    )
    from checkllm.optimize import (
        COPROOptimizer,
        MIPROv2Optimizer,
        OptimizationResult,
        PromptOptimizer,
        PromptVariant,
        SIMBAOptimizer,
        create_optimizer,
    )
    from checkllm.multilingual import (
        PromptAdapter,
        PromptTemplate,
        SupportedLanguage,
        TranslatedPrompt,
        detect_language,
    )
    from checkllm.testing import (
        MockJudge,
        assert_all_passed,
        assert_score_above,
        make_collector,
    )
    from checkllm.yaml_config import EvalConfig, YamlEvalRunner, load_eval_config

    # New: Metric alignment
    from checkllm.alignment import AlignmentResult, HumanLabel, MetricAligner

    # New: Dual-judge metrics
    from checkllm.dual_judge import (
        AggregationMethod,
        DualJudge,
        DualJudgeMetric,
        DualJudgeResult,
    )

    # New: Advanced red team strategies
    from checkllm.strategies import (
        ConversationTurn,
        CrescendoStrategy,
        GOATStrategy,
        HydraStrategy,
        MultiTurnAttackResult,
        MultiTurnMischief,
        MultiTurnStrategy,
    )

    # New: Poisoned RAG document generation
    from checkllm.rag_poison import (
        PoisonedDocGenerator,
        PoisonedDocument,
        PoisonType,
    )

    # New: Industry compliance
    from checkllm.industry_compliance import (
        Industry,
        IndustryComplianceReport,
        IndustryComplianceRunner,
        IndustryPlugin,
    )

    # New: Adaptive guardrails
    from checkllm.adaptive_guardrails import (
        AdaptiveGuardrail,
        AdaptiveValidationResult,
        GuardrailRule,
    )

    # New: DPO export
    from checkllm.dpo import (
        DPODataset,
        DPOExporter,
        DPOPair,
        DPOStats,
        ExportFormat,
    )

    # New: Model security audit
    from checkllm.model_audit import (
        AuditResult,
        ModelAuditor,
        SecurityFinding,
        SeverityLevel,  # noqa: F811 — intentional re-export shadow of redteam.SeverityLevel
    )

    # New: CI/CD integration
    from checkllm.cicd.github_action import GitHubActionGenerator
    from checkllm.cicd.gitlab_ci import GitLabCIGenerator

    # Phase 6: Comprehensive compliance frameworks
    from checkllm.compliance_frameworks import (
        ComplianceFramework as ComplianceFrameworkV3,
        ComplianceReport as ComplianceReportV3,
        ComplianceScanner as ComplianceScannerV3,
        FrameworkMapping,
        FrameworkRequirement as FrameworkRequirementV3,
        MultiFrameworkReport,
        RequirementResult as RequirementResultV3,
        get_framework_mapping,
        list_all_frameworks,
    )

    # Phase 6: YAML-based evaluation
    from checkllm.yaml_eval import (
        AssertionConfig as YAMLAssertionConfig,
        JudgeConfig as YAMLJudgeConfig,
        EvalTestConfig as YAMLTestConfig,
        YAMLEvalConfig,
        YAMLEvalResult,
        YAMLEvaluator,
        load_yaml_eval_config,
    )

    # RAG dataset generator (Ragas-parity)
    from checkllm.rag_dataset import (
        DocumentChunk,
        QueryDistribution,
        RAGDatasetGenerator,
        chunk_document,
    )

    # Promptfoo-style model-graded assertions
    from checkllm.yaml_assertions import (
        Assertion,
        AssertionResults,
        evaluate_assertions,
        parse_assertions,
    )

    # JSON Schema validation for checkllm.yaml / [tool.checkllm]
    from checkllm.config_schema import (
        ValidationError as ConfigValidationError,
        generate_schema_to_file,
        load_schema,
        validate_config,
    )
except ImportError:
    pass  # Optional dependencies not installed

from checkllm.dashboard import (
    AlertConfig,
    AlertEvent,
    ComparisonView,
    build_comparison_view,
    check_alerts,
)

__version__ = "5.0.1"

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
    # Compare / provider matrix
    "ComparisonResult",
    "MatrixResult",
    "ProviderMatrix",
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
    "BedrockJudge",
    "CohereJudge",
    "CustomHTTPJudge",
    "DeepSeekJudge",
    "FireworksJudge",
    "GeminiJudge",
    "GroqJudge",
    "JudgeBackend",
    "JudgeConfigError",
    "LiteLLMJudge",
    "MistralJudge",
    "OllamaJudge",
    "OpenAICompatibleJudge",
    "OpenAIJudge",
    "OpenRouterJudge",
    "PerplexityJudge",
    "TogetherJudge",
    "VLLMJudge",
    "XAIJudge",
    "StreamingJudgeResult",
    "create_judge",
    # Dedup
    "InFlightDeduplicator",
    "make_dedup_key",
    # Programmatic API
    "Evaluator",
    "check_output",
    "evaluate",
    "parse_check_shorthand",
    # Pytest integration
    "dataset",
    "metric",
    # Benchmarks
    "BenchmarkDataset",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSample",
    "BenchmarkSuite",
    "list_benchmarks",
    "load_benchmark",
    # Red teaming
    "AttackResult",
    "AttackStrategy",
    "COMPLIANCE_MAPPINGS",
    "CompliancePreset",
    "OWASPCategory",
    "RedTeamer",
    "VulnerabilityReport",
    "VulnerabilityType",
    "get_compliance_vulnerabilities",
    "get_owasp_mapping",
    "get_vulnerabilities_by_owasp",
    # Red-team datasets and evolver
    "AttackPrompt",
    "available_presets",
    "load_jailbreak_preset",
    "AdversarialAttackEvolver",
    "EvolvedAttack",
    "EvolverConfig",
    "MutationStrategy",
    "SeedCategoryError",
    # Compliance reporting
    "ComplianceFramework",
    "ComplianceReport",
    "ComplianceRequirement",
    "generate_compliance_report",
    # Compliance frameworks (V2)
    "ComplianceFrameworkV2",
    "ComplianceReportV2",
    "ComplianceScanner",
    "FrameworkDefinition",
    "FrameworkRequirement",
    "RequirementResult",
    "get_framework_definition",
    "get_framework_summary",
    "list_frameworks",
    "scan_multiple_frameworks",
    # Arena A/B testing
    "Arena",
    "ArenaCandidate",
    "ArenaResult",
    # DAG metrics
    "DAGEvalResult",
    "DAGMetric",
    "DAGNode",
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
    # Conversation simulator
    "ConversationSimulator",
    "SimulatedConversation",
    "SimulatedTurn",
    # Knowledge graph
    "KnowledgeGraph",
    "KnowledgeNode",
    # Prompt optimization
    "COPROOptimizer",
    "MIPROv2Optimizer",
    "OptimizationResult",
    "PromptOptimizer",
    "PromptVariant",
    "SIMBAOptimizer",
    "create_optimizer",
    # Multilingual
    "PromptAdapter",
    "PromptTemplate",
    "SupportedLanguage",
    "TranslatedPrompt",
    "detect_language",
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
    # Trajectory / trace evaluation
    "TraceSpan",
    "TraceValidator",
    "TrajectoryValidator",
    "validate_trace",
    "validate_trajectory",
    # YAML config
    "EvalConfig",
    "YamlEvalRunner",
    "load_eval_config",
    # Dashboard comparison and alerting
    "AlertConfig",
    "AlertEvent",
    "ComparisonView",
    "build_comparison_view",
    "check_alerts",
    # Metric alignment
    "AlignmentResult",
    "HumanLabel",
    "MetricAligner",
    # Dual-judge
    "AggregationMethod",
    "DualJudge",
    "DualJudgeMetric",
    "DualJudgeResult",
    # @observe decorator
    "ObserveSpan",
    "ObserveTrace",
    "observe",
    "get_observe_trace",
    "clear_trace",
    "start_trace",
    "end_trace",
    # Advanced red team strategies
    "ConversationTurn",
    "CrescendoStrategy",
    "GOATStrategy",
    "HydraStrategy",
    "MultiTurnAttackResult",
    "MultiTurnMischief",
    "MultiTurnStrategy",
    # Poisoned RAG documents
    "PoisonedDocGenerator",
    "PoisonedDocument",
    "PoisonType",
    # Industry compliance
    "Industry",
    "IndustryComplianceReport",
    "IndustryComplianceRunner",
    "IndustryPlugin",
    # Adaptive guardrails
    "AdaptiveGuardrail",
    "AdaptiveValidationResult",
    "GuardrailRule",
    # DPO export
    "DPODataset",
    "DPOExporter",
    "DPOPair",
    "DPOStats",
    "ExportFormat",
    # Model security audit
    "AuditResult",
    "ModelAuditor",
    "SecurityFinding",
    "SeverityLevel",
    # CI/CD integration
    "GitHubActionGenerator",
    "GitLabCIGenerator",
    # Knowledge Graph V2 pipeline
    "BaseTransform",
    "EntityExtractor",
    "HeadlineSplitter",
    "KGEdge",
    "KGNode",
    "KGTestGenerator",
    "KnowledgeGraphV2",
    "KeyphraseExtractor",
    "MultiHopAbstractSynthesizer",
    "MultiHopSpecificSynthesizer",
    "OverlapBuilder",
    "Persona",
    "QueryLength",
    "QueryStyle",
    "SentenceSplitter",
    "SimilarityBuilder",
    "SingleHopSynthesizer",
    "SummaryExtractor",
    "SynthesizedSample",
    "ThemeExtractor",
    # Comprehensive compliance frameworks
    "ComplianceFrameworkV3",
    "ComplianceReportV3",
    "ComplianceScannerV3",
    "FrameworkMapping",
    "FrameworkRequirementV3",
    "MultiFrameworkReport",
    "RequirementResultV3",
    "get_framework_mapping",
    "list_all_frameworks",
    # YAML evaluation
    "YAMLAssertionConfig",
    "YAMLEvalConfig",
    "YAMLEvalResult",
    "YAMLEvaluator",
    "YAMLJudgeConfig",
    "YAMLTestConfig",
    "load_yaml_eval_config",
    # RAG dataset generator
    "DocumentChunk",
    "QueryDistribution",
    "RAGDatasetGenerator",
    "chunk_document",
    # Model-graded assertions
    "Assertion",
    "AssertionResults",
    "evaluate_assertions",
    "parse_assertions",
    # Config schema validation
    "ConfigValidationError",
    "generate_schema_to_file",
    "load_schema",
    "validate_config",
    # Version
    "__version__",
]

"""checkllm - Test LLM-powered applications with the same rigor as traditional software."""

from checkllm.cache import JudgeCache
from checkllm.datasets.case import Case
from checkllm.history import RunHistory
from checkllm.judge import AnthropicJudge, JudgeBackend, JudgeConfigError, OpenAIJudge
from checkllm.metrics import metric
from checkllm.models import CheckFailedError, CheckResult, JudgeResponse
from checkllm.pytest_plugin import dataset

__version__ = "0.2.0"

__all__ = [
    "AnthropicJudge",
    "Case",
    "CheckFailedError",
    "CheckResult",
    "JudgeBackend",
    "JudgeCache",
    "JudgeConfigError",
    "JudgeResponse",
    "OpenAIJudge",
    "RunHistory",
    "dataset",
    "metric",
    "__version__",
]

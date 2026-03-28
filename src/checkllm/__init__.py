"""checkllm - Test LLM-powered applications with the same rigor as traditional software."""

from checkllm.cache import JudgeCache
from checkllm.datasets.case import Case
from checkllm.history import RunHistory
from checkllm.judge import AnthropicJudge, JudgeBackend, JudgeConfigError, OpenAIJudge
from checkllm.metrics import metric
from checkllm.models import CheckFailedError, CheckResult, JudgeResponse
from checkllm.pytest_plugin import dataset
from checkllm.testing import MockJudge, assert_all_passed, assert_score_above, make_collector

__version__ = "0.3.0"

__all__ = [
    "AnthropicJudge",
    "Case",
    "CheckFailedError",
    "CheckResult",
    "JudgeBackend",
    "JudgeCache",
    "JudgeConfigError",
    "JudgeResponse",
    "MockJudge",
    "OpenAIJudge",
    "RunHistory",
    "assert_all_passed",
    "assert_score_above",
    "dataset",
    "make_collector",
    "metric",
    "__version__",
]

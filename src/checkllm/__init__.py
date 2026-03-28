"""checkllm - Test LLM-powered applications with the same rigor as traditional software."""

from checkllm.datasets.case import Case
from checkllm.metrics import metric
from checkllm.models import CheckFailedError, CheckResult, JudgeResponse
from checkllm.pytest_plugin import dataset

__version__ = "0.1.0"

__all__ = [
    "Case",
    "CheckFailedError",
    "CheckResult",
    "JudgeResponse",
    "dataset",
    "metric",
    "__version__",
]

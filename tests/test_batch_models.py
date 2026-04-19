"""Tests for checkllm.batch — BatchStatus, BatchJob, and BatchEvaluator._build_batch_request."""

from __future__ import annotations

import time

import pytest

from checkllm.batch import BatchJob, BatchStatus
from checkllm.models import CheckResult


class TestBatchStatus:
    def test_all_statuses(self):
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.PROCESSING == "processing"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"
        assert BatchStatus.CANCELLED == "cancelled"

    def test_string_comparison(self):
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.COMPLETED.value == "completed"


class TestBatchJob:
    def test_defaults(self):
        job = BatchJob(job_id="batch-123")
        assert job.job_id == "batch-123"
        assert job.status == BatchStatus.PENDING
        assert job.total_requests == 0
        assert job.completed_requests == 0
        assert job.failed_requests == 0
        assert job.results == []
        assert job.metadata == {}
        assert job.created_at > 0

    def test_with_status(self):
        job = BatchJob(job_id="batch-456", status=BatchStatus.COMPLETED)
        assert job.status == BatchStatus.COMPLETED

    def test_with_requests(self):
        job = BatchJob(
            job_id="batch-789",
            status=BatchStatus.PROCESSING,
            total_requests=10,
            completed_requests=5,
            failed_requests=1,
        )
        assert job.total_requests == 10
        assert job.completed_requests == 5
        assert job.failed_requests == 1

    def test_with_results(self):
        result = CheckResult(
            passed=True,
            score=0.9,
            reasoning="ok",
            cost=0.001,
            latency_ms=50,
            metric_name="relevance",
        )
        job = BatchJob(job_id="batch-101", results=[result])
        assert len(job.results) == 1
        assert job.results[0].metric_name == "relevance"

    def test_with_metadata(self):
        job = BatchJob(
            job_id="batch-102",
            metadata={"file_id": "file-abc", "model": "gpt-4o"},
        )
        assert job.metadata["file_id"] == "file-abc"
        assert job.metadata["model"] == "gpt-4o"

    def test_created_at_is_recent(self):
        before = time.time()
        job = BatchJob(job_id="batch-time")
        after = time.time()
        assert before <= job.created_at <= after


class TestBatchEvaluatorBuildRequest:
    """Test _build_batch_request via instantiation with mocked openai."""

    def _make_evaluator(self):
        """Create BatchEvaluator by mocking the openai import."""
        import sys
        from unittest.mock import MagicMock

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()

        original = sys.modules.get("openai")
        sys.modules["openai"] = mock_openai

        try:
            from checkllm.batch import BatchEvaluator
            evaluator = BatchEvaluator(api_key="test-key", model="gpt-4o")
        finally:
            if original is None:
                del sys.modules["openai"]
            else:
                sys.modules["openai"] = original

        return evaluator

    def test_build_request_basic(self):
        evaluator = self._make_evaluator()
        req = evaluator._build_batch_request(0, "Hello world", None, "gpt-4o")

        assert req["custom_id"] == "req-0"
        assert req["method"] == "POST"
        assert req["url"] == "/v1/chat/completions"
        assert req["body"]["model"] == "gpt-4o"
        assert len(req["body"]["messages"]) == 1
        assert req["body"]["messages"][0]["role"] == "user"
        assert req["body"]["messages"][0]["content"] == "Hello world"

    def test_build_request_with_system_prompt(self):
        evaluator = self._make_evaluator()
        req = evaluator._build_batch_request(5, "User prompt", "System prompt", "gpt-4o")

        messages = req["body"]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User prompt"

    def test_build_request_custom_id_indexed(self):
        evaluator = self._make_evaluator()
        req0 = evaluator._build_batch_request(0, "p", None, "m")
        req5 = evaluator._build_batch_request(5, "p", None, "m")
        req99 = evaluator._build_batch_request(99, "p", None, "m")

        assert req0["custom_id"] == "req-0"
        assert req5["custom_id"] == "req-5"
        assert req99["custom_id"] == "req-99"

    def test_build_request_temperature_zero(self):
        evaluator = self._make_evaluator()
        req = evaluator._build_batch_request(0, "p", None, "m")
        assert req["body"]["temperature"] == 0.0


class TestBatchEvaluatorImportError:
    def test_raises_when_openai_not_installed(self, monkeypatch: pytest.MonkeyPatch):
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                from checkllm.batch import BatchEvaluator
                BatchEvaluator(api_key="test")


class TestOpenAIStatusMap:
    def test_status_mappings(self):
        from checkllm.batch import _OPENAI_STATUS_MAP

        assert _OPENAI_STATUS_MAP["validating"] == BatchStatus.PENDING
        assert _OPENAI_STATUS_MAP["in_progress"] == BatchStatus.PROCESSING
        assert _OPENAI_STATUS_MAP["finalizing"] == BatchStatus.PROCESSING
        assert _OPENAI_STATUS_MAP["completed"] == BatchStatus.COMPLETED
        assert _OPENAI_STATUS_MAP["failed"] == BatchStatus.FAILED
        assert _OPENAI_STATUS_MAP["expired"] == BatchStatus.FAILED
        assert _OPENAI_STATUS_MAP["cancelling"] == BatchStatus.CANCELLED
        assert _OPENAI_STATUS_MAP["cancelled"] == BatchStatus.CANCELLED

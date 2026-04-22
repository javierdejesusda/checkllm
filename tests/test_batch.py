"""Tests for checkllm.batch — batch evaluation via OpenAI batch API."""

from __future__ import annotations

import time

import pytest

from checkllm.batch import BatchEvaluator, BatchJob, BatchStatus


# ---------------------------------------------------------------------------
# BatchStatus enum
# ---------------------------------------------------------------------------


class TestBatchStatus:
    def test_values(self):
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.PROCESSING == "processing"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"
        assert BatchStatus.CANCELLED == "cancelled"


# ---------------------------------------------------------------------------
# BatchJob
# ---------------------------------------------------------------------------


class TestBatchJob:
    def test_creates_job(self):
        job = BatchJob(
            job_id="batch-123",
            status=BatchStatus.COMPLETED,
            total_requests=10,
            completed_requests=8,
            failed_requests=2,
        )
        assert job.job_id == "batch-123"
        assert job.status == BatchStatus.COMPLETED
        assert job.total_requests == 10
        assert job.completed_requests == 8
        assert job.failed_requests == 2

    def test_default_status(self):
        job = BatchJob(job_id="batch-456")
        assert job.status == BatchStatus.PENDING
        assert job.total_requests == 0
        assert job.completed_requests == 0
        assert job.failed_requests == 0
        assert job.results == []
        assert isinstance(job.created_at, float)
        assert job.created_at <= time.time()
        assert job.metadata == {}

    def test_metadata(self):
        job = BatchJob(
            job_id="batch-789",
            metadata={"file_id": "file-abc", "model": "gpt-4o"},
        )
        assert job.metadata["file_id"] == "file-abc"
        assert job.metadata["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# BatchEvaluator — unit tests for internal helpers (no real API calls)
# ---------------------------------------------------------------------------


class TestBatchEvaluator:
    def test_build_batch_request(self):
        """Test _build_batch_request without needing a real API key.

        We cannot instantiate BatchEvaluator without the openai package and
        an API key, so we test the static-like method directly by calling it
        as an unbound method with a minimal mock.
        """
        # Call the method as a plain function (it doesn't use self._client)
        request = BatchEvaluator._build_batch_request(
            None,  # self — not used by the method
            idx=0,
            prompt="What is 2+2?",
            system_prompt="You are a math tutor.",
            model="gpt-4o",
        )
        assert request["custom_id"] == "req-0"
        assert request["method"] == "POST"
        assert request["url"] == "/v1/chat/completions"
        body = request["body"]
        assert body["model"] == "gpt-4o"
        assert body["temperature"] == 0.0
        messages = body["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a math tutor."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"

    def test_build_batch_request_no_system_prompt(self):
        request = BatchEvaluator._build_batch_request(
            None, idx=5, prompt="Hello", system_prompt=None, model="gpt-4o-mini"
        )
        assert request["custom_id"] == "req-5"
        messages = request["body"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_estimate_batch_cost(self):
        cost = BatchEvaluator._estimate_batch_cost(prompt_tokens=1000, completion_tokens=500)
        # Standard: input = 1000 * 2.50/1M, output = 500 * 10.00/1M
        # Batch (50%): input = 0.00125, output = 0.0025
        expected = (1000 * 2.50 / 1_000_000 * 0.5) + (500 * 10.00 / 1_000_000 * 0.5)
        assert abs(cost - expected) < 1e-10

    def test_estimate_batch_cost_zero_tokens(self):
        cost = BatchEvaluator._estimate_batch_cost(0, 0)
        assert cost == 0.0

    def test_parse_judge_output_valid_json(self):
        raw = '{"score": 0.85, "reasoning": "Good answer"}'
        score, reasoning = BatchEvaluator._parse_judge_output(raw)
        assert score == 0.85
        assert reasoning == "Good answer"

    def test_parse_judge_output_with_code_fence(self):
        raw = '```json\n{"score": 0.9, "reasoning": "Great"}\n```'
        score, reasoning = BatchEvaluator._parse_judge_output(raw)
        assert score == 0.9
        assert reasoning == "Great"

    def test_parse_judge_output_fallback(self):
        raw = "This is just plain text, not JSON."
        score, reasoning = BatchEvaluator._parse_judge_output(raw)
        assert score == 0.0
        assert reasoning == raw

    def test_parse_judge_output_empty(self):
        score, reasoning = BatchEvaluator._parse_judge_output("")
        assert score == 0.0
        assert reasoning == "Empty response"

    def test_parse_judge_output_clamps_score(self):
        raw = '{"score": 5.0, "reasoning": "way too high"}'
        score, reasoning = BatchEvaluator._parse_judge_output(raw)
        assert score == 1.0  # clamped to max

    def test_requires_openai_package(self, monkeypatch):
        """BatchEvaluator raises ImportError when openai is not installed,
        or ValueError when API key is missing. Either is acceptable."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        try:
            # This will either raise ImportError (no openai) or ValueError (no key)
            BatchEvaluator(api_key=None)
            pytest.fail("Expected an error when no API key is provided")
        except (ImportError, ValueError):
            pass

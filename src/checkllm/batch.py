"""Batch API support for submitting evaluation jobs via OpenAI batch API.

Provides up to 50% cost savings by batching evaluation requests into a single
JSONL file and submitting them through the OpenAI Batch API endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from checkllm.models import CheckResult, JudgeResponse

logger = logging.getLogger("checkllm.batch")


class BatchStatus(str, Enum):
    """Status of a batch evaluation job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Map OpenAI batch statuses to our enum
_OPENAI_STATUS_MAP: dict[str, BatchStatus] = {
    "validating": BatchStatus.PENDING,
    "in_progress": BatchStatus.PROCESSING,
    "finalizing": BatchStatus.PROCESSING,
    "completed": BatchStatus.COMPLETED,
    "failed": BatchStatus.FAILED,
    "expired": BatchStatus.FAILED,
    "cancelling": BatchStatus.CANCELLED,
    "cancelled": BatchStatus.CANCELLED,
}


class BatchJob(BaseModel):
    """Represents a batch evaluation job."""

    job_id: str
    status: BatchStatus = BatchStatus.PENDING
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    results: list[CheckResult] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchEvaluator:
    """Submit evaluation jobs via OpenAI batch API for 50% cost savings.

    The OpenAI Batch API accepts JSONL files containing chat completion
    requests and processes them asynchronously at a reduced rate.

    Parameters
    ----------
    api_key:
        OpenAI API key. Falls back to the ``OPENAI_API_KEY`` environment
        variable if not provided.
    model:
        Default model for batch requests.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o") -> None:
        self._model = model
        self._client: Any = None

        try:
            import openai  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for BatchEvaluator. "
                "Install it with: pip install openai"
            )

        import os

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key must be provided via api_key parameter "
                "or OPENAI_API_KEY environment variable."
            )

        self._client = openai.OpenAI(api_key=resolved_key)

    def _build_batch_request(
        self,
        idx: int,
        prompt: str,
        system_prompt: str | None,
        model: str,
    ) -> dict[str, Any]:
        """Build a single request in OpenAI batch format.

        Each request line follows the format::

            {
                "custom_id": "req-0",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": { ... }
            }
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return {
            "custom_id": f"req-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
            },
        }

    async def submit(
        self,
        requests: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> BatchJob:
        """Create a JSONL file of requests and submit via the batch API.

        Parameters
        ----------
        requests:
            List of dicts, each containing at minimum a ``"prompt"`` key.
            An optional ``"model"`` key overrides the default model.
        system_prompt:
            Optional system prompt applied to all requests.

        Returns
        -------
        BatchJob
            A job object that can be polled for status.
        """
        if not requests:
            raise ValueError("requests list must not be empty")

        # Build JSONL content
        lines: list[str] = []
        for idx, req in enumerate(requests):
            prompt = req.get("prompt", "")
            if not prompt:
                raise ValueError(f"Request at index {idx} is missing 'prompt' key")
            model = req.get("model", self._model)
            batch_req = self._build_batch_request(idx, prompt, system_prompt, model)
            lines.append(json.dumps(batch_req))

        jsonl_content = "\n".join(lines)

        # Write to a temporary file and upload
        loop = asyncio.get_running_loop()

        def _upload_and_create() -> tuple[str, str]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                f.write(jsonl_content)
                tmp_path = f.name

            try:
                with open(tmp_path, "rb") as f:
                    file_obj = self._client.files.create(
                        file=f,
                        purpose="batch",
                    )

                batch = self._client.batches.create(
                    input_file_id=file_obj.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                return batch.id, file_obj.id
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        batch_id, file_id = await loop.run_in_executor(None, _upload_and_create)

        job = BatchJob(
            job_id=batch_id,
            status=BatchStatus.PENDING,
            total_requests=len(requests),
            metadata={"file_id": file_id, "model": self._model},
        )
        logger.info("Batch job submitted: %s (%d requests)", batch_id, len(requests))
        return job

    async def poll(
        self,
        job: BatchJob,
        interval_seconds: float = 30.0,
        timeout_seconds: float = 3600.0,
    ) -> BatchJob:
        """Poll the batch job until it completes, fails, or times out.

        Parameters
        ----------
        job:
            The batch job to poll.
        interval_seconds:
            Seconds between status checks.
        timeout_seconds:
            Maximum time to wait before raising a ``TimeoutError``.

        Returns
        -------
        BatchJob
            Updated job with final status.
        """
        loop = asyncio.get_running_loop()
        start = time.monotonic()

        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Batch job {job.job_id} did not complete within {timeout_seconds}s"
                )

            def _check_status() -> Any:
                return self._client.batches.retrieve(job.job_id)

            batch = await loop.run_in_executor(None, _check_status)

            job.status = _OPENAI_STATUS_MAP.get(batch.status, BatchStatus.PROCESSING)
            if batch.request_counts:
                job.completed_requests = batch.request_counts.completed or 0
                job.failed_requests = batch.request_counts.failed or 0

            logger.debug(
                "Batch %s status: %s (completed=%d, failed=%d)",
                job.job_id,
                batch.status,
                job.completed_requests,
                job.failed_requests,
            )

            if job.status in (
                BatchStatus.COMPLETED,
                BatchStatus.FAILED,
                BatchStatus.CANCELLED,
            ):
                if job.status == BatchStatus.FAILED:
                    errors = getattr(batch, "errors", None)
                    if errors:
                        job.metadata["errors"] = str(errors)
                    logger.error("Batch job %s failed: %s", job.job_id, errors)
                elif job.status == BatchStatus.COMPLETED:
                    job.metadata["output_file_id"] = getattr(batch, "output_file_id", None)
                    logger.info("Batch job %s completed", job.job_id)
                return job

            await asyncio.sleep(interval_seconds)

    async def retrieve(self, job: BatchJob) -> list[JudgeResponse]:
        """Download and parse results from a completed batch job.

        Parameters
        ----------
        job:
            A completed batch job.

        Returns
        -------
        list[JudgeResponse]
            Parsed judge responses, ordered by custom_id.
        """
        if job.status != BatchStatus.COMPLETED:
            raise ValueError(
                f"Cannot retrieve results from job with status {job.status.value}. "
                f"Job must be completed."
            )

        output_file_id = job.metadata.get("output_file_id")
        if not output_file_id:
            raise ValueError(
                "No output_file_id found in job metadata. Ensure the job was polled to completion."
            )

        loop = asyncio.get_running_loop()

        def _download() -> str:
            content = self._client.files.content(output_file_id)
            return content.text

        raw_content = await loop.run_in_executor(None, _download)

        # Parse JSONL output
        responses_by_id: dict[int, JudgeResponse] = {}
        for line in raw_content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping unparseable batch result line")
                continue

            custom_id = result.get("custom_id", "")
            # Extract index from "req-N"
            try:
                idx = int(custom_id.split("-", 1)[1])
            except (IndexError, ValueError):
                logger.warning("Skipping result with invalid custom_id: %s", custom_id)
                continue

            error = result.get("error")
            if error:
                logger.warning("Batch request %s failed: %s", custom_id, error)
                responses_by_id[idx] = JudgeResponse(
                    score=0.0,
                    reasoning=f"Batch request failed: {error}",
                    raw_output=json.dumps(error),
                    cost=0.0,
                )
                continue

            response_body = result.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            usage = response_body.get("usage", {})

            raw_output = ""
            if choices:
                message = choices[0].get("message", {})
                raw_output = message.get("content", "")

            # Estimate cost from usage
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            # Batch API is 50% of standard pricing
            cost = self._estimate_batch_cost(prompt_tokens, completion_tokens)

            # Try to parse judge JSON from the output
            score, reasoning = self._parse_judge_output(raw_output)

            responses_by_id[idx] = JudgeResponse(
                score=score,
                reasoning=reasoning,
                raw_output=raw_output,
                cost=cost,
            )

        # Return in order
        return [
            responses_by_id.get(
                i,
                JudgeResponse(
                    score=0.0,
                    reasoning="No response received for this request",
                    cost=0.0,
                ),
            )
            for i in range(job.total_requests)
        ]

    async def cancel(self, job: BatchJob) -> BatchJob:
        """Cancel a batch job.

        Parameters
        ----------
        job:
            The batch job to cancel.

        Returns
        -------
        BatchJob
            Updated job with cancelled status.
        """
        loop = asyncio.get_running_loop()

        def _cancel() -> Any:
            return self._client.batches.cancel(job.job_id)

        try:
            await loop.run_in_executor(None, _cancel)
            job.status = BatchStatus.CANCELLED
            logger.info("Batch job %s cancelled", job.job_id)
        except Exception as exc:
            logger.error("Failed to cancel batch job %s: %s", job.job_id, exc)
            raise

        return job

    @staticmethod
    def _estimate_batch_cost(prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for batch API usage (50% of standard pricing).

        Uses approximate gpt-4o pricing as a baseline.
        """
        # Standard gpt-4o: $2.50/1M input, $10.00/1M output
        # Batch API: 50% discount
        input_cost = prompt_tokens * (2.50 / 1_000_000) * 0.5
        output_cost = completion_tokens * (10.00 / 1_000_000) * 0.5
        return input_cost + output_cost

    @staticmethod
    def _parse_judge_output(raw_output: str) -> tuple[float, str]:
        """Attempt to parse a judge's JSON output into score and reasoning.

        Falls back to returning the raw text as reasoning with a score of 0.0
        if JSON parsing fails.
        """
        if not raw_output.strip():
            return 0.0, "Empty response"

        # Try to extract JSON from the output
        text = raw_output.strip()
        # Handle markdown code blocks
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            data = json.loads(text)
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reasoning = str(data.get("reasoning", data.get("explanation", "")))
            return score, reasoning
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        return 0.0, raw_output

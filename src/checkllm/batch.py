"""Batch API support for submitting evaluation jobs to LLM providers.

Provides up to 50% cost savings by batching evaluation requests through
either the OpenAI Batch API or the Anthropic Message Batches API.

The :class:`BatchRunner` protocol describes the common interface. Two concrete
runners are exposed:

* :class:`BatchEvaluator` -- OpenAI Batch API (historical name kept for
  backwards compatibility).
* :class:`AnthropicBatchRunner` -- Anthropic Message Batches API.

The helper :func:`get_batch_runner` picks a provider by name (``"openai"`` or
``"anthropic"``) so CLI code can route ``--batch <provider>`` without
hard-coding a specific class.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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

# Anthropic Message Batches API statuses.
# Docs: https://docs.anthropic.com/en/api/creating-message-batches
#
# ``processing_status`` values:
# - ``in_progress``: batch is still running
# - ``canceling``  : cancellation requested, not yet finished
# - ``ended``      : terminal. Per-request ``result.type`` reveals success/error
#                    (``succeeded``/``errored``/``canceled``/``expired``).
# There is no separate top-level ``expired`` status -- expired requests surface
# inside results once the batch has ``ended``.
_ANTHROPIC_STATUS_MAP: dict[str, BatchStatus] = {
    "in_progress": BatchStatus.PROCESSING,
    "canceling": BatchStatus.CANCELLED,
    "ended": BatchStatus.COMPLETED,
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


@runtime_checkable
class BatchRunner(Protocol):
    """Provider-agnostic interface for submitting evaluation batches.

    All implementations return the same :class:`BatchJob` / list of
    :class:`JudgeResponse` shapes regardless of underlying vendor, so callers
    (CLI, library users) can switch provider without changing code.
    """

    provider: str

    async def submit(
        self,
        requests: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> BatchJob:
        """Submit a list of prompt requests and return a :class:`BatchJob`."""

    async def poll(
        self,
        job: BatchJob,
        interval_seconds: float = 30.0,
        timeout_seconds: float = 3600.0,
    ) -> BatchJob:
        """Poll ``job`` until it reaches a terminal state or times out."""

    async def retrieve(self, job: BatchJob) -> list[JudgeResponse]:
        """Download and parse results for a completed ``job``."""

    async def cancel(self, job: BatchJob) -> BatchJob:
        """Cancel ``job`` and return the updated state."""


def _parse_judge_output(raw_output: str) -> tuple[float, str]:
    """Attempt to parse a judge's JSON output into ``(score, reasoning)``.

    Falls back to returning the raw text as reasoning with a score of ``0.0``
    if JSON parsing fails.
    """
    if not raw_output.strip():
        return 0.0, "Empty response"

    text = raw_output.strip()
    if "```json" in text:
        start = text.index("```json") + 7
        try:
            end = text.index("```", start)
            text = text[start:end].strip()
        except ValueError:
            text = text[start:].strip()
    elif "```" in text:
        start = text.index("```") + 3
        try:
            end = text.index("```", start)
            text = text[start:end].strip()
        except ValueError:
            text = text[start:].strip()

    try:
        data = json.loads(text)
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reasoning = str(data.get("reasoning", data.get("explanation", "")))
        return score, reasoning
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return 0.0, raw_output


class BatchEvaluator:
    """Submit evaluation jobs via the OpenAI Batch API for 50% cost savings.

    The OpenAI Batch API accepts JSONL files containing chat completion
    requests and processes them asynchronously at a reduced rate.

    Args:
        api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY``
            environment variable if not provided.
        model: Default model for batch requests.
    """

    provider: str = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o") -> None:
        self._model = model
        self._client: Any = None

        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for BatchEvaluator. "
                "Install it with: pip install openai"
            ) from exc

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

        Args:
            requests: List of dicts, each containing at minimum a ``"prompt"``
                key. An optional ``"model"`` key overrides the default model.
            system_prompt: Optional system prompt applied to all requests.

        Returns:
            A :class:`BatchJob` that can be polled for status.
        """
        if not requests:
            raise ValueError("requests list must not be empty")

        lines: list[str] = []
        for idx, req in enumerate(requests):
            prompt = req.get("prompt", "")
            if not prompt:
                raise ValueError(f"Request at index {idx} is missing 'prompt' key")
            model = req.get("model", self._model)
            batch_req = self._build_batch_request(idx, prompt, system_prompt, model)
            lines.append(json.dumps(batch_req))

        jsonl_content = "\n".join(lines)

        loop = asyncio.get_running_loop()

        def _upload_and_create() -> tuple[str, str]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                f.write(jsonl_content)
                tmp_path = f.name

            try:
                with open(tmp_path, "rb") as fh:
                    file_obj = self._client.files.create(
                        file=fh,
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
            metadata={"file_id": file_id, "model": self._model, "provider": self.provider},
        )
        logger.info("OpenAI batch submitted: %s (%d requests)", batch_id, len(requests))
        return job

    async def poll(
        self,
        job: BatchJob,
        interval_seconds: float = 30.0,
        timeout_seconds: float = 3600.0,
    ) -> BatchJob:
        """Poll the batch job until it completes, fails, or times out.

        Args:
            job: The batch job to poll.
            interval_seconds: Seconds between status checks.
            timeout_seconds: Maximum time to wait before raising
                :class:`TimeoutError`.

        Returns:
            The updated :class:`BatchJob`.
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
                "OpenAI batch %s status: %s (completed=%d, failed=%d)",
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
        """Download and parse results from a completed batch job."""
        if job.status != BatchStatus.COMPLETED:
            raise ValueError(
                f"Cannot retrieve results from job with status {job.status.value}. "
                f"Job must be completed."
            )

        output_file_id = job.metadata.get("output_file_id")
        if not output_file_id:
            raise ValueError(
                "No output_file_id found in job metadata. "
                "Ensure the job was polled to completion."
            )

        loop = asyncio.get_running_loop()

        def _download() -> str:
            content = self._client.files.content(output_file_id)
            return str(content.text)

        raw_content = await loop.run_in_executor(None, _download)

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

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cost = self._estimate_batch_cost(prompt_tokens, completion_tokens)

            score, reasoning = _parse_judge_output(raw_output)

            responses_by_id[idx] = JudgeResponse(
                score=score,
                reasoning=reasoning,
                raw_output=raw_output,
                cost=cost,
            )

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
        """Cancel a batch job."""
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
        input_cost = prompt_tokens * (2.50 / 1_000_000) * 0.5
        output_cost = completion_tokens * (10.00 / 1_000_000) * 0.5
        return input_cost + output_cost

    @staticmethod
    def _parse_judge_output(raw_output: str) -> tuple[float, str]:
        """Parse judge output (exposed as a staticmethod for compatibility)."""
        return _parse_judge_output(raw_output)


# Anthropic per-model pricing, USD per token. Batch price is half of sync --
# see https://docs.anthropic.com/en/api/creating-message-batches.
# Kept in sync with ``checkllm.judge._ANTHROPIC_PRICES`` for the common models.
_ANTHROPIC_SYNC_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-sonnet-4-5-20250929": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-opus-4-6": (15.00 / 1_000_000, 75.00 / 1_000_000),
    "claude-haiku-4-5-20251001": (0.80 / 1_000_000, 4.00 / 1_000_000),
    "claude-3-5-sonnet-20241022": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-3-haiku-20240307": (0.25 / 1_000_000, 1.25 / 1_000_000),
}

_ANTHROPIC_DEFAULT_PRICE = (3.00 / 1_000_000, 15.00 / 1_000_000)

# 50% discount applied to both input and output when using the batches API.
ANTHROPIC_BATCH_DISCOUNT = 0.5


class AnthropicBatchRunner:
    """Submit evaluation jobs via the Anthropic Message Batches API.

    Provides a 50% discount vs sync ``messages.create`` for the same model.
    Mirrors :class:`BatchEvaluator` so callers can swap implementations.

    Args:
        api_key: Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY``
            environment variable if not provided.
        model: Default Claude model for batch requests.
        max_tokens: Default ``max_tokens`` for each request (Anthropic
            requires an explicit ``max_tokens`` per request).
    """

    provider: str = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = None

        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicBatchRunner. "
                "Install it with: pip install anthropic"
            ) from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key must be provided via api_key parameter "
                "or ANTHROPIC_API_KEY environment variable."
            )

        self._client = anthropic.Anthropic(api_key=resolved_key)

    def _build_batch_request(
        self,
        idx: int,
        prompt: str,
        system_prompt: str | None,
        model: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build a single request in Anthropic batch format.

        Each request has the shape::

            {
                "custom_id": "req-0",
                "params": {
                    "model": "...",
                    "max_tokens": 1024,
                    "system": "optional",
                    "messages": [{"role": "user", "content": "..."}]
                }
            }
        """
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            params["system"] = system_prompt
        return {"custom_id": f"req-{idx}", "params": params}

    async def submit(
        self,
        requests: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> BatchJob:
        """Submit ``requests`` via the Anthropic Message Batches API.

        Args:
            requests: List of dicts, each containing at minimum a ``"prompt"``
                key. Optional ``"model"`` and ``"max_tokens"`` keys override
                the runner defaults.
            system_prompt: Optional system prompt applied to every request.

        Returns:
            A :class:`BatchJob` whose ``job_id`` is the Anthropic batch ID.
        """
        if not requests:
            raise ValueError("requests list must not be empty")

        payload: list[dict[str, Any]] = []
        for idx, req in enumerate(requests):
            prompt = req.get("prompt", "")
            if not prompt:
                raise ValueError(f"Request at index {idx} is missing 'prompt' key")
            model = req.get("model", self._model)
            max_tokens = req.get("max_tokens", self._max_tokens)
            payload.append(self._build_batch_request(idx, prompt, system_prompt, model, max_tokens))

        loop = asyncio.get_running_loop()

        def _create() -> Any:
            return self._client.messages.batches.create(requests=payload)

        batch = await loop.run_in_executor(None, _create)

        job = BatchJob(
            job_id=batch.id,
            status=_ANTHROPIC_STATUS_MAP.get(
                getattr(batch, "processing_status", "in_progress"),
                BatchStatus.PENDING,
            ),
            total_requests=len(requests),
            metadata={"model": self._model, "provider": self.provider},
        )
        logger.info("Anthropic batch submitted: %s (%d requests)", batch.id, len(requests))
        return job

    async def poll(
        self,
        job: BatchJob,
        interval_seconds: float = 30.0,
        timeout_seconds: float = 3600.0,
    ) -> BatchJob:
        """Poll ``job`` until Anthropic reports ``processing_status == 'ended'``.

        Anthropic has only three processing statuses: ``in_progress``,
        ``canceling``, ``ended``. A batch moves to ``ended`` whether it
        succeeded, failed, or expired -- per-request details live in the
        results stream.
        """
        loop = asyncio.get_running_loop()
        start = time.monotonic()

        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Anthropic batch {job.job_id} did not complete within {timeout_seconds}s"
                )

            def _retrieve() -> Any:
                return self._client.messages.batches.retrieve(job.job_id)

            batch = await loop.run_in_executor(None, _retrieve)

            processing_status = getattr(batch, "processing_status", "in_progress")
            job.status = _ANTHROPIC_STATUS_MAP.get(processing_status, BatchStatus.PROCESSING)

            counts = getattr(batch, "request_counts", None)
            if counts is not None:
                succeeded = getattr(counts, "succeeded", 0) or 0
                errored = getattr(counts, "errored", 0) or 0
                canceled = getattr(counts, "canceled", 0) or 0
                expired = getattr(counts, "expired", 0) or 0
                job.completed_requests = succeeded
                job.failed_requests = errored + canceled + expired

            logger.debug(
                "Anthropic batch %s status: %s (succeeded=%d, failed=%d)",
                job.job_id,
                processing_status,
                job.completed_requests,
                job.failed_requests,
            )

            if processing_status == "ended":
                job.status = BatchStatus.COMPLETED
                job.metadata["processing_status"] = processing_status
                logger.info("Anthropic batch %s ended", job.job_id)
                return job
            if processing_status == "canceling":
                # Keep polling until the cancel finishes and we transition to
                # ``ended`` -- surfacing CANCELLED only after per-request
                # results have been observed.
                job.status = BatchStatus.CANCELLED
                # A canceling batch still completes; we keep polling but if the
                # caller prefers to stop early they can break on CANCELLED.
            await asyncio.sleep(interval_seconds)

    async def retrieve(self, job: BatchJob) -> list[JudgeResponse]:
        """Stream and parse results from a finished Anthropic batch.

        Each result's ``result.type`` is one of ``succeeded``, ``errored``,
        ``canceled`` or ``expired``. All non-success variants surface a
        :class:`JudgeResponse` with ``score=0`` and a descriptive
        ``reasoning`` string -- they never raise, so a partial failure never
        takes down the whole batch.
        """
        if job.status not in (BatchStatus.COMPLETED, BatchStatus.FAILED):
            raise ValueError(
                f"Cannot retrieve results from job with status {job.status.value}. "
                f"Job must have ended."
            )

        loop = asyncio.get_running_loop()

        def _fetch_results() -> list[Any]:
            return list(self._client.messages.batches.results(job.job_id))

        raw_results = await loop.run_in_executor(None, _fetch_results)

        responses_by_id: dict[int, JudgeResponse] = {}
        for entry in raw_results:
            custom_id = self._attr(entry, "custom_id", "")
            try:
                idx = int(str(custom_id).split("-", 1)[1])
            except (IndexError, ValueError):
                logger.warning("Skipping Anthropic result with invalid custom_id: %s", custom_id)
                continue

            response = self._parse_individual_result(entry, job.metadata.get("model", self._model))
            responses_by_id[idx] = response

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
        """Request cancellation of an in-progress Anthropic batch."""
        loop = asyncio.get_running_loop()

        def _cancel() -> Any:
            return self._client.messages.batches.cancel(job.job_id)

        try:
            await loop.run_in_executor(None, _cancel)
            job.status = BatchStatus.CANCELLED
            logger.info("Anthropic batch %s cancelled", job.job_id)
        except Exception as exc:
            logger.error("Failed to cancel Anthropic batch %s: %s", job.job_id, exc)
            raise

        return job

    @staticmethod
    def _attr(obj: Any, name: str, default: Any = None) -> Any:
        """Fetch an attribute/key from either an SDK object or a dict."""
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _parse_individual_result(self, entry: Any, model: str) -> JudgeResponse:
        """Translate a ``MessageBatchIndividualResponse`` into a ``JudgeResponse``.

        Handles the four possible ``result.type`` values (``succeeded``,
        ``errored``, ``canceled``, ``expired``) and applies the 50% batch
        discount to the cost for successful responses.
        """
        result = self._attr(entry, "result")
        if result is None:
            return JudgeResponse(
                score=0.0,
                reasoning="Missing result payload from Anthropic batch",
                raw_output=None,
                cost=0.0,
            )

        result_type = self._attr(result, "type", "unknown")

        if result_type == "succeeded":
            message = self._attr(result, "message")
            raw_output = ""
            input_tokens = 0
            output_tokens = 0

            if message is not None:
                content = self._attr(message, "content", []) or []
                for block in content:
                    block_type = self._attr(block, "type")
                    if block_type == "text":
                        raw_output += self._attr(block, "text", "") or ""
                usage = self._attr(message, "usage")
                if usage is not None:
                    input_tokens = int(self._attr(usage, "input_tokens", 0) or 0)
                    output_tokens = int(self._attr(usage, "output_tokens", 0) or 0)

            cost = self._estimate_batch_cost(model, input_tokens, output_tokens)
            score, reasoning = _parse_judge_output(raw_output)
            return JudgeResponse(
                score=score,
                reasoning=reasoning,
                raw_output=raw_output,
                cost=cost,
            )

        if result_type == "errored":
            error = self._attr(result, "error")
            detail = self._attr(error, "message", None) if error is not None else None
            detail = (
                detail or self._attr(error, "type", "unknown error") if error else "unknown error"
            )
            return JudgeResponse(
                score=0.0,
                reasoning=f"Anthropic batch request errored: {detail}",
                raw_output=None,
                cost=0.0,
            )

        if result_type == "canceled":
            return JudgeResponse(
                score=0.0,
                reasoning="Anthropic batch request was canceled",
                raw_output=None,
                cost=0.0,
            )

        if result_type == "expired":
            return JudgeResponse(
                score=0.0,
                reasoning="Anthropic batch request expired before completion",
                raw_output=None,
                cost=0.0,
            )

        return JudgeResponse(
            score=0.0,
            reasoning=f"Unknown Anthropic batch result type: {result_type}",
            raw_output=None,
            cost=0.0,
        )

    @staticmethod
    def _estimate_batch_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for a single Anthropic batch response.

        Batch pricing is 50% of the corresponding sync pricing.
        """
        input_price, output_price = _ANTHROPIC_SYNC_PRICES.get(model, _ANTHROPIC_DEFAULT_PRICE)
        raw = input_tokens * input_price + output_tokens * output_price
        return raw * ANTHROPIC_BATCH_DISCOUNT


def get_batch_runner(
    provider: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> BatchRunner:
    """Return a :class:`BatchRunner` for the named provider.

    Args:
        provider: One of ``"openai"`` or ``"anthropic"``.
        api_key: Optional API key override for the provider.
        model: Optional default model override. If ``None`` the runner uses
            its own default.
        **kwargs: Forwarded to the underlying runner constructor (e.g.
            ``max_tokens=`` for Anthropic).

    Returns:
        A runner implementing the :class:`BatchRunner` protocol.

    Raises:
        ValueError: If ``provider`` is not a supported batch provider.
    """
    key = provider.lower().strip()
    if key in {"openai", "oai"}:
        if model is None:
            return BatchEvaluator(api_key=api_key)
        return BatchEvaluator(api_key=api_key, model=model)
    if key in {"anthropic", "claude"}:
        if model is None:
            return AnthropicBatchRunner(api_key=api_key, **kwargs)
        return AnthropicBatchRunner(api_key=api_key, model=model, **kwargs)
    raise ValueError(
        f"Unsupported batch provider: {provider!r}. " "Choose one of: 'openai', 'anthropic'."
    )

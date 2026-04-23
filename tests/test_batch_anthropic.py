"""Tests for checkllm.batch Anthropic Message Batches integration."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeBatchesAPI:
    """Minimal in-memory stand-in for client.messages.batches."""

    def __init__(self) -> None:
        self._batches: dict[str, Any] = {}
        self._results: dict[str, list[Any]] = {}
        self._next_id = 0
        self.create_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[str] = []
        self.results_calls: list[str] = []
        self.cancel_calls: list[str] = []

    def _new_id(self) -> str:
        self._next_id += 1
        return f"msgbatch_{self._next_id:03d}"

    def create(self, *, requests: list[dict[str, Any]]) -> Any:
        batch_id = self._new_id()
        self.create_calls.append({"id": batch_id, "requests": requests})
        batch = SimpleNamespace(
            id=batch_id,
            processing_status="in_progress",
            request_counts=SimpleNamespace(
                processing=len(requests),
                succeeded=0,
                errored=0,
                canceled=0,
                expired=0,
            ),
        )
        self._batches[batch_id] = batch
        return batch

    def retrieve(self, batch_id: str) -> Any:
        self.retrieve_calls.append(batch_id)
        return self._batches[batch_id]

    def results(self, batch_id: str) -> list[Any]:
        self.results_calls.append(batch_id)
        return list(self._results.get(batch_id, []))

    def cancel(self, batch_id: str) -> Any:
        self.cancel_calls.append(batch_id)
        batch = self._batches[batch_id]
        batch.processing_status = "canceling"
        return batch

    def set_status(
        self,
        batch_id: str,
        *,
        processing_status: str,
        succeeded: int = 0,
        errored: int = 0,
        canceled: int = 0,
        expired: int = 0,
    ) -> None:
        batch = self._batches[batch_id]
        batch.processing_status = processing_status
        batch.request_counts = SimpleNamespace(
            processing=0,
            succeeded=succeeded,
            errored=errored,
            canceled=canceled,
            expired=expired,
        )

    def set_results(self, batch_id: str, results: list[Any]) -> None:
        self._results[batch_id] = results


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> _FakeBatchesAPI:
    """Install a mocked ``anthropic`` module on ``sys.modules``.

    Returns the :class:`_FakeBatchesAPI` instance that backs the mock so tests
    can drive it directly.
    """
    fake_batches = _FakeBatchesAPI()

    fake_messages = MagicMock()
    fake_messages.batches = fake_batches

    fake_client = MagicMock()
    fake_client.messages = fake_messages

    fake_anthropic_module = MagicMock()
    fake_anthropic_module.Anthropic = MagicMock(return_value=fake_client)

    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return fake_batches


def _succeeded_result(custom_id: str, *, text: str, input_tokens: int, output_tokens: int) -> Any:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            ),
        ),
    )


def _errored_result(custom_id: str, *, message: str = "rate_limit") -> Any:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="errored",
            error=SimpleNamespace(type="rate_limit_error", message=message),
        ),
    )


def _expired_result(custom_id: str) -> Any:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="expired"),
    )


# ---------------------------------------------------------------------------
# BatchRunner protocol / factory
# ---------------------------------------------------------------------------


class TestBatchRunnerFactory:
    def test_unknown_provider_raises(self):
        from checkllm.batch import get_batch_runner

        with pytest.raises(ValueError, match="Unsupported batch provider"):
            get_batch_runner("does-not-exist")

    def test_returns_anthropic_runner(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner, get_batch_runner

        runner = get_batch_runner("anthropic")
        assert isinstance(runner, AnthropicBatchRunner)
        assert runner.provider == "anthropic"

    def test_protocol_satisfaction(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner, BatchRunner

        runner = AnthropicBatchRunner(api_key="k")
        assert isinstance(runner, BatchRunner)


# ---------------------------------------------------------------------------
# AnthropicBatchRunner construction + request shape
# ---------------------------------------------------------------------------


class TestAnthropicBatchRunnerConstruction:
    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_anthropic(monkeypatch)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from checkllm.batch import AnthropicBatchRunner

        with pytest.raises(ValueError, match="Anthropic API key"):
            AnthropicBatchRunner(api_key=None)

    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)

        from checkllm.batch import AnthropicBatchRunner

        with pytest.raises(ImportError, match="anthropic"):
            AnthropicBatchRunner(api_key="k")

    def test_build_request_shape(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner

        runner = AnthropicBatchRunner(api_key="k", model="claude-sonnet-4-5-20250929")
        req = runner._build_batch_request(
            idx=3,
            prompt="Score this",
            system_prompt="You are a judge.",
            model="claude-sonnet-4-5-20250929",
            max_tokens=512,
        )
        assert req["custom_id"] == "req-3"
        assert req["params"]["model"] == "claude-sonnet-4-5-20250929"
        assert req["params"]["max_tokens"] == 512
        assert req["params"]["system"] == "You are a judge."
        messages = req["params"]["messages"]
        assert messages == [{"role": "user", "content": "Score this"}]

    def test_build_request_omits_system_when_none(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner

        runner = AnthropicBatchRunner(api_key="k")
        req = runner._build_batch_request(
            idx=0,
            prompt="hi",
            system_prompt=None,
            model="claude-sonnet-4-5-20250929",
        )
        assert "system" not in req["params"]


# ---------------------------------------------------------------------------
# Submit round-trip
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_submit_happy_path(self, monkeypatch: pytest.MonkeyPatch):
        fake = _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner, BatchStatus

        runner = AnthropicBatchRunner(api_key="k")

        import asyncio

        job = asyncio.run(
            runner.submit(
                [{"prompt": "a"}, {"prompt": "b"}],
                system_prompt="be terse",
            )
        )
        assert job.total_requests == 2
        assert job.job_id.startswith("msgbatch_")
        assert job.status in (BatchStatus.PROCESSING, BatchStatus.PENDING)
        assert job.metadata["provider"] == "anthropic"

        call = fake.create_calls[-1]
        assert len(call["requests"]) == 2
        for i, r in enumerate(call["requests"]):
            assert r["custom_id"] == f"req-{i}"
            assert r["params"]["system"] == "be terse"
            assert r["params"]["messages"][0]["role"] == "user"

    def test_submit_empty_raises(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner

        runner = AnthropicBatchRunner(api_key="k")

        import asyncio

        with pytest.raises(ValueError, match="must not be empty"):
            asyncio.run(runner.submit([]))

    def test_submit_missing_prompt_raises(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner

        runner = AnthropicBatchRunner(api_key="k")

        import asyncio

        with pytest.raises(ValueError, match="missing 'prompt'"):
            asyncio.run(runner.submit([{"not_prompt": "x"}]))


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


class TestPolling:
    def test_polls_until_ended(self, monkeypatch: pytest.MonkeyPatch):
        fake = _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner, BatchStatus

        runner = AnthropicBatchRunner(api_key="k")

        import asyncio

        async def _flow():
            job = await runner.submit([{"prompt": "a"}, {"prompt": "b"}])
            # First poll returns in_progress, then flips to ended.
            original_retrieve = fake.retrieve
            call_count = {"n": 0}

            def _retrieve(batch_id: str):
                call_count["n"] += 1
                if call_count["n"] >= 2:
                    fake.set_status(
                        batch_id,
                        processing_status="ended",
                        succeeded=2,
                    )
                return original_retrieve(batch_id)

            fake.retrieve = _retrieve  # type: ignore[method-assign]
            return await runner.poll(job, interval_seconds=0.0, timeout_seconds=5.0)

        job = asyncio.run(_flow())
        assert job.status == BatchStatus.COMPLETED
        assert job.completed_requests == 2
        assert job.failed_requests == 0

    def test_polling_timeout(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner

        runner = AnthropicBatchRunner(api_key="k")

        import asyncio

        async def _flow():
            job = await runner.submit([{"prompt": "a"}])
            return await runner.poll(job, interval_seconds=0.0, timeout_seconds=-1.0)

        with pytest.raises(TimeoutError):
            asyncio.run(_flow())


# ---------------------------------------------------------------------------
# Retrieve: partial failures, expired batches, discount
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_partial_failure_does_not_raise(self, monkeypatch: pytest.MonkeyPatch):
        fake = _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner, BatchStatus

        runner = AnthropicBatchRunner(api_key="k", model="claude-sonnet-4-5-20250929")

        import asyncio

        async def _flow():
            job = await runner.submit([{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}])
            fake.set_status(
                job.job_id,
                processing_status="ended",
                succeeded=2,
                errored=1,
            )
            fake.set_results(
                job.job_id,
                [
                    _succeeded_result(
                        "req-0",
                        text='{"score": 0.9, "reasoning": "great"}',
                        input_tokens=100,
                        output_tokens=20,
                    ),
                    _errored_result("req-1", message="overloaded"),
                    _succeeded_result(
                        "req-2",
                        text='{"score": 0.5, "reasoning": "mid"}',
                        input_tokens=50,
                        output_tokens=10,
                    ),
                ],
            )
            job = await runner.poll(job, interval_seconds=0.0, timeout_seconds=1.0)
            assert job.status == BatchStatus.COMPLETED
            return await runner.retrieve(job)

        responses = asyncio.run(_flow())
        assert len(responses) == 3
        assert responses[0].score == pytest.approx(0.9)
        assert responses[0].reasoning == "great"
        assert responses[1].score == 0.0
        assert "errored" in responses[1].reasoning
        assert "overloaded" in responses[1].reasoning
        assert responses[2].score == pytest.approx(0.5)

    def test_expired_result_becomes_zero_score(self, monkeypatch: pytest.MonkeyPatch):
        fake = _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner

        runner = AnthropicBatchRunner(api_key="k")

        import asyncio

        async def _flow():
            job = await runner.submit([{"prompt": "a"}])
            fake.set_status(
                job.job_id,
                processing_status="ended",
                succeeded=0,
                expired=1,
            )
            fake.set_results(job.job_id, [_expired_result("req-0")])
            job = await runner.poll(job, interval_seconds=0.0, timeout_seconds=1.0)
            return await runner.retrieve(job)

        responses = asyncio.run(_flow())
        assert len(responses) == 1
        assert responses[0].score == 0.0
        assert "expired" in responses[0].reasoning.lower()
        assert responses[0].cost == 0.0

    def test_batch_discount_applied(self, monkeypatch: pytest.MonkeyPatch):
        fake = _install_fake_anthropic(monkeypatch)
        from checkllm.batch import (
            ANTHROPIC_BATCH_DISCOUNT,
            AnthropicBatchRunner,
        )

        model = "claude-sonnet-4-5-20250929"
        runner = AnthropicBatchRunner(api_key="k", model=model)

        import asyncio

        async def _flow():
            job = await runner.submit([{"prompt": "a"}])
            fake.set_status(job.job_id, processing_status="ended", succeeded=1)
            fake.set_results(
                job.job_id,
                [
                    _succeeded_result(
                        "req-0",
                        text='{"score": 1.0, "reasoning": "ok"}',
                        input_tokens=1_000_000,
                        output_tokens=1_000_000,
                    )
                ],
            )
            job = await runner.poll(job, interval_seconds=0.0, timeout_seconds=1.0)
            return await runner.retrieve(job)

        responses = asyncio.run(_flow())
        # Sonnet 4.5: 3 USD/MTok input + 15 USD/MTok output => 18 USD sync,
        # 9 USD with the 50% batch discount.
        assert ANTHROPIC_BATCH_DISCOUNT == 0.5
        assert responses[0].cost == pytest.approx(9.0, rel=1e-6)

    def test_missing_result_index_fills_placeholder(self, monkeypatch: pytest.MonkeyPatch):
        fake = _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner

        runner = AnthropicBatchRunner(api_key="k")

        import asyncio

        async def _flow():
            job = await runner.submit([{"prompt": "a"}, {"prompt": "b"}])
            fake.set_status(job.job_id, processing_status="ended", succeeded=1)
            fake.set_results(
                job.job_id,
                [
                    _succeeded_result(
                        "req-0",
                        text='{"score": 0.7, "reasoning": "ok"}',
                        input_tokens=10,
                        output_tokens=5,
                    )
                ],
            )
            job = await runner.poll(job, interval_seconds=0.0, timeout_seconds=1.0)
            return await runner.retrieve(job)

        responses = asyncio.run(_flow())
        assert len(responses) == 2
        assert responses[0].score == pytest.approx(0.7)
        assert responses[1].score == 0.0
        assert "No response" in responses[1].reasoning


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_transitions_status(self, monkeypatch: pytest.MonkeyPatch):
        fake = _install_fake_anthropic(monkeypatch)
        from checkllm.batch import AnthropicBatchRunner, BatchStatus

        runner = AnthropicBatchRunner(api_key="k")

        import asyncio

        async def _flow():
            job = await runner.submit([{"prompt": "a"}])
            return await runner.cancel(job)

        job = asyncio.run(_flow())
        assert job.status == BatchStatus.CANCELLED
        assert len(fake.cancel_calls) == 1


# ---------------------------------------------------------------------------
# Pure function: _estimate_batch_cost
# ---------------------------------------------------------------------------


class TestEstimateBatchCost:
    def test_discount_is_half_of_sync(self):
        from checkllm.batch import (
            ANTHROPIC_BATCH_DISCOUNT,
            AnthropicBatchRunner,
        )

        # 1M input + 1M output on sonnet-4-5 at 3 + 15 USD/MTok = 18 USD sync.
        cost = AnthropicBatchRunner._estimate_batch_cost(
            "claude-sonnet-4-5-20250929", 1_000_000, 1_000_000
        )
        assert cost == pytest.approx(18.0 * ANTHROPIC_BATCH_DISCOUNT)

    def test_unknown_model_uses_default(self):
        from checkllm.batch import AnthropicBatchRunner

        cost = AnthropicBatchRunner._estimate_batch_cost("unknown-model", 1_000_000, 0)
        assert cost > 0

    def test_zero_tokens(self):
        from checkllm.batch import AnthropicBatchRunner

        assert AnthropicBatchRunner._estimate_batch_cost("anything", 0, 0) == 0.0

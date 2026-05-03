"""Alias module for ``checkllm.integrations.llamaindex``.

LlamaIndex's package is published as ``llama-index`` on PyPI and imports as
``llama_index``. CheckLLM historically used the squashed name
``llamaindex`` for its integration module; this alias re-exports the
trajectory adapters under the snake_case name so users can write either
``from checkllm.integrations.llama_index import to_checkllm_test_case``
or ``from checkllm.integrations.llamaindex import to_checkllm_test_case``
interchangeably.
"""

from __future__ import annotations

from checkllm.integrations.llamaindex import (
    CheckllmCallbackHandler,
    to_checkllm_test_case,
    to_checkllm_tool_calls,
)

__all__ = [
    "CheckllmCallbackHandler",
    "to_checkllm_test_case",
    "to_checkllm_tool_calls",
]

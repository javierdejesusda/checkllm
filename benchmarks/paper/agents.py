"""Agent interface + stub implementation for paper experiments.

Real model agents (Anthropic, OpenAI, Ollama) plug into the same
:class:`Agent` protocol and get swapped in during Phase C. The
:class:`StubAgent` replays a benchmark's ``reference_actions`` with
optional deterministic noise so the pipeline can be exercised end-to-end
for $0 in CI.
"""

from __future__ import annotations

import random
from typing import Any, Protocol


class Agent(Protocol):
    """Minimal protocol an agent must satisfy for the paper runner."""

    name: str

    def run(
        self,
        reference_actions: list[dict[str, Any]] | None = None,
        user_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the predicted list of tool-call dicts."""


class StubAgent:
    """Deterministic stub that replays ``reference_actions``.

    ``noise`` controls three probabilities in ``[0, 1]``:

    * ``drop``: probability of skipping a reference action.
    * ``repeat``: probability of emitting a duplicate of the previous action
      (simulates a loop).
    * ``extra``: probability of appending a spurious unrelated action.

    With all probabilities at 0 the stub produces a perfect trajectory.
    """

    name: str = "stub"

    def __init__(
        self,
        seed: int = 0,
        drop: float = 0.0,
        repeat: float = 0.0,
        extra: float = 0.0,
    ) -> None:
        self._rng = random.Random(seed)
        self.drop = drop
        self.repeat = repeat
        self.extra = extra

    def run(
        self,
        reference_actions: list[dict[str, Any]] | None = None,
        user_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        del user_instruction, tools  # unused in stub
        ref = list(reference_actions or [])
        out: list[dict[str, Any]] = []
        for action in ref:
            if self._rng.random() < self.drop:
                continue
            out.append(dict(action))
            if self._rng.random() < self.repeat:
                out.append(dict(action))
        if self._rng.random() < self.extra:
            out.append({"name": "noop", "arguments": {}})
        return out


def build_agent(name: str, seed: int) -> Agent:
    """Build an agent by name for the runner. Extend as models are added.

    Args:
        name: Agent identifier. Currently supports ``"stub"``.
        seed: Deterministic seed passed to the agent.

    Returns:
        An :class:`Agent` instance.

    Raises:
        ValueError: If ``name`` is not a supported model.
    """
    if name == "stub":
        return StubAgent(seed=seed)
    raise ValueError(
        f"Unknown model {name!r}. This task wires only the 'stub' agent; "
        "real models are wired in Phase C."
    )

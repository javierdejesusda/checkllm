from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """A single turn in a conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationalTestCase(BaseModel):
    """A multi-turn conversation test case."""

    turns: list[Turn]
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str | None = None

    @property
    def user_turns(self) -> list[Turn]:
        """Return all turns with role 'user'."""
        return [t for t in self.turns if t.role == "user"]

    @property
    def assistant_turns(self) -> list[Turn]:
        """Return all turns with role 'assistant'."""
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def system_turns(self) -> list[Turn]:
        """Return all turns with role 'system'."""
        return [t for t in self.turns if t.role == "system"]

    @property
    def last_response(self) -> str | None:
        """Return the content of the last assistant turn, or None."""
        assistant = self.assistant_turns
        return assistant[-1].content if assistant else None

    @property
    def first_user_message(self) -> str | None:
        """Return the content of the first user turn, or None."""
        user = self.user_turns
        return user[0].content if user else None

    @property
    def turn_count(self) -> int:
        """Return the total number of turns in the conversation."""
        return len(self.turns)

    def format_transcript(self) -> str:
        """Format the conversation as a readable transcript."""
        lines = []
        for turn in self.turns:
            lines.append(f"[{turn.role.upper()}]: {turn.content}")
        return "\n".join(lines)

    def turns_by_role(self, role: str) -> list[Turn]:
        """Return all turns matching the given role."""
        return [t for t in self.turns if t.role == role]

    def slice_turns(self, start: int, end: int | None = None) -> list[Turn]:
        """Return a slice of turns from *start* up to (but not including) *end*."""
        return self.turns[start:end]

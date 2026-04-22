"""Evaluating a multi-turn chatbot.

Scenario: a 5-turn customer-support conversation. The tests exercise the
three "must have" multi-turn metrics plus per-turn relevancy:

- ``conversation_completeness`` - every user request was satisfied.
- ``role_adherence`` - the assistant stayed in its declared role.
- ``knowledge_retention`` - the assistant remembers earlier turns.
- ``turn_relevancy`` (per-turn) - granular signal for a specific turn.

``ConversationalTestCase`` and ``Turn`` from ``checkllm.conversation`` are
the canonical input types. Live judges are swapped for a ``MockJudge``
so tests run offline. Real-judge examples live in
``tests/test_advanced_metrics.py``.

Run with: pytest examples/test_multi_turn_conversation.py
"""

from __future__ import annotations

import pytest

from checkllm import ConversationalTestCase, Turn
from checkllm.testing import MockJudge, make_collector


ROLE_DESCRIPTION = (
    "You are a helpful support agent for ACME Cloud. You answer questions "
    "about billing, account settings, and service status. You politely "
    "refuse anything outside that scope."
)


def _build_transcript() -> ConversationalTestCase:
    """Five-turn session where the user asks about billing and plan changes."""
    turns = [
        Turn(role="system", content=ROLE_DESCRIPTION),
        Turn(role="user", content="Hi, I'm on the Pro plan. Can you tell me my next invoice date?"),
        Turn(
            role="assistant",
            content=(
                "Sure. Your Pro plan renews on the 14th of each month, so your next "
                "invoice will be generated on June 14th."
            ),
        ),
        Turn(
            role="user",
            content="Great. If I upgrade to Business now, does billing still happen on the 14th?",
        ),
        Turn(
            role="assistant",
            content=(
                "Upgrades take effect immediately. You'll see a pro-rated charge "
                "today, and your regular billing date remains the 14th."
            ),
        ),
        Turn(role="user", content="Perfect. Please go ahead and upgrade me."),
        Turn(
            role="assistant",
            content=(
                "I've queued the upgrade to Business. You'll receive a confirmation "
                "email and, as mentioned, your next invoice stays on the 14th."
            ),
        ),
        Turn(role="user", content="Thanks! One last thing - what's the weather today?"),
        Turn(
            role="assistant",
            content=(
                "I'm only able to help with ACME Cloud account questions, so I can't "
                "check the weather. Anything else I can do for your account?"
            ),
        ),
    ]
    return ConversationalTestCase(turns=turns, expected_outcome="plan upgraded")


@pytest.fixture
def chat_collector() -> object:
    """Offline collector using MockJudge."""
    judge = MockJudge(default_score=0.9, default_reasoning="Mock: transcript looks coherent")
    return make_collector(judge=judge, threshold=0.8)


def test_conversation_completeness(chat_collector) -> None:
    """Every user request in the transcript should be satisfied."""
    convo = _build_transcript()
    chat_collector.conversation_completeness(conversation=convo, threshold=0.8)
    assert all(r.passed for r in chat_collector.results)


def test_role_adherence(chat_collector) -> None:
    """The assistant must stay within its declared role."""
    convo = _build_transcript()
    chat_collector.role_adherence(
        output=convo.last_response or "",
        role_description=ROLE_DESCRIPTION,
        query=convo.user_turns[-1].content,
        threshold=0.85,
    )
    assert all(r.passed for r in chat_collector.results)


def test_knowledge_retention(chat_collector) -> None:
    """The assistant must remember information from earlier turns."""
    convo = _build_transcript()
    chat_collector.knowledge_retention(conversation=convo, threshold=0.8)
    # Spot-check: the "14th" fact was introduced in turn 2 and echoed in turn 6.
    echoed = sum(1 for t in convo.assistant_turns if "14th" in t.content)
    assert echoed >= 2, "Assistant should reuse the 14th-of-month fact"
    assert all(r.passed for r in chat_collector.results)


def test_per_turn_relevancy(chat_collector) -> None:
    """Per-turn check on the final out-of-scope refusal."""
    convo = _build_transcript()
    last_user = convo.user_turns[-1].content
    last_assistant = convo.last_response or ""

    chat_collector.relevance(
        output=last_assistant,
        query=last_user,
        threshold=0.6,  # refusals read as only partially relevant
    )
    # Deterministic guard for the refusal wording.
    chat_collector.icontains_any(
        last_assistant,
        ["only able to help", "can't help", "outside my scope"],
    )
    assert all(r.passed for r in chat_collector.results)


def test_topic_stays_on_scope(chat_collector) -> None:
    """Topic guard: assistant must not answer off-topic (weather) questions."""
    convo = _build_transcript()
    last_assistant = convo.last_response or ""

    chat_collector.not_contains(last_assistant, "sunny")
    chat_collector.not_contains(last_assistant, "rain")
    chat_collector.max_tokens(last_assistant, limit=80)
    assert all(r.passed for r in chat_collector.results)


def test_full_multi_turn_suite_composed(chat_collector) -> None:
    """Compose all three multi-turn metrics plus deterministic guards."""
    convo = _build_transcript()

    # Deterministic sanity checks
    assert convo.turn_count == 9
    assert len(convo.assistant_turns) == 4

    # Judge metrics
    chat_collector.conversation_completeness(conversation=convo, threshold=0.8)
    chat_collector.role_adherence(
        output=convo.last_response or "",
        role_description=ROLE_DESCRIPTION,
        query=convo.user_turns[-1].content,
        threshold=0.85,
    )
    chat_collector.knowledge_retention(conversation=convo, threshold=0.8)

    assert all(r.passed for r in chat_collector.results), chat_collector.results


# Run with: pytest examples/test_multi_turn_conversation.py

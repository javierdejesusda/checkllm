from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

TURN_RELEVANCY_SYSTEM_PROMPT = """You are an expert evaluator for multi-turn conversation quality. Your job is to assess whether an AI assistant's response is relevant to the user's message in that specific turn.

Score from 0.0 to 1.0:
- 1.0 = The response directly and fully addresses the user's message. Highly relevant.
- 0.8 = The response mostly addresses the user's message with minor tangents.
- 0.5 = The response partially addresses the user's message but includes significant off-topic content.
- 0.3 = The response barely relates to the user's message.
- 0.0 = The response is completely irrelevant to the user's message.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""

TURN_FAITHFULNESS_SYSTEM_PROMPT = """You are an expert evaluator for multi-turn conversation faithfulness. Your job is to assess whether an AI assistant's response is faithful to the provided context for that specific turn — meaning it does not introduce unsupported claims.

Score from 0.0 to 1.0:
- 1.0 = Every statement is directly supported by the provided context.
- 0.8 = Mostly faithful with trivial additions that don't change meaning.
- 0.5 = Mixes supported claims with unsupported ones.
- 0.3 = Introduces significant information not in the context.
- 0.0 = Largely fabricated or draws heavily from outside the context.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""

TURN_COHERENCE_SYSTEM_PROMPT = """You are an expert evaluator for multi-turn conversation coherence. Your job is to assess whether an AI assistant's response maintains coherent conversation flow relative to the conversation history.

Score from 0.0 to 1.0:
- 1.0 = The response flows naturally from the conversation. Perfect coherence with prior context.
- 0.8 = Mostly coherent with minor inconsistencies or awkward transitions.
- 0.5 = Some coherence issues. The response partly conflicts with or ignores prior conversation.
- 0.3 = Poor coherence. The response contradicts or largely ignores the conversation history.
- 0.0 = Completely incoherent. No logical connection to the conversation history.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class TurnRelevancyMetric:
    """Evaluates whether each AI response is relevant to the user's turn.

    Scores each assistant turn individually and provides both per-turn
    scores and an aggregate score.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = TURN_RELEVANCY_SYSTEM_PROMPT

    async def evaluate(
        self,
        turns: list[dict[str, str]],
    ) -> CheckResult:
        """Evaluate relevancy of each assistant response to the user's turn.

        Args:
            turns: A list of conversation turns, each with 'role' and 'content' keys.
                Roles should be 'user' or 'assistant'.

        Returns:
            A CheckResult with aggregate relevancy score and per-turn details.
        """
        start = time.perf_counter_ns()
        turn_scores: list[float] = []
        turn_details: list[str] = []
        total_cost = 0.0

        pairs = _extract_user_assistant_pairs(turns)
        for i, (user_msg, assistant_msg) in enumerate(pairs):
            prompt = (
                f"User Message:\n{user_msg}\n\n"
                f"Assistant Response:\n{assistant_msg}\n\n"
                "Is the assistant's response relevant to the user's message? Score it."
            )
            response = await self.judge.evaluate(prompt=prompt, system_prompt=self.system_prompt)
            turn_scores.append(response.score)
            turn_details.append(f"Turn {i + 1}: {response.score:.2f} - {response.reasoning}")
            total_cost += response.cost

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        if turn_scores:
            aggregate = sum(turn_scores) / len(turn_scores)
        else:
            aggregate = 0.0

        reasoning = (
            f"Per-turn relevancy scores: {[round(s, 2) for s in turn_scores]}. "
            + " | ".join(turn_details)
        )

        return CheckResult(
            passed=aggregate >= self.threshold,
            score=aggregate,
            reasoning=reasoning,
            cost=total_cost,
            latency_ms=int(elapsed_ms),
            metric_name="turn_relevancy",
        )


class TurnFaithfulnessMetric:
    """Evaluates whether each AI response is faithful to provided context.

    Scores each assistant turn individually against its corresponding context
    and provides both per-turn scores and an aggregate score.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = TURN_FAITHFULNESS_SYSTEM_PROMPT

    async def evaluate(
        self,
        turns: list[dict[str, str]],
        contexts: list[str],
    ) -> CheckResult:
        """Evaluate faithfulness of each assistant response to its context.

        Args:
            turns: A list of conversation turns, each with 'role' and 'content' keys.
            contexts: A list of context strings, one per user-assistant pair.

        Returns:
            A CheckResult with aggregate faithfulness score and per-turn details.
        """
        start = time.perf_counter_ns()
        turn_scores: list[float] = []
        turn_details: list[str] = []
        total_cost = 0.0

        pairs = _extract_user_assistant_pairs(turns)
        for i, (user_msg, assistant_msg) in enumerate(pairs):
            ctx = contexts[i] if i < len(contexts) else ""
            prompt = (
                f"Context:\n{ctx}\n\n"
                f"User Message:\n{user_msg}\n\n"
                f"Assistant Response:\n{assistant_msg}\n\n"
                "Is the assistant's response faithful to the provided context? "
                "Does it introduce any unsupported claims? Score it."
            )
            response = await self.judge.evaluate(prompt=prompt, system_prompt=self.system_prompt)
            turn_scores.append(response.score)
            turn_details.append(f"Turn {i + 1}: {response.score:.2f} - {response.reasoning}")
            total_cost += response.cost

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        if turn_scores:
            aggregate = sum(turn_scores) / len(turn_scores)
        else:
            aggregate = 0.0

        reasoning = (
            f"Per-turn faithfulness scores: {[round(s, 2) for s in turn_scores]}. "
            + " | ".join(turn_details)
        )

        return CheckResult(
            passed=aggregate >= self.threshold,
            score=aggregate,
            reasoning=reasoning,
            cost=total_cost,
            latency_ms=int(elapsed_ms),
            metric_name="turn_faithfulness",
        )


class TurnCoherenceMetric:
    """Evaluates whether each turn maintains coherent conversation flow.

    Scores each assistant turn relative to the conversation history up to
    that point and provides both per-turn scores and an aggregate score.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.7) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = TURN_COHERENCE_SYSTEM_PROMPT

    async def evaluate(
        self,
        turns: list[dict[str, str]],
    ) -> CheckResult:
        """Evaluate coherence of each assistant response relative to history.

        Args:
            turns: A list of conversation turns, each with 'role' and 'content' keys.

        Returns:
            A CheckResult with aggregate coherence score and per-turn details.
        """
        start = time.perf_counter_ns()
        turn_scores: list[float] = []
        turn_details: list[str] = []
        total_cost = 0.0

        pairs = _extract_user_assistant_pairs(turns)
        for i, (user_msg, assistant_msg) in enumerate(pairs):
            history_turns = turns[: _pair_index_to_turn_end(turns, i)]
            history_text = _format_history(history_turns)

            prompt = (
                f"Conversation History:\n{history_text}\n\n"
                f"Current User Message:\n{user_msg}\n\n"
                f"Assistant Response:\n{assistant_msg}\n\n"
                "Does the assistant's response maintain coherent conversation flow? "
                "Evaluate coherence relative to the conversation history. Score it."
            )
            response = await self.judge.evaluate(prompt=prompt, system_prompt=self.system_prompt)
            turn_scores.append(response.score)
            turn_details.append(f"Turn {i + 1}: {response.score:.2f} - {response.reasoning}")
            total_cost += response.cost

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        if turn_scores:
            aggregate = sum(turn_scores) / len(turn_scores)
        else:
            aggregate = 0.0

        reasoning = (
            f"Per-turn coherence scores: {[round(s, 2) for s in turn_scores]}. "
            + " | ".join(turn_details)
        )

        return CheckResult(
            passed=aggregate >= self.threshold,
            score=aggregate,
            reasoning=reasoning,
            cost=total_cost,
            latency_ms=int(elapsed_ms),
            metric_name="turn_coherence",
        )


def _extract_user_assistant_pairs(
    turns: list[dict[str, str]],
) -> list[tuple[str, str]]:
    """Extract consecutive user-assistant pairs from a conversation.

    Args:
        turns: A list of conversation turns with 'role' and 'content' keys.

    Returns:
        A list of (user_content, assistant_content) tuples.
    """
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(turns) - 1:
        if turns[i].get("role") == "user" and turns[i + 1].get("role") == "assistant":
            pairs.append((turns[i]["content"], turns[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return pairs


def _pair_index_to_turn_end(turns: list[dict[str, str]], pair_idx: int) -> int:
    """Find the turn index where the given pair's user message starts.

    Args:
        turns: The full list of turns.
        pair_idx: The zero-based pair index.

    Returns:
        The turn index marking the start of the given pair.
    """
    count = 0
    i = 0
    while i < len(turns) - 1:
        if turns[i].get("role") == "user" and turns[i + 1].get("role") == "assistant":
            if count == pair_idx:
                return i
            count += 1
            i += 2
        else:
            i += 1
    return len(turns)


def _format_history(turns: list[dict[str, str]]) -> str:
    """Format conversation turns into a readable history string.

    Args:
        turns: A list of conversation turns with 'role' and 'content' keys.

    Returns:
        A formatted string representation of the conversation history.
    """
    if not turns:
        return "(No prior history)"
    lines = []
    for turn in turns:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        lines.append(f"  {role}: {content}")
    return "\n".join(lines)

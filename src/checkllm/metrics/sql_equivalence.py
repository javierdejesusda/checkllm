from __future__ import annotations

import time

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult

SQL_EQUIVALENCE_SYSTEM_PROMPT = """You are an expert SQL equivalence evaluator. Your task is to determine whether two SQL queries are semantically equivalent, meaning they would produce the same result set on any valid database conforming to the given schema.

Two queries are semantically equivalent if they return identical rows and columns for all possible database states, even if they use different syntax, join orders, subquery structures, or aliasing.

Evaluation process:
1. Parse both queries to understand their logical intent.
2. Compare the SELECT columns (accounting for aliases and expressions).
3. Compare the FROM/JOIN structure (accounting for equivalent join reorderings).
4. Compare WHERE/HAVING conditions (accounting for equivalent logical expressions).
5. Compare GROUP BY, ORDER BY, and LIMIT clauses.
6. If a schema is provided, verify both queries are valid against it.
7. Determine if the queries produce the same result set in all cases.

Score from 0.0 to 1.0:
- 1.0 = The queries are fully semantically equivalent; they produce identical results for all valid inputs.
- 0.8 = The queries are nearly equivalent; minor differences (e.g., column ordering, extra columns) that do not affect the core result.
- 0.5 = The queries share the same intent but have structural differences that could produce different results in edge cases.
- 0.3 = The queries target similar data but have significant logical differences.
- 0.0 = The queries are completely different in logic and would produce entirely different results.

Key evaluation criteria:
1. Do both queries select the same columns (semantically, not just syntactically)?
2. Do they filter on the same conditions?
3. Do they handle NULLs, duplicates, and edge cases the same way?
4. Would rewriting one query lead to the other?

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class SQLEquivalenceMetric:
    """Judges whether two SQL queries are semantically equivalent.

    Compares a generated SQL query against a reference query, optionally
    considering a database schema, to determine if they would produce
    the same result set for all valid database states.
    """

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = SQL_EQUIVALENCE_SYSTEM_PROMPT

    async def evaluate(
        self, output: str, reference: str, schema: str | None = None
    ) -> CheckResult:
        """Evaluate semantic equivalence of two SQL queries.

        Args:
            output: The generated SQL query.
            reference: The expected/reference SQL query.
            schema: Optional database schema for validation context.

        Returns:
            CheckResult with equivalence score.
        """
        parts = []
        if schema:
            parts.append(f"Database Schema:\n{schema}\n")
        parts.append(f"Reference SQL:\n{reference}\n")
        parts.append(f"Generated SQL:\n{output}\n")
        parts.append(
            "Are these two SQL queries semantically equivalent? "
            "Would they produce the same result set? Score the equivalence."
        )
        prompt = "\n".join(parts)

        start = time.perf_counter_ns()
        response = await self.judge.evaluate(
            prompt=prompt, system_prompt=self.system_prompt
        )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="sql_equivalence",
        )

"""Generate tutorial notebooks under docs/notebooks.

Run once with ``python scripts/_build_tutorials.py``. Notebooks are written
with outputs cleared so that ``git diff`` stays clean.
"""

from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).resolve().parents[1] / "docs" / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def _nb(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = cells
    return nb


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


FAKE_JUDGE_CELL = '''from checkllm.judge import JudgeResponse


class FakeJudge:
    """Deterministic in-process judge used to keep this notebook offline.

    Real runs should swap this for ``OpenAIJudge`` / ``AnthropicJudge`` etc.
    """

    def __init__(self, score: float = 0.9, reasoning: str = "looks good") -> None:
        self._score = score
        self._reasoning = reasoning

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> JudgeResponse:
        return JudgeResponse(
            score=self._score,
            reasoning=self._reasoning,
            raw_output=str(self._score),
            cost=0.0,
        )


judge = FakeJudge(score=0.9, reasoning="offline stub")
'''


SWITCH_TO_REAL_CELL = """# -------------------------------------------------------------------
# SWITCH TO A REAL PROVIDER
# -------------------------------------------------------------------
# Uncomment and replace the FakeJudge above once you have API keys:
#
# from checkllm.judge import OpenAIJudge
# judge = OpenAIJudge(api_key="sk-...", model="gpt-4o-mini")
#
# from checkllm.judge import AnthropicJudge
# judge = AnthropicJudge(api_key="...", model="claude-3-5-sonnet-20241022")
"""


# ---------------------------------------------------------------------------
# 01 — Quickstart
# ---------------------------------------------------------------------------
nb1 = _nb(
    [
        md(
            "# 01 — Quickstart\n\n"
            "Welcome to **checkllm**, the pytest of LLM testing.\n\n"
            "This notebook walks through:\n\n"
            "1. Installing checkllm.\n"
            "2. Running your first deterministic check.\n"
            "3. Running an LLM-judged metric **without a real API key**.\n"
            "4. Interpreting the structured results.\n\n"
            "Everything below is fully runnable offline — we plug in a recorded "
            "`FakeJudge` for the judged metrics. A cell near the bottom shows how "
            "to switch to a real provider."
        ),
        md("## 1. Install\n\nUncomment if checkllm is not yet installed:"),
        code("# !pip install checkllm"),
        md(
            "## 2. Deterministic checks\n\n"
            "Deterministic checks never call an LLM. They are the safest place "
            "to start — free, instant, and fully reproducible."
        ),
        code(
            "from checkllm.deterministic import DeterministicChecks\n"
            "\n"
            "det = DeterministicChecks()\n"
            'output = "Python is a high-level programming language created by Guido van Rossum."\n'
            "\n"
            'print("contains Python:", det.contains(output, "Python").passed)\n'
            'print("contains Guido :", det.contains(output, "Guido").passed)\n'
            'print("no JavaScript :", det.not_contains(output, "JavaScript").passed)\n'
            'print("under 50 tokens:", det.max_tokens(output, limit=50).passed)\n'
        ),
        md(
            "## 3. LLM-judged metrics with a FakeJudge\n\n"
            "Judged metrics call an LLM to grade the response. To keep the "
            "notebook runnable offline we wire in a `FakeJudge` that returns a "
            "pre-recorded score."
        ),
        code(FAKE_JUDGE_CELL),
        code(
            "# Jupyter supports top-level ``await``; outside notebooks use\n"
            "# ``asyncio.run`` instead.\n"
            'result = await judge.evaluate(prompt="Is the sky blue?")\n'
            'print("score     :", result.score)\n'
            'print("reasoning :", result.reasoning)\n'
            'print("cost      :", result.cost)\n'
        ),
        md(
            "## 4. Interpreting results\n\n"
            "`JudgeResponse` is a Pydantic model. The fields you typically "
            "care about are:\n\n"
            "| Field | Meaning |\n"
            "|-------|---------|\n"
            "| `score` | value in `[0.0, 1.0]`, higher is better |\n"
            "| `reasoning` | textual justification from the judge |\n"
            "| `cost` | estimated USD (0.0 for the offline stub) |\n\n"
            "Pair the score with a threshold (commonly 0.7 or 0.8) and fail the "
            "surrounding test when the score falls below it."
        ),
        md("## 5. Switch to a real provider"),
        code(SWITCH_TO_REAL_CELL),
    ]
)
nbf.write(nb1, OUT / "01_quickstart.ipynb")


# ---------------------------------------------------------------------------
# 02 — RAG evaluation
# ---------------------------------------------------------------------------
nb2 = _nb(
    [
        md(
            "# 02 — RAG Evaluation\n\n"
            "Evaluate a toy Retrieval-Augmented Generation pipeline end to end:\n\n"
            "* **Faithfulness** — is the answer grounded in the retrieved context?\n"
            "* **Context Precision / Recall** — are the right documents retrieved?\n"
            "* **NDCG / MRR** — ranking-quality of the retriever.\n\n"
            "All judged metrics use an offline `FakeJudge`."
        ),
        md("## 1. A toy RAG corpus and retriever"),
        code(
            "CORPUS = [\n"
            '    {"id": "d1", "text": "Python was created by Guido van Rossum in 1991."},\n'
            '    {"id": "d2", "text": "Python emphasizes code readability."},\n'
            '    {"id": "d3", "text": "Rust was first released in 2010 by Graydon Hoare."},\n'
            '    {"id": "d4", "text": "Java was released by Sun Microsystems in 1995."},\n'
            "]\n"
            "\n"
            "\n"
            "def retrieve(query: str, k: int = 2):\n"
            '    """Keyword retriever that returns the top-k documents containing query terms."""\n'
            "    terms = set(query.lower().split())\n"
            "    scored = []\n"
            "    for doc in CORPUS:\n"
            '        overlap = sum(1 for t in terms if t in doc["text"].lower())\n'
            "        if overlap:\n"
            "            scored.append((overlap, doc))\n"
            "    scored.sort(key=lambda x: x[0], reverse=True)\n"
            "    return [d for _, d in scored[:k]]\n"
            "\n"
            "\n"
            'question = "Who created Python?"\n'
            "retrieved = retrieve(question, k=2)\n"
            "for doc in retrieved:\n"
            "    print(doc)\n"
        ),
        md("## 2. Deterministic retrieval-quality metrics (no LLM needed)"),
        code(
            'relevant_ids = {"d1", "d2"}\n'
            'retrieved_ids = [d["id"] for d in retrieved]\n'
            "\n"
            "hits = [1 if rid in relevant_ids else 0 for rid in retrieved_ids]\n"
            "precision = sum(hits) / len(hits) if hits else 0.0\n"
            "recall = sum(hits) / len(relevant_ids) if relevant_ids else 0.0\n"
            'print("Precision@k:", round(precision, 3))\n'
            'print("Recall@k   :", round(recall, 3))\n'
        ),
        md("## 3. Ranking-quality: NDCG and MRR"),
        code(
            "import math\n"
            "\n"
            "\n"
            "def dcg(gains):\n"
            "    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))\n"
            "\n"
            "\n"
            "def ndcg(ranked_ids, relevant_ids):\n"
            "    gains = [1.0 if rid in relevant_ids else 0.0 for rid in ranked_ids]\n"
            "    ideal = sorted(gains, reverse=True)\n"
            "    if not any(ideal):\n"
            "        return 0.0\n"
            "    return dcg(gains) / dcg(ideal)\n"
            "\n"
            "\n"
            "def mrr(ranked_ids, relevant_ids):\n"
            "    for i, rid in enumerate(ranked_ids, start=1):\n"
            "        if rid in relevant_ids:\n"
            "            return 1.0 / i\n"
            "    return 0.0\n"
            "\n"
            "\n"
            'print("NDCG:", round(ndcg(retrieved_ids, relevant_ids), 3))\n'
            'print("MRR :", round(mrr(retrieved_ids, relevant_ids), 3))\n'
        ),
        md("## 4. Faithfulness of the generated answer (offline judge)"),
        code(FAKE_JUDGE_CELL),
        code(
            'answer = "Python was created by Guido van Rossum in 1991."\n'
            'context = "\\n".join(d["text"] for d in retrieved)\n'
            "\n"
            "prompt = (\n"
            '    f"Question: {question}\\nContext: {context}\\nAnswer: {answer}\\n"\n'
            '    "Is the answer faithful to the context? Score 0-1."\n'
            ")\n"
            "result = await judge.evaluate(prompt=prompt)\n"
            'print("faithfulness:", result.score, "|", result.reasoning)\n'
        ),
        md("## 5. Switch to a real provider"),
        code(SWITCH_TO_REAL_CELL),
    ]
)
nbf.write(nb2, OUT / "02_rag_evaluation.ipynb")


# ---------------------------------------------------------------------------
# 03 — Conversational evaluation
# ---------------------------------------------------------------------------
nb3 = _nb(
    [
        md(
            "# 03 — Conversational Evaluation\n\n"
            "Multi-turn chat evaluation with `ConversationalTestCase`.\n\n"
            "We cover:\n\n"
            "* Building a transcript from `Turn` objects.\n"
            "* Role-adherence and conversation-completeness heuristics.\n"
            "* Per-turn judged scoring with an offline judge."
        ),
        md("## 1. Build a multi-turn test case"),
        code(
            "from checkllm.conversation import ConversationalTestCase, Turn\n"
            "\n"
            "tc = ConversationalTestCase(\n"
            "    turns=[\n"
            '        Turn(role="system",    content="You are a helpful support agent."),\n'
            '        Turn(role="user",      content="My package hasn\'t arrived."),\n'
            '        Turn(role="assistant", content="I\'m sorry about that. Could you share your order number?"),\n'
            '        Turn(role="user",      content="Sure, it is #A-123."),\n'
            '        Turn(role="assistant", content="Thanks! Your order shipped yesterday and should arrive tomorrow."),\n'
            "    ]\n"
            ")\n"
            "\n"
            'print("# turns      :", tc.turn_count)\n'
            'print("first user  :", tc.first_user_message)\n'
            'print("last reply  :", tc.last_response)\n'
            "print()\n"
            "print(tc.format_transcript())\n"
        ),
        md("## 2. Deterministic turn-level checks"),
        code(
            "assistant_turns = tc.assistant_turns\n"
            "\n"
            "# Every assistant turn must acknowledge the user; simple containment proxy.\n"
            'has_apology = any("sorry" in t.content.lower() for t in assistant_turns)\n'
            'asks_order_num = any("order number" in t.content.lower() for t in assistant_turns)\n'
            "\n"
            'print("expressed empathy:", has_apology)\n'
            'print("collected info  :", asks_order_num)\n'
        ),
        md("## 3. Per-turn judged scoring (offline)"),
        code(FAKE_JUDGE_CELL),
        code(
            "per_turn_scores = []\n"
            "for turn in tc.assistant_turns:\n"
            "    prompt = (\n"
            '        "Assistant reply: " + turn.content + "\\n"\n'
            '        "Rate role adherence on 0-1 (support agent)."\n'
            "    )\n"
            "    resp = await judge.evaluate(prompt=prompt)\n"
            "    per_turn_scores.append(resp.score)\n"
            "\n"
            'print("per-turn     :", per_turn_scores)\n'
            'print("conv. average:", sum(per_turn_scores) / len(per_turn_scores))\n'
        ),
        md("## 4. Switch to a real provider"),
        code(SWITCH_TO_REAL_CELL),
    ]
)
nbf.write(nb3, OUT / "03_conversational_eval.ipynb")


# ---------------------------------------------------------------------------
# 04 — Agent trajectory
# ---------------------------------------------------------------------------
nb4 = _nb(
    [
        md(
            "# 04 — Agent Trajectory Evaluation\n\n"
            "Evaluate a tool-using agent with `AgentTestCase`:\n\n"
            "* Validate that expected tool calls happened.\n"
            "* Detect loops and step efficiency.\n"
            "* Judge the final answer with an offline judge."
        ),
        md("## 1. Record an agent trajectory"),
        code(
            "from checkllm.agents import AgentStep, AgentTestCase, ToolCall\n"
            "\n"
            "tc = AgentTestCase(\n"
            '    query="What is the weather in Paris and Berlin?",\n'
            "    steps=[\n"
            "        AgentStep(\n"
            '            action="call_tool",\n'
            '            tool_call=ToolCall(name="weather", parameters={"city": "Paris"}, result="22C sunny"),\n'
            "        ),\n"
            "        AgentStep(\n"
            '            action="call_tool",\n'
            '            tool_call=ToolCall(name="weather", parameters={"city": "Berlin"}, result="18C cloudy"),\n'
            "        ),\n"
            '        AgentStep(action="respond"),\n'
            "    ],\n"
            "    expected_tools=[\n"
            '        ToolCall(name="weather", parameters={"city": "Paris"}),\n'
            '        ToolCall(name="weather", parameters={"city": "Berlin"}),\n'
            "    ],\n"
            '    final_output="Paris is 22C sunny, Berlin is 18C cloudy.",\n'
            ")\n"
            "\n"
            "print(tc.format_trace())\n"
        ),
        md("## 2. Validate expected tool calls"),
        code(
            "from checkllm.agents import validate_tool_calls\n"
            "\n"
            "result = validate_tool_calls(tc)\n"
            'print("tool-call validation:", result.passed, "| score:", result.score)\n'
            'print("reasoning:", result.reasoning)\n'
        ),
        md("## 3. Loop detection and step efficiency"),
        code(
            "from collections import Counter\n"
            "\n"
            "tool_names = [call.name for call in tc.tool_calls]\n"
            "most_common, count = Counter(tool_names).most_common(1)[0]\n"
            'print(f"tool {most_common!r} called {count} times")\n'
            'print("total steps        :", len(tc.steps))\n'
            "has_loop = count > 5\n"
            'print("suspected loop     :", has_loop)\n'
        ),
        md("## 4. Judge the final answer offline"),
        code(FAKE_JUDGE_CELL),
        code(
            "prompt = (\n"
            '    "Query: " + tc.query + "\\n"\n'
            '    "Answer: " + (tc.final_output or "") + "\\n"\n'
            '    "Is the answer correct and helpful? Score 0-1."\n'
            ")\n"
            "resp = await judge.evaluate(prompt=prompt)\n"
            'print("answer-quality:", resp.score, "|", resp.reasoning)\n'
        ),
        md("## 5. Switch to a real provider"),
        code(SWITCH_TO_REAL_CELL),
    ]
)
nbf.write(nb4, OUT / "04_agent_trajectory.ipynb")


# ---------------------------------------------------------------------------
# 05 — Red teaming
# ---------------------------------------------------------------------------
nb5 = _nb(
    [
        md(
            "# 05 — Red Teaming\n\n"
            "Generate adversarial prompts, run them against a stub target, and "
            "roll the results up into an OWASP-style scorecard.\n\n"
            "This notebook mocks the model response with a safe refusal so it "
            "runs offline. Swap in your real model in the last cell."
        ),
        md("## 1. Build a stub target"),
        code(
            "async def safe_target(prompt: str) -> str:\n"
            '    """Simulated model that safely refuses every adversarial ask."""\n'
            "    bad_terms = (\n"
            '        "ignore", "jailbreak", "bypass", "override",\n'
            '        "malicious", "exploit", "pretend",\n'
            "    )\n"
            "    if any(term in prompt.lower() for term in bad_terms):\n"
            '        return "I can\'t help with that request."\n'
            '    return "Here is a helpful answer."\n'
        ),
        md("## 2. Run a small red-team scan"),
        code(
            "from checkllm.redteam import RedTeamer, VulnerabilityType\n"
            "\n"
            "red = RedTeamer()  # heuristic detection — no judge needed\n"
            "report = await red.scan(\n"
            "    target=safe_target,\n"
            "    vulnerability_types=[\n"
            "        VulnerabilityType.PROMPT_INJECTION,\n"
            "        VulnerabilityType.JAILBREAK,\n"
            "    ],\n"
            "    attacks_per_type=3,\n"
            ")\n"
            "print(report.summary())\n"
        ),
        md("## 3. OWASP scorecard and vulnerability rollups"),
        code(
            'print("Attacks run          :", report.total_attacks)\n'
            'print("Attacks succeeded    :", report.successful_attacks)\n'
            'print("OWASP score (0-1)    :", round(report.owasp_score, 3))\n'
            'print("Risk level           :", report.risk_level)\n'
            'print("Vulnerabilities/type :")\n'
            "for vt, count in sorted(report.by_type.items()):\n"
            '    print(f"  {vt}: {count}")\n'
        ),
        md(
            "## 4. Switch to a real provider\n\n"
            "Replace `safe_target` with an async wrapper around your production "
            "model. Optionally pass a real judge so that attack success is "
            "evaluated by an LLM instead of heuristics:\n\n"
            "```python\n"
            "from checkllm.judge import OpenAIJudge\n"
            "from checkllm.redteam import RedTeamer\n\n"
            "judge = OpenAIJudge(api_key='sk-...', model='gpt-4o-mini')\n"
            "red = RedTeamer(judge=judge)\n"
            "```"
        ),
    ]
)
nbf.write(nb5, OUT / "05_redteam.ipynb")


# Clear outputs on every notebook we just wrote.
for path in sorted(OUT.glob("*.ipynb")):
    nb = nbf.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbf.write(nb, path)

print("Wrote tutorials to", OUT)
print(os.listdir(OUT))

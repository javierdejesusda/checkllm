"""CheckLLM-unique capability showcase harness.

Generates a structured report demonstrating capabilities that have no direct
counterpart in DeepEval, Ragas, or promptfoo: compliance scanning, red-team
strategies, knowledge-graph test synthesis, and agent trajectory metrics.

This is a capability demonstration, not a ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from checkllm.judge import JudgeBackend


@dataclass
class ShowcaseReport:
    """Structured result for one capability showcase section.

    Attributes:
        section: Short identifier for the capability being demonstrated.
        summary: Human-readable description of what was run and what happened.
        total: Number of items evaluated (tests, trajectories, samples, etc.).
        pass_count: Number of items that passed or were produced successfully.
        samples: Up to five example findings for display in the Markdown report.
    """

    section: str
    summary: str
    total: int
    pass_count: int
    samples: list[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the report section as a Markdown string.

        Returns:
            A Markdown-formatted string for the section.
        """
        lines = [
            f"## {self.section}",
            "",
            f"- Total: {self.total}",
            f"- Passed: {self.pass_count}",
            "",
            self.summary,
            "",
        ]
        if self.samples:
            lines.append("### Example findings")
            for s in self.samples[:5]:
                lines.append(f"- {s}")
        return "\n".join(lines)


TargetFn = Callable[[str], Awaitable[str]]


async def run_compliance_showcase(
    target: TargetFn,
    judge: JudgeBackend,
    frameworks: list[str],
    attacks_per_type: int = 1,
) -> ShowcaseReport:
    """Run a compliance scan showcase using ComplianceScanner.

    Args:
        target: Async callable accepting a prompt string and returning the
            model response.
        judge: JudgeBackend instance for evaluating responses.
        frameworks: List of ComplianceFramework enum member names as strings,
            e.g. ``["OWASP_LLM_TOP10"]``.
        attacks_per_type: Unused (ComplianceScanner does not expose this
            parameter); kept for API symmetry with other showcase runners.

    Returns:
        A ShowcaseReport with section ``"compliance"``.
    """
    try:
        from checkllm.compliance_frameworks import (
            ComplianceFramework,
            ComplianceScanner,
        )

        scanner = ComplianceScanner(judge=judge)
        enum_frameworks = [ComplianceFramework[name] for name in frameworks]
        multi_report = await scanner.scan(
            target=target,
            frameworks=enum_frameworks,
        )
        # Aggregate across all per-framework ComplianceReport objects.
        total = sum(r.total_requirements for r in multi_report.reports)
        pass_count = sum(r.passed for r in multi_report.reports)
        summary = f"Scanned {', '.join(frameworks)} via ComplianceScanner"
        samples = [{"framework": f} for f in frameworks]
    except Exception as exc:
        total = 0
        pass_count = 0
        summary = f"compliance showcase error: {exc}"
        samples = []

    return ShowcaseReport(
        section="compliance",
        summary=summary,
        total=total,
        pass_count=pass_count,
        samples=samples,
    )


async def run_redteam_showcase(
    target: TargetFn,
    judge: JudgeBackend,
    vulnerability_types: list[str],
    strategies: list[str],
    attacks_per_type: int = 1,
) -> ShowcaseReport:
    """Run a red-team scan showcase using RedTeamer.

    Args:
        target: Async callable accepting a prompt string and returning the
            model response.
        judge: JudgeBackend instance for evaluating responses.
        vulnerability_types: List of VulnerabilityType enum member names as
            strings, e.g. ``["PROMPT_INJECTION"]``.
        strategies: List of StrategyType enum member names as strings, e.g.
            ``["BASE64"]``.
        attacks_per_type: Number of attack prompts per (type, strategy) pair.

    Returns:
        A ShowcaseReport with section ``"redteam"``.
    """
    try:
        from checkllm.redteam import RedTeamer, VulnerabilityType
        from checkllm.redteam_strategies import StrategyType

        vuln_enum = [VulnerabilityType[v] for v in vulnerability_types]
        strat_enum = [StrategyType[s] for s in strategies]
        red = RedTeamer(judge=judge)
        report = await red.scan(
            target=target,
            vulnerability_types=vuln_enum,
            strategies=strat_enum,
            attacks_per_type=attacks_per_type,
        )
        results = getattr(report, "results", [])
        total = len(results)
        # AttackResult.vulnerable is True when the attack SUCCEEDED (bad).
        # A "pass" here means the model was NOT vulnerable.
        pass_count = sum(1 for r in results if not getattr(r, "vulnerable", True))
        summary = (
            f"Ran {len(strategies)} strategies against "
            f"{len(vulnerability_types)} vulnerability types"
        )
        samples = [{"vuln": v, "strategies": strategies} for v in vulnerability_types]
    except Exception as exc:
        total = 0
        pass_count = 0
        summary = f"redteam showcase error: {exc}"
        samples = []

    return ShowcaseReport(
        section="redteam",
        summary=summary,
        total=total,
        pass_count=pass_count,
        samples=samples,
    )


async def run_kg_synthesis_showcase(
    judge: JudgeBackend,
    documents: list[str],
    num_samples: int = 10,
) -> ShowcaseReport:
    """Run a knowledge-graph test synthesis showcase using KGTestGenerator.

    Args:
        judge: JudgeBackend instance used for entity/theme extraction and
            question synthesis.
        documents: Raw document texts to generate test samples from.
        num_samples: Total number of test samples to generate.

    Returns:
        A ShowcaseReport with section ``"knowledge_graph"``.
    """
    try:
        from checkllm.knowledge_graph import KGTestGenerator

        gen = KGTestGenerator(judge=judge)
        samples = await gen.generate(
            documents=documents,
            num_samples=num_samples,
            synthesizers={"single_hop": 1.0},
            personas=1,
        )
        total = len(samples)
        pass_count = total
        summary = f"Generated {total} test cases from {len(documents)} documents"
        sample_previews = [{"query": getattr(s, "query", str(s))[:120]} for s in samples[:5]]
    except Exception as exc:
        total = 0
        pass_count = 0
        summary = f"kg showcase error: {exc}"
        sample_previews = []

    return ShowcaseReport(
        section="knowledge_graph",
        summary=summary,
        total=total,
        pass_count=pass_count,
        samples=sample_previews,
    )


async def run_trajectory_showcase(
    judge: JudgeBackend,
    trajectories: list[tuple[list[dict], list[str]]],
) -> ShowcaseReport:
    """Run an agent trajectory evaluation showcase.

    Each trajectory is a list of step dicts containing at least a ``"tool"``
    key. The expected tool sequence is a parallel list of tool name strings.

    Args:
        judge: JudgeBackend instance for semantic sequence matching.
        trajectories: List of ``(steps, expected_tools)`` pairs. Each
            ``steps`` entry is a dict with a ``"tool"`` key. ``expected_tools``
            is the ordered list of tool names that should have been called.

    Returns:
        A ShowcaseReport with section ``"trajectory"``. ``total`` equals the
        number of trajectories evaluated.
    """
    results = []
    try:
        from checkllm.metrics.trajectory import TrajectoryToolSequenceMetric

        metric = TrajectoryToolSequenceMetric(judge=judge)
        for traj, expected in trajectories:
            actual_tools = [step.get("tool", "") for step in traj]
            r = await metric.evaluate(
                actual_tool_sequence=actual_tools,
                expected_tool_sequence=expected,
            )
            results.append(r)
        total = len(results)
        pass_count = sum(1 for r in results if getattr(r, "passed", False))
        summary = f"Scored {len(trajectories)} trajectories against expected tool sequences"
        samples = [
            {
                "score": getattr(r, "score", None),
                "reasoning": getattr(r, "reasoning", "")[:120],
            }
            for r in results[:5]
        ]
    except Exception as exc:
        total = len(trajectories)
        pass_count = 0
        summary = f"trajectory showcase error: {exc}"
        samples = []

    return ShowcaseReport(
        section="trajectory",
        summary=summary,
        total=total,
        pass_count=pass_count,
        samples=samples,
    )


def write_showcase_markdown(reports: list[ShowcaseReport], path) -> None:
    """Write all showcase reports to a Markdown file.

    Args:
        reports: List of ShowcaseReport objects to render.
        path: Destination file path (str or Path). Parent directories are
            created automatically.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(r.to_markdown() for r in reports)
    path.write_text(
        "# CheckLLM Capability Showcase\n\n"
        "Features in this document have no direct competitor in DeepEval, Ragas, "
        "or promptfoo. This is a capability demonstration, not a ranking.\n\n" + body,
        encoding="utf-8",
    )

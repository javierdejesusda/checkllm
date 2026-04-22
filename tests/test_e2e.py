"""End-to-end smoke test using checkllm as an end user would."""


def test_deterministic_checks_pass(check):
    """Use the check fixture for basic deterministic checks."""
    output = "Python is a high-level programming language created by Guido van Rossum."

    check.contains(output, "Python")
    check.contains(output, "Guido van Rossum")
    check.not_contains(output, "JavaScript")
    check.regex(output, pattern=r"created by \w+")
    check.max_tokens(output, limit=50)


def test_deterministic_checks_catch_failures(check):
    """Verify that passing checks don't raise."""
    check.contains("hello world", "hello")
    check.not_contains("hello world", "goodbye")
    check.latency(250, max_ms=1000)
    check.cost(0.002, max_usd=0.01)


def test_json_schema_validation(check):
    """Verify JSON schema check works end-to-end."""
    from pydantic import BaseModel

    class Response(BaseModel):
        answer: str
        confidence: float

    check.json_schema(
        '{"answer": "42", "confidence": 0.95}',
        schema=Response,
    )


def test_dataset_driven():
    """Verify dataset loading works end-to-end."""
    from checkllm.datasets.loader import load_dataset
    from checkllm.datasets.case import Case

    def my_cases():
        yield Case(input="What is Python?", query="programming language")
        yield Case(input="What is 2+2?", query="math")

    cases = load_dataset(my_cases)
    assert len(cases) == 2
    assert cases[0].input == "What is Python?"


def test_snapshot_roundtrip(tmp_path):
    """Verify snapshot save/load works end-to-end."""
    from checkllm.regression.snapshot import (
        Snapshot,
        TestRunRecord,
        MetricRecord,
        save_snapshot,
        load_snapshot,
    )

    snap = Snapshot(
        version=1,
        tests={
            "test_my_agent": [
                TestRunRecord(
                    metrics={
                        "hallucination": MetricRecord(score=0.92, passed=True),
                        "relevance": MetricRecord(score=0.88, passed=True),
                    }
                ),
                TestRunRecord(
                    metrics={
                        "hallucination": MetricRecord(score=0.90, passed=True),
                        "relevance": MetricRecord(score=0.85, passed=True),
                    }
                ),
            ]
        },
    )

    path = tmp_path / "baseline.json"
    save_snapshot(snap, path)
    loaded = load_snapshot(path)

    assert loaded.get_scores("test_my_agent", "hallucination") == [0.92, 0.90]
    assert loaded.get_scores("test_my_agent", "relevance") == [0.88, 0.85]


def test_regression_detection():
    """Verify regression detection works end-to-end."""
    from checkllm.regression.snapshot import Snapshot, TestRunRecord, MetricRecord
    from checkllm.regression.compare import compare_snapshot

    baseline = Snapshot(
        version=1,
        tests={
            "test_quality": [
                TestRunRecord(metrics={"score": MetricRecord(score=s, passed=True)})
                for s in [0.90, 0.92, 0.88, 0.91, 0.89]
            ]
        },
    )
    current = Snapshot(
        version=1,
        tests={
            "test_quality": [
                TestRunRecord(metrics={"score": MetricRecord(score=s, passed=False)})
                for s in [0.50, 0.52, 0.48, 0.51, 0.49]
            ]
        },
    )

    report = compare_snapshot(baseline, current)
    assert report.has_regressions is True
    assert report.regressions[0].test_name == "test_quality"


def test_html_report_generation(tmp_path):
    """Verify HTML report generates a valid file."""
    from checkllm.models import CheckResult
    from checkllm.reporting.html import generate_html_report

    results = {
        "test_my_agent": [
            CheckResult(
                passed=True,
                score=0.95,
                reasoning="Grounded",
                cost=0.003,
                latency_ms=450,
                metric_name="hallucination",
            ),
            CheckResult(
                passed=True,
                score=0.88,
                reasoning="Relevant",
                cost=0.002,
                latency_ms=320,
                metric_name="relevance",
            ),
            CheckResult(
                passed=False,
                score=0.3,
                reasoning="Too verbose",
                cost=0.001,
                latency_ms=200,
                metric_name="rubric",
            ),
        ]
    }

    report_path = tmp_path / "report.html"
    generate_html_report(results, report_path)

    content = report_path.read_text()
    assert "checkllm Report" in content
    assert "hallucination" in content
    assert "Too verbose" in content


def test_cli_version():
    """Verify CLI is accessible."""
    from typer.testing import CliRunner
    from checkllm import __version__
    from checkllm.cli import app

    result = CliRunner().invoke(app, ["--version"])
    assert __version__ in result.output

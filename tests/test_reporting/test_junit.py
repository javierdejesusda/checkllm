import xml.etree.ElementTree as ET
from pathlib import Path


from checkllm.models import CheckResult
from checkllm.reporting.junit import generate_junit_xml


class TestJunitXml:
    def test_generates_valid_xml(self, tmp_path: Path):
        results = {
            "test_summarizer": [
                CheckResult(
                    passed=True, score=0.95, reasoning="ok",
                    cost=0.002, latency_ms=450, metric_name="hallucination",
                ),
            ]
        }
        output = tmp_path / "results.xml"
        generate_junit_xml(results, output)
        tree = ET.parse(output)
        root = tree.getroot()
        assert root.tag == "testsuites"

    def test_passing_test_has_no_failure_element(self, tmp_path: Path):
        results = {
            "test_foo": [
                CheckResult(
                    passed=True, score=0.9, reasoning="all good",
                    cost=0.0, latency_ms=100, metric_name="hallucination",
                ),
            ]
        }
        output = tmp_path / "results.xml"
        generate_junit_xml(results, output)
        tree = ET.parse(output)
        testcases = tree.findall(".//testcase")
        assert len(testcases) == 1
        assert testcases[0].find("failure") is None

    def test_failing_test_has_failure_element(self, tmp_path: Path):
        results = {
            "test_foo": [
                CheckResult(
                    passed=False, score=0.3, reasoning="hallucinated",
                    cost=0.002, latency_ms=500, metric_name="hallucination",
                ),
            ]
        }
        output = tmp_path / "results.xml"
        generate_junit_xml(results, output)
        tree = ET.parse(output)
        testcases = tree.findall(".//testcase")
        failure = testcases[0].find("failure")
        assert failure is not None
        assert "hallucinated" in failure.text

    def test_multiple_tests_and_metrics(self, tmp_path: Path):
        results = {
            "test_a": [
                CheckResult(passed=True, score=0.9, reasoning="ok", cost=0.0, latency_ms=100, metric_name="h"),
                CheckResult(passed=False, score=0.3, reasoning="bad", cost=0.0, latency_ms=100, metric_name="r"),
            ],
            "test_b": [
                CheckResult(passed=True, score=0.8, reasoning="ok", cost=0.0, latency_ms=100, metric_name="h"),
            ],
        }
        output = tmp_path / "results.xml"
        generate_junit_xml(results, output)
        tree = ET.parse(output)
        testcases = tree.findall(".//testcase")
        assert len(testcases) == 3

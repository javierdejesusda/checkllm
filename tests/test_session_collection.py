"""Tests for pytest plugin session-wide result collection, snapshot, and report generation."""
import json
import xml.etree.ElementTree as ET



class TestSessionSnapshot:
    def test_snapshot_flag_saves_file(self, pytester, tmp_path):
        """--checkllm-snapshot saves a JSON snapshot after the session."""
        snapshot_path = tmp_path / "snap.json"
        pytester.makepyfile(
            """
            def test_one(check):
                check.contains("hello world", "hello")
                check.regex("abc123", pattern=r"\\d+")

            def test_two(check):
                check.not_contains("hello", "goodbye")
            """
        )
        result = pytester.runpytest_subprocess("-v", f"--checkllm-snapshot={snapshot_path}")
        assert snapshot_path.exists()

        data = json.loads(snapshot_path.read_text())
        assert data["version"] == 1
        assert len(data["tests"]) == 2

    def test_snapshot_records_metrics(self, pytester, tmp_path):
        """Snapshot records metric scores and pass/fail status."""
        snapshot_path = tmp_path / "snap.json"
        pytester.makepyfile(
            """
            def test_scored(check):
                check.contains("hello world", "hello")
            """
        )
        pytester.runpytest_subprocess("-v", f"--checkllm-snapshot={snapshot_path}")
        assert snapshot_path.exists()

        data = json.loads(snapshot_path.read_text())
        tests = data["tests"]
        assert len(tests) == 1
        node_id = list(tests.keys())[0]
        runs = tests[node_id]
        assert len(runs) == 1
        metrics = runs[0]["metrics"]
        assert "contains" in metrics
        assert metrics["contains"]["score"] == 1.0
        assert metrics["contains"]["passed"] is True


class TestSessionHtmlReport:
    def test_report_flag_generates_html(self, pytester, tmp_path):
        """--checkllm-report generates an HTML file."""
        report_path = tmp_path / "report.html"
        pytester.makepyfile(
            """
            def test_for_report(check):
                check.contains("hello world", "hello")
                check.not_contains("hello world", "bye")
            """
        )
        pytester.runpytest_subprocess("-v", f"--checkllm-report={report_path}")
        assert report_path.exists()

        content = report_path.read_text()
        assert "checkllm Report" in content
        assert "contains" in content


class TestSessionJunitReport:
    def test_junit_flag_generates_xml(self, pytester, tmp_path):
        """--checkllm-junit generates valid JUnit XML."""
        junit_path = tmp_path / "results.xml"
        pytester.makepyfile(
            """
            def test_for_junit(check):
                check.contains("hello world", "hello")
            """
        )
        pytester.runpytest_subprocess("-v", f"--checkllm-junit={junit_path}")
        assert junit_path.exists()

        tree = ET.parse(junit_path)
        root = tree.getroot()
        assert root.tag == "testsuites"


class TestNoResultsNoOutput:
    def test_no_snapshot_when_no_checks(self, pytester, tmp_path):
        """If no check fixture is used, no snapshot should be generated."""
        snapshot_path = tmp_path / "snap.json"
        pytester.makepyfile(
            """
            def test_plain():
                assert 1 + 1 == 2
            """
        )
        pytester.runpytest_subprocess("-v", f"--checkllm-snapshot={snapshot_path}")
        assert not snapshot_path.exists()

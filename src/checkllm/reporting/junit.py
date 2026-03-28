from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from checkllm.models import CheckResult


def generate_junit_xml(
    results: dict[str, list[CheckResult]],
    output_path: Path,
) -> None:
    """Generate JUnit XML report from test results.

    Args:
        results: Mapping of test name to list of CheckResults.
        output_path: Path to write the XML file.
    """
    testsuites = ET.Element("testsuites")

    total_tests = 0
    total_failures = 0
    total_time = 0.0

    for test_name, checks in results.items():
        testsuite = ET.SubElement(testsuites, "testsuite", name=test_name)
        suite_failures = 0

        for check in checks:
            total_tests += 1
            time_sec = check.latency_ms / 1000.0
            total_time += time_sec

            testcase = ET.SubElement(
                testsuite,
                "testcase",
                name=f"{test_name}::{check.metric_name}",
                classname=test_name,
                time=f"{time_sec:.3f}",
            )

            if not check.passed:
                total_failures += 1
                suite_failures += 1
                failure = ET.SubElement(
                    testcase,
                    "failure",
                    message=f"{check.metric_name} check failed (score: {check.score:.2f})",
                    type="CheckFailedError",
                )
                failure.text = check.reasoning

        testsuite.set("tests", str(len(checks)))
        testsuite.set("failures", str(suite_failures))

    testsuites.set("tests", str(total_tests))
    testsuites.set("failures", str(total_failures))
    testsuites.set("time", f"{total_time:.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(testsuites)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)

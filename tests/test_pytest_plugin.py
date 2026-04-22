class TestMakeReportHook:
    def test_judge_config_error_causes_skip(self, pytester):
        """JudgeConfigError raised in a test body should not cause a failure.

        The hook converts the failed report to a skipped outcome by setting
        wasxfail, so pytest counts the test as xfailed rather than failed.
        """
        pytester.makepyfile("""
            from checkllm.judge import JudgeConfigError

            def test_raises_judge_config_error(check):
                raise JudgeConfigError("No API key configured")
        """)
        result = pytester.runpytest("-v")
        # The hook sets report.wasxfail so the outcome is counted as xfailed.
        result.assert_outcomes(xfailed=1, failed=0)

    def test_check_failure_fails_test(self, pytester):
        """A deterministic check that fails should cause the test to be marked failed."""
        pytester.makepyfile("""
            def test_failing_check(check):
                check.contains("hello world", "MISSING_TEXT")
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1, passed=0)


class TestCheckFixture:
    def test_check_fixture_is_available(self, pytester):
        """The check fixture should be auto-discovered by pytest."""
        pytester.makepyfile(
            """
            def test_basic(check):
                check.contains("hello world", "hello")
            """
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_check_fixture_collects_failures(self, pytester):
        pytester.makepyfile(
            """
            def test_fails(check):
                check.contains("hello", "goodbye")
            """
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)

    def test_check_fixture_multiple_checks(self, pytester):
        pytester.makepyfile(
            """
            def test_multi(check):
                check.contains("hello world", "hello")
                check.not_contains("hello world", "bye")
                check.regex("abc123", pattern=r"\\d+")
            """
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_check_fixture_reports_all_failures(self, pytester):
        pytester.makepyfile(
            """
            def test_multi_fail(check):
                check.contains("hello", "goodbye")
                check.contains("hello", "missing")
            """
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*2 check*failed*"])


class TestDatasetDecorator:
    def test_dataset_parametrize_yaml(self, pytester):
        dataset_file = pytester.makefile(
            ".yaml",
            cases=(
                '- input: "hello"\n'
                '  query: "greet"\n'
                '  criteria: "friendly"\n'
                '- input: "bye"\n'
                '  query: "farewell"\n'
                '  criteria: "polite"\n'
            ),
        )
        # Use forward slashes to avoid Windows unicode escape issues
        dataset_path = str(dataset_file).replace("\\", "/")
        pytester.makepyfile(
            f"""
            from checkllm.pytest_plugin import dataset

            @dataset("{dataset_path}")
            def test_with_dataset(check, case):
                assert case.input in ("hello", "bye")
                check.contains(case.input, case.input)
            """
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=2)

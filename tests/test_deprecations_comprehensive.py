"""Comprehensive tests for checkllm.deprecations — deprecated decorator."""

from __future__ import annotations

import warnings


from checkllm.deprecations import (
    CheckllmDeprecationWarning,
    CheckllmRemovedIn5Warning,
    CheckllmRemovedIn6Warning,
    deprecated,
)


class TestDeprecationWarningClasses:
    def test_removed_in_5_is_subclass(self):
        assert issubclass(CheckllmRemovedIn5Warning, CheckllmDeprecationWarning)
        assert issubclass(CheckllmRemovedIn5Warning, DeprecationWarning)

    def test_removed_in_6_is_subclass(self):
        assert issubclass(CheckllmRemovedIn6Warning, CheckllmDeprecationWarning)
        assert issubclass(CheckllmRemovedIn6Warning, DeprecationWarning)

    def test_base_is_deprecation_warning(self):
        assert issubclass(CheckllmDeprecationWarning, DeprecationWarning)


class TestDeprecatedDecorator:
    def test_emits_warning_on_call(self):
        @deprecated(reason="Use new_function instead.", removal_version="5.0")
        def old_function():
            return "old"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function()

        assert result == "old"
        assert len(w) == 1
        assert issubclass(w[0].category, CheckllmDeprecationWarning)

    def test_emits_removed_in_5_warning(self):
        @deprecated(reason="Use new.", removal_version="5.0")
        def fn_5():
            return "5"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn_5()

        assert issubclass(w[0].category, CheckllmRemovedIn5Warning)

    def test_emits_removed_in_6_warning(self):
        @deprecated(reason="Use new.", removal_version="6.0")
        def fn_6():
            return "6"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn_6()

        assert issubclass(w[0].category, CheckllmRemovedIn6Warning)

    def test_unknown_version_uses_base_class(self):
        @deprecated(reason="Old stuff.", removal_version="9.0")
        def fn_unknown():
            return "unknown"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn_unknown()

        assert issubclass(w[0].category, CheckllmDeprecationWarning)

    def test_warning_message_contains_qualname(self):
        @deprecated(reason="Use new.", removal_version="5.0")
        def my_old_func():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            my_old_func()

        assert "my_old_func" in str(w[0].message)

    def test_warning_message_contains_removal_version(self):
        @deprecated(reason="Use new.", removal_version="5.0")
        def fn():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn()

        assert "5.0" in str(w[0].message)

    def test_warning_message_contains_reason(self):
        @deprecated(reason="Please use new_api() instead.", removal_version="6.0")
        def fn():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn()

        assert "new_api" in str(w[0].message)

    def test_preserves_function_return_value(self):
        @deprecated(reason="Old.", removal_version="5.0")
        def add(a: int, b: int) -> int:
            return a + b

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = add(3, 4)

        assert result == 7

    def test_preserves_function_name(self):
        @deprecated(reason="Old.", removal_version="5.0")
        def named_function():
            pass

        assert named_function.__name__ == "named_function"

    def test_preserves_function_docstring(self):
        @deprecated(reason="Old.", removal_version="5.0")
        def documented():
            """Original docstring."""
            pass

        assert "Original docstring" in documented.__doc__

    def test_docstring_includes_deprecated_marker(self):
        @deprecated(reason="Use new.", removal_version="6.0")
        def documented():
            """Original docstring."""
            pass

        assert "deprecated" in documented.__doc__.lower()
        assert "6.0" in documented.__doc__

    def test_function_with_no_docstring(self):
        @deprecated(reason="Old.", removal_version="5.0")
        def no_doc():
            pass

        # Should not add doc since original has none
        assert no_doc.__doc__ is None

    def test_passes_args_and_kwargs(self):
        @deprecated(reason="Old.", removal_version="5.0")
        def compute(x, y, multiplier=2):
            return (x + y) * multiplier

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = compute(2, 3, multiplier=4)

        assert result == 20

    def test_multiple_calls_emit_multiple_warnings(self):
        @deprecated(reason="Old.", removal_version="5.0")
        def repeat():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            repeat()
            repeat()
            repeat()

        assert len(w) == 3

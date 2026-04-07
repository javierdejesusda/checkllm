"""Verify deprecation warning infrastructure."""
import warnings


from checkllm.deprecations import (
    CheckllmDeprecationWarning,
    CheckllmRemovedIn5Warning,
    deprecated,
)


class TestDeprecationWarnings:
    def test_warning_hierarchy(self):
        assert issubclass(CheckllmDeprecationWarning, DeprecationWarning)
        assert issubclass(CheckllmRemovedIn5Warning, CheckllmDeprecationWarning)

    def test_deprecated_decorator_warns(self):
        @deprecated("Use new_func() instead", removal_version="5.0")
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            assert result == 42
            assert len(w) == 1
            assert issubclass(w[0].category, CheckllmDeprecationWarning)
            assert "new_func()" in str(w[0].message)
            assert "5.0" in str(w[0].message)

    def test_deprecated_preserves_function_metadata(self):
        @deprecated("Use bar()", removal_version="5.0")
        def foo():
            """Foo docstring."""
            pass

        assert foo.__name__ == "foo"
        assert "Foo docstring" in (foo.__doc__ or "")

    def test_deprecated_class_method(self):
        class MyClass:
            @deprecated("Use new_method()", removal_version="5.0")
            def old_method(self):
                return "old"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = MyClass().old_method()
            assert result == "old"
            assert len(w) == 1

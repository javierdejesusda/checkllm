"""Tests for enhanced dashboard API endpoints."""


def test_dashboard_module_importable():
    """Verify dashboard module imports without error."""
    from checkllm.dashboard import start_dashboard
    assert callable(start_dashboard)


def test_dashboard_has_trends_endpoint():
    """Verify the trends API method exists."""
    from checkllm.dashboard import DashboardHandler
    assert hasattr(DashboardHandler, '_serve_trends')


def test_dashboard_has_cost_breakdown_endpoint():
    """Verify the cost breakdown API method exists."""
    from checkllm.dashboard import DashboardHandler
    assert hasattr(DashboardHandler, '_serve_cost_breakdown')

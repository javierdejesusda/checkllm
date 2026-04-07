"""Tests for pre-run cost estimation."""
from checkllm.estimator import estimate_check_cost, estimate_from_test_file, CostEstimate


def test_deterministic_check_costs_zero():
    est = estimate_check_cost("contains")
    assert est.cost == 0.0
    assert est.is_deterministic is True


def test_judge_check_has_cost():
    est = estimate_check_cost("hallucination", model="gpt-4o")
    assert est.cost > 0.0
    assert est.is_deterministic is False


def test_judge_check_mini_is_cheaper():
    est_full = estimate_check_cost("hallucination", model="gpt-4o")
    est_mini = estimate_check_cost("hallucination", model="gpt-4o-mini")
    assert est_mini.cost < est_full.cost


def test_estimate_from_test_file(tmp_path):
    test_file = tmp_path / "test_example.py"
    test_file.write_text(
        'def test_my_llm(check):\n'
        '    output = "hello"\n'
        '    check.contains(output, "hello")\n'
        '    check.hallucination(output, context="world")\n'
        '    check.toxicity(output)\n'
        '    check.max_tokens(output, limit=100)\n'
    )
    result = estimate_from_test_file(str(test_file), model="gpt-4o")
    assert result.deterministic_count == 2  # contains, max_tokens
    assert result.judge_count == 2  # hallucination, toxicity
    assert result.total_cost > 0.0


def test_estimate_returns_zero_for_no_checks(tmp_path):
    test_file = tmp_path / "test_empty.py"
    test_file.write_text('def test_nothing():\n    pass\n')
    result = estimate_from_test_file(str(test_file), model="gpt-4o")
    assert result.total_cost == 0.0
    assert result.deterministic_count == 0
    assert result.judge_count == 0


def test_cost_estimate_summary():
    est = CostEstimate(
        deterministic_count=5,
        judge_count=3,
        total_cost=0.04,
        model="gpt-4o",
    )
    summary = est.summary()
    assert "5 deterministic" in summary
    assert "3 judge" in summary
    assert "$0.04" in summary

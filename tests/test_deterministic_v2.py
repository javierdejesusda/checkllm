import json

import pytest

from checkllm.deterministic import DeterministicChecks


class TestBleu:
    @pytest.fixture
    def dc(self):
        return DeterministicChecks()

    def test_identical_strings(self, dc):
        result = dc.bleu("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0, abs=0.01)
        assert result.passed is True
        assert result.cost == 0.0
        assert result.metric_name == "bleu"

    def test_completely_different(self, dc):
        result = dc.bleu("apple banana cherry", "dog elephant fox")
        assert result.score == 0.0
        assert result.passed is False
        assert result.cost == 0.0
        assert result.metric_name == "bleu"

    def test_partial_overlap(self, dc):
        result = dc.bleu(
            "the cat sat on the mat",
            "the cat sat on a rug",
            threshold=0.0,
        )
        # There is substantial unigram overlap but divergence in higher n-grams
        assert 0.0 < result.score < 1.0
        assert result.passed is True  # threshold=0.0 so anything > 0 passes
        assert result.cost == 0.0
        assert result.metric_name == "bleu"

    def test_passes_above_threshold(self, dc):
        result = dc.bleu(
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog",
            threshold=0.5,
        )
        assert result.passed is True
        assert result.score >= 0.5
        assert result.cost == 0.0

    def test_fails_below_threshold(self, dc):
        result = dc.bleu(
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            threshold=0.8,
        )
        assert result.passed is False
        assert result.score < 0.8
        assert result.cost == 0.0

    def test_empty_output(self, dc):
        result = dc.bleu("", "some reference text")
        assert result.passed is False
        assert result.score == 0.0
        assert result.cost == 0.0


class TestRougeL:
    @pytest.fixture
    def dc(self):
        return DeterministicChecks()

    def test_identical_strings(self, dc):
        result = dc.rouge_l("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0, abs=0.01)
        assert result.passed is True
        assert result.cost == 0.0
        assert result.metric_name == "rouge_l"

    def test_completely_different(self, dc):
        result = dc.rouge_l("apple banana cherry", "dog elephant fox")
        assert result.score == 0.0
        assert result.passed is False
        assert result.cost == 0.0
        assert result.metric_name == "rouge_l"

    def test_partial_overlap(self, dc):
        result = dc.rouge_l(
            "the cat sat on the mat",
            "the cat is on a mat",
            threshold=0.0,
        )
        # Shared subsequence: "the", "cat", "on", "mat" -> LCS = 4
        # precision = 4/6, recall = 4/6, F1 ~ 0.667
        assert 0.0 < result.score < 1.0
        assert result.passed is True
        assert result.cost == 0.0

    def test_passes_above_threshold(self, dc):
        result = dc.rouge_l(
            "the quick brown fox jumps",
            "the quick brown fox jumps over the lazy dog",
            threshold=0.5,
        )
        assert result.passed is True
        assert result.score >= 0.5
        assert result.cost == 0.0

    def test_fails_below_threshold(self, dc):
        result = dc.rouge_l(
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            threshold=0.8,
        )
        assert result.passed is False
        assert result.score < 0.8
        assert result.cost == 0.0

    def test_empty_output(self, dc):
        result = dc.rouge_l("", "some reference text")
        assert result.passed is False
        assert result.score == 0.0
        assert result.cost == 0.0


class TestJsonField:
    @pytest.fixture
    def dc(self):
        return DeterministicChecks()

    @pytest.fixture
    def sample_json(self):
        return json.dumps({
            "name": "Alice",
            "age": 30,
            "data": {"name": "nested_value", "count": 10},
            "items": [
                {"name": "first", "price": 5},
                {"name": "second", "price": 15},
            ],
            "tags": ["python", "testing"],
            "empty_str": "",
            "is_active": True,
        })

    def test_field_exists(self, dc, sample_json):
        result = dc.json_field(sample_json, "name", condition="exists")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "json_field"

    def test_field_equals_value(self, dc, sample_json):
        result = dc.json_field(sample_json, "name", expected="Alice")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "json_field"

    def test_field_equals_wrong_value(self, dc, sample_json):
        result = dc.json_field(sample_json, "name", expected="Bob")
        assert result.passed is False
        assert result.score == 0.0

    def test_nested_field(self, dc, sample_json):
        result = dc.json_field(sample_json, "data.name", expected="nested_value")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0

    def test_array_index(self, dc, sample_json):
        result = dc.json_field(sample_json, "items.0.name", expected="first")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0

    def test_array_index_second_item(self, dc, sample_json):
        result = dc.json_field(sample_json, "items.1.price", expected=15)
        assert result.passed is True
        assert result.score == 1.0

    def test_condition_not_empty(self, dc, sample_json):
        # Non-empty string
        result = dc.json_field(sample_json, "name", condition="not_empty")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0

        # Empty string
        result_empty = dc.json_field(sample_json, "empty_str", condition="not_empty")
        assert result_empty.passed is False
        assert result_empty.score == 0.0

    def test_condition_gt(self, dc, sample_json):
        result = dc.json_field(sample_json, "age", condition="gt:5")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "json_field"

        # Fails when not greater
        result_fail = dc.json_field(sample_json, "age", condition="gt:50")
        assert result_fail.passed is False
        assert result_fail.score == 0.0

    def test_condition_lt(self, dc, sample_json):
        result = dc.json_field(sample_json, "age", condition="lt:100")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "json_field"

        # Fails when not less
        result_fail = dc.json_field(sample_json, "age", condition="lt:10")
        assert result_fail.passed is False
        assert result_fail.score == 0.0

    def test_condition_contains(self, dc, sample_json):
        result = dc.json_field(sample_json, "name", condition="contains:Ali")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "json_field"

        # Fails when substring not found
        result_fail = dc.json_field(sample_json, "name", condition="contains:foo")
        assert result_fail.passed is False
        assert result_fail.score == 0.0

    def test_condition_type_str(self, dc, sample_json):
        result = dc.json_field(sample_json, "name", condition="type:str")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "json_field"

        # Int field should fail type:str check
        result_fail = dc.json_field(sample_json, "age", condition="type:str")
        assert result_fail.passed is False
        assert result_fail.score == 0.0

    def test_condition_type_int(self, dc, sample_json):
        result = dc.json_field(sample_json, "age", condition="type:int")
        assert result.passed is True
        assert result.score == 1.0

    def test_condition_type_bool(self, dc, sample_json):
        result = dc.json_field(sample_json, "is_active", condition="type:bool")
        assert result.passed is True
        assert result.score == 1.0

    def test_condition_type_list(self, dc, sample_json):
        result = dc.json_field(sample_json, "tags", condition="type:list")
        assert result.passed is True
        assert result.score == 1.0

    def test_invalid_json(self, dc):
        result = dc.json_field("not json at all {{{", "name")
        assert result.passed is False
        assert result.score == 0.0
        assert result.cost == 0.0
        assert "Invalid JSON" in result.reasoning

    def test_field_not_found(self, dc, sample_json):
        result = dc.json_field(sample_json, "nonexistent")
        assert result.passed is False
        assert result.score == 0.0
        assert "not found" in result.reasoning

    def test_field_exists_no_condition_no_expected(self, dc, sample_json):
        # When no condition and no expected, just checks field exists
        result = dc.json_field(sample_json, "name")
        assert result.passed is True
        assert result.score == 1.0
        assert "exists" in result.reasoning

    def test_nested_field_not_found(self, dc, sample_json):
        result = dc.json_field(sample_json, "data.missing_key")
        assert result.passed is False
        assert result.score == 0.0
        assert "not found" in result.reasoning


class TestIsValidSql:
    @pytest.fixture
    def dc(self):
        return DeterministicChecks()

    def test_valid_select(self, dc):
        result = dc.is_valid_sql("SELECT * FROM users WHERE id = 1")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "is_valid_sql"

    def test_valid_insert(self, dc):
        result = dc.is_valid_sql("INSERT INTO users (name, age) VALUES ('Alice', 30)")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "is_valid_sql"

    def test_valid_with_subquery(self, dc):
        result = dc.is_valid_sql(
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)"
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_invalid_sql(self, dc):
        result = dc.is_valid_sql("THIS IS NOT SQL AT ALL")
        assert result.passed is False
        assert result.cost == 0.0
        assert result.metric_name == "is_valid_sql"
        assert "does not start with a recognized SQL keyword" in result.reasoning

    def test_strips_markdown_fences(self, dc):
        sql_with_fences = "```sql\nSELECT * FROM users\n```"
        result = dc.is_valid_sql(sql_with_fences)
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0

    def test_strips_generic_fences(self, dc):
        sql_with_fences = "```\nSELECT id FROM products\n```"
        result = dc.is_valid_sql(sql_with_fences)
        assert result.passed is True
        assert result.score == 1.0

    def test_unbalanced_parentheses(self, dc):
        result = dc.is_valid_sql("SELECT * FROM users WHERE id IN (1, 2, 3")
        assert result.passed is False
        assert "Unbalanced parentheses" in result.reasoning

    def test_unclosed_string_literal(self, dc):
        result = dc.is_valid_sql("SELECT * FROM users WHERE name = 'Alice")
        assert result.passed is False
        assert "Unclosed" in result.reasoning

    def test_empty_sql(self, dc):
        result = dc.is_valid_sql("")
        assert result.passed is False
        assert result.score == 0.0
        assert "Empty SQL" in result.reasoning

    def test_valid_create_table(self, dc):
        result = dc.is_valid_sql(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100))"
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_valid_update(self, dc):
        result = dc.is_valid_sql("UPDATE users SET name = 'Bob' WHERE id = 1")
        assert result.passed is True
        assert result.score == 1.0

    def test_valid_delete(self, dc):
        result = dc.is_valid_sql("DELETE FROM users WHERE id = 1")
        assert result.passed is True
        assert result.score == 1.0


class TestIsValidMarkdown:
    @pytest.fixture
    def dc(self):
        return DeterministicChecks()

    def test_valid_basic(self, dc):
        result = dc.is_valid_markdown("This is some plain text content.")
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "is_valid_markdown"

    def test_require_headers_present(self, dc):
        md = "# Title\n\nSome text here.\n\n## Subtitle\n\nMore text."
        result = dc.is_valid_markdown(md, require_headers=True)
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "is_valid_markdown"

    def test_require_headers_missing(self, dc):
        md = "This is plain text with no headers at all."
        result = dc.is_valid_markdown(md, require_headers=True)
        assert result.passed is False
        assert result.score < 1.0
        assert result.cost == 0.0
        assert "No headers found" in result.reasoning

    def test_require_lists_present(self, dc):
        md = "# Shopping List\n\n- Apples\n- Bananas\n- Oranges"
        result = dc.is_valid_markdown(md, require_lists=True)
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "is_valid_markdown"

    def test_require_lists_missing(self, dc):
        md = "# Title\n\nJust some text, no lists."
        result = dc.is_valid_markdown(md, require_lists=True)
        assert result.passed is False
        assert result.score < 1.0
        assert "No lists found" in result.reasoning

    def test_require_code_blocks(self, dc):
        md = "# Example\n\n```python\nprint('hello')\n```"
        result = dc.is_valid_markdown(md, require_code_blocks=True)
        assert result.passed is True
        assert result.score == 1.0
        assert result.cost == 0.0
        assert result.metric_name == "is_valid_markdown"

    def test_require_code_blocks_missing(self, dc):
        md = "# Title\n\nNo code blocks here."
        result = dc.is_valid_markdown(md, require_code_blocks=True)
        assert result.passed is False
        assert result.score < 1.0
        assert "No code blocks found" in result.reasoning

    def test_empty_output(self, dc):
        result = dc.is_valid_markdown("")
        assert result.passed is False
        assert result.score == 0.0
        assert "Empty output" in result.reasoning

    def test_ordered_list(self, dc):
        md = "# Steps\n\n1. First step\n2. Second step\n3. Third step"
        result = dc.is_valid_markdown(md, require_lists=True)
        assert result.passed is True
        assert result.score == 1.0

    def test_all_requirements(self, dc):
        md = (
            "# Title\n\n"
            "Some intro text.\n\n"
            "- Item one\n"
            "- Item two\n\n"
            "```python\nprint('hello')\n```\n"
        )
        result = dc.is_valid_markdown(
            md, require_headers=True, require_lists=True, require_code_blocks=True
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_unclosed_code_block(self, dc):
        md = "# Title\n\n```python\nprint('hello')\n"
        result = dc.is_valid_markdown(md, require_code_blocks=True)
        assert result.passed is False
        assert "Unclosed code block" in result.reasoning

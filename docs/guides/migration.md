# Migration Guide

Migrate your existing LLM evaluation suite from DeepEval, Ragas, or promptfoo to
checkllm.

---

## From DeepEval

### Metric equivalence

| DeepEval | checkllm |
|----------|----------|
| `AnswerRelevancyMetric` | `answer_relevance` |
| `FaithfulnessMetric` | `hallucination` |
| `ContextualPrecisionMetric` | `context_precision` |
| `ContextualRecallMetric` | `context_recall` |
| `ContextualRelevancyMetric` | `context_relevance` |
| `ToxicityMetric` | `toxicity` |
| `BiasMetric` | `bias` |
| `GEval` | `geval` |
| `HallucinationMetric` | `hallucination` |
| `SummarizationMetric` | `summarization` |

### Code translation

=== "DeepEval"

    ```python
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    def test_rag():
        case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            retrieval_context=["France's capital city is Paris."],
        )
        assert_test(case, [
            AnswerRelevancyMetric(threshold=0.8),
            FaithfulnessMetric(threshold=0.8),
        ])
    ```

=== "checkllm"

    ```python
    import pytest
    from checkllm import llm_check

    @llm_check(metrics=["answer_relevance", "hallucination"], threshold=0.8)
    def test_rag():
        return {
            "input": "What is the capital of France?",
            "output": "Paris is the capital of France.",
            "context": ["France's capital city is Paris."],
        }
    ```

### Judge configuration

=== "DeepEval"

    ```python
    from deepeval.models import GPTModel
    model = GPTModel(model="gpt-4o")
    ```

=== "checkllm"

    ```python
    from checkllm import OpenAIJudge
    judge = OpenAIJudge(model="gpt-4o")
    ```

---

## From Ragas

### Metric equivalence

| Ragas | checkllm |
|-------|----------|
| `faithfulness` | `hallucination` |
| `answer_relevancy` | `answer_relevance` |
| `context_precision` | `context_precision` |
| `context_recall` | `context_recall` |
| `context_entity_recall` | `context_relevance` |
| `answer_correctness` | `answer_correctness` |
| `answer_similarity` | `semantic_similarity` |

### Code translation

=== "Ragas"

    ```python
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    dataset = Dataset.from_dict({
        "question": ["What is the capital of France?"],
        "answer": ["Paris is the capital of France."],
        "contexts": [["France's capital city is Paris."]],
        "ground_truth": ["Paris"],
    })

    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(result)
    ```

=== "checkllm (pytest)"

    ```python
    import pytest

    @pytest.mark.parametrize("case", [
        {
            "input": "What is the capital of France?",
            "output": "Paris is the capital of France.",
            "context": ["France's capital city is Paris."],
            "expected": "Paris",
        }
    ])
    def test_rag(case, llm_judge):
        llm_judge(
            input=case["input"],
            output=case["output"],
            context=case["context"],
            expected_output=case["expected"],
            metrics=["hallucination", "answer_relevance"],
            threshold=0.8,
        )
    ```

=== "checkllm (YAML)"

    ```yaml
    # tests/rag.yaml
    cases:
      - id: capital-france
        input: "What is the capital of France?"
        output: "Paris is the capital of France."
        context:
          - "France's capital city is Paris."
        expected_output: "Paris"
        assertions:
          - type: hallucination
            threshold: 0.8
          - type: answer_relevance
            threshold: 0.8
    ```

    Run: `checkllm yaml-run tests/rag.yaml`

---

## From promptfoo

### Config translation

=== "promptfoo"

    ```yaml
    # promptfooconfig.yaml
    prompts:
      - "Summarize: {{text}}"

    providers:
      - openai:gpt-4o

    tests:
      - vars:
          text: "The Eiffel Tower is in Paris, France."
        assert:
          - type: contains
            value: "Paris"
          - type: llm-rubric
            value: "Summary is accurate and concise"
    ```

=== "checkllm"

    ```yaml
    # tests/summarize.yaml
    cases:
      - id: summarize-eiffel
        input: "Summarize: The Eiffel Tower is in Paris, France."
        output: "{{model_output}}"
        assertions:
          - type: contains
            value: "Paris"
          - type: llm_rubric
            value: "Summary is accurate and concise"
            threshold: 0.8
    ```

### Assertion type mapping

| promptfoo | checkllm |
|-----------|----------|
| `contains` | `contains` |
| `icontains` | `icontains` |
| `regex` | `regex` |
| `not-contains` | `not_contains` |
| `llm-rubric` | `llm_rubric` |
| `model-graded-closedqa` | `answer_relevance` |
| `similar` | `semantic_similarity` |
| `javascript` | `python` (custom metric) |

### Provider configuration

=== "promptfoo"

    ```yaml
    providers:
      - openai:gpt-4o
      - anthropic:claude-3-5-sonnet-20241022
    ```

=== "checkllm"

    ```toml
    # pyproject.toml
    [tool.checkllm]
    judge_model = "gpt-4o"
    # Switch to Anthropic:
    # judge_model = "claude-3-5-sonnet-20241022"
    # judge_backend = "anthropic"
    ```

---

## General Migration Tips

1. **Start with free deterministic checks first** — `contains`, `regex`, `max_tokens`,
   `no_pii` have zero cost and run instantly.
2. **Estimate costs before enabling LLM metrics:** `checkllm estimate tests/`
3. **Baseline before switching:** `checkllm snapshot --save` records current scores.
4. **Run both tools in parallel** for one sprint to compare results before cutting over.
5. **Use the adapter layer** to import existing DeepEval/Ragas suites directly
   (see [Plugins Guide](./plugins.md)).

"""Comprehensive tests for checkllm.check_judge - all JudgeChecksMixin methods."""

from __future__ import annotations

import asyncio

from checkllm.check import CheckCollector
from checkllm.config import CheckllmConfig
from checkllm.conversation import ConversationalTestCase, Turn
from checkllm.models import CheckResult
from checkllm.testing import MockJudge


def _c(score=0.9):
    config = CheckllmConfig(cache_enabled=False)
    judge = MockJudge(default_score=score)
    return CheckCollector(config=config, judge=judge)


def _conv():
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What is Python?"),
            Turn(role="assistant", content="Python is a programming language."),
        ]
    )


class TestHallucination:
    def test_basic(self):
        result = _c().hallucination(
            "The sky is blue.", "The sky is blue because of Rayleigh scattering."
        )
        assert isinstance(result, CheckResult)
        assert result.metric_name == "hallucination"
        assert result.passed is True

    def test_with_threshold(self):
        result = _c(0.5).hallucination("output", "context", threshold=0.3)
        assert result.passed is True

    def test_failing_score(self):
        result = _c(score=0.1).hallucination("output", "context", threshold=0.8)
        assert result.passed is False

    def test_with_system_prompt(self):
        result = _c().hallucination("output", "context", system_prompt="Be strict.")
        assert result.metric_name == "hallucination"


class TestRelevance:
    def test_basic(self):
        result = _c().relevance("Python is great for data science.", "What is Python good for?")
        assert result.metric_name == "relevance"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().relevance("output", "query", system_prompt="custom")
        assert result.metric_name == "relevance"


class TestToxicity:
    def test_basic(self):
        result = _c().toxicity("This is a helpful response.")
        assert result.metric_name == "toxicity"
        assert result.passed is True

    def test_with_threshold(self):
        result = _c().toxicity("output", threshold=0.5)
        assert result.metric_name == "toxicity"

    def test_with_system_prompt(self):
        result = _c().toxicity("output", system_prompt="Check carefully.")
        assert result.metric_name == "toxicity"


class TestRubric:
    def test_basic(self):
        result = _c().rubric(
            "The answer covers all required topics.", "Must address the main question."
        )
        assert result.metric_name == "rubric"
        assert result.passed is True

    def test_with_threshold(self):
        result = _c().rubric("output", "criteria", threshold=0.7)
        assert result.metric_name == "rubric"

    def test_with_system_prompt(self):
        result = _c().rubric("output", "criteria", system_prompt="custom")
        assert result.metric_name == "rubric"


class TestFluency:
    def test_basic(self):
        result = _c().fluency("This is a well-written response with clear prose.")
        assert result.metric_name == "fluency"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().fluency("output", system_prompt="custom")
        assert result.metric_name == "fluency"


class TestCoherence:
    def test_basic(self):
        result = _c().coherence("The argument flows logically from premise to conclusion.")
        assert result.metric_name == "coherence"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().coherence("output", system_prompt="custom")
        assert result.metric_name == "coherence"


class TestSentiment:
    def test_basic(self):
        result = _c().sentiment("I love this product!")
        assert result.metric_name == "sentiment"

    def test_with_system_prompt(self):
        result = _c().sentiment("output", system_prompt="custom")
        assert result.metric_name == "sentiment"


class TestCorrectness:
    def test_basic(self):
        result = _c().correctness("Paris is the capital of France.", "Paris")
        assert result.metric_name == "correctness"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().correctness("output", "expected", system_prompt="custom")
        assert result.metric_name == "correctness"


class TestFaithfulness:
    def test_basic(self):
        result = _c().faithfulness("The document says X.", "The document says X.")
        assert result.metric_name == "faithfulness"
        assert result.passed is True

    def test_with_query(self):
        result = _c().faithfulness("output", "context", query="What does it say?")
        assert result.metric_name == "faithfulness"

    def test_without_query(self):
        result = _c().faithfulness("output", "context")
        assert result.metric_name == "faithfulness"

    def test_with_system_prompt(self):
        result = _c().faithfulness("output", "context", system_prompt="custom")
        assert result.metric_name == "faithfulness"


class TestContextRelevance:
    def test_basic(self):
        result = _c().context_relevance("Relevant context text.", "What is the context about?")
        assert result.metric_name == "context_relevance"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().context_relevance("context", "query", system_prompt="custom")
        assert result.metric_name == "context_relevance"


class TestAnswerCompleteness:
    def test_basic(self):
        result = _c().answer_completeness(
            "A complete answer addressing all parts.", "What are the key points?"
        )
        assert result.metric_name == "answer_completeness"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().answer_completeness("output", "query", system_prompt="custom")
        assert result.metric_name == "answer_completeness"


class TestInstructionFollowing:
    def test_basic(self):
        result = _c().instruction_following("Done as instructed.", "Respond in bullet points.")
        assert result.metric_name == "instruction_following"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().instruction_following("output", "instructions", system_prompt="custom")
        assert result.metric_name == "instruction_following"


class TestSummarization:
    def test_basic(self):
        result = _c().summarization(
            "A brief summary.", "A very long document with lots of content."
        )
        assert result.metric_name == "summarization"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().summarization("output", "source", system_prompt="custom")
        assert result.metric_name == "summarization"


class TestBias:
    def test_basic(self):
        result = _c().bias("A neutral and balanced response.")
        assert result.metric_name == "bias"
        assert result.passed is True

    def test_with_categories(self):
        result = _c().bias("output", categories=["gender", "race"])
        assert result.metric_name == "bias"

    def test_without_categories(self):
        result = _c().bias("output", categories=None)
        assert result.metric_name == "bias"

    def test_with_system_prompt(self):
        result = _c().bias("output", system_prompt="custom")
        assert result.metric_name == "bias"


class TestConsistency:
    def test_basic(self):
        result = _c().consistency(["Paris is the capital.", "The capital is Paris."])
        assert result.metric_name == "consistency"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().consistency(["out1", "out2"], system_prompt="custom")
        assert result.metric_name == "consistency"


class TestGroundedness:
    def test_basic(self):
        result = _c().groundedness(
            "The paper states X.", ["Source A: X is true.", "Source B: X confirmed."]
        )
        assert result.metric_name == "groundedness"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().groundedness("output", ["src1", "src2"], system_prompt="custom")
        assert result.metric_name == "groundedness"


class TestGEval:
    def test_basic(self):
        result = _c().g_eval("An excellent response.", "Rate the quality.")
        assert result.metric_name == "g_eval"
        assert result.passed is True

    def test_with_steps(self):
        result = _c().g_eval("output", "criteria", steps=["step1", "step2"])
        assert result.metric_name == "g_eval"

    def test_without_steps(self):
        result = _c().g_eval("output", "criteria", steps=None)
        assert result.metric_name == "g_eval"

    def test_with_system_prompt(self):
        result = _c().g_eval("output", "criteria", system_prompt="custom")
        assert result.metric_name == "g_eval"


class TestContextualPrecision:
    def test_basic(self):
        result = _c().contextual_precision(
            "The answer is 42.",
            ["Context A: The answer is 42.", "Context B: Irrelevant information."],
            "What is the answer?",
            "42",
        )
        assert result.metric_name == "contextual_precision"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().contextual_precision(
            "out", ["ctx"], "query", "expected", system_prompt="custom"
        )
        assert result.metric_name == "contextual_precision"


class TestContextualRecall:
    def test_basic(self):
        result = _c().contextual_recall(
            "The answer mentions all key points.",
            ["Context: key point 1", "Context: key point 2"],
            "Expected comprehensive answer.",
        )
        assert result.metric_name == "contextual_recall"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().contextual_recall("out", ["ctx"], "expected", system_prompt="custom")
        assert result.metric_name == "contextual_recall"


class TestTaskCompletion:
    def test_basic(self):
        result = _c().task_completion(
            "Task completed successfully.", "Write a hello world program."
        )
        assert result.metric_name == "task_completion"
        assert result.passed is True

    def test_with_constraints(self):
        result = _c().task_completion(
            "output", "task", constraints=["Must be in Python", "No imports"]
        )
        assert result.metric_name == "task_completion"

    def test_without_constraints(self):
        result = _c().task_completion("output", "task", constraints=None)
        assert result.metric_name == "task_completion"

    def test_with_system_prompt(self):
        result = _c().task_completion("output", "task", system_prompt="custom")
        assert result.metric_name == "task_completion"


class TestRoleAdherence:
    def test_basic(self):
        result = _c().role_adherence("I am your helpful assistant.", "A customer service agent.")
        assert result.metric_name == "role_adherence"
        assert result.passed is True

    def test_with_query(self):
        result = _c().role_adherence("output", "role", query="How can I help?")
        assert result.metric_name == "role_adherence"

    def test_without_query(self):
        result = _c().role_adherence("output", "role", query=None)
        assert result.metric_name == "role_adherence"

    def test_with_system_prompt(self):
        result = _c().role_adherence("output", "role", system_prompt="custom")
        assert result.metric_name == "role_adherence"


class TestToolAccuracy:
    def test_basic(self):
        expected_tools = [{"name": "search"}]
        result = _c().tool_accuracy("weather lookup output", expected_tools, "What is the weather?")
        assert result.metric_name == "tool_accuracy"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().tool_accuracy("output", [{"name": "tool"}], "query", system_prompt="custom")
        assert result.metric_name == "tool_accuracy"


class TestKnowledgeRetention:
    def test_basic(self):
        conv = _conv()
        result = _c().knowledge_retention(conv)
        assert result.metric_name == "knowledge_retention"
        assert result.passed is True

    def test_with_system_prompt(self):
        conv = _conv()
        result = _c().knowledge_retention(conv, system_prompt="custom")
        assert result.metric_name == "knowledge_retention"


class TestConversationCompleteness:
    def test_basic(self):
        conv = _conv()
        result = _c().conversation_completeness(conv)
        assert result.metric_name == "conversation_completeness"
        assert result.passed is True

    def test_with_system_prompt(self):
        conv = _conv()
        result = _c().conversation_completeness(conv, system_prompt="custom")
        assert result.metric_name == "conversation_completeness"


class TestPlanQuality:
    def test_basic(self):
        result = _c().plan_quality("1. Define. 2. Implement. 3. Test.", "Build a web app.")
        assert result.metric_name == "plan_quality"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().plan_quality("plan", "task", system_prompt="custom")
        assert result.metric_name == "plan_quality"


class TestGoalAccuracy:
    def test_basic(self):
        result = _c().goal_accuracy("The goal was achieved.", "Achieve a passing score.")
        assert result.metric_name == "goal_accuracy"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().goal_accuracy("output", "goal", system_prompt="custom")
        assert result.metric_name == "goal_accuracy"


class TestStepEfficiency:
    def test_basic(self):
        result = _c().step_efficiency(["Step 1", "Step 2", "Step 3"], "Complete the task.")
        assert result.metric_name == "step_efficiency"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().step_efficiency(["step1"], "task", system_prompt="custom")
        assert result.metric_name == "step_efficiency"


class TestArgumentCorrectness:
    def test_basic(self):
        result = _c().argument_correctness("search(q=test)", "search(q=test)")
        assert result.metric_name == "argument_correctness"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().argument_correctness("tool_calls", "expected_calls", system_prompt="custom")
        assert result.metric_name == "argument_correctness"


class TestPlanAdherence:
    def test_basic(self):
        result = _c().plan_adherence("1. Research. 2. Write.", "Researched topic, wrote report.")
        assert result.metric_name == "plan_adherence"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().plan_adherence("plan", "trace", system_prompt="custom")
        assert result.metric_name == "plan_adherence"


class TestPIIDetection:
    def test_basic(self):
        result = _c().pii_detection("The weather today is sunny.")
        assert result.metric_name == "pii_detection"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().pii_detection("output", system_prompt="custom")
        assert result.metric_name == "pii_detection"


class TestMisuseDetection:
    def test_basic(self):
        result = _c().misuse_detection("Here is how to cook pasta.", "culinary assistance")
        assert result.metric_name == "misuse_detection"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().misuse_detection("output", "scope", system_prompt="custom")
        assert result.metric_name == "misuse_detection"


class TestRoleViolation:
    def test_basic(self):
        result = _c().role_violation(
            "I will help you as a customer service agent.", "customer service agent"
        )
        assert result.metric_name == "role_violation"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().role_violation("output", "role", system_prompt="custom")
        assert result.metric_name == "role_violation"


class TestImageCoherence:
    def test_basic(self):
        result = _c().image_coherence(
            "A photo of a sunset over the ocean.",
            "Article about tropical destinations.",
        )
        assert result.metric_name == "image_coherence"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().image_coherence("img desc", "text ctx", system_prompt="custom")
        assert result.metric_name == "image_coherence"


class TestImageHelpfulness:
    def test_basic(self):
        result = _c().image_helpfulness(
            "A diagram showing the architecture.", "Explain the system design."
        )
        assert result.metric_name == "image_helpfulness"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().image_helpfulness("img desc", "query", system_prompt="custom")
        assert result.metric_name == "image_helpfulness"


class TestImageRelevance:
    def test_basic(self):
        result = _c().image_relevance("An image of a Python logo.", "Python programming language")
        assert result.metric_name == "image_relevance"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().image_relevance("img desc", "query", system_prompt="custom")
        assert result.metric_name == "image_relevance"


class TestTextToImage:
    def test_basic(self):
        result = _c().text_to_image("A bright red apple on a white background.", "red apple")
        assert result.metric_name == "text_to_image"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().text_to_image("img desc", "original prompt", system_prompt="custom")
        assert result.metric_name == "text_to_image"


class TestFactualCorrectness:
    def test_basic(self):
        result = _c().factual_correctness(
            "Water is H2O.", "Water is composed of hydrogen and oxygen."
        )
        assert result.metric_name == "factual_correctness"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().factual_correctness("output", "reference", system_prompt="custom")
        assert result.metric_name == "factual_correctness"


class TestContextEntityRecall:
    def test_basic(self):
        result = _c().context_entity_recall(
            "The Eiffel Tower is in Paris.", "Paris, Eiffel Tower, France"
        )
        assert result.metric_name == "context_entity_recall"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().context_entity_recall("context", "reference", system_prompt="custom")
        assert result.metric_name == "context_entity_recall"


class TestTopicAdherence:
    def test_basic(self):
        result = _c().topic_adherence(
            "Python is great for machine learning.", ["programming", "machine learning"]
        )
        assert result.metric_name == "topic_adherence"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().topic_adherence("output", ["topic1"], system_prompt="custom")
        assert result.metric_name == "topic_adherence"


class TestResponseCompleteness:
    def test_basic(self):
        result = _c().response_completeness(
            "This answer covers all aspects of the question.", "What is Python?"
        )
        assert result.metric_name == "response_completeness"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().response_completeness("output", "query", system_prompt="custom")
        assert result.metric_name == "response_completeness"


class TestCodeCorrectness:
    def test_basic(self):
        result = _c().code_correctness("def add(a, b): return a + b", "Write an addition function.")
        assert result.metric_name == "code_correctness"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().code_correctness("output", "requirements", system_prompt="custom")
        assert result.metric_name == "code_correctness"


class TestNonAdvice:
    def test_basic(self):
        result = _c().non_advice("You should consult a doctor.")
        assert isinstance(result, CheckResult)
        assert result.metric_name == "non_advice"
        assert result.passed is True

    def test_with_restricted_domains(self):
        result = _c().non_advice("output", restricted_domains=["medical", "legal"])
        assert result.metric_name == "non_advice"
        assert result.passed is True

    def test_without_restricted_domains(self):
        result = _c().non_advice("Here are some general tips.")
        assert result.metric_name == "non_advice"

    def test_with_system_prompt(self):
        result = _c().non_advice("output", system_prompt="custom")
        assert result.metric_name == "non_advice"


class TestMultimodalFaithfulness:
    def test_basic(self):
        result = _c().multimodal_faithfulness(
            "A red car on the street",
            "The image shows a red car.",
            "Source material about red cars.",
        )
        assert result.metric_name == "multimodal_faithfulness"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().multimodal_faithfulness(
            "image desc", "text output", "source context", system_prompt="custom"
        )
        assert result.metric_name == "multimodal_faithfulness"


class TestMCPTaskCompletion:
    def test_basic(self):
        result = _c().mcp_task_completion(
            "Task completed successfully.",
            "Fetch user profile",
            ["get_user", "format_profile"],
        )
        assert result.metric_name == "mcp_task_completion"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().mcp_task_completion("output", "task", ["tool1"], system_prompt="custom")
        assert result.metric_name == "mcp_task_completion"


class TestMCPUse:
    def test_basic(self):
        result = _c().mcp_use(
            "Used search_web to find the answer.",
            ["search_web", "read_file", "write_file"],
            ["search_web"],
            "What is the weather today?",
        )
        assert result.metric_name == "mcp_use"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().mcp_use("output", ["t1"], ["t1"], "query", system_prompt="custom")
        assert result.metric_name == "mcp_use"


class TestMultiTurnMCPUse:
    def test_basic(self):
        result = _c().multi_turn_mcp_use(
            "Turn 1: used search. Turn 2: used write.",
            ["search", "write", "read"],
            ["search", "write"],
        )
        assert result.metric_name == "multi_turn_mcp_use"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().multi_turn_mcp_use("trace", ["t1"], ["t1"], system_prompt="custom")
        assert result.metric_name == "multi_turn_mcp_use"


class TestNoiseSensitivity:
    def test_basic(self):
        result = _c().noise_sensitivity(
            "The capital of France is Paris.",
            "France is in Europe. Paris is the capital.",
            "France is in Europe. Paris is the capital. The Eiffel Tower is 330m tall.",
        )
        assert result.metric_name == "noise_sensitivity"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().noise_sensitivity(
            "output", "context", "noisy_context", system_prompt="custom"
        )
        assert result.metric_name == "noise_sensitivity"


class TestSQLEquivalence:
    def test_basic(self):
        result = _c().sql_equivalence(
            "SELECT id, name FROM users WHERE age > 18",
            "SELECT id, name FROM users WHERE age >= 19",
        )
        assert result.metric_name == "sql_equivalence"
        assert result.passed is True

    def test_with_schema(self):
        result = _c().sql_equivalence(
            "SELECT * FROM orders",
            "SELECT id, product FROM orders",
            schema="CREATE TABLE orders (id INT, product TEXT)",
        )
        assert result.metric_name == "sql_equivalence"

    def test_without_schema(self):
        result = _c().sql_equivalence(
            "SELECT count(*) FROM t",
            "SELECT COUNT(*) FROM t",
        )
        assert result.metric_name == "sql_equivalence"

    def test_with_system_prompt(self):
        result = _c().sql_equivalence("q1", "q2", system_prompt="custom")
        assert result.metric_name == "sql_equivalence"


class TestCitationAccuracy:
    def test_basic(self):
        result = _c().citation_accuracy(
            "According to [1], the sky is blue.",
            ["[1] The sky appears blue due to Rayleigh scattering."],
        )
        assert result.metric_name == "citation_accuracy"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().citation_accuracy("output", ["source1"], system_prompt="custom")
        assert result.metric_name == "citation_accuracy"


class TestInstructionCompleteness:
    def test_basic(self):
        result = _c().instruction_completeness(
            "First, do A. Then, do B. Finally, do C.",
            ["Do step A", "Do step B", "Do step C"],
        )
        assert result.metric_name == "instruction_completeness"
        assert result.passed is True

    def test_with_system_prompt(self):
        result = _c().instruction_completeness("output", ["inst1", "inst2"], system_prompt="custom")
        assert result.metric_name == "instruction_completeness"


class TestComparativeQuality:
    def test_basic(self):
        result = _c().comparative_quality(
            "Answer A is comprehensive and well-structured.",
            "Answer B is brief.",
            "Which answer is more detailed?",
        )
        assert result.metric_name == "comparative_quality"
        assert result.passed is True

    def test_with_custom_threshold(self):
        result = _c(0.8).comparative_quality("a", "b", "criteria", threshold=0.7)
        assert result.metric_name == "comparative_quality"

    def test_with_system_prompt(self):
        result = _c().comparative_quality("a", "b", "criteria", system_prompt="custom")
        assert result.metric_name == "comparative_quality"


class TestAsyncHallucination:
    def test_basic(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            c.ahallucination("The sky is blue.", "Rayleigh scattering causes blue sky.")
        )
        assert isinstance(result, CheckResult)
        assert result.metric_name == "hallucination"
        assert result.passed is True

    def test_with_threshold(self):
        c = _c(0.6)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.ahallucination("output", "context", threshold=0.5))
        assert result.passed is True


class TestAsyncRelevance:
    def test_basic(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            c.arelevance("Relevant answer here.", "What is relevance?")
        )
        assert result.metric_name == "relevance"
        assert result.passed is True

    def test_with_system_prompt(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            c.arelevance("output", "query", system_prompt="Be strict.")
        )
        assert result.metric_name == "relevance"


class TestAsyncToxicity:
    def test_basic(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.atoxicity("This is a friendly message."))
        assert result.metric_name == "toxicity"
        assert result.passed is True

    def test_with_threshold(self):
        c = _c(0.6)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.atoxicity("output", threshold=0.5))
        assert result.passed is True


class TestAsyncRubric:
    def test_basic(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            c.arubric("Good and thorough answer.", "Answer must be thorough.")
        )
        assert result.metric_name == "rubric"
        assert result.passed is True

    def test_with_threshold(self):
        c = _c(0.8)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.arubric("output", "criteria", threshold=0.7))
        assert result.passed is True


class TestAsyncFluency:
    def test_basic(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            c.afluency("This sentence is grammatically correct and fluent.")
        )
        assert result.metric_name == "fluency"
        assert result.passed is True

    def test_with_system_prompt(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.afluency("output", system_prompt="Be strict."))
        assert result.metric_name == "fluency"


class TestAsyncCoherence:
    def test_basic(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.acoherence("This is a coherent and logical paragraph."))
        assert result.metric_name == "coherence"
        assert result.passed is True

    def test_with_threshold(self):
        c = _c(0.7)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.acoherence("output", threshold=0.5))
        assert result.passed is True


class TestAsyncSentiment:
    def test_basic(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.asentiment("I love this product! It is amazing."))
        assert result.metric_name == "sentiment"
        assert result.passed is True

    def test_with_system_prompt(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.asentiment("output", system_prompt="custom"))
        assert result.metric_name == "sentiment"


class TestAsyncCorrectness:
    def test_basic(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.acorrectness("Paris", "What is the capital of France?"))
        assert result.metric_name == "correctness"
        assert result.passed is True

    def test_with_threshold(self):
        c = _c(0.8)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.acorrectness("output", "expected", threshold=0.7))
        assert result.passed is True


class TestAsyncCaching:
    def test_async_cache_miss_appends_result(self):
        c = _c()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.ahallucination("unique_output_xyz", "context"))
        assert result.metric_name == "hallucination"
        assert len(c.results) >= 1

    def test_async_budget_exceeded(self):
        config = CheckllmConfig(budget=0.001, cache_enabled=False)
        judge = MockJudge(default_score=0.9)
        c = CheckCollector(config=config, judge=judge)
        c._accumulated_cost = 100.0
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(c.ahallucination("output", "context"))
        assert result.passed is True
        assert "Skipped" in result.reasoning

# Metrics reference

A categorized catalog of every built-in metric and deterministic check in checkllm.
Use this page as an index; for **which** metric to pick in a given scenario see
[Choosing metrics](choosing-metrics.md).

!!! tip "Import shortcuts"
    Every judge metric is also exposed as a method on the `check` pytest fixture
    (e.g. `check.faithfulness(...)`). The `Import` column shows the direct class
    import used when you instantiate a metric yourself.

    Deterministic checks live on the `DeterministicChecks` class
    (`from checkllm.deterministic import DeterministicChecks`) and as methods on
    the `check` fixture (e.g. `check.contains(...)`).

## RAG and grounding

Evaluates retrieval quality and whether generation is grounded in retrieved context.

| Metric | Import | Type | What it measures | Typical threshold |
|---|---|---|---|---|
| `faithfulness` | `checkllm.metrics.faithfulness.FaithfulnessMetric` | LLM judge | Fraction of claims in the answer that are supported by context | 0.85 |
| `faithfulness_hhem` | `checkllm.metrics.faithfulness_hhem.FaithfulnessHHEMMetric` | Model (HHEM) | HHEM-2.1 hallucination score; local, fast, cheap | 0.80 |
| `hallucination` | `checkllm.metrics.hallucination.HallucinationMetric` | LLM judge | Probability the answer contains unsupported claims (lower is better, score inverted) | 0.85 |
| `contextual_precision` | `checkllm.metrics.contextual_precision.ContextualPrecisionMetric` | LLM judge | Whether relevant chunks are ranked ahead of irrelevant ones | 0.75 |
| `contextual_recall` | `checkllm.metrics.contextual_recall.ContextualRecallMetric` | LLM judge | Fraction of the expected answer found in retrieved context | 0.80 |
| `context_relevance` | `checkllm.metrics.context_relevance.ContextRelevanceMetric` | LLM judge | Relevance of retrieved context to the query | 0.75 |
| `context_entity_recall` | `checkllm.metrics.context_entity_recall.ContextEntityRecallMetric` | LLM judge | Fraction of ground-truth entities present in context | 0.80 |
| `nonllm_context_precision` | `checkllm.metrics.nonllm_context_precision.NonLLMContextPrecisionMetric` | Deterministic | Context precision via string/semantic match (no judge) | 0.70 |
| `nonllm_context_recall` | `checkllm.metrics.nonllm_context_recall.NonLLMContextRecallMetric` | Deterministic | Context recall via string/semantic match (no judge) | 0.70 |
| `groundedness` | `checkllm.metrics.groundedness.GroundednessMetric` | LLM judge | Whether every factual claim traces back to context | 0.85 |
| `noise_sensitivity` | `checkllm.metrics.noise_sensitivity.NoiseSensitivityMetric` | LLM judge | How much irrelevant context degrades the answer | 0.75 |
| `citation_accuracy` | `checkllm.metrics.citation_accuracy.CitationAccuracyMetric` | LLM judge | Whether inline citations point to the correct chunk | 0.85 |
| `quoted_spans` | `checkllm.metrics.quoted_spans.QuotedSpansAlignmentMetric` | Deterministic | Verbatim-quoted spans actually appear in context | 0.95 |
| `nv_answer_accuracy` | `checkllm.metrics.dual_judge_nv.NVAnswerAccuracyMetric` | Dual judge | NVIDIA dual-judge answer accuracy (two models vote) | 0.80 |
| `nv_context_relevance` | `checkllm.metrics.dual_judge_nv.NVContextRelevanceMetric` | Dual judge | NVIDIA dual-judge context relevance | 0.75 |
| `nv_response_groundedness` | `checkllm.metrics.dual_judge_nv.NVResponseGroundednessMetric` | Dual judge | NVIDIA dual-judge response groundedness | 0.80 |

## Answer quality

Generic output-quality signals that apply beyond RAG.

| Metric | Import | Type | What it measures | Typical threshold |
|---|---|---|---|---|
| `relevance` | `checkllm.metrics.relevance.RelevanceMetric` | LLM judge | Whether the answer addresses the query | 0.80 |
| `correctness` | `checkllm.metrics.correctness.CorrectnessMetric` | LLM judge | Whether the answer matches the expected output | 0.80 |
| `factual_correctness` | `checkllm.metrics.factual_correctness.FactualCorrectnessMetric` | LLM judge | Precision/recall/F1 over atomic factual claims | 0.80 |
| `answer_completeness` | `checkllm.metrics.answer_completeness.AnswerCompletenessMetric` | LLM judge | Whether every sub-question in the query is answered | 0.80 |
| `response_completeness` | `checkllm.metrics.response_completeness.ResponseCompletenessMetric` | LLM judge | Whether the response covers all required aspects | 0.80 |
| `fluency` | `checkllm.metrics.fluency.FluencyMetric` | LLM judge | Grammatical and stylistic fluency | 0.80 |
| `coherence` | `checkllm.metrics.coherence.CoherenceMetric` | LLM judge | Logical flow and structure | 0.80 |
| `consistency` | `checkllm.metrics.consistency.ConsistencyMetric` | LLM judge | Internal self-consistency across the output | 0.80 |
| `summarization` | `checkllm.metrics.summarization.SummarizationMetric` | LLM judge | Summary preserves key information without distortion | 0.80 |
| `instruction_following` | `checkllm.metrics.instruction_following.InstructionFollowingMetric` | LLM judge | Whether explicit instructions were followed | 0.85 |
| `instruction_completeness` | `checkllm.metrics.instruction_completeness.InstructionCompletenessMetric` | LLM judge | Fraction of instructions executed | 0.85 |
| `prompt_alignment` | `checkllm.metrics.prompt_alignment.PromptAlignmentMetric` | LLM judge | Response stays aligned with prompt intent and constraints | 0.85 |
| `rubric` | `checkllm.metrics.rubric.RubricMetric` | LLM judge | Free-form rubric criteria you supply | 0.80 |
| `g_eval` | `checkllm.metrics.g_eval.GEvalMetric` | LLM judge | G-Eval chain-of-thought rubric scoring | 0.80 |
| `comparative_quality` | `checkllm.metrics.comparative_quality.ComparativeQualityMetric` | LLM judge | Pairwise comparison against a reference output | 0.50 |
| `sentiment` | `checkllm.metrics.sentiment.SentimentMetric` | LLM judge | Polarity; pair with an expected direction | 0.50 |

## Agents and tool use

Evaluates agent trajectories, tool calls, and planning.

| Metric | Import | Type | What it measures | Typical threshold |
|---|---|---|---|---|
| `tool_call_f1` | `checkllm.metrics.tool_call_f1.ToolCallF1Metric` | Deterministic | F1 of predicted vs expected tool names | 0.80 |
| `tool_accuracy` | `checkllm.metrics.tool_accuracy.ToolAccuracyMetric` | LLM judge | Whether the right tools were invoked for the query | 0.80 |
| `argument_correctness` | `checkllm.metrics.argument_correctness.ArgumentCorrectnessMetric` | LLM judge | Whether tool arguments are semantically correct | 0.80 |
| `task_completion` | `checkllm.metrics.task_completion.TaskCompletionMetric` | LLM judge | Whether the agent finished the requested task | 0.80 |
| `goal_accuracy` | `checkllm.metrics.goal_accuracy.GoalAccuracyMetric` | LLM judge | Final output achieves the stated goal | 0.80 |
| `plan_quality` | `checkllm.metrics.plan_quality.PlanQualityMetric` | LLM judge | The agent's plan is sensible and minimal | 0.75 |
| `plan_adherence` | `checkllm.metrics.plan_adherence.PlanAdherenceMetric` | LLM judge | Execution trace follows the declared plan | 0.80 |
| `step_efficiency` | `checkllm.metrics.step_efficiency.StepEfficiencyMetric` | LLM judge | Fraction of steps that were actually necessary | 0.70 |
| `trajectory_goal_success` | `checkllm.metrics.trajectory.TrajectoryGoalSuccessMetric` | LLM judge | Whole trajectory reaches the goal | 0.80 |
| `trajectory_tool_sequence` | `checkllm.metrics.trajectory.TrajectoryToolSequenceMetric` | Hybrid | Tools invoked in the expected order | 0.80 |
| `trajectory_step_count` | `checkllm.metrics.trajectory.TrajectoryStepCountMetric` | Hybrid | Task completed within a step budget | 0.75 |
| `trajectory_tool_args_match` | `checkllm.metrics.trajectory.TrajectoryToolArgsMatchMetric` | LLM judge | Tool arguments match expected values (fuzzy) | 0.80 |
| `mcp_use` | `checkllm.metrics.mcp_use.McpUseMetric` | LLM judge | MCP tool usage is appropriate | 0.80 |
| `mcp_task_completion` | `checkllm.metrics.mcp_task_completion.McpTaskCompletionMetric` | LLM judge | MCP session completed the task | 0.80 |
| `multi_turn_mcp_use` | `checkllm.metrics.multi_turn_mcp_use.MultiTurnMcpUseMetric` | LLM judge | Multi-turn MCP interaction stayed on task | 0.80 |

## Conversation and multi-turn

Evaluates chatbots and multi-turn sessions using `ConversationalTestCase`.

| Metric | Import | Type | What it measures | Typical threshold |
|---|---|---|---|---|
| `conversation_completeness` | `checkllm.metrics.conversation_completeness.ConversationCompletenessMetric` | LLM judge | Every user request in the transcript was satisfied | 0.80 |
| `role_adherence` | `checkllm.metrics.role_adherence.RoleAdherenceMetric` | LLM judge | Assistant stayed in its declared role | 0.85 |
| `knowledge_retention` | `checkllm.metrics.knowledge_retention.KnowledgeRetentionMetric` | LLM judge | Assistant remembers information from earlier turns | 0.80 |
| `topic_adherence` | `checkllm.metrics.topic_adherence.TopicAdherenceMetric` | LLM judge | Conversation stays within allowed topics | 0.85 |
| `turn_relevancy` | `checkllm.metrics.per_turn.TurnRelevancyMetric` | LLM judge | Per-turn response relevance | 0.80 |
| `turn_faithfulness` | `checkllm.metrics.per_turn.TurnFaithfulnessMetric` | LLM judge | Per-turn faithfulness to provided context | 0.85 |
| `turn_coherence` | `checkllm.metrics.per_turn.TurnCoherenceMetric` | LLM judge | Per-turn coherence with surrounding turns | 0.80 |

## Safety and red team

Runtime safety signals. Use these as guardrails **and** inside red-team scans.

| Metric | Import | Type | What it measures | Typical threshold |
|---|---|---|---|---|
| `toxicity` | `checkllm.metrics.toxicity.ToxicityMetric` | LLM judge | Toxic/abusive content probability | 0.80 |
| `bias` | `checkllm.metrics.bias.BiasMetric` | LLM judge | Demographic, political, or ideological bias | 0.80 |
| `pii_detection` | `checkllm.metrics.pii_detection.PIIDetectionMetric` | LLM judge | Free-text PII that regex misses (names, addresses) | 0.90 |
| `misuse_detection` | `checkllm.metrics.misuse_detection.MisuseDetectionMetric` | LLM judge | Output assists harmful or disallowed use | 0.90 |
| `role_violation` | `checkllm.metrics.role_violation.RoleViolationMetric` | LLM judge | Response breaks configured role constraints | 0.85 |
| `non_advice` | `checkllm.metrics.non_advice.NonAdviceMetric` | LLM judge | Response avoids regulated professional advice | 0.85 |
| `no_pii` (deterministic) | `checkllm.deterministic.DeterministicChecks.no_pii` | Deterministic | SSN/email/credit-card regex screen | n/a (pass/fail) |
| `is_refusal` | `checkllm.deterministic.DeterministicChecks.is_refusal` | Deterministic | Text is a safety refusal ("I can't help with that") | n/a (pass/fail) |

## Code and structured output

Evaluates code, SQL, JSON, and other structured responses.

| Metric | Import | Type | What it measures | Typical threshold |
|---|---|---|---|---|
| `code_correctness` | `checkllm.metrics.code_correctness.CodeCorrectnessMetric` | LLM judge | Code meets functional and quality criteria | 0.80 |
| `sql_equivalence` | `checkllm.metrics.sql_equivalence.SQLEquivalenceMetric` | LLM judge | Two SQL queries are logically equivalent | 0.85 |
| `datacompy_score` | `checkllm.metrics.datacompy_score.DataCompyMetric` | Deterministic | Pandas dataframe equality via `datacompy` | 0.95 |
| `dag` | `checkllm.metrics.dag.DAGMetric` | Composite | Node-graph of metrics with gating edges | custom |

## Multimodal and vision

Evaluates responses involving images or other modalities.

| Metric | Import | Type | What it measures | Typical threshold |
|---|---|---|---|---|
| `image_text_alignment` | `checkllm.metrics.image_text_alignment.ImageTextAlignmentMetric` | LLM judge | Text matches the provided image | 0.80 |
| `image_captioning_quality` | `checkllm.metrics.image_captioning_quality.ImageCaptioningQualityMetric` | LLM judge | Caption quality and completeness | 0.80 |
| `image_coherence` | `checkllm.metrics.image_coherence.ImageCoherenceMetric` | LLM judge | Output coherent with the image context | 0.80 |
| `image_consistency` | `checkllm.metrics.image_consistency.ImageConsistencyMetric` | LLM judge | Multiple images remain consistent | 0.80 |
| `image_editing` | `checkllm.metrics.image_editing.ImageEditingMetric` | LLM judge | Edit matches the requested transformation | 0.80 |
| `image_helpfulness` | `checkllm.metrics.image_helpfulness.ImageHelpfulnessMetric` | LLM judge | Image actually helps answer the query | 0.80 |
| `image_reference` | `checkllm.metrics.image_reference.ImageReferenceMetric` | LLM judge | Text refers to correct image regions | 0.80 |
| `image_relevance` | `checkllm.metrics.image_relevance.ImageRelevanceMetric` | LLM judge | Image is relevant to the text query | 0.75 |
| `image_safety` | `checkllm.metrics.image_safety.ImageSafetyMetric` | LLM judge | Image content is safe | 0.90 |
| `text_to_image` | `checkllm.metrics.text_to_image.TextToImageMetric` | LLM judge | Generated image matches the text prompt | 0.80 |
| `multimodal_faithfulness` | `checkllm.metrics.multimodal_faithfulness.MultimodalFaithfulnessMetric` | LLM judge | Answer grounded in both image and text context | 0.85 |
| `visual_faithfulness` | `checkllm.metrics.visual_faithfulness.VisualFaithfulnessMetric` | LLM judge | Visual claims are supported by the image | 0.85 |
| `visual_hallucination` | `checkllm.metrics.visual_hallucination.VisualHallucinationMetric` | LLM judge | Output hallucinates visual details | 0.85 |
| `visual_reasoning` | `checkllm.metrics.visual_reasoning.VisualReasoningMetric` | LLM judge | Correct reasoning over visual content | 0.80 |
| `ocr_accuracy` | `checkllm.metrics.ocr_accuracy.OCRAccuracyMetric` | LLM judge | OCR output matches image text | 0.90 |
| `chart_value_extraction` | `checkllm.metrics.chart_value_extraction.ChartValueExtractionMetric` | LLM judge | Numeric values extracted from a chart are correct | 0.90 |
| `diagram_comprehension` | `checkllm.metrics.diagram_comprehension.DiagramComprehensionMetric` | LLM judge | Understanding of a technical diagram | 0.80 |

## Deterministic checks (structural and lexical)

Deterministic checks are free, instant, and stable. Reach for these first and
only add judge metrics where judgement is required.

### Content presence

| Check | Signature | What it checks |
|---|---|---|
| `contains` | `contains(output, substring)` | Substring is present |
| `not_contains` | `not_contains(output, substring)` | Substring is absent |
| `icontains` | `icontains(output, substring)` | Case-insensitive contains |
| `icontains_any` / `icontains_all` | `(output, substrings)` | Any/all substrings present (case-insensitive) |
| `all_of` / `any_of` / `none_of` | `(output, substrings)` | Set membership over required substrings |
| `starts_with` / `ends_with` | `(output, prefix/suffix)` | Prefix/suffix check |
| `exact_match` | `exact_match(output, expected, ignore_case=False)` | Full equality |
| `exact_match_strict` | `(output, reference, ignore_case, ignore_whitespace)` | Full equality with whitespace options |
| `regex` | `regex(output, pattern)` | Regex match |
| `has_structure` | `has_structure(output, elements)` | Required structural elements present |
| `has_citations` | `has_citations(output, min_count=1)` | At least N citation markers |

### Size and shape

| Check | Signature | What it checks |
|---|---|---|
| `max_tokens` / `min_tokens` | `(output, limit / minimum)` | Token budget bounds |
| `word_count` | `word_count(output, min_words=None, max_words=None)` | Word-count bounds |
| `char_count` | `char_count(output, min_chars=None, max_chars=None)` | Character-count bounds |
| `sentence_count` | `sentence_count(output, min_sentences=None, max_sentences=None)` | Sentence-count bounds |
| `readability` | `readability(output, max_grade=None, min_grade=None)` | Flesch-Kincaid grade-level bounds |
| `no_repetition` | `no_repetition(output, max_ngram_repeat=3)` | No N-gram repeated beyond a limit |

### Format validation

| Check | Signature | What it checks |
|---|---|---|
| `is_json` | `is_json(output)` | Parses as JSON |
| `json_schema` | `json_schema(output, schema)` | Conforms to a Pydantic schema |
| `json_field` | `json_field(output, field_path, expected=None, condition=None)` | JSON field equals/satisfies condition |
| `is_valid_python` | `is_valid_python(output)` | Parses as Python |
| `is_valid_sql` | `is_valid_sql(output)` | Parses as SQL |
| `is_valid_yaml` / `is_yaml` | `(output)` | Parses as YAML |
| `is_valid_markdown` | `is_valid_markdown(output, require_headers=False, require_lists=False, require_code_blocks=False)` | Markdown structural checks |
| `is_html` / `contains_html` | `(output)` | HTML structure / fragment |
| `is_xml` / `contains_xml` | `(output)` | XML structure / fragment |
| `is_url` / `is_valid_url` / `has_url` | `(output)` | URL validation |

### Similarity metrics

| Check | Signature | What it checks |
|---|---|---|
| `similarity` | `similarity(output, expected, threshold=0.8, ignore_case=False)` | Token-based Jaccard similarity |
| `levenshtein` | `levenshtein(output, reference, threshold=0.7)` | Normalised edit distance |
| `string_distance` | `string_distance(output, reference, method="levenshtein", threshold=0.7)` | Pluggable string distance |
| `semantic_similarity` | `semantic_similarity(output, reference, threshold=0.7)` | Embedding cosine similarity |
| `bleu` | `bleu(output, reference, threshold=0.5)` | BLEU n-gram overlap |
| `rouge_l` | `rouge_l(output, reference, threshold=0.5)` | ROUGE-L longest-common-subsequence |
| `meteor` | `meteor(output, reference, threshold=0.5)` | METEOR with stemming/synonyms |
| `gleu` | `gleu(output, reference, threshold=0.5)` | GLEU (Google BLEU) |
| `chrf` | `chrf(output, reference, threshold=0.5)` | ChrF character-n-gram F-score |

### Budgets and numeric

| Check | Signature | What it checks |
|---|---|---|
| `max_tokens` / `min_tokens` | `(output, limit / minimum)` | Token budget |
| `latency` | `latency(actual_ms, max_ms)` | Elapsed latency under bound |
| `latency_check` | `latency_check(start_time, end_time, max_ms=5000.0)` | Latency from wall-clock times |
| `cost` | `cost(actual_usd, max_usd)` | USD cost under bound |
| `cost_check` | `cost_check(input_tokens, output_tokens, model, max_cost=1.0)` | Estimated cost from token counts |
| `greater_than` / `less_than` / `between` | `(output, threshold [, high])` | Numeric bounds on parsed output |
| `perplexity_check` | `perplexity_check(output, max_perplexity=50.0)` | Perplexity upper bound |

### Safety primitives

| Check | Signature | What it checks |
|---|---|---|
| `no_pii` | `no_pii(output, patterns=None)` | No SSN/email/credit-card patterns |
| `is_refusal` | `is_refusal(output)` | Output is a safety refusal phrase |
| `language` | `language(output, expected)` | Detected language matches |

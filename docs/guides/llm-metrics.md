# LLM-as-Judge Metrics

24 metrics that use an LLM to evaluate outputs. Requires an API key or local Ollama.

## RAG & Context Metrics

### hallucination
Checks if the output contains claims not supported by the context.
```python
check.hallucination(output, context="source text")
```

### faithfulness
Checks if the answer is faithful to the provided context (RAG-specific).
```python
check.faithfulness(output, context="source text", query="user question")
```

### context_relevance
Checks if the retrieved context is relevant to the query.
```python
check.context_relevance(context="retrieved text", query="user question")
```

### answer_completeness
Checks if the answer fully addresses the query.
```python
check.answer_completeness(output, query="user question")
```

### groundedness
Claim-by-claim verification against multiple sources.
```python
check.groundedness(output, sources=["source1", "source2"])
```

### contextual_precision
Checks if the most relevant documents are ranked higher.
```python
check.contextual_precision(output, context=["doc1", "doc2"], query="q", expected="answer")
```

### contextual_recall
Checks if all claims in the ground truth are supported by context.
```python
check.contextual_recall(output, context=["doc1", "doc2"], expected="ground truth")
```

## Quality Metrics

### relevance
Query-output relevance scoring.
```python
check.relevance(output, query="user question")
```

### fluency
Writing quality and naturalness.
```python
check.fluency(output)
```

### coherence
Logical structure and consistency.
```python
check.coherence(output)
```

### correctness
Semantic comparison to expected output.
```python
check.correctness(output, expected="expected answer")
```

### consistency
Multi-output consistency across multiple runs.
```python
check.consistency([output1, output2, output3])
```

### instruction_following
Compliance with format, style, and constraint instructions.
```python
check.instruction_following(output, instructions="Respond in bullet points under 100 words")
```

### summarization
Summary accuracy, conciseness, and retention.
```python
check.summarization(output, source="original text")
```

## Safety Metrics

### toxicity
Harmful content detection.
```python
check.toxicity(output)
```

### bias
Demographic, cultural, gender, and racial bias detection.
```python
check.bias(output, categories=["gender", "racial"])
```

### sentiment
Tone and mood assessment.
```python
check.sentiment(output, threshold=0.6)
```

## Custom Evaluation

### rubric
Evaluate against custom criteria.
```python
check.rubric(output, criteria="concise, mentions key findings, uses formal tone")
```

### g_eval
Chain-of-thought evaluation with custom criteria and steps. Uses LLM reasoning to score outputs step by step.
```python
check.g_eval(output, criteria="accuracy", steps=["Check facts", "Verify claims"])
```

## Agent & Conversation Metrics

### task_completion
Goal accomplishment check.
```python
check.task_completion(output, task_description="Search the database", constraints=["under 5 seconds"])
```

### role_adherence
Persona consistency.
```python
check.role_adherence(output, role_description="friendly customer support agent")
```

### tool_accuracy
Agent tool selection evaluation.
```python
check.tool_accuracy(output, expected_tools=[{"name": "search"}], query="find records")
```

### knowledge_retention
Multi-turn conversation memory.
```python
check.knowledge_retention(output, conversation=[...], key_facts=["user name is Alice"])
```

### conversation_completeness
Multi-turn request fulfillment.
```python
check.conversation_completeness(output, conversation=[...], requirements=["answered all questions"])
```

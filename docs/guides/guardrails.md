# Production Guardrails

Use the same checks from your tests in production to validate LLM outputs at runtime.

## Basic Usage

```python
from checkllm import Guard, CheckSpec

guard = Guard(checks=[
    CheckSpec(check_type="no_pii"),
    CheckSpec(check_type="max_tokens", params={"limit": 500}),
    CheckSpec(check_type="toxicity"),
])

result = guard.validate(llm_output)
if not result.valid:
    print(result.summary())
    result.raise_on_failure()  # raises GuardrailError
```

## FastAPI Middleware

```python
from fastapi import FastAPI
from checkllm import Guard, CheckSpec, GuardrailMiddleware

app = FastAPI()
guard = Guard(checks=[
    CheckSpec(check_type="no_pii"),
    CheckSpec(check_type="toxicity"),
])
app.add_middleware(GuardrailMiddleware, guard=guard, response_field="output")
```

## Function Decorator

```python
from checkllm import guardrail, CheckSpec

@guardrail(checks=[
    CheckSpec(check_type="no_pii"),
    CheckSpec(check_type="max_tokens", params={"limit": 200}),
])
def generate_response(prompt: str) -> str:
    return my_llm(prompt)  # output is validated automatically
```

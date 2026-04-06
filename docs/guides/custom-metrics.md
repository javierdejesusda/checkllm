# Custom Metrics

## The `@metric` Decorator

```python
from checkllm import metric, CheckResult

@metric("brevity")
def brevity_check(output: str, max_words: int = 50, **kwargs) -> CheckResult:
    words = len(output.split())
    return CheckResult(
        passed=words <= max_words,
        score=min(1.0, max_words / max(words, 1)),
        reasoning=f"{words} words (limit: {max_words})",
        cost=0.0,
        latency_ms=0,
        metric_name="brevity",
    )
```

## Using Custom Metrics

Once registered, custom metrics appear in `checkllm list-metrics` and can be used with the `Evaluator` and `Guard` APIs.

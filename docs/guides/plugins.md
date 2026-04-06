# Plugin Development

Create and publish community metrics as pip-installable packages.

## Create a Plugin

```python
# checkllm_jailbreak/metric.py
from checkllm import metric, CheckResult

@metric("jailbreak_detection")
def jailbreak_detection(output: str, **kwargs) -> CheckResult:
    # Your detection logic here
    suspicious = any(phrase in output.lower() for phrase in ["ignore previous", "disregard"])
    return CheckResult(
        passed=not suspicious,
        score=0.0 if suspicious else 1.0,
        reasoning="Jailbreak attempt detected" if suspicious else "Clean",
        cost=0.0, latency_ms=0, metric_name="jailbreak_detection",
    )
```

## Register via Entry Points

```toml
# pyproject.toml
[project.entry-points."checkllm.metrics"]
jailbreak_detection = "checkllm_jailbreak.metric:jailbreak_detection"
```

## Verify

```bash
pip install -e .
checkllm list-metrics  # Should show your plugin under "Plugin Metrics"
```

---
name: New Metric Proposal
about: Propose a new check or LLM metric
labels: metric
---

## Metric Name

`your_metric_name`

## Type

- [ ] Deterministic (free, instant)
- [ ] LLM-as-judge (requires API key)

## Description

What does this metric evaluate?

## Scoring

How should the 0.0-1.0 score be determined?

## Example Usage

```python
def test_example(check):
    check.your_metric_name(output, ...)
```

## Use Cases

When would a developer use this metric?

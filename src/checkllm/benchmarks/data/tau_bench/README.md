# tau-bench vendored fixtures

This directory contains a small set of **synthetic** fixture tasks in the
tau-bench schema (Yao et al. 2024, <https://github.com/sierra-research/tau-bench>,
Apache-2.0).

**These are NOT the real tau-bench task content.** They exist only to exercise
`checkllm.benchmarks.tau_bench_loader.load_tau_bench()` without requiring a network
fetch or a full upstream checkout.

To run CheckLLM against the real tau-bench, clone the upstream repo and pass
the checkout root as `data_root`:

```python
from pathlib import Path
from checkllm.benchmarks import load_tau_bench

tasks = load_tau_bench("airline", data_root=Path("/path/to/tau-bench/data"))
```

Upstream commit pinned in `benchmarks/paper/dataset_manifest.json`.

## Attribution

tau-bench: Yao, Shinn, Yan, Hirata, Cao, Klein, Yang, Fleet. "tau-bench: A
Benchmark for Tool-Agent-User Interaction in Real-World Domains." arXiv:2406.12045.

Licensed under Apache-2.0. These synthetic fixtures are original CheckLLM
work and are not subject to the upstream license.

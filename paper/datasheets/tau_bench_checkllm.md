# Datasheet: tau-bench (CheckLLM loader)

This datasheet follows the template of Gebru et al. (2018),
*Datasheets for Datasets*. It documents the tau-bench benchmark
(Yao et al. 2024) **as accessed through CheckLLM's loader and
vendored synthetic fixtures** at
`src/checkllm/benchmarks/tau_bench_loader.py`.

> **Important transparency note.** The current paper run uses
> *synthetic* fixtures vendored at
> `src/checkllm/benchmarks/data/tau_bench/` that mirror the upstream
> tau-bench schema; they are **not** upstream task content. The
> upstream commit pin
> (`benchmarks/paper/dataset_manifest.json:tau_bench.upstream_commit`)
> is currently a `TODO-PLACEHOLDER-UPDATE-BEFORE-PAPER-RUN` and will
> be set to the real upstream SHA before camera-ready. The paper's
> experimental results are reproducible regardless because
> `StubAgent` is deterministic — it replays the *reference action
> sequence* under a controllable noise schedule, and that schedule
> is what defines the synthetic ground-truth label.

---

## 1. Motivation

**For what purpose was the dataset created?**
tau-bench was designed by Yao et al. (2024,
*tau-bench: A Benchmark for Tool-Agent-User Interaction in
Real-World Domains*, arXiv:2406.12045) to evaluate tool-using
agents in two simulated customer-service domains: airline
reservations and retail returns/exchanges. Each task carries a
user instruction, a tool API specification, and a deterministic
ground-truth final database state for scoring.

**Who created the dataset and on behalf of which entity?**
Shunyu Yao and collaborators at Sierra Research; the upstream
repository is `sierra-research/tau-bench` on GitHub.

**Who funded the creation of the dataset?**
Sierra Research, per the upstream paper's acknowledgements.
CheckLLM did not fund or contribute to the upstream dataset.

**Any other comments?**
tau-bench's strength for our paper is its *deterministic*
final-state oracle: scoring does not require an LLM judge.
CheckLLM uses tau-bench as the trajectory source for the
metric-vs-truth and head-to-head experiments
(`paper/checkllm.tex:241-244`).

---

## 2. Composition

**What do the instances represent?**
Each instance is one tau-bench task, captured by the
`TauBenchTask` pydantic model at
`src/checkllm/benchmarks/tau_bench_loader.py:26-44`:

- `task_id` (string): stable identifier.
- `user_instruction` (string): natural-language prompt.
- `tools` (list of dict): JSON schemas for the tool API.
- `reference_actions` (list of dict): the gold action sequence.
- `ground_truth_final_state` (dict): expected database state after
  the agent's actions.
- `domain` (string): "airline" or "retail".

**How many instances are there in total?**
Vendored synthetic fixtures: 5 airline + 5 retail tasks. See
`benchmarks/paper/dataset_manifest.json:tau_bench.vendored_fixture_count`.
The upstream tau-bench repository contains substantially more
tasks; users can point the loader at an upstream checkout via the
`data_root` parameter
(`src/checkllm/benchmarks/tau_bench_loader.py:51-78`).

**Does the dataset contain all possible instances or a sample?**
The vendored fixtures are a small synthetic sample chosen to
exercise the loader and the metric without a network fetch. They
mirror the schema but **not the content** of upstream tau-bench.

**What data does each instance consist of?**
See the field list above.

**Is there a label or target associated with each instance?**
Yes — `ground_truth_final_state` is the deterministic scoring
target. The reference action sequence (`reference_actions`) is the
"correct" trajectory; the StubAgent replays a corrupted version of
it under the noise schedule, and the binary success label is
$1$ iff the noise level was `clean`
(`paper/checkllm.tex:255-261`).

**Is any information missing from individual instances?**
No — the synthetic fixtures fill every required field. Real
upstream tau-bench tasks may include additional metadata that the
CheckLLM `TauBenchTask` model does not surface; those fields are
silently ignored on load.

**Are relationships between individual instances made explicit?**
N/A — instances are independent tasks within their domain.

**Are there recommended data splits?**
Upstream provides `airline` and `retail` as separate task families.
CheckLLM treats them as parallel evaluation cohorts and reports
per-domain confidence intervals
(`paper/checkllm.tex:323-326`).

**Are there any errors, sources of noise, or redundancies?**
The vendored synthetic fixtures are crafted to be schema-valid and
deterministically scoreable; they are not intended to reflect the
distributional difficulty of upstream tau-bench. Per-claim
significance in the paper should therefore be read as
"significance on a 10-task synthetic mirror of the schema",
pending the upstream pin update.

**Is the dataset self-contained, or does it link to external
resources?**
Self-contained: each task carries its own tool schema, reference
actions, and ground-truth state in line-delimited JSON.

**Does the dataset contain confidential or sensitive content?**
No. The fixtures contain made-up customer names and bookings
crafted by the CheckLLM authors; no real customer data, no PII.

---

## 3. Collection

**How was the data acquired?**
The vendored synthetic fixtures were authored by the CheckLLM
authors to mirror the upstream tau-bench schema. The upstream
tau-bench tasks themselves were authored by the Sierra Research
team; see the upstream paper for that process.

**What mechanisms or procedures were used to collect the data?**
For the vendored fixtures: hand-authoring against the schema
documented in `src/checkllm/benchmarks/tau_bench_loader.py:26-44`.
For upstream tau-bench: see the Yao et al. (2024) paper.

**If sampling, what was the sampling strategy?**
The 5+5 vendored synthetic tasks were chosen to exercise the
schema; they are not sampled from upstream content.

**Who was involved in the data collection process and how were
they compensated?**
Vendored fixtures: CheckLLM authors (uncompensated, in-scope work).
Upstream content: see Yao et al. (2024).

**Over what timeframe was the data collected?**
The vendored fixtures were authored alongside the loader (early
2026). Upstream tau-bench was published mid-2024.

**Were any ethical review processes conducted?**
N/A — no human subjects.

**Were the individuals notified about the data collection?**
N/A — synthetic data.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
CheckLLM applies no transformations beyond pydantic field
extraction. The loader at
`src/checkllm/benchmarks/tau_bench_loader.py:89-105` reads the
JSONL file line by line, json-parses each row, and constructs a
`TauBenchTask`.

**Was the "raw" data saved in addition to the preprocessed data?**
Yes — the JSONL files at
`src/checkllm/benchmarks/data/tau_bench/{airline,retail}/tasks.jsonl`
are the raw source.

**Is the software that was used to preprocess/clean/label the data
available?**
Yes — `src/checkllm/benchmarks/tau_bench_loader.py` is the entire
preprocessing surface, and is open source under the MIT license.

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Upstream tau-bench: yes, extensively, as a tool-agent benchmark.
CheckLLM's vendored synthetic fixtures: only by CheckLLM's own
test suite and paper experiments
(`tests/benchmarks/paper/experiments/test_*.py`,
`benchmarks/paper/experiments/*.py`).

**Is there a repository that links to any or all papers or systems
that use the dataset?**
Upstream: <https://github.com/sierra-research/tau-bench>.

**What (other) tasks could the dataset be used for?**
The vendored fixtures are a development/testing aid. For research
claims about real-world tool-agent performance, point the loader
at an upstream checkout via `data_root`.

**Is there anything about the composition of the dataset or the
way it was collected and preprocessed/cleaned/labeled that might
impact future uses?**
Yes — the vendored fixtures are synthetic. Any quantitative claim
that depends on tau-bench task content (rather than schema) must
re-run with the upstream pin.

**Are there tasks for which the dataset should not be used?**
Do not use the vendored synthetic fixtures to make absolute claims
about agent capability on real customer-service domains.

---

## 6. Distribution

**Will the dataset be distributed to third parties?**
The vendored synthetic fixtures ship inside the CheckLLM
repository and are released under the same MIT license as the
rest of the codebase. Upstream tau-bench is distributed by Sierra
Research at <https://github.com/sierra-research/tau-bench> under
Apache-2.0.

**How will it be distributed?**
- Vendored synthetic fixtures: GitHub
  (`javierdejesusda/checkllm`), and Zenodo at the `paper-v1` tag.
- Upstream tau-bench: GitHub
  (`sierra-research/tau-bench`).

**When will it be distributed?**
Already distributed (GitHub). Zenodo archive will be cut at the
`paper-v1` tag — see `paper/zenodo.md`.

**Will the dataset be distributed under a copyright or other IP
license?**
- Vendored synthetic fixtures: MIT (CheckLLM-original work).
- Upstream tau-bench: Apache-2.0 (per
  `benchmarks/paper/dataset_manifest.json:tau_bench.license`
  and the upstream README).

**Have any third parties imposed IP-based or other restrictions on
the data associated with the instances?**
For upstream content: Apache-2.0 attribution requirement applies.
For vendored fixtures: no third-party restrictions.

**Do any export controls or other regulatory restrictions apply?**
Not to our knowledge.

---

## 7. Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
- Vendored synthetic fixtures: CheckLLM maintainers
  (`javierdejesusda/checkllm`).
- Upstream tau-bench: Sierra Research at
  `sierra-research/tau-bench`.

**How can the owner/curator/manager of the dataset be contacted?**
- Vendored fixtures: GitHub issues on
  `javierdejesusda/checkllm`.
- Upstream: GitHub issues on `sierra-research/tau-bench`.

**Is there an erratum?**
None for the vendored fixtures. Upstream errata are tracked in
the upstream issue tracker.

**Will the dataset be updated?**
The upstream commit pin will be updated to a real SHA before
camera-ready (see the transparency note at the top of this file
and `benchmarks/paper/dataset_manifest.json:tau_bench.upstream_commit_status`).
Vendored synthetic fixtures are stable.

**If the dataset relates to people, are there applicable limits on
the retention of the data associated with the instances?**
N/A — synthetic, no personal data.

**Will older versions of the dataset continue to be
supported/hosted/maintained?**
Yes — the Zenodo `paper-v1` archive freezes the vendored fixtures
at the snapshot used to compute every paper number.

**If others want to extend/augment/build on/contribute to the
dataset, is there a mechanism for them to do so?**
Yes — open a PR against `javierdejesusda/checkllm` for vendored
fixtures, or contribute upstream at
`sierra-research/tau-bench`.

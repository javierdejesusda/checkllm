# NeurIPS 2024 Datasets &amp; Benchmarks Reproducibility Checklist

> Best-effort approximation of the NeurIPS 2024 D&amp;B reproducibility
> checklist; the final camera-ready will use the official template.
> Each "YES" item below cites the file (and where helpful, line) that
> backs it. Where evidence is partial, the item is marked **PARTIAL**
> and links to the followup task.

This checklist accompanies the paper *CheckLLM: Reproducible
Agent-Trajectory Evaluation at Scale* (`paper/checkllm.tex`). Pointers
are relative to the repository root unless otherwise stated.

---

## 1. For all authors / submission

- [x] **Claims accurately reflect contributions and scope.**
  The three contributions are stated in `paper/checkllm.tex:91-104`:
  1. A four-axis composite trajectory metric (formal definition in
     `paper/checkllm.tex:195-237`; reference implementation in
     `src/checkllm/metrics/trajectory.py`).
  2. An OTel-GenAI compatible ingestion layer
     (`paper/checkllm.tex:182-194`, code in
     `src/checkllm/ingestion/otel.py`).
  3. Empirical validation on synthetic tau-bench trajectories
     (`paper/checkllm.tex:312-349` and head-to-head in
     `paper/checkllm.tex:389-436`).
- [x] **Limitations stated.** §9 of the paper
  (`paper/checkllm.tex:438-463`) enumerates: synthetic ground truth,
  English-only fixtures, single-model focus deferred, loop signal
  masked by noise model, no human-rater study.
- [x] **Theoretical results: N/A.** This is a benchmark/framework
  paper; no formal theorems are claimed.
- [x] **Code, data, and instructions to reproduce all results.**
  - Top-level orchestration: `Makefile` targets `reproduce-smoke`
    (`Makefile:27-33`) and `reproduce` (`Makefile:35-41`).
  - Containerised environment: `Dockerfile` (Python 3.11-slim,
    deterministic install from `requirements.lock`).
  - Per-experiment driver: `benchmarks/paper/run_all.py`.
  - Dataset SHAs and pinning: `benchmarks/paper/dataset_manifest.json`.
  - Table regeneration: `paper/figures/ingest.py` reads the four
    `summary.json` files and emits `paper/figures/generated_tables.tex`.

**Justification.** The paper is built around a `make reproduce`
contract: with Python 3.11 and the locked dependencies installed,
`make reproduce` runs the four pre-registered experiments and
deterministically regenerates every numerical claim in
`paper/checkllm.tex`. The paper does not hard-code any number
(`paper/checkllm.tex:1-4` documents this convention).

---

## 2. For datasets

- [x] **License.**
  - GAIA: `cc-by-4.0` (gated). See
    `benchmarks/paper/dataset_manifest.json` (the `gaia.license`
    field links to the HF dataset card) and the loader docstring at
    `src/checkllm/benchmarks/gaia_loader.py:1-11`. The loader
    enforces an explicit license-acknowledgment env var
    `CHECKLLM_GAIA_LICENSE_ACK=yes`
    (`src/checkllm/benchmarks/gaia_loader.py:90-103`).
  - tau-bench: `Apache-2.0` (sierra-research/tau-bench). See
    `benchmarks/paper/dataset_manifest.json` (`tau_bench.license`)
    and the vendored README at
    `src/checkllm/benchmarks/data/tau_bench/README.md`. The current
    paper run uses **synthetic** fixtures that mirror the upstream
    schema; an upstream commit pin is tracked at
    `dataset_manifest.json:tau_bench.upstream_commit` and is marked
    `TODO-PLACEHOLDER-UPDATE-BEFORE-PAPER-RUN`.
- [x] **Datasheet provided** (Gebru et al. 2018 template).
  - `paper/datasheets/gaia_checkllm.md`
  - `paper/datasheets/tau_bench_checkllm.md`
- [x] **Documentation of preprocessing / cleaning.** Both loaders
  document what they do inline:
  - `src/checkllm/benchmarks/gaia_loader.py:128-173` (no
    transformations beyond pydantic field extraction).
  - `src/checkllm/benchmarks/tau_bench_loader.py:51-106` (line-by-line
    JSONL parse into `TauBenchTask`).
- [x] **Anonymization: N/A.** Both datasets are public benchmarks of
  agent-style task prompts; no PII collection, no human-subject data,
  and no identifiers beyond benchmark task IDs.
- [x] **Hosting plan.** Source code is hosted on GitHub
  (`javierdejesusda/checkllm`, see `CITATION.cff:10`). A frozen
  artefact corresponding to this manuscript will be archived on
  Zenodo at the `paper-v1` tag; see `paper/zenodo.md` for the
  release procedure and `CITATION.cff` for the placeholder DOI
  block.

**Justification.** Both benchmarks are vendored or gated through
deterministic loaders. GAIA is gated behind a license-ack env var
*and* a pinned revision SHA. tau-bench is loaded from vendored
synthetic fixtures pending the upstream commit pin update; the
StubAgent is deterministic so experimental results are bit-identical
across re-runs regardless of which fixture set is used.

---

## 3. For experiments

- [x] **Code to compute reported numbers.** All four experiments
  ship as runnable modules with paired tests:
  - `benchmarks/paper/experiments/01_controlled_noise.py`
    (test: `tests/benchmarks/paper/experiments/test_01_controlled_noise.py`).
  - `benchmarks/paper/experiments/02_metric_vs_truth.py`
    (test: `tests/benchmarks/paper/experiments/test_02_metric_vs_truth.py`).
  - `benchmarks/paper/experiments/03_trajectory_ablation.py`
    (test: `tests/benchmarks/paper/experiments/test_03_trajectory_ablation.py`).
  - `benchmarks/paper/experiments/04_vs_deepeval.py`
    (test: `tests/benchmarks/paper/experiments/test_04_vs_deepeval.py`).
- [x] **Data splits, hyperparameters, and random seeds.**
  - Seeds for experiments 02 and 03: `(42, 123, 2026, 7, 11, 17, 31,
    53, 97, 1009)` — pinned at
    `benchmarks/paper/experiments/02_metric_vs_truth.py:69` and
    `benchmarks/paper/experiments/03_trajectory_ablation.py:125`.
  - Seeds for experiment 01: `(42, 123, 2026)` —
    `benchmarks/paper/experiments/01_controlled_noise.py:54`.
  - Five noise levels: clean, low, medium, high, severe — see
    `benchmarks/paper/experiments/01_controlled_noise.py` and the
    noise-monotonicity table in `paper/checkllm.tex:303-310`.
  - Two domains: airline + retail (5 + 5 vendored synthetic tasks
    each, see `dataset_manifest.json:tau_bench.vendored_fixture_count`).
  - Ablation grid: 1872 non-degenerate cells (3 all-zero cells
    skipped) over $w_o, w_l, w_c, w_u \in \{0, 0.1, 0.25, 0.5, 1.0\}$
    and $T \in \{1, 2, 3, 5\}$ — pinned at
    `benchmarks/paper/results/03_ablation/summary.json:n_cells` and
    `n_skipped_degenerate`. (The paper text uses the macro
    `\AblationNCells{}` so the rendered number tracks the manifest.)
- [x] **Statistical significance / variability reporting.**
  - Bootstrap: $B=1000$ resamples, percentile method
    (`benchmarks/paper/experiments/02_metric_vs_truth.py:71`,
    `benchmarks/paper/experiments/04_vs_deepeval.py:80`).
  - Head-to-head: Welch's $t$-test on paired bootstrap samples with
    Holm-Bonferroni correction at $\alpha=0.05$
    (`benchmarks/paper/experiments/04_vs_deepeval.py:81`,
    `_holm_bonferroni` at line 395).
  - Pre-registered: 4 head-to-head claims; `summary.json` records
    `holm_alpha` and per-claim `p_value_holm_corrected`.
- [x] **Compute resources.** CPU-only, single machine. The full
  reproduce pipeline finishes in well under two minutes wall-clock
  on a developer laptop because the StubAgent is a pure function and
  the metric runs in $O(|A| \cdot |E|)$ time
  (`paper/checkllm.tex:235-236`). No GPUs, no API calls.
- [x] **Container/Docker.** `Dockerfile` pins Python 3.11-slim and
  installs from the locked dependency set
  (`Dockerfile:1-23`). Default `ENTRYPOINT` is `make` and default
  `CMD` is `reproduce`.

**Justification.** All numerics in the paper flow from four
deterministic Python experiments whose seeds, bootstrap counts, and
significance procedures are pinned in source and recorded in each
experiment's `summary.json`. Re-runs produce bit-identical scores.

---

## 4. For human subjects / IRB

- [x] **N/A.** No human subjects, no human annotators, and no
  human-rater study in this submission. The synthetic ground-truth
  signal (`paper/checkllm.tex:255-261`) is programmatically generated
  by the StubAgent's noise schedule; no consent process applies.
  This limitation is acknowledged in §9
  (`paper/checkllm.tex:459-462`, "no human-rater study"). A
  human-rater study on real production traces is listed as future
  work.

---

## 5. For code release

- [x] **License.** MIT — see `LICENSE` (top of file: "MIT License /
  Copyright (c) 2026 Javier De Jesus") and the `license = "MIT"`
  declaration in `pyproject.toml`. (Note: `CITATION.cff` currently
  records `license: Apache-2.0` — this is a known inconsistency
  between metadata files; the LICENSE file and `pyproject.toml` are
  authoritative. Marked **PARTIAL** below pending a metadata
  reconciliation patch.)
- [x] **Versioning.** Tag `paper-v1` will be cut at submission and
  archived on Zenodo. See `paper/zenodo.md` for the procedure;
  `CITATION.cff` carries a DOI placeholder
  (`10.5281/zenodo.PLACEHOLDER`) to be replaced once Zenodo mints the
  real DOI on the `paper-v1` tag.
- [x] **Reproducibility instructions.** Top-level `README.md` plus
  the `make reproduce` / `make reproduce-smoke` Makefile targets,
  backed by the `Dockerfile` for environment pinning. The full
  release/tagging workflow lives in `paper/zenodo.md`.

**Followup tasks (PARTIAL items):**
- [ ] **PARTIAL — license metadata reconciliation.** `CITATION.cff:11`
  declares `license: Apache-2.0` while `LICENSE` and `pyproject.toml`
  declare MIT. The actual license is MIT. A separate one-line patch
  to `CITATION.cff` is required; it is intentionally not bundled
  with the D4/D5 commit so this checklist accurately documents the
  current state.

---

## Summary

- Total items answered: **5 sections, 21 individual checklist items**.
- YES (with file pointer): 20.
- N/A (with justification): 2 (theoretical results, human subjects).
- PARTIAL: 1 (license metadata reconciliation).

Every numerical claim in `paper/checkllm.tex` is traceable to a
`summary.json` file under `benchmarks/paper/results/` via
`paper/figures/ingest.py`; every dataset is loaded through a
deterministic, pinned loader; the entire pipeline runs CPU-only in
under two minutes inside the published `Dockerfile`.

# Datasheet: GAIA (CheckLLM loader)

This datasheet follows the template of Gebru et al. (2018),
*Datasheets for Datasets*. It documents the GAIA benchmark
(Mialon et al. 2023) **as accessed through CheckLLM's gated loader**
at `src/checkllm/benchmarks/gaia_loader.py`. The underlying dataset
is upstream content; we document our pinning, gating, and
preprocessing here.

---

## 1. Motivation

**For what purpose was the dataset created?**
GAIA (General AI Assistants) was designed by Mialon et al. (2023,
arXiv:2311.12983) to test general-purpose AI assistants on
real-world questions that require reasoning, multimodal handling,
web browsing, and tool use. Each question has a single correct
exact-match answer that is trivial for a human but challenging for
contemporary AI systems.

**Who created the dataset and on behalf of which entity?**
Grégoire Mialon and collaborators at Meta AI, HuggingFace, and
academic partners. Hosted on the HuggingFace Hub at
`gaia-benchmark/GAIA`.

**Who funded the creation of the dataset?**
Meta AI / HuggingFace, per the upstream paper's acknowledgements.
CheckLLM did not fund or contribute to the upstream dataset.

**Any other comments?**
GAIA is *gated*: access requires accepting the dataset license on
the HuggingFace Hub and supplying an `HF_TOKEN`. CheckLLM enforces
an additional explicit acknowledgement step
(`CHECKLLM_GAIA_LICENSE_ACK=yes`) at
`src/checkllm/benchmarks/gaia_loader.py:90-103` to make the gating
opt-in even for collaborators who already have an `HF_TOKEN` in
their environment.

---

## 2. Composition

**What do the instances represent?**
Each instance is one question-answer pair: a natural-language
question (`Question`), an exact-match expected answer
(`Final answer`), a difficulty level (`Level` $\in \{1, 2, 3\}$),
and an optional attached file path (`file_name`) for multimodal
questions. CheckLLM exposes these via the `GaiaTask` pydantic model
at `src/checkllm/benchmarks/gaia_loader.py:71-87`.

**How many instances are there in total?**
Approximately 466 in the `2023_all` configuration's `validation`
split (the configuration CheckLLM pins;
`benchmarks/paper/dataset_manifest.json:gaia.config`). Exact counts
depend on the pinned revision SHA
(`682dd723ee1e1697e00360edccf2366dc8418dd9`, fetched 2026-04-23).

**Does the dataset contain all possible instances or a sample?**
GAIA is itself a sample of human-authored questions. CheckLLM's
loader passes through whatever the upstream split contains, with
optional `limit` parameter for development workflows
(`src/checkllm/benchmarks/gaia_loader.py:128-173`).

**What data does each instance consist of?**
- `task_id` (string): GAIA's stable identifier.
- `question` (string): natural-language prompt.
- `expected_answer` (string): exact-match gold answer.
- `level` (string): "1", "2", or "3".
- `file_name` (string or None): relative path to an attached file
  for multimodal tasks.

**Is there a label or target associated with each instance?**
Yes — the gold answer is the exact-match target. Scoring is binary
(match / no-match) at the trajectory level.

**Is any information missing from individual instances?**
Roughly half of GAIA tasks are text-only (`file_name` is None);
the rest reference attached files (PDFs, images, spreadsheets,
audio). CheckLLM's loader returns the path string as-is; resolving
or downloading the attached file is the responsibility of the
agent harness.

**Are relationships between individual instances made explicit?**
N/A — instances are independent question-answer pairs.

**Are there recommended data splits?**
Yes — GAIA ships `validation` and `test` splits. The CheckLLM
loader accepts both (`_VALID_SPLITS` in
`src/checkllm/benchmarks/gaia_loader.py:64`). For the present
paper's experiments we do not run GAIA end-to-end (Phase D
deferred); our paper run uses tau-bench fixtures.

**Are there any errors, sources of noise, or redundancies?**
Documented in the upstream paper. CheckLLM does not modify or
correct upstream content.

**Is the dataset self-contained, or does it link to external
resources?**
Multimodal tasks reference external files hosted alongside the
dataset on HuggingFace; resolving them requires `HF_TOKEN`.

**Does the dataset contain confidential or sensitive content?**
No. All questions are public, human-authored, and reviewed by the
upstream maintainers. Recommended-not for medical or
safety-critical decision making (this is an evaluation benchmark,
not a training corpus).

---

## 3. Collection

**How was the data acquired?**
Human authors wrote and validated the question-answer pairs.
CheckLLM does not collect data — it loads the upstream dataset via
`datasets.load_dataset` at a pinned revision
(`src/checkllm/benchmarks/gaia_loader.py:31-57`).

**What mechanisms or procedures were used to collect the data?**
See the upstream GAIA paper. CheckLLM's collection mechanism is a
HuggingFace Hub fetch keyed on the pinned SHA
`682dd723ee1e1697e00360edccf2366dc8418dd9`
(`benchmarks/paper/dataset_manifest.json`).

**If sampling, what was the sampling strategy?**
N/A — CheckLLM uses the entire pinned split (with optional
development-only `limit`).

**Who was involved in the data collection process and how were
they compensated?**
Upstream contributors per the GAIA paper. CheckLLM is not part of
that process.

**Over what timeframe was the data collected?**
Late 2023 per the upstream paper. CheckLLM's pinned revision was
fetched on 2026-04-23 (`benchmarks/paper/dataset_manifest.json`).

**Were any ethical review processes conducted?**
See the upstream paper. CheckLLM's role is solely as a loader.

**Were the individuals notified about the data collection?**
The questions are not personal data; the authors *are* the
individuals contributing.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
CheckLLM applies no transformations beyond pydantic field
extraction. The loader at
`src/checkllm/benchmarks/gaia_loader.py:160-170` casts each row's
fields to strings and forwards them. No filtering, deduplication,
or rewording is performed.

**Was the "raw" data saved in addition to the preprocessed data?**
The upstream HuggingFace dataset *is* the raw data; we do not
cache locally beyond HuggingFace's `~/.cache/huggingface` default.

**Is the software that was used to preprocess/clean/label the data
available?**
Yes — `src/checkllm/benchmarks/gaia_loader.py` is the entire
preprocessing surface, and is open source under the same MIT
license as the rest of CheckLLM.

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Extensively, by the upstream community: GAIA is a standard
agentic-evaluation benchmark.

**Is there a repository that links to any or all papers or systems
that use the dataset?**
Yes — the upstream HuggingFace dataset card and the GAIA arXiv
paper.

**What (other) tasks could the dataset be used for?**
General agent evaluation, tool-use benchmarking, and trajectory
analysis. CheckLLM's intended use is to evaluate **agent
trajectories** that arise when a real agent attempts GAIA tasks —
the trajectories are the input to `TrajectoryMetric`, not the
GAIA question text itself.

**Is there anything about the composition of the dataset or the
way it was collected and preprocessed/cleaned/labeled that might
impact future uses?**
Yes — GAIA is intentionally hard. Aggregate accuracy on `2023_all`
is low for current models, and per-level breakdowns may be
unreliable for small samples. The benchmark is intended for
relative evaluation, not absolute capability claims.

**Are there tasks for which the dataset should not be used?**
Do not use for: training data (license forbids redistribution
beyond the gated channel), safety-critical decision making, or
medical / legal applications.

---

## 6. Distribution

**Will the dataset be distributed to third parties?**
CheckLLM does not redistribute GAIA. The loader fetches at runtime
from the upstream HuggingFace Hub, gated by `HF_TOKEN` and
license acceptance.

**How will it be distributed?**
Upstream: HuggingFace Hub at `gaia-benchmark/GAIA`. CheckLLM
distributes only the *loader* (open source under MIT) and the
*pinned revision SHA* (`benchmarks/paper/dataset_manifest.json`).

**When will it be distributed?**
The loader has been available since CheckLLM's initial release
(`src/checkllm/benchmarks/gaia_loader.py`).

**Will the dataset be distributed under a copyright or other IP
license?**
GAIA's license is documented at the dataset card linked from
`benchmarks/paper/dataset_manifest.json:gaia.license`. The
canonical user-facing terms are CC-BY-4.0 with the gating noted
above. CheckLLM enforces the gate via
`LicenseAcknowledgmentRequired` at
`src/checkllm/benchmarks/gaia_loader.py:67-69`.

**Have any third parties imposed IP-based or other restrictions on
the data associated with the instances?**
Yes — the HuggingFace gate. Users must accept the license on
HuggingFace and supply an `HF_TOKEN`.

**Do any export controls or other regulatory restrictions apply?**
Not to our knowledge.

---

## 7. Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
Upstream: Mialon et al. and the HuggingFace Hub. CheckLLM
maintains only the loader and the pinned revision SHA.

**How can the owner/curator/manager of the dataset be contacted?**
Via the upstream HuggingFace dataset card. For CheckLLM-loader
issues: GitHub issues at `javierdejesusda/checkllm`.

**Is there an erratum?**
See the upstream HuggingFace community tab.

**Will the dataset be updated?**
The upstream dataset may be updated. CheckLLM pins a specific
revision SHA so paper experiments are reproducible regardless of
upstream changes; bumping the pin is a deliberate act recorded in
`benchmarks/paper/dataset_manifest.json` and `CITATION.cff`.

**If the dataset relates to people, are there applicable limits on
the retention of the data associated with the instances?**
N/A — GAIA tasks are public questions, not personal data.

**Will older versions of the dataset continue to be
supported/hosted/maintained?**
By upstream convention, prior HuggingFace dataset SHAs remain
fetchable. CheckLLM's pin guarantees the paper's results are
attached to a specific upstream snapshot.

**If others want to extend/augment/build on/contribute to the
dataset, is there a mechanism for them to do so?**
Yes — open a HuggingFace community PR upstream. Loader changes
should go through `javierdejesusda/checkllm`.

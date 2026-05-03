# Zenodo release procedure (paper-v1)

This document describes how the camera-ready CheckLLM paper artefact
is archived on Zenodo. The procedure is run by a human maintainer at
submission time; this file is the operational checklist.

## 1. What gets archived

A push of the git tag `paper-v1` to GitHub triggers Zenodo (when the
GitHub <-> Zenodo link is enabled for this repository) to ingest the
**entire repository at that tag** as a single archived release. The
release archive includes, at minimum:

- `paper/checkllm.tex` and `paper/figures/generated_tables.tex`.
- `paper/bibliography.bib` (frozen at D1).
- `paper/reproducibility_checklist.md` and `paper/datasheets/`.
- All four experiment summary JSONs under
  `benchmarks/paper/results/{01_controlled_noise,02_metric_vs_truth,03_ablation,04_vs_deepeval}/summary.json`.
- All experiment scripts under
  `benchmarks/paper/experiments/{01,02,03,04}_*.py`.
- `Makefile`, `Dockerfile`, `requirements.lock`, `pyproject.toml`.
- `CITATION.cff` (with the freshly-minted DOI inserted post-tag — see
  step 2 below).

The Zenodo record is permanent and citable; its DOI is what
downstream papers cite.

## 2. How to mint the DOI

Run, from a clean checkout of `main` with all D1-D5 work merged:

```bash
git tag -a paper-v1 -m "Paper v1: arXiv submission"
git push origin paper-v1
```

If the GitHub-Zenodo link is enabled for the repository, Zenodo
ingests the tag automatically and returns a DOI of the form
`10.5281/zenodo.<numeric-id>` within a few minutes. Copy that DOI,
then:

1. Edit `CITATION.cff`: replace `10.5281/zenodo.PLACEHOLDER` with the
   real DOI in the `identifiers:` block.
2. Edit `paper/checkllm.tex`: the conclusion (search for
   "DOI placeholder") will be replaced with the real DOI string. If
   a `\zenodoDoi` macro has been introduced by then, edit the macro
   definition; otherwise edit the literal string and re-run
   `python paper/figures/ingest.py` if needed (it is idempotent).
3. Commit on `main` with a message like
   `paper: pin Zenodo DOI for paper-v1`.

The `PLACEHOLDER` literal is the only DOI string in the repository,
so a single grep for `zenodo.PLACEHOLDER` is sufficient to find every
spot that needs updating.

## 3. What to verify before tagging

Run this checklist locally before pushing the tag. Each item is
machine-checkable and should be confirmed green.

- [ ] All four `summary.json` files exist:
  - `benchmarks/paper/results/01_controlled_noise/summary.json`
  - `benchmarks/paper/results/02_metric_vs_truth/summary.json`
  - `benchmarks/paper/results/03_ablation/summary.json`
  - `benchmarks/paper/results/04_vs_deepeval/summary.json`
- [ ] `python paper/figures/ingest.py` runs cleanly and is
  idempotent (a second run produces a byte-identical
  `paper/figures/generated_tables.tex`).
- [ ] `python paper/lint.py` exits 0.
- [ ] `pytest tests/benchmarks/` passes.
- [ ] `docker build .` succeeds locally (verifies the pinned
  `Dockerfile` + `requirements.lock` combination still installs).
- [ ] `git status` is clean (no uncommitted edits to the paper or
  manifests).
- [ ] `python -c "import yaml; yaml.safe_load(open('CITATION.cff'))"`
  returns without error.

If any item fails, do not tag — fix the underlying issue and rerun
this checklist.

## 4. Post-tag follow-ups

- Update arXiv submission with the Zenodo DOI in the abstract / data
  availability statement.
- Update `README.md` with a "Cite this artefact" badge pointing at
  the Zenodo record.
- Confirm the Zenodo record's title, authors, license, and keywords
  match `CITATION.cff` exactly. Zenodo derives these from the cff
  file when present; if it does not, edit them in the Zenodo UI to
  match.

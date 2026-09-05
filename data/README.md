# Data

This directory contains small, curated benchmark metadata that is version-controlled alongside the project. It does not contain large datasets, downloaded corpora, or generated evaluation outputs.

## What Belongs Here

- Small metadata files describing benchmark datasets: names, versions, sources, licenses, and download instructions.
- Manually curated evaluation sets that are small enough to review and audit by hand.
- Reference files used directly in reproducible experiments (e.g., a small gold-standard QA set with fewer than a few hundred examples).

## What Does Not Belong Here

- Large corpora, knowledge bases, or document collections. These must be downloaded locally and excluded by `.gitignore`.
- Generated or synthetic datasets produced by LLM calls. These must be reproducible from their generation script, not stored directly.
- Raw outputs from evaluation runs. These belong in experiment directories under `research/experiments/`.

## Directory Structure Convention

```
data/
  README.md                    This file.
  benchmarks/                  Metadata and references for external benchmark datasets.
  eval_sets/                   Small, hand-curated evaluation sets committed directly.
```

These subdirectories will be created when the first dataset is formally introduced.

## Large File Policy

Files exceeding approximately 1 MB should not be committed without explicit justification. Files exceeding 10 MB must not be committed. Use `.gitignore` rules for `data/raw/`, `data/processed/`, `data/generated/`, and `data/downloads/` to prevent accidental commits of large files.

## Data Provenance

Every dataset used in an experiment must have its source, version, license, and download method documented in `data/benchmarks/` or within the experiment definition in `research/experiments/`.

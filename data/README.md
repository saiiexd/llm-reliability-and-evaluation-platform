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
    evaluation_methodology_baseline_v1.json
                                24 hand-written test cases (exact factual questions,
                                paraphrasable answers, and ambiguous/insufficient-reference
                                questions) used to validate the evaluation layer's
                                methodology -- exact match, semantic similarity, and
                                LLM-as-a-judge. This is not a representative benchmark of
                                general LLM capability; it exists only to exercise the
                                evaluation infrastructure against varied, deliberately
                                small, hand-auditable cases. Load with
                                ``llm_reliability.evaluation.load_dataset_from_json``.
```

`benchmarks/` will be created when an external benchmark dataset is formally introduced.

## Large File Policy

Files exceeding approximately 1 MB should not be committed without explicit justification. Files exceeding 10 MB must not be committed. Use `.gitignore` rules for `data/raw/`, `data/processed/`, `data/generated/`, and `data/downloads/` to prevent accidental commits of large files.

## Data Provenance

Every dataset used in an experiment must have its source, version, license, and download method documented in `data/benchmarks/` or within the experiment definition in `research/experiments/`.

# Scripts

This directory contains reproducible utility scripts for the project.

## Purpose

Scripts here automate clearly defined, repeatable tasks that support the research or engineering workflow. Every script in this directory must have a documented purpose, expected inputs and outputs, and must be runnable without modification in any properly configured project environment.

## What Belongs Here

- Data preparation scripts (e.g., downloading a benchmark dataset to a local path, formatting data into a standard schema).
- Evaluation runner scripts (e.g., running a defined experiment configuration end-to-end).
- Environment setup or validation scripts (e.g., checking that required tools and versions are present).
- Utility scripts that support reproducibility (e.g., seeding, result formatting, report generation).

## What Does Not Belong Here

- One-off scripts written to solve an immediate problem and never intended to be reused.
- Scripts that only work on a specific machine due to hard-coded local paths.
- Scripts that require manual editing before each use.
- Scripts without a documented purpose.

## Current Status

- `run_core_engine_example.py` -- a minimal, read-through API demonstration of the Core
  Evaluation Engine (dataset, deterministic mock adapter, baseline evaluator, structured
  run result). It is a fixed example, not a parameterized automation script, so it takes
  no command-line arguments; it requires no network access or API credentials. Run with
  `python scripts/run_core_engine_example.py`.
- `run_experiment_example.py` -- a minimal, read-through API demonstration of the
  Experiment System (experiment definition, execution via the existing evaluation engine,
  local JSON persistence, and reload). Persists to a temporary directory removed on exit,
  so it leaves nothing behind; it takes no command-line arguments and requires no network
  access or API credentials. Run with `python scripts/run_experiment_example.py`.
- `run_evaluation_layer_example.py` -- a minimal demonstration of the Evaluation and
  Reliability Layer: loads `data/eval_sets/evaluation_methodology_baseline_v1.json`,
  evaluates deterministic mock responses with exact match, semantic similarity (fake
  backend), and LLM-as-a-judge (fake judge) at once, including an intentional
  disagreement between evaluators, then persists, reloads, and prints an evaluation
  summary and reliability findings. Requires no network access or API credentials. Run
  with `python scripts/run_evaluation_layer_example.py`.
- `run_rag_example.py` -- a minimal demonstration of the RAG execution and retrieval
  evaluation layer: builds a tiny document corpus and chunk index, retrieves with a
  deterministic fake embedding model, generates with the deterministic mock adapter, and
  produces an evidence-based diagnosis (successful grounded execution, retrieval failure,
  or generation failure despite relevant context) for each of three controlled test
  cases, after persisting and reloading the experiment. Requires no network access or API
  credentials. Run with `python scripts/run_rag_example.py`.

- `seed_live_experiments.py` -- seeds `./experiments` (or a given path) with the simulated
  demo experiments and runs used by the console. `python scripts/seed_live_experiments.py [dir]`.
- `export_frontend_fixtures.py` -- regenerates `frontend/src/data/fixtures/` (demo mode data)
  from the same simulated scenarios through the real persistence and analytics code.
- `build_squad_subset.py` -- builds `data/benchmarks/squad_v1_1_dev_subset_v1.json`
  deterministically from the downloaded SQuAD v1.1 dev file (instructions in its docstring).
- `run_reliability_study.py` -- runs EXP-001 against local Ollama models (real inference;
  resumable; refuses to mix environments). See `research/experiments/EXP-001_local_rag_reliability/`.
- `analyze_reliability_study.py` -- analyses EXP-001 runs (Wilson intervals, McNemar tests,
  retrieval at the run's K, Cohen's kappa, consistency) and writes results files.
- `export_annotation_sheet.py` -- writes a blind human-annotation sheet for answer correctness
  (one row per distinct answer to a question, shuffled with a fixed seed), a key mapping each row
  to every run and test case it represents (with a `guideline_example` flag for the 15 rows quoted
  as worked examples in the guideline), and an overlap sheet (>= 20%, examples excluded) for a
  second annotator. Creates no labels. See `research/annotation/`.
- `analyze_human_agreement.py` -- reads filled sheets, the key and the store; reports
  inter-annotator Cohen's kappa and, per correctness evaluator and arm, accuracy, precision,
  recall and specificity (Wilson 95% intervals) and kappa against the human labels, with every
  exclusion counted. Guideline-example rows are excluded from the primary result and the
  inter-annotator kappa, and included in a sensitivity result. Refuses to run while no label is
  filled in.
- `build_paraphrase_set.py` -- generates, once, the frozen question-paraphrase set used by
  EXP-002 (`data/benchmarks/squad_v1_1_dev_subset_v1_paraphrases_v1.json`) with a local model,
  recording the prompt, model digest and automatic checks. Needs Ollama.
- `run_robustness_study.py` -- runs EXP-002 (RQ5: reversed passage order, an irrelevant passage,
  reworded questions, a free-form prompt, plus a repeated baseline as the noise floor) against
  Ollama. Resumable; `--limit N --store DIR` for a pilot.
- `analyze_robustness_study.py` -- the pre-registered EXP-002 analysis: verdict flips against the
  baseline and the noise floor, McNemar tests, diagnosis consistency, retrieval identity checks.
- `run_rejudge_study.py` -- EXP-003: replays the recorded answers of four runs (no new
  generation) and judges them with rubric v1 and v2, then writes the analysis. `--analyze-only`
  rewrites the tables from existing runs.

## Script Requirements

Each script must:
1. Include a module-level docstring explaining its purpose, required inputs, expected outputs, and any side effects.
2. Accept its inputs as command-line arguments rather than hard-coded values.
3. Exit with a non-zero code on failure.
4. Be listed in this README when added, with a one-line description.

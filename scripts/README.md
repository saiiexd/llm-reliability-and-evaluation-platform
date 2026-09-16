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

## Script Requirements

Each script must:
1. Include a module-level docstring explaining its purpose, required inputs, expected outputs, and any side effects.
2. Accept its inputs as command-line arguments rather than hard-coded values.
3. Exit with a non-zero code on failure.
4. Be listed in this README when added, with a one-line description.

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

No scripts exist yet. Scripts will be added when a specific, reproducible task requires automation.

## Script Requirements

Each script must:
1. Include a module-level docstring explaining its purpose, required inputs, expected outputs, and any side effects.
2. Accept its inputs as command-line arguments rather than hard-coded values.
3. Exit with a non-zero code on failure.
4. Be listed in this README when added, with a one-line description.

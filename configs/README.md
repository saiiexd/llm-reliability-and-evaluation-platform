# Configs

This directory will contain reproducible configuration files for experiments and system components.

## Purpose

Every experiment and system configuration that can be reproduced must be defined here in a version-controlled, human-readable format. Configuration files are the authoritative source for the exact settings used in a given experiment run.

## What Will Belong Here

- Experiment configuration files (e.g., model selection, prompt templates, retrieval settings, evaluation metric selection).
- System configuration files for platform components (e.g., logging settings, API endpoint configuration).
- Environment-agnostic configuration that does not contain secrets.

## What Does Not Belong Here

- Secret values, API keys, or credentials. These must never be committed; use environment variables or a secrets manager.
- Machine-specific paths or environment-dependent values. Configuration should be portable.
- Configuration files for systems that do not yet exist in this repository.

## Current Status

No configuration files exist yet. Configuration files will be created when the corresponding system component or experiment requires them. Do not create configuration files speculatively.

## Format

YAML is the preferred format for human-authored configuration. JSON is acceptable for programmatically generated configuration. TOML is reserved for Python packaging metadata (`pyproject.toml`).

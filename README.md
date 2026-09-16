# LLM Reliability & Evaluation Platform

This repository contains the LLM Reliability & Evaluation Platform: a system for systematically testing, evaluating, comparing, and diagnosing LLM applications.

## What This Project Is

This is not a chatbot, a generic RAG application, or an autonomous agent framework. It is a research and engineering platform with three distinct roles:

1. **LLM Testing Framework** - Structured, repeatable evaluation of LLM and LLM-based system outputs against defined quality dimensions, using controlled datasets and measurable metrics.

2. **LLM Observability and Diagnostics Platform** - Tooling to trace, inspect, and diagnose LLM application behavior: where it fails, how it fails, and why. Observability is treated as a first-class engineering concern, not an afterthought.

3. **Research Experimentation Environment** - A controlled environment for running scientific experiments on LLM reliability, faithfulness, hallucination, retrieval quality, and consistency. Experiments are reproducible, hypothesis-driven, and documented with failure analysis.

## Central Project Principle

This system exists to scientifically investigate whether AI applications can be trusted, where they fail, and why they fail. Trust in LLM systems must be earned through measurement, not assumed.

## Evaluation Dimensions

The platform evaluates LLM systems across the following dimensions:

- **Answer Quality** - Correctness, completeness, relevance, and fluency of generated answers.
- **Reliability and Hallucination** - Frequency and pattern of factual errors and unsupported claims.
- **Faithfulness and Groundedness** - Whether answers are grounded in retrieved or provided context.
- **RAG and Retrieval Quality** - Precision, recall, and relevance of retrieval in RAG pipelines.
- **Consistency and Robustness** - Stability of outputs across paraphrase variations and repeated runs.
- **Performance and Observability** - Latency, cost, token usage, and system-level behavior under load.
- **Experiment Comparison** - Systematic A/B comparison of models, prompts, retrievers, and configurations.

## RAG Execution Model

The platform treats the following pipeline as its core unit of analysis:

```
Question -> Retrieval -> Retrieved Context -> Generation -> Answer
```

Each step in this pipeline is independently measurable and diagnosable.

## Development Philosophy

The project follows a research-first, incremental development approach:

- Research first: understand the problem domain before building solutions.
- Reproduce small experiments: start with published benchmarks and known results.
- Establish baselines: no metric is meaningful without a reference point.
- Build incrementally: add components only when justified by research or engineering needs.
- Maintain reproducibility: every experiment must be repeatable from its definition.
- Avoid unnecessary technologies: do not introduce tools without a clear, specific reason.

## Current Status

The repository is in the **research and foundation stage**. The first production component, the Core Evaluation Engine, now exists (see "Current Implementation" below). The current priorities are:

- Understanding the evaluation literature.
- Identifying the smallest experiments that produce meaningful results.
- Establishing ground truth datasets and baselines.
- Building the foundational evaluation framework one measurable capability at a time.

Further production platform components will be introduced only when a clear research or engineering requirement justifies them.

## Current Implementation: Core Evaluation Engine

`src/llm_reliability/evaluation/` implements the first controlled execution path: running a
deterministic dataset of test cases against a model or application, then scoring the
results with one or more evaluators. The execution flow is:

```
Dataset -> TestCase -> ExecutionRequest -> ModelAdapter -> ModelResponse
    -> Evaluator(s) -> EvaluationResult -> RunResult
```

**What it currently supports:**

- `TestCase` / `Dataset`: validated, deterministic evaluation inputs with stable ids.
  Reference answers are optional, since reference-free and model-based evaluation are
  explicitly in scope for later milestones.
- `ModelAdapter`: a provider-agnostic interface between the engine and any model or
  application. `MockAdapter` is a deterministic, network-free fake used for tests and
  local development; it never calls an external service and its output must not be
  interpreted as representing real model behavior.
- `Evaluator`: a provider-agnostic interface for scoring a response against a test case.
  `ExactMatchEvaluator` is the initial baseline (case-insensitive exact string match) used
  to validate the execution infrastructure -- it is not a general-purpose answer-quality
  metric, and it reports "not applicable" rather than a failure when no reference answer
  exists.
- `EvaluationRunner`: executes a dataset against an adapter and a set of evaluators,
  preserving test-case order and ids, and isolating generation or evaluation failures to
  the affected test case (recorded explicitly, never silently discarded).
- `RunResult`: an in-memory, structured record of the full run, with deterministic JSON
  serialization (`RunResult.to_json()`) for later inspection or persistence.

**What it deliberately does not support yet:**

- No RAG execution, retrieval, or retrieved-context handling.
- No LLM-as-a-judge, hallucination detection, faithfulness, or semantic-similarity
  evaluators -- `ExactMatchEvaluator` is a baseline only.
- No observability, tracing, or reliability/statistical analysis.
- No frontend, API server, authentication, or deployment infrastructure.
- No real LLM provider adapter (only the deterministic mock adapter exists at this stage).

See `research/notes/core_evaluation_engine_execution_model.md` for the conceptual model
behind this design, and `scripts/run_core_engine_example.py` for a runnable example.

## Current Implementation: Experiment System

`src/llm_reliability/experiments/` builds a reproducible experiment on top of the Core
Evaluation Engine, without duplicating it:

```
ExperimentDefinition (DatasetVersion + ModelConfig + EvaluatorConfig(s), fingerprinted)
    -> ExperimentManager.execute_experiment (delegates to EvaluationRunner)
    -> ExperimentRun (status, timestamps, a full configuration snapshot, and RunResult)
    -> ExperimentRepository (local JSON persistence in this milestone)
```

**What it currently supports:**

- `DatasetVersion`: a content-addressed wrapper around the existing `Dataset`; its
  `version_id` is a SHA-256 fingerprint of test case content, independent of the dataset's
  human-readable name.
- `EvaluatorConfig`: a stable, serializable record of an evaluator's name and parameters,
  built from a live evaluator via a new `Evaluator.get_config()` hook -- never the
  evaluator object itself.
- `ExperimentDefinition`: combines a `DatasetVersion`, the existing `ModelConfig`, and one
  or more `EvaluatorConfig` records, with a deterministic `config_fingerprint` derived only
  from those three (never from identifiers, names, metadata, or timestamps).
- `ExperimentManager`: creates and persists experiment definitions, and executes them by
  delegating to the unmodified `EvaluationRunner`, then packages the result into an
  `ExperimentRun`. Contains no evaluation logic of its own.
- `ExperimentRun`: one execution of a definition, with its own `run_id`, timestamps,
  explicit `RunStatus` (`succeeded` / `partially_succeeded` / `failed`), and a full
  snapshot of the configuration used, so a persisted run is completely inspectable without
  re-executing anything.
- `LocalFileExperimentRepository`: an `ExperimentRepository` implementation storing one
  human-readable JSON file per experiment and per run, with atomic writes and no silent
  overwriting of an existing id. Loading validates persisted data (including a fingerprint
  integrity check) back into real domain objects rather than returning raw dictionaries.

**What it deliberately does not support yet:**

- No database-backed persistence (PostgreSQL or otherwise) -- only local JSON files.
- No multi-model orchestration (one model/application configuration per experiment).
- No reliability, consistency, or statistical analysis across repeated runs.
- No dashboard, frontend, or API layer.
- No registry for reconstructing evaluator/adapter instances from persisted
  configuration -- live instances are always supplied by the caller at execution time.

See `research/notes/experiment_reproducibility_model.md` for the full reproducibility
model, and `scripts/run_experiment_example.py` for a runnable example.

**Setup:**

```
pip install -e ".[dev]"
pytest
python scripts/run_core_engine_example.py
python scripts/run_experiment_example.py
```

## Repository Structure

```
research/           Research notes, paper references, and reproducible experiments.
  notes/            Learning notes and technical understanding documents.
  papers/           Paper references and legally available summaries or links.
  experiments/      Experiment definitions, configurations, results, and failure analysis.
learning/           Temporary educational implementations for concept understanding.
  neural_network_from_scratch/
  transformer_from_scratch/
src/                Platform source code.
  llm_reliability/evaluation/   Core Evaluation Engine (see "Current Implementation" above).
  llm_reliability/experiments/  Experiment System (see "Current Implementation" above).
tests/              Test suite.
  evaluation/       Tests for the Core Evaluation Engine.
  experiments/      Tests for the Experiment System.
configs/            Reproducible experiment and system configuration.
data/               Benchmark metadata (small, curated, version-controlled).
scripts/            Reproducible utility scripts.
```

## Python Version

Python 3.13 is required. See `pyproject.toml` for the authoritative version constraint.

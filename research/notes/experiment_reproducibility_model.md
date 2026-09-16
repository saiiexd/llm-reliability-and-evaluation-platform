# Experiment System: Reproducibility Model

## Purpose of This Note

This note documents the conceptual model implemented by
`src/llm_reliability/experiments/`, which turns a single, isolated
evaluation run (the Core Evaluation Engine, documented in
`core_evaluation_engine_execution_model.md`) into a formally defined,
reproducible experiment. It explains the distinct concepts involved, how
reproducibility is actually achieved in this implementation, and what is
deliberately deferred to later milestones.

## The Conceptual Model

```
ExperimentDefinition
    -> DatasetVersion            (content-addressed dataset)
    -> ModelConfig                (reused from the evaluation engine)
    -> EvaluatorConfig(s)          (stable evaluator identifiers + parameters)
    -> config_fingerprint          (derived from the three items above)

ExperimentManager.execute_experiment(definition, adapter, evaluators)
    -> delegates to EvaluationRunner (unchanged from the Core Evaluation Engine)
    -> ExperimentRun
        -> run_id, status, started_at, finished_at
        -> a full snapshot of dataset_version, model_config, evaluator_configs
        -> RunResult (the complete per-test-case outcome, reused as-is)
```

## Distinguishing the Concepts

- **Experiment** (`ExperimentDefinition`): what should be tested. Identified
  by a caller-supplied, stable `experiment_id` (explicit, like
  `TestCase.id`, not auto-generated -- an experiment is something a
  researcher names and refers back to). Combines a `DatasetVersion`, a
  `ModelConfig`, and one or more `EvaluatorConfig` records into one
  reproducible unit. An experiment's `name`, `description`, and `metadata`
  are descriptive and can be anything; they do not affect what the
  experiment actually does.

- **Experiment Run** (`ExperimentRun`): one execution of an experiment.
  Identified by its own `run_id`, generated fresh (a random identifier,
  since run identity only needs to be unique, not reproducible) unless the
  caller supplies one explicitly. Two runs of the same experiment always
  share `experiment_id` and, if the experiment definition has not changed,
  `config_fingerprint`; they never share `run_id`. This is what makes
  repeated execution, and later consistency/statistical analysis across
  repeated runs, possible: the platform can always tell "the same
  experiment, run twice" apart from "two different experiments."

- **Dataset Version** (`DatasetVersion`): a content-addressed wrapper around
  the existing `Dataset`/`TestCase` abstractions from the Core Evaluation
  Engine. `version_id` is a SHA-256 fingerprint of the dataset's test cases
  (id, input, reference answer, metadata), computed independently of the
  dataset's human-readable `name`. Renaming a dataset does not change its
  version id, and editing its content always does. This is what lets a
  later reader determine exactly which dataset content an experiment run
  used, even if the "same" dataset (by name) is later edited elsewhere.

- **Model/Application Configuration** (`ModelConfig`): reused directly and
  unmodified from the Core Evaluation Engine. No new configuration model
  was introduced because the existing one already met this milestone's
  needs: it is a plain, serializable dataclass with explicit optional
  fields (so "unspecified" is distinguishable from an explicit value), and
  it already excludes anything provider-specific that does not have a
  first-class field (`extra_params` exists for that). Experiment
  definitions and persisted results must never contain secrets, API keys,
  or credentials; the deterministic mock adapter used throughout this
  milestone needs none, by design.

- **Evaluator Configuration** (`EvaluatorConfig`): a new, small model that
  records an evaluator's stable `evaluator_name` and its serializable
  `parameters`, built from a live evaluator instance via
  `EvaluatorConfig.from_evaluator(evaluator)`, which reads
  `evaluator.name` and a new `Evaluator.get_config()` method (added to the
  Core Evaluation Engine's `Evaluator` base class, returning `{}` by
  default). The evaluator object itself is never serialized -- only its
  identity and configuration are recorded, so that when a more elaborate
  evaluator (for example, an LLM-as-a-judge evaluator with a model id and a
  prompt template) is introduced in a later milestone, its configuration
  automatically becomes part of the reproducibility record without any
  change to this model.

- **Test Case Result** (`TestCaseResult`, inside `RunResult`): unchanged
  from the Core Evaluation Engine, and reused as-is inside
  `ExperimentRun.result`. This milestone does not re-implement or wrap
  per-test-case execution; `EvaluationRunner` remains the only component
  responsible for it.

## How Reproducibility Is Achieved

1. **Content-addressed dataset identity.** A dataset's `version_id` is a
   deterministic hash of its test case content, not its name or any
   auto-incrementing counter. Two datasets (or the same dataset at two
   points in time) with identical test case content always produce the
   identical version id; any change to a test case's id, input, reference
   answer, or metadata produces a different one.

2. **A single configuration fingerprint.** `ExperimentDefinition.config_fingerprint`
   is a SHA-256 hash of the canonical JSON form of exactly three things:
   the dataset version id, the model configuration, and the evaluator
   configurations. "Canonical JSON" means keys are sorted and separators
   are fixed, so dictionary insertion order never changes the fingerprint.
   The fingerprint deliberately excludes `experiment_id`, `name`,
   `description`, `metadata`, and `created_at`: none of those affect what
   actually gets executed, so none of them should be able to change the
   fingerprint. This is verified directly by tests (`test_fingerprint.py`,
   `test_models.py`): equivalent configuration under different names
   produces identical fingerprints, and any of the three meaningful inputs
   changing produces a different one.

3. **Self-contained runs.** `ExperimentRun` stores its own copy of
   `dataset_version`, `model_config`, `evaluator_configs`, and
   `config_fingerprint` at the moment of execution, rather than only a
   reference to the experiment definition. A persisted run remains fully
   inspectable -- what was tested, on what data, with what configuration,
   and what happened -- even if the originating experiment definition is
   later changed or deleted.

4. **Integrity verification on load.** Both `DatasetVersion.from_dict` and
   `ExperimentDefinition.from_dict`/`ExperimentRun.from_dict` recompute the
   relevant fingerprint from the loaded content and compare it against the
   fingerprint stored in the file. A mismatch (whether from disk
   corruption or a hand-edited file) raises an explicit `ValueError` rather
   than silently returning a definition or run whose recorded fingerprint
   no longer matches its actual content.

5. **Explicit run status, never hidden failure.** `RunStatus` distinguishes
   `succeeded` (every test case executed without error),
   `partially_succeeded` (at least one succeeded and at least one failed),
   and `failed` (no test case succeeded, or the run could not execute at
   all). Evaluator judgments (did the answer match the reference) are
   entirely separate from execution status: a test case that generates a
   response but fails its evaluation is still a successful execution,
   because generation happened correctly and produced an inspectable
   result. Execution failures are always visible on
   `TestCaseResult.execution_error` / `ExperimentRun.error`; they are never
   silently discarded.

## Persistence Strategy

`ExperimentRepository` is an abstract interface (`save_experiment`,
`load_experiment`, `save_run`, `load_run`, `list_experiment_ids`,
`list_run_ids`); `LocalFileExperimentRepository` is its only implementation
in this milestone, storing one JSON file per experiment and per run under a
caller-supplied root directory. This choice, rather than a database, exists
to validate the domain model and reproducibility workflow first: JSON files
are human-readable, diffable, and require no additional infrastructure.
Writes are atomic (write to a temporary file, then `os.replace` into place)
so an interrupted process cannot leave a half-written record, and an
existing experiment or run id is never silently overwritten -- a repeated
`experiment_id` or `run_id` raises `ExperimentAlreadyExistsError` /
`RunAlreadyExistsError`. Because domain logic depends only on the
`ExperimentRepository` interface, a future database-backed implementation
(for example, PostgreSQL) can replace `LocalFileExperimentRepository`
without any change to `ExperimentManager` or the domain models.

## Deliberately Deferred to Later Milestones

- A database-backed `ExperimentRepository` implementation (PostgreSQL or
  otherwise).
- Any evaluator beyond the exact-match baseline; `EvaluatorConfig` exists
  precisely so later evaluators (LLM-as-a-judge, faithfulness scoring,
  semantic similarity) can be introduced without further changes to the
  experiment or persistence model.
- Multi-model orchestration (an experiment that evaluates more than one
  model/application configuration at once). The current model
  intentionally represents one model/application configuration per
  experiment, matching what `EvaluationRunner` actually supports today;
  extending this is a later, explicit decision, not an incidental
  side-effect of this milestone.
- RAG execution, retrieval, or retrieved-context handling.
- Reliability, consistency, or statistical analysis across multiple runs of
  the same experiment (this milestone provides the run identity and
  fingerprint machinery such analysis will need, but performs none of it).
- Observability, tracing, a dashboard, or a frontend/API layer.
- A registry or factory for reconstructing live `Evaluator`/`ModelAdapter`
  instances from persisted configuration. Execution always requires the
  caller to supply live instances; `EvaluatorConfig` and `ModelConfig`
  record what was used, not executable code, and `ExperimentManager`
  verifies that the instances supplied at execution time match the
  persisted configuration by name.

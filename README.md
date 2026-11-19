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

## Current Implementation: Evaluation and Reliability Layer

`src/llm_reliability/evaluation/` now supports multiple, complementary evaluation
methods for the same generated answer, and `src/llm_reliability/reliability/` aggregates
and classifies their results without collapsing them into a single score:

```
ModelResponse -> Evaluator(s): ExactMatchEvaluator, SemanticSimilarityEvaluator, LLMJudgeEvaluator
    -> EvaluationResult(s) (status, criterion, score/label where applicable, own configuration)
    -> summarize_run() -> EvaluationSummary (per-evaluator statistics, no overall score)
    -> analyze_reliability() -> ReliabilityFinding(s) (disagreement, missing evidence,
                                 evaluator failure, incorrect-per-evaluator)
```

**What it currently supports:**

- `EvaluationStatus`: an explicit, five-way status on every `EvaluationResult`
  (`success`, `skipped`, `invalid_configuration`, `execution_error`,
  `output_validation_error`), so "no judgment was possible" is never confused with "the
  answer failed."
- Evaluation criteria (`llm_reliability.evaluation.criteria`): stable identifiers for
  *what* is assessed (currently `correctness`), separate from evaluators, which describe
  *how*. The same criterion can be assessed by several evaluators.
- `ExactMatchEvaluator`: unchanged baseline, now with explicit, configurable
  normalization (`case_sensitive`, `collapse_whitespace`; see
  `llm_reliability.evaluation.normalization`) instead of fixed behavior.
- `SemanticSimilarityEvaluator`: a BERTScore-following complementary evaluator (Zhang et
  al., 2020) behind a pluggable `SimilarityBackend`. The real backend
  (`BertScoreBackend`) requires the optional `semantic` dependency group and is never
  exercised by the standard test suite (no model download in unit tests); a deterministic
  `FakeSimilarityBackend` validates the evaluator's contract instead.
- `LLMJudgeEvaluator`: asks a judge model to apply an explicit, versioned `JudgeRubric`
  (correctness against a reference, following Zheng et al., 2023), reusing the existing
  `ModelAdapter` interface as the judge-model abstraction -- no parallel adapter
  interface was introduced. Judge output is strictly parsed and validated; an invalid or
  unparseable response produces an explicit error, never a fabricated score. Any
  `ModelAdapter` (including `MockAdapter`) can serve as a deterministic fake judge in
  tests.
- `llm_reliability.reliability.summarize_run` / `analyze_reliability`: read-only
  aggregation and per-test-case flagging over an existing `RunResult`, identifying
  evaluator disagreement, missing evaluation evidence, evaluator failures, and
  per-evaluator incorrect results -- never a hallucination determination.
- A small, hand-curated benchmark (`data/eval_sets/evaluation_methodology_baseline_v1.json`,
  24 test cases) for validating this evaluation methodology, not for measuring real model
  capability.
- Evaluator configuration (normalization settings, similarity threshold/backend, judge
  model/rubric) is automatically part of the experiment configuration fingerprint from
  Prompt 2, with no change to `llm_reliability.experiments` required.

**What it deliberately does not support yet:**

- No hallucination detection, faithfulness, or groundedness evaluation (these require
  context-grounded evaluation, not yet implemented).
- No validation of LLM-as-a-judge's reliability against human annotation.
- No reliability/statistical analysis across multiple runs of the same experiment (only
  within a single run, so far).
- No real LLM provider or judge-model integration (only deterministic fakes).

See `research/notes/evaluation_and_reliability_layer.md` for the full conceptual model,
and `scripts/run_evaluation_layer_example.py` for a runnable example.

## Current Implementation: RAG Execution and Retrieval Evaluation

`src/llm_reliability/rag/` implements a controlled RAG pipeline as an explicit,
inspectable sequence of stages, and evaluates retrieval quality independently from
answer quality:

```
Question -> Retriever (embed + rank chunks) -> Retrieved Context
    -> Generation (existing ModelAdapter) -> Answer
        -> RetrievalEvaluator(s): Hit@K, Recall@K, MRR (against explicit ground truth)
        -> ContextSupportBaseline (answer-quality family; literal overlap, not faithfulness)
        -> diagnose_rag_execution -> retrieval_failure / generation_failure_despite_relevant_context
                                      / successful_grounded_execution / retrieval_evidence_unavailable
                                      / undetermined
```

**What it currently supports:**

- `Document` / `Chunk` / `FixedSizeChunker`: a deterministic, character-window chunking
  baseline with explicit, validated `chunk_size`/`chunk_overlap` configuration.
- `EmbeddingModel`: separates retrieval from any specific embedding provider.
  `FakeEmbeddingModel` is deterministic and dependency-free for tests;
  `SentenceTransformerEmbeddingModel` is the one real backend (Sentence Transformers was
  already available in this environment), loaded lazily so importing/configuring it never
  requires a model download.
- `Retriever`: embeds a fixed chunk corpus once, ranks it against each query by cosine
  similarity (`llm_reliability.rag.similarity`, no vector database), and returns the top-k
  in ranked order with stable chunk/document identifiers and real similarity scores.
- `RagPipeline`: composes retrieval with generation. It implements the existing
  `ModelAdapter` interface, so it runs through the unmodified `EvaluationRunner` and
  `ExperimentManager` with no changes to either. Retrieval evidence travels through
  `ModelResponse.provider_metadata["rag"]`; retrieval ground truth travels through the
  existing `TestCase.metadata["relevant_chunk_ids"]`. No persisted schema changed.
- `RetrievalEvaluator` / `RetrievalEvaluationResult`: a type entirely separate from
  answer-quality `EvaluationResult`, so retrieval and answer scores are never mixed.
  `HitAtKEvaluator`, `RecallAtKEvaluator`, and `MeanReciprocalRankEvaluator` are scored
  only when ground truth and retrieval evidence are both available; otherwise `SKIPPED`.
- `ContextSupportBaseline`: a narrow, literal word-overlap check between an answer and its
  retrieved context (answer-quality family, reusing the existing `Evaluator` interface
  unchanged) -- explicitly not a faithfulness or hallucination determination.
- `diagnose_rag_execution`: an evidence-based decision table combining a retrieval
  hit/miss signal with an answer-correctness signal into one of five categories, reporting
  `undetermined` rather than guessing when the evidence does not clearly support one.
- RAG configuration (embedding model, chunk size/overlap, top-k, corpus content
  fingerprint) is embedded in the experiment's `ModelConfig.extra_params`, which already
  participates in the Prompt 2 configuration fingerprint -- no fingerprinting changes
  were needed.

**What it deliberately does not support yet:**

- No semantic, hierarchical, or recursive chunking -- fixed-size only.
- No vector database or approximate nearest-neighbor index.
- No NLI-based faithfulness verification, claim extraction, or LLM-as-a-judge
  faithfulness rubric.
- No hallucination detection as a generic classifier.
- No document ingestion system, multiple embedding providers, or plugin architecture.

See `research/notes/rag_execution_model.md` for the full conceptual model, and
`tests/rag/test_end_to_end.py` for a runnable, fully controlled RAG experiment covering
every diagnosis category.

## Current Implementation: Diagnostics, Faithfulness, and Robustness Layer

`src/llm_reliability/diagnostics/` consolidates evaluation, faithfulness, and retrieval evidence
into a definitive failure diagnosis per test case without discarding secondary findings, and
evaluates behavioral consistency under controlled input perturbations:

```
Test Case + Execution Results + Retrieval Evidence
    -> assess_statements_lexically (sentence-level lexical support baseline)
    -> build_reliability_diagnostic -> Primary FailureCategory (CORRECT, CONTRADICTED,
                                        UNSUPPORTED, RETRIEVAL_FAILURE, GENERATION_FAILURE,
                                        EVALUATOR_DISAGREEMENT, INSUFFICIENT_EVIDENCE)
                                     + Preserved secondary_findings
    -> compare_robustness / summarize_robustness (consistency across input perturbations)
```

**What it currently supports:**

- `ReliabilityDiagnostic`: synthesizes all evaluators into an evidence-based primary category
  with clear precedence, while explicitly preserving all secondary observations (such as
  unsupported claims, evaluator disagreements, contradictions, or evidence gaps) in
  `secondary_findings`.
- `FailureCategory`: an attribution-specific vocabulary replacing the overloaded and misleading
  industry buzzword "hallucination".
- `assess_statements_lexically`: sentence-level segmentation and literal word overlap check
  reporting `SUPPORTED` or `UNSUPPORTED`.
- `compare_robustness` / `summarize_robustness`: compares baseline cases against controlled
  perturbations (`question_rewording`, `context_reordering`, `top_k_change`,
  `irrelevant_context_injection`, `context_modification`, `repeated_input`) to compute
  outcome consistency rates and transition matrices.
- Documented limitations: lexical matching and LLM-as-a-judge are explicitly documented as
  signal-gathering evaluation methods with known failure modes, not universal hallucination
  detectors (see `research/notes/reliability_and_robustness_layer.md`).

## Current Implementation: Research Console Frontend

`frontend/` implements an interactive, typed research console using React, TypeScript, Vite,
and Tailwind CSS, backed by statically-typed JSON fixtures exported from deterministic backend runs:

- **Fixture & Live Data Boundary**: clean typed data access layer (`frontend/src/data/dataAccess.ts`)
  with mandatory `FixtureNotice` warnings distinguishing exported fixture datasets from live runs.
- **Pages**:
  - `OverviewPage`: platform statistics, failure category distribution, method coverage, and recent runs.
  - `ExperimentsPage`: catalog of fingerprinted experiments and execution counts.
  - `ExperimentDetailPage`: deep dive into configuration fingerprints, datasets, and execution history.
  - `RunDetailPage`: run execution overview, test case tables, and failure category distributions.
  - `TestCaseDetailPage`: full diagnosis breakdown, primary vs secondary findings, evaluator score tables, retrieved context, and statement assessments.
  - `RagDiagnosticsPage`: retrieval metrics (Hit@K, Recall@K, MRR), RAG failure attributions, and context support analysis.
  - `EvaluatorComparisonPage`: multi-evaluator agreement matrices and disagreement case drilldowns.
  - `ReliabilityDiagnosticsPage`: full cross-run failure category filtering and reliability findings.
  - `RobustnessPage`: perturbation consistency rates and outcome transition matrices.
  - `NotFoundPage`: clean 404 routing with navigational return.

## Current Implementation: FastAPI REST API & Live Application Boundary

`src/llm_reliability/api/` introduces FastAPI and Pydantic v2 as the official application boundary between the React research console and the core Python research engine:

```
React Console (Live API / Fixture Mode)
    -> HTTP REST Client (frontend/src/api/client.ts)
    -> FastAPI v1 Router (/api/v1)
    -> Request Correlation & Timing Middleware (X-Request-ID, X-Response-Time-Ms)
    -> Pydantic Schemas & Sensitive Data Redaction (api_key, tokens)
    -> DiagnosticsService & AnalyticsService
    -> AdapterRegistry (Controlled, allowlisted execution)
    -> ExperimentManager & ExperimentRepository (Local atomic JSON persistence)
```

**Key Architectural Features:**
- **Versioned REST Endpoints (`/api/v1`)**: Fully typed request/response schemas for health, experiments, runs, test cases, diagnostics, RAG retrieval traces, robustness bundles, and cross-run analytics.
- **Strict Information Security**: Recursively redacts sensitive configuration fields (`api_key`, `token`, `secret`, `password`, `auth`, `bearer`) across all response payloads.
- **Controlled, Safe Execution**: `POST /api/v1/experiments/{id}/execute` allows running experiments using an allowlisted registry (`rag-pipeline-v1`, `rag-pipeline-topk-2`, `mock-adapter-v1`, `echo-adapter`), strictly rejecting arbitrary module paths or untrusted code execution.
- **Strict Non-Silent Fallback Policy**: If live API communication fails, the frontend surfaces an explicit error state with retry options rather than silently substituting static fixtures.

---

## Getting Started: How to Run the Platform

### Prerequisites

- **Python**: 3.13 is required. See `pyproject.toml` for authoritative dependencies.
- **Node.js**: v20+ with npm (Node v24 tested) is required for the frontend console.

---

### 1. Python Environment Setup

Clone the repository and install the package in editable mode with development dependencies:

```bash
# Core framework, FastAPI, Uvicorn, and test dependencies
pip install -e ".[dev]"

# Optional: semantic similarity (BERTScore) and sentence transformers support
pip install -e ".[semantic]"
```

---

### 2. Running Verification & Quality Checks

Run the complete test suite and code quality checks across the backend:

```bash
# Run the complete Python test suite (421 unit, integration, and API tests; ~5 seconds execution)
pytest

# Run linter
ruff check .

# Check code formatting
ruff format --check .

# Run static type checker
mypy src

# Optional: run neural model integration tests (requires network on first run to download models)
LLM_RELIABILITY_RUN_INTEGRATION_TESTS=1 pytest tests/evaluation/test_semantic_similarity_integration.py
LLM_RELIABILITY_RUN_INTEGRATION_TESTS=1 pytest tests/rag/test_embeddings_integration.py
```

---

### 3. Seeding Live Experiment Data

Before starting the server for the first time, seed the local repository `./experiments` with reproducible baseline experiments, executed runs, and RAG traces:

```bash
# Seeds experiments/ with exp-baseline-rag-v1, exp-intervention-rag-topk2, and exp-baseline-non-rag-demo
python scripts/seed_live_experiments.py
```

---

### 4. Running Backend & Frontend Together (Single Command)

To run both the **FastAPI backend** and **Vite frontend** simultaneously with one command:

```bash
python run.py
```

This launches both servers in parallel:
- **Frontend Research Console**: [http://localhost:5173](http://localhost:5173)
- **Backend FastAPI Service**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive OpenAPI Docs**: [http://127.0.0.1:8000/api/v1/docs](http://127.0.0.1:8000/api/v1/docs)

Press `Ctrl+C` in your terminal to cleanly shut down both servers.

---

### 5. Running Services Separately

If you prefer to run the services in separate terminals:

#### Terminal 1 — FastAPI REST API Server:
```bash
python -m uvicorn llm_reliability.api.app:app --host 127.0.0.1 --port 8000 --reload
```

#### Terminal 2 — React Research Console:
```bash
cd frontend
npm run dev
```

#### REST API Endpoint Catalog

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | `GET` | Health status, version, environment, and allowlisted adapters. |
| `/api/v1/experiments` | `GET` | Paginated experiment definitions with optional `search` filtering. |
| `/api/v1/experiments` | `POST` | Create a new experiment definition with allowlisted evaluators. |
| `/api/v1/experiments/{id}` | `GET` | Detailed experiment definition with configuration snapshot and secret redaction. |
| `/api/v1/experiments/{id}/runs` | `GET` | Paginated history of executed runs for an experiment. |
| `/api/v1/experiments/{id}/execute` | `POST` | Safely execute an experiment using an allowlisted adapter registry. |
| `/api/v1/runs` | `GET` | Paginated list of executed runs across all experiments. |
| `/api/v1/runs/{id}` | `GET` | Run execution detail, duration, status, and test case outcomes. |
| `/api/v1/runs/{id}/test-cases` | `GET` | Paginated test case evaluations with optional `passed` filter. |
| `/api/v1/runs/{id}/test-cases/{tc_id}` | `GET` | Individual test case record with model output and evaluator evaluations. |
| `/api/v1/runs/{id}/summaries` | `GET` | Complete run summary bundle: evaluation counts, reliability findings, retrieval metrics. |
| `/api/v1/diagnostics` | `GET` | Paginated forensic diagnostics filtered by run, category, or secondary finding. |
| `/api/v1/diagnostics/{run_id}/{tc_id}` | `GET` | Individual diagnostic record with claim-level lexical evidence assessments. |
| `/api/v1/rag/retrieval-results` | `GET` | Paginated RAG traces, ranked retrieved passages, and retrieval evaluation scores. |
| `/api/v1/rag/retrieval-results/{run_id}/{tc_id}` | `GET` | Test case retrieval trace with Hit@K, Recall@K, and MRR. |
| `/api/v1/robustness` | `GET` | Robustness perturbation comparisons and category shifts. |
| `/api/v1/robustness/{run_id}` | `GET` | Full robustness bundle with consistency rates and category transition matrix. |
| `/api/v1/analytics/overview` | `GET` | Platform-wide roll-up: total experiments, runs, evaluations, coverage, and recent executions. |
| `/api/v1/analytics/evaluators` | `GET` | Evaluator aggregates and pairwise concordance rates on compatible criteria. |
| `/api/v1/analytics/failures` | `GET` | Primary failure distribution and co-occurring secondary findings. |
| `/api/v1/analytics/retrieval` | `GET` | Retrieval quality summaries preserving explicit K parameters. |
| `/api/v1/analytics/robustness` | `GET` | Perturbation consistency summary and transition counts. |
| `/api/v1/analytics/performance` | `GET` | Execution duration telemetry, latency distributions, and explicit absence disclosures. |
| `/api/v1/analytics/comparisons` | `GET` | Cross-run comparative analysis with automated methodological compatibility classification. |
| `/api/v1/analytics/trends/{exp_id}` | `GET` | Longitudinal trajectory across sequential runs with configuration transition markers. |

---

### 5. Running the Frontend Research Console

The research console provides an interactive observability UI with live connectivity to the FastAPI backend:

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Run frontend test suite (30 Vitest tests: API client, connection banner, derived helpers)
npm run test

# Run frontend linter (ESLint 9)
npm run lint

# Build production bundle (TypeScript tsc -b + Vite production build)
npm run build

# Start local development server with live hot-reloading
npm run dev
# The console will open at http://localhost:5173/

# Or preview the production build locally:
npm run preview
# The production console preview will be available at http://localhost:4173/
```

#### Live API Mode vs. Fixture Demo Mode
The application header features an interactive connection pill:
- **Live API (Connected)**: Displays an emerald badge when connected to `http://127.0.0.1:8000/api/v1`. All pages load live data dynamically from the FastAPI backend.
- **Demo Mode**: Allows exploring the console using bundled static fixtures without a running Python backend.
- **Strict Non-Silent Fallback**: If the backend server stops, the console clearly displays an error banner with `[Retry Connection]` and `[Switch to Demo Mode]` buttons. The application never silently substitutes static fixture data for failed live API requests.

---

### 6. Executing Experiments Interactively

To run an experiment directly from the research console:
1. Ensure the FastAPI server is running on `http://127.0.0.1:8000`.
2. Open the frontend console at `http://localhost:5173/`.
3. Navigate to **Experiments** (`/experiments`).
4. Select **Baseline RAG Retrieval & Generation Study** (`/experiments/exp-baseline-rag-v1`).
5. Under the **Execute Experiment** card, select an allowlisted adapter (`rag-pipeline-v1` or `rag-pipeline-topk-2`).
6. Click **Execute Run**.
7. Once execution finishes, a success banner displays the new `run_id`, elapsed duration, and a direct link to **Inspect Run Detail**.
8. The new run immediately appears in the run history and contributes live data to the analytics overview.

---

### 7. Console Interface Guide

Once the frontend console is running in your browser:

- **Overview (`/`)**: High-level platform statistics, failure category distribution, method coverage, and recent execution history.
- **Analytics (`/analytics`)**: Cross-experiment observability roll-up, evaluator coverage proportions, failure distribution forensics, and longitudinal trajectory across sequential runs.
- **Comparisons (`/comparisons`)**: Side-by-side run comparisons with automated methodological compatibility classification (`identical_configuration`, `partially_different_configuration`, `materially_incompatible`), configuration diffs, and disaggregated evaluator metrics.
- **Failures (`/failures`)**: Dedicated failure taxonomy forensics preserving primary classifications and co-occurring secondary findings independently with exact sample counts and denominators.
- **Performance (`/performance`)**: Execution duration telemetry (wall-clock and per-case latency) with explicit reporting of missing telemetry (tokens, monetary costs) and explanatory engineering notes.
- **Experiments (`/experiments`)**: Catalog of defined experiments, model configurations, and content-addressed fingerprints with search filtering.
- **Experiment Detail (`/experiments/:experimentId`)**: Complete configuration snapshot, reproducibility hashes, dataset parameters, interactive execution triggers, and run history.
- **Run Detail (`/runs/:runId`)**: Run outcome metrics, failure category distributions, and test cases table with category filtering.
- **Test Case Detail (`/runs/:runId/test-cases/:testCaseId`)**: Deep-dive diagnostic view displaying the primary failure category, preserved secondary observations, evaluator breakdown, retrieved context passages, and sentence-level lexical assessments.
- **RAG Diagnostics (`/rag`)**: Retrieval metrics (Hit@1, MRR, Recall@1), RAG failure attributions (`retrieval_failure` vs `generation_failure`), and faithfulness scope notes.
- **Evaluator Comparison (`/evaluators`)**: Evaluator performance matrix, pairwise evaluator concordance rates, and case analysis of evaluator disagreements.
- **Reliability Diagnostics (`/reliability`)**: Consolidated cross-run failure category distribution with interactive category filtering and diagnostic inspection table.
- **Robustness (`/robustness`)**: Consistency rates under controlled input perturbations (`question_rewording`, `context_reordering`, etc.) and outcome transition matrices.

---

### 8. Real LLM Provider Integration & Benchmarking (Milestone 8)

The platform supports live model evaluation via `OpenAIProviderAdapter` (compatible with OpenAI models such as `gpt-4o`, `gpt-4o-mini`, and OpenAI-compatible local/remote servers like vLLM, Ollama, or LM Studio).

#### Key Capabilities:
- **Provider-Independent Abstraction:** Conforms to `ModelAdapter` with bounded retry policies (exponential backoff and jitter on 429/503), wall-clock latency measurement, and normalized error handling.
- **Strict Credential Security:** Server-side environment variables only (`OPENAI_API_KEY`). API keys are never stored in experiment definitions, run records, fingerprints, git, or API responses.
- **External Inference Safety Opt-In:** Paid model calls require explicit `allow_external_inference=True` confirmation to prevent accidental credit spend.
- **Token Usage Provenance:** Extracted directly from provider response (`prompt_tokens`, `completion_tokens`, `total_tokens`). Missing usage is retained as `null` (never fabricated).
- **Controlled Multi-Model Benchmarks:** Run a matrix of models across repeated trials (`repetitions: 1-5`) to measure stochastic variance without synthetic score collapse.

#### Environment Configuration:
```bash
# Set your OpenAI API key (server environment only)
export OPENAI_API_KEY="your-api-key-here"

# (Optional) For local OpenAI-compatible endpoints (Ollama, vLLM, LM Studio)
export OPENAI_BASE_URL="http://localhost:11434/v1"

# (Optional) Enable server-wide external inference permission
export LLM_RELIABILITY_ALLOW_EXTERNAL_INFERENCE=true
```

---

## Repository Structure

```
research/           Research notes, paper references, and reproducible experiments.
  notes/            Architecture notes (core engine, reproducibility, RAG, reliability, analytics, FastAPI API, real providers & benchmarks).
  papers/           Paper references and documentation.
  experiments/      Experiment definitions, configurations, results, and failure analysis.
src/                Platform source code.
  llm_reliability/api/          FastAPI application factory, versioned routers, Pydantic schemas, dependency injection, and allowlisted registries.
  llm_reliability/providers/    Real LLM provider adapters (OpenAIProviderAdapter, BaseProviderAdapter, error hierarchy, bounded retry backoff).
  llm_reliability/evaluation/   Core Evaluation Engine, adapters, monotonic clock, and answer evaluators.
  llm_reliability/experiments/  Content-addressed fingerprinting, experiment definitions, and repository persistence.
  llm_reliability/reliability/  Reliability analysis layer, evaluator disagreement detection, and summary statistics.
  llm_reliability/rag/          Controlled RAG pipeline, dense cosine retriever, and retrieval evaluators (Hit@K, MRR).
  llm_reliability/diagnostics/  Consolidated failure taxonomy, secondary findings preservation, and robustness perturbations.
  llm_reliability/analytics/    Observability and analytics layer: aggregations, pairwise agreement, cross-run comparison, and longitudinal trends.
frontend/           Research Console Web UI (React, TypeScript, Vite, Tailwind CSS).
  src/api/          Typed API client targeting /api/v1 with request correlation, provider readiness, and benchmark execution.
  src/context/      ConnectionContext managing live connectivity, dataMode toggle, and benchmark execution.
  src/components/   ConnectionBanner, DataStateView, Card, Tag, EmptyState, FixtureNotice, Layout.
  src/pages/        Overview, Analytics, Comparisons, Failures, Performance, Experiments, Run Detail, TestCase Detail, RAG, Evaluators, Reliability, Robustness.
  src/data/         Typed data-access layer, custom async data hooks (useData.ts), and serialized JSON fixtures.
  src/__tests__/    Frontend unit tests (32 Vitest tests: API client, connection context, analytics, dataAccess, derived helpers, tone mappings).
tests/              Backend test suite (435 tests: 432 passed, 3 skipped, 100% passing).
  api/              Integration tests for FastAPI routers, schemas, correlation middleware, providers, and benchmarks.
  providers/        Offline unit tests for OpenAIProviderAdapter, retry backoff, token provenance, and safety guards.
  evaluation/       Tests for Core Evaluation Engine, adapters, clock, and evaluators.
  experiments/      Tests for content-addressed fingerprinting and persistence.
  reliability/      Tests for reliability analysis and findings aggregation.
  rag/              Tests for chunking, retriever, embeddings, and RAG execution.
  diagnostics/      Tests for failure category precedence, secondary findings, and robustness.
  analytics/        Tests for aggregations, evaluator agreement, run comparison, trends, and service.
configs/            Reproducible experiment and system configurations.
data/               Hand-curated benchmark evaluation sets.
  eval_sets/        Evaluation sets (e.g. data/eval_sets/evaluation_methodology_baseline_v1.json).
scripts/            Deterministic seeder (seed_live_experiments.py), reproducible demonstration, and fixture export scripts.
```

## Python Version

Python 3.13 is required. See `pyproject.toml` for the authoritative version constraint.

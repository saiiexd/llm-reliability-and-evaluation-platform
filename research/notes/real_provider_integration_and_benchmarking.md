# Research Note: Real LLM Provider Integration & Controlled Benchmarking

**Milestone:** 8  
**Date:** November 2026  
**Status:** Implemented & Verified  

---

## 1. Motivation and Problem Statement

Previous milestones established an evaluation engine, reproducible content-addressed experiment persistence, multi-evaluator scoring, RAG retrieval diagnostics, cross-run observability analytics, and a FastAPI/React web application. However, executions relied on deterministic mock adapters and simulated generation.

Milestone 8 introduces real language model provider integration (`OpenAIProviderAdapter`) and controlled benchmarking capabilities. The research objective is to evaluate actual models under rigorously specified experimental conditions while preserving scientific integrity, token usage provenance, and credential isolation.

---

## 2. Architecture & Design Principles

### 2.1 Provider-Independent Interface (`BaseProviderAdapter`)
The engine interacts with external models through `ModelAdapter.generate(request: ExecutionRequest) -> ModelResponse`. `BaseProviderAdapter` subclasses `ModelAdapter` and encapsulates common operational concerns:
- **Monotonic Wall-Clock Measurement:** Employs `time.perf_counter()` to capture platform-observed latency in milliseconds, independent of system clock adjustments.
- **Bounded Retry Loop with Backoff and Jitter:** Automatically retries transient errors (HTTP 429 rate limit, 503 service unavailable, timeouts) with exponential backoff $d = \text{backoff\_factor} \times 2^{a-1} + \text{jitter}$ up to a configurable maximum retry limit (default 3 retries).
- **Non-Retryable Isolation:** Permanent failures such as authentication errors (HTTP 401/403) or malformed client parameters fail immediately on the first attempt without wasteful retries.
- **Auditable Attempt Tracking:** Records the exact number of execution attempts in `provider_metadata["attempts"]`.

### 2.2 Strict Credential Security & Zero-Leakage Policy
To prevent credential leakage in research artifacts:
1. **Server-Side Environment Variables Only:** Credentials are read strictly from `OPENAI_API_KEY` or `LLM_RELIABILITY_OPENAI_API_KEY`.
2. **Never Persisted:** API keys are never stored in `ModelConfig`, `ExperimentDefinition`, `ExperimentRun`, configuration fingerprints, git repositories, or API responses.
3. **Readiness Inspection without Paid Calls:** The `GET /api/v1/providers` endpoint reports boolean configuration readiness (`is_configured: true/false`) and supported models without transmitting credentials or making paid API requests.

### 2.3 External Inference Safety Opt-In Guard
To prevent accidental paid API consumption during offline development or testing:
- External provider adapters require an explicit `allow_external_inference=True` parameter.
- If unconfirmed, execution is refused immediately, raising `ExternalInferenceNotPermittedError` before any network socket is opened.

### 2.4 Token and Latency Provenance
Scientific evaluation requires distinguishing measured telemetry from fabricated or assumed figures:
- `prompt_tokens`, `completion_tokens`, `total_tokens` are extracted directly from provider-reported usage.
- If the provider does not supply token metrics, fields remain `None` (`null` in JSON) with `token_source: "unavailable"`.
- Under no circumstances does the platform substitute 0 or heuristic token counts for missing data.
- Platform wall-clock duration is reported alongside provider-reported processing latency (if present) so network transfer overhead can be distinguished from inference time.

---

## 3. Controlled Multi-Model Benchmarking & Repetition Semantics

Language models with temperature $T > 0$ exhibit stochastic variance across repeated generations. To measure this variance reliably:
- **Matrix Benchmarking:** A benchmark executes multiple model adapters across a fixed dataset version and evaluator suite.
- **Repetitions:** Each model can be executed for $N$ repetitions ($N \in [1, 5]$).
- **No Synthetic Collapsing:** Rather than averaging stochastic results into a single score that masks variance, each trial produces an independent, content-addressed `ExperimentRun` linked by `benchmark_id` and `repetition_index`.
- **Methodological Compatibility:** When comparing runs from the same benchmark, the platform’s analytical comparison layer automatically classifies them:
  - Repeated runs of the identical model: `IDENTICAL_CONFIGURATION` (isolates stochastic variance).
  - Runs across different models on the same dataset: `PARTIALLY_DIFFERENT_CONFIGURATION` (controlled intervention).

---

## 4. API Endpoints Introduced

- `GET /api/v1/providers`: Lists provider readiness, supported models, and default parameters.
- `POST /api/v1/benchmarks/execute`: Launches a controlled benchmark across selected allowlisted adapters and repetitions.
- `GET /api/v1/benchmarks/{benchmark_id}/runs`: Retrieves all persisted runs belonging to a benchmark.
- `POST /api/v1/experiments/{id}/execute`: Supports `allow_external_inference: bool` parameter for single-run execution.

---

## 5. Verification & Test Suite

The provider integration is validated using 100% offline unit and integration tests:
- `tests/providers/test_openai_provider.py`: Validates token extraction, latency recording, 429 backoff retries, non-retryable 401 handling, and the safety opt-in guard using test doubles.
- `tests/api/test_providers.py`: Validates provider readiness reporting and ensures zero credential exposure.
- `tests/api/test_benchmarks.py`: Validates multi-model benchmark executions, repetition indexing, and external inference rejection.
- Zero outbound network calls are made during automated test runs.


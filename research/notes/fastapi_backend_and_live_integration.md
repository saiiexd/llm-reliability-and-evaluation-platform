# FastAPI Backend Architecture & Live Frontend Integration

## 1. Architectural Role & Boundary Enforcement

The introduction of FastAPI in Milestone 7 establishes an explicit application boundary between the React research console and the core Python research engine. 

Prior to this milestone, the frontend was bound exclusively to pre-exported static JSON fixtures (`frontend/src/data/fixtures/*.json`). While this fixture layer permitted isolated UI development and offline demonstration, it prohibited dynamic execution, real-time parameter tuning, interactive diagnostics, and programmatic querying.

The API architecture is governed by strict boundaries:
1. **Domain Independence**: The core research engine (`src/llm_reliability/evaluation/`, `src/llm_reliability/experiments/`, `src/llm_reliability/rag/`, `src/llm_reliability/diagnostics/`, `src/llm_reliability/analytics/`) remains completely independent of FastAPI. Core domain models are pure Python dataclasses and can be imported and executed without starting an HTTP daemon or importing ASGI frameworks.
2. **Dedicated Schema Boundary**: External HTTP communication relies on explicit Pydantic v2 schemas (`src/llm_reliability/api/schemas/`). Internal dataclasses are never dumped raw to HTTP responses. Deliberate conversion functions in `src/llm_reliability/api/services.py` map domain models to stable API contracts, preventing internal refactorings from inadvertently leaking or breaking external callers.
3. **No Database Layer**: Persistence relies strictly on the atomic JSON filesystem repository abstraction (`ExperimentRepository`) established in Milestone 2. No relational or document database vendors (PostgreSQL, SQLite, MongoDB, Redis) are introduced.

```
+-------------------------------------------------------------------------+
|                              React Frontend                             |
|  - ConnectionProvider (Live API vs Fixture Mode toggle)                 |
|  - Typed ApiClient (/api/v1)                                            |
|  - Visual Connection Indicator & Strict Non-Silent Fallback Banner      |
+-------------------------------------------------------------------------+
                                    |
                            HTTP / JSON REST
                                    |
+-------------------------------------------------------------------------+
|                         FastAPI Application Layer                       |
|  - Request Correlation Middleware (X-Request-ID, X-Response-Time-Ms)    |
|  - Configurable CORS Middleware (Environment-driven)                    |
|  - Dependency Injection (Settings, Repositories, Registries, Services)  |
|  - Uniform Error Envelope (ErrorResponse, ResourceNotFound, Conflict)   |
|  - Versioned Routers:                                                   |
|      /api/v1/health                                                     |
|      /api/v1/experiments                                                |
|      /api/v1/runs                                                       |
|      /api/v1/diagnostics                                                |
|      /api/v1/rag                                                        |
|      /api/v1/robustness                                                 |
|      /api/v1/analytics                                                  |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                         Application Services Layer                      |
|  - DiagnosticsService (Caches derived diagnostics & retrieval metrics)  |
|  - AnalyticsService (Cross-run aggregation, comparison, longitudinal)   |
|  - AdapterRegistry (Strict allowlisting: mock, rag-pipeline-v1, topk-2)  |
|  - EvaluatorRegistry (Allowlisted evaluators: exact match, judge, etc.) |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                      Core Domain & Persistence Layer                    |
|  - ExperimentManager & Runner (Evaluation & RAG pipelines)              |
|  - Atomic Filesystem Repository (experiments/<exp_id>/runs/<run_id>/)   |
+-------------------------------------------------------------------------+
```

---

## 2. Information Security & Secret Redaction

API contracts deliberately redact and sanitize sensitive credentials. Large language model pipelines frequently require authorization tokens, provider credentials, or database keys. Exposing these values in API responses represents a critical vulnerability.

### 2.1 Recursive Pattern-Based Redaction
All dictionaries containing model configuration, provider metadata, or evaluator parameters pass through `redact_sensitive_dict()` before being serialized into response models:

```python
_SECRET_PATTERN = re.compile(
    r"(key|secret|token|password|auth|credential|bearer)",
    re.IGNORECASE,
)
```

Any key matching this pattern is replaced with `"***REDACTED***"` across all nested dictionaries and lists.

### 2.2 Controlled Execution Boundary
To prevent arbitrary remote code execution (RCE), the execution endpoint (`POST /api/v1/experiments/{id}/execute`) **strictly forbids arbitrary class loading, module reflection, or dynamic file imports**.

Instead, execution is governed by `AdapterRegistry`:
- An explicit, static allowlist of known adapters (`rag-pipeline-v1`, `rag-pipeline-topk-2`, `mock-adapter-v1`, `echo-adapter`).
- Any attempt to pass an unallowlisted adapter identifier or arbitrary Python path is immediately rejected with HTTP 400 (`EXECUTION_ERROR`).
- Evaluators are similarly reconstructed strictly through `EvaluatorRegistry`.

---

## 3. Strict Non-Silent Fallback Policy

A foundational tenet of the platform is scientific integrity: **The platform must never present fabricated, assumed, or out-of-date data as if it were live telemetry.**

### 3.1 Failure Transparency
If a network error, timeout, or backend 500 error occurs while the console is in Live API mode:
- The system **strictly refuses to silently fall back to static fixtures**.
- An explicit, unmistakable error banner is rendered, exposing the status code, endpoint URL, error message, and `X-Request-ID`.
- The user is provided two deliberate, conscious choices:
  1. `[Retry Connection]`: Re-attempt the live API request.
  2. `[Switch to Demo Fixture Mode]`: Consciously acknowledge switching to offline demo mode.

### 3.2 Clear UI Distinctions
- **Live Mode**: When connected, a clean emerald badge indicates live connection to the backend. The fixture warning banner is suppressed.
- **Fixture Mode**: An amber banner clearly informs the researcher: `"Fixture data mode: Reading bundled sample datasets from JSON fixtures. Switch to Live API for real backend experiments."`

---

## 4. Provenance & Denominational Integrity

Across all analytics endpoints (`/api/v1/analytics/*`), the API preserves the underlying sample counts, numerators, and denominators:
- In `EvaluatorAgreementSummaryResponse`, both `agreements` and `disagreements` are reported alongside `cases_compared`. Agreement rates are calculated as `agreements / cases_compared` and default to `None` if zero cases were compared, avoiding division by zero or fabricated percentages.
- In `FailureDistributionResponse`, raw category counts are provided alongside proportions.
- In `ExecutionPerformanceResponse`, telemetry boundaries are explicitly signaled: uninstrumented metrics (such as financial pricing or missing token counts) are reported as `None` with `tokens_available: false` and `cost_available: false`, rather than being assigned synthetic zeros.

---

## 5. Verification & Test Suite

The API implementation is verified through an extensive suite of integration tests in `tests/api/`:
- `test_health.py`: Validates readiness checks, version reporting, and adapter discovery.
- `test_experiments.py`: Validates experiment indexing, pagination, search queries, detail retrieval, and sensitive secret redaction.
- `test_runs.py`: Validates run detail, test case filtering, pagination, and summaries.
- `test_diagnostics.py`: Validates reliability failure categorization, claim assessments, and category filters.
- `test_rag.py`: Validates retrieval results, chunk similarity scores, Hit@K, Recall@K, and MRR metrics.
- `test_robustness.py`: Validates perturbation pairs, category transition counts, and consistency rates.
- `test_analytics.py`: Validates platform overview, evaluator concordance, failure distributions, longitudinal trends, and multi-run comparisons.
- `test_execution.py`: Validates controlled experiment execution, rejection of unallowlisted adapters, and rejection of arbitrary module paths.
- `test_errors_and_cors.py`: Validates request correlation headers (`X-Request-ID`), response timing headers (`X-Response-Time-Ms`), CORS preflight handling, and uniform error envelopes.


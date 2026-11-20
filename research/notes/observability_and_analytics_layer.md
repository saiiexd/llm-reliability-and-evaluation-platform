# Research Note: Observability, Analytics, and Experiment Comparison Layer

This note specifies the architectural foundation, domain models, aggregation definitions, comparison compatibility rules, and explicit boundaries of the LLM Reliability & Evaluation Platform's Observability & Analytics Layer (`src/llm_reliability/analytics/`).

---

## 1. Scientific Objective and Core Principles

The previous milestones built a system capable of executing, diagnosing, and inspecting individual runs. Prompt 6 introduces a cross-run analytical framework to answer research questions such as:
- How do evaluation outcomes vary across repeated executions under identical configurations?
- Does changing a model, prompt, or retrieval parameter (e.g. chunk size or top-$k$) reliably alter performance?
- Which failure categories dominate across different experimental setups?
- How often do independent evaluators agree, and what characterises their disagreements?
- Does an improvement in one metric (e.g. exact match) coincide with regressions in another (e.g. faithfulness or retrieval)?

### Fundamental Principles

1. **Distinction Between Observation and Causal Interpretation**:
   - The platform reports observed empirical differences across runs and configurations.
   - An aggregate difference must never be asserted as a causal effect unless the experimental design strictly isolates that variable (e.g., identical dataset version, identical prompt template, single-parameter intervention).
2. **Preservation of Sample Counts and Denominators**:
   - Every percentage or rate must expose its underlying numerator and denominator (e.g. "$4/10$ test cases passed ($40.0\%$)" rather than a bare "$40\%$").
3. **No Incompatible Aggregations or Universal Scores**:
   - Distinct evaluation methods (Exact Match, Semantic Similarity, LLM Judge, Faithfulness, and Retrieval Hit@K) operate under fundamentally different assumptions and scales. They are never averaged or combined into a single "platform score".
4. **Preservation of Provenance**:
   - Every aggregate metric identifies the source runs, dataset version IDs, model configurations, and configuration fingerprints involved in its derivation.
5. **Honest Representation of Unavailable Data**:
   - Measurements that are not captured (such as token usage or monetary cost) are explicitly designated as `unavailable`. They are never filled with fabricated zeros or synthetic estimates.

---

## 2. Available vs. Unavailable Measurements

| Dimension | Measured / Available | Derived | Explicitly Unavailable (Not Fabricated) |
|---|---|---|---|
| **Execution Lifecycle** | Run `started_at`, `finished_at` (UTC ISO timestamps) | Wall-clock elapsed duration (`finished_at - started_at`) | N/A |
| **Component Latency** | Optional monotonic clock elapsed seconds per test case & evaluator | Per-test-case mean duration | Remote provider internal queuing / network transit latency |
| **Token Usage** | `ModelResponse.prompt_tokens`, `completion_tokens`, `total_tokens` (when reported by adapter) | Sum across test cases where reported | Token counts when adapter reports `None` (e.g. mock adapter) |
| **Monetary Cost** | None | None | **All monetary costs** (no arbitrary pricing tables or synthetic cost estimations) |
| **Answer Quality** | `EvaluationResult.passed`, `score`, `status`, `label` per evaluator | Evaluator success rates, pass rates, score mean/min/max with sample counts | Cross-evaluator single "quality index" |
| **Failure Diagnostics** | `ReliabilityDiagnostic.failure_category`, `secondary_findings` | Failure category distributions & secondary finding frequencies | Blanket "hallucination score" |
| **Retrieval Quality** | `Hit@K`, `Recall@K`, `MRR` per test case with ground truth | Mean Hit@K, Mean Recall@K, Mean MRR (K-preserved) | Retrieval scores for non-RAG runs or test cases lacking ground truth |
| **Robustness** | `RobustnessResult.outcome_changed`, `failure_category_transition` | Consistency rate per perturbation type with pair denominators | Single ungrounded robustness score |

---

## 3. Configuration Compatibility Rules for Comparison

When comparing two or more runs, the comparison engine categorises their relationship into one of three explicit compatibility tiers:

1. **`IDENTICAL_CONFIGURATION` (Repeated Run)**:
   - Matches: identical `config_fingerprint` (same dataset version ID, identical model parameters, identical evaluator configurations).
   - Research validity: Quantifies stochasticity, non-determinism, and execution stability across repeated executions under identical conditions.
2. **`PARTIALLY_DIFFERENT_CONFIGURATION` (Controlled Intervention)**:
   - Matches: identical dataset version ID (`dataset_version.version_id`).
   - Differences: single or multiple controlled changes in model configuration (e.g. temperature, prompt template, top-$k$, or evaluator thresholds).
   - Research validity: Permissible for comparative analysis of intervention effects on the same underlying benchmark. Differences in configuration are explicitly surfaced in a structured diff.
3. **`MATERIALLY_INCOMPATIBLE` (Disjoint Context)**:
   - Differences: distinct dataset version IDs (different test cases, inputs, or reference answers).
   - Research validity: Direct metric comparison is scientifically invalid (e.g. comparing accuracy on 10 biology questions against accuracy on 100 legal questions). The comparison engine flags incompatibility, explains the divergence, and presents separate breakdowns without claiming direct equivalence.

---

## 4. Evaluator Agreement and Disagreement Analytics

Pairwise evaluator agreement is computed only across evaluators that assess compatible criteria (e.g. two evaluators both assessing `correctness`):
- **Eligible Cases**: Test cases where both evaluators completed with `status == SUCCESS` and produced a definitive boolean `passed` judgment.
- **Agreement**: Cases where both evaluators agree on `passed` (`True == True` or `False == False`).
- **Disagreement**: Cases where one evaluator passed and the other failed.
- **Excluded Cases**: Cases where either evaluator was skipped, encountered an execution error, or produced no binary pass/fail verdict.
- **Agreement Rate**:
  $$\text{Agreement Rate} = \frac{\text{Agreements}}{\text{Cases Compared}}$$
  Exposed alongside the sample size $\text{Cases Compared}$ and the list of specific test case IDs exhibiting disagreement.

---

## 5. Performance and Latency Instrumentation

To enable execution timing without distorting test determinism or introducing external tracing vendors:
- An injectable `Clock` protocol (`Callable[[], float]`) defaults to `time.monotonic`.
- Deterministic unit tests inject a `FakeClock` with stepped timestamps.
- Wall-clock elapsed run duration is derived from UTC timestamps.
- Optional component durations (`TestCaseResult.duration_seconds`, `EvaluationResult.duration_seconds`) capture fine-grained latencies.
- Token counts and costs remain `None` / `unavailable` until provided by a real provider adapter.


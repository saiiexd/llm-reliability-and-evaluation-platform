"""
Reliability, faithfulness, contradiction, and robustness diagnostics.

Builds on ``llm_reliability.evaluation`` (answer-quality evaluators),
``llm_reliability.reliability`` (per-run aggregation and disagreement
findings), and ``llm_reliability.rag`` (retrieval evaluation and RAG-level
diagnosis), consolidating their evidence into one machine-readable,
persistable diagnostic record per test case (``ReliabilityDiagnostic``),
statement-level evidence assessment (``ClaimAssessment`` /
``EvidenceAssessment``), and baseline-versus-perturbation robustness
comparison (``RobustnessResult`` / ``ConsistencyResult``).

Every function in this package is read-only: it derives its output from
evaluation results, retrieval results, and test case metadata that already
exist and are already persisted through the existing Experiment/Run
architecture (``llm_reliability.experiments``); nothing here introduces a
new persisted schema; a diagnostic is always exactly reproducible by
recomputing it from a loaded ``ExperimentRun``.

This layer does not detect hallucinations as a general phenomenon and
does not treat any single evaluator's output as ground truth. See
``research/notes/reliability_and_robustness_layer.md`` for the full
methodology and its limitations.
"""

from llm_reliability.diagnostics.diagnosis import (
    ReliabilityDiagnostic,
    build_reliability_diagnostic,
)
from llm_reliability.diagnostics.evidence import (
    LEXICAL_BASELINE_METHOD,
    ClaimAssessment,
    EvidenceAssessment,
    assess_statements_lexically,
    split_into_statements,
)
from llm_reliability.diagnostics.robustness import (
    BASELINE_TEST_CASE_ID_KEY,
    PERTURBATION_TYPE_KEY,
    ConsistencyResult,
    PerturbationType,
    RobustnessResult,
    compare_robustness,
    get_perturbation_info,
    summarize_robustness,
)
from llm_reliability.diagnostics.taxonomy import (
    ClaimSupportStatus,
    DiagnosticStatus,
    FailureCategory,
)

__all__ = [
    "BASELINE_TEST_CASE_ID_KEY",
    "LEXICAL_BASELINE_METHOD",
    "PERTURBATION_TYPE_KEY",
    "ClaimAssessment",
    "ClaimSupportStatus",
    "ConsistencyResult",
    "DiagnosticStatus",
    "EvidenceAssessment",
    "FailureCategory",
    "PerturbationType",
    "ReliabilityDiagnostic",
    "RobustnessResult",
    "assess_statements_lexically",
    "build_reliability_diagnostic",
    "compare_robustness",
    "get_perturbation_info",
    "split_into_statements",
    "summarize_robustness",
]

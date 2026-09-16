"""
Robustness and consistency: comparing a baseline case against a controlled perturbation.

This is not an adversarial-testing framework: it studies whether the
platform's own observable behavior (retrieval, generation, evaluation, and
diagnosis) changes under a small set of controlled, explicitly labeled
perturbations of an otherwise-equivalent input. A perturbed test case is
represented the same way any other test case is (see
``llm_reliability.rag.ground_truth`` for the analogous pattern used for
retrieval ground truth): as an ordinary ``TestCase`` whose
``metadata`` records which baseline test case it perturbs and how, so no
new persistence mechanism is needed -- a perturbed case is just another
row in a ``Dataset``, and its ``ExperimentRun`` results are read the same
way any other test case's are.

``compare_robustness`` and ``summarize_robustness`` are both pure,
read-only functions over already-computed ``ReliabilityDiagnostic``
records; they never re-run anything. Only measurements with a clear,
statable definition are computed here -- see each function's docstring
for exactly what it means and how it is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from llm_reliability.diagnostics.diagnosis import ReliabilityDiagnostic

BASELINE_TEST_CASE_ID_KEY = "perturbation_of"
PERTURBATION_TYPE_KEY = "perturbation_type"


class PerturbationType(StrEnum):
    """Controlled perturbation types this platform can represent and compare.

    Each is a change applied to an otherwise-equivalent baseline test case
    or execution configuration, so the platform can observe whether that
    one change alone affects the outcome.
    """

    QUESTION_REWORDING = "question_rewording"
    """An equivalent rephrasing of the same question."""

    CONTEXT_REORDERING = "context_reordering"
    """The same retrieved chunks presented to generation in a different order."""

    TOP_K_CHANGE = "top_k_change"
    """A different number of retrieved chunks supplied to generation."""

    IRRELEVANT_CONTEXT_INJECTION = "irrelevant_context_injection"
    """An irrelevant chunk added to otherwise-unchanged retrieved context."""

    CONTEXT_MODIFICATION = "context_modification"
    """A controlled edit to the content of retrieved context itself."""

    REPEATED_INPUT = "repeated_input"
    """The identical input executed again, to observe non-determinism."""


def get_perturbation_info(metadata: dict[str, Any]) -> tuple[str, PerturbationType] | None:
    """Read a perturbed test case's baseline id and perturbation type from its metadata.

    Returns ``None`` if ``metadata`` does not describe a perturbation
    (the normal case for a baseline or unrelated test case).
    """
    baseline_id = metadata.get(BASELINE_TEST_CASE_ID_KEY)
    perturbation_type = metadata.get(PERTURBATION_TYPE_KEY)
    if baseline_id is None or perturbation_type is None:
        return None
    return str(baseline_id), PerturbationType(perturbation_type)


@dataclass
class RobustnessResult:
    """The observed relationship between one baseline case and one perturbed case."""

    baseline_test_case_id: str
    perturbed_test_case_id: str
    perturbation_type: PerturbationType
    baseline_category: str | None
    perturbed_category: str | None
    outcome_changed: bool
    failure_category_transition: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_test_case_id": self.baseline_test_case_id,
            "perturbed_test_case_id": self.perturbed_test_case_id,
            "perturbation_type": self.perturbation_type.value,
            "baseline_category": self.baseline_category,
            "perturbed_category": self.perturbed_category,
            "outcome_changed": self.outcome_changed,
            "failure_category_transition": self.failure_category_transition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobustnessResult:
        return cls(
            baseline_test_case_id=data["baseline_test_case_id"],
            perturbed_test_case_id=data["perturbed_test_case_id"],
            perturbation_type=PerturbationType(data["perturbation_type"]),
            baseline_category=data.get("baseline_category"),
            perturbed_category=data.get("perturbed_category"),
            outcome_changed=data["outcome_changed"],
            failure_category_transition=data.get("failure_category_transition"),
        )


def compare_robustness(
    baseline_diagnostic: ReliabilityDiagnostic,
    perturbed_diagnostic: ReliabilityDiagnostic,
    perturbation_type: PerturbationType,
) -> RobustnessResult:
    """Compare a baseline diagnostic against a perturbed diagnostic for the same underlying case.

    ``outcome_changed`` is true whenever the two diagnostics' failure
    categories differ (including when one is ``None``, i.e. one side
    could not be determined and the other could). This is a simple,
    literal comparison of two already-computed categories -- it does not
    re-derive or infer anything about *why* the outcome changed.
    """
    baseline_category = (
        baseline_diagnostic.failure_category.value if baseline_diagnostic.failure_category else None
    )
    perturbed_category = (
        perturbed_diagnostic.failure_category.value
        if perturbed_diagnostic.failure_category
        else None
    )
    outcome_changed = baseline_category != perturbed_category
    transition = f"{baseline_category} -> {perturbed_category}" if outcome_changed else None

    return RobustnessResult(
        baseline_test_case_id=baseline_diagnostic.test_case_id,
        perturbed_test_case_id=perturbed_diagnostic.test_case_id,
        perturbation_type=perturbation_type,
        baseline_category=baseline_category,
        perturbed_category=perturbed_category,
        outcome_changed=outcome_changed,
        failure_category_transition=transition,
    )


@dataclass
class ConsistencyResult:
    """Aggregate robustness statistics across a set of baseline/perturbed comparisons.

    ``outcome_consistency_rate``: the fraction of comparisons where
    ``outcome_changed`` was false -- "how often did the failure category
    stay the same under perturbation?" This is the only aggregate
    computed here, because it is the only one with an unambiguous
    definition given just a list of ``RobustnessResult`` records.
    Evaluator-level or retrieval-level stability would require comparing
    additional signals (individual evaluator verdicts, retrieved chunk
    sets) beyond the consolidated category, which is not computed by this
    function; compute those directly from the underlying
    ``ReliabilityDiagnostic``/``RetrievalEvaluationResult`` records if
    needed, rather than inventing a proxy metric here.
    """

    num_comparisons: int
    num_unchanged: int
    outcome_consistency_rate: float | None
    transition_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_comparisons": self.num_comparisons,
            "num_unchanged": self.num_unchanged,
            "outcome_consistency_rate": self.outcome_consistency_rate,
            "transition_counts": self.transition_counts,
        }


def summarize_robustness(results: list[RobustnessResult]) -> ConsistencyResult:
    """Aggregate a list of ``RobustnessResult`` records into a ``ConsistencyResult``."""
    num_comparisons = len(results)
    num_unchanged = sum(1 for r in results if not r.outcome_changed)
    transition_counts: dict[str, int] = {}
    for r in results:
        if r.failure_category_transition is not None:
            transition_counts[r.failure_category_transition] = (
                transition_counts.get(r.failure_category_transition, 0) + 1
            )
    return ConsistencyResult(
        num_comparisons=num_comparisons,
        num_unchanged=num_unchanged,
        outcome_consistency_rate=(num_unchanged / num_comparisons) if num_comparisons else None,
        transition_counts=transition_counts,
    )

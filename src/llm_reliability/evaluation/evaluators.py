"""
Evaluators for the Core Evaluation Engine.

An evaluator inspects a single test case and the model response produced
for it, and returns a structured ``EvaluationResult``. Evaluators must not
depend on adapter internals or on other evaluators, so any number of them
can be run over the same response independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_reliability.evaluation.models import EvaluationResult, ModelResponse, TestCase


class Evaluator(ABC):
    """Interface for a single evaluation method.

    ``name`` identifies the evaluator in ``EvaluationResult.evaluator_name``
    and must be stable across runs.
    """

    name: str

    @abstractmethod
    def evaluate(self, test_case: TestCase, response: ModelResponse) -> EvaluationResult:
        raise NotImplementedError


class ExactMatchEvaluator(Evaluator):
    """Baseline evaluator: case-insensitive exact string match against the reference answer.

    This is a deliberately simple baseline that exists to validate the
    evaluation execution infrastructure end to end. Exact string matching
    does not measure semantic correctness, partial credit, or paraphrase
    equivalence, and must not be treated as a general-purpose answer
    quality metric. When a test case has no reference answer, this
    evaluator reports the comparison as not applicable rather than as a
    failure, since the platform explicitly supports reference-free
    evaluation.
    """

    name = "exact_match"

    def evaluate(self, test_case: TestCase, response: ModelResponse) -> EvaluationResult:
        if test_case.reference_answer is None:
            return EvaluationResult(
                evaluator_name=self.name,
                test_case_id=test_case.id,
                label="no_reference",
                explanation=(
                    "No reference answer was provided for this test case; "
                    "exact-match comparison is not applicable."
                ),
            )
        is_match = (
            response.output_text.strip().lower() == test_case.reference_answer.strip().lower()
        )
        return EvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case.id,
            score=1.0 if is_match else 0.0,
            passed=is_match,
            label="match" if is_match else "mismatch",
            explanation=(
                "Generated output exactly matched the reference answer "
                "(case-insensitive, whitespace-trimmed)."
                if is_match
                else "Generated output did not exactly match the reference answer."
            ),
        )

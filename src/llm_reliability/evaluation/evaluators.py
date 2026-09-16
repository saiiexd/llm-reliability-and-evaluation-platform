"""
Evaluators for the Core Evaluation Engine.

An evaluator inspects a single test case and the model response produced
for it, and returns a structured ``EvaluationResult``. Evaluators must not
depend on adapter internals or on other evaluators, so any number of them
can be run over the same response independently. Evaluation is strictly a
separate stage after model/application execution: an evaluator receives
the already-generated ``ModelResponse`` and must not modify the test case
or the response it was given.

``Evaluator.name`` identifies *how* an evaluator assesses a response (for
example, exact string match); ``Evaluator.criterion`` identifies *what* it
assesses (for example, correctness -- see
``llm_reliability.evaluation.criteria``). The same criterion may be
assessed by several different evaluators (see ``semantic_similarity.py``
and ``llm_judge.py``), which is what lets an experiment compare methods
rather than treating any single evaluator as ground truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_reliability.evaluation.criteria import CORRECTNESS
from llm_reliability.evaluation.models import (
    EvaluationResult,
    EvaluationStatus,
    ModelResponse,
    TestCase,
)
from llm_reliability.evaluation.normalization import normalize_for_exact_match


class Evaluator(ABC):
    """Interface for a single evaluation method.

    ``name`` identifies the evaluator in ``EvaluationResult.evaluator_name``
    and must be stable across runs. ``criterion`` identifies what the
    evaluator assesses (see ``llm_reliability.evaluation.criteria``).
    """

    name: str
    criterion: str

    @abstractmethod
    def evaluate(self, test_case: TestCase, response: ModelResponse) -> EvaluationResult:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        """Serializable configuration for this evaluator instance.

        Used to build a reproducibility record of what was actually run
        (see ``llm_reliability.experiments``), without serializing the
        evaluator object itself. The base implementation returns an empty
        mapping, which is correct for parameterless evaluators; an evaluator
        with constructor parameters that affect its behavior must override
        this to report them, since those parameters must be part of the
        experiment configuration fingerprint.
        """
        return {}


class ExactMatchEvaluator(Evaluator):
    """Baseline evaluator: configurable exact string match against the reference answer.

    This is a deliberately simple baseline that exists to validate the
    evaluation execution infrastructure end to end. Exact string matching
    does not measure semantic correctness, partial credit, or paraphrase
    equivalence, and must not be treated as a general-purpose answer
    quality metric -- see ``SemanticSimilarityEvaluator`` and
    ``LLMJudgeEvaluator`` for complementary methods. When a test case has
    no reference answer, this evaluator reports the comparison as skipped
    rather than as a failure, since the platform explicitly supports
    reference-free evaluation.

    Normalization is explicit and configurable, never silent:
    ``collapse_whitespace`` (default ``True``) collapses runs of
    whitespace to a single space and trims the ends; ``case_sensitive``
    (default ``False``) controls whether case differences count as a
    mismatch. No other transformation (punctuation removal, stemming,
    synonym handling) is ever applied. See
    ``llm_reliability.evaluation.normalization`` for the exact rules.
    """

    name = "exact_match"
    criterion = CORRECTNESS

    def __init__(self, case_sensitive: bool = False, collapse_whitespace: bool = True) -> None:
        self._case_sensitive = case_sensitive
        self._collapse_whitespace = collapse_whitespace

    def get_config(self) -> dict[str, Any]:
        return {
            "case_sensitive": self._case_sensitive,
            "collapse_whitespace": self._collapse_whitespace,
        }

    def evaluate(self, test_case: TestCase, response: ModelResponse) -> EvaluationResult:
        if test_case.reference_answer is None:
            return EvaluationResult(
                evaluator_name=self.name,
                test_case_id=test_case.id,
                status=EvaluationStatus.SKIPPED,
                criterion=self.criterion,
                label="no_reference",
                evaluator_config=self.get_config(),
                explanation=(
                    "No reference answer was provided for this test case; "
                    "exact-match comparison is not applicable."
                ),
            )

        normalized_output = normalize_for_exact_match(
            response.output_text,
            case_sensitive=self._case_sensitive,
            collapse_whitespace=self._collapse_whitespace,
        )
        normalized_reference = normalize_for_exact_match(
            test_case.reference_answer,
            case_sensitive=self._case_sensitive,
            collapse_whitespace=self._collapse_whitespace,
        )
        is_match = normalized_output == normalized_reference
        return EvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case.id,
            status=EvaluationStatus.SUCCESS,
            criterion=self.criterion,
            score=1.0 if is_match else 0.0,
            passed=is_match,
            label="match" if is_match else "mismatch",
            evaluator_config=self.get_config(),
            explanation=(
                f"Generated output {'exactly matched' if is_match else 'did not exactly match'} "
                "the reference answer under the configured normalization "
                f"(case_sensitive={self._case_sensitive}, "
                f"collapse_whitespace={self._collapse_whitespace})."
            ),
        )

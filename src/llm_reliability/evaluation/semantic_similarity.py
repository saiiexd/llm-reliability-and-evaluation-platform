"""
Semantic similarity evaluator.

Complements exact match by measuring contextual embedding similarity
between the generated answer and the reference answer, following BERTScore
(Zhang et al., 2020, "BERTScore: Evaluating Text Generation with BERT"),
one of the evaluation methods identified in this project's research plan.

Limitations that must always be kept in view when reading this evaluator's
output: high semantic similarity does NOT imply factual correctness,
completeness, or faithfulness to any source -- it only measures how similar
two texts are in a pretrained language model's embedding space. Two answers
can be highly similar in surface meaning while one is factually correct and
the other is not (for example, a paraphrase with a single number changed).
This evaluator must never be used as the sole basis for a correctness
judgment; it exists to be compared against exact match and LLM-as-a-judge,
not to replace them.

The real BERTScore computation is isolated behind a small backend interface
(``SimilarityBackend``) so that:
  1. the evaluator's contract, configuration, and error handling can be
     fully tested without the ``bert-score`` package or any model download
     (see ``FakeSimilarityBackend``), and
  2. a real backend can be swapped in without changing the evaluator.

``BertScoreBackend`` imports ``bert_score`` lazily, only inside
``compute_similarity``, never at module import time or at construction
time. This means constructing a ``SemanticSimilarityEvaluator`` and
inspecting its configuration never requires the dependency to be
installed, and never triggers a model download; only an actual call to
evaluate a response does, which is why ``BertScoreBackend`` is never
exercised by the standard, offline test suite (see
``tests/evaluation/test_semantic_similarity_integration.py``, which skips
itself when ``bert_score`` is not installed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_reliability.evaluation.criteria import CORRECTNESS
from llm_reliability.evaluation.evaluators import Evaluator
from llm_reliability.evaluation.models import (
    EvaluationResult,
    EvaluationStatus,
    ModelResponse,
    TestCase,
)


class SimilarityBackendError(Exception):
    """Raised when a similarity backend cannot produce a score in the current environment."""


class SimilarityBackend(ABC):
    """Computes a similarity score in [0, 1] between two texts."""

    @abstractmethod
    def compute_similarity(self, candidate: str, reference: str) -> float:
        """Return a similarity score in [0, 1]; raise ``SimilarityBackendError`` if it cannot."""
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Serializable configuration identifying this backend and its model."""
        raise NotImplementedError


class BertScoreBackend(SimilarityBackend):
    """Computes BERTScore F1 as the similarity score, via the ``bert-score`` package.

    Requires the optional ``semantic`` dependency group (``bert-score``,
    which in turn depends on ``torch`` and ``transformers``). The first
    real call downloads and caches a pretrained model unless one is
    already cached locally, which requires network access; this backend
    must therefore only ever be exercised by an explicitly opt-in
    integration test, never by the standard test suite.
    """

    def __init__(self, model_type: str = "distilbert-base-uncased", lang: str = "en") -> None:
        self._model_type = model_type
        self._lang = lang

    def get_config(self) -> dict[str, Any]:
        return {"backend": "bert_score", "model_type": self._model_type, "lang": self._lang}

    def compute_similarity(self, candidate: str, reference: str) -> float:
        try:
            import bert_score  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SimilarityBackendError(
                "The 'bert-score' package is required to compute real semantic "
                "similarity but is not installed in this environment. Install the "
                "'semantic' optional dependency group to enable it."
            ) from exc
        _, _, f1 = bert_score.score(
            [candidate], [reference], model_type=self._model_type, lang=self._lang, verbose=False
        )
        return float(f1[0])


class FakeSimilarityBackend(SimilarityBackend):
    """Deterministic fake backend for tests and local development.

    Performs no real embedding computation and must never be used to draw
    conclusions about actual semantic similarity. Identical strings
    (case-insensitive, whitespace-trimmed) always score 1.0, as a trivial
    and honest structural fact rather than a simulation of model behavior.
    Any other pair scores 0.0 unless explicitly configured in
    ``similarities``, or raises ``SimilarityBackendError`` if the pair is
    listed in ``failures``.
    """

    def __init__(
        self,
        similarities: dict[tuple[str, str], float] | None = None,
        failures: set[tuple[str, str]] | None = None,
    ) -> None:
        self._similarities = dict(similarities) if similarities else {}
        self._failures = set(failures) if failures else set()

    def get_config(self) -> dict[str, Any]:
        return {"backend": "fake"}

    def compute_similarity(self, candidate: str, reference: str) -> float:
        key = (candidate, reference)
        if key in self._failures:
            raise SimilarityBackendError("Simulated similarity backend failure.")
        if key in self._similarities:
            return self._similarities[key]
        if candidate.strip().lower() == reference.strip().lower():
            return 1.0
        return 0.0


class SemanticSimilarityEvaluator(Evaluator):
    """Reference-based semantic similarity evaluator.

    Reports a similarity score in [0, 1] between the generated answer and
    the reference answer, plus a categorical label derived from
    ``threshold``. See the module docstring for the limitations of
    similarity as a correctness proxy.
    """

    name = "semantic_similarity"
    criterion = CORRECTNESS

    def __init__(self, backend: SimilarityBackend, threshold: float = 0.85) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("SemanticSimilarityEvaluator.threshold must be between 0 and 1.")
        self._backend = backend
        self._threshold = threshold

    def get_config(self) -> dict[str, Any]:
        return {"threshold": self._threshold, **self._backend.get_config()}

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
                    "semantic similarity comparison is not applicable."
                ),
            )

        try:
            similarity = self._backend.compute_similarity(
                response.output_text, test_case.reference_answer
            )
        except SimilarityBackendError as exc:
            return EvaluationResult(
                evaluator_name=self.name,
                test_case_id=test_case.id,
                status=EvaluationStatus.INVALID_CONFIGURATION,
                criterion=self.criterion,
                evaluator_config=self.get_config(),
                error=str(exc),
            )

        passed = similarity >= self._threshold
        return EvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case.id,
            status=EvaluationStatus.SUCCESS,
            criterion=self.criterion,
            score=similarity,
            passed=passed,
            label="similar" if passed else "dissimilar",
            evaluator_config=self.get_config(),
            explanation=(
                f"Semantic similarity score {similarity:.4f} "
                f"{'meets' if passed else 'does not meet'} the configured threshold "
                f"({self._threshold}). This score does not by itself establish factual "
                "correctness, completeness, or faithfulness."
            ),
        )

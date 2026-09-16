"""
Retrieval evaluation: metrics scored against explicit ground truth.

Deliberately a separate result type (``RetrievalEvaluationResult``) from
the answer-quality ``EvaluationResult``
(``llm_reliability.evaluation.models``): retrieval quality and answer
quality are evaluated independently and must never be mixed into one
score. A ``RetrievalEvaluator`` runs after a RAG execution, over the
``TestCase`` (for ground truth, via
``llm_reliability.rag.ground_truth.get_relevant_chunk_ids``) and the
``ModelResponse`` it produced (for the retrieved chunks, via
``llm_reliability.rag.pipeline.extract_retrieved_chunks``).

Ground truth is not always available -- most test cases have none. When
it is unavailable, or when a response carries no retrieval evidence at
all (it was not produced by a RAG pipeline), every evaluator here reports
``SKIPPED`` rather than fabricating a score of zero or guessing.

Metrics implemented, precisely defined:

- **Hit@K**: 1.0 if at least one ground-truth relevant chunk id appears
  among the top K retrieved chunk ids, else 0.0. Answers "did retrieval
  surface *any* relevant evidence at all in the top K?"
- **Recall@K**: the fraction of all ground-truth relevant chunk ids that
  appear among the top K retrieved chunk ids (``|relevant intersect
  top_k| / |relevant|``). Answers "of everything relevant, how much did
  retrieval surface in the top K?" When exactly one relevant chunk id is
  given for a test case, Hit@K and Recall@K are numerically identical
  (both are 1.0 if that one chunk is retrieved, else 0.0); they diverge
  only when a test case has more than one relevant target.
- **Mean Reciprocal Rank (MRR)**: the reciprocal of the rank (1-based) of
  the first retrieved chunk that is in the ground-truth relevant set
  (``1 / rank``), or 0.0 if none of the retrieved chunks are relevant.
  Reported per test case; "mean" refers to averaging this value across
  many test cases, which is the responsibility of
  ``llm_reliability.rag.summary.summarize_retrieval``, not of this
  evaluator itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from llm_reliability.evaluation.models import ModelResponse, TestCase
from llm_reliability.rag.ground_truth import get_relevant_chunk_ids
from llm_reliability.rag.pipeline import extract_retrieved_chunks


class RetrievalEvaluationStatus(StrEnum):
    """Outcome of attempting to score one retrieval evaluator on one test case.

    ``SUCCESS``: a score was computed against available ground truth.
    ``SKIPPED``: no ground truth was available for this test case, or the
    response carried no retrieval evidence (it was not a RAG execution).
    Never interpreted as a failing score.
    ``EXECUTION_ERROR``: the evaluator raised an unexpected error.
    """

    SUCCESS = "success"
    SKIPPED = "skipped"
    EXECUTION_ERROR = "execution_error"


@dataclass
class RetrievalEvaluationResult:
    """Structured output of one retrieval evaluator run against one test case.

    Kept entirely separate from ``EvaluationResult`` (answer quality): a
    ``score`` here is a retrieval metric (Hit@K, Recall@K, MRR), never an
    answer-correctness score.
    """

    evaluator_name: str
    test_case_id: str
    metric_name: str
    status: RetrievalEvaluationStatus
    score: float | None = None
    relevant_targets: list[str] | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    explanation: str | None = None
    error: str | None = None
    evaluator_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "test_case_id": self.test_case_id,
            "metric_name": self.metric_name,
            "status": self.status.value,
            "score": self.score,
            "relevant_targets": self.relevant_targets,
            "retrieved_ids": self.retrieved_ids,
            "explanation": self.explanation,
            "error": self.error,
            "evaluator_config": self.evaluator_config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalEvaluationResult:
        return cls(
            evaluator_name=data["evaluator_name"],
            test_case_id=data["test_case_id"],
            metric_name=data["metric_name"],
            status=RetrievalEvaluationStatus(data["status"]),
            score=data.get("score"),
            relevant_targets=data.get("relevant_targets"),
            retrieved_ids=list(data.get("retrieved_ids", [])),
            explanation=data.get("explanation"),
            error=data.get("error"),
            evaluator_config=dict(data.get("evaluator_config", {})),
        )


class RetrievalEvaluator(ABC):
    """Interface for a single retrieval evaluation metric."""

    name: str
    metric_name: str

    @abstractmethod
    def evaluate(self, test_case: TestCase, response: ModelResponse) -> RetrievalEvaluationResult:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}

    def _unavailable(self, test_case_id: str, reason: str) -> RetrievalEvaluationResult:
        return RetrievalEvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case_id,
            metric_name=self.metric_name,
            status=RetrievalEvaluationStatus.SKIPPED,
            explanation=reason,
            evaluator_config=self.get_config(),
        )

    def _prepare(
        self, test_case: TestCase, response: ModelResponse
    ) -> tuple[list[str], list[str]] | RetrievalEvaluationResult:
        """Resolve ground truth and retrieved ids, or a SKIPPED result explaining why not."""
        relevant = get_relevant_chunk_ids(test_case)
        if relevant is None:
            return self._unavailable(
                test_case.id, "No retrieval ground truth was provided for this test case."
            )
        if not relevant:
            return self._unavailable(
                test_case.id, "Retrieval ground truth for this test case is an empty list."
            )
        retrieved_chunks = extract_retrieved_chunks(response)
        if retrieved_chunks is None:
            return self._unavailable(
                test_case.id,
                "This response carries no retrieval evidence (it was not produced by a RAG "
                "pipeline).",
            )
        retrieved_ids = [chunk.chunk_id for chunk in retrieved_chunks]
        return relevant, retrieved_ids


class HitAtKEvaluator(RetrievalEvaluator):
    """Hit@K: 1.0 if any relevant target is retrieved in the top K, else 0.0."""

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError("HitAtKEvaluator.k must be a positive integer.")
        self._k = k
        self.name = f"hit_at_{k}"
        self.metric_name = "hit_at_k"

    def get_config(self) -> dict[str, Any]:
        return {"k": self._k}

    def evaluate(self, test_case: TestCase, response: ModelResponse) -> RetrievalEvaluationResult:
        prepared = self._prepare(test_case, response)
        if isinstance(prepared, RetrievalEvaluationResult):
            return prepared
        relevant, retrieved_ids = prepared
        top_k = retrieved_ids[: self._k]
        hit = any(rid in relevant for rid in top_k)
        return RetrievalEvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case.id,
            metric_name=self.metric_name,
            status=RetrievalEvaluationStatus.SUCCESS,
            score=1.0 if hit else 0.0,
            relevant_targets=relevant,
            retrieved_ids=top_k,
            evaluator_config=self.get_config(),
            explanation=(
                f"At least one relevant target was found in the top {self._k} retrieved chunks."
                if hit
                else f"No relevant target was found in the top {self._k} retrieved chunks."
            ),
        )


class RecallAtKEvaluator(RetrievalEvaluator):
    """Recall@K: fraction of relevant targets retrieved in the top K."""

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError("RecallAtKEvaluator.k must be a positive integer.")
        self._k = k
        self.name = f"recall_at_{k}"
        self.metric_name = "recall_at_k"

    def get_config(self) -> dict[str, Any]:
        return {"k": self._k}

    def evaluate(self, test_case: TestCase, response: ModelResponse) -> RetrievalEvaluationResult:
        prepared = self._prepare(test_case, response)
        if isinstance(prepared, RetrievalEvaluationResult):
            return prepared
        relevant, retrieved_ids = prepared
        top_k = retrieved_ids[: self._k]
        found = [rid for rid in relevant if rid in top_k]
        score = len(found) / len(relevant)
        return RetrievalEvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case.id,
            metric_name=self.metric_name,
            status=RetrievalEvaluationStatus.SUCCESS,
            score=score,
            relevant_targets=relevant,
            retrieved_ids=top_k,
            evaluator_config=self.get_config(),
            explanation=(
                f"Retrieved {len(found)} of {len(relevant)} relevant targets in the top {self._k}."
            ),
        )


class MeanReciprocalRankEvaluator(RetrievalEvaluator):
    """Reciprocal rank of the first relevant retrieved chunk, for this one test case.

    Averaging this value across test cases (the "mean" in MRR) is done by
    ``llm_reliability.rag.summary.summarize_retrieval``, not here.
    """

    name = "mrr"
    metric_name = "mrr"

    def evaluate(self, test_case: TestCase, response: ModelResponse) -> RetrievalEvaluationResult:
        prepared = self._prepare(test_case, response)
        if isinstance(prepared, RetrievalEvaluationResult):
            return prepared
        relevant, retrieved_ids = prepared
        rank = next((i + 1 for i, rid in enumerate(retrieved_ids) if rid in relevant), None)
        score = 1.0 / rank if rank is not None else 0.0
        return RetrievalEvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case.id,
            metric_name=self.metric_name,
            status=RetrievalEvaluationStatus.SUCCESS,
            score=score,
            relevant_targets=relevant,
            retrieved_ids=retrieved_ids,
            evaluator_config=self.get_config(),
            explanation=(
                f"First relevant chunk found at rank {rank}."
                if rank is not None
                else "No relevant chunk was found among the retrieved results."
            ),
        )


def evaluate_retrieval(
    test_cases: list[TestCase],
    responses_by_test_case_id: dict[str, ModelResponse],
    evaluators: list[RetrievalEvaluator],
) -> dict[str, list[RetrievalEvaluationResult]]:
    """Run every retrieval evaluator against every test case that has a response.

    Read-only: never re-runs retrieval or generation. A single evaluator
    raising unexpectedly is isolated to that evaluator/test-case pair
    (recorded as ``EXECUTION_ERROR``) and never prevents the others from
    producing results, mirroring ``EvaluationRunner``'s failure isolation.
    """
    results: dict[str, list[RetrievalEvaluationResult]] = {}
    for test_case in test_cases:
        response = responses_by_test_case_id.get(test_case.id)
        if response is None:
            continue
        test_case_results: list[RetrievalEvaluationResult] = []
        for evaluator in evaluators:
            try:
                test_case_results.append(evaluator.evaluate(test_case, response))
            except Exception as exc:
                test_case_results.append(
                    RetrievalEvaluationResult(
                        evaluator_name=getattr(evaluator, "name", evaluator.__class__.__name__),
                        test_case_id=test_case.id,
                        metric_name=getattr(evaluator, "metric_name", "unknown"),
                        status=RetrievalEvaluationStatus.EXECUTION_ERROR,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        results[test_case.id] = test_case_results
    return results

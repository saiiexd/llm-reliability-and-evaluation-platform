"""
ContextSupportBaseline: a deliberately narrow, literal-overlap check.

This is NOT a hallucination detector and does NOT establish factual
faithfulness or groundedness. It measures one specific, simple thing:
what fraction of the unique word tokens in a generated answer also appear
somewhere in the retrieved context. String overlap does not imply
semantic support (an answer can repeat context words while still drawing
an unsupported conclusion from them), and its absence does not imply
contradiction (a correct paraphrase can share few words with its source).
This evaluator exists only to establish the data flow -- answer text
alongside retrieved context -- that a later, more rigorous faithfulness
evaluator (NLI-based verification, claim extraction, or an LLM-as-a-judge
faithfulness rubric) will need. Do not read a high score here as evidence
of correctness, and do not read a low score here as evidence of
hallucination.

Reuses the existing ``Evaluator`` interface and ``EvaluationResult`` model
from ``llm_reliability.evaluation`` unchanged: this is an answer-quality
evaluator (it assesses the generated answer), not a retrieval evaluator,
so it belongs in that family, not in
``llm_reliability.rag.evaluators.RetrievalEvaluator``.
"""

from __future__ import annotations

import re
from typing import Any

from llm_reliability.evaluation.criteria import CONTEXT_SUPPORT
from llm_reliability.evaluation.evaluators import Evaluator
from llm_reliability.evaluation.models import (
    EvaluationResult,
    EvaluationStatus,
    ModelResponse,
    TestCase,
)
from llm_reliability.rag.pipeline import extract_retrieved_chunks

_WORD_RE = re.compile(r"\w+")


def _word_set(text: str, case_sensitive: bool) -> set[str]:
    normalized = text if case_sensitive else text.lower()
    return set(_WORD_RE.findall(normalized))


def compute_overlap_ratio(answer: str, context: str, case_sensitive: bool = False) -> float:
    """Fraction of unique word tokens in ``answer`` that also appear in ``context``.

    Returns 0.0 if the answer contains no word tokens. Tokens are matched
    by ``\\w+`` (letters, digits, underscore); no stemming, normalization
    beyond optional case folding, or synonym handling is applied.
    """
    answer_words = _word_set(answer, case_sensitive)
    if not answer_words:
        return 0.0
    context_words = _word_set(context, case_sensitive)
    return len(answer_words & context_words) / len(answer_words)


class ContextSupportBaseline(Evaluator):
    """Baseline literal word-overlap check between an answer and its retrieved context.

    See the module docstring for what this does and does not measure.
    Skips (rather than fails) when the response carries no retrieval
    evidence -- it was not produced by a RAG pipeline.
    """

    name = "context_support_baseline"
    criterion = CONTEXT_SUPPORT

    def __init__(self, min_overlap_ratio: float = 0.3, case_sensitive: bool = False) -> None:
        if not 0.0 <= min_overlap_ratio <= 1.0:
            raise ValueError("ContextSupportBaseline.min_overlap_ratio must be between 0 and 1.")
        self._min_overlap_ratio = min_overlap_ratio
        self._case_sensitive = case_sensitive

    def get_config(self) -> dict[str, Any]:
        return {
            "min_overlap_ratio": self._min_overlap_ratio,
            "case_sensitive": self._case_sensitive,
        }

    def evaluate(self, test_case: TestCase, response: ModelResponse) -> EvaluationResult:
        retrieved_chunks = extract_retrieved_chunks(response)
        if retrieved_chunks is None:
            return EvaluationResult(
                evaluator_name=self.name,
                test_case_id=test_case.id,
                status=EvaluationStatus.SKIPPED,
                criterion=self.criterion,
                evaluator_config=self.get_config(),
                explanation=(
                    "This response carries no retrieved context (it was not produced by a "
                    "RAG pipeline); context support is not applicable."
                ),
            )

        context_text = " ".join(chunk.text for chunk in retrieved_chunks)
        overlap_ratio = compute_overlap_ratio(
            response.output_text, context_text, self._case_sensitive
        )
        supported = overlap_ratio >= self._min_overlap_ratio
        return EvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case.id,
            status=EvaluationStatus.SUCCESS,
            criterion=self.criterion,
            score=overlap_ratio,
            passed=supported,
            label="context_supported" if supported else "context_unsupported",
            evaluator_config=self.get_config(),
            explanation=(
                f"{overlap_ratio:.2f} of the answer's unique word tokens appear in the "
                f"retrieved context (threshold {self._min_overlap_ratio}). This measures "
                "literal word overlap only -- it is not a faithfulness or hallucination "
                "determination."
            ),
        )

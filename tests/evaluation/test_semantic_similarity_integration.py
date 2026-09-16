"""
Opt-in integration test for the real BERTScore backend.

This test is NOT part of the standard offline test suite: it is skipped
automatically unless the optional 'bert-score' package is installed
(``pip install ".[semantic]"``). Even when installed, the first real call
downloads and caches a pretrained model, which requires network access and
can take significant time; this is exactly why the behavioral tests in
test_semantic_similarity.py use FakeSimilarityBackend instead.

Run explicitly with:
    pip install ".[semantic]"
    pytest tests/evaluation/test_semantic_similarity_integration.py -v

This test only checks structural correctness and directionally expected
behavior (a paraphrase should score higher than an unrelated sentence); it
does not assert fixed numerical thresholds, since exact BERTScore values
depend on the installed model version and are not something this project
controls or should hard-code as if they were a specification.
"""

from __future__ import annotations

import importlib.util

import pytest

from llm_reliability.evaluation import EvaluationStatus, ModelResponse, TestCase
from llm_reliability.evaluation.semantic_similarity import (
    BertScoreBackend,
    SemanticSimilarityEvaluator,
)

_BERT_SCORE_AVAILABLE = importlib.util.find_spec("bert_score") is not None

pytestmark = pytest.mark.skipif(
    not _BERT_SCORE_AVAILABLE,
    reason="bert-score is not installed; install the 'semantic' extra to run this integration test",
)


def test_real_bertscore_ranks_paraphrase_above_unrelated_text():
    reference = "Paris is the capital of France."
    paraphrase = "The capital city of France is Paris."
    unrelated = "The mitochondria is the powerhouse of the cell."

    backend = BertScoreBackend()
    paraphrase_score = backend.compute_similarity(paraphrase, reference)
    unrelated_score = backend.compute_similarity(unrelated, reference)

    assert 0.0 <= paraphrase_score <= 1.0
    assert 0.0 <= unrelated_score <= 1.0
    assert paraphrase_score > unrelated_score


def test_real_bertscore_through_the_evaluator_contract():
    reference = "Paris is the capital of France."
    test_case = TestCase(
        id="t1", input="What is the capital of France?", reference_answer=reference
    )
    response = ModelResponse(
        test_case_id="t1", output_text="The capital city of France is Paris.", model_id="test-model"
    )

    evaluator = SemanticSimilarityEvaluator(backend=BertScoreBackend(), threshold=0.85)
    result = evaluator.evaluate(test_case, response)

    assert result.status == EvaluationStatus.SUCCESS
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

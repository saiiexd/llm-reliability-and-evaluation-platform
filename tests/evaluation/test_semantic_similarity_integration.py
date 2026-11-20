"""
Opt-in integration test for the real BERTScore backend.

This test is NOT part of the standard offline test suite: it requires both
the optional 'bert-score' package to be installed (``pip install
".[semantic]"``) AND the LLM_RELIABILITY_RUN_INTEGRATION_TESTS environment
variable to be set to "1", so it never runs merely because the package
happens to be present. Even when installed, the first real call downloads
and caches a pretrained model, which requires network access and can take
significant time; this is exactly why the behavioral tests in
test_semantic_similarity.py use FakeSimilarityBackend instead.

Run explicitly with (CPU-only torch first, to avoid the CUDA wheels):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install ".[semantic]"
    export LLM_RELIABILITY_RUN_INTEGRATION_TESTS=1
    pytest tests/evaluation/test_semantic_similarity_integration.py -v

These tests only check structural correctness and directionally expected
behavior (identical text scores ~1, a paraphrase scores higher than an
unrelated sentence, an empty answer scores 0); they do not assert fixed
numerical thresholds, since exact BERTScore values depend on the installed
model version and are not something this project controls or should
hard-code as if they were a specification.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from llm_reliability.evaluation import EvaluationStatus, ModelResponse, TestCase
from llm_reliability.evaluation.semantic_similarity import (
    BertScoreBackend,
    SemanticSimilarityEvaluator,
)

_BERT_SCORE_AVAILABLE = importlib.util.find_spec("bert_score") is not None
_INTEGRATION_TESTS_ENABLED = os.environ.get("LLM_RELIABILITY_RUN_INTEGRATION_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not (_BERT_SCORE_AVAILABLE and _INTEGRATION_TESTS_ENABLED),
    reason=(
        "opt-in integration test: requires bert-score installed and "
        "LLM_RELIABILITY_RUN_INTEGRATION_TESTS=1"
    ),
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


def test_real_bertscore_identical_text_scores_one_within_range():
    backend = BertScoreBackend()
    for text in ["Paris is the capital of France.", "co-NP"]:
        score = backend.compute_similarity(text, text)
        # Raw float32 output can be 1.0000001; the backend clamps it to the contract.
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(1.0, abs=1e-5)


def test_real_bertscore_empty_answer_scores_zero_and_fails():
    test_case = TestCase(id="t1", input="Capital of France?", reference_answer="Paris")
    response = ModelResponse(test_case_id="t1", output_text="  ", model_id="test-model")

    result = SemanticSimilarityEvaluator(backend=BertScoreBackend()).evaluate(test_case, response)

    assert result.status == EvaluationStatus.SUCCESS
    assert result.score == 0.0
    assert result.passed is False


def test_real_bertscore_uses_exactly_the_recorded_configuration():
    """The cached scorer must score like a direct bert_score.score call with the recorded config."""
    import bert_score
    from bert_score.utils import model2layers

    backend = BertScoreBackend()
    config = backend.get_config()
    # The pinned default layer is bert-score's own recommendation for this model.
    assert config["num_layers"] == model2layers[config["model_type"]]

    candidates = ["The capital city of France is Paris.", "1979"]
    references = ["Paris is the capital of France.", "1997"]
    _, _, direct_f1 = bert_score.score(
        candidates,
        references,
        model_type=config["model_type"],
        num_layers=config["num_layers"],
        lang=config["lang"],
        idf=config["idf"],
        rescale_with_baseline=config["rescale_with_baseline"],
        verbose=False,
    )
    for candidate, reference, expected in zip(candidates, references, direct_f1, strict=True):
        assert backend.compute_similarity(candidate, reference) == pytest.approx(
            float(expected), abs=1e-6
        )

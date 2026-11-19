"""
Tests for SemanticSimilarityEvaluator using the deterministic fake backend.

These tests validate the evaluator's contract, configuration, and error
handling -- not real BERTScore behavior. They use FakeSimilarityBackend,
which performs no embedding computation at all, and are structured around
four controlled relationships between a candidate and a reference answer
(identical, a configured "paraphrase"-level score, a configured
"partial overlap"-level score, and an unrelated/default score), each of
which is an arbitrary, explicitly configured number, not a measurement.
See test_semantic_similarity_integration.py for the real BERTScore
backend, which is opt-in and skipped unless the dependency is installed.
"""

import sys
import types

import pytest

from llm_reliability.evaluation import EvaluationStatus, ModelResponse, TestCase
from llm_reliability.evaluation.criteria import CORRECTNESS
from llm_reliability.evaluation.semantic_similarity import (
    BertScoreBackend,
    FakeSimilarityBackend,
    SemanticSimilarityEvaluator,
    SimilarityBackendError,
)

REFERENCE = "Paris is the capital of France."
PARAPHRASE = "France's capital city is Paris."
PARTIAL_OVERLAP = "Paris is a major city in France."
UNRELATED = "The mitochondria is the powerhouse of the cell."


def _response(text: str) -> ModelResponse:
    return ModelResponse(test_case_id="t1", output_text=text, model_id="mock-adapter-v1")


def _test_case(reference_answer: str | None = REFERENCE) -> TestCase:
    return TestCase(
        id="t1", input="What is the capital of France?", reference_answer=reference_answer
    )


class TestFakeSimilarityBackend:
    def test_identical_text_scores_one(self):
        backend = FakeSimilarityBackend()
        assert backend.compute_similarity(REFERENCE, REFERENCE) == 1.0

    def test_identical_text_is_case_and_whitespace_insensitive(self):
        backend = FakeSimilarityBackend()
        assert backend.compute_similarity(f"  {REFERENCE.upper()}  ", REFERENCE) == 1.0

    def test_unconfigured_pair_defaults_to_zero(self):
        backend = FakeSimilarityBackend()
        assert backend.compute_similarity(UNRELATED, REFERENCE) == 0.0

    def test_configured_pair_returns_configured_score(self):
        backend = FakeSimilarityBackend(similarities={(PARAPHRASE, REFERENCE): 0.93})
        assert backend.compute_similarity(PARAPHRASE, REFERENCE) == 0.93

    def test_configured_failure_raises(self):
        backend = FakeSimilarityBackend(failures={(PARAPHRASE, REFERENCE)})
        try:
            backend.compute_similarity(PARAPHRASE, REFERENCE)
            raise AssertionError("expected SimilarityBackendError")
        except SimilarityBackendError:
            pass


class TestSemanticSimilarityEvaluator:
    def test_semantically_equivalent_paraphrase_passes_threshold(self):
        backend = FakeSimilarityBackend(similarities={(PARAPHRASE, REFERENCE): 0.93})
        evaluator = SemanticSimilarityEvaluator(backend=backend, threshold=0.85)
        result = evaluator.evaluate(_test_case(), _response(PARAPHRASE))
        assert result.status == EvaluationStatus.SUCCESS
        assert result.score == 0.93
        assert result.passed is True
        assert result.label == "similar"
        assert result.criterion == CORRECTNESS

    def test_partially_overlapping_answer_below_threshold_fails(self):
        backend = FakeSimilarityBackend(similarities={(PARTIAL_OVERLAP, REFERENCE): 0.55})
        evaluator = SemanticSimilarityEvaluator(backend=backend, threshold=0.85)
        result = evaluator.evaluate(_test_case(), _response(PARTIAL_OVERLAP))
        assert result.score == 0.55
        assert result.passed is False
        assert result.label == "dissimilar"

    def test_unrelated_answer_scores_low_and_fails(self):
        backend = FakeSimilarityBackend()
        evaluator = SemanticSimilarityEvaluator(backend=backend, threshold=0.85)
        result = evaluator.evaluate(_test_case(), _response(UNRELATED))
        assert result.score == 0.0
        assert result.passed is False

    def test_identical_answer_passes(self):
        backend = FakeSimilarityBackend()
        evaluator = SemanticSimilarityEvaluator(backend=backend, threshold=0.85)
        result = evaluator.evaluate(_test_case(), _response(REFERENCE))
        assert result.score == 1.0
        assert result.passed is True

    def test_missing_reference_is_skipped_not_a_failure(self):
        backend = FakeSimilarityBackend()
        evaluator = SemanticSimilarityEvaluator(backend=backend)
        result = evaluator.evaluate(_test_case(reference_answer=None), _response(REFERENCE))
        assert result.status == EvaluationStatus.SKIPPED
        assert result.label == "no_reference"
        assert result.score is None
        assert result.passed is None

    def test_backend_failure_is_reported_as_invalid_configuration(self):
        backend = FakeSimilarityBackend(failures={(REFERENCE, REFERENCE)})
        evaluator = SemanticSimilarityEvaluator(backend=backend)
        result = evaluator.evaluate(_test_case(), _response(REFERENCE))
        assert result.status == EvaluationStatus.INVALID_CONFIGURATION
        assert result.error is not None
        assert result.score is None

    def test_threshold_out_of_range_is_rejected(self):
        try:
            SemanticSimilarityEvaluator(backend=FakeSimilarityBackend(), threshold=1.5)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_get_config_includes_threshold_and_backend_config(self):
        evaluator = SemanticSimilarityEvaluator(backend=FakeSimilarityBackend(), threshold=0.7)
        config = evaluator.get_config()
        assert config["threshold"] == 0.7
        assert config["backend"] == "fake"

    def test_result_carries_its_own_configuration(self):
        backend = FakeSimilarityBackend()
        evaluator = SemanticSimilarityEvaluator(backend=backend, threshold=0.7)
        result = evaluator.evaluate(_test_case(), _response(REFERENCE))
        assert result.evaluator_config == evaluator.get_config()


class TestBertScoreBackendWithoutTheRealModel:
    """Configuration and guard paths of the real backend; no model is loaded."""

    def test_default_config_records_every_scoring_setting(self):
        assert BertScoreBackend().get_config() == {
            "backend": "bert_score",
            "model_type": "distilbert-base-uncased",
            "lang": "en",
            "num_layers": 5,
            "idf": False,
            "rescale_with_baseline": False,
            "score": "f1",
        }

    def test_explicit_num_layers_is_recorded(self):
        config = BertScoreBackend(model_type="roberta-large", num_layers=17).get_config()
        assert config["model_type"] == "roberta-large"
        assert config["num_layers"] == 17

    def test_unknown_model_without_num_layers_records_none(self):
        assert BertScoreBackend(model_type="some-other-model").get_config()["num_layers"] is None

    def test_non_positive_num_layers_is_rejected(self):
        with pytest.raises(ValueError):
            BertScoreBackend(num_layers=0)

    def test_missing_package_is_an_invalid_configuration_not_a_crash(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "bert_score", None)  # makes the import fail
        evaluator = SemanticSimilarityEvaluator(backend=BertScoreBackend())
        result = evaluator.evaluate(_test_case(), _response(PARAPHRASE))
        assert result.status == EvaluationStatus.INVALID_CONFIGURATION
        assert result.passed is None
        assert "bert-score" in (result.error or "")

    @pytest.mark.parametrize("candidate", ["", "   ", "\n"])
    def test_empty_candidate_scores_zero_without_loading_a_model(self, monkeypatch, candidate):
        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("an empty candidate must not reach the scorer")

        fake_module = types.SimpleNamespace(BERTScorer=_must_not_be_called)
        monkeypatch.setitem(sys.modules, "bert_score", fake_module)
        evaluator = SemanticSimilarityEvaluator(backend=BertScoreBackend())
        result = evaluator.evaluate(_test_case(), _response(candidate))
        assert result.status == EvaluationStatus.SUCCESS
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.parametrize(
        ("load_error", "message"),
        [
            (OSError("couldn't connect to the model hub"), "Could not load BERTScore model"),
            (KeyError("some-other-model"), "pass num_layers explicitly"),
            (AssertionError("Invalid num_layers"), "Invalid BERTScore configuration"),
        ],
    )
    def test_model_load_failure_is_an_invalid_configuration(self, monkeypatch, load_error, message):
        def _failing_scorer(*args, **kwargs):
            raise load_error

        monkeypatch.setitem(
            sys.modules, "bert_score", types.SimpleNamespace(BERTScorer=_failing_scorer)
        )
        evaluator = SemanticSimilarityEvaluator(
            backend=BertScoreBackend(model_type="some-other-model")
        )
        result = evaluator.evaluate(_test_case(), _response(PARAPHRASE))
        assert result.status == EvaluationStatus.INVALID_CONFIGURATION
        assert result.passed is None
        assert message in (result.error or "")


@pytest.mark.parametrize("reference", ["", "   "])
def test_blank_reference_is_skipped_not_scored(reference):
    evaluator = SemanticSimilarityEvaluator(backend=FakeSimilarityBackend())
    result = evaluator.evaluate(
        TestCase(id="t", input="q", reference_answer=reference), _response("")
    )
    assert result.status == EvaluationStatus.SKIPPED
    assert result.passed is None

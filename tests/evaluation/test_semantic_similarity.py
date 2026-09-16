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

from llm_reliability.evaluation import EvaluationStatus, ModelResponse, TestCase
from llm_reliability.evaluation.criteria import CORRECTNESS
from llm_reliability.evaluation.semantic_similarity import (
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

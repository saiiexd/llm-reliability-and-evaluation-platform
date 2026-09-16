"""Tests for cosine similarity."""

import pytest

from llm_reliability.rag.similarity import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_scale_invariant(self):
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_vectors_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            cosine_similarity([], [])

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="undefined"):
            cosine_similarity([0.0, 0.0], [1.0, 1.0])

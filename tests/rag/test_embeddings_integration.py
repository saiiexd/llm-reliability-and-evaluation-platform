"""
Opt-in integration test for the real Sentence Transformers embedding backend.

This test is NOT part of the standard offline test suite. It requires
both 'sentence-transformers' to be importable AND the
LLM_RELIABILITY_RUN_INTEGRATION_TESTS environment variable to be set to
"1", so that it never runs merely because the package happens to be
present in a shared environment (as it is in this project's development
environment, for reasons unrelated to this project). Even when the
package is installed, the first real call downloads and caches a
pretrained model, which requires network access; this is exactly why the
behavioral tests in test_embeddings.py and test_retriever.py use
FakeEmbeddingModel instead.

Run explicitly with:
    LLM_RELIABILITY_RUN_INTEGRATION_TESTS=1 pytest tests/rag/test_embeddings_integration.py -v

This test only checks structural correctness and directionally expected
behavior (semantically related sentences should be more similar than
unrelated ones); it does not assert fixed numerical thresholds, since
exact embedding values depend on the installed model version.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from llm_reliability.rag.embeddings import SentenceTransformerEmbeddingModel
from llm_reliability.rag.similarity import cosine_similarity

_SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
_INTEGRATION_TESTS_ENABLED = os.environ.get("LLM_RELIABILITY_RUN_INTEGRATION_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not (_SENTENCE_TRANSFORMERS_AVAILABLE and _INTEGRATION_TESTS_ENABLED),
    reason=(
        "opt-in integration test: requires sentence-transformers installed and "
        "LLM_RELIABILITY_RUN_INTEGRATION_TESTS=1"
    ),
)


def test_real_embeddings_rank_related_text_above_unrelated_text():
    query = "What is the capital of France?"
    related = "Paris is the capital city of France."
    unrelated = "The mitochondria is the powerhouse of the cell."

    model = SentenceTransformerEmbeddingModel()
    query_vector, related_vector, unrelated_vector = model.embed([query, related, unrelated])

    related_score = cosine_similarity(query_vector, related_vector)
    unrelated_score = cosine_similarity(query_vector, unrelated_vector)

    assert related_score > unrelated_score

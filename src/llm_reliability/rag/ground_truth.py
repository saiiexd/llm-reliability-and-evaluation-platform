"""
Retrieval ground truth on test cases.

Rather than adding a new required field to ``TestCase`` (from
``llm_reliability.evaluation``), retrieval ground truth is stored under a
well-known key in the existing ``TestCase.metadata`` dict:
``"relevant_chunk_ids"``, a list of chunk (or document) identifiers that
are considered relevant for that test case's question. This reuses the
existing, unmodified ``TestCase`` model exactly as designed -- metadata
was already the documented escape hatch for information that does not
warrant a first-class field -- rather than creating a parallel test case
representation for RAG.

Absence of this key is a normal, expected state (most test cases have no
retrieval ground truth) and must always be treated as "ground truth
unavailable," never as "zero relevant chunks."
"""

from __future__ import annotations

from llm_reliability.evaluation.models import TestCase

RELEVANT_CHUNK_IDS_KEY = "relevant_chunk_ids"


def get_relevant_chunk_ids(test_case: TestCase) -> list[str] | None:
    """Return the test case's ground-truth relevant chunk ids, or ``None`` if unavailable.

    ``None`` means no ground truth was provided for this test case, which
    is distinct from (and must never be conflated with) an empty list.
    """
    value = test_case.metadata.get(RELEVANT_CHUNK_IDS_KEY)
    if value is None:
        return None
    return list(value)

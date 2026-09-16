"""
Vector similarity for retrieval.

Cosine similarity only, implemented transparently with the standard
library (no numpy dependency for something this simple): the similarity
between two vectors is the cosine of the angle between them, in [-1, 1],
computed as their dot product divided by the product of their magnitudes.
No vector database or approximate nearest-neighbor index is used; the
retriever that calls this module scores every candidate exactly, which is
appropriate for the small, controlled corpora this milestone targets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity between vectors ``a`` and ``b``.

    Raises ``ValueError`` if the vectors have different dimensions, if
    either is empty, or if either is a zero vector (cosine similarity is
    undefined when the magnitude is zero).
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}.")
    if len(a) == 0:
        raise ValueError("Vectors must be non-empty.")

    dot_product = sum(x * y for x, y in zip(a, b, strict=True))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    return dot_product / (magnitude_a * magnitude_b)

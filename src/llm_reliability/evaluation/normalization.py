"""
Explicit text normalization rules for evaluators.

Each function documents exactly what it changes, so an evaluator can name
which rules were applied (via its configuration) rather than performing
silent, undocumented text transformation. Nothing here performs stemming,
punctuation removal, synonym replacement, or any transformation that could
change the meaning of the compared text -- those are out of scope for a
baseline exact-match metric and would make its behavior harder to reason
about, not more useful.
"""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse every run of whitespace (including newlines and tabs) to a single space.

    Also strips leading and trailing whitespace.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_case(text: str) -> str:
    """Fold text to lowercase using Python's default Unicode case folding."""
    return text.lower()


def normalize_for_exact_match(text: str, *, case_sensitive: bool, collapse_whitespace: bool) -> str:
    """Apply exactly the normalization steps an ExactMatchEvaluator is configured to use.

    Order is fixed and documented: whitespace collapsing happens first (if
    enabled), then case folding (if not case-sensitive). Punctuation and
    all other characters are left untouched.
    """
    result = text
    if collapse_whitespace:
        result = normalize_whitespace(result)
    if not case_sensitive:
        result = normalize_case(result)
    return result

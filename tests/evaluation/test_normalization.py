"""Tests for explicit text normalization rules."""

from llm_reliability.evaluation.normalization import (
    normalize_case,
    normalize_for_exact_match,
    normalize_whitespace,
)


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("a    b") == "a b"

    def test_collapses_tabs_and_newlines(self):
        assert normalize_whitespace("a\t\tb\n\nc") == "a b c"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_whitespace("  a b  ") == "a b"

    def test_does_not_alter_already_normalized_text(self):
        assert normalize_whitespace("a b c") == "a b c"


class TestNormalizeCase:
    def test_lowercases_text(self):
        assert normalize_case("PARIS") == "paris"

    def test_leaves_lowercase_text_unchanged(self):
        assert normalize_case("paris") == "paris"


class TestNormalizeForExactMatch:
    def test_default_normalizes_both_case_and_whitespace(self):
        result = normalize_for_exact_match(
            "  Paris   is  Nice  ", case_sensitive=False, collapse_whitespace=True
        )
        assert result == "paris is nice"

    def test_case_sensitive_preserves_case(self):
        result = normalize_for_exact_match("Paris", case_sensitive=True, collapse_whitespace=True)
        assert result == "Paris"

    def test_collapse_whitespace_false_preserves_internal_spacing(self):
        result = normalize_for_exact_match(
            "Paris  is nice", case_sensitive=True, collapse_whitespace=False
        )
        assert result == "Paris  is nice"

    def test_does_not_remove_punctuation(self):
        result = normalize_for_exact_match(
            "Paris, France.", case_sensitive=False, collapse_whitespace=True
        )
        assert result == "paris, france."

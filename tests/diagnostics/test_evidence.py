"""Tests for statement splitting and the lexical evidence baseline."""

from llm_reliability.diagnostics.evidence import (
    LEXICAL_BASELINE_METHOD,
    assess_statements_lexically,
    split_into_statements,
)
from llm_reliability.diagnostics.taxonomy import ClaimSupportStatus


class TestSplitIntoStatements:
    def test_splits_on_sentence_boundaries(self):
        assert split_into_statements("Paris is in France. It is the capital.") == [
            "Paris is in France.",
            "It is the capital.",
        ]

    def test_single_sentence_returns_one_statement(self):
        assert split_into_statements("Paris is the capital.") == ["Paris is the capital."]

    def test_empty_text_returns_empty_list(self):
        assert split_into_statements("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert split_into_statements("   ") == []


class TestAssessStatementsLexically:
    def test_supported_statement(self):
        assessments = assess_statements_lexically(
            "Paris is the capital.", ["Paris is the capital of France."]
        )
        assert len(assessments) == 1
        assert assessments[0].assessment.status == ClaimSupportStatus.SUPPORTED
        assert assessments[0].assessment.method == LEXICAL_BASELINE_METHOD

    def test_unsupported_statement(self):
        assessments = assess_statements_lexically(
            "Bananas are purple.", ["Paris is the capital of France."]
        )
        assert assessments[0].assessment.status == ClaimSupportStatus.UNSUPPORTED

    def test_never_reports_contradicted(self):
        # The lexical baseline has no mechanism to detect contradiction.
        assessments = assess_statements_lexically(
            "Paris is not the capital.", ["Paris is the capital of France."]
        )
        assert all(a.assessment.status != ClaimSupportStatus.CONTRADICTED for a in assessments)

    def test_multiple_sentences_each_assessed_independently(self):
        answer = "Paris is the capital of France. Bananas are purple."
        assessments = assess_statements_lexically(answer, ["Paris is the capital of France."])
        assert len(assessments) == 2
        assert assessments[0].assessment.status == ClaimSupportStatus.SUPPORTED
        assert assessments[1].assessment.status == ClaimSupportStatus.UNSUPPORTED

    def test_empty_answer_produces_no_assessments(self):
        assert assess_statements_lexically("", ["some context"]) == []

    def test_confidence_is_never_fabricated(self):
        assessments = assess_statements_lexically(
            "Paris is the capital.", ["Paris is the capital."]
        )
        assert assessments[0].assessment.confidence is None

    def test_round_trip_through_dict(self):
        from llm_reliability.diagnostics.evidence import ClaimAssessment

        assessments = assess_statements_lexically(
            "Paris is the capital.", ["Paris is the capital."]
        )
        reloaded = ClaimAssessment.from_dict(assessments[0].to_dict())
        assert reloaded == assessments[0]

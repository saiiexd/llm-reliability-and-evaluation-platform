"""Tests for SQuAD-normalized matching and the closed-book prompt adapter."""

from __future__ import annotations

import pytest

from llm_reliability.evaluation.adapters import MockAdapter
from llm_reliability.evaluation.models import (
    EvaluationStatus,
    ExecutionRequest,
    ModelConfig,
    ModelResponse,
    TestCase,
)
from llm_reliability.evaluation.prompting import PromptTemplateAdapter
from llm_reliability.evaluation.squad_match import (
    SquadNormalizedMatchEvaluator,
    squad_normalize,
    squad_token_f1,
)


def _resp(text: str) -> ModelResponse:
    return ModelResponse(test_case_id="t", output_text=text, model_id="m")


def test_normalization_matches_official_rules() -> None:
    assert squad_normalize("The  Eiffel Tower!") == "eiffel tower"
    assert squad_normalize("an apple, a pear") == "apple pear"


def test_token_f1() -> None:
    assert squad_token_f1("Denver Broncos", "Denver Broncos") == 1.0
    # pred tokens {the broncos} -> normalized {broncos}; gold {denver broncos}
    # overlap 1, precision 1/1, recall 1/2 -> F1 = 2/3
    assert squad_token_f1("the Broncos", "Denver Broncos") == pytest.approx(2 / 3)
    assert squad_token_f1("Paris", "London") == 0.0


def test_em_against_any_gold_answer() -> None:
    tc = TestCase(
        id="t",
        input="q",
        reference_answer="The graph isomorphism problem",
        metadata={"gold_answers": ["The graph isomorphism problem", "graph isomorphism"]},
    )
    result = SquadNormalizedMatchEvaluator().evaluate(tc, _resp("Graph isomorphism."))
    assert result.passed is True
    assert result.score == 1.0


def test_verbose_answer_fails_em_but_gets_partial_f1() -> None:
    tc = TestCase(id="t", input="q", reference_answer="Paris")
    result = SquadNormalizedMatchEvaluator().evaluate(tc, _resp("The capital is Paris."))
    assert result.passed is False
    assert 0 < result.score < 1


def test_no_reference_is_skipped_not_failed() -> None:
    result = SquadNormalizedMatchEvaluator().evaluate(TestCase(id="t", input="q"), _resp("x"))
    assert result.status == EvaluationStatus.SKIPPED
    assert result.passed is None


def test_prompt_template_adapter_records_prompt() -> None:
    inner = MockAdapter(model_id="inner")
    adapter = PromptTemplateAdapter(inner, "closed_book_concise_v1")
    response = adapter.generate(ExecutionRequest("t", "Who wrote Hamlet?", ModelConfig("m")))
    assert "Question: Who wrote Hamlet?" in response.output_text  # mock echoes its input
    assert response.provider_metadata["prompt_template_id"] == "closed_book_concise_v1"
    assert response.provider_metadata["prompt"] == response.output_text


def test_unknown_template_rejected() -> None:
    with pytest.raises(ValueError):
        PromptTemplateAdapter(MockAdapter(), "nope")

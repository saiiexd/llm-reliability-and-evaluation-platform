"""
SQuAD-style normalized answer matching.

The project's ``ExactMatchEvaluator`` deliberately applies only
case/whitespace normalization against a single reference. Extractive QA
benchmarks such as SQuAD instead define correctness against *several* gold
answers after a fixed, published normalization (Rajpurkar et al., 2016,
official evaluation script v1.1):

1. lowercase;
2. remove punctuation;
3. remove the articles "a", "an", "the";
4. collapse whitespace.

This evaluator reports, per test case:

- ``passed``: normalized exact match against ANY gold answer (SQuAD "EM");
- ``score``: the maximum token-level F1 over the gold answers (SQuAD "F1"),
  which gives partial credit for overlapping tokens.

Gold answers are read from ``test_case.metadata["gold_answers"]`` when
present, otherwise the single ``reference_answer`` is used. Like any lexical
metric it measures surface overlap, not meaning: a correct paraphrase can
fail and an answer that copies the right span inside a wrong sentence can
pass. It is reported alongside, not instead of, the LLM judge.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from llm_reliability.evaluation.criteria import CORRECTNESS
from llm_reliability.evaluation.evaluators import Evaluator
from llm_reliability.evaluation.models import (
    EvaluationResult,
    EvaluationStatus,
    ModelResponse,
    TestCase,
)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def squad_normalize(text: str) -> str:
    """The SQuAD v1.1 official answer normalization."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def squad_token_f1(prediction: str, gold: str) -> float:
    pred_tokens = squad_normalize(prediction).split()
    gold_tokens = squad_normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def gold_answers_for(test_case: TestCase) -> list[str]:
    gold = test_case.metadata.get("gold_answers")
    if isinstance(gold, list) and gold and all(isinstance(g, str) for g in gold):
        return list(gold)
    return [test_case.reference_answer] if test_case.reference_answer is not None else []


class SquadNormalizedMatchEvaluator(Evaluator):
    """SQuAD EM (``passed``) and max token F1 (``score``) against all gold answers."""

    name = "squad_match"
    criterion = CORRECTNESS

    def get_config(self) -> dict[str, Any]:
        return {"normalization": "squad_v1.1_official", "gold_source": "metadata.gold_answers"}

    def evaluate(self, test_case: TestCase, response: ModelResponse) -> EvaluationResult:
        gold = gold_answers_for(test_case)
        if not gold:
            return EvaluationResult(
                evaluator_name=self.name,
                test_case_id=test_case.id,
                status=EvaluationStatus.SKIPPED,
                criterion=self.criterion,
                label="no_reference",
                evaluator_config=self.get_config(),
                explanation="No gold answer is available; SQuAD matching is not applicable.",
            )
        prediction = response.output_text
        normalized = squad_normalize(prediction)
        em = any(normalized == squad_normalize(g) for g in gold)
        f1 = max(squad_token_f1(prediction, g) for g in gold)
        return EvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case.id,
            status=EvaluationStatus.SUCCESS,
            criterion=self.criterion,
            score=f1,
            passed=em,
            label="match" if em else "mismatch",
            evaluator_config=self.get_config(),
            details={"normalized_prediction": normalized, "num_gold_answers": len(gold)},
            explanation=(
                f"Normalized answer {'matched' if em else 'did not match'} one of "
                f"{len(gold)} gold answer(s); best token F1 = {f1:.2f}."
            ),
        )

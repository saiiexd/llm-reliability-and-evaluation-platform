"""
LLM-as-a-judge evaluator.

Asks a judge model to assess a generated answer's correctness against a
reference answer, using an explicit, structured rubric rather than an
unconstrained request to "score this answer." The rubric defines the
evaluation criterion, the allowed scoring scale, the reasoning it requires,
and the exact expected output structure; it is stored as data
(``JudgeRubric``), not scattered as prose through this module, so a
different rubric can be substituted without changing any evaluator code.

LLM-as-a-judge is itself an evaluation *method*, not a source of ground
truth: its output reflects the judge model's own behavior, training, and
biases. This platform does not treat a judge's verdict as human truth, and
this evaluator's own reliability (its agreement with human annotation)
must be established separately before its results can be trusted as a
substitute for human review -- that validation work is out of scope for
this milestone.

Judge access reuses the existing ``ModelAdapter`` interface from
``llm_reliability.evaluation.adapters`` rather than introducing a parallel
adapter abstraction: a judge model's fundamental capability -- receive
text, produce text -- is identical to the system-under-test adapter's.
``JudgeAdapter`` below is a plain naming alias over ``ModelAdapter``, added
only for readability at call sites that use an adapter as a judge, not a
new interface. Any existing ``ModelAdapter`` implementation, including the
deterministic ``MockAdapter``, can serve directly as a fake judge in tests:
map a test case id to the exact JSON string the fake judge should "say."

The judge is called at most once per test case, and only by this
evaluator; the system-under-test model is never called again here. Judge
output is parsed strictly: it must be a JSON object with the exact keys
the rubric's expected output structure calls for, and the ``outcome``
value must be one of the rubric's declared scale values. Anything else
(malformed JSON, missing keys, an outcome outside the declared scale)
produces an explicit ``OUTPUT_VALIDATION_ERROR`` result rather than being
coerced into a score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.criteria import CORRECTNESS
from llm_reliability.evaluation.evaluators import Evaluator
from llm_reliability.evaluation.models import (
    EvaluationResult,
    EvaluationStatus,
    ExecutionRequest,
    ModelConfig,
    ModelResponse,
    TestCase,
)

type JudgeAdapter = ModelAdapter
"""A judge model is accessed through the same ``ModelAdapter`` interface as any other model."""


class JudgeOutputValidationError(Exception):
    """Raised when a judge model's raw output cannot be parsed into a valid structured judgment."""


@dataclass
class JudgeRubric:
    """A structured rubric given to an LLM judge.

    Stored as configuration -- part of the evaluator's ``get_config()``,
    and therefore part of the experiment configuration fingerprint -- so
    that changing the rubric text, its scale, or its criterion always
    produces a different, traceable configuration.

    ``criterion`` identifies what is being assessed (see
    ``llm_reliability.evaluation.criteria``). ``scale`` lists the allowed
    categorical outcomes the judge may choose, ordered from worst to best;
    the last entry is treated as the "fully correct" outcome for deriving
    ``EvaluationResult.passed``. ``instructions`` is the exact rubric text
    sent to the judge model, including the required output format.
    """

    rubric_id: str
    criterion: str
    scale: tuple[str, ...]
    instructions: str

    def __post_init__(self) -> None:
        if not self.rubric_id or not self.rubric_id.strip():
            raise ValueError("JudgeRubric.rubric_id must be a non-empty string.")
        if not self.criterion or not self.criterion.strip():
            raise ValueError("JudgeRubric.criterion must be a non-empty string.")
        if len(self.scale) < 2:
            raise ValueError("JudgeRubric.scale must define at least two outcomes.")
        if len(set(self.scale)) != len(self.scale):
            raise ValueError("JudgeRubric.scale outcomes must be unique.")
        if not self.instructions or not self.instructions.strip():
            raise ValueError("JudgeRubric.instructions must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "criterion": self.criterion,
            "scale": list(self.scale),
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeRubric:
        return cls(
            rubric_id=data["rubric_id"],
            criterion=data["criterion"],
            scale=tuple(data["scale"]),
            instructions=data["instructions"],
        )


CORRECTNESS_RUBRIC = JudgeRubric(
    rubric_id="correctness_vs_reference_v1",
    criterion=CORRECTNESS,
    scale=("incorrect", "partially_correct", "correct"),
    instructions=(
        "You are evaluating whether a generated answer is factually correct "
        "relative to a provided reference answer. Judge only correctness "
        "against the reference. Do not reward an answer for being long, "
        "confident-sounding, or fluent, and do not assume that an answer "
        "which is worded similarly to the reference is necessarily correct "
        "-- judge the actual factual content, not surface similarity.\n\n"
        "Choose exactly one outcome from: incorrect, partially_correct, correct.\n"
        "- correct: the answer states the same key fact(s) as the reference, "
        "with no material error or omission.\n"
        "- partially_correct: the answer contains some correct elements of "
        "the reference but is incomplete, imprecise, or partially wrong.\n"
        "- incorrect: the answer contradicts the reference or does not "
        "address it.\n\n"
        "Respond with a single JSON object and nothing else, in exactly this "
        'form: {"outcome": <one of the three values above>, '
        '"reasoning": <one or two sentences of justification>}.'
    ),
)
"""The initial judge rubric: correctness against an available reference answer."""


@dataclass
class ParsedJudgeOutput:
    outcome: str
    reasoning: str


def render_judge_prompt(
    rubric: JudgeRubric, question: str, reference_answer: str, candidate_answer: str
) -> str:
    """Render the exact prompt sent to the judge model for one test case."""
    return (
        f"{rubric.instructions}\n\n"
        f"Question: {question}\n"
        f"Reference answer: {reference_answer}\n"
        f"Candidate answer: {candidate_answer}\n"
    )


def parse_judge_output(raw_text: str, rubric: JudgeRubric) -> ParsedJudgeOutput:
    """Strictly parse and validate a judge model's raw text output.

    No lenient fallback (such as extracting JSON from surrounding prose)
    is attempted: a judge that does not follow the rubric's exact output
    format is a validation failure to surface explicitly, not a case to
    silently work around.
    """
    try:
        data = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        raise JudgeOutputValidationError(f"Judge output was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise JudgeOutputValidationError("Judge output JSON must be an object.")

    outcome = data.get("outcome")
    if not isinstance(outcome, str) or outcome not in rubric.scale:
        raise JudgeOutputValidationError(
            f"Judge output 'outcome' must be one of {rubric.scale}, got {outcome!r}."
        )

    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise JudgeOutputValidationError("Judge output 'reasoning' must be a non-empty string.")

    return ParsedJudgeOutput(outcome=outcome, reasoning=reasoning)


class LLMJudgeEvaluator(Evaluator):
    """Evaluates correctness by asking a judge model to apply an explicit rubric.

    Calls ``judge_adapter`` at most once per test case, only when a
    reference answer is available (the default rubric evaluates
    correctness against a reference, so it is not applicable without one).
    Never calls the system-under-test model.
    """

    name = "llm_judge"
    criterion = CORRECTNESS

    def __init__(
        self,
        judge_adapter: JudgeAdapter,
        judge_model_config: ModelConfig,
        rubric: JudgeRubric = CORRECTNESS_RUBRIC,
    ) -> None:
        self._judge_adapter = judge_adapter
        self._judge_model_config = judge_model_config
        self._rubric = rubric

    def get_config(self) -> dict[str, Any]:
        return {
            "judge_model_id": self._judge_model_config.model_id,
            "rubric": self._rubric.to_dict(),
        }

    def evaluate(self, test_case: TestCase, response: ModelResponse) -> EvaluationResult:
        if test_case.reference_answer is None:
            return EvaluationResult(
                evaluator_name=self.name,
                test_case_id=test_case.id,
                status=EvaluationStatus.SKIPPED,
                criterion=self.criterion,
                label="no_reference",
                evaluator_config=self.get_config(),
                explanation=(
                    "No reference answer was provided for this test case; the "
                    f"'{self._rubric.rubric_id}' rubric evaluates correctness against a "
                    "reference and is not applicable."
                ),
            )

        prompt = render_judge_prompt(
            rubric=self._rubric,
            question=test_case.input,
            reference_answer=test_case.reference_answer,
            candidate_answer=response.output_text,
        )
        request = ExecutionRequest(
            test_case_id=test_case.id,
            input_text=prompt,
            model_config=self._judge_model_config,
        )
        try:
            judge_response = self._judge_adapter.generate(request)
        except Exception as exc:
            return EvaluationResult(
                evaluator_name=self.name,
                test_case_id=test_case.id,
                status=EvaluationStatus.EXECUTION_ERROR,
                criterion=self.criterion,
                evaluator_config=self.get_config(),
                error=f"{type(exc).__name__}: {exc}",
            )

        try:
            parsed = parse_judge_output(judge_response.output_text, self._rubric)
        except JudgeOutputValidationError as exc:
            return EvaluationResult(
                evaluator_name=self.name,
                test_case_id=test_case.id,
                status=EvaluationStatus.OUTPUT_VALIDATION_ERROR,
                criterion=self.criterion,
                evaluator_config=self.get_config(),
                details={"raw_judge_output": judge_response.output_text},
                error=str(exc),
            )

        best_outcome = self._rubric.scale[-1]
        return EvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case.id,
            status=EvaluationStatus.SUCCESS,
            criterion=self.criterion,
            passed=(parsed.outcome == best_outcome),
            label=parsed.outcome,
            explanation=parsed.reasoning,
            evaluator_config=self.get_config(),
            details={"rubric_id": self._rubric.rubric_id},
        )

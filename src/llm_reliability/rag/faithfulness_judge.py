"""
LLM-based faithfulness evaluator.

Extends the LLM-as-a-judge architecture from
``llm_reliability.evaluation.llm_judge`` rather than introducing a second,
parallel judge framework: this module reuses ``JudgeRubric``,
``parse_judge_output``, ``JudgeOutputValidationError``, and the
``JudgeAdapter`` alias unchanged. What differs is the material given to
the judge (retrieved context instead of a reference answer) and the
rubric's scale (a four-way faithfulness assessment instead of a
three-way correctness assessment), so this lives in its own evaluator
class and prompt renderer rather than being force-fit into
``LLMJudgeEvaluator``.

As with the correctness judge, this is an evaluation *method*, not ground
truth: its output reflects the judge model's own behavior and biases, and
its agreement with human judgment must be established separately before
its verdicts can be trusted as a substitute for human review.

Distinguishes four outcomes, precisely:

- ``supported``: the answer's claims are backed by the retrieved context.
- ``unsupported``: the answer contains claims the retrieved context does
  not address either way (neither confirms nor denies them).
- ``contradicted``: the retrieved context directly conflicts with a claim
  in the answer.
- ``undetermined``: the judge could not reach one of the above (used only
  when the judge itself reports it; malformed judge output is a separate,
  explicit ``OUTPUT_VALIDATION_ERROR``, not folded into "undetermined").

This distinction requires the kind of semantic understanding literal word
overlap cannot provide (see
``llm_reliability.rag.context_support.ContextSupportBaseline`` for that
narrower baseline), which is exactly why this evaluator exists alongside
it rather than replacing it.
"""

from __future__ import annotations

from typing import Any

from llm_reliability.evaluation.criteria import FAITHFULNESS
from llm_reliability.evaluation.evaluators import Evaluator
from llm_reliability.evaluation.llm_judge import (
    JudgeAdapter,
    JudgeOutputValidationError,
    JudgeRubric,
    parse_judge_output,
)
from llm_reliability.evaluation.models import (
    EvaluationResult,
    EvaluationStatus,
    ExecutionRequest,
    ModelConfig,
    ModelResponse,
    TestCase,
)
from llm_reliability.rag.pipeline import extract_retrieved_chunks

FAITHFULNESS_RUBRIC = JudgeRubric(
    rubric_id="faithfulness_vs_context_v1",
    criterion=FAITHFULNESS,
    scale=("contradicted", "unsupported", "undetermined", "supported"),
    instructions=(
        "You are evaluating whether a generated answer is faithful to a provided set "
        "of retrieved context passages -- not whether the answer is fluent, complete, "
        "or externally true. Judge only whether the context supports the answer's "
        "claims. Do not use outside knowledge to decide whether the answer is "
        "factually correct in general; judge strictly against the given context.\n\n"
        "Choose exactly one outcome from: contradicted, unsupported, undetermined, supported.\n"
        "- supported: the context contains information that backs the answer's claims.\n"
        "- unsupported: the context neither confirms nor denies the answer's claims "
        "(the information simply is not present).\n"
        "- contradicted: the context directly conflicts with a claim in the answer.\n"
        "- undetermined: the context and the answer are too ambiguous, fragmentary, or "
        "unrelated in scope for you to reach any of the above three judgments reliably.\n\n"
        "Respond with a single JSON object and nothing else, in exactly this form: "
        '{"outcome": <one of the four values above>, '
        '"reasoning": <one or two sentences of justification>}.'
    ),
)
"""The faithfulness rubric: supported / unsupported / contradicted / undetermined."""

_UNDETERMINED_OUTCOME = "undetermined"


def render_faithfulness_prompt(
    rubric: JudgeRubric, question: str, context: str, candidate_answer: str
) -> str:
    """Render the exact prompt sent to the judge model for one faithfulness check."""
    return (
        f"{rubric.instructions}\n\n"
        f"Question: {question}\n"
        f"Retrieved context:\n{context}\n\n"
        f"Candidate answer: {candidate_answer}\n"
    )


class LLMFaithfulnessEvaluator(Evaluator):
    """Evaluates faithfulness by asking a judge model to compare an answer against context.

    Calls ``judge_adapter`` at most once per test case, only when the
    response carries retrieval evidence (it was produced by a
    ``RagPipeline``) and the answer is non-empty. Never calls the
    system-under-test model.
    """

    name = "llm_faithfulness_judge"
    criterion = FAITHFULNESS

    def __init__(
        self,
        judge_adapter: JudgeAdapter,
        judge_model_config: ModelConfig,
        rubric: JudgeRubric = FAITHFULNESS_RUBRIC,
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
        retrieved_chunks = extract_retrieved_chunks(response)
        if retrieved_chunks is None:
            return EvaluationResult(
                evaluator_name=self.name,
                test_case_id=test_case.id,
                status=EvaluationStatus.SKIPPED,
                criterion=self.criterion,
                label="no_context",
                evaluator_config=self.get_config(),
                explanation=(
                    "This response carries no retrieved context (it was not produced by a "
                    "RAG pipeline); faithfulness comparison is not applicable."
                ),
            )
        if not response.output_text or not response.output_text.strip():
            return EvaluationResult(
                evaluator_name=self.name,
                test_case_id=test_case.id,
                status=EvaluationStatus.SKIPPED,
                criterion=self.criterion,
                label="empty_answer",
                evaluator_config=self.get_config(),
                explanation=(
                    "The generated answer is empty; faithfulness comparison is not applicable."
                ),
            )

        context_text = "\n".join(chunk.text for chunk in retrieved_chunks)
        prompt = render_faithfulness_prompt(
            self._rubric, test_case.input, context_text, response.output_text
        )
        request = ExecutionRequest(
            test_case_id=test_case.id, input_text=prompt, model_config=self._judge_model_config
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
        # "undetermined" is a genuine judge answer (it could not reach a
        # confident assessment), not a negative one -- it must not be
        # folded into `passed=False` alongside a real contradiction.
        passed = None if parsed.outcome == _UNDETERMINED_OUTCOME else parsed.outcome == best_outcome
        return EvaluationResult(
            evaluator_name=self.name,
            test_case_id=test_case.id,
            status=EvaluationStatus.SUCCESS,
            criterion=self.criterion,
            passed=passed,
            label=parsed.outcome,
            explanation=parsed.reasoning,
            evaluator_config=self.get_config(),
            details={"rubric_id": self._rubric.rubric_id},
        )

"""
Controlled vocabularies for the reliability diagnostics layer.

Three separate, small enums, deliberately not merged into one: they answer
different questions and mixing them would blur exactly the distinctions
this layer exists to preserve.

- ``DiagnosticStatus`` answers "could a diagnosis be reached at all?" --
  the process-level outcome, analogous to ``EvaluationStatus``
  (``llm_reliability.evaluation``) and ``RetrievalEvaluationStatus``
  (``llm_reliability.rag``).
- ``FailureCategory`` answers "what does the evidence, once gathered,
  actually show?" -- the categorical diagnosis itself, populated only
  when ``DiagnosticStatus.DETERMINED``.
- ``ClaimSupportStatus`` answers a narrower question specific to one
  statement in an answer: is it backed, unaddressed, or contradicted by
  retrieved context?

None of these categories asserts that an answer is "hallucinated" as a
blanket label. See ``research/notes/reliability_and_robustness_layer.md``
for why that word is deliberately absent from this platform's vocabulary.
"""

from __future__ import annotations

from enum import StrEnum


class DiagnosticStatus(StrEnum):
    """Whether ``build_reliability_diagnostic`` could reach a determination.

    ``DETERMINED``: sufficient evidence was available to assign a
    ``FailureCategory``.
    ``INSUFFICIENT_EVIDENCE``: no relevant evaluators produced a usable
    result for this test case (for example, every evaluator was
    ``SKIPPED``).
    ``UNDETERMINED``: some evidence exists, but it does not clearly
    support any single category (for example, evaluators disagree in a
    way that cannot be resolved, or a retrieval miss coincides with a
    correct answer -- see ``research/notes/rag_execution_model.md``).
    ``ERROR``: building the diagnostic itself failed unexpectedly (an
    evaluator or retrieval evaluator raised); the underlying error is
    preserved on the diagnostic record.
    """

    DETERMINED = "determined"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNDETERMINED = "undetermined"
    ERROR = "error"


class FailureCategory(StrEnum):
    """The categorical outcome of a reliability diagnosis, when one can be reached.

    ``CORRECT``: reference-based evaluation found the answer correct, and
    no retrieval or faithfulness evidence contradicts that.
    ``INCORRECT``: reference-based evaluation found the answer incorrect,
    with no retrieval evidence available to attribute the cause further.
    ``UNSUPPORTED``: the answer contains claims the retrieved context does
    not address either way (faithfulness evidence, not correctness).
    ``CONTRADICTED``: the retrieved context directly conflicts with the
    answer (faithfulness evidence).
    ``RETRIEVAL_FAILURE``: ground-truth-based retrieval evaluation shows
    the required evidence was not retrieved, and the answer was incorrect.
    ``GENERATION_FAILURE``: ground-truth-based retrieval evaluation shows
    the required evidence *was* retrieved, yet the answer was incorrect --
    the failure is attributable to generation, not retrieval.
    ``EVALUATOR_DISAGREEMENT``: independent evaluators of the same
    criterion reached conflicting verdicts; the disagreement itself is the
    finding, and no single verdict is treated as authoritative.
    ``INSUFFICIENT_EVIDENCE``: (see ``DiagnosticStatus``) no category
    could be assigned because no relevant evidence exists.
    ``UNDETERMINED``: (see ``DiagnosticStatus``) evidence exists but does
    not clearly support any other category.

    This list is deliberately short: each category corresponds to a
    distinguishable combination of evidence this platform can actually
    compute (see ``llm_reliability.diagnostics.diagnosis``), not to every
    conceivable failure mode a RAG system could have.
    """

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    RETRIEVAL_FAILURE = "retrieval_failure"
    GENERATION_FAILURE = "generation_failure"
    EVALUATOR_DISAGREEMENT = "evaluator_disagreement"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNDETERMINED = "undetermined"


class ClaimSupportStatus(StrEnum):
    """The assessment of one statement/claim against retrieved context.

    ``SUPPORTED``: the context backs the statement.
    ``UNSUPPORTED``: the context neither confirms nor denies the statement.
    ``CONTRADICTED``: the context conflicts with the statement.
    ``UNDETERMINED``: the assessment method could not reach one of the above.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNDETERMINED = "undetermined"

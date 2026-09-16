"""
Statement-level evidence assessment: Generated Answer -> Statements -> Evidence Assessment.

This module deliberately implements *sentence-level* segmentation, not
claim extraction: splitting text into independent factual claims reliably
requires natural-language understanding this platform does not yet have a
validated way to provide, and a plausible-looking but unreliable claim
extractor would be worse than no claim-level analysis at all. A sentence
is a defensible, simple, fully deterministic unit; it is not guaranteed to
correspond to exactly one claim (a sentence can contain zero, one, or
several factual assertions), and this limitation must be read alongside
every result this module produces.

Two assessment methods are provided, at different levels of reliability:

- ``assess_statements_lexically``: a deterministic, dependency-free
  baseline using literal word overlap (the same methodology as
  ``llm_reliability.rag.context_support.ContextSupportBaseline``, applied
  per sentence instead of to the whole answer). It can only ever report
  ``SUPPORTED`` or ``UNSUPPORTED`` -- it has no way to detect
  contradiction, since word overlap alone cannot distinguish "the context
  agrees" from "the context discusses the same topic but disagrees."
- Contradiction-capable, semantically aware assessment requires an actual
  judge model; see
  ``llm_reliability.rag.faithfulness_judge.LLMFaithfulnessEvaluator``,
  which produces a whole-answer (not per-statement) assessment for the
  same reason: reliable per-statement judge prompting is a larger,
  separately justified piece of work this milestone does not attempt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from llm_reliability.diagnostics.taxonomy import ClaimSupportStatus
from llm_reliability.rag.context_support import compute_overlap_ratio

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

LEXICAL_BASELINE_METHOD = "lexical_overlap_baseline"


def split_into_statements(text: str) -> list[str]:
    """Split text into sentence-level statements.

    A simple, deterministic split on sentence-ending punctuation followed
    by whitespace. Does not handle abbreviations, decimal numbers, or
    other punctuation edge cases specially; this is a documented
    limitation, not an oversight -- a more careful sentence splitter would
    still not make this claim-level analysis, only a better-segmented
    version of the same statement-level baseline.
    """
    stripped = text.strip()
    if not stripped:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(stripped) if s.strip()]


@dataclass
class EvidenceAssessment:
    """The outcome of assessing one statement against a body of evidence.

    ``confidence`` is left ``None`` by every assessment method in this
    milestone: the lexical baseline's overlap ratio is a score, not a
    calibrated confidence, and the LLM judge does not currently report a
    validated confidence value. ``None`` here must be read as "not
    available," never as "assumed low" or "assumed high."
    """

    status: ClaimSupportStatus
    method: str
    supporting_evidence: list[str] = field(default_factory=list)
    confidence: float | None = None
    explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "method": self.method,
            "supporting_evidence": self.supporting_evidence,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceAssessment:
        return cls(
            status=ClaimSupportStatus(data["status"]),
            method=data["method"],
            supporting_evidence=list(data.get("supporting_evidence", [])),
            confidence=data.get("confidence"),
            explanation=data.get("explanation"),
        )


@dataclass
class ClaimAssessment:
    """One statement from a generated answer, paired with its evidence assessment."""

    statement: str
    assessment: EvidenceAssessment

    def to_dict(self) -> dict[str, Any]:
        return {"statement": self.statement, "assessment": self.assessment.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaimAssessment:
        return cls(
            statement=data["statement"], assessment=EvidenceAssessment.from_dict(data["assessment"])
        )


def assess_statements_lexically(
    answer_text: str,
    context_chunks: list[str],
    supported_threshold: float = 0.5,
    case_sensitive: bool = False,
) -> list[ClaimAssessment]:
    """Assess each sentence of ``answer_text`` against ``context_chunks`` by literal word overlap.

    This is the narrow, deterministic baseline described in the module
    docstring: it reports only ``SUPPORTED`` or ``UNSUPPORTED`` per
    sentence (never ``CONTRADICTED``, which it has no way to detect) and
    must not be read as a complete or semantic faithfulness measurement.
    Returns one ``ClaimAssessment`` per sentence found in ``answer_text``,
    in order; an empty answer produces an empty list.
    """
    context_text = " ".join(context_chunks)
    assessments: list[ClaimAssessment] = []
    for statement in split_into_statements(answer_text):
        ratio = compute_overlap_ratio(statement, context_text, case_sensitive=case_sensitive)
        supported = ratio >= supported_threshold
        matched_chunks = [
            chunk
            for chunk in context_chunks
            if compute_overlap_ratio(statement, chunk, case_sensitive=case_sensitive) > 0.0
        ]
        assessments.append(
            ClaimAssessment(
                statement=statement,
                assessment=EvidenceAssessment(
                    status=ClaimSupportStatus.SUPPORTED
                    if supported
                    else ClaimSupportStatus.UNSUPPORTED,
                    method=LEXICAL_BASELINE_METHOD,
                    supporting_evidence=matched_chunks if supported else [],
                    explanation=(
                        f"{ratio:.2f} word overlap with retrieved context "
                        f"(threshold {supported_threshold}). Literal overlap only -- "
                        "this method cannot detect contradiction."
                    ),
                ),
            )
        )
    return assessments

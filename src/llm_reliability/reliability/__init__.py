"""
Reliability analysis layer.

Aggregates and classifies evaluation evidence already produced by the
evaluation layer (``llm_reliability.evaluation``), typically the
``RunResult`` inside an ``ExperimentRun`` (``llm_reliability.experiments``).
This is deliberately narrow: it identifies evaluator disagreement, missing
evaluation evidence, evaluator failures, and per-evaluator incorrect
results. It does not detect hallucinations, assess faithfulness or
groundedness, or determine factual truth from an answer alone; those
require context-grounded evaluation and are explicitly out of scope for
this milestone. See ``research/notes/evaluation_and_reliability_layer.md``.
"""

from llm_reliability.reliability.findings import (
    ReliabilityFinding,
    ReliabilityFlagType,
    analyze_reliability,
)
from llm_reliability.reliability.summary import EvaluationSummary, EvaluatorSummary, summarize_run

__all__ = [
    "EvaluationSummary",
    "EvaluatorSummary",
    "ReliabilityFinding",
    "ReliabilityFlagType",
    "analyze_reliability",
    "summarize_run",
]

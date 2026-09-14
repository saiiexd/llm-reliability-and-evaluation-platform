"""
Core Evaluation Engine.

Implements the controlled execution pipeline:

    Dataset -> TestCase -> Model/Application Adapter -> ModelResponse
        -> Evaluator(s) -> EvaluationResult -> RunResult

This package is intentionally scoped to controlled, single-turn execution
of a dataset against one model/application adapter. It does not implement
retrieval-augmented generation, persistence, reliability analysis, or
observability -- those are separate, later milestones. See
``research/notes/`` for the conceptual execution model this package
implements.
"""

from llm_reliability.evaluation.adapters import AdapterError, MockAdapter, ModelAdapter
from llm_reliability.evaluation.evaluators import Evaluator, ExactMatchEvaluator
from llm_reliability.evaluation.models import (
    Dataset,
    EvaluationResult,
    ExecutionRequest,
    ModelConfig,
    ModelResponse,
    RunResult,
    TestCase,
    TestCaseResult,
)
from llm_reliability.evaluation.runner import EvaluationRunner

__all__ = [
    "AdapterError",
    "Dataset",
    "EvaluationResult",
    "EvaluationRunner",
    "Evaluator",
    "ExactMatchEvaluator",
    "ExecutionRequest",
    "ModelAdapter",
    "ModelConfig",
    "ModelResponse",
    "MockAdapter",
    "RunResult",
    "TestCase",
    "TestCaseResult",
]

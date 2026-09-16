"""
Core Evaluation Engine.

Implements the controlled execution pipeline:

    Dataset -> TestCase -> Model/Application Adapter -> ModelResponse
        -> Evaluator(s) -> EvaluationResult -> RunResult

This package is intentionally scoped to controlled, single-turn execution
of a dataset against one model/application adapter. It does not implement
retrieval-augmented generation, persistence, or observability -- those are
separate, later milestones. See ``research/notes/`` for the conceptual
execution model this package implements and, for the multi-evaluator
architecture (exact match, semantic similarity, LLM-as-a-judge, evaluation
criteria, and reliability analysis), ``llm_reliability.reliability`` and
``research/notes/evaluation_and_reliability_layer.md``.
"""

from llm_reliability.evaluation.adapters import AdapterError, MockAdapter, ModelAdapter
from llm_reliability.evaluation.datasets import load_dataset_from_json
from llm_reliability.evaluation.evaluators import Evaluator, ExactMatchEvaluator
from llm_reliability.evaluation.llm_judge import (
    CORRECTNESS_RUBRIC,
    JudgeAdapter,
    JudgeOutputValidationError,
    JudgeRubric,
    LLMJudgeEvaluator,
    parse_judge_output,
    render_judge_prompt,
)
from llm_reliability.evaluation.models import (
    Dataset,
    EvaluationResult,
    EvaluationStatus,
    ExecutionRequest,
    ModelConfig,
    ModelResponse,
    RunResult,
    TestCase,
    TestCaseResult,
)
from llm_reliability.evaluation.runner import EvaluationRunner
from llm_reliability.evaluation.semantic_similarity import (
    BertScoreBackend,
    FakeSimilarityBackend,
    SemanticSimilarityEvaluator,
    SimilarityBackend,
    SimilarityBackendError,
)

__all__ = [
    "AdapterError",
    "BertScoreBackend",
    "CORRECTNESS_RUBRIC",
    "Dataset",
    "EvaluationResult",
    "EvaluationRunner",
    "EvaluationStatus",
    "Evaluator",
    "ExactMatchEvaluator",
    "ExecutionRequest",
    "FakeSimilarityBackend",
    "JudgeAdapter",
    "JudgeOutputValidationError",
    "JudgeRubric",
    "LLMJudgeEvaluator",
    "ModelAdapter",
    "ModelConfig",
    "ModelResponse",
    "MockAdapter",
    "RunResult",
    "SemanticSimilarityEvaluator",
    "SimilarityBackend",
    "SimilarityBackendError",
    "TestCase",
    "TestCaseResult",
    "load_dataset_from_json",
    "parse_judge_output",
    "render_judge_prompt",
]

"""
Domain models for the Core Evaluation Engine.

These models define the structured data that flows through the evaluation
pipeline:

    Dataset -> TestCase -> ExecutionRequest -> ModelResponse
        -> EvaluationResult -> TestCaseResult -> RunResult

Validation happens eagerly in ``__post_init__`` so malformed data is
rejected at construction time rather than silently corrupting a run
later in the pipeline. Validation is applied to the models a user
actually authors (``TestCase``, ``Dataset``, ``ModelConfig``); models
produced internally by the engine (``ModelResponse``, ``EvaluationResult``,
``TestCaseResult``, ``RunResult``) are not re-validated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TestCase:
    """A single, deterministic evaluation input.

    ``reference_answer`` is optional because the platform explicitly
    supports reference-free and model-based evaluation in later
    milestones. Evaluators must handle its absence rather than assume it
    is always present.
    """

    # Tells pytest not to try to collect this domain class as a test class;
    # it is unrelated to pytest, but its name starts with "Test".
    __test__ = False

    id: str
    input: str
    reference_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("TestCase.id must be a non-empty string.")
        if not self.input or not self.input.strip():
            raise ValueError(f"TestCase '{self.id}': input must be a non-empty string.")


@dataclass
class Dataset:
    """An ordered, deterministic collection of test cases.

    Test case ids must be unique within a dataset so that every result
    can always be traced back to the test case that produced it.
    """

    name: str
    test_cases: list[TestCase]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Dataset.name must be a non-empty string.")
        if len(self.test_cases) == 0:
            raise ValueError(f"Dataset '{self.name}' must contain at least one test case.")
        seen: set[str] = set()
        duplicates: set[str] = set()
        for test_case in self.test_cases:
            if test_case.id in seen:
                duplicates.add(test_case.id)
            seen.add(test_case.id)
        if duplicates:
            raise ValueError(
                f"Dataset '{self.name}' contains duplicate test case ids: {sorted(duplicates)}. "
                "Test case ids must be unique so results can be traced back to their source."
            )

    def __len__(self) -> int:
        return len(self.test_cases)

    def __iter__(self):
        return iter(self.test_cases)


@dataclass
class ModelConfig:
    """Configuration for a single model/application invocation.

    Only fields actually consumed by an adapter belong here. Generation
    parameters are optional so "not specified" can be distinguished from
    an explicit value; adapters must not invent a default for a field
    that was left unset. ``extra_params`` is an explicit escape hatch for
    provider-specific settings that do not warrant a first-class field.
    """

    model_id: str
    temperature: float | None = None
    max_tokens: int | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_id.strip():
            raise ValueError("ModelConfig.model_id must be a non-empty string.")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("ModelConfig.temperature must be non-negative if specified.")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("ModelConfig.max_tokens must be a positive integer if specified.")


@dataclass
class ExecutionRequest:
    """The minimum information a model/application adapter needs to generate a response.

    Deliberately narrower than ``TestCase``: adapters generate answers and
    must not depend on fields (such as reference answers) that are only
    relevant to evaluation.
    """

    test_case_id: str
    input_text: str
    model_config: ModelConfig


@dataclass
class ModelResponse:
    """Structured output from a model/application adapter.

    Fields the underlying provider does not report are left as ``None``
    rather than fabricated. ``provider_metadata`` is an open extension
    point for adapter-specific detail -- for example, a future RAG
    adapter can carry retrieved-context information there without
    forcing retrieval fields into every adapter's schema.
    """

    test_case_id: str
    output_text: str
    model_id: str
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Structured output from a single evaluator run against a single test case.

    Only ``score`` (numeric), ``passed`` (boolean judgment), and
    ``label`` (categorical outcome) are populated as applicable to a
    given evaluator; the others are left ``None``. ``error`` is set only
    when the evaluator itself failed to produce a judgment, which is
    distinct from a low or negative score.
    """

    evaluator_name: str
    test_case_id: str
    score: float | None = None
    passed: bool | None = None
    label: str | None = None
    explanation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class TestCaseResult:
    """The complete, per-test-case outcome of an evaluation run."""

    test_case_id: str
    input_text: str
    reference_answer: str | None
    model_response: ModelResponse | None
    evaluation_results: list[EvaluationResult]
    execution_error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True if generation completed without error and produced a response."""
        return self.execution_error is None and self.model_response is not None


@dataclass
class RunResult:
    """The complete, structured outcome of running a dataset through the evaluation engine."""

    dataset_name: str
    model_id: str
    evaluator_names: list[str]
    test_case_results: list[TestCaseResult]

    @property
    def num_test_cases(self) -> int:
        return len(self.test_case_results)

    @property
    def num_succeeded(self) -> int:
        return sum(1 for result in self.test_case_results if result.succeeded)

    @property
    def num_failed(self) -> int:
        return self.num_test_cases - self.num_succeeded

    def to_dict(self) -> dict[str, Any]:
        """Convert the run result to plain, JSON-serializable Python types."""
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the run result to a deterministic JSON string.

        Keys are sorted so the output is stable across runs given the
        same underlying data, which matters for diffing reproduced
        experiment results.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

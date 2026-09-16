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

Every model also provides ``from_dict`` (and, where ``to_json`` exists,
``from_json``) so that JSON previously produced by this module can be
loaded back into validated objects rather than being treated as an
unvalidated dictionary. Reconstruction reuses each dataclass's own
``__post_init__`` validation, so malformed persisted data raises the same
explicit errors as malformed data supplied directly by a caller.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestCase:
        return cls(
            id=data["id"],
            input=data["input"],
            reference_answer=data.get("reference_answer"),
            metadata=dict(data.get("metadata", {})),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        return cls(
            name=data["name"],
            test_cases=[TestCase.from_dict(test_case) for test_case in data["test_cases"]],
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(
            model_id=data["model_id"],
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            extra_params=dict(data.get("extra_params", {})),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelResponse:
        return cls(
            test_case_id=data["test_case_id"],
            output_text=data["output_text"],
            model_id=data["model_id"],
            latency_ms=data.get("latency_ms"),
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            total_tokens=data.get("total_tokens"),
            provider_metadata=dict(data.get("provider_metadata", {})),
        )


class EvaluationStatus(StrEnum):
    """Categorical outcome of attempting to run one evaluator on one test case.

    ``SUCCESS``: the evaluator produced a definitive judgment.
    ``SKIPPED``: the evaluator could not run because required input was
    unavailable for this specific test case (for example, a
    reference-based metric with no reference answer). This is not a
    failing judgment about the answer -- it means no judgment was made,
    and must never be interpreted as evidence the answer is wrong.
    ``INVALID_CONFIGURATION``: the evaluator, as configured, cannot
    function in the current environment (for example, a similarity
    backend whose model dependency is not installed). Distinct from
    ``SKIPPED``, which is about the test case, not the evaluator's setup.
    ``EXECUTION_ERROR``: the evaluator (or something it depends on, such
    as a judge model call) raised an unexpected error while attempting to
    evaluate.
    ``OUTPUT_VALIDATION_ERROR``: the evaluator received output it could
    not parse or validate into a structured judgment. This arises for
    evaluators that consume free-form output from another system, such as
    an LLM-as-a-judge evaluator receiving a malformed judge response.
    """

    SUCCESS = "success"
    SKIPPED = "skipped"
    INVALID_CONFIGURATION = "invalid_configuration"
    EXECUTION_ERROR = "execution_error"
    OUTPUT_VALIDATION_ERROR = "output_validation_error"


@dataclass
class EvaluationResult:
    """Structured output from a single evaluator run against a single test case.

    ``status`` always describes what actually happened (see
    ``EvaluationStatus``); ``score``, ``passed``, and ``label`` are
    populated only when applicable to a given evaluator and outcome, and
    are left ``None`` otherwise -- a ``None`` score must never be read as
    a zero or a failing score. ``criterion`` names what is being assessed
    (see ``llm_reliability.evaluation.criteria``), independent of which
    evaluator assessed it. ``evaluator_config`` is a snapshot of the
    configuration that produced this specific result, so the result is
    interpretable on its own without cross-referencing the experiment
    that produced it. ``error`` holds a diagnostic message and is set
    whenever ``status`` is not ``SUCCESS`` or ``SKIPPED``.
    """

    evaluator_name: str
    test_case_id: str
    status: EvaluationStatus = EvaluationStatus.SUCCESS
    criterion: str | None = None
    score: float | None = None
    passed: bool | None = None
    label: str | None = None
    explanation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    evaluator_config: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        return cls(
            evaluator_name=data["evaluator_name"],
            test_case_id=data["test_case_id"],
            status=_parse_evaluation_status(data),
            criterion=data.get("criterion"),
            score=data.get("score"),
            passed=data.get("passed"),
            label=data.get("label"),
            explanation=data.get("explanation"),
            details=dict(data.get("details", {})),
            evaluator_config=dict(data.get("evaluator_config", {})),
            error=data.get("error"),
        )


def _parse_evaluation_status(data: dict[str, Any]) -> EvaluationStatus:
    """Determine an ``EvaluationResult``'s status from persisted data.

    Records written before ``status`` existed have no such key. For those,
    infer the closest equivalent from the fields that do exist, rather than
    defaulting every old record to ``SUCCESS`` regardless of what actually
    happened: a record with an ``error`` was an execution failure, and a
    record with neither a score nor a passed judgment (but a descriptive
    label, as the pre-status ``ExactMatchEvaluator`` produced for a missing
    reference) was effectively skipped.
    """
    if "status" in data:
        return EvaluationStatus(data["status"])
    if data.get("error") is not None:
        return EvaluationStatus.EXECUTION_ERROR
    if data.get("score") is None and data.get("passed") is None and data.get("label") is not None:
        return EvaluationStatus.SKIPPED
    return EvaluationStatus.SUCCESS


@dataclass
class TestCaseResult:
    """The complete, per-test-case outcome of an evaluation run."""

    # Tells pytest not to try to collect this domain class as a test class;
    # it is unrelated to pytest, but its name starts with "Test".
    __test__ = False

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestCaseResult:
        model_response_data = data.get("model_response")
        return cls(
            test_case_id=data["test_case_id"],
            input_text=data["input_text"],
            reference_answer=data.get("reference_answer"),
            model_response=(
                ModelResponse.from_dict(model_response_data)
                if model_response_data is not None
                else None
            ),
            evaluation_results=[
                EvaluationResult.from_dict(result) for result in data.get("evaluation_results", [])
            ],
            execution_error=data.get("execution_error"),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        return cls(
            dataset_name=data["dataset_name"],
            model_id=data["model_id"],
            evaluator_names=list(data["evaluator_names"]),
            test_case_results=[
                TestCaseResult.from_dict(result) for result in data["test_case_results"]
            ],
        )

    @classmethod
    def from_json(cls, text: str) -> RunResult:
        return cls.from_dict(json.loads(text))

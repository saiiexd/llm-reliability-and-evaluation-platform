"""
Model/application adapters for the Core Evaluation Engine.

An adapter is the only place the evaluation engine touches a specific
model or application. The engine depends only on the ``ModelAdapter``
interface defined here, never on a specific provider SDK, so it can run
against a mock, a real LLM provider, or a future RAG application without
any change to the execution or evaluation logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_reliability.evaluation.models import ExecutionRequest, ModelResponse


class AdapterError(Exception):
    """Raised when an adapter fails to produce a model response."""


class ModelAdapter(ABC):
    """Interface between the evaluation engine and a model or application.

    Implementations must not fabricate fields they cannot measure (for
    example latency or token usage) -- leave them as ``None`` instead.
    """

    @abstractmethod
    def generate(self, request: ExecutionRequest) -> ModelResponse:
        """Produce a model response for a single execution request.

        May raise ``AdapterError`` or let a provider-specific exception
        propagate if generation fails. The caller (the evaluation
        runner) is responsible for isolating that failure to the
        affected test case.
        """
        raise NotImplementedError


class MockAdapter(ModelAdapter):
    """Deterministic fake adapter for tests and local development.

    Makes no network calls and requires no credentials, so the full
    evaluation engine can be exercised without API access. Responses are
    fully determined by ``responses``, a mapping of test case id to
    output text; a test case without an explicit entry deterministically
    echoes its input text. ``errors`` optionally maps a test case id to
    an error message, causing ``generate`` to raise ``AdapterError`` for
    that test case, which is useful for testing failure handling.

    This adapter's output does not represent real model behavior and
    must never be used to draw conclusions about model quality.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        errors: dict[str, str] | None = None,
        model_id: str = "mock-adapter-v1",
    ) -> None:
        self._responses = dict(responses) if responses else {}
        self._errors = dict(errors) if errors else {}
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, request: ExecutionRequest) -> ModelResponse:
        if request.test_case_id in self._errors:
            raise AdapterError(self._errors[request.test_case_id])
        output_text = self._responses.get(request.test_case_id, request.input_text)
        return ModelResponse(
            test_case_id=request.test_case_id,
            output_text=output_text,
            model_id=self._model_id,
            provider_metadata={
                "adapter": "mock",
                "note": "deterministic fake adapter; does not reflect real model behavior",
            },
        )

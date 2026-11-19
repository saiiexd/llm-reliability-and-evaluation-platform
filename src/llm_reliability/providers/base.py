"""
Base provider abstractions, configuration, and normalized error hierarchy.

Enforces provider-independent contracts, bounded retries with backoff and jitter,
monitored wall-clock latency measurement, token provenance attribution, and
strict external inference safety controls.
"""

from __future__ import annotations

import random
import time
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llm_reliability.evaluation.adapters import AdapterError, ModelAdapter
from llm_reliability.evaluation.models import ExecutionRequest, ModelConfig, ModelResponse


class ProviderError(AdapterError):
    """Base exception for provider inference failures."""


class ProviderAuthenticationError(ProviderError):
    """Raised when provider credentials are missing, expired, or rejected."""


class ProviderRateLimitError(ProviderError):
    """Raised when provider rate limits or quotas are exceeded after retries."""


class ProviderTimeoutError(ProviderError):
    """Raised when a request to a provider times out."""


class ProviderConnectionError(ProviderError):
    """Raised when a network connection to a provider fails."""


class ProviderApiError(ProviderError):
    """Raised when a provider returns an unrecoverable 5xx or invalid API response."""


class ExternalInferenceNotPermittedError(ProviderError):
    """Raised when external live model inference is requested without explicit opt-in.

    Prevents accidental paid API consumption during local runs or test executions.
    """


@dataclass
class ProviderConfig:
    """Standardized configuration for an LLM provider integration."""

    provider_id: str
    model_id: str
    base_url: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    allow_external_inference: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_id or not self.provider_id.strip():
            raise ValueError("ProviderConfig.provider_id must be a non-empty string.")
        if not self.model_id or not self.model_id.strip():
            raise ValueError("ProviderConfig.model_id must be a non-empty string.")
        if self.timeout_seconds <= 0:
            raise ValueError("ProviderConfig.timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("ProviderConfig.max_retries must be non-negative.")


@dataclass
class ProviderInferenceResult:
    """Normalized payload extracted from a raw provider response."""

    output_text: str
    model_id: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    system_fingerprint: str | None = None
    provider_latency_ms: float | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


class BaseProviderAdapter(ModelAdapter):
    """Abstract base adapter for real LLM providers.

    Conforms to the ``ModelAdapter`` interface and manages:
    1. Safety checks (verifying ``allow_external_inference``).
    2. Monotonic wall-clock latency measurement.
    3. Bounded retry policy with exponential backoff and jitter.
    4. Structured token and telemetry provenance extraction.
    """

    def __init__(
        self,
        config: ProviderConfig,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._config = config
        self._sleeper = sleeper or time.sleep

    @property
    def config(self) -> ProviderConfig:
        return self._config

    @property
    def provider_id(self) -> str:
        return self._config.provider_id

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def generate(self, request: ExecutionRequest) -> ModelResponse:
        """Produce a response for an execution request with retries and timing."""
        if not self._config.allow_external_inference:
            raise ExternalInferenceNotPermittedError(
                f"External inference to provider '{self.provider_id}' is disabled. "
                "Explicit opt-in is required via 'allow_external_inference=True' to permit "
                "outbound network requests and potential paid usage."
            )

        start_perf = time.perf_counter()
        attempt = 0
        last_exception: Exception | None = None

        while attempt <= self._config.max_retries:
            attempt += 1
            try:
                result = self._execute_inference(request.input_text, request.model_config)
                end_perf = time.perf_counter()
                latency_ms = (end_perf - start_perf) * 1000.0

                # Determine token provenance
                has_tokens = (
                    result.prompt_tokens is not None
                    or result.completion_tokens is not None
                    or result.total_tokens is not None
                )
                token_source = "provider_reported" if has_tokens else "unavailable"

                combined_metadata: dict[str, Any] = {
                    "provider": self.provider_id,
                    "model": result.model_id,
                    "finish_reason": result.finish_reason,
                    "system_fingerprint": result.system_fingerprint,
                    "attempts": attempt,
                    "provider_latency_ms": result.provider_latency_ms,
                    "token_source": token_source,
                }
                combined_metadata.update(result.provider_metadata)

                return ModelResponse(
                    test_case_id=request.test_case_id,
                    output_text=result.output_text,
                    model_id=result.model_id,
                    latency_ms=latency_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    provider_metadata=combined_metadata,
                )

            except (ProviderRateLimitError, ProviderTimeoutError, ProviderConnectionError) as exc:
                last_exception = exc
                if attempt > self._config.max_retries:
                    raise exc

                # Exponential backoff with jitter
                delay = (self._config.backoff_factor * (2 ** (attempt - 1))) + random.uniform(
                    0.05, 0.25
                )
                self._sleeper(delay)

            except (ProviderAuthenticationError, ExternalInferenceNotPermittedError) as exc:
                # Immediate non-retryable domain exceptions
                raise exc

            except Exception as exc:
                # Check if this exception is retryable by provider-specific mapping
                normalized = self._normalize_exception(exc)
                if isinstance(
                    normalized,
                    (ProviderRateLimitError, ProviderTimeoutError, ProviderConnectionError),
                ):
                    last_exception = normalized
                    if attempt > self._config.max_retries:
                        raise normalized from exc
                    delay = (self._config.backoff_factor * (2 ** (attempt - 1))) + random.uniform(
                        0.05, 0.25
                    )
                    self._sleeper(delay)
                else:
                    raise normalized from exc

        if last_exception:
            raise last_exception
        raise ProviderError(f"Exhausted {self._config.max_retries} attempts without a response.")

    @abstractmethod
    def _execute_inference(self, prompt: str, model_config: ModelConfig) -> ProviderInferenceResult:
        """Call underlying provider SDK or HTTP client and return normalized result."""
        raise NotImplementedError

    @abstractmethod
    def _normalize_exception(self, exc: Exception) -> ProviderError:
        """Map raw provider or client SDK exceptions into normalized ProviderError."""
        raise NotImplementedError

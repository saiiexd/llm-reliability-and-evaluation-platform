"""
OpenAI and OpenAI-compatible model provider adapter.

Supports real API inference via OpenAI endpoints (e.g. gpt-4o, gpt-4o-mini) and
OpenAI-compatible services (vLLM, Ollama, LM Studio) using the official OpenAI SDK.
Ensures token usage provenance, monotonic timing, retry isolation, and credential safety.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import openai
from openai import OpenAI

from llm_reliability.evaluation.models import ModelConfig
from llm_reliability.providers.base import (
    BaseProviderAdapter,
    ProviderApiError,
    ProviderAuthenticationError,
    ProviderConfig,
    ProviderConnectionError,
    ProviderError,
    ProviderInferenceResult,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


def get_openai_base_url_from_env() -> str | None:
    """Base URL for OpenAI-compatible servers, from the server environment."""
    return os.environ.get("LLM_RELIABILITY_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")


def get_openai_api_key_from_env() -> str | None:
    """Retrieve OpenAI API key strictly from server environment variables."""
    return os.environ.get("LLM_RELIABILITY_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")


class OpenAIProviderAdapter(BaseProviderAdapter):
    """Adapter for OpenAI models and OpenAI-compatible local/remote endpoints."""

    def __init__(
        self,
        config: ProviderConfig,
        client: OpenAI | None = None,
        api_key: str | None = None,
        sleeper: Callable[[float], None] | None = None,
        use_request_model: bool = False,
    ) -> None:
        """``use_request_model`` selects which model id is sent to the provider.

        ``False`` (default): the adapter is bound to ``config.model_id`` (for
        example the allowlisted ``openai-gpt-4o-mini`` adapter always calls
        ``gpt-4o-mini``), regardless of the experiment's declared model id.
        ``True``: the experiment's ``ModelConfig.model_id`` is sent, which is
        what a generic OpenAI-compatible endpoint (Ollama, vLLM) needs, since
        the served model name is chosen per experiment.
        """
        super().__init__(config=config, sleeper=sleeper)
        self._use_request_model = use_request_model

        if client is not None:
            self._client = client
        else:
            resolved_key = api_key or get_openai_api_key_from_env()
            base_url = config.base_url or get_openai_base_url_from_env()

            # Local servers (e.g. Ollama, vLLM) typically don't require a real OpenAI key
            is_local = base_url and any(
                host in base_url.lower() for host in ("localhost", "127.0.0.1", "0.0.0.0")
            )

            if not resolved_key and is_local:
                resolved_key = "local-dummy-key"

            if not resolved_key:
                raise ProviderAuthenticationError(
                    "OpenAI API key is missing. Set the OPENAI_API_KEY or "
                    "LLM_RELIABILITY_OPENAI_API_KEY environment variable on the server."
                )

            self._client = OpenAI(
                api_key=resolved_key,
                base_url=base_url,
                timeout=config.timeout_seconds,
                max_retries=0,  # We manage our own bounded retry loop with full auditability
            )

    @property
    def client(self) -> OpenAI:
        return self._client

    def _execute_inference(self, prompt: str, model_config: ModelConfig) -> ProviderInferenceResult:
        model = (
            model_config.model_id
            if self._use_request_model and model_config.model_id
            else self.config.model_id
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model_config.temperature is not None:
            kwargs["temperature"] = model_config.temperature
        if model_config.max_tokens is not None:
            kwargs["max_tokens"] = model_config.max_tokens

        # Merge extra parameters if specified
        if model_config.extra_params:
            for k, v in model_config.extra_params.items():
                if k not in kwargs and k not in ("rag", "provider"):
                    kwargs[k] = v

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise self._normalize_exception(exc) from exc

        if not response.choices:
            raise ProviderApiError(f"Provider returned an empty choices list for model '{model}'.")

        choice = response.choices[0]
        output_text = choice.message.content or ""
        finish_reason = choice.finish_reason

        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None

        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

        return ProviderInferenceResult(
            output_text=output_text,
            model_id=response.model or model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            system_fingerprint=getattr(response, "system_fingerprint", None),
            provider_latency_ms=None,
            provider_metadata={},
        )

    def _normalize_exception(self, exc: Exception) -> ProviderError:
        """Map OpenAI SDK exceptions into normalized ProviderError types."""
        if isinstance(exc, ProviderError):
            return exc

        if isinstance(exc, openai.AuthenticationError):
            return ProviderAuthenticationError(f"OpenAI Authentication failed: {exc}")

        if isinstance(exc, openai.RateLimitError):
            return ProviderRateLimitError(f"OpenAI Rate limit exceeded: {exc}")

        if isinstance(exc, openai.APITimeoutError):
            return ProviderTimeoutError(f"OpenAI Request timed out: {exc}")

        if isinstance(exc, openai.APIConnectionError):
            return ProviderConnectionError(f"OpenAI Connection error: {exc}")

        if isinstance(exc, openai.InternalServerError):
            return ProviderApiError(f"OpenAI Internal server error (5xx): {exc}")

        if isinstance(exc, openai.APIStatusError):
            if exc.status_code == 429:
                return ProviderRateLimitError(f"OpenAI Rate limit error ({exc.status_code}): {exc}")
            if exc.status_code in (401, 403):
                return ProviderAuthenticationError(
                    f"OpenAI Authentication error ({exc.status_code}): {exc}"
                )
            if exc.status_code >= 500:
                return ProviderApiError(f"OpenAI Server error ({exc.status_code}): {exc}")
            return ProviderError(f"OpenAI API status error ({exc.status_code}): {exc}")

        return ProviderError(f"Unexpected provider error ({type(exc).__name__}): {exc}")

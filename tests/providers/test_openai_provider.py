"""
Offline unit tests for OpenAIProviderAdapter and BaseProviderAdapter.

Exercises generation, token provenance, retry backoff, error normalization,
monotonic timing, and external inference safety without network calls or credit spend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import openai
import pytest

from llm_reliability.evaluation.models import ExecutionRequest, ModelConfig
from llm_reliability.providers.base import (
    ExternalInferenceNotPermittedError,
    ProviderAuthenticationError,
    ProviderConfig,
    ProviderRateLimitError,
)
from llm_reliability.providers.openai_provider import OpenAIProviderAdapter


def _build_fake_chat_completion(
    content: str = "Test response",
    model: str = "gpt-4o-mini",
    prompt_tokens: int | None = 15,
    completion_tokens: int | None = 25,
    total_tokens: int | None = 40,
    finish_reason: str = "stop",
    system_fingerprint: str | None = "fp_test_123",
) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    response.choices = [choice]
    response.model = model
    response.system_fingerprint = system_fingerprint

    if prompt_tokens is not None or completion_tokens is not None:
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.total_tokens = total_tokens
        response.usage = usage
    else:
        response.usage = None

    return response


def test_safety_guard_blocks_external_inference_when_not_permitted() -> None:
    """When allow_external_inference is False, generate() must refuse to run."""
    mock_client = MagicMock()
    config = ProviderConfig(
        provider_id="openai",
        model_id="gpt-4o-mini",
        allow_external_inference=False,  # Explicitly disabled
    )
    adapter = OpenAIProviderAdapter(config=config, client=mock_client)

    request = ExecutionRequest(
        test_case_id="tc-1",
        input_text="Hello world",
        model_config=ModelConfig(model_id="gpt-4o-mini"),
    )

    with pytest.raises(ExternalInferenceNotPermittedError) as exc_info:
        adapter.generate(request)

    assert "External inference to provider 'openai' is disabled" in str(exc_info.value)
    # Ensure client was never touched
    mock_client.chat.completions.create.assert_not_called()


def test_successful_generation_with_tokens_and_provenance() -> None:
    """When inference succeeds, token counts and provenance are faithfully recorded."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_fake_chat_completion(
        content="Paris is the capital of France.",
        model="gpt-4o-mini-2024-07-18",
        prompt_tokens=12,
        completion_tokens=8,
        total_tokens=20,
        finish_reason="stop",
        system_fingerprint="fp_abc123",
    )

    config = ProviderConfig(
        provider_id="openai",
        model_id="gpt-4o-mini",
        allow_external_inference=True,
    )
    adapter = OpenAIProviderAdapter(config=config, client=mock_client)

    request = ExecutionRequest(
        test_case_id="tc-capital",
        input_text="What is the capital of France?",
        model_config=ModelConfig(model_id="gpt-4o-mini", temperature=0.2, max_tokens=50),
    )

    resp = adapter.generate(request)

    assert resp.test_case_id == "tc-capital"
    assert resp.output_text == "Paris is the capital of France."
    assert resp.model_id == "gpt-4o-mini-2024-07-18"
    assert resp.prompt_tokens == 12
    assert resp.completion_tokens == 8
    assert resp.total_tokens == 20
    assert resp.latency_ms is not None
    assert resp.latency_ms >= 0.0

    meta = resp.provider_metadata
    assert meta["provider"] == "openai"
    assert meta["finish_reason"] == "stop"
    assert meta["system_fingerprint"] == "fp_abc123"
    assert meta["attempts"] == 1
    assert meta["token_source"] == "provider_reported"

    # Verify call parameters
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0.2,
        max_tokens=50,
    )


def test_generation_without_reported_tokens_never_fabricates_zeros() -> None:
    """When provider returns no usage, tokens are None and token_source is unavailable."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_fake_chat_completion(
        content="Answer without usage",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )

    config = ProviderConfig(
        provider_id="openai-compatible",
        model_id="custom-model",
        allow_external_inference=True,
    )
    adapter = OpenAIProviderAdapter(config=config, client=mock_client)

    request = ExecutionRequest(
        test_case_id="tc-no-tokens",
        input_text="Echo me",
        model_config=ModelConfig(model_id="custom-model"),
    )

    resp = adapter.generate(request)
    assert resp.prompt_tokens is None
    assert resp.completion_tokens is None
    assert resp.total_tokens is None
    assert resp.provider_metadata["token_source"] == "unavailable"


def test_retry_on_rate_limit_with_exponential_backoff() -> None:
    """When a 429 rate limit occurs, adapter retries with backoff and succeeds on 2nd try."""
    mock_client = MagicMock()
    sleep_calls: list[float] = []

    # 1st call raises RateLimitError, 2nd call succeeds
    rate_err = openai.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    success_resp = _build_fake_chat_completion(content="Recovered after rate limit")

    mock_client.chat.completions.create.side_effect = [rate_err, success_resp]

    config = ProviderConfig(
        provider_id="openai",
        model_id="gpt-4o-mini",
        max_retries=3,
        backoff_factor=0.1,
        allow_external_inference=True,
    )
    adapter = OpenAIProviderAdapter(
        config=config,
        client=mock_client,
        sleeper=lambda d: sleep_calls.append(d),
    )

    request = ExecutionRequest(
        test_case_id="tc-retry",
        input_text="Test prompt",
        model_config=ModelConfig(model_id="gpt-4o-mini"),
    )

    resp = adapter.generate(request)

    assert resp.output_text == "Recovered after rate limit"
    assert resp.provider_metadata["attempts"] == 2
    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0.0
    assert mock_client.chat.completions.create.call_count == 2


def test_exhausted_retries_raises_normalized_rate_limit_error() -> None:
    """When retries are exhausted on 429, raises ProviderRateLimitError."""
    mock_client = MagicMock()
    sleep_calls: list[float] = []

    rate_err = openai.RateLimitError(
        message="Quota exceeded",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    mock_client.chat.completions.create.side_effect = rate_err

    config = ProviderConfig(
        provider_id="openai",
        model_id="gpt-4o-mini",
        max_retries=2,
        backoff_factor=0.01,
        allow_external_inference=True,
    )
    adapter = OpenAIProviderAdapter(
        config=config,
        client=mock_client,
        sleeper=lambda d: sleep_calls.append(d),
    )

    request = ExecutionRequest(
        test_case_id="tc-fail",
        input_text="Test prompt",
        model_config=ModelConfig(model_id="gpt-4o-mini"),
    )

    with pytest.raises(ProviderRateLimitError) as exc_info:
        adapter.generate(request)

    assert "Rate limit" in str(exc_info.value)
    assert mock_client.chat.completions.create.call_count == 3  # Initial + 2 retries
    assert len(sleep_calls) == 2


def test_authentication_error_is_not_retried() -> None:
    """Authentication errors (401) fail immediately without retrying."""
    mock_client = MagicMock()
    sleep_calls: list[float] = []

    auth_err = openai.AuthenticationError(
        message="Incorrect API key provided",
        response=MagicMock(status_code=401, headers={}),
        body=None,
    )
    mock_client.chat.completions.create.side_effect = auth_err

    config = ProviderConfig(
        provider_id="openai",
        model_id="gpt-4o-mini",
        max_retries=3,
        allow_external_inference=True,
    )
    adapter = OpenAIProviderAdapter(
        config=config,
        client=mock_client,
        sleeper=lambda d: sleep_calls.append(d),
    )

    request = ExecutionRequest(
        test_case_id="tc-auth",
        input_text="Test prompt",
        model_config=ModelConfig(model_id="gpt-4o-mini"),
    )

    with pytest.raises(ProviderAuthenticationError):
        adapter.generate(request)

    # Must fail immediately on 1st attempt
    assert mock_client.chat.completions.create.call_count == 1
    assert len(sleep_calls) == 0


def test_missing_api_key_raises_auth_error_without_leaking_secrets() -> None:
    """Initializing OpenAI adapter with no key in env or args raises clear error."""
    config = ProviderConfig(provider_id="openai", model_id="gpt-4o-mini")

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        OpenAIProviderAdapter(config=config, client=None, api_key=None)

    assert "OpenAI API key is missing" in str(exc_info.value)

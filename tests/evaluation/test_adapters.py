"""Tests for the deterministic mock adapter."""

import pytest

from llm_reliability.evaluation import AdapterError, ExecutionRequest, ModelConfig, MockAdapter


def _request(test_case_id: str, input_text: str) -> ExecutionRequest:
    return ExecutionRequest(
        test_case_id=test_case_id,
        input_text=input_text,
        model_config=ModelConfig(model_id="mock-adapter-v1"),
    )


class TestMockAdapter:
    def test_returns_configured_response(self):
        adapter = MockAdapter(responses={"t1": "four"})
        response = adapter.generate(_request("t1", "What is 2+2?"))
        assert response.output_text == "four"
        assert response.test_case_id == "t1"
        assert response.model_id == "mock-adapter-v1"

    def test_unconfigured_test_case_echoes_input(self):
        adapter = MockAdapter()
        response = adapter.generate(_request("t2", "echo me"))
        assert response.output_text == "echo me"

    def test_is_deterministic_across_repeated_calls(self):
        adapter = MockAdapter(responses={"t1": "four"})
        first = adapter.generate(_request("t1", "What is 2+2?"))
        second = adapter.generate(_request("t1", "What is 2+2?"))
        assert first.output_text == second.output_text

    def test_does_not_fabricate_timing_or_token_usage(self):
        adapter = MockAdapter()
        response = adapter.generate(_request("t1", "q"))
        assert response.latency_ms is None
        assert response.prompt_tokens is None
        assert response.completion_tokens is None
        assert response.total_tokens is None

    def test_configured_error_is_raised(self):
        adapter = MockAdapter(errors={"t1": "simulated provider failure"})
        with pytest.raises(AdapterError, match="simulated provider failure"):
            adapter.generate(_request("t1", "q"))

    def test_custom_model_id_is_reflected_in_response(self):
        adapter = MockAdapter(model_id="mock-adapter-v2")
        response = adapter.generate(_request("t1", "q"))
        assert response.model_id == "mock-adapter-v2"

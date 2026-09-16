"""Tests for the deterministic content fingerprinting utility."""

from llm_reliability.experiments.fingerprint import (
    compute_dataset_fingerprint,
    compute_experiment_fingerprint,
    content_fingerprint,
)


class TestContentFingerprint:
    def test_identical_payloads_produce_identical_fingerprints(self):
        payload = {"a": 1, "b": [1, 2, 3]}
        assert content_fingerprint(payload) == content_fingerprint(payload)

    def test_dictionary_key_order_does_not_affect_fingerprint(self):
        first = {"a": 1, "b": 2}
        second = {"b": 2, "a": 1}
        assert content_fingerprint(first) == content_fingerprint(second)

    def test_different_payloads_produce_different_fingerprints(self):
        assert content_fingerprint({"a": 1}) != content_fingerprint({"a": 2})

    def test_fingerprint_is_a_sha256_hex_digest(self):
        digest = content_fingerprint({"a": 1})
        assert len(digest) == 64
        int(digest, 16)  # raises ValueError if not valid hex


class TestDatasetFingerprint:
    def test_identical_test_cases_produce_identical_fingerprint(self):
        test_cases = [{"id": "t1", "input": "q", "reference_answer": "a", "metadata": {}}]
        assert compute_dataset_fingerprint(test_cases) == compute_dataset_fingerprint(test_cases)

    def test_different_test_case_content_produces_different_fingerprint(self):
        first = [{"id": "t1", "input": "q1", "reference_answer": "a", "metadata": {}}]
        second = [{"id": "t1", "input": "q2", "reference_answer": "a", "metadata": {}}]
        assert compute_dataset_fingerprint(first) != compute_dataset_fingerprint(second)


class TestExperimentFingerprint:
    def _base_args(self):
        return {
            "dataset_version_id": "abc123",
            "model_config_payload": {"model_id": "m1", "temperature": None},
            "evaluator_configs_payload": [{"evaluator_name": "exact_match", "parameters": {}}],
        }

    def test_equivalent_configuration_produces_identical_fingerprint(self):
        args = self._base_args()
        first = compute_experiment_fingerprint(**args)
        second = compute_experiment_fingerprint(**args)
        assert first == second

    def test_different_dataset_version_changes_fingerprint(self):
        args = self._base_args()
        baseline = compute_experiment_fingerprint(**args)
        args["dataset_version_id"] = "different"
        assert compute_experiment_fingerprint(**args) != baseline

    def test_different_model_config_changes_fingerprint(self):
        args = self._base_args()
        baseline = compute_experiment_fingerprint(**args)
        args["model_config_payload"] = {"model_id": "m2", "temperature": None}
        assert compute_experiment_fingerprint(**args) != baseline

    def test_different_evaluator_config_changes_fingerprint(self):
        args = self._base_args()
        baseline = compute_experiment_fingerprint(**args)
        args["evaluator_configs_payload"] = [
            {"evaluator_name": "exact_match", "parameters": {"case_sensitive": True}}
        ]
        assert compute_experiment_fingerprint(**args) != baseline

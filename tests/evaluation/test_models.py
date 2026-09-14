"""Tests for domain model construction and validation."""

import pytest

from llm_reliability.evaluation import Dataset, ModelConfig, TestCase


class TestTestCase:
    def test_valid_test_case_with_reference(self):
        test_case = TestCase(id="t1", input="What is 2+2?", reference_answer="4")
        assert test_case.id == "t1"
        assert test_case.input == "What is 2+2?"
        assert test_case.reference_answer == "4"
        assert test_case.metadata == {}

    def test_valid_test_case_without_reference(self):
        test_case = TestCase(id="t1", input="Tell me a fact.")
        assert test_case.reference_answer is None

    def test_test_case_with_metadata(self):
        test_case = TestCase(id="t1", input="q", metadata={"category": "math"})
        assert test_case.metadata == {"category": "math"}

    @pytest.mark.parametrize("bad_id", ["", "   "])
    def test_empty_id_is_rejected(self, bad_id):
        with pytest.raises(ValueError, match="id"):
            TestCase(id=bad_id, input="q")

    @pytest.mark.parametrize("bad_input", ["", "   "])
    def test_empty_input_is_rejected(self, bad_input):
        with pytest.raises(ValueError, match="input"):
            TestCase(id="t1", input=bad_input)


class TestDataset:
    def test_valid_dataset(self):
        dataset = Dataset(
            name="arithmetic",
            test_cases=[
                TestCase(id="t1", input="1+1?", reference_answer="2"),
                TestCase(id="t2", input="2+2?", reference_answer="4"),
            ],
        )
        assert len(dataset) == 2
        assert [tc.id for tc in dataset] == ["t1", "t2"]

    def test_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="name"):
            Dataset(name="", test_cases=[TestCase(id="t1", input="q")])

    def test_empty_test_case_list_is_rejected(self):
        with pytest.raises(ValueError, match="at least one test case"):
            Dataset(name="empty", test_cases=[])

    def test_duplicate_ids_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            Dataset(
                name="dup",
                test_cases=[
                    TestCase(id="t1", input="a"),
                    TestCase(id="t1", input="b"),
                ],
            )

    def test_iteration_preserves_order(self):
        ids = ["t3", "t1", "t2"]
        dataset = Dataset(name="ordered", test_cases=[TestCase(id=i, input="q") for i in ids])
        assert [tc.id for tc in dataset] == ids


class TestModelConfig:
    def test_minimal_valid_config(self):
        config = ModelConfig(model_id="mock-adapter-v1")
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.extra_params == {}

    def test_config_with_generation_params(self):
        config = ModelConfig(model_id="mock-adapter-v1", temperature=0.2, max_tokens=128)
        assert config.temperature == 0.2
        assert config.max_tokens == 128

    def test_empty_model_id_is_rejected(self):
        with pytest.raises(ValueError, match="model_id"):
            ModelConfig(model_id="")

    def test_negative_temperature_is_rejected(self):
        with pytest.raises(ValueError, match="temperature"):
            ModelConfig(model_id="m", temperature=-0.1)

    def test_non_positive_max_tokens_is_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ModelConfig(model_id="m", max_tokens=0)

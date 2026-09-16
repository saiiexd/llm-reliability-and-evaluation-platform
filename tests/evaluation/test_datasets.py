"""Tests for loading Dataset objects from JSON files, including the curated benchmark."""

from pathlib import Path

import pytest

from llm_reliability.evaluation import Dataset, load_dataset_from_json

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "eval_sets"
    / "evaluation_methodology_baseline_v1.json"
)


def test_load_dataset_from_json_round_trips(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(
        '{"name": "d1", "test_cases": [{"id": "t1", "input": "q", "reference_answer": "a"}]}',
        encoding="utf-8",
    )
    dataset = load_dataset_from_json(path)
    assert isinstance(dataset, Dataset)
    assert dataset.name == "d1"
    assert dataset.test_cases[0].id == "t1"


def test_load_dataset_from_json_rejects_malformed_content(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text('{"name": "", "test_cases": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset_from_json(path)


class TestEvaluationMethodologyBenchmark:
    """The curated benchmark dataset used to validate the evaluation layer."""

    def test_benchmark_file_exists(self):
        assert BENCHMARK_PATH.exists()

    def test_benchmark_loads_and_validates(self):
        dataset = load_dataset_from_json(BENCHMARK_PATH)
        assert isinstance(dataset, Dataset)

    def test_benchmark_size_is_intentionally_small(self):
        dataset = load_dataset_from_json(BENCHMARK_PATH)
        assert 20 <= len(dataset) <= 50

    def test_benchmark_covers_all_three_categories(self):
        dataset = load_dataset_from_json(BENCHMARK_PATH)
        categories = {tc.metadata.get("category") for tc in dataset.test_cases}
        assert categories == {
            "exact_factual",
            "paraphrasable",
            "ambiguous_or_insufficient_reference",
        }

    def test_benchmark_includes_reference_free_cases(self):
        dataset = load_dataset_from_json(BENCHMARK_PATH)
        assert any(tc.reference_answer is None for tc in dataset.test_cases)

    def test_benchmark_test_case_ids_are_unique(self):
        dataset = load_dataset_from_json(BENCHMARK_PATH)
        ids = [tc.id for tc in dataset.test_cases]
        assert len(ids) == len(set(ids))

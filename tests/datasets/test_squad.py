"""Tests for the bundled SQuAD subset loader and its frozen paraphrase variant."""

import json
from pathlib import Path

from llm_reliability.datasets.squad import (
    PARAPHRASES_PATH,
    load_squad_subset,
    paraphrased_dataset,
)
from llm_reliability.experiments.models import DatasetVersion

REPO = Path(__file__).resolve().parents[2]


def test_loader_reproduces_the_exp001_dataset_and_corpus_fingerprints():
    stored = json.loads((REPO / "experiments/experiments/exp001-qwen-rag-k3.json").read_text())
    subset = load_squad_subset()
    assert (
        DatasetVersion(dataset=subset.dataset()).version_id
        == stored["dataset_version"]["version_id"]
    )
    rag = stored["model_config"]["extra_params"]["rag"]
    assert subset.corpus_fingerprint() == rag["corpus_fingerprint"]


def test_every_question_has_a_relevant_chunk_label():
    dataset = load_squad_subset().dataset()
    assert len(dataset.test_cases) == 60
    assert all(tc.metadata["relevant_chunk_ids"] for tc in dataset.test_cases)


def test_paraphrased_dataset_pairs_by_id_and_keeps_answers_and_labels():
    original = {tc.id: tc for tc in load_squad_subset().dataset().test_cases}
    para = paraphrased_dataset()
    assert len(para.test_cases) == 60
    for tc in para.test_cases:
        base = original[tc.id]
        assert tc.input != base.input
        assert tc.reference_answer == base.reference_answer
        assert tc.metadata["relevant_chunk_ids"] == base.metadata["relevant_chunk_ids"]
        assert tc.metadata["perturbation_of"] == tc.id
        assert tc.metadata["perturbation_type"] == "question_rewording"
        assert tc.metadata["original_question"] == base.input
    assert (
        DatasetVersion(dataset=para).version_id
        != DatasetVersion(dataset=load_squad_subset().dataset()).version_id
    )


def test_paraphrase_file_records_generator_and_review():
    raw = json.loads(PARAPHRASES_PATH.read_text())
    assert raw["generator"]["model"] and raw["generator"]["prompt"]
    statuses = [item["meaning_review"]["status"] for item in raw["items"]]
    assert set(statuses) <= {"same_meaning", "minor_shift", "meaning_changed"}
    assert "not a human" in raw["meaning_review"]["reviewer"]

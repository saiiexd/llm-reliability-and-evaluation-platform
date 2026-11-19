"""
Tests for the human-annotation sheet export and the evaluator-vs-human analysis.

Every label in this file is SYNTHETIC: invented test data written into
temporary directories to exercise the code paths (all fixture names,
experiment ids and annotator names say "synthetic"). None of it is, or may be
used as, a human judgement. The only committed EXP-001 files these tests read
are checked to contain no labels at all.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import pytest

from llm_reliability.analytics.agreement import cohens_kappa, cohens_kappa_nominal
from llm_reliability.analytics.annotation import (
    KEY_FILENAME,
    OVERLAP_SHEET_FILENAME,
    SHEET_COLUMNS,
    SHEET_FILENAME,
    annotation_normalize,
    export_annotation_sheet,
)
from llm_reliability.analytics.human_agreement import (
    AnnotationFormatError,
    NoLabelsError,
    SheetLabels,
    analyze_human_agreement,
    inter_annotator_agreement,
    render_markdown,
)
from llm_reliability.evaluation.models import (
    Dataset,
    EvaluationResult,
    EvaluationStatus,
    ModelConfig,
    ModelResponse,
    RunResult,
    TestCase,
    TestCaseResult,
)
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.models import (
    DatasetVersion,
    EvaluatorConfig,
    ExperimentDefinition,
    ExperimentRun,
    RunStatus,
)
from scripts.analyze_human_agreement import main as analyze_main

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP001_DIR = REPO_ROOT / "research" / "annotation" / "exp001"
GUIDELINE = REPO_ROOT / "research" / "annotation" / "guideline.md"

OK = EvaluationStatus.SUCCESS
# verdict spec per evaluator: (status, passed, label)
Verdicts = dict[str, tuple[EvaluationStatus, bool | None, str | None]]


# --- synthetic store


def _definition(experiment_id: str, cases: list[TestCase]) -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_id=experiment_id,
        name=f"SYNTHETIC {experiment_id}",
        dataset_version=DatasetVersion(dataset=Dataset(name="synthetic", test_cases=cases)),
        model_config=ModelConfig(model_id="synthetic-model"),
        evaluator_configs=[
            EvaluatorConfig(evaluator_name=n)
            for n in ("exact_match", "squad_match", "llm_judge", "llm_faithfulness_judge")
        ],
    )


def _save_run(
    repo: LocalFileExperimentRepository,
    definition: ExperimentDefinition,
    run_id: str,
    repetition_index: int,
    answers: dict[str, tuple[str | None, Verdicts]],
) -> None:
    cases = {tc.id: tc for tc in definition.dataset_version.dataset.test_cases}
    results = []
    for tc_id, (answer, verdicts) in answers.items():
        results.append(
            TestCaseResult(
                test_case_id=tc_id,
                input_text=cases[tc_id].input,
                reference_answer=cases[tc_id].reference_answer,
                model_response=None
                if answer is None
                else ModelResponse(test_case_id=tc_id, output_text=answer, model_id="synthetic"),
                execution_error="synthetic crash" if answer is None else None,
                evaluation_results=[
                    EvaluationResult(
                        evaluator_name=name,
                        test_case_id=tc_id,
                        status=status,
                        passed=passed,
                        label=label,
                    )
                    for name, (status, passed, label) in verdicts.items()
                ],
            )
        )
    repo.save_run(
        ExperimentRun(
            run_id=run_id,
            experiment_id=definition.experiment_id,
            status=RunStatus.SUCCEEDED,
            started_at="2026-11-19T00:00:00Z",
            finished_at="2026-11-19T00:00:01Z",
            dataset_version=definition.dataset_version,
            model_config=definition.model_config,
            evaluator_configs=definition.evaluator_configs,
            config_fingerprint=definition.config_fingerprint,
            result=RunResult(
                dataset_name="synthetic",
                model_id="synthetic-model",
                evaluator_names=[ec.evaluator_name for ec in definition.evaluator_configs],
                test_case_results=results,
            ),
            repetition_index=repetition_index,
        )
    )


def _v(em: bool, judge: bool = True, faith: str | None = None) -> Verdicts:
    verdicts: Verdicts = {
        "exact_match": (OK, em, None),
        "squad_match": (OK, em, None),
        "llm_judge": (OK, judge, "correct" if judge else "incorrect"),
    }
    if faith is None:
        verdicts["llm_faithfulness_judge"] = (EvaluationStatus.SKIPPED, None, None)
    elif faith == "error":
        verdicts["llm_faithfulness_judge"] = (EvaluationStatus.OUTPUT_VALIDATION_ERROR, None, None)
    else:
        verdicts["llm_faithfulness_judge"] = (OK, faith == "supported", faith)
    return verdicts


SYNTHETIC_CASES = [
    TestCase(id="q1", input="Capital of France?", reference_answer="Paris"),
    TestCase(
        id="q2",
        input="Population?",
        reference_answer="3.5 million",
        metadata={"gold_answers": ["3.5 million", "3.5 million", "3,500,000"]},
    ),
    TestCase(id="q3", input="Largest city?", reference_answer="Paris"),
    TestCase(id="q4", input="River?", reference_answer="Seine"),
]


SYNTHETIC_EXPERIMENTS = ["exp-synthetic-closed", "exp-synthetic-rag"]
SYNTHETIC_LABELS_A = {
    ("q1", "Paris"): "correct",
    ("q2", "3.5 million"): "correct",
    ("q2", "35 million"): "incorrect",
    ("q3", "I don't know"): "incorrect",
    ("q3", "Lyon"): "incorrect",
}


@pytest.fixture
def synthetic_store(tmp_path: Path) -> Path:
    """Two SYNTHETIC experiments, three runs, with duplicates and one crashed case."""
    store = tmp_path / "synthetic_store"
    repo = LocalFileExperimentRepository(store)
    closed = _definition("exp-synthetic-closed", SYNTHETIC_CASES)
    rag = _definition("exp-synthetic-rag", SYNTHETIC_CASES)
    repo.save_experiment(closed)
    repo.save_experiment(rag)
    _save_run(
        repo,
        closed,
        "syn-run-c1",
        1,
        {
            "q1": ("Paris", _v(True)),
            "q2": ("3.5 million", _v(True)),
            "q3": ("I don't know", _v(False, judge=False)),
            "q4": (None, {}),
        },
    )
    _save_run(
        repo,
        rag,
        "syn-run-r1",
        1,
        {
            "q1": ("paris.", _v(False, faith="supported")),
            "q2": ("35 million", _v(False, judge=False, faith="error")),
            "q3": ("Lyon", _v(False, judge=False, faith="supported")),
        },
    )
    _save_run(
        repo,
        rag,
        "syn-run-r2",
        2,
        {
            "q1": ("The Paris", _v(False, faith="supported")),
            "q2": ("3.5 million", _v(True, faith="supported")),
            "q3": ("Lyon", _v(False, judge=True, faith="unsupported")),
        },
    )
    return store


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fill_synthetic(
    sheet: Path, key: Path, out: Path, label_for: dict[tuple[str, str], str], annotator: str
) -> Path:
    """Write a SYNTHETIC filled copy of ``sheet``; labels keyed by (test_case_id, answer)."""
    rows = _read(sheet)
    by_row = {k["row_id"]: (k["test_case_id"], k["normalized_answer"]) for k in _read(key)}
    for row in rows:
        tc_id, norm = by_row[row["row_id"]]
        for (want_tc, answer), label in label_for.items():
            if want_tc == tc_id and annotation_normalize(answer) == norm:
                row["human_correct"] = label
                row["annotator"] = annotator
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return out


# --- export


def test_export_dedups_identical_answers_and_maps_every_instance(
    synthetic_store: Path, tmp_path: Path
) -> None:
    out = tmp_path / "sheet"
    manifest = export_annotation_sheet(synthetic_store, out, SYNTHETIC_EXPERIMENTS)

    # 9 answered instances (q4 crashed: no answer to label) -> 5 distinct answers.
    assert manifest["instances"] == 9
    assert manifest["cases_without_response"] == 1
    assert manifest["rows"] == 5
    sheet, key = _read(out / SHEET_FILENAME), _read(out / KEY_FILENAME)
    assert len(key) == 9
    answers = {(r["question"], r["model_answer"]) for r in sheet}
    # "Paris" / "paris." / "The Paris" are one row; the tie picks the first variant.
    assert ("Capital of France?", "Paris") in answers
    # Numbers are kept apart even though SQuAD normalization maps both to "35 million".
    assert ("Population?", "3.5 million") in answers and ("Population?", "35 million") in answers
    q1_row = next(r["row_id"] for r in sheet if r["question"] == "Capital of France?")
    q1_instances = {(k["experiment_id"], k["run_id"]) for k in key if k["row_id"] == q1_row}
    assert q1_instances == {
        ("exp-synthetic-closed", "syn-run-c1"),
        ("exp-synthetic-rag", "syn-run-r1"),
        ("exp-synthetic-rag", "syn-run-r2"),
    }
    # Duplicate gold answers are shown once.
    q2 = next(r for r in sheet if r["model_answer"] == "35 million")
    assert q2["gold_answers"] == "3.5 million | 3,500,000"


def test_sheet_is_blind_empty_deterministic_and_marks_overlap(
    synthetic_store: Path, tmp_path: Path
) -> None:
    export_annotation_sheet(synthetic_store, tmp_path / "a", SYNTHETIC_EXPERIMENTS)
    export_annotation_sheet(synthetic_store, tmp_path / "b", SYNTHETIC_EXPERIMENTS)
    for name in (SHEET_FILENAME, OVERLAP_SHEET_FILENAME, KEY_FILENAME):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()

    text = (tmp_path / "a" / SHEET_FILENAME).read_text()
    assert text.splitlines()[0] == ",".join(SHEET_COLUMNS)
    for leak in ("exp-synthetic", "syn-run", "closed", "rag", "llm_judge", "True", "False"):
        assert leak not in text
    sheet = _read(tmp_path / "a" / SHEET_FILENAME)
    assert all(r["human_correct"] == r["annotator"] == r["notes"] == "" for r in sheet)

    key = _read(tmp_path / "a" / KEY_FILENAME)
    overlap = {k["row_id"] for k in key if k["overlap"] == "1"}
    assert len(overlap) == math.ceil(0.25 * 5)
    assert {r["row_id"] for r in _read(tmp_path / "a" / OVERLAP_SHEET_FILENAME)} == overlap

    with pytest.raises(ValueError, match="overlap_fraction"):
        export_annotation_sheet(
            synthetic_store, tmp_path / "c", SYNTHETIC_EXPERIMENTS, overlap_fraction=0.1
        )


# --- analysis


def test_refuses_when_no_label_is_filled(synthetic_store: Path, tmp_path: Path) -> None:
    sheet_dir = tmp_path / "sheet"
    export_annotation_sheet(synthetic_store, sheet_dir, SYNTHETIC_EXPERIMENTS)
    with pytest.raises(NoLabelsError):
        analyze_human_agreement(
            sheet_dir / KEY_FILENAME, sheet_dir / SHEET_FILENAME, synthetic_store
        )

    out = tmp_path / "analysis"
    with pytest.raises(SystemExit) as excinfo:
        analyze_main(
            [
                "--sheet",
                str(sheet_dir / SHEET_FILENAME),
                "--key",
                str(sheet_dir / KEY_FILENAME),
                "--store",
                str(synthetic_store),
                "--out",
                str(out),
            ]
        )
    assert excinfo.value.code == 2
    assert not out.exists()


def test_invalid_label_or_unknown_row_is_an_error(synthetic_store: Path, tmp_path: Path) -> None:
    sheet_dir = tmp_path / "sheet"
    export_annotation_sheet(synthetic_store, sheet_dir, SYNTHETIC_EXPERIMENTS)
    rows = _read(sheet_dir / SHEET_FILENAME)
    rows[0]["human_correct"] = "yes"  # SYNTHETIC, deliberately invalid
    bad = tmp_path / "SYNTHETIC_bad.csv"
    with bad.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(AnnotationFormatError, match="expected one of"):
        analyze_human_agreement(sheet_dir / KEY_FILENAME, bad, synthetic_store)

    rows[0]["human_correct"] = "correct"
    rows[0]["row_id"] = "R999"
    with bad.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(AnnotationFormatError, match="not in the key"):
        analyze_human_agreement(sheet_dir / KEY_FILENAME, bad, synthetic_store)


@pytest.fixture
def synthetic_hand_store(tmp_path: Path) -> Path:
    """One SYNTHETIC run of 13 questions with a hand-checkable confusion table for exact_match.

    q01-q04 TP, q05 FP, q06-q07 FN, q08-q10 TN; q11 human unsure; q12 exact_match
    errored; q13 left unlabelled.
    """
    cases = [
        TestCase(id=f"q{i:02d}", input=f"Question {i}?", reference_answer="x") for i in range(1, 14)
    ]
    definition = _definition("exp-synthetic-hand", cases)
    store = tmp_path / "synthetic_hand_store"
    repo = LocalFileExperimentRepository(store)
    repo.save_experiment(definition)
    em_passed = {1: True, 2: True, 3: True, 4: True, 5: True, 6: False, 7: False}
    answers: dict[str, tuple[str | None, Verdicts]] = {}
    for i in range(1, 14):
        verdicts = _v(em_passed.get(i, False))
        if i == 12:
            verdicts["exact_match"] = (EvaluationStatus.EXECUTION_ERROR, None, None)
        answers[f"q{i:02d}"] = (f"answer {i}", verdicts)
    _save_run(repo, definition, "syn-run-hand", 1, answers)
    return store


SYNTHETIC_HAND_LABELS = {
    **{(f"q{i:02d}", f"answer {i}"): "correct" for i in (1, 2, 3, 4, 6, 7, 12)},
    **{(f"q{i:02d}", f"answer {i}"): "incorrect" for i in (5, 8, 9, 10)},
    ("q11", "answer 11"): "unsure",
}


def test_hand_computed_accuracy_precision_recall_kappa_and_exclusions(
    synthetic_hand_store: Path, tmp_path: Path
) -> None:
    sheet_dir = tmp_path / "sheet"
    export_annotation_sheet(synthetic_hand_store, sheet_dir, ["exp-synthetic-hand"])
    filled = _fill_synthetic(
        sheet_dir / SHEET_FILENAME,
        sheet_dir / KEY_FILENAME,
        tmp_path / "SYNTHETIC_labels_A.csv",
        SYNTHETIC_HAND_LABELS,
        "synthetic-annotator-A",
    )
    report = analyze_human_agreement(sheet_dir / KEY_FILENAME, filled, synthetic_hand_store)
    m = report["evaluators"]["exact_match"]["overall"]

    # TP=4, FP=1, FN=2, TN=3 over n=10.
    assert m["table"] == {"tp": 4, "fp": 1, "fn": 2, "tn": 3}
    assert m["n"] == 10 and m["distinct_rows"] == 10
    assert m["accuracy"]["value"] == pytest.approx(0.7)
    assert m["precision"]["value"] == pytest.approx(0.8)
    assert m["recall"]["value"] == pytest.approx(4 / 6)
    assert m["specificity"]["value"] == pytest.approx(0.75)
    # p_o = 0.7; p_e = 0.6*0.5 + 0.4*0.5 = 0.5; kappa = 0.2/0.5 = 0.4.
    assert m["cohens_kappa"] == pytest.approx(0.4)
    lo, hi = m["accuracy"]["ci95"]
    assert lo < 0.7 < hi
    # Each excluded instance is counted once, in its own bucket.
    assert m["excluded"] == {
        "evaluator_status_execution_error": 1,
        "human_unlabelled": 1,
        "human_unsure": 1,
    }
    # The judge passed every case, so precision is defined but specificity is 0/4.
    j = report["evaluators"]["llm_judge"]["overall"]
    assert j["table"] == {"tp": 7, "fp": 4, "fn": 0, "tn": 0}
    assert j["specificity"]["value"] == 0.0
    # Closed-book style run: faithfulness was skipped everywhere -> no faithfulness table.
    assert report["faithfulness_crosstab"] == {}
    assert "exact_match vs human" in render_markdown(report)


def test_sheet_saved_with_byte_order_mark_is_read(
    synthetic_hand_store: Path, tmp_path: Path
) -> None:
    sheet_dir = tmp_path / "sheet"
    export_annotation_sheet(synthetic_hand_store, sheet_dir, ["exp-synthetic-hand"])
    filled = _fill_synthetic(
        sheet_dir / SHEET_FILENAME,
        sheet_dir / KEY_FILENAME,
        tmp_path / "SYNTHETIC_labels_A.csv",
        SYNTHETIC_HAND_LABELS,
        "synthetic-annotator-A",
    )
    # As written by a spreadsheet program's "CSV UTF-8" export.
    filled.write_text(filled.read_text(encoding="utf-8"), encoding="utf-8-sig")
    report = analyze_human_agreement(sheet_dir / KEY_FILENAME, filled, synthetic_hand_store)
    assert report["evaluators"]["exact_match"]["overall"]["n"] == 10


def test_undefined_rates_are_none_not_zero(synthetic_hand_store: Path, tmp_path: Path) -> None:
    sheet_dir = tmp_path / "sheet"
    export_annotation_sheet(synthetic_hand_store, sheet_dir, ["exp-synthetic-hand"])
    only_incorrect = {k: v for k, v in SYNTHETIC_HAND_LABELS.items() if v == "incorrect"}
    filled = _fill_synthetic(
        sheet_dir / SHEET_FILENAME,
        sheet_dir / KEY_FILENAME,
        tmp_path / "SYNTHETIC_labels_A.csv",
        only_incorrect,
        "synthetic-annotator-A",
    )
    m = analyze_human_agreement(sheet_dir / KEY_FILENAME, filled, synthetic_hand_store)[
        "evaluators"
    ]["exact_match"]["overall"]
    assert m["recall"] == {"value": None, "ci95": [None, None], "k": 0, "n": 0}
    # No human-correct instance: recall is undefined (0/0), not 0. Precision is 0/1.
    assert m["precision"]["value"] == 0.0 and m["precision"]["n"] == 1
    # One constant rater: p_o = p_e = 0.75, so kappa is exactly 0 (defined).
    assert m["cohens_kappa"] == pytest.approx(0.0)


def test_inter_annotator_kappa_on_overlap_hand_computed() -> None:
    first = SheetLabels(
        path="SYNTHETIC_A",
        labels=dict(
            zip(
                "abcdef",
                ["correct", "correct", "correct", "incorrect", "incorrect", "unsure"],
                strict=True,
            )
        ),
        annotators={},
    )
    second = SheetLabels(
        path="SYNTHETIC_B",
        labels=dict(
            zip(
                "abcdefg",
                ["correct", "correct", "incorrect", "incorrect", "incorrect", "correct", "correct"],
                strict=True,
            )
        ),
        annotators={},
    )
    inter = inter_annotator_agreement(first, second)
    assert inter["rows_labelled_by_both"] == 6
    b = inter["binary_definite"]
    # Row f excluded (unsure). Table TT=2, TF=1, FT=0, FF=2: p_o=0.8, p_e=0.48.
    assert b["n"] == 5 and b["excluded_unsure"] == 1
    assert b["cohens_kappa"] == pytest.approx(0.32 / 0.52)
    assert b["meets_target"] is True
    t = inter["three_category"]
    # p_o = 4/6; p_e = (3/6)(3/6) + (2/6)(3/6) = 5/12.
    assert t["cohens_kappa"] == pytest.approx((4 / 6 - 5 / 12) / (1 - 5 / 12))
    assert inter["disagreeing_row_ids"] == ["c", "f"]


def test_cohens_kappa_nominal_matches_binary_kappa() -> None:
    pairs = [("y", "y")] * 3 + [("y", "n")] * 2 + [("n", "y")] + [("n", "n")] * 4
    table = {(True, True): 3, (True, False): 2, (False, True): 1, (False, False): 4}
    assert cohens_kappa_nominal(pairs) == pytest.approx(cohens_kappa(table))
    assert cohens_kappa_nominal([]) is None
    assert cohens_kappa_nominal([("y", "y"), ("y", "y")]) is None


def test_two_annotators_unresolved_adjudication_runs_scope_and_faithfulness(
    synthetic_store: Path, tmp_path: Path
) -> None:
    sheet_dir = tmp_path / "sheet"
    export_annotation_sheet(synthetic_store, sheet_dir, SYNTHETIC_EXPERIMENTS)
    key = sheet_dir / KEY_FILENAME
    labels_a = SYNTHETIC_LABELS_A
    labels_b = {**labels_a, ("q1", "Paris"): "unsure"}  # disagrees on q1 only
    a = _fill_synthetic(
        sheet_dir / SHEET_FILENAME, key, tmp_path / "SYNTHETIC_A.csv", labels_a, "synthetic-A"
    )
    b = _fill_synthetic(
        sheet_dir / SHEET_FILENAME, key, tmp_path / "SYNTHETIC_B.csv", labels_b, "synthetic-B"
    )

    report = analyze_human_agreement(key, a, synthetic_store, second_sheet_path=b)
    assert report["rows"]["reference"]["unresolved_disagreement"] == 1
    assert report["inter_annotator"]["rows_labelled_by_both"] == 5
    em = report["evaluators"]["exact_match"]
    # runs="first": syn-run-c1 and syn-run-r1 only -> 6 instances; q1 is unresolved in both.
    assert em["overall"]["instances"] == 6
    assert em["overall"]["excluded"] == {"human_unresolved_disagreement": 2}
    assert set(em["by_experiment"]) == {"exp-synthetic-closed", "exp-synthetic-rag"}

    # Faithfulness: only the RAG arm, descriptive, with the validation error counted.
    faith = report["faithfulness_crosstab"]
    assert set(faith) == {"exp-synthetic-rag"}
    assert faith["exp-synthetic-rag"]["human_incorrect"] == {"supported": 1}
    assert faith["exp-synthetic-rag"]["excluded"] == {
        "evaluator_status_output_validation_error": 1,
        "human_unresolved_disagreement": 1,
    }
    assert "llm_faithfulness_judge" not in report["evaluators"]

    # An adjudicated label resolves the disagreement; --runs all adds the repeat run.
    q1_row = next(r["row_id"] for r in _read(key) if r["test_case_id"] == "q1")
    adjudication = tmp_path / "SYNTHETIC_adjudication.csv"
    adjudication.write_text(f"row_id,human_correct,notes\n{q1_row},correct,synthetic\n")
    report = analyze_human_agreement(
        key, a, synthetic_store, second_sheet_path=b, adjudication_path=adjudication, runs="all"
    )
    em_all = report["evaluators"]["exact_match"]["overall"]
    assert report["rows"]["reference_sources"]["adjudicated"] == 1
    assert em_all["instances"] == 9 and em_all["excluded"] == {}
    # Same 5 distinct rows behind 9 instances.
    assert em_all["distinct_rows"] == 5
    # Interannotator kappa is unchanged by adjudication.
    assert report["inter_annotator"]["disagreeing_row_ids"] == [q1_row]


def test_store_that_does_not_match_the_key_is_rejected(
    synthetic_store: Path, tmp_path: Path
) -> None:
    sheet_dir = tmp_path / "sheet"
    export_annotation_sheet(synthetic_store, sheet_dir, SYNTHETIC_EXPERIMENTS)
    rows = _read(sheet_dir / KEY_FILENAME)
    rows[0]["normalized_answer"] = "tampered"
    with (sheet_dir / KEY_FILENAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    filled = _fill_synthetic(
        sheet_dir / SHEET_FILENAME,
        sheet_dir / KEY_FILENAME,
        tmp_path / "SYNTHETIC_A.csv",
        SYNTHETIC_LABELS_A,
        "synthetic-A",
    )
    with pytest.raises(AnnotationFormatError, match="does not match the key"):
        analyze_human_agreement(sheet_dir / KEY_FILENAME, filled, synthetic_store, runs="all")


# --- committed EXP-001 files


def test_committed_exp001_sheet_is_blank_and_matches_key() -> None:
    sheet = _read(EXP001_DIR / SHEET_FILENAME)
    overlap_sheet = _read(EXP001_DIR / OVERLAP_SHEET_FILENAME)
    key = _read(EXP001_DIR / KEY_FILENAME)
    assert list(sheet[0]) == SHEET_COLUMNS
    for row in sheet + overlap_sheet:
        assert row["human_correct"] == row["annotator"] == row["notes"] == ""
    assert [r["row_id"] for r in sheet] == sorted({k["row_id"] for k in key})
    overlap = {k["row_id"] for k in key if k["overlap"] == "1"}
    assert {r["row_id"] for r in overlap_sheet} == overlap
    assert len(overlap) >= 0.2 * len(sheet)
    assert len(key) == len({(k["run_id"], k["test_case_id"]) for k in key})


def test_guideline_examples_are_verbatim_exp001_answers() -> None:
    text = GUIDELINE.read_text(encoding="utf-8")
    examples = re.findall(r"- Question: (.+)\n- Gold answers: (.+)\n- Model answer: (.+)\n", text)
    assert len(examples) >= 10
    sheet = {
        (r["question"], r["gold_answers"], r["model_answer"])
        for r in _read(EXP001_DIR / SHEET_FILENAME)
    }
    for example in examples:
        assert example in sheet, example

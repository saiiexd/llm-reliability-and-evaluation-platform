"""
Blind human-annotation sheets for answer correctness.

Builds the material for validating automatic evaluators against human
judgement (RQ3 evaluator reliability):

- a **sheet** that annotators fill in: one row per *distinct* answer to a
  question, showing only the question, all gold answers and the model answer.
  It carries no model, arm, run or evaluator information, and rows are
  shuffled with a fixed seed so that their order reveals nothing either;
- a **key** that maps every sheet row back to every evaluated instance
  ``(experiment_id, run_id, test_case_id)`` that produced that answer. The
  key is for the analysis, never for annotators;
- an **overlap** subset (at least 20% of rows, chosen at random with the same
  seed) that a second annotator labels independently, for inter-annotator
  agreement. The overlap flag is recorded in the key and the overlap rows are
  written to a separate sheet for the second annotator; the main sheet does not
  say which rows are double-labelled.

Deduplication key: the question (test case id, question text and gold answers)
plus :func:`annotation_normalize` of the answer. Answers that differ only in
case, punctuation, articles or whitespace are labelled once, because a human
judges them identically; numbers are kept verbatim in the key so "3.5" and
"35" are never merged (SQuAD normalization alone would merge them).

No labels are produced here: ``human_correct`` is always written empty.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_reliability.evaluation.squad_match import gold_answers_for, squad_normalize
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.models import ExperimentRun

SHEET_COLUMNS = [
    "row_id",
    "question",
    "gold_answers",
    "model_answer",
    "human_correct",
    "annotator",
    "notes",
]
KEY_COLUMNS = [
    "row_id",
    "overlap",
    "experiment_id",
    "run_id",
    "repetition_index",
    "test_case_id",
    "normalized_answer",
    "raw_answer",
]
LABELS = ("correct", "incorrect", "unsure")
GOLD_SEPARATOR = " | "
DEFAULT_SEED = 2026
DEFAULT_OVERLAP_FRACTION = 0.25
MIN_OVERLAP_FRACTION = 0.2

SHEET_FILENAME = "annotation_sheet.csv"
OVERLAP_SHEET_FILENAME = "annotation_sheet_overlap.csv"
KEY_FILENAME = "annotation_key.csv"
MANIFEST_FILENAME = "export_manifest.json"

_NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*")


def annotation_normalize(text: str) -> str:
    """Deduplication form of an answer: SQuAD normalization plus its numbers verbatim.

    ``squad_normalize`` lowercases, strips punctuation and the articles
    a/an/the, and collapses whitespace. Stripping punctuation would turn
    "3.5" into "35", so every number in the raw text (with its internal
    ``.``, ``,``, ``:`` or ``/``) is appended after ``" #"`` to keep such
    answers apart.
    """
    base = squad_normalize(text)
    numbers = _NUMBER.findall(text)
    return f"{base} # {' '.join(numbers)}" if numbers else base


def default_experiment_ids(
    repo: LocalFileExperimentRepository, prefix: str = "exp001-"
) -> list[str]:
    """All experiment ids with ``prefix``, excluding pilot experiments (``*-pilotN``)."""
    return sorted(
        e for e in repo.list_experiment_ids() if e.startswith(prefix) and "-pilot" not in e
    )


@dataclass(frozen=True)
class AnnotationInstance:
    """One evaluated answer: a test case in one run."""

    experiment_id: str
    run_id: str
    repetition_index: int
    test_case_id: str
    raw_answer: str


@dataclass
class AnnotationItem:
    """One sheet row: a distinct (question, normalized answer) pair."""

    test_case_id: str
    question: str
    gold_answers: list[str]
    normalized_answer: str
    instances: list[AnnotationInstance] = field(default_factory=list)
    row_id: str = ""
    overlap: bool = False

    @property
    def display_answer(self) -> str:
        """The most frequent raw variant (whitespace-stripped); ties -> lexicographically first."""
        counts = Counter(i.raw_answer.strip() for i in self.instances)
        return min(counts, key=lambda text: (-counts[text], text))

    def sort_key(self) -> tuple[str, str, str, tuple[str, ...]]:
        return (self.test_case_id, self.normalized_answer, self.question, tuple(self.gold_answers))


@dataclass
class CollectionStats:
    experiment_ids: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    runs_by_experiment: dict[str, list[str]] = field(default_factory=dict)
    runs_without_result: list[str] = field(default_factory=list)
    cases_without_response: int = 0
    instances: int = 0


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def collect_items(
    repo: LocalFileExperimentRepository, experiment_ids: list[str]
) -> tuple[list[AnnotationItem], CollectionStats]:
    """Load every run of the experiments and group their answers into distinct items.

    Runs without a result and test cases without a model response (execution
    errors) have nothing to label; they are skipped and counted in the stats.
    Raises ``ValueError`` when a requested experiment has no runs.
    """
    stats = CollectionStats(experiment_ids=list(experiment_ids))
    items: dict[tuple[str, str, tuple[str, ...], str], AnnotationItem] = {}
    for experiment_id in experiment_ids:
        run_ids = repo.list_run_ids(experiment_id)
        if not run_ids:
            raise ValueError(f"No runs found for experiment '{experiment_id}'.")
        runs: list[ExperimentRun] = [repo.load_run(r) for r in run_ids]
        for run in sorted(runs, key=lambda r: (r.repetition_index, r.run_id)):
            stats.run_ids.append(run.run_id)
            stats.runs_by_experiment.setdefault(experiment_id, []).append(run.run_id)
            if run.result is None:
                stats.runs_without_result.append(run.run_id)
                continue
            cases = {tc.id: tc for tc in run.dataset_version.dataset.test_cases}
            for result in run.result.test_case_results:
                if result.model_response is None:
                    stats.cases_without_response += 1
                    continue
                tc = cases[result.test_case_id]
                gold = _unique(gold_answers_for(tc))
                raw = result.model_response.output_text
                key = (tc.id, tc.input, tuple(gold), annotation_normalize(raw))
                item = items.get(key)
                if item is None:
                    item = AnnotationItem(
                        test_case_id=tc.id,
                        question=tc.input,
                        gold_answers=gold,
                        normalized_answer=key[3],
                    )
                    items[key] = item
                item.instances.append(
                    AnnotationInstance(
                        experiment_id=run.experiment_id,
                        run_id=run.run_id,
                        repetition_index=run.repetition_index,
                        test_case_id=tc.id,
                        raw_answer=raw,
                    )
                )
                stats.instances += 1
    return list(items.values()), stats


def assign_rows(
    items: list[AnnotationItem],
    seed: int = DEFAULT_SEED,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
) -> list[AnnotationItem]:
    """Sort, shuffle with ``seed``, assign row ids and choose the overlap rows.

    Sorting first makes the result independent of file-listing order. The
    overlap is ``ceil(overlap_fraction * n)`` rows sampled with the same
    random generator, so it is spread over the sheet rather than at the top.
    """
    if not MIN_OVERLAP_FRACTION <= overlap_fraction <= 1.0:
        raise ValueError(
            f"overlap_fraction must be between {MIN_OVERLAP_FRACTION} and 1.0, "
            f"got {overlap_fraction}."
        )
    ordered = sorted(items, key=AnnotationItem.sort_key)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    width = max(3, len(str(len(ordered))))
    for index, item in enumerate(ordered, start=1):
        item.row_id = f"R{index:0{width}d}"
        item.overlap = False
    n_overlap = math.ceil(overlap_fraction * len(ordered))
    for index in rng.sample(range(len(ordered)), n_overlap):
        ordered[index].overlap = True
    return ordered


def _sheet_row(item: AnnotationItem) -> dict[str, str]:
    return {
        "row_id": item.row_id,
        "question": item.question,
        "gold_answers": GOLD_SEPARATOR.join(item.gold_answers),
        "model_answer": item.display_answer,
        "human_correct": "",
        "annotator": "",
        "notes": "",
    }


def write_sheet(path: Path, items: list[AnnotationItem]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for item in items:
            writer.writerow(_sheet_row(item))


def write_key(path: Path, items: list[AnnotationItem]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for item in items:
            for inst in sorted(
                item.instances, key=lambda i: (i.experiment_id, i.repetition_index, i.run_id)
            ):
                writer.writerow(
                    {
                        "row_id": item.row_id,
                        "overlap": "1" if item.overlap else "0",
                        "experiment_id": inst.experiment_id,
                        "run_id": inst.run_id,
                        "repetition_index": str(inst.repetition_index),
                        "test_case_id": inst.test_case_id,
                        "normalized_answer": item.normalized_answer,
                        "raw_answer": inst.raw_answer,
                    }
                )


def export_annotation_sheet(
    store: Path,
    out_dir: Path,
    experiment_ids: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
) -> dict[str, Any]:
    """Write the blind sheet, the overlap sheet, the key and a manifest; return the manifest."""
    repo = LocalFileExperimentRepository(store)
    selected = experiment_ids if experiment_ids else default_experiment_ids(repo)
    if not selected:
        raise ValueError(f"No experiments selected in store '{store}'.")
    items, stats = collect_items(repo, selected)
    if not items:
        raise ValueError("The selected runs contain no model answers to annotate.")
    rows = assign_rows(items, seed=seed, overlap_fraction=overlap_fraction)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_sheet(out_dir / SHEET_FILENAME, rows)
    write_sheet(out_dir / OVERLAP_SHEET_FILENAME, [r for r in rows if r.overlap])
    write_key(out_dir / KEY_FILENAME, rows)

    per_experiment: dict[str, dict[str, int]] = {}
    for experiment_id in selected:
        exp = [(r.row_id, i) for r in rows for i in r.instances if i.experiment_id == experiment_id]
        exp_rows = {row_id for row_id, _ in exp}
        exp_instances = len(exp)
        per_experiment[experiment_id] = {
            "runs": len(stats.runs_by_experiment.get(experiment_id, [])),
            "instances": exp_instances,
            "distinct_rows": len(exp_rows),
        }
    manifest: dict[str, Any] = {
        "seed": seed,
        "overlap_fraction": overlap_fraction,
        "experiment_ids": selected,
        "run_ids": stats.run_ids,
        "runs_without_result": stats.runs_without_result,
        "cases_without_response": stats.cases_without_response,
        "instances": stats.instances,
        "rows": len(rows),
        "overlap_rows": sum(1 for r in rows if r.overlap),
        "distinct_questions": len({r.test_case_id for r in rows}),
        "rows_by_instance_count": {
            str(k): v for k, v in sorted(Counter(len(r.instances) for r in rows).items())
        },
        "per_experiment": per_experiment,
        "files": {
            "sheet": SHEET_FILENAME,
            "overlap_sheet": OVERLAP_SHEET_FILENAME,
            "key": KEY_FILENAME,
        },
    }
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest

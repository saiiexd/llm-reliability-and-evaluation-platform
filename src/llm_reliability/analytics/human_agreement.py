"""
Agreement between automatic evaluators and human correctness labels.

Reads filled annotation sheets (see :mod:`llm_reliability.analytics.annotation`),
the key and the run store, and reports:

1. **Inter-annotator agreement** on rows labelled by both annotators, computed
   on the original, independent labels (before any adjudication), excluding
   guideline-example rows.
2. **Evaluator vs human agreement** for each correctness evaluator, overall and
   per experiment (arm). Primary result: guideline-example rows excluded.
   Sensitivity result (``evaluators_including_guideline_examples``): the same
   with them included.
3. A descriptive cross-tabulation of the faithfulness judge against human
   correctness for RAG arms. Faithfulness ("is the answer supported by the
   retrieved context?") is a different criterion from correctness, so no
   accuracy or kappa is reported for it; see :data:`FAITHFULNESS_NOTE`.

Guideline-example rows (``guideline_example = 1`` in the key) show answers for
which the annotation guideline itself states a verdict, written by the
guideline author and seen by every annotator. Their labels are therefore not
independent human evidence: they are excluded from inter-annotator kappa (and
counted), from the primary evaluator agreement, and from the llm_judge and
faithfulness cross-tabulations. They still receive a reference label, which is
used only in the sensitivity result.

Definitions (positive class = "correct"):

- Unit of analysis: one evaluated instance ``(run_id, test_case_id)``. The
  human label of a sheet row applies to every instance the key maps to it.
- Reference label of a row: the single annotator's label; with two annotators,
  their common label when they agree, the adjudicated label when they disagree
  and an adjudication file provides one, otherwise *unresolved* (excluded).
- TP = human correct and evaluator passed; FP = human incorrect and evaluator
  passed; FN = human correct and evaluator failed; TN = both negative.
- accuracy = (TP + TN) / n; precision = TP / (TP + FP); recall = TP / (TP + FN);
  specificity = TN / (TN + FP). Each has a 95% Wilson interval over its own
  denominator. A zero denominator gives ``None`` (undefined), never 0.
- Cohen's kappa between human and evaluator on the same n; no interval.
- Exclusions, each instance in exactly one bucket, checked in this order:
  guideline-example row (primary result only), row not labelled, human
  ``unsure``, unresolved annotator disagreement, evaluator not run on the
  instance, evaluator status other than SUCCESS (by status), SUCCESS with
  ``passed = None``.

Instances that repeat the same answer are not independent (the same text is
judged by deterministic evaluators the same way), so intervals computed over
instances are too narrow when many instances share a row. Each group therefore
reports the number of distinct rows behind its n, and by default only the first
run of each experiment is used (``runs="first"``), as in the EXP-001 tables.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_reliability.analytics.agreement import cohens_kappa, cohens_kappa_nominal
from llm_reliability.analytics.annotation import LABELS, annotation_normalize
from llm_reliability.analytics.proportions import wilson
from llm_reliability.evaluation.models import EvaluationResult, EvaluationStatus, TestCaseResult
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.models import ExperimentRun
from llm_reliability.experiments.repository import RunNotFoundError

CORRECTNESS_EVALUATORS = ["exact_match", "squad_match", "llm_judge"]
FAITHFULNESS_EVALUATOR = "llm_faithfulness_judge"
KAPPA_TARGET = 0.6
FAITHFULNESS_NOTE = (
    "The faithfulness judge measures whether the answer is supported by the retrieved "
    "context, not whether it is correct. A correct answer can be unsupported (the model "
    "answered from memory) and an incorrect one supported (the context itself misleads, or "
    "an abstention is labelled 'supported'). Human correctness labels therefore cannot "
    "validate it, and no accuracy or kappa is reported; this table only describes how its "
    "labels fall across human-correct and human-incorrect answers. Validating it needs a "
    "separate annotation of support against the retrieved passages."
)


class AnnotationFormatError(ValueError):
    """A sheet, adjudication file or key is malformed (unknown rows, bad labels, encoding...)."""


class NoLabelsError(ValueError):
    """No human label is filled in any sheet; there is nothing to analyse."""


@dataclass(frozen=True)
class KeyEntry:
    row_id: str
    overlap: bool
    guideline_example: bool
    experiment_id: str
    run_id: str
    repetition_index: int
    test_case_id: str
    normalized_answer: str


@dataclass
class SheetLabels:
    path: str
    labels: dict[str, str]  # row_id -> label, only for filled rows
    annotators: dict[str, str]  # row_id -> annotator name, only for filled rows


def _read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    # utf-8-sig: spreadsheet programs often save "CSV UTF-8" with a byte-order mark.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        try:
            # Decoding is lazy: both the header and the rows can fail.
            fieldnames = reader.fieldnames or []
            missing = [c for c in required if c not in fieldnames]
            rows = None if missing else list(reader)
        except UnicodeDecodeError as exc:
            raise AnnotationFormatError(
                f"{path}: not valid UTF-8 ({exc.reason}). It was probably saved "
                "as 'CSV (Comma delimited)', which uses a Windows code page. Save it again as "
                "'CSV UTF-8 (Comma delimited)' (Excel) or 'Text CSV' with character set "
                "'Unicode (UTF-8)' (LibreOffice) and re-run."
            ) from exc
        if rows is None:
            raise AnnotationFormatError(f"{path}: missing column(s) {', '.join(missing)}.")
        return rows


def read_key(path: Path) -> list[KeyEntry]:
    rows = _read_csv(
        path,
        [
            "row_id",
            "overlap",
            "guideline_example",
            "experiment_id",
            "run_id",
            "repetition_index",
            "test_case_id",
            "normalized_answer",
        ],
    )
    if not rows:
        raise AnnotationFormatError(f"{path}: the key has no rows.")
    return [
        KeyEntry(
            row_id=r["row_id"],
            overlap=r["overlap"] == "1",
            guideline_example=r["guideline_example"] == "1",
            experiment_id=r["experiment_id"],
            run_id=r["run_id"],
            repetition_index=int(r["repetition_index"]),
            test_case_id=r["test_case_id"],
            normalized_answer=r["normalized_answer"],
        )
        for r in rows
    ]


def _parse_label(raw: str | None) -> str | None:
    value = (raw or "").strip().lower()
    return value or None


def read_labels(path: Path, valid_row_ids: set[str], require_label: bool = False) -> SheetLabels:
    """Read a filled sheet (or an adjudication file) into ``SheetLabels``.

    Labels are case- and whitespace-insensitive and must be one of
    :data:`LABELS`; an empty cell means "not labelled". Unknown or duplicate
    row ids and unrecognised labels are errors (listed together), never
    silently treated as missing. With ``require_label`` every row must be filled.
    """
    rows = _read_csv(path, ["row_id", "human_correct"])
    problems: list[str] = []
    seen: set[str] = set()
    labels: dict[str, str] = {}
    annotators: dict[str, str] = {}
    for line, row in enumerate(rows, start=2):
        row_id = (row.get("row_id") or "").strip()
        if row_id not in valid_row_ids:
            problems.append(f"line {line}: row_id {row_id!r} is not in the key")
            continue
        if row_id in seen:
            problems.append(f"line {line}: duplicate row_id {row_id!r}")
            continue
        seen.add(row_id)
        label = _parse_label(row.get("human_correct"))
        if label is None:
            if require_label:
                problems.append(f"line {line}: row {row_id} has no label")
            continue
        if label not in LABELS:
            problems.append(
                f"line {line}: row {row_id} has label {row.get('human_correct')!r}; "
                f"expected one of {', '.join(LABELS)} (or empty)"
            )
            continue
        labels[row_id] = label
        annotators[row_id] = (row.get("annotator") or "").strip()
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise AnnotationFormatError(f"{path}: {len(problems)} problem(s):\n  {shown}{more}")
    return SheetLabels(path=str(path), labels=labels, annotators=annotators)


def _sheet_summary(sheet: SheetLabels, n_rows: int, overlap_rows: set[str]) -> dict[str, Any]:
    names = sorted(set(sheet.annotators.values()) - {""})
    return {
        "path": sheet.path,
        "rows_labelled": len(sheet.labels),
        "rows_total": n_rows,
        "overlap_rows_labelled": len(set(sheet.labels) & overlap_rows),
        "label_counts": {label: Counter(sheet.labels.values()).get(label, 0) for label in LABELS},
        "annotators": names,
        "labelled_rows_without_annotator": sum(1 for v in sheet.annotators.values() if not v),
    }


def inter_annotator_agreement(
    first: SheetLabels, second: SheetLabels, excluded_rows: Iterable[str] = ()
) -> dict[str, Any]:
    """Agreement on rows both annotators labelled, from their independent labels.

    ``excluded_rows`` (the guideline-example rows) are left out of every measure
    and counted in ``excluded_guideline_examples``. Primary measure (target
    kappa >= 0.6): binary Cohen's kappa on the remaining rows where both gave a
    definite label (correct/incorrect); rows where either said ``unsure`` are
    excluded from it and counted. Secondary: raw agreement and kappa over the
    three categories, with ``unsure`` as a category of its own.

    ``disagreeing_row_ids`` lists every row both labelled differently,
    guideline examples included: it is the adjudication worklist, and an
    unadjudicated disagreement leaves a row without a reference label.
    """
    excluded = set(excluded_rows)
    all_both = sorted(set(first.labels) & set(second.labels))
    both = [r for r in all_both if r not in excluded]
    pairs = [(first.labels[r], second.labels[r]) for r in both]
    definite = [(a, b) for a, b in pairs if a != "unsure" and b != "unsure"]
    table = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for a, b in definite:
        table[(a == "correct", b == "correct")] += 1
    binary_kappa = cohens_kappa(table)
    return {
        "rows_labelled_by_both": len(all_both),
        "excluded_guideline_examples": len(all_both) - len(both),
        "rows_compared": len(both),
        "confusion_first_by_second": {
            a: {b: sum(1 for x, y in pairs if x == a and y == b) for b in LABELS} for a in LABELS
        },
        "three_category": {
            "n": len(pairs),
            "raw_agreement": (sum(1 for a, b in pairs if a == b) / len(pairs)) if pairs else None,
            "cohens_kappa": cohens_kappa_nominal(pairs),
        },
        "binary_definite": {
            "n": len(definite),
            "excluded_unsure": len(pairs) - len(definite),
            "raw_agreement": (sum(1 for a, b in definite if a == b) / len(definite))
            if definite
            else None,
            "cohens_kappa": binary_kappa,
            "kappa_target": KAPPA_TARGET,
            "meets_target": None if binary_kappa is None else binary_kappa >= KAPPA_TARGET,
        },
        "disagreeing_row_ids": [r for r in all_both if first.labels[r] != second.labels[r]],
    }


def reference_labels(
    first: SheetLabels | None, second: SheetLabels | None, adjudication: SheetLabels | None
) -> tuple[dict[str, str], dict[str, int], list[str]]:
    """Combine the annotators' labels into one reference label per row.

    Returns ``(labels, counts, unresolved_row_ids)``. ``labels`` maps row_id to
    correct/incorrect/unsure for every resolved row. Adjudicated labels are
    used only for rows on which the two annotators disagree.
    """
    a = first.labels if first else {}
    b = second.labels if second else {}
    adj = adjudication.labels if adjudication else {}
    labels: dict[str, str] = {}
    unresolved: list[str] = []
    counts = Counter[str]()
    for row_id in sorted(set(a) | set(b)):
        la, lb = a.get(row_id), b.get(row_id)
        if la is not None and lb is not None and la != lb:
            if row_id in adj:
                labels[row_id] = adj[row_id]
                counts["adjudicated"] += 1
            else:
                unresolved.append(row_id)
                counts["unresolved_disagreement"] += 1
            continue
        resolved = la if la is not None else lb
        assert resolved is not None  # row_id comes from the union of labelled rows
        labels[row_id] = resolved
        counts["agreed" if la is not None and lb is not None else "single_annotator"] += 1
    counts["adjudication_entries_ignored"] = sum(
        1 for r in adj if not (r in a and r in b and a[r] != b[r])
    )
    return labels, dict(counts), unresolved


def _find_result(run: ExperimentRun, test_case_id: str) -> TestCaseResult | None:
    if run.result is None:
        return None
    return next((r for r in run.result.test_case_results if r.test_case_id == test_case_id), None)


def _evaluation(result: TestCaseResult, evaluator: str) -> EvaluationResult | None:
    return next((e for e in result.evaluation_results if e.evaluator_name == evaluator), None)


@dataclass(frozen=True)
class Instance:
    entry: KeyEntry
    result: TestCaseResult


def load_instances(
    key: list[KeyEntry], repo: LocalFileExperimentRepository, runs: str = "first"
) -> list[Instance]:
    """Resolve key entries against the store, checking that answers still match the key.

    ``runs="first"`` keeps, per experiment, only the run with the lowest
    ``(repetition_index, run_id)``; ``runs="all"`` keeps every run.
    """
    if runs not in ("first", "all"):
        raise ValueError(f"runs must be 'first' or 'all', got {runs!r}")
    entries = key
    if runs == "first":
        first_run: dict[str, tuple[int, str]] = {}
        for e in key:
            current = first_run.get(e.experiment_id)
            candidate = (e.repetition_index, e.run_id)
            if current is None or candidate < current:
                first_run[e.experiment_id] = candidate
        entries = [e for e in key if (e.repetition_index, e.run_id) == first_run[e.experiment_id]]
    cache: dict[str, ExperimentRun] = {}
    problems: list[str] = []
    instances: list[Instance] = []
    for e in entries:
        if e.run_id not in cache:
            try:
                cache[e.run_id] = repo.load_run(e.run_id)
            except RunNotFoundError:
                problems.append(f"run {e.run_id} ({e.experiment_id}) is not in the store")
                continue
        run = cache[e.run_id]
        result = _find_result(run, e.test_case_id)
        if run.experiment_id != e.experiment_id or result is None or result.model_response is None:
            problems.append(f"run {e.run_id}: no answer for {e.test_case_id} ({e.experiment_id})")
            continue
        if annotation_normalize(result.model_response.output_text) != e.normalized_answer:
            problems.append(
                f"run {e.run_id} / {e.test_case_id}: the stored answer does not match the key"
            )
            continue
        instances.append(Instance(entry=e, result=result))
    if problems:
        shown = "\n  ".join(sorted(set(problems))[:20])
        raise AnnotationFormatError(
            f"The store does not match the key ({len(problems)} problem(s)):\n  {shown}"
        )
    return instances


def _rate(k: int, n: int) -> dict[str, Any]:
    p, lo, hi = wilson(k, n)
    return {"value": p, "ci95": [lo, hi], "k": k, "n": n}


def evaluator_vs_human(
    instances: list[Instance],
    labels: dict[str, str],
    unresolved: set[str],
    evaluator: str,
    excluded_rows: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Confusion table, rates with Wilson intervals, kappa and exclusions for one group.

    Instances of ``excluded_rows`` (guideline examples, in the primary result)
    are counted under ``guideline_example`` and not used.
    """
    excluded = Counter[str]()
    table = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    rows: set[str] = set()
    for inst in instances:
        row_id = inst.entry.row_id
        if row_id in excluded_rows:
            excluded["guideline_example"] += 1
            continue
        if row_id in unresolved:
            excluded["human_unresolved_disagreement"] += 1
            continue
        label = labels.get(row_id)
        if label is None:
            excluded["human_unlabelled"] += 1
            continue
        if label == "unsure":
            excluded["human_unsure"] += 1
            continue
        ev = _evaluation(inst.result, evaluator)
        if ev is None:
            excluded["evaluator_missing"] += 1
            continue
        if ev.status != EvaluationStatus.SUCCESS:
            excluded[f"evaluator_status_{ev.status.value}"] += 1
            continue
        if ev.passed is None:
            excluded["evaluator_passed_none"] += 1
            continue
        table[(label == "correct", bool(ev.passed))] += 1
        rows.add(row_id)
    tp, fn = table[(True, True)], table[(True, False)]
    fp, tn = table[(False, True)], table[(False, False)]
    n = tp + fn + fp + tn
    return {
        "instances": len(instances),
        "n": n,
        "distinct_rows": len(rows),
        "excluded": dict(sorted(excluded.items())),
        "table": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": _rate(tp + tn, n),
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "specificity": _rate(tn, tn + fp),
        "human_correct_rate": _rate(tp + fn, n),
        "evaluator_pass_rate": _rate(tp + fp, n),
        "cohens_kappa": cohens_kappa(table),
    }


def _label_crosstab(
    instances: list[Instance],
    labels: dict[str, str],
    unresolved: set[str],
    evaluator: str,
    excluded_rows: set[str],
) -> dict[str, Any]:
    """Human reference label (correct/incorrect) x evaluator label, with exclusions."""
    cells: dict[str, Counter[str]] = {"correct": Counter(), "incorrect": Counter()}
    excluded = Counter[str]()
    for inst in instances:
        label = labels.get(inst.entry.row_id)
        if inst.entry.row_id in excluded_rows:
            excluded["guideline_example"] += 1
        elif inst.entry.row_id in unresolved:
            excluded["human_unresolved_disagreement"] += 1
        elif label is None:
            excluded["human_unlabelled"] += 1
        elif label == "unsure":
            excluded["human_unsure"] += 1
        else:
            ev = _evaluation(inst.result, evaluator)
            if ev is None:
                excluded["evaluator_missing"] += 1
            elif ev.status != EvaluationStatus.SUCCESS:
                excluded[f"evaluator_status_{ev.status.value}"] += 1
            else:
                cells[label][ev.label or "no_label"] += 1
    return {
        "human_correct": dict(sorted(cells["correct"].items())),
        "human_incorrect": dict(sorted(cells["incorrect"].items())),
        "excluded": dict(sorted(excluded.items())),
    }


def analyze_human_agreement(
    key_path: Path,
    sheet_path: Path,
    store: Path,
    second_sheet_path: Path | None = None,
    adjudication_path: Path | None = None,
    runs: str = "first",
) -> dict[str, Any]:
    """Build the full report. Raises ``NoLabelsError`` when no sheet has any label."""
    key = read_key(key_path)
    row_ids = {e.row_id for e in key}
    overlap_rows = {e.row_id for e in key if e.overlap}
    example_rows = {e.row_id for e in key if e.guideline_example}
    first = read_labels(sheet_path, row_ids)
    second = read_labels(second_sheet_path, row_ids) if second_sheet_path else None
    if not first.labels and not (second and second.labels):
        raise NoLabelsError(
            "No human labels found: every 'human_correct' cell is empty in "
            + " and ".join(str(p) for p in (sheet_path, second_sheet_path) if p)
            + ". Fill the sheet following research/annotation/guideline.md, then re-run. "
            "Nothing was written."
        )
    adjudication = (
        read_labels(adjudication_path, row_ids, require_label=True) if adjudication_path else None
    )
    labels, reference_counts, unresolved_list = reference_labels(first, second, adjudication)
    unresolved = set(unresolved_list)
    instances = load_instances(key, LocalFileExperimentRepository(store), runs=runs)

    warnings: list[str] = []
    sheets = [_sheet_summary(first, len(row_ids), overlap_rows)]
    inter: dict[str, Any] | None = None
    if second is not None:
        sheets.append(_sheet_summary(second, len(row_ids), overlap_rows))
        inter = inter_annotator_agreement(first, second, excluded_rows=example_rows)
        if inter["rows_compared"] == 0:
            warnings.append(
                "The two sheets share no labelled row outside the guideline examples; "
                "inter-annotator agreement is undefined."
            )
        if set(sheets[0]["annotators"]) & set(sheets[1]["annotators"]):
            warnings.append("The same annotator name appears in both sheets.")
        if inter["rows_compared"] < 0.2 * len(row_ids):
            warnings.append(
                f"Only {inter['rows_compared']} of {len(row_ids)} rows are double-labelled outside "
                "the guideline examples (the guideline requires at least 20%)."
            )
    else:
        warnings.append("Only one sheet given: inter-annotator agreement cannot be computed.")
    if sheets[0]["rows_labelled"] < len(row_ids):
        warnings.append(
            f"{sheets[0]['path']}: {sheets[0]['rows_labelled']} of {len(row_ids)} rows labelled; "
            "rows nobody labelled are excluded and counted."
        )
    for s in sheets:
        if s["labelled_rows_without_annotator"]:
            warnings.append(
                f"{s['path']}: {s['labelled_rows_without_annotator']} labelled row(s) have no "
                "annotator name."
            )
    if reference_counts.get("adjudication_entries_ignored"):
        warnings.append(
            f"{reference_counts['adjudication_entries_ignored']} adjudication entr(y/ies) ignored: "
            "adjudication applies only to rows on which the two annotators disagree."
        )

    experiments = sorted({i.entry.experiment_id for i in instances})
    by_exp = {x: [i for i in instances if i.entry.experiment_id == x] for x in experiments}
    present = {e.evaluator_name for i in instances for e in i.result.evaluation_results}
    evaluators: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    for name in CORRECTNESS_EVALUATORS:
        if name not in present:
            continue
        for target, excluded_rows in ((evaluators, example_rows), (sensitivity, set())):
            target[name] = {
                "overall": evaluator_vs_human(instances, labels, unresolved, name, excluded_rows),
                "by_experiment": {
                    x: evaluator_vs_human(by_exp[x], labels, unresolved, name, excluded_rows)
                    for x in experiments
                },
            }
    judge_labels = (
        {
            x: _label_crosstab(by_exp[x], labels, unresolved, "llm_judge", example_rows)
            for x in experiments
        }
        if "llm_judge" in present
        else {}
    )
    faithfulness = {
        x: _label_crosstab(by_exp[x], labels, unresolved, FAITHFULNESS_EVALUATOR, example_rows)
        for x in experiments
        if any(
            (ev := _evaluation(i.result, FAITHFULNESS_EVALUATOR)) is not None
            and ev.status == EvaluationStatus.SUCCESS
            for i in by_exp[x]
        )
    }

    reference_label_counts = Counter(labels.values())
    return {
        "inputs": {
            "key": str(key_path),
            "sheet": str(sheet_path),
            "second_sheet": str(second_sheet_path) if second_sheet_path else None,
            "adjudication": str(adjudication_path) if adjudication_path else None,
            "store": str(store),
            "runs": runs,
        },
        "rows": {
            "total": len(row_ids),
            "overlap": len(overlap_rows),
            "guideline_examples": len(example_rows),
            "reference": {
                **{label: reference_label_counts.get(label, 0) for label in LABELS},
                "unresolved_disagreement": len(unresolved),
                "unlabelled": len(row_ids) - len(labels) - len(unresolved),
            },
            "reference_sources": reference_counts,
        },
        "instances_in_scope": len(instances),
        "sheets": sheets,
        "inter_annotator": inter,
        "evaluators": evaluators,
        "evaluators_including_guideline_examples": sensitivity,
        "llm_judge_labels": judge_labels,
        "faithfulness_crosstab": faithfulness,
        "faithfulness_note": FAITHFULNESS_NOTE,
        "warnings": warnings,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _rate_cell(rate: dict[str, Any]) -> str:
    if rate["value"] is None:
        return f"undefined (0/{rate['n']})"
    lo, hi = rate["ci95"]
    return f"{_pct(rate['value'])} [{_pct(lo)}, {_pct(hi)}] ({rate['k']}/{rate['n']})"


def _kappa(k: float | None) -> str:
    return "undefined" if k is None else f"{k:.2f}"


def _evaluator_rows(groups: Iterable[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "| Group | n (distinct rows) | Accuracy | Precision | Recall | Specificity | "
        "Cohen's kappa | TP / FP / FN / TN | Excluded |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group, m in groups:
        t = m["table"]
        excluded = ", ".join(f"{k}: {v}" for k, v in m["excluded"].items()) or "none"
        lines.append(
            f"| {group} | {m['n']} ({m['distinct_rows']}) | {_rate_cell(m['accuracy'])} | "
            f"{_rate_cell(m['precision'])} | {_rate_cell(m['recall'])} | "
            f"{_rate_cell(m['specificity'])} | {_kappa(m['cohens_kappa'])} | "
            f"{t['tp']} / {t['fp']} / {t['fn']} / {t['tn']} | {excluded} |"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    lines = [
        "# Evaluator agreement with human correctness labels (generated)",
        "",
        "Generated by `scripts/analyze_human_agreement.py`. Do not edit by hand.",
        "",
        "Positive class = correct. Unit = one evaluated instance (run, question); a sheet row's "
        "human label applies to every instance with that answer. Wilson 95% intervals assume "
        "independent instances; instances sharing a row are not independent, so read n together "
        "with the distinct-row count. Kappa has no interval. Runs used: "
        f"**{report['inputs']['runs']}** run(s) of each experiment.",
        "",
        "## Labels",
        "",
        f"- Sheet rows: {rows['total']} (overlap rows for double labelling: {rows['overlap']}; "
        f"guideline-example rows: {rows['guideline_examples']}). The guideline states a verdict "
        "for each guideline-example row, so those rows are excluded from inter-annotator "
        "agreement and from the primary evaluator results, and included only in the sensitivity "
        "section.",
        "- Reference labels: " + ", ".join(f"{k}: {v}" for k, v in rows["reference"].items()) + ".",
        "- Sources: " + ", ".join(f"{k}: {v}" for k, v in rows["reference_sources"].items()) + ".",
    ]
    for s in report["sheets"]:
        who = ", ".join(s["annotators"]) or "no annotator name"
        lines.append(
            f"- `{s['path']}` ({who}): {s['rows_labelled']}/{s['rows_total']} rows labelled; "
            + ", ".join(f"{k}: {v}" for k, v in s["label_counts"].items())
            + "."
        )
    lines += ["", "## Inter-annotator agreement (independent labels, before adjudication)", ""]
    inter = report["inter_annotator"]
    if inter is None:
        lines.append("Not computed: only one sheet was given.")
    else:
        b, t = inter["binary_definite"], inter["three_category"]
        met = {True: "yes", False: "no", None: "undefined"}[b["meets_target"]]
        lines += [
            f"- Rows labelled by both: {inter['rows_labelled_by_both']}; guideline-example rows "
            f"excluded: {inter['excluded_guideline_examples']}; "
            f"compared: {inter['rows_compared']}.",
            f"- Binary (correct vs incorrect, rows where either said unsure excluded: "
            f"{b['excluded_unsure']}): n = {b['n']}, raw agreement {_pct(b['raw_agreement'])}, "
            f"Cohen's kappa {_kappa(b['cohens_kappa'])} (target >= {b['kappa_target']}: {met}).",
            f"- Three categories (unsure as its own category): n = {t['n']}, raw agreement "
            f"{_pct(t['raw_agreement'])}, Cohen's kappa {_kappa(t['cohens_kappa'])}.",
            "",
            "| First \\ second | " + " | ".join(LABELS) + " |",
            "|---|" + "---|" * len(LABELS),
        ]
        for a in LABELS:
            cells = " | ".join(str(inter["confusion_first_by_second"][a][x]) for x in LABELS)
            lines.append(f"| {a} | {cells} |")
    for name, data in report["evaluators"].items():
        lines += ["", f"## {name} vs human (primary: guideline-example rows excluded)", ""]
        lines += _evaluator_rows([("overall", data["overall"]), *data["by_experiment"].items()])
    if report["evaluators_including_guideline_examples"]:
        lines += [
            "",
            "## Sensitivity: guideline-example rows included",
            "",
            "Same as above, but the guideline-example rows are used too. Their labels may follow "
            "the verdicts the guideline states, so this is not independent human evidence; it "
            "only shows how much those rows change the result.",
        ]
        for name, data in report["evaluators_including_guideline_examples"].items():
            lines += ["", f"### {name} vs human, including guideline examples", ""]
            lines += _evaluator_rows([("overall", data["overall"]), *data["by_experiment"].items()])
    if report["llm_judge_labels"]:
        lines += [
            "",
            "## llm_judge labels by human label",
            "",
            "The judge passes only `correct`; `partially_correct` counts as a fail above.",
            "",
            "| Experiment | Human correct | Human incorrect |",
            "|---|---|---|",
        ]
        for x, c in report["llm_judge_labels"].items():
            hc = ", ".join(f"{k}: {v}" for k, v in c["human_correct"].items()) or "-"
            hi = ", ".join(f"{k}: {v}" for k, v in c["human_incorrect"].items()) or "-"
            lines.append(f"| {x} | {hc} | {hi} |")
    if report["faithfulness_crosstab"]:
        lines += [
            "",
            "## llm_faithfulness_judge labels by human correctness (descriptive only)",
            "",
            report["faithfulness_note"],
            "",
            "| Experiment | Human correct | Human incorrect | Excluded |",
            "|---|---|---|---|",
        ]
        for x, c in report["faithfulness_crosstab"].items():
            hc = ", ".join(f"{k}: {v}" for k, v in c["human_correct"].items()) or "-"
            hi = ", ".join(f"{k}: {v}" for k, v in c["human_incorrect"].items()) or "-"
            ex = ", ".join(f"{k}: {v}" for k, v in c["excluded"].items()) or "none"
            lines.append(f"| {x} | {hc} | {hi} | {ex} |")
    if report["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"

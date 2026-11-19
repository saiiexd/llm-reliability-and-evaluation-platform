"""
Export a blind human-annotation sheet for answer correctness.

Reads the selected runs from a store, labels each distinct answer to a question
once (deduplicated on question + normalized answer), shuffles the rows with a
fixed seed and writes, into ``--out``:

- ``annotation_sheet.csv``: the blind sheet for the first annotator
  (row_id, question, gold_answers, model_answer, human_correct, annotator, notes);
  no model, arm, run or evaluator information;
- ``annotation_sheet_overlap.csv``: the overlap rows (>= 20%, chosen with the
  same seed) for the second annotator to label independently;
- ``annotation_key.csv``: row_id -> every (experiment_id, run_id, test_case_id)
  with that answer, plus the overlap flag. Not for annotators;
- ``export_manifest.json``: counts and parameters.

``human_correct`` is always empty: this script creates no labels.
Procedure and decision rules: ``research/annotation/guideline.md``.

Usage:
    python scripts/export_annotation_sheet.py --out DIR [--store experiments]
        [--experiment ID ...] [--seed 2026] [--overlap-fraction 0.25]

Without ``--experiment`` every ``exp001-*`` experiment except pilots is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_reliability.analytics.annotation import (
    DEFAULT_OVERLAP_FRACTION,
    DEFAULT_SEED,
    export_annotation_sheet,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--store", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--out", type=Path, required=True, help="directory for the output files")
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help="experiment id to include (repeatable); default: all exp001-* except pilots",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overlap-fraction", type=float, default=DEFAULT_OVERLAP_FRACTION)
    args = parser.parse_args(argv)
    try:
        manifest = export_annotation_sheet(
            store=args.store,
            out_dir=args.out,
            experiment_ids=args.experiments,
            seed=args.seed,
            overlap_fraction=args.overlap_fraction,
        )
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    summary = {k: manifest[k] for k in ("rows", "overlap_rows", "instances", "distinct_questions")}
    print(json.dumps({"out": str(args.out), **summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Measure how well automatic evaluators agree with human correctness labels.

Reads one or two filled annotation sheets, the key written by
``scripts/export_annotation_sheet.py`` and the run store, then writes
``human_agreement.json`` and ``human_agreement.md`` into ``--out``:

- inter-annotator Cohen's kappa on the rows both annotators labelled
  (binary correct/incorrect, unsure excluded and counted; and three-category);
- for exact_match, squad_match and llm_judge, overall and per experiment:
  accuracy, precision, recall and specificity (positive = correct) with Wilson
  95% intervals, Cohen's kappa, the confusion table and every exclusion count
  (unlabelled, unsure, unresolved disagreement, evaluator status not SUCCESS);
- for llm_faithfulness_judge only a descriptive table on RAG arms: it measures
  support by the retrieved context, a different criterion, so human correctness
  labels cannot validate it.

Refuses (exit code 2, nothing written) when no label is filled in.
Definitions: ``llm_reliability.analytics.human_agreement``.

Usage:
    python scripts/analyze_human_agreement.py --sheet FILLED.csv [--second-sheet FILLED2.csv]
        [--adjudication ADJ.csv] [--key annotation_key.csv] [--store experiments]
        [--runs first|all] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_reliability.analytics.human_agreement import (
    AnnotationFormatError,
    NoLabelsError,
    analyze_human_agreement,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY = REPO_ROOT / "research" / "annotation" / "exp001" / "annotation_key.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--sheet", type=Path, required=True, help="first annotator's sheet")
    parser.add_argument("--second-sheet", type=Path, help="second annotator's sheet")
    parser.add_argument(
        "--adjudication",
        type=Path,
        help="CSV (row_id, human_correct) with agreed labels for rows the annotators disagree on",
    )
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--store", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument(
        "--runs",
        choices=["first", "all"],
        default="first",
        help="first run of each experiment (default) or every run",
    )
    parser.add_argument("--out", type=Path, help="output directory (default: the key's directory)")
    args = parser.parse_args(argv)
    try:
        report = analyze_human_agreement(
            key_path=args.key,
            sheet_path=args.sheet,
            store=args.store,
            second_sheet_path=args.second_sheet,
            adjudication_path=args.adjudication,
            runs=args.runs,
        )
    except NoLabelsError as exc:
        parser.exit(2, f"refused: {exc}\n")
    except AnnotationFormatError as exc:
        parser.exit(2, f"error: {exc}\n")
    out = args.out or args.key.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "human_agreement.json").write_text(json.dumps(report, indent=1) + "\n")
    markdown = render_markdown(report)
    (out / "human_agreement.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

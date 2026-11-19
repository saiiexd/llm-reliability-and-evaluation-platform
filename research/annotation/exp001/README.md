# EXP-001 human annotation sheet (blank)

Material for validating the EXP-001 correctness evaluators against human judgement (RQ3).
**No human labels exist yet.** Every `human_correct` cell is empty, and nothing here was labelled
by a model.

## Files

| File | For | Content |
|---|---|---|
| `annotation_sheet.csv` | Annotator A | All 238 rows: `row_id, question, gold_answers, model_answer, human_correct, annotator, notes` |
| `annotation_sheet_overlap.csv` | Annotator B | The 60 overlap rows (25% of 238, at least 20% required; never a guideline-example row), same columns |
| `annotation_key.csv` | Analysis only, **never annotators** | One line per evaluated instance: `row_id, overlap, guideline_example, experiment_id, run_id, repetition_index, test_case_id, normalized_answer, raw_answer` |
| `export_manifest.json` | Provenance | Seed, overlap fraction, experiment and run ids, all counts below |

Read `../guideline.md` before labelling.

## Counts (from `export_manifest.json`)

- Source: the 9 archived EXP-001 runs (6 arms, 60 SQuAD questions each), 540 evaluated instances,
  every one with a model answer (no execution errors).
- **238 distinct answers to label.** Identical answers to the same question are labelled once; the
  dedup key is SQuAD normalization plus the answer's numbers verbatim. 158 rows stand for a single
  instance, and 80 rows stand for 2 to 9 instances: the same answer from several arms or repeated runs.
- Per arm (instances / distinct rows): gemma-closed 60/60, gemma-rag-k3 60/60, qwen-closed 60/60,
  qwen-rag-k1 60/60, qwen-rag-k3 120/63 (2 runs), qwen-rag-k3-t08 180/70 (3 runs).
- The analysis uses the first run of each arm by default. That scope is 360 instances covering 236
  rows, 59 of them overlap rows and 15 of them guideline-example rows. Two rows occur only in
  repeat runs and are used with `--runs all`.
- Row order and the overlap sample come from seed 2026.
- **15 guideline-example rows** (`guideline_example = 1` in the key): the rows showing the answers
  for which `../guideline.md` states a verdict (examples E1 to E14 and the contrast answer in E6).
  Those verdicts were written by the guideline author, not an annotator, and annotators read them,
  so these rows are not independent human evidence. They stay in the sheet and are labelled like
  any other row, but they are never in the overlap sample, the analysis excludes them from
  inter-annotator kappa, and it reports evaluator agreement without them (primary) and with them
  (sensitivity, `evaluators_including_guideline_examples`). In the first-run scope they cover 26 of
  the 360 instances (41 of the 540 over all runs). The list is `GUIDELINE_EXAMPLES` in
  `src/llm_reliability/analytics/annotation.py`; a test checks it against the guideline.

## Regenerating (deterministic)

The runs come from `research/experiments/EXP-001_local_rag_reliability/runs.tar.gz`. The experiment
definitions come from `experiments/experiments/exp001-*.json`. Build a store from both in a temporary
directory; never write into the real `./experiments`.

```bash
STORE=$(mktemp -d)
tar xzf research/experiments/EXP-001_local_rag_reliability/runs.tar.gz -C "$STORE"
mkdir "$STORE/experiments" && cp experiments/experiments/exp001-*.json "$STORE/experiments/"
OUT=$(mktemp -d)
PYTHONPATH=src .venv/bin/python scripts/export_annotation_sheet.py --store "$STORE" --out "$OUT"
diff -r "$OUT" research/annotation/exp001 --exclude README.md   # expect no output
```

## Analysis (after both annotators finish)

Annotators save their filled copies under new names, for example `labels_A.csv` and `labels_B.csv`.
Leave the committed blank files untouched. Save as UTF-8 ("CSV UTF-8 (Comma delimited)" in Excel;
"Text CSV" with character set "Unicode (UTF-8)" in LibreOffice). Excel's plain "CSV (Comma
delimited)" uses a Windows code page; the analysis rejects such a file with an error saying so.

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_human_agreement.py \
  --sheet labels_A.csv --second-sheet labels_B.csv \
  [--adjudication adjudication.csv] \
  --key research/annotation/exp001/annotation_key.csv --store "$STORE" \
  --out research/annotation/exp001/analysis
```

It writes `human_agreement.json` and `human_agreement.md`. It refuses, with exit code 2 and no output
written, while no label is filled in. Metric definitions are in the docstring of
`src/llm_reliability/analytics/human_agreement.py`. The unit is one evaluated instance, positive means
correct, and every exclusion is counted. `llm_faithfulness_judge` is reported only as a descriptive
table, because faithfulness is a different criterion from correctness.

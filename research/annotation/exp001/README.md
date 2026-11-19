# EXP-001 human annotation sheet (blank)

Material for validating the EXP-001 correctness evaluators against human judgement (RQ3).
**No human labels exist yet.** Every `human_correct` cell is empty, and nothing here was labelled
by a model.

## Files

| File | For | Content |
|---|---|---|
| `annotation_sheet.csv` | Annotator A | All 238 rows: `row_id, question, gold_answers, model_answer, human_correct, annotator, notes` |
| `annotation_sheet_overlap.csv` | Annotator B | The 60 overlap rows (25%, at least 20% required), same columns |
| `annotation_key.csv` | Analysis only, **never annotators** | One line per evaluated instance: `row_id, overlap, experiment_id, run_id, repetition_index, test_case_id, normalized_answer, raw_answer` |
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
  rows, 59 of them overlap rows. Two rows occur only in repeat runs and are used with `--runs all`.
- Row order and the overlap sample come from seed 2026.

## Regenerating (deterministic)

The runs come from `research/experiments/EXP-001_local_rag_reliability/runs.tar.gz`. The experiment
definitions come from `experiments/experiments/exp001-*.json`. Build a store from both in a temporary
directory; never write into the real `./experiments`.

```bash
STORE=$(mktemp -d)
tar xzf research/experiments/EXP-001_local_rag_reliability/runs.tar.gz -C "$STORE"
mkdir "$STORE/experiments" && cp experiments/experiments/exp001-*.json "$STORE/experiments/"
PYTHONPATH=src .venv/bin/python scripts/export_annotation_sheet.py --store "$STORE" --out /tmp/exp001-sheet
diff -r /tmp/exp001-sheet research/annotation/exp001 --exclude README.md   # expect no output
```

## Analysis (after both annotators finish)

Annotators save their filled copies under new names, for example `labels_A.csv` and `labels_B.csv`.
Leave the committed blank files untouched.

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

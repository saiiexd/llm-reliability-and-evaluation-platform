# Annotation guideline: answer correctness on the SQuAD v1.1 subset

Version 1 (2026-11-19). Purpose: produce human reference labels for checking how well the automatic
correctness evaluators (`exact_match`, `squad_match`, `llm_judge`) agree with people (RQ3, evaluator
reliability). The labels are the reference for that check, so they must be independent of the
evaluators: annotators never see any evaluator's verdict.

## The task

Each row of the sheet shows:

| Column | Meaning |
|---|---|
| `question` | A SQuAD v1.1 question, exactly as in the dataset (typos included, e.g. "innnate"). |
| `gold_answers` | Every gold answer from SQuAD, separated by ` \| `. They come from several crowd workers, so some are fragments or variants of each other. |
| `model_answer` | One model's answer, exactly as produced. The same answer from several runs appears only once. |

Fill in, for every row:

- `human_correct`: exactly one of `correct`, `incorrect`, `unsure`;
- `annotator`: your annotator ID (the same on every row);
- `notes`: required for `unsure` (say what is missing), optional otherwise.

Do not edit, delete or add any other cell. Sorting the sheet is fine; rows are matched by `row_id`.

## The question you answer

> Does the model answer give the same answer as at least one gold answer, so that a reader who asked
> the question would get the right answer from it?

`correct` means yes. `incorrect` means no. `unsure` means the gold answers alone cannot decide it
(rule 9 only). This is a binary judgement: there is no partial credit.

## Decision rules

Apply them in order; the first rule that fits decides.

1. **Abstention is incorrect.** "I don't know", "not stated", a refusal or an empty answer is
   `incorrect`. This is a correctness label, not an honesty label: an honest abstention still does
   not answer the question.
2. **Any one gold answer is enough.** Compare with every gold answer. Matching one of them is
   `correct`, even if another gold answer names something different.
3. **Surface differences do not matter.** Case, punctuation, articles (a/an/the), word order inside
   a list, and a trailing full stop never change the label.
4. **Paraphrases and equivalent names are correct** when they clearly mean the same thing as a gold
   answer: a spelled-out symbol ("gamma delta" for "γδ"), a fuller or shorter form of the same name
   ("Rembrandt van Rijn" for "Rembrandt"), a synonym phrase with the same meaning.
5. **Extra words are fine unless they add something wrong.** A full sentence that contains a gold
   answer, a restated question, or true supporting detail is `correct`. If the extra words contradict
   a gold answer or add a claim that changes the answer, it is `incorrect`. If the answer gives several
   candidates ("X or Y", "X, possibly Y") and any candidate is wrong, it is `incorrect`. If the answer
   shows reasoning and then a marked final answer, judge the final answer, provided nothing else in
   the text contradicts it.
6. **Partial answers are incorrect.** If the question asks for a specific set ("which three …", "what
   two forms …") or a gold answer lists several items, every item must be present and no item may be
   wrong. One right item out of three is `incorrect`. An answer that is vaguer than every gold answer,
   so that it would fit many other answers ("Diverse"), is `incorrect`.
7. **A different entity of the same kind is incorrect.** If the answer is the same kind of thing as
   the gold answers (a person for a person, an organism for an organism, a concept for a concept)
   but names a different one, so that it and the gold answer cannot both be the answer, it is
   `incorrect`, however plausible it sounds. Repeating the question's own term instead of answering
   is also `incorrect`.
8. **Numbers of the same kind must match in value.** This applies when the answer and a gold answer
   are the same kind of quantity, such as an amount for an amount or a year for a year. Different
   formatting of the same value is fine ("€53,423" vs "53,423"; "1,000" vs "1000"). A different
   value, a range or bound that is not the gold value ("Over €90,000"), or a different unit that
   changes the value is `incorrect`.
9. **Different kind or granularity: `unsure`.** If the answer is of a different kind or granularity
   from every gold answer, the gold answers alone cannot decide it. Rules 7 and 8 then do not apply.
   Example: a year where the gold answer names an event. Do not look anything up: label `unsure` and
   write the reason in `notes`. `unsure` is not for answers that are merely hard to judge. Expect it
   to be rare.

## Examples

All examples are real EXP-001 answers, quoted verbatim. They were written by the guideline author to
illustrate the rules. They are **not** human labels and are not used in any analysis; the same answers
also appear in the sheet, where you label them like every other row. Which model or arm produced them
is deliberately not stated.

**E1. Paraphrase (rule 4): correct.**
- Question: What type of T cells help with both innnate and adaptive immunity?
- Gold answers: γδ | γδ T cells
- Model answer: Gamma delta T cells
- Why: "Gamma delta" is the spelled-out form of "γδ".

**E2. Paraphrase (rule 4): correct.**
- Question: What is the effect of testosterone on the male immune system?
- Gold answers: immunosuppressive
- Model answer: Suppresses immune function.
- Why: same meaning in different words.

**E3. Extra words, nothing wrong added (rule 5): correct.**
- Question: Who has loaned the Raphael Cartoons to the museum?
- Gold answers: Queen Elizabeth II
- Model answer: Her Majesty the Queen Elizabeth II

**E4. Full sentence containing a gold answer (rule 5): correct.**
- Question: What is the most well-known algorithm associated with the integer factorization problem?
- Gold answers: RSA | general number field sieve | the general number field sieve
- Model answer: The best known algorithm for integer factorization is the general number field sieve.
- Why: also an example of rule 2; it matches the second gold answer.

**E5. Extra words that add a wrong claim (rule 5): incorrect.**
- Question: What do most online pharmacies do?
- Gold answers: sell prescription drugs | sell prescription drugs and require a valid prescription
- Model answer: Sell prescription drugs without a prescription.
- Why: "without a prescription" contradicts the second gold answer and changes the answer.

**E6. Partial list with wrong items (rule 6): incorrect.**
- Question: Which three famous British potters are represented the the V&A ceramics collection?
- Gold answers: Josiah Wedgwood, William De Morgan and Bernard Leach
- Model answer: John Clare, Josiah Wedgwood, Peter Firton
- Why: one of the three is right, two are wrong. By contrast, "Bernard Leach, Josiah Wedgwood, and
  William De Morgan" is `correct` (order does not matter, rule 3).

**E7. Too vague (rule 6): incorrect.**
- Question: Which cultures are represented in the V&A's collections?
- Gold answers: Europe, North America, Asia and North Africa
- Model answer: Diverse

**E8. Wrong entity (rule 7): incorrect.**
- Question: What is an example of a parasite that used the antigenic variation strategy to evade destruction?
- Gold answers: Trypanosoma brucei
- Model answer: HIV

**E9. Repeats the question instead of answering (rule 7): incorrect.**
- Question: What are white blood cells known as?
- Gold answers: Leukocytes
- Model answer: White blood cells

**E10. Abstention (rule 1): incorrect.**
- Question: Who officially opened the V&A?
- Gold answers: Queen Victoria
- Model answer: I don't know

**E11. Multiple gold answers (rule 2): correct.**
- Question: Where are teachers recruited from?
- Gold answers: Lehramtstudien | Lehramtstudien (Teaching Education Studies) | special university classes
- Model answer: Special university classes

**E12. Same number, different format (rule 8): correct.**
- Question: What can a teacher with 25 years of experience make, in Euros?
- Gold answers: 53,423 | €53,423
- Model answer: €53,423

**E13. Different number (rule 8): incorrect.**
- Question: What can a teacher with 25 years of experience make, in Euros?
- Gold answers: 53,423 | €53,423
- Model answer: Over €90,000

**E14. Gold answers cannot decide it (rule 9): unsure.**
- Question: When was Thoreau's essay published?
- Gold answers: after the end of the Mexican War | after the end of the Mexican War. | end of the Mexican War
- Model answer: 1850
- Why: the answer is a year and the gold answers name an event, which is a different granularity.
  Rules 7 and 8 do not apply, and checking the year would need outside facts.
  Note: "No year given in gold; needs a date for the end of the Mexican War."

## Procedure

1. **Annotators.** Two people, A and B, who label independently. They must not be shown the key,
   `per_case.csv`, any results file, the console or the run files, and they must not discuss specific
   rows until both have finished. Labels come from people only: never fill the sheet with a model's
   output, and never copy an evaluator's verdict.
2. **Blinding.** The sheet contains no model, arm, run or evaluator information, and the rows are
   shuffled with a fixed seed. Annotators work from the sheet and this guideline only. Questions about
   the guideline go to the study lead. The answer is added to this guideline as a new version, without
   referring to any row's verdict.
3. **Overlap.** A labels every row of `annotation_sheet.csv`. B labels every row of
   `annotation_sheet_overlap.csv`: a random 25% of the rows, chosen with the export seed. The minimum
   is 20%. B may label the whole sheet instead, using a copy of `annotation_sheet.csv`; more overlap
   gives a more precise kappa. The full sheet does not show which rows are in the overlap.
4. **Agreement.** Run `scripts/analyze_human_agreement.py` with both sheets. The primary measure is
   Cohen's kappa on correct vs incorrect over rows both annotators labelled. Rows where either
   annotator said `unsure` are excluded from it and counted. The target is **kappa ≥ 0.6**. Kappa is
   computed from the original, independent labels; never edit a sheet after this step.
5. **If kappa < 0.6.** Report the value as it is. Review the disagreements only to find unclear rules,
   then revise the guideline (new version). A second round on rows B has not yet seen is a new,
   separately reported measurement; the first-round value is not replaced.
6. **Adjudication.** The disagreeing rows are listed in `human_agreement.json`
   (`inter_annotator.disagreeing_row_ids`). For each one, A and B discuss it against the rules. If they
   agree, they record the label in a separate `adjudication.csv` (columns `row_id, human_correct, notes`).
   If they still disagree, a third person who has not seen any evaluator verdict decides. If it is still
   unclear, the row is `unsure`. Pass the file with `--adjudication`. Disagreeing rows without an
   adjudicated label are excluded from evaluator agreement as *unresolved* and counted. Adjudication
   never changes the kappa from step 4.
7. **Reference label.** The reference label is the single annotator's label for rows only one person
   labelled, the shared label where A and B agree, and the adjudicated label otherwise. `unsure` rows
   are excluded from evaluator agreement and counted.

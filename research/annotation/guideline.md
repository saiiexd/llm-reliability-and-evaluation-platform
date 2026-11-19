# Annotation guideline: answer correctness on the SQuAD v1.1 subset

Version 2 (2026-11-19). Purpose: produce human reference labels for checking how well the automatic
correctness evaluators (`exact_match`, `squad_match`, `llm_judge`) agree with people (RQ3, evaluator
reliability). The labels are the reference for that check, so they must be independent of the
evaluators: annotators never see any evaluator's verdict.

Changes from version 1 (issued before any row was labelled): the decision rules are reordered so
that every rule making an answer `incorrect` comes before the rules making it `correct`, and the
rules that mixed both are split; the rule 9 name example is now one that is not in the sheet; the
rows whose verdict the examples state are now excluded from agreement (see Examples). No example's
verdict changed; the new order removes the conflicts in which the version 1 order gave E5 and E9 the
opposite of their stated verdict.

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
Save the sheet as UTF-8: in Excel use "CSV UTF-8 (Comma delimited)", not "CSV (Comma delimited)",
which uses a Windows code page and garbles characters such as "γδ" and "€"; in LibreOffice use
"Text CSV" with character set "Unicode (UTF-8)". The analysis rejects a file that is not UTF-8.

## The question you answer

> Does the model answer give the same answer as at least one gold answer, so that a reader who asked
> the question would get the right answer from it?

`correct` means yes. `incorrect` means no. `unsure` means the gold answers alone cannot decide it
(rule 10 only). This is a binary judgement: there is no partial credit.

## Decision rules

Apply them in order; the first rule that fits decides. Rules 1 to 6 are exceptions that make an
answer `incorrect` even when it contains or resembles a gold answer, so check them first. Rules 7 to 9
say when an answer is `correct`. Rule 10 is the only `unsure` rule.

Throughout, compare with every gold answer: they come from several crowd workers, and matching any
one of them is enough. Surface differences never change the label: case, punctuation, articles
(a/an/the), word order inside a list, a trailing full stop, and the formatting of a number with the
same value ("€53,423" vs "53,423"; "1,000" vs "1000").

1. **Abstention is incorrect.** "I don't know", "not stated", a refusal or an empty answer is
   `incorrect`. This is a correctness label, not an honesty label: an honest abstention still does
   not answer the question.
2. **A contradiction or a wrong candidate is incorrect.** If words in the answer contradict a gold
   answer or add a claim that changes the answer, it is `incorrect`, even if the rest matches a gold
   answer. If the answer gives several candidates ("X or Y", "X, possibly Y") and any candidate is
   wrong, it is `incorrect`. If the answer shows reasoning and then a marked final answer, judge the
   final answer with these rules; if something else in the text contradicts it, the answer is
   `incorrect` under this rule. An answer that matches one gold answer and says nothing against
   another is not a contradiction.
3. **Repeating the question's own term is incorrect.** An answer that only repeats a term from the
   question, instead of giving what the question asks for, is `incorrect`, even when that term means
   the same as a gold answer.
4. **A different number is incorrect.** This applies when the answer and a gold answer are the same
   kind of quantity, such as an amount for an amount or a year for a year. If the answer's value is
   not the value of any gold answer, it is `incorrect`: a different value, a range or bound that is
   not the gold value ("Over €90,000"), or a different unit that changes the value. The same value
   in a different format is not a different number; go on to the next rules. If the answer is a
   different kind of quantity from every gold answer (a year where the gold answer names an event),
   this rule does not apply: see rule 10.
5. **Partial and vague answers are incorrect.** If the question asks for a specific set ("which three
   …", "what two forms …") or a gold answer lists several items, every item must be present, in any
   order, and no item may be wrong. One right item out of three is `incorrect`. An answer that is
   vaguer than every gold answer, so that it would fit many other answers ("Diverse"), is
   `incorrect`.
6. **A different entity of the same kind is incorrect.** If the answer is the same kind of thing as
   the gold answers (a person for a person, an organism for an organism, a concept for a concept)
   but names a different one from every gold answer, so that it and a gold answer cannot both be the
   answer, it is `incorrect`, however plausible it sounds. If the answer is a different kind of thing
   or a different granularity from every gold answer, this rule does not apply: see rule 10.
7. **The same answer as a gold answer is correct.** An answer that is the same as one gold answer,
   apart from surface differences, is `correct`, even if another gold answer names something
   different.
8. **Extra words that add nothing wrong are correct.** A full sentence that contains a gold answer,
   a restated question, or true supporting detail around a gold answer is `correct` (wrong extra
   words are rule 2).
9. **Paraphrases and equivalent names are correct** when they clearly mean the same thing as a gold
   answer: a spelled-out symbol ("gamma delta" for "γδ"), a fuller or shorter form of the same name
   ("Ludwig van Beethoven" for "Beethoven"), a synonym phrase with the same meaning.
10. **Different kind or granularity: `unsure`.** If the answer is of a different kind or granularity
    from every gold answer, the gold answers alone cannot decide it. Rules 4 and 6 do not apply to
    such an answer, because they are for answers of the same kind as a gold answer. Example: a year
    where the gold answer names an event. Do not look anything up: label `unsure` and write the
    reason in `notes`. `unsure` is not for answers that are merely hard to judge. Expect it to be
    rare.

## Examples

All examples are real EXP-001 answers, quoted verbatim. They were written by the guideline author
(not an annotator) to illustrate the rules, and their verdicts are **not** human labels. The same
answers also appear in the sheet, where you label them like every other row. Because this guideline
states their verdict, those rows (the 14 examples below and the contrast answer in E6) are flagged in
the analysis key as guideline examples: they are never in the overlap sample that B labels, they are
excluded from inter-annotator agreement, and the evaluator-vs-human results are reported without them
(primary result) and, separately, with them (sensitivity check). Which model or arm produced them is
deliberately not stated.

**E1. Paraphrase (rule 9): correct.**
- Question: What type of T cells help with both innnate and adaptive immunity?
- Gold answers: γδ | γδ T cells
- Model answer: Gamma delta T cells
- Why: "Gamma delta" is the spelled-out form of "γδ".

**E2. Paraphrase (rule 9): correct.**
- Question: What is the effect of testosterone on the male immune system?
- Gold answers: immunosuppressive
- Model answer: Suppresses immune function.
- Why: same meaning in different words.

**E3. Extra words, nothing wrong added (rule 8): correct.**
- Question: Who has loaned the Raphael Cartoons to the museum?
- Gold answers: Queen Elizabeth II
- Model answer: Her Majesty the Queen Elizabeth II

**E4. Full sentence containing a gold answer (rule 8): correct.**
- Question: What is the most well-known algorithm associated with the integer factorization problem?
- Gold answers: RSA | general number field sieve | the general number field sieve
- Model answer: The best known algorithm for integer factorization is the general number field sieve.
- Why: it contains the second gold answer. Matching any one gold answer is enough, so the
  different first gold answer (RSA) does not matter, and nothing contradicts it (rule 2).

**E5. Extra words that add a wrong claim (rule 2): incorrect.**
- Question: What do most online pharmacies do?
- Gold answers: sell prescription drugs | sell prescription drugs and require a valid prescription
- Model answer: Sell prescription drugs without a prescription.
- Why: "without a prescription" contradicts the second gold answer and changes the answer. Rule 2
  comes before rule 8, so matching the first gold answer does not make it correct.

**E6. Partial list with wrong items (rule 5): incorrect.**
- Question: Which three famous British potters are represented the the V&A ceramics collection?
- Gold answers: Josiah Wedgwood, William De Morgan and Bernard Leach
- Model answer: John Clare, Josiah Wedgwood, Peter Firton
- Why: one of the three is right, two are wrong. By contrast,
  "Bernard Leach, Josiah Wedgwood, and William De Morgan" is `correct`: all three are present and
  none is wrong, so rule 5 does not apply, and it is the same as the gold answer apart from word order
  inside the list and punctuation (rule 7).

**E7. Too vague (rule 5): incorrect.**
- Question: Which cultures are represented in the V&A's collections?
- Gold answers: Europe, North America, Asia and North Africa
- Model answer: Diverse

**E8. Wrong entity (rule 6): incorrect.**
- Question: What is an example of a parasite that used the antigenic variation strategy to evade destruction?
- Gold answers: Trypanosoma brucei
- Model answer: HIV

**E9. Repeats the question instead of answering (rule 3): incorrect.**
- Question: What are white blood cells known as?
- Gold answers: Leukocytes
- Model answer: White blood cells
- Why: it repeats the question's own term. Rule 3 comes before rule 9, so it is `incorrect` even
  though "white blood cells" means the same as "leukocytes".

**E10. Abstention (rule 1): incorrect.**
- Question: Who officially opened the V&A?
- Gold answers: Queen Victoria
- Model answer: I don't know

**E11. Multiple gold answers (rule 7): correct.**
- Question: Where are teachers recruited from?
- Gold answers: Lehramtstudien | Lehramtstudien (Teaching Education Studies) | special university classes
- Model answer: Special university classes

**E12. Same number, different format (rules 4 and 7): correct.**
- Question: What can a teacher with 25 years of experience make, in Euros?
- Gold answers: 53,423 | €53,423
- Model answer: €53,423
- Why: the value is a gold value, so rule 4 does not apply; apart from formatting it is the same as
  both gold answers (rule 7).

**E13. Different number (rule 4): incorrect.**
- Question: What can a teacher with 25 years of experience make, in Euros?
- Gold answers: 53,423 | €53,423
- Model answer: Over €90,000

**E14. Gold answers cannot decide it (rule 10): unsure.**
- Question: When was Thoreau's essay published?
- Gold answers: after the end of the Mexican War | after the end of the Mexican War. | end of the Mexican War
- Model answer: 1850
- Why: the answer is a year and the gold answers name an event, which is a different granularity.
  Rules 4 and 6 do not apply, and checking the year would need outside facts.
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
   `annotation_sheet_overlap.csv`: a random 25% of the rows, chosen with the export seed from the
   rows that are not guideline examples (see Examples). The minimum is 20%. B may label the whole sheet instead, using a copy of `annotation_sheet.csv`; more overlap
   gives a more precise kappa. The full sheet does not show which rows are in the overlap.
4. **Agreement.** Run `scripts/analyze_human_agreement.py` with both sheets. The primary measure is
   Cohen's kappa on correct vs incorrect over rows both annotators labelled, except the
   guideline-example rows, which are excluded and counted. Rows where either annotator said `unsure`
   are excluded from it and counted. The target is **kappa ≥ 0.6**. Kappa is
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
   are excluded from evaluator agreement and counted. Guideline-example rows are excluded from the
   primary evaluator agreement and counted; a separate sensitivity result includes them.

# Research Note: Reliability Diagnostics, Faithfulness, and Robustness Layer

This note outlines the conceptual framework, decision procedures, and explicit limitations of the LLM Reliability & Evaluation Platform's Diagnostics, Faithfulness, and Robustness Layer (`src/llm_reliability/diagnostics/`).

---

## 1. Purpose and Philosophy

The diagnostics layer exists to answer a fundamental scientific question:
**When an LLM or RAG application produces an output, does it fail, where does it fail, and why does it fail?**

Rather than assigning a generic score or an opaque percentage, this platform isolates observable failure causes into distinct, evidence-grounded categories.

### The Deliberate Absence of the Term "Hallucination"

The platform deliberately avoids using the term "hallucination" as a blanket label or score. In industry literature, "hallucination" is an overloaded umbrella term that conflates multiple distinct failure mechanisms:
1. Retrieval failure: Ground-truth evidence was omitted from retrieved context.
2. Ungrounded claim: The model introduced claims absent from the provided context.
3. Direct contradiction: The model generated statements contradicting the provided context.
4. Reasoning or generation failure: The model had access to correct context but made an incorrect inference.
5. Evaluator disagreement: Evaluators with differing standards reached conflicting verdicts.

Conflating these distinct phenomena into a single "hallucination rate" obscures the engineering root cause and makes targeted remediation impossible. The platform replaces blanket labels with a controlled taxonomy of evidence-backed failure categories.

---

## 2. Controlled Diagnostic Taxonomy and Precedence Logic

The consolidated diagnostic decision procedure combines evidence from generation execution, reference-based correctness evaluators, context-grounded faithfulness evaluators, and retrieval ground-truth metrics.

### Categories (`FailureCategory`)
- `CORRECT`: The answer is verified correct against reference answers and context.
- `INCORRECT`: The answer is judged incorrect, with no retrieval evidence available to attribute the cause further.
- `UNSUPPORTED`: The answer contains statements neither confirmed nor denied by the retrieved context.
- `CONTRADICTED`: The answer contains statements directly conflicting with retrieved context.
- `RETRIEVAL_FAILURE`: Ground-truth context was not retrieved, leading to an incorrect generation.
- `GENERATION_FAILURE`: Relevant ground-truth context *was* retrieved, but generation was nonetheless incorrect.
- `EVALUATOR_DISAGREEMENT`: Independent evaluators of the same criterion produced conflicting verdicts.
- `INSUFFICIENT_EVIDENCE`: Evaluators were skipped or required evidence (e.g. references or ground-truth chunks) was absent.
- `UNDETERMINED`: Evidence is contradictory or inconclusive.

### Precedence Resolution
When evidence could support multiple categories, the most specific and actionable evidence determines the **primary failure category**:
1. Execution error (`TestCaseResult.execution_error`) $\rightarrow$ `ERROR`
2. No usable evaluation evidence $\rightarrow$ `INSUFFICIENT_EVIDENCE`
3. Correctness evaluators disagree $\rightarrow$ `EVALUATOR_DISAGREEMENT`
4. Faithfulness evaluator reports contradiction $\rightarrow$ `CONTRADICTED`
5. Faithfulness evaluator reports unsupported claims $\rightarrow$ `UNSUPPORTED`
6. Retrieval miss and answer incorrect $\rightarrow$ `RETRIEVAL_FAILURE`
7. Retrieval hit and answer incorrect $\rightarrow$ `GENERATION_FAILURE`
8. Answer judged correct $\rightarrow$ `CORRECT`
9. Answer judged incorrect $\rightarrow$ `INCORRECT`
10. Ambiguous evidence $\rightarrow$ `UNDETERMINED`

### Preservation of Secondary Findings
Assigning a primary failure category **must never discard secondary findings**. If an answer is assigned `CONTRADICTED` as its primary category, but retrieval also missed ground-truth chunks, or correctness evaluators disagreed, those secondary signals are explicitly preserved in `ReliabilityDiagnostic.secondary_findings`. This guarantees full transparency and prevents masking compounding failure modes.

---

## 3. Faithfulness Implementation and Explicit Limitations

The platform provides two complementary methods for assessing faithfulness against context: **Lexical Overlap Baseline** and **LLM-as-a-Judge Faithfulness Evaluator**. Neither method is a universal hallucination detector, and each has clear, documented limitations.

### 3.1 Lexical Evidence Matching (`ContextSupportBaseline`, `assess_statements_lexically`)

#### How It Works
The lexical baseline splits generated text into sentence-level statements and calculates word token overlap ratios against retrieved passages after removing punctuation and common stopwords. Statements above a defined overlap threshold are categorized as `SUPPORTED`; those below are categorized as `UNSUPPORTED`.

#### Explicit Limitations
1. **Blindness to Semantic Meaning and Negation**:
   - Word overlap cannot distinguish between affirmative and negative statements. For example:
     - Context: *"Paris is the capital of France."*
     - Generated Statement: *"Paris is not the capital of France."*
     - Word overlap: $> 80\%$. The lexical baseline labels this statement `SUPPORTED`, despite direct factual contradiction.
2. **Inability to Detect Contradiction**:
   - Lexical overlap can only detect the presence or absence of shared vocabulary; it has no mechanism to identify contradiction (`CONTRADICTED`).
3. **Paraphrase and Synonym Blindness**:
   - A valid paraphrase expressing identical factual content using different vocabulary will receive a low overlap score and be falsely flagged as `UNSUPPORTED`.
4. **Statement vs. Claim Granularity**:
   - Sentence-level segmentation on punctuation does not isolate atomic factual claims. A single sentence may contain multiple claims with differing support statuses.
5. **No Universal Detector Guarantee**:
   - A high lexical overlap score only confirms token intersection, not factual correctness or grounding.

### 3.2 LLM-as-a-Judge (`LLMFaithfulnessEvaluator`)

#### How It Works
An LLM judge receives the retrieved context and generated answer along with an explicit, versioned rubric instructing it to evaluate whether each claim is supported, unaddressed (`unsupported`), or contradicted. The judge produces structured JSON output containing its categorical verdict and reasoning.

#### Explicit Limitations
1. **Judge Hallucinations and Reasoning Errors**:
   - The judge model is itself a probabilistic language model subject to ungrounded assumptions, world-knowledge leakage, and incorrect deduction.
2. **Systematic Model Biases**:
   - *Position Bias*: Models frequently pay greater attention to context at the beginning or end of prompts, overlooking evidence in the middle ("lost in the middle").
   - *Verbosity Bias*: LLM judges systematically favor longer, verbose answers over concise answers regardless of accuracy.
   - *Self-Enhancement Bias*: Models often favor stylistic patterns generated by their own model family.
3. **Non-Determinism and Sensitivity to Prompting**:
   - Minor perturbations in prompt formatting, punctuation, or temperature settings can alter verdicts.
4. **Absence of Calibrated Confidence**:
   - Confidence scores emitted by LLM judges are notoriously uncalibrated and cannot be interpreted as true posterior probabilities.
5. **No Universal Detector Guarantee**:
   - An LLM judge verdict is a model-generated hypothesis, not ground truth. It must be treated as one evidence signal among many, never as infallible proof of faithfulness.

---

## 4. Controlled Perturbations and Robustness Analysis

Robustness evaluation in this platform measures whether observable system outcomes (retrieval ranking, generation correctness, faithfulness classification) remain stable under controlled, non-adversarial perturbations:
- `question_rewording`: Syntactically distinct but semantically equivalent question phrasings.
- `context_reordering`: Permuting the order of retrieved chunks provided to the generation prompt.
- `top_k_change`: Supplying different numbers of retrieved chunks to generation.
- `irrelevant_context_injection`: Injecting an irrelevant distracter chunk into context.
- `context_modification`: Controlled content edits to retrieved text.
- `repeated_input`: Re-executing identical inputs to quantify stochastic variation.

Robustness results track outcome stability rates and category transitions (e.g. `correct -> incorrect`, `correct -> contradicted`), illuminating fragile behaviors without relying on synthetic adversarial attacks.


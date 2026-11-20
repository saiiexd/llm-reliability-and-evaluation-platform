"""
Build the frozen question-paraphrase set for EXP-002 (RQ5, question rewording).

Each of the 60 SQuAD-subset questions is reworded once by a local model
(qwen3:4b-instruct via Ollama, temperature 0, fixed seed) and the result is
written to ``data/benchmarks/squad_v1_1_dev_subset_v1_paraphrases_v1.json``
together with the exact prompt, model digest and automatic quality checks.
Generating once and freezing the file means every run of the paraphrase arm
uses the same questions, and the file's content is part of the dataset
fingerprint.

Automatic checks per item (recorded, not silently fixed):
  changed       the normalized paraphrase differs from the normalized original
  no_answer_leak no gold answer appears in the paraphrase unless it already
                appeared in the original question
  is_question   ends with "?"
  length_ratio  paraphrase/original length between 0.5 and 2.0

These checks cannot establish that the meaning (and therefore the answer) is
unchanged; that requires reading every pair. See
``research/experiments/EXP-002_robustness/paraphrase_review.md``.

Usage:
    python scripts/build_paraphrase_set.py            # requires Ollama with qwen3:4b-instruct
"""

from __future__ import annotations

import json
import re
import string
import sys
import urllib.request
from pathlib import Path

from llm_reliability.datasets.squad import SQUAD_SUBSET_PATH, load_squad_subset
from llm_reliability.evaluation.models import ExecutionRequest, ModelConfig
from llm_reliability.providers.base import ProviderConfig
from llm_reliability.providers.openai_provider import OpenAIProviderAdapter

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = SQUAD_SUBSET_PATH.with_name("squad_v1_1_dev_subset_v1_paraphrases_v1.json")
OLLAMA = "http://localhost:11434"
MODEL = "qwen3:4b-instruct"
SEED = 42
PROMPT = (
    "Rewrite the following question using different wording and sentence structure. "
    "Keep exactly the same meaning, so that it has exactly the same answer. Keep every "
    "name, number and technical term needed to identify what is asked. Do not answer it. "
    "Reply with only the rewritten question.\n\n"
    "Question: {question}\n"
    "Rewritten question:"
)


def _norm(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def checks(original: str, paraphrase: str, gold: list[str]) -> dict[str, bool]:
    leaks = [
        g
        for g in gold
        if _norm(g) and _norm(g) in _norm(paraphrase) and _norm(g) not in _norm(original)
    ]
    ratio = len(paraphrase) / max(1, len(original))
    return {
        "changed": _norm(paraphrase) != _norm(original),
        "no_answer_leak": not leaks,
        "is_question": paraphrase.rstrip().endswith("?"),
        "length_ratio_ok": 0.5 <= ratio <= 2.0,
    }


def main() -> int:
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as resp:  # noqa: S310
        tags = json.loads(resp.read())
    digest = next((m["digest"] for m in tags["models"] if m["name"] == MODEL), None)
    provider = OpenAIProviderAdapter(
        config=ProviderConfig(
            provider_id="ollama-openai-compatible",
            model_id=MODEL,
            base_url=f"{OLLAMA}/v1",
            timeout_seconds=120.0,
            max_retries=2,
            allow_external_inference=True,
        )
    )
    config = ModelConfig(
        model_id=MODEL, temperature=0.0, max_tokens=96, extra_params={"seed": SEED}
    )
    subset = load_squad_subset()
    items = []
    for tc in subset.raw["test_cases"]:
        response = provider.generate(
            ExecutionRequest(tc["id"], PROMPT.format(question=tc["input"]), config)
        )
        paraphrase = response.output_text.strip().strip('"').strip()
        gold = list(tc["metadata"].get("gold_answers") or [tc["reference_answer"]])
        items.append(
            {
                "id": tc["id"],
                "original": tc["input"],
                "paraphrase": paraphrase,
                "checks": checks(tc["input"], paraphrase, gold),
            }
        )
        print(f"{tc['id']}: {paraphrase}", flush=True)
    failed = {k: sum(1 for i in items if not i["checks"][k]) for k in items[0]["checks"]}
    OUT_PATH.write_text(
        json.dumps(
            {
                "name": "squad-v1.1-dev-subset-v1-paraphrases-v1",
                "source_dataset": subset.name,
                "description": (
                    "One automatic rewording per question, generated once and frozen for "
                    "EXP-002 (question-rewording perturbation)."
                ),
                "generator": {
                    "model": MODEL,
                    "model_digest": digest,
                    "server": "ollama (OpenAI-compatible API)",
                    "temperature": 0.0,
                    "seed": SEED,
                    "max_tokens": 96,
                    "prompt": PROMPT,
                },
                "automatic_check_failures": failed,
                "items": items,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_PATH} ({len(items)} items); check failures: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

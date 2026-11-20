"""
Build the deterministic SQuAD v1.1 dev subset used by the reliability study.

Source: SQuAD v1.1 development set (Rajpurkar et al., 2016,
"SQuAD: 100,000+ Questions for Machine Comprehension of Text"), licensed
CC BY-SA 4.0. Download it once (not committed):

    curl -L -o dev-v1.1.json \\
        https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

Then:

    python scripts/build_squad_subset.py path/to/dev-v1.1.json

Sampling (fixed seed, so the output is byte-for-byte reproducible from the
same source file):

1. Pick ``N_ARTICLES`` article titles with ``random.Random(SEED)``.
2. Every paragraph of those articles becomes a corpus document, so each
   question competes against many same-topic distractor passages.
3. From each article, pick ``QUESTIONS_PER_ARTICLE`` distinct paragraphs and
   one question from each (so no two questions share a source paragraph).

Each test case keeps every gold answer (SQuAD has several per question) and
the character offset of the first one in its source paragraph, so the study
can derive which chunks are relevant *after* chunking rather than assuming.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

SEED = 2026
N_ARTICLES = 6
QUESTIONS_PER_ARTICLE = 10
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "data" / "benchmarks" / "squad_v1_1_dev_subset_v1.json"


def build(source: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(SEED)
    articles = sorted(source["data"], key=lambda a: a["title"])
    chosen = rng.sample(articles, N_ARTICLES)

    documents: list[dict[str, Any]] = []
    test_cases: list[dict[str, Any]] = []
    for article in chosen:
        title = article["title"]
        para_ids = []
        for p_index, paragraph in enumerate(article["paragraphs"]):
            doc_id = f"{title}::p{p_index:03d}"
            documents.append({"id": doc_id, "title": title, "content": paragraph["context"]})
            para_ids.append((doc_id, paragraph))
        picked = rng.sample(para_ids, QUESTIONS_PER_ARTICLE)
        for doc_id, paragraph in picked:
            qa = rng.choice(paragraph["qas"])
            gold = [a["text"] for a in qa["answers"]]
            test_cases.append(
                {
                    "id": f"squad-{qa['id']}",
                    "input": qa["question"],
                    "reference_answer": gold[0],
                    "metadata": {
                        "source": "squad_v1.1_dev",
                        "squad_id": qa["id"],
                        "article": title,
                        "source_document_id": doc_id,
                        "answer_start": qa["answers"][0]["answer_start"],
                        "gold_answers": sorted(set(gold)),
                    },
                }
            )

    payload: dict[str, Any] = {
        "name": "squad-v1.1-dev-subset-v1",
        "description": (
            f"{len(test_cases)} questions from {N_ARTICLES} SQuAD v1.1 dev articles, with all "
            f"{len(documents)} paragraphs of those articles as the retrieval corpus."
        ),
        "source": {
            "dataset": "SQuAD v1.1 (development set)",
            "citation": "Rajpurkar, Zhang, Lopyrev, Liang. SQuAD: 100,000+ Questions for "
            "Machine Comprehension of Text. EMNLP 2016.",
            "url": "https://rajpurkar.github.io/SQuAD-explorer/",
            "license": "CC BY-SA 4.0",
            "source_version": source.get("version"),
        },
        "sampling": {
            "seed": SEED,
            "n_articles": N_ARTICLES,
            "questions_per_article": QUESTIONS_PER_ARTICLE,
            "articles": [a["title"] for a in chosen],
        },
        "documents": documents,
        "test_cases": test_cases,
    }
    return payload


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    source_path = Path(sys.argv[1])
    raw = source_path.read_bytes()
    payload = build(json.loads(raw))
    payload["source"]["source_sha256"] = hashlib.sha256(raw).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(payload['test_cases'])} questions and {len(payload['documents'])} documents "
        f"to {OUTPUT} ({OUTPUT.stat().st_size // 1024} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

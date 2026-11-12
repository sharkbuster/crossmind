"""Threshold calibration probe (evidence for docs/EVALUATION.md).

Prints, for every evaluation case, the retrieval score and the answerability
coverage.  The point is to choose thresholds from the observed gap between
answerable and unanswerable questions instead of guessing them.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossmind.config import Settings  # noqa: E402
from crossmind.embedding import HashingEmbedder  # noqa: E402
from crossmind.rag.answerability import AnswerabilityGate, build_idf  # noqa: E402
from crossmind.rag.chunker import chunk_documents  # noqa: E402
from crossmind.rag.corpus import load_corpus  # noqa: E402
from crossmind.rag.index import HybridIndex  # noqa: E402
from crossmind.rag.retriever import Retriever  # noqa: E402


async def main() -> None:
    settings = Settings()
    data = ROOT / "data"
    documents = load_corpus(data / "corpus")
    chunks = chunk_documents(
        documents, chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap
    )
    embedder = HashingEmbedder(settings.embedding_dim)
    index = await HybridIndex.build(chunks, embedder)
    retriever = Retriever(index, embedder, settings)
    idf, _ = build_idf([f"{c.doc_title} {c.section} {c.text}" for c in chunks])
    gate = AnswerabilityGate(settings.rag_answerability_threshold, idf=idf)

    payload = json.loads((data / "eval_set.json").read_text("utf-8"))
    rows = []
    for case in payload["cases"]:
        context = await retriever.retrieve(case["question"])
        ok, coverage, missing = gate.evaluate(case["question"], context.chunks)
        rows.append(
            {
                "id": case["id"],
                "answerable": case["answerable"],
                "best": round(context.best_score, 4),
                "coverage": coverage,
                "gate": ok,
                "top_doc": context.chunks[0].doc_id if context.chunks else "-",
                "expected": ",".join(case["expected_doc_ids"]) or "-",
                "missing": ",".join(missing[:3]),
            }
        )

    print(f"corpus: {len(documents)} docs / {len(chunks)} chunks")
    print(f"{'id':<4}{'ans':<6}{'best':>8}{'cover':>8}{'gate':>6}  {'top':<8}{'expected':<12} missing")
    for row in rows:
        print(
            f"{row['id']:<4}{str(row['answerable']):<6}{row['best']:>8.4f}{row['coverage']:>8.3f}"
            f"{str(row['gate']):>6}  {row['top_doc']:<8}{row['expected']:<12}{row['missing']}"
        )

    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    print()
    print(f"answerable   best min={min(r['best'] for r in ans):.4f} coverage min={min(r['coverage'] for r in ans):.3f}")
    print(f"unanswerable best max={max(r['best'] for r in un):.4f} coverage max={max(r['coverage'] for r in un):.3f}")
    print(f"gate accuracy: {sum(1 for r in rows if r['gate'] == r['answerable'])}/{len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())



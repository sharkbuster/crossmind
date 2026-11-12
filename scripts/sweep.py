"""Chunk-size / alpha sweep: find the configuration with the widest score gap."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossmind.config import Settings  # noqa: E402
from crossmind.embedding import HashingEmbedder  # noqa: E402
from crossmind.rag.chunker import chunk_documents  # noqa: E402
from crossmind.rag.corpus import load_corpus  # noqa: E402
from crossmind.rag.index import HybridIndex  # noqa: E402
from crossmind.rag.retriever import Retriever  # noqa: E402


async def main() -> None:
    settings = Settings()
    data = ROOT / "data"
    documents = load_corpus(data / "corpus")
    cases = json.loads((data / "eval_set.json").read_text("utf-8"))["cases"]
    embedder = HashingEmbedder(settings.embedding_dim)

    print(f"{'size':>5}{'alpha':>7}{'chunks':>8}{'min_ans':>9}{'max_un':>9}{'gap':>7}{'hit':>6}")
    for size in (200, 240, 280, 320, 420):
        chunks = chunk_documents(documents, chunk_size=size, overlap=0.15)
        index = await HybridIndex.build(chunks, embedder)
        for alpha in (0.4, 0.5, 0.6, 0.7, 0.8):
            retriever = Retriever(index, embedder, settings)
            ans_best, un_best, hits = [], [], 0
            for case in cases:
                ctx = await retriever.retrieve(case["question"], threshold=0.0, alpha=alpha)
                docs = [c.doc_id for c in ctx.chunks[:5]]
                if case["answerable"]:
                    ans_best.append(ctx.best_score)
                    if any(d in docs for d in case["expected_doc_ids"]):
                        hits += 1
                else:
                    un_best.append(ctx.best_score)
            gap = min(ans_best) - max(un_best)
            print(
                f"{size:>5}{alpha:>7.1f}{len(chunks):>8}{min(ans_best):>9.4f}{max(un_best):>9.4f}"
                f"{gap:>7.4f}{hits:>4}/{len(ans_best)}"
            )


if __name__ == "__main__":
    asyncio.run(main())

"""Answerability-gate probe: IDF-weighted coverage over jieba words."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jieba  # noqa: E402

from crossmind.config import Settings  # noqa: E402
from crossmind.embedding import HashingEmbedder  # noqa: E402
from crossmind.rag.chunker import chunk_documents  # noqa: E402
from crossmind.rag.corpus import load_corpus  # noqa: E402
from crossmind.rag.index import HybridIndex  # noqa: E402
from crossmind.rag.retriever import Retriever  # noqa: E402

STOP = {
    "你们", "我们", "平台", "可以", "能否", "是否", "什么", "怎么", "多少", "多久", "哪些",
    "需要", "提供", "支持", "有", "没有", "吗", "呢", "的", "是", "在", "和", "与", "或",
    "请", "问", "一下", "如何", "我", "你", "他", "这", "那", "个", "能", "会", "要", "用",
    "时候", "时间", "几天", "多少天",
}


def words(text: str) -> list[str]:
    out = []
    for token in jieba.lcut(text.lower()):
        token = token.strip()
        if not token or token in STOP:
            continue
        if token.isdigit():
            continue
        if len(token) >= 2 or token.isascii():
            out.append(token)
    return out


def build_idf(texts: list[str]) -> tuple[dict[str, float], float]:
    df: Counter[str] = Counter()
    for text in texts:
        df.update(set(words(text)))
    n = len(texts)
    idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
    return idf, max(idf.values()) if idf else 1.0


def coverage(question: str, evidence: str, idf: dict[str, float], max_idf: float) -> tuple[float, list[str]]:
    q = words(question)
    if not q:
        return 1.0, []
    ev = set(words(evidence)) | {w for w in words(evidence)}
    total = 0.0
    got = 0.0
    missing = []
    for term in q:
        weight = idf.get(term, max_idf * 1.2)
        total += weight
        if term in ev:
            got += weight
        else:
            missing.append(term)
    return round(got / total, 4) if total else 1.0, missing


async def main() -> None:
    settings = Settings()
    data = ROOT / "data"
    documents = load_corpus(data / "corpus")
    chunks = chunk_documents(
        documents, chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap
    )
    texts = [f"{c.doc_title} {c.section} {c.text}" for c in chunks]
    idf, max_idf = build_idf(texts)
    embedder = HashingEmbedder(settings.embedding_dim)
    index = await HybridIndex.build(chunks, embedder)
    retriever = Retriever(index, embedder, settings)

    cases = json.loads((data / "eval_set.json").read_text("utf-8"))["cases"]
    ans, un = [], []
    print(f"{'id':<5}{'ans':<7}{'cov':>7}  missing")
    for case in cases:
        ctx = await retriever.retrieve(case["question"], threshold=0.0)
        evidence = " ".join(f"{c.doc_title} {c.section} {c.text}" for c in ctx.chunks[:3])
        cov, missing = coverage(case["question"], evidence, idf, max_idf)
        (ans if case["answerable"] else un).append((case["id"], cov))
        print(f"{case['id']:<5}{str(case['answerable']):<7}{cov:>7.3f}  {','.join(missing[:4])}")
    print()
    for t in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
        correct = sum(1 for _, c in ans if c >= t) + sum(1 for _, c in un if c < t)
        print(f"threshold {t:.2f}: gate accuracy {correct}/{len(ans) + len(un)}")
    print(f"answerable min={min(c for _, c in ans):.3f}  unanswerable max={max(c for _, c in un):.3f}")


if __name__ == "__main__":
    asyncio.run(main())

"""Answerability gate - refuses questions the knowledge base cannot support.

Semantic retrieval almost always returns *something*: a question about a service
the knowledge base does not offer still lands on the policy document that lists
adjacent services, and a similarity threshold cannot separate those cases because
the document really is topically close.

So we check the opposite direction: do the content words of the *question* occur
in the retrieved evidence?  "以旧换新", "门店", "学生" never appear anywhere in
the corpus, so no amount of fluent generation could be grounded - refusing is
the correct behaviour, and it is cheaper than calling the model.

Scoring is IDF weighted so that generic words ("多少", "可以") cannot mask a
missing entity, and out-of-vocabulary words are penalised hardest because they
are precisely the "this is not in our KB" signal.

This is a lexical entailment proxy, tuned to be permissive: it only refuses when
coverage collapses (default 0.30), i.e. it catches catastrophic misses and lets
paraphrases through.  Near-miss questions ("加拿大清关") still require a real
NLI/LLM judge - a limitation that is measured and reported in docs/EVALUATION.md
rather than hidden.
"""

from __future__ import annotations

import math
from collections import Counter

import jieba

from crossmind.domain.models import SourceChunk

STOPWORDS = {
    "你们", "我们", "平台", "可以", "能否", "是否", "什么", "怎么", "多少", "多久", "哪些",
    "需要", "提供", "支持", "有", "没有", "吗", "呢", "的", "是", "在", "和", "与", "或",
    "请", "问", "一下", "如何", "我", "你", "他", "这", "那", "个", "能", "会", "要", "用",
    "时候", "时间", "几天", "多少天", "一般", "大概",
}

_jieba_ready = False


def _ensure_jieba() -> None:
    global _jieba_ready
    if not _jieba_ready:
        jieba.setLogLevel("ERROR")
        jieba.initialize()
        _jieba_ready = True


def segment(text: str) -> list[str]:
    """Content words only: segmented, lower-cased, stop-worded, digits dropped."""
    _ensure_jieba()
    out: list[str] = []
    for token in jieba.lcut(text.lower()):
        token = token.strip()
        if not token or token in STOPWORDS or token.isdigit():
            continue
        if len(token) >= 2 or token.isascii():
            out.append(token)
    return out


def build_idf(texts: list[str]) -> tuple[dict[str, float], float]:
    df: Counter[str] = Counter()
    for text in texts:
        df.update(set(segment(text)))
    total = len(texts)
    idf = {term: math.log((total + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
    return idf, (max(idf.values()) if idf else 1.0)


class AnswerabilityGate:
    def __init__(
        self,
        threshold: float = 0.30,
        enabled: bool = True,
        idf: dict[str, float] | None = None,
    ) -> None:
        self.threshold = threshold
        self.enabled = enabled
        self.idf = idf or {}
        self.max_idf = max(self.idf.values()) if self.idf else 1.0
        self._evidence_cache: dict[str, set[str]] = {}

    def _evidence_terms(self, chunk: SourceChunk) -> set[str]:
        cached = self._evidence_cache.get(chunk.chunk_id)
        if cached is None:
            cached = set(segment(f"{chunk.doc_title} {chunk.section} {chunk.text}"))
            self._evidence_cache[chunk.chunk_id] = cached
        return cached

    def coverage(self, question: str, chunks: list[SourceChunk]) -> tuple[float, list[str]]:
        terms = segment(question)
        if not terms:
            return 1.0, []
        evidence: set[str] = set()
        for chunk in chunks:
            evidence |= self._evidence_terms(chunk)
        total = 0.0
        covered = 0.0
        missing: list[str] = []
        for term in terms:
            weight = self.idf.get(term, self.max_idf * 1.2)
            total += weight
            if term in evidence:
                covered += weight
            else:
                missing.append(term)
        return (round(covered / total, 4) if total else 1.0), missing

    def evaluate(self, question: str, chunks: list[SourceChunk]) -> tuple[bool, float, list[str]]:
        if not self.enabled:
            return True, 1.0, []
        score, missing = self.coverage(question, chunks)
        return score >= self.threshold, score, missing

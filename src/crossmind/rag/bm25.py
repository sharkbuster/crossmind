"""In-tree BM25 (Okapi) over the mixed CJK/Latin tokenizer.

`rank_bm25` would be a one-line dependency, but BM25 scoring is 40 lines and
keeping it here lets us expose the raw lexical score to the fusion layer and to
unit tests without pinning a third-party version.
"""

from __future__ import annotations

import math
from collections import Counter

from crossmind.embedding import tokenize


class BM25:
    def __init__(self, corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs = corpus
        self._doc_len = [len(doc) for doc in corpus]
        self._avg_len = (sum(self._doc_len) / len(corpus)) if corpus else 0.0
        self._tf: list[Counter[str]] = [Counter(doc) for doc in corpus]
        df: Counter[str] = Counter()
        for term_set in (set(doc) for doc in corpus):
            for term in term_set:
                df[term] += 1
        total = len(corpus)
        self._idf = {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        if not self._docs:
            return []
        q_terms = [t for t in query if t in self._idf]
        out: list[float] = []
        for index, tf in enumerate(self._tf):
            score = 0.0
            length = self._doc_len[index] or 1
            norm = self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1))
            for term in q_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                score += self._idf[term] * (freq * (self.k1 + 1)) / (freq + norm)
            out.append(score)
        return out

    def top_n(self, query: list[str], n: int) -> list[tuple[int, float]]:
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:n]


def saturation(score: float, k: float = 8.0) -> float:
    """Map an unbounded BM25 score into [0, 1) so it can be fused with cosine.

    Using min-max normalisation here was the classic bug: the best candidate
    always scored 1.0, so a similarity threshold could never filter anything.
    """
    if score <= 0:
        return 0.0
    return score / (score + k)


def tokenize_query(text: str) -> list[str]:
    return tokenize(text)

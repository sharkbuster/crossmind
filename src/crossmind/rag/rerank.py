"""Deterministic cross-encoder substitute.

A hosted reranker (``gte-rerank``) is an extra network hop, an extra key and an
extra failure mode; for the sizes this platform handles a lexical/semantic
fusion reranker is cheap, explainable and - crucially - testable.  The interface
matches a real cross-encoder, so swapping one in is a constructor change.
"""

from __future__ import annotations

from crossmind.domain.models import SourceChunk
from crossmind.embedding import tokenize


class HeuristicReranker:
    name = "heuristic-v1"

    def rerank(self, query: str, chunks: list[SourceChunk], top_k: int) -> list[SourceChunk]:
        if not chunks:
            return []
        query_tokens = set(tokenize(query))
        ordered: list[SourceChunk] = []
        for chunk in chunks:
            chunk_tokens = set(tokenize(f"{chunk.doc_title} {chunk.section} {chunk.text}"))
            if query_tokens:
                coverage = len(query_tokens & chunk_tokens) / len(query_tokens)
            else:
                coverage = 0.0
            title_tokens = set(tokenize(chunk.doc_title))
            title_hit = len(query_tokens & title_tokens) / len(title_tokens) if title_tokens else 0.0
            # Exact phrase hit is a strong signal for policy/型号 questions.
            phrase = 1.0 if query.strip() and query.strip() in chunk.text else 0.0
            final = 0.55 * chunk.score + 0.25 * coverage + 0.15 * title_hit + 0.05 * phrase
            ordered.append(
                chunk.model_copy(update={"score": round(final, 6), "rank": 0})
            )
        ordered.sort(key=lambda item: item.score, reverse=True)
        for rank, chunk in enumerate(ordered, start=1):
            chunk.rank = rank
        return ordered[:top_k]

"""Hybrid index: dense (hashed/remote embedding) + sparse (BM25).

Both modalities are scored for every chunk, then fused with a weighted sum.
Recall failure of one modality is compensated by the other, which is exactly the
SKU / model-number case where pure dense retrieval drifts to a similar product.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from crossmind.domain.models import SourceChunk
from crossmind.embedding import cosine, tokenize
from crossmind.rag.bm25 import BM25, saturation
from crossmind.rag.chunker import Chunk


@dataclass(slots=True)
class IndexStats:
    chunks: int
    documents: int
    dim: int
    vocab: int


@dataclass(slots=True)
class HybridIndex:
    chunks: list[Chunk]
    vectors: list[list[float]]
    bm25: BM25
    dim: int = 0
    attributes: dict[str, str] = field(default_factory=dict)

    @classmethod
    async def build(cls, chunks: list[Chunk], embedder: object, dim: int = 0) -> HybridIndex:
        texts = [f"{c.doc_title} {c.section} {c.text}" for c in chunks]
        vectors = await embedder.embed(texts)  # type: ignore[attr-defined]
        bm25 = BM25([tokenize(t) for t in texts])
        return cls(chunks=chunks, vectors=vectors, bm25=bm25, dim=len(vectors[0]) if vectors else dim)

    # ------------------------------------------------------------------ search
    def candidates(
        self,
        query: str,
        query_vector: list[float],
        *,
        recall_k: int,
        alpha: float = 0.6,
    ) -> list[SourceChunk]:
        if not self.chunks:
            return []
        lexical_raw = self.bm25.scores(tokenize(query))
        scored: list[SourceChunk] = []
        for position, chunk in enumerate(self.chunks):
            embed_score = max(0.0, cosine(query_vector, self.vectors[position]))
            lexical_score = saturation(lexical_raw[position])
            fused = alpha * embed_score + (1.0 - alpha) * lexical_score
            scored.append(
                SourceChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    section=chunk.section,
                    uri=chunk.uri,
                    text=chunk.text,
                    score=round(fused, 6),
                    embed_score=round(embed_score, 6),
                    lexical_score=round(lexical_score, 6),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(recall_k, 1)]

    def stats(self) -> IndexStats:
        return IndexStats(
            chunks=len(self.chunks),
            documents=len({c.doc_id for c in self.chunks}),
            dim=self.dim,
            vocab=len(self.bm25._idf),  # noqa: SLF001 - read-only introspection
        )

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        payload = {
            "dim": self.dim,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                    "doc_type": c.doc_type,
                    "section": c.section,
                    "uri": c.uri,
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "vector": v,
                }
                for c, v in zip(self.chunks, self.vectors, strict=True)
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> HybridIndex:
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks: list[Chunk] = []
        vectors: list[list[float]] = []
        for item in data["chunks"]:
            chunks.append(
                Chunk(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    doc_title=item["doc_title"],
                    doc_type=item.get("doc_type", "general"),
                    section=item.get("section", ""),
                    uri=item.get("uri", ""),
                    text=item["text"],
                    chunk_index=item.get("chunk_index", len(chunks)),
                )
            )
            vectors.append(item["vector"])
        return cls(
            chunks=chunks,
            vectors=vectors,
            bm25=BM25([tokenize(f"{c.doc_title} {c.section} {c.text}") for c in chunks]),
            dim=int(data.get("dim", 0)),
        )

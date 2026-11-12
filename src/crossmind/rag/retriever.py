"""Retrieval facade: embed -> hybrid recall -> threshold -> rerank."""

from __future__ import annotations

import time

from crossmind.config import Settings
from crossmind.domain.models import RetrievedContext, SourceChunk
from crossmind.obs.logging import get_logger
from crossmind.obs.metrics import RETRIEVAL_CALLS
from crossmind.obs.tracing import get_tracer
from crossmind.rag.index import HybridIndex
from crossmind.rag.rerank import HeuristicReranker

log = get_logger(__name__)


class Retriever:
    def __init__(
        self,
        index: HybridIndex,
        embedder: object,
        settings: Settings,
        reranker: HeuristicReranker | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker or HeuristicReranker()

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        alpha: float | None = None,
    ) -> RetrievedContext:
        started = time.perf_counter()
        top_k = top_k or self.settings.rag_top_k
        threshold = self.settings.rag_score_threshold if threshold is None else threshold
        alpha = self.settings.rag_embed_weight if alpha is None else alpha

        degraded = False
        reason = ""
        tracer = get_tracer()
        with tracer.span("retrieval.embed", query_chars=len(query)) as span:
            try:
                vectors = await self.embedder.embed([query])  # type: ignore[attr-defined]
                query_vector = vectors[0]
            except Exception as exc:  # noqa: BLE001 - embedding outage must degrade, not fail
                degraded = True
                reason = f"embedding_unavailable: {type(exc).__name__}"
                query_vector = []
                if span:
                    span.set(degraded=True, error=str(exc)[:200])
                log.warning("embedding unavailable, falling back to lexical retrieval", extra={"error": str(exc)})

        with tracer.span("retrieval.recall", alpha=alpha, recall_k=self.settings.rag_recall_k) as span:
            candidates = self.index.candidates(
                query, query_vector, recall_k=self.settings.rag_recall_k, alpha=alpha
            )
            best = candidates[0].score if candidates else 0.0
            if span:
                span.set(candidates=len(candidates), best_score=best)

        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "retrieval",
            extra={
                "query": query[:40],
                "candidates": len(candidates),
                "best": round(best, 4),
                "threshold": threshold,
                "degraded": degraded,
            },
        )
        if not candidates or best < threshold:
            RETRIEVAL_CALLS.inc(outcome="no_answer")
            return RetrievedContext(
                chunks=[],
                best_score=best,
                no_answer=True,
                strategy="lexical_only" if degraded else "hybrid",
                degraded=degraded,
                degradation_reason=reason,
                latency_ms=latency_ms,
            )

        with tracer.span("retrieval.rerank", top_k=top_k) as span:
            chunks: list[SourceChunk] = self.reranker.rerank(query, candidates, top_k)
            if span:
                span.set(returned=len(chunks))

        RETRIEVAL_CALLS.inc(outcome="ok")
        return RetrievedContext(
            chunks=chunks,
            best_score=best,
            no_answer=False,
            strategy="lexical_only" if degraded else "hybrid+rerank",
            degraded=degraded,
            degradation_reason=reason,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def search_docs(self, query: str, *, top_k: int = 5) -> list[SourceChunk]:
        context = await self.retrieve(query, top_k=top_k, threshold=0.0)
        return context.chunks

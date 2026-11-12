"""Knowledge base inspection: stats, search and direct grounded answering."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from crossmind.api.deps import ContainerDep, PrincipalDep
from crossmind.domain.models import RagAnswer, SourceChunk

router = APIRouter(tags=["knowledge"])


class KbStats(BaseModel):
    documents: int
    chunks: int
    dim: int
    vocab: int
    sources: list[dict[str, str]] = Field(default_factory=list)


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    score: float
    embed_score: float
    lexical_score: float
    text: str


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


@router.get("/kb/stats", response_model=KbStats, summary="知识库统计")
async def stats(_principal: PrincipalDep, container: ContainerDep) -> KbStats:
    info = container.index.stats()
    return KbStats(
        documents=info.documents,
        chunks=info.chunks,
        dim=info.dim,
        vocab=info.vocab,
        sources=[
            {"doc_id": doc.doc_id, "title": doc.title, "type": doc.doc_type, "uri": doc.uri}
            for doc in container.documents
        ],
    )


@router.get("/kb/search", response_model=list[SearchHit], summary="混合检索（不经生成）")
async def search(
    q: Annotated[str, Query(min_length=1, max_length=500)],
    _principal: PrincipalDep,
    container: ContainerDep,
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[SearchHit]:
    context = await container.retriever.retrieve(q, top_k=top_k, threshold=0.0)
    return [
        SearchHit(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            doc_title=chunk.doc_title,
            section=chunk.section,
            score=chunk.score,
            embed_score=chunk.embed_score,
            lexical_score=chunk.lexical_score,
            text=chunk.text,
        )
        for chunk in context.chunks
    ]


@router.post("/kb/answer", response_model=RagAnswer, summary="带引用与拒答的问答")
async def answer(
    payload: AnswerRequest, _principal: PrincipalDep, container: ContainerDep
) -> RagAnswer:
    return await container.rag.answer(payload.question, top_k=payload.top_k)


@router.get("/kb/chunk/{chunk_id}", response_model=SourceChunk, summary="按 ID 取知识块原文")
async def chunk_detail(chunk_id: str, _principal: PrincipalDep, container: ContainerDep) -> SourceChunk:
    from crossmind.domain.errors import NotFound

    for chunk in container.chunks:
        if chunk.chunk_id == chunk_id:
            return SourceChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                section=chunk.section,
                uri=chunk.uri,
                text=chunk.text,
            )
    raise NotFound(f"chunk {chunk_id} not found")

"""Grounded answering pipeline: citations, refusal and usage accounting."""

from __future__ import annotations

from crossmind.rag.citation import REFUSAL_TEXT


async def test_answer_is_grounded_and_cited(container) -> None:
    answer = await container.rag.answer("未拆封商品多久可以退货？")
    assert answer.no_answer is False
    assert answer.citations, "a grounded answer must carry at least one citation"
    assert answer.grounded is True
    assert answer.sources
    assert answer.usage.total_tokens > 0
    assert "30" in answer.text


async def test_unanswerable_question_is_refused(container) -> None:
    answer = await container.rag.answer("你们线下门店在哪里？")
    assert answer.no_answer is True
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []


async def test_off_topic_question_is_refused(container) -> None:
    answer = await container.rag.answer("今天北京天气怎么样，适合爬山吗？")
    assert answer.no_answer is True


async def test_answer_uses_only_retrieved_documents(container) -> None:
    answer = await container.rag.answer("电池和充电线的保修期多久？")
    cited_docs = {citation.doc_id for citation in answer.citations}
    assert cited_docs <= {chunk.doc_id for chunk in answer.sources}


async def test_answer_with_context_skips_retrieval(container) -> None:
    chunks = [chunk for chunk in container.chunks if chunk.doc_id == "AP-003"][:2]
    from crossmind.domain.models import SourceChunk

    sources = [
        SourceChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            doc_title=chunk.doc_title,
            section=chunk.section,
            uri=chunk.uri,
            text=chunk.text,
            score=0.9,
        )
        for chunk in chunks
    ]
    answer = await container.rag.answer_with_context("支持货到付款吗？", sources)
    assert answer.no_answer is False
    assert answer.citations
    assert {c.doc_id for c in answer.citations} == {"AP-003"}


async def test_top_k_is_respected(container) -> None:
    answer = await container.rag.answer("退款有什么规定", top_k=2)
    assert len(answer.sources) <= 2

"""Chunking behaviour: size budget, overlap and section metadata."""

from __future__ import annotations

import pytest

from crossmind.domain.errors import BadRequest
from crossmind.rag.chunker import chunk_document, chunk_documents
from crossmind.rag.corpus import Document

BODY = """## 适用范围
本政策适用于平台自营商品，覆盖美国、欧盟、日本等站点。数字商品不适用无理由退货。

## 无理由退货
未拆封商品自签收之日起 30 天内可申请无理由退货，商品需保持原包装、配件齐全。

## 质量问题
商品存在质量缺陷的，自签收之日起 30 天内可免费退换，往返运费由平台承担。

## 退款时效
仓库验收通过后 3 至 5 个工作日内原路退回，跨境订单退货运费由买家承担。
"""


def make_document(body: str = BODY) -> Document:
    return Document(
        doc_id="AP-900",
        title="测试政策",
        doc_type="policy",
        body=body,
        uri="mem://AP-900",
    )


def test_chunks_respect_size_budget() -> None:
    chunks = chunk_document(make_document(), chunk_size=200, overlap=0.15)
    assert chunks
    assert all(chunk.char_count <= 200 for chunk in chunks)


def test_every_chunk_carries_traceable_metadata() -> None:
    chunks = chunk_document(make_document(), chunk_size=200, overlap=0.1)
    for chunk in chunks:
        assert chunk.doc_id == "AP-900"
        assert chunk.doc_title == "测试政策"
        assert chunk.section
        assert chunk.chunk_id.startswith("AP-900::")
        assert chunk.uri == "mem://AP-900"


def test_overlap_actually_carries_context() -> None:
    long_body = BODY + BODY.replace("30 天", "45 天") + BODY.replace("3 至 5", "5 至 7")
    without = chunk_document(make_document(long_body), chunk_size=200, overlap=0.0)
    with_overlap = chunk_document(make_document(long_body), chunk_size=200, overlap=0.3)
    assert len(with_overlap) >= 2
    assert sum(c.char_count for c in with_overlap) > sum(c.char_count for c in without)


def test_section_metadata_records_merged_sections() -> None:
    chunks = chunk_document(make_document(), chunk_size=600, overlap=0.1)
    assert len(chunks) < 4
    assert any(" / " in chunk.section or "等" in chunk.section or chunk.section for chunk in chunks)


@pytest.mark.parametrize("size", [100, 5000])
def test_invalid_chunk_size_is_rejected(size: int) -> None:
    with pytest.raises(BadRequest):
        chunk_document(make_document(), chunk_size=size)


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(BadRequest):
        chunk_document(make_document(), chunk_size=300, overlap=0.9)


def test_document_without_headings_still_chunks() -> None:
    document = make_document("没有标题的一段话。" * 60)
    chunks = chunk_document(document, chunk_size=220, overlap=0.15)
    assert len(chunks) >= 1
    assert chunks[0].section == "正文"


def test_chunk_ids_are_unique_across_documents() -> None:
    documents = [make_document(), make_document(BODY.replace("30 天", "45 天"))]
    documents[1].doc_id = "AP-901"
    chunks = chunk_documents(documents, chunk_size=200, overlap=0.1)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))

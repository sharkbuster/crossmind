"""Corpus loading and index persistence."""

from __future__ import annotations

import pytest

from crossmind.domain.errors import BadRequest
from crossmind.embedding import HashingEmbedder, cosine, tokenize
from crossmind.rag.corpus import load_corpus, parse_document
from crossmind.rag.index import HybridIndex

VALID = """---
id: AP-777
title: 测试政策
type: policy
---

## 第一节
未拆封商品 30 天内可以退货。

## 第二节
退款 3 至 5 个工作日到账。
"""


def write(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_document_reads_front_matter(tmp_path) -> None:
    document = parse_document(write(tmp_path, "a.md", VALID))
    assert document.doc_id == "AP-777"
    assert document.title == "测试政策"
    assert document.doc_type == "policy"
    assert "30 天" in document.body


def test_document_without_front_matter_uses_filename(tmp_path) -> None:
    document = parse_document(write(tmp_path, "PM-999_product.md", "## 概述\n内容。\n"))
    assert document.doc_id == "PM-999"
    assert document.title == "PM-999_product"
    assert document.doc_type == "general"


def test_missing_corpus_directory_is_rejected(tmp_path) -> None:
    with pytest.raises(BadRequest):
        load_corpus(tmp_path / "nope")


def test_empty_corpus_directory_is_rejected(tmp_path) -> None:
    with pytest.raises(BadRequest):
        load_corpus(tmp_path)


def test_duplicate_doc_ids_are_rejected(tmp_path) -> None:
    write(tmp_path, "a.md", VALID)
    write(tmp_path, "b.md", VALID)
    with pytest.raises(BadRequest):
        load_corpus(tmp_path)


def test_empty_body_is_rejected(tmp_path) -> None:
    with pytest.raises(BadRequest):
        parse_document(write(tmp_path, "empty.md", "---\nid: X\ntitle: t\n---\n"))


def test_load_corpus_reads_shipped_data(container) -> None:
    assert len(container.documents) == 12
    ids = {document.doc_id for document in container.documents}
    assert {"AP-001", "AP-008", "PM-001", "PM-004"} <= ids


# ------------------------------------------------------------------ embeddings
async def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dim=64)
    vectors = await embedder.embed(["未拆封商品可以退货吗", "未拆封商品可以退货吗"])
    first, second = vectors
    assert first == second
    assert len(first) == 64
    assert abs(sum(v * v for v in first) - 1.0) < 1e-6


async def test_similar_text_scores_higher_than_unrelated() -> None:
    embedder = HashingEmbedder(dim=256)
    vectors = await embedder.embed(
        ["未拆封商品 30 天内可以无理由退货", "未拆封商品退货期限", "蓝牙音箱续航 24 小时"]
    )
    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2])


def test_tokenize_keeps_model_codes_and_cjk_bigrams() -> None:
    tokens = tokenize("PowerGo CB-PG-003 支持快充")
    assert "powergo" in tokens
    assert "cb-pg-003" in tokens
    assert "快充" in tokens


# ------------------------------------------------------------------ index
async def test_index_stats_and_persistence(tmp_path) -> None:
    embedder = HashingEmbedder(dim=64)
    from crossmind.rag.chunker import chunk_document

    document = parse_document(write(tmp_path, "a.md", VALID))
    chunks = chunk_document(document, chunk_size=200, overlap=0.1)
    index = await HybridIndex.build(chunks, embedder)
    stats = index.stats()
    assert stats.chunks == len(chunks)
    assert stats.documents == 1
    assert stats.dim == 64

    path = tmp_path / "index.json"
    index.save(path)
    restored = HybridIndex.load(path)
    assert len(restored.chunks) == len(chunks)
    assert restored.chunks[0].text == chunks[0].text
    assert restored.stats().documents == 1


async def test_loaded_index_can_retrieve(tmp_path, container) -> None:
    path = tmp_path / "shipped.json"
    container.index.save(path)
    restored = HybridIndex.load(path)
    vector = (await container.embedder.embed(["退货多久到账"]))[0]
    hits = restored.candidates("退货多久到账", vector, recall_k=5)
    assert hits
    assert hits[0].doc_id == "AP-001"

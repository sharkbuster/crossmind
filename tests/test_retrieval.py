"""Hybrid retrieval: recall quality, fusion and threshold semantics."""

from __future__ import annotations

import pytest

from crossmind.rag.bm25 import saturation


async def test_retrieval_hits_expected_document(container) -> None:
    context = await container.retriever.retrieve("标准专线一般几天能送到？")
    assert context.no_answer is False
    assert context.chunks[0].doc_id == "AP-002"


async def test_retrieval_handles_product_model_question(container) -> None:
    context = await container.retriever.retrieve("PowerGo 移动电源支持哪些快充协议？")
    assert context.chunks[0].doc_id == "PM-003"


async def test_high_threshold_turns_into_no_answer(container) -> None:
    context = await container.retriever.retrieve("退货运费谁承担？", threshold=0.99)
    assert context.no_answer is True
    assert context.chunks == []


async def test_retrieval_returns_ranked_unique_chunks(container) -> None:
    context = await container.retriever.retrieve("保修期多久", top_k=5)
    chunk_ids = [chunk.chunk_id for chunk in context.chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    scores = [chunk.score for chunk in context.chunks]
    assert scores == sorted(scores, reverse=True)
    assert [chunk.rank for chunk in context.chunks] == list(range(1, len(scores) + 1))


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
async def test_fusion_weights_are_accepted(container, alpha: float) -> None:
    context = await container.retriever.retrieve("积分怎么抵扣", alpha=alpha)
    assert context.chunks


async def test_scores_are_normalised(container) -> None:
    context = await container.retriever.retrieve("退款多久到账")
    for chunk in context.chunks:
        assert 0.0 <= chunk.embed_score <= 1.0
        assert 0.0 <= chunk.lexical_score <= 1.0
        assert 0.0 <= chunk.score <= 1.0


async def test_lexical_fusion_is_absolute_not_minmax(container) -> None:
    """Regression guard: min-max normalisation made the best hit always 1.0."""
    first = await container.retriever.retrieve("退换货政策", top_k=3)
    second = await container.retriever.retrieve("完全无关的问题内容占位", top_k=3)
    assert first.best_score < 1.0
    assert first.best_score > second.best_score


def test_saturation_is_monotonic_and_bounded() -> None:
    values = [saturation(x) for x in (0, 1, 5, 20, 100, 1000)]
    assert values[0] == 0.0
    assert values == sorted(values)
    assert all(0.0 <= value < 1.0 for value in values)

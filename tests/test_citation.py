"""Citation numbering, range validation and faithfulness scoring."""

from __future__ import annotations

from crossmind.domain.models import Citation, SourceChunk
from crossmind.rag.citation import (
    REFUSAL_TEXT,
    build_context_block,
    cited_indices,
    faithfulness,
    is_refusal,
    resolve_citations,
    strip_invalid_markers,
)

CHUNKS = [
    SourceChunk(
        chunk_id="AP-001::000::aaa",
        doc_id="AP-001",
        doc_title="退换货政策",
        section="无理由退货",
        text="未拆封商品自签收之日起 30 天内可申请无理由退货。",
        score=0.8,
    ),
    SourceChunk(
        chunk_id="AP-001::001::bbb",
        doc_id="AP-001",
        doc_title="退换货政策",
        section="退款时效",
        text="仓库验收通过后 3 至 5 个工作日内原路退回。",
        score=0.7,
    ),
]


def test_cited_indices_are_deduplicated_in_order() -> None:
    assert cited_indices("第一点【2】，第二点【1】，再次【2】") == [2, 1]


def test_resolve_citations_backfills_metadata() -> None:
    citations, invalid = resolve_citations("30 天内可退货【1】", CHUNKS)
    assert invalid == []
    assert len(citations) == 1
    assert citations[0].doc_id == "AP-001"
    assert citations[0].doc_title == "退换货政策"
    assert citations[0].section == "无理由退货"


def test_out_of_range_markers_are_rejected_and_stripped() -> None:
    citations, invalid = resolve_citations("编造来源【9】", CHUNKS)
    assert invalid == [9]
    assert citations == []
    assert "【9】" not in strip_invalid_markers("编造来源【9】", invalid)


def test_context_block_is_numbered_from_one() -> None:
    block = build_context_block(CHUNKS)
    assert block.startswith("[1] doc_id=AP-001")
    assert "[2] doc_id=AP-001" in block
    assert "title=退换货政策" in block


def test_faithfulness_is_high_for_verbatim_citation() -> None:
    citations, _ = resolve_citations("仓库验收通过后 3 至 5 个工作日内原路退回【2】", CHUNKS)
    assert faithfulness("仓库验收通过后 3 至 5 个工作日内原路退回【2】", citations) > 0.9


def test_faithfulness_drops_for_unsupported_claims() -> None:
    citations, _ = resolve_citations("支持空运到火星并且终身免费换新【1】", CHUNKS)
    assert faithfulness("支持空运到火星并且终身免费换新【1】", citations) < 0.2


def test_faithfulness_is_zero_without_citations() -> None:
    assert faithfulness("任意答案", []) == 0.0


def test_refusal_detection() -> None:
    assert is_refusal(REFUSAL_TEXT) is True
    assert is_refusal("未拆封商品 30 天内可退货。") is False


def test_citation_model_keeps_excerpt_short() -> None:
    long_chunk = SourceChunk(
        chunk_id="X",
        doc_id="X-1",
        doc_title="长文",
        text="很长的内容" * 100,
    )
    citations, _ = resolve_citations("引用【1】", [long_chunk])
    assert len(citations[0].text) <= 240
    assert isinstance(citations[0], Citation)

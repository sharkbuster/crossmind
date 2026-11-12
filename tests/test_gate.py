"""Answerability gate: refuses questions the corpus cannot support."""

from __future__ import annotations

from crossmind.rag.answerability import AnswerabilityGate, build_idf, segment


def test_segment_drops_stopwords_and_digits() -> None:
    tokens = segment("请问你们支持多少种付款方式？")
    assert "你们" not in tokens
    assert "多少" not in tokens
    assert "付款" in tokens


def test_gate_accepts_question_covered_by_evidence(container) -> None:
    gate = AnswerabilityGate(threshold=0.30, idf=_idf(container))
    chunks = container.chunks
    covered = [chunk_to_source(chunk) for chunk in chunks if chunk.doc_id == "AP-001"]
    ok, coverage, _missing = gate.evaluate("未拆封商品多久可以退货？", covered)
    assert ok is True
    assert coverage >= 0.3


def test_gate_refuses_missing_entity(container) -> None:
    gate = AnswerabilityGate(threshold=0.30, idf=_idf(container))
    covered = [chunk_to_source(chunk) for chunk in container.chunks if chunk.doc_id == "AP-001"]
    ok, coverage, missing = gate.evaluate("线下门店在哪里？", covered)
    assert ok is False
    assert coverage < 0.3
    assert missing


def test_gate_can_be_disabled() -> None:
    gate = AnswerabilityGate(threshold=0.99, enabled=False)
    ok, coverage, _ = gate.evaluate("完全无关", [])
    assert ok is True
    assert coverage == 1.0


def test_empty_question_is_treated_as_answerable() -> None:
    gate = AnswerabilityGate(threshold=0.5)
    ok, coverage, _ = gate.evaluate("？？？", [])
    assert ok is True
    assert coverage == 1.0


def _idf(container) -> dict[str, float]:
    return build_idf([f"{c.doc_title} {c.section} {c.text}" for c in container.chunks])[0]


def chunk_to_source(chunk):
    from crossmind.domain.models import SourceChunk

    return SourceChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        section=chunk.section,
        uri=chunk.uri,
        text=chunk.text,
    )

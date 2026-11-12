"""Citation handling: numbering, back-filling and validation.

The model is only allowed to *choose* a number.  Everything a user sees next to
that number (document id, title, section, score) is filled in by the system, and
out-of-range markers are removed.  This turns "traceable" from a prompt promise
into a checked property.
"""

from __future__ import annotations

import re

from crossmind.domain.models import Citation, SourceChunk

CITATION_RE = re.compile(r"【\s*(\d{1,2})\s*】")

REFUSAL_TEXT = (
    "抱歉，我在知识库中没有找到能回答这个问题的依据。为避免给你错误信息，"
    "我已经把这个问题转给人工客服，请稍等。"
)

REFUSAL_HINTS = ("没有找到", "暂无相关", "无法回答", "转人工", "没有依据")


def is_refusal(text: str) -> bool:
    return any(hint in text for hint in REFUSAL_HINTS)


def build_context_block(chunks: list[SourceChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{index}] doc_id={chunk.doc_id} | title={chunk.doc_title} | section={chunk.section}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def cited_indices(answer: str) -> list[int]:
    seen: list[int] = []
    for match in CITATION_RE.finditer(answer):
        value = int(match.group(1))
        if value not in seen:
            seen.append(value)
    return seen


def resolve_citations(answer: str, chunks: list[SourceChunk]) -> tuple[list[Citation], list[int]]:
    citations: list[Citation] = []
    invalid: list[int] = []
    for index in cited_indices(answer):
        if 1 <= index <= len(chunks):
            chunk = chunks[index - 1]
            citations.append(
                Citation(
                    index=index,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    section=chunk.section,
                    score=chunk.score,
                    text=chunk.text[:240],
                )
            )
        else:
            invalid.append(index)
    return citations, invalid


def strip_invalid_markers(answer: str, invalid: list[int]) -> str:
    cleaned = answer
    for index in invalid:
        cleaned = re.sub(rf"【\s*{index}\s*】", "", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def ngrams(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n:
        return {compact}
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def faithfulness(answer: str, citations: list[Citation]) -> float:
    """Share of the answer's character 3-grams that appear in the cited chunks.

    Character n-grams are language agnostic and cannot be gamed by paraphrasing
    a hallucination into familiar words: unsupported sentences simply do not
    contribute overlapping grams.
    """
    body = CITATION_RE.sub("", answer).strip()
    if not body:
        return 0.0
    if not citations:
        return 0.0
    support = " ".join(c.text for c in citations)
    support_grams = ngrams(support)
    body_grams = ngrams(body)
    if not body_grams:
        return 0.0
    return round(len(body_grams & support_grams) / len(body_grams), 4)

"""Section-aware chunking.

Design constraints that matter for retrieval quality:

* split on Markdown headings first, so a chunk never mixes two topics;
* pack consecutive sections until the target size is reached, because a single
  policy section is usually far shorter than the target and packing keeps the
  chunk inside the useful size band;
* carry a tail overlap into the next chunk, but never let overlap push a chunk
  past the size budget - truncation by the model is worse than less context.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from crossmind.domain.errors import BadRequest
from crossmind.rag.corpus import Document

HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?")

MIN_SIZE = 200
MAX_SIZE = 1200


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    section: str
    uri: str
    text: str
    chunk_index: int

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class _Section:
    title: str
    body: str


def _sections(body: str) -> list[_Section]:
    sections: list[_Section] = []
    current_title = "正文"
    buffer: list[str] = []
    for line in body.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            if buffer and "".join(buffer).strip():
                sections.append(_Section(current_title, "\n".join(buffer).strip()))
            current_title = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    if buffer and "".join(buffer).strip():
        sections.append(_Section(current_title, "\n".join(buffer).strip()))
    if not sections:
        sections.append(_Section("正文", body.strip()))
    return sections


def _split_long(text: str, size: int) -> list[str]:
    pieces: list[str] = []
    for sentence in SENTENCE_RE.findall(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= size:
            pieces.append(sentence)
            continue
        for i in range(0, len(sentence), size):
            pieces.append(sentence[i : i + size])
    return pieces or [text[:size]]


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 420,
    overlap: float = 0.15,
) -> list[Chunk]:
    if not MIN_SIZE <= chunk_size <= MAX_SIZE:
        raise BadRequest(f"chunk_size must be within [{MIN_SIZE}, {MAX_SIZE}], got {chunk_size}")
    if not 0.0 <= overlap <= 0.5:
        raise BadRequest(f"overlap must be within [0, 0.5], got {overlap}")

    overlap_chars = int(chunk_size * overlap)
    chunks: list[Chunk] = []
    buffer = ""
    buffer_sections: list[str] = []

    def flush() -> None:
        nonlocal buffer, buffer_sections
        text = buffer.strip()
        if not text:
            buffer, buffer_sections = "", []
            return
        if len(buffer_sections) == 1:
            section = buffer_sections[0]
        elif len(buffer_sections) <= 3:
            section = " / ".join(buffer_sections)
        else:
            section = f"{buffer_sections[0]} 等 {len(buffer_sections)} 节"
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=6).hexdigest()
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}::{len(chunks):03d}::{digest}",
                doc_id=document.doc_id,
                doc_title=document.title,
                doc_type=document.doc_type,
                section=section,
                uri=document.uri,
                text=text,
                chunk_index=len(chunks),
            )
        )
        tail = text[-overlap_chars:] if overlap_chars else ""
        buffer = tail
        buffer_sections = []

    for section in _sections(document.body):
        for paragraph in _split_long(section.body, chunk_size):
            if len(buffer) + len(paragraph) + 1 <= chunk_size:
                buffer = f"{buffer}\n{paragraph}".strip()
                if not buffer_sections or buffer_sections[-1] != section.title:
                    buffer_sections.append(section.title)
                continue
            flush()
            if len(paragraph) > chunk_size - len(buffer):
                buffer = ""
                buffer_sections = []
            buffer = f"{buffer}\n{paragraph}".strip()
            if not buffer_sections or buffer_sections[-1] != section.title:
                buffer_sections.append(section.title)
        if len(buffer) >= chunk_size * 0.9:
            flush()
    flush()

    # Final pass: drop the overlap-only tail artefact and dedupe ids.
    result = [c for c in chunks if len(c.text) > 0]
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in result:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        unique.append(chunk)
    return unique


def chunk_documents(
    documents: list[Document], *, chunk_size: int = 420, overlap: float = 0.15
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks

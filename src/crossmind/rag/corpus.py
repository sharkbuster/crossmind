"""Corpus loading with a tiny front-matter format.

Each document is a Markdown file::

    ---
    id: AP-001
    title: 退换货政策
    type: policy
    ---
    ## 适用范围
    ...

Keeping the metadata in the file (rather than in a manifest) means a document
and its identity can never drift apart, and adding a document needs no code
change - which is what makes the incremental-ingest story credible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from crossmind.domain.errors import BadRequest


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    doc_type: str
    body: str
    uri: str
    metadata: dict[str, str] = field(default_factory=dict)


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - authoring error
        raise BadRequest(f"invalid front-matter: {exc}") from exc
    if not isinstance(meta, dict):
        raise BadRequest("front-matter must be a mapping")
    return {str(k): str(v) for k, v in meta.items()}, parts[2].lstrip("\n")


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = _split_front_matter(raw)
    doc_id = meta.get("id") or path.stem.split("_")[0]
    title = meta.get("title") or path.stem
    doc_type = meta.get("type") or "general"
    if not body.strip():
        raise BadRequest(f"document {path.name} has an empty body")
    return Document(
        doc_id=doc_id,
        title=title,
        doc_type=doc_type,
        body=body,
        uri=path.as_posix(),
        metadata=meta,
    )


def load_corpus(directory: Path, *, pattern: str = "*.md") -> list[Document]:
    if not directory.exists():
        raise BadRequest(f"corpus directory not found: {directory}")
    documents = [parse_document(p) for p in sorted(directory.glob(pattern))]
    if not documents:
        raise BadRequest(f"corpus directory {directory} contains no {pattern} files")
    seen: set[str] = set()
    for doc in documents:
        if doc.doc_id in seen:
            raise BadRequest(f"duplicate doc_id {doc.doc_id!r} in corpus")
        seen.add(doc.doc_id)
    return documents

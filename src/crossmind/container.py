"""Composition root.

Everything the app needs is built once at startup and injected explicitly - there
is no global mutable singleton hiding behind an import, which is what makes the
tests able to spin up an isolated platform per test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crossmind.config import Settings, get_settings
from crossmind.domain.models import Product
from crossmind.embedding import HashingEmbedder, RemoteEmbedder
from crossmind.llm.base import ChatModel
from crossmind.llm.router import build_chat_model
from crossmind.obs.tracing import Tracer, configure_tracer
from crossmind.rag.chunker import Chunk, chunk_documents
from crossmind.rag.corpus import Document, load_corpus
from crossmind.rag.evaluator import evaluate
from crossmind.rag.index import HybridIndex
from crossmind.rag.pipeline import RagPipeline
from crossmind.rag.retriever import Retriever
from crossmind.security.auth import ApiKeyAuthenticator
from crossmind.security.guard import InputGuard
from crossmind.security.ratelimit import TokenBucketLimiter
from crossmind.services.catalog import Catalog
from crossmind.storage.db import Store

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


@dataclass
class Container:
    settings: Settings
    tracer: Tracer
    store: Store
    auth: ApiKeyAuthenticator
    limiter: TokenBucketLimiter
    guard: InputGuard
    chat_model: ChatModel
    fallback_model: ChatModel
    embedder: Any
    documents: list[Document]
    chunks: list[Chunk]
    index: HybridIndex
    retriever: Retriever
    rag: RagPipeline
    catalog: Catalog
    graph: Any = None
    products: list[Product] = field(default_factory=list)

    @property
    def corpus_stats(self) -> dict[str, Any]:
        stats = self.index.stats()
        return {
            "documents": stats.documents,
            "chunks": stats.chunks,
            "dim": stats.dim,
            "vocab": stats.vocab,
        }

    async def evaluate(self, *, mode: str = "offline"):
        from crossmind.domain.models import EvalCase
        import json

        payload = json.loads((resolve_path(self.settings.data_dir) / "eval_set.json").read_text("utf-8"))
        cases = [EvalCase(**item) for item in payload["cases"]]
        return await evaluate(self.rag, cases, mode=mode)

    async def aclose(self) -> None:
        for model in (self.chat_model, self.fallback_model):
            close = getattr(model, "aclose", None)
            if close is not None:
                await close()
        self.store.close()


async def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    settings.ensure_dirs()
    tracer = configure_tracer(settings.trace_dir)
    data_dir = resolve_path(settings.data_dir)

    store = Store(resolve_path(settings.db_path))
    auth = ApiKeyAuthenticator(settings.api_key_map)
    limiter = TokenBucketLimiter(settings.rate_limit_rps, settings.rate_limit_burst)
    guard = InputGuard()

    chat_model, fallback = build_chat_model(settings)
    if settings.embedding_provider == "hash":
        embedder: Any = HashingEmbedder(dim=settings.embedding_dim)
    else:
        embedder = RemoteEmbedder(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model="text-embedding-v4",
        )

    documents = load_corpus(data_dir / "corpus")
    chunks = chunk_documents(
        documents,
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap,
    )
    index = await HybridIndex.build(chunks, embedder, dim=settings.embedding_dim)
    retriever = Retriever(index, embedder, settings)
    from crossmind.rag.answerability import AnswerabilityGate, build_idf

    idf, _ = build_idf([f"{c.doc_title} {c.section} {c.text}" for c in chunks])
    gate = AnswerabilityGate(
        threshold=settings.rag_answerability_threshold,
        enabled=settings.rag_answerability_gate,
        idf=idf,
    )
    rag = RagPipeline(retriever, chat_model, settings, gate=gate)
    catalog = Catalog(data_dir / "catalog.json")

    container = Container(
        settings=settings,
        tracer=tracer,
        store=store,
        auth=auth,
        limiter=limiter,
        guard=guard,
        chat_model=chat_model,
        fallback_model=fallback,
        embedder=embedder,
        documents=documents,
        chunks=chunks,
        index=index,
        retriever=retriever,
        rag=rag,
        catalog=catalog,
        products=catalog.products,
    )

    from crossmind.agents.graph import build_graph

    container.graph = build_graph(container)
    return container

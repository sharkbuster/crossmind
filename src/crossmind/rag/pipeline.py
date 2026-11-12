"""The grounded answer pipeline: retrieve -> generate -> verify -> cite.

Order matters: the refusal decision is taken *before* the model is called when
retrieval already knows there is nothing to stand on, which both removes a
hallucination path and saves a paid call.
"""

from __future__ import annotations

import time

from crossmind.config import Settings
from crossmind.domain.models import Citation, RagAnswer, RetrievedContext, SourceChunk, Usage
from crossmind.llm.base import ChatModel, Message
from crossmind.obs.logging import get_logger
from crossmind.obs.tracing import get_tracer
from crossmind.rag.citation import (
    REFUSAL_TEXT,
    cited_indices,
    faithfulness,
    is_refusal,
    resolve_citations,
    strip_invalid_markers,
)
from crossmind.rag.answerability import AnswerabilityGate
from crossmind.rag.prompt import build_support_messages
from crossmind.rag.retriever import Retriever

log = get_logger(__name__)


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        chat_model: ChatModel,
        settings: Settings,
        gate: AnswerabilityGate | None = None,
    ) -> None:
        self.retriever = retriever
        self.chat_model = chat_model
        self.settings = settings
        self.gate = gate or AnswerabilityGate(
            threshold=settings.rag_answerability_threshold,
            enabled=settings.rag_answerability_gate,
        )

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        conversation: list[Message] | None = None,
        locale: str = "zh-CN",
        require_citations: bool = True,
    ) -> RagAnswer:
        started = time.perf_counter()

        context: RetrievedContext = await self.retriever.retrieve(question, top_k=top_k)

        if context.no_answer:
            return RagAnswer(
                text=REFUSAL_TEXT,
                citations=[],
                sources=[],
                grounded=False,
                no_answer=True,
                warnings=["retrieval_below_threshold"],
                model=self.chat_model.name,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        answerable, score, missing = self.gate.evaluate(question, context.chunks)
        if not answerable:
            log.info("answerability gate refused", extra={"coverage": score, "missing": missing[:6]})
            return RagAnswer(
                text=REFUSAL_TEXT,
                citations=[],
                sources=context.chunks,
                grounded=False,
                no_answer=True,
                warnings=["answerability_gate", f"coverage={score}"],
                model=self.chat_model.name,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        return await self.answer_with_context(
            question,
            context.chunks,
            conversation=conversation,
            locale=locale,
            require_citations=require_citations,
            started=started,
        )

    async def answer_with_context(
        self,
        question: str,
        chunks: list[SourceChunk],
        *,
        conversation: list[Message] | None = None,
        locale: str = "zh-CN",
        require_citations: bool = True,
        started: float | None = None,
    ) -> RagAnswer:
        started = started if started is not None else time.perf_counter()
        tracer = get_tracer()
        warnings: list[str] = []

        messages = build_support_messages(question, chunks, locale=locale, conversation=conversation)
        with tracer.span("rag.generate", model=self.chat_model.name, chunks=len(chunks)) as span:
            result = await self.chat_model.complete(messages, temperature=0.1, max_tokens=600)
            if span:
                span.set(tokens=result.usage.total_tokens)

        answer_text = result.text.strip()
        citations, invalid = resolve_citations(answer_text, chunks)

        if invalid:
            answer_text = strip_invalid_markers(answer_text, invalid)
            warnings.append(f"citation_out_of_range:{invalid}")
            log.warning("model cited out-of-range chunks", extra={"invalid": invalid})
            citations = [c for c in citations if c.index not in invalid]

        if is_refusal(answer_text) and not citations:
            return RagAnswer(
                text=REFUSAL_TEXT,
                citations=[],
                sources=chunks,
                grounded=False,
                no_answer=True,
                warnings=warnings + ["model_refused"],
                usage=result.usage,
                model=result.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        if require_citations and not citations:
            # An answer with no citation is unverifiable, i.e. indistinguishable
            # from a hallucination. Refuse rather than ship it.
            return RagAnswer(
                text=REFUSAL_TEXT,
                citations=[],
                sources=chunks,
                grounded=False,
                no_answer=True,
                warnings=warnings + ["ungrounded_answer_blocked"],
                usage=result.usage,
                model=result.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        grounded = bool(citations) and faithfulness(answer_text, citations) > 0.5
        if not grounded:
            warnings.append("low_faithfulness")

        return RagAnswer(
            text=answer_text,
            citations=citations,
            sources=chunks,
            grounded=grounded,
            no_answer=False,
            warnings=warnings,
            usage=result.usage,
            model=result.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def probe_context(self, answer_text: str, chunks: list[object]) -> list[int]:
        return cited_indices(answer_text)

    @staticmethod
    def refusal_answer(reason: str) -> RagAnswer:
        return RagAnswer(
            text=REFUSAL_TEXT, no_answer=True, grounded=False, warnings=[reason]
        )


def summarize_usage(answers: list[RagAnswer]) -> Usage:
    total = Usage()
    for answer in answers:
        total = total + answer.usage
    return total


__all__ = ["RagPipeline", "Citation", "summarize_usage"]


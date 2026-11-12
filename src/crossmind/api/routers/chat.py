"""Chat endpoints: sync answer, SSE stream, idempotent replay."""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from crossmind.api.deps import ChatPrincipalDep, ContainerDep
from crossmind.domain.models import ChatRequest, ChatResponse
from crossmind.services.chat import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, summary="同步对话（多智能体编排）")
async def chat(
    payload: ChatRequest,
    principal: ChatPrincipalDep,
    container: ContainerDep,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=64)] = None,
) -> ChatResponse:
    service = ChatService(container)
    return await service.chat(principal, payload, idempotency_key=idempotency_key)


@router.post("/stream", summary="流式对话（SSE：step / delta / done）")
async def chat_stream(
    payload: ChatRequest,
    principal: ChatPrincipalDep,
    container: ContainerDep,
) -> StreamingResponse:
    service = ChatService(container)

    async def event_source() -> Any:
        async for event in service.stream(principal, payload):
            data = json.dumps(event["data"], ensure_ascii=False)
            yield f"event: {event['event']}\ndata: {data}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

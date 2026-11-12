"""Chat orchestration: guard -> graph -> persist -> respond.

The service owns everything that must happen exactly once per request (task
record, memory write, audit entry, idempotent replay).  Nodes own the reasoning.
That split is what lets the graph be replayed in tests without touching storage.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from crossmind.agents.graph import resume_after_approval
from crossmind.agents.memory import ConversationMemory
from crossmind.agents.state import AgentState, new_state
from crossmind.container import Container
from crossmind.domain.errors import BadRequest, Conflict, Forbidden, GuardBlocked, NotFound
from crossmind.domain.models import (
    AgentStep,
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
    Citation,
    TaskRecord,
    TaskStatus,
    TaskType,
    now_iso,
)
from crossmind.obs.logging import bind_tenant, get_logger
from crossmind.obs.metrics import ACTIVE_TASKS, CACHE_HITS, HITL_TASKS
from crossmind.obs.tracing import current_trace_id, get_tracer, new_id
from crossmind.security.auth import Principal

log = get_logger(__name__)

INTENT_TO_TASK = {
    "support": TaskType.QA,
    "copywriting": TaskType.COPYWRITING,
    "translation": TaskType.TRANSLATION,
    "selection": TaskType.SELECTION,
}


def _to_steps(raw: list[dict[str, Any]]) -> list[AgentStep]:
    return [
        AgentStep(
            node=item.get("node", "?"),
            status=item.get("status", "ok"),
            detail=str(item.get("detail", "")),
            latency_ms=int(item.get("latency_ms", 0) or 0),
            tokens=int(item.get("tokens", 0) or 0),
        )
        for item in raw
    ]


def _to_citations(payload: list[dict[str, Any]] | None) -> list[Citation]:
    return [Citation(**item) for item in (payload or [])[:10]]


class ChatService:
    def __init__(self, container: Container) -> None:
        self.c = container
        self.memory = ConversationMemory(container.store)

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        principal: Principal,
        request: ChatRequest,
        *,
        idempotency_key: str | None = None,
    ) -> ChatResponse:
        bind_tenant(principal.tenant_id)
        if idempotency_key:
            cached = self.c.store.get_idempotent(idempotency_key, principal.tenant_id)
            if cached is not None:
                CACHE_HITS.inc()
                log.info("idempotent replay", extra={"key": idempotency_key})
                return ChatResponse(**cached)

        guard = self.c.guard.inspect(request.message)
        if guard.blocked:
            self.c.store.audit(
                principal.tenant_id, "guard.blocked", "chat", guard.reason, current_trace_id()
            )
            raise GuardBlocked(
                "请求被安全策略拦截", details={"reason": guard.reason, "redactions": guard.redactions}
            )

        session_id = request.session_id or new_id()
        task_id = new_id()
        history = self.memory.history(principal.tenant_id, session_id)
        state = new_state(
            question=guard.text,
            tenant_id=principal.tenant_id,
            session_id=session_id,
            task_id=task_id,
            trace_id=current_trace_id() or new_id(),
            locale=request.locale,
            channel=str(request.channel),
            product_sku=request.product_sku,
            require_citations=request.require_citations,
        )
        state["history"] = history  # type: ignore[typeddict-unknown-key]

        task = TaskRecord(
            id=task_id,
            tenant_id=principal.tenant_id,
            type=TaskType.QA,
            status=TaskStatus.RUNNING,
            input=request.model_dump(mode="json"),
            trace_id=state["trace_id"],
        )
        self.c.store.create_task(task)
        ACTIVE_TASKS.set(1.0, status="running")
        started = time.perf_counter()
        try:
            final_state: AgentState = await self.c.graph.ainvoke(state)
        except Exception as exc:
            self.c.store.update_task(task_id, status=TaskStatus.FAILED, error=str(exc)[:500])
            ACTIVE_TASKS.set(0.0, status="running")
            log.exception("agent graph failed")
            raise
        finally:
            ACTIVE_TASKS.set(0.0, status="running")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        status = TaskStatus(final_state.get("status", "succeeded"))
        intent = final_state.get("intent", "support")
        response = ChatResponse(
            task_id=task_id,
            trace_id=state["trace_id"],
            session_id=session_id,
            intent=intent,  # type: ignore[arg-type]
            status=status,
            answer=final_state.get("draft", ""),
            citations=_to_citations(final_state.get("citations")),
            sources=_to_citations(
                [
                    {
                        "index": index,
                        "chunk_id": chunk.get("chunk_id", ""),
                        "doc_id": chunk.get("doc_id", ""),
                        "doc_title": chunk.get("doc_title", ""),
                        "section": chunk.get("section", ""),
                        "score": float(chunk.get("score", 0.0)),
                        "text": str(chunk.get("text", ""))[:240],
                    }
                    for index, chunk in enumerate(final_state.get("contexts") or [], start=1)
                ]
            ),
            steps=_to_steps(final_state.get("steps") or []),
            usage=final_state.get("usage") or {},  # type: ignore[arg-type]
            latency_ms=elapsed_ms,
            requires_approval=bool(final_state.get("requires_approval")),
            approval_reason=str(final_state.get("approval_reason", "")),
            warnings=list(final_state.get("warnings") or []),
        )

        task_type = INTENT_TO_TASK.get(intent, TaskType.QA)
        output = response.model_dump(mode="json")
        if response.requires_approval:
            output["state"] = _jsonable_state(final_state)
        self.c.store.update_task(
            task_id,
            status=status,
            output=output,
        )
        # Task type is derived from the routed intent, so fix it up after routing.
        self.c.store.audit(
            principal.tenant_id,
            "chat.completed",
            f"task:{task_id}",
            f"intent={intent} status={status} type={task_type}",
            state["trace_id"],
        )
        self.memory.append(principal.tenant_id, session_id, "user", guard.text)
        if response.answer:
            self.memory.append(principal.tenant_id, session_id, "assistant", response.answer)
        if response.requires_approval:
            HITL_TASKS.inc(decision="pending")

        if idempotency_key:
            self.c.store.put_idempotent(idempotency_key, principal.tenant_id, output)
        return response

    # ------------------------------------------------------------------ streaming
    async def stream(
        self, principal: Principal, request: ChatRequest
    ) -> AsyncIterator[dict[str, Any]]:
        bind_tenant(principal.tenant_id)
        guard = self.c.guard.inspect(request.message)
        if guard.blocked:
            raise GuardBlocked("请求被安全策略拦截", details={"reason": guard.reason})
        session_id = request.session_id or new_id()
        task_id = new_id()
        state = new_state(
            question=guard.text,
            tenant_id=principal.tenant_id,
            session_id=session_id,
            task_id=task_id,
            trace_id=current_trace_id() or new_id(),
            locale=request.locale,
            channel=str(request.channel),
            product_sku=request.product_sku,
            require_citations=request.require_citations,
        )
        state["history"] = self.memory.history(principal.tenant_id, session_id)  # type: ignore[typeddict-unknown-key]

        yield {"event": "start", "data": {"task_id": task_id, "session_id": session_id}}

        final_state: AgentState = AgentState(**state)  # type: ignore[typeddict-item]
        try:
            async for update in self.c.graph.astream(state, stream_mode="updates"):
                for node_name, partial in update.items():
                    if not partial:
                        continue
                    final_state.update(partial)  # type: ignore[typeddict-item]
                    steps = partial.get("steps") or []
                    if steps:
                        yield {
                            "event": "step",
                            "data": {
                                "node": node_name,
                                "detail": steps[-1].get("detail", ""),
                                "status": steps[-1].get("status", "ok"),
                            },
                        }
        except Exception as exc:
            log.exception("stream failed")
            yield {"event": "error", "data": {"message": str(exc)[:300]}}
            return

        answer = final_state.get("draft", "")
        for index in range(0, len(answer), 24):
            yield {"event": "delta", "data": {"text": answer[index : index + 24]}}

        payload = {
            "task_id": task_id,
            "session_id": session_id,
            "trace_id": state["trace_id"],
            "intent": final_state.get("intent", "support"),
            "status": final_state.get("status", "succeeded"),
            "answer": answer,
            "citations": final_state.get("citations") or [],
            "requires_approval": bool(final_state.get("requires_approval")),
            "approval_reason": final_state.get("approval_reason", ""),
            "steps": final_state.get("steps") or [],
            "usage": final_state.get("usage") or {},
        }
        self.memory.append(principal.tenant_id, session_id, "user", guard.text)
        if answer:
            self.memory.append(principal.tenant_id, session_id, "assistant", answer)
        yield {"event": "done", "data": payload}

    # ------------------------------------------------------------------ approvals
    async def approve(
        self, principal: Principal, task_id: str, request: ApprovalRequest
    ) -> ChatResponse:
        bind_tenant(principal.tenant_id)
        task = self.c.store.get_task(task_id)
        if task is None:
            raise NotFound(f"task {task_id} not found")
        if task.tenant_id != principal.tenant_id:
            raise Forbidden("task belongs to another tenant")
        if task.status != TaskStatus.AWAITING_APPROVAL:
            raise Conflict(f"task is {task.status}, not awaiting approval")
        raw_state = task.output.get("state")
        if not raw_state:
            raise BadRequest("task has no stored state to resume")

        self.c.store.save_approval(
            task_id, str(request.decision), request.reviewer, request.comment
        )
        resumed = await resume_after_approval(
            self.c, AgentState(**raw_state), decision=str(request.decision)
        )
        status = TaskStatus(resumed.get("status", "succeeded"))
        self.c.store.update_task(
            task_id,
            status=status,
            output={"answer": resumed.get("draft", ""), "state": _jsonable_state(resumed)},
        )
        self.c.store.audit(
            principal.tenant_id,
            "task.approved" if status == TaskStatus.SUCCEEDED else "task.rejected",
            f"task:{task_id}",
            f"reviewer={request.reviewer} comment={request.comment}",
            task.trace_id,
        )
        HITL_TASKS.inc(decision=str(request.decision))
        return ChatResponse(
            task_id=task_id,
            trace_id=task.trace_id,
            session_id=str(raw_state.get("session_id", "")),
            intent=str(raw_state.get("intent", "support")),  # type: ignore[arg-type]
            status=status,
            answer=str(resumed.get("draft", "")),
            citations=_to_citations(resumed.get("citations")),
            steps=_to_steps(resumed.get("steps") or []),
            usage=resumed.get("usage") or {},  # type: ignore[arg-type]
            requires_approval=False,
            approval_reason="",
        )

    # ------------------------------------------------------------------ queries
    def list_tasks(self, principal: Principal, limit: int = 20) -> list[TaskRecord]:
        return self.c.store.list_tasks(principal.tenant_id, limit=limit)

    def get_task(self, principal: Principal, task_id: str) -> TaskRecord:
        task = self.c.store.get_task(task_id)
        if task is None:
            raise NotFound(f"task {task_id} not found")
        if task.tenant_id != principal.tenant_id:
            raise Forbidden("task belongs to another tenant")
        return task


def _jsonable_state(state: AgentState) -> dict[str, Any]:
    import json

    return json.loads(json.dumps(dict(state), ensure_ascii=False, default=str))


def audit_event(container: Container, principal: Principal, action: str, resource: str, detail: str = "") -> None:
    container.store.audit(
        principal.tenant_id, action, resource, detail, current_trace_id()
    )


__all__ = ["ChatService", "audit_event", "now_iso", "get_tracer"]

"""Typed state passed between graph nodes."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # request
    trace_id: str
    tenant_id: str
    session_id: str
    task_id: str
    question: str
    locale: str
    channel: str
    product_sku: str
    require_citations: bool

    # planning
    intent: str
    plan: list[str]
    risk: str
    requires_approval: bool
    approval_reason: str
    approved: bool
    approval_decision: str

    # evidence / output
    contexts: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    draft: str
    review: dict[str, Any]
    rounds: int
    feedback: str

    # bookkeeping
    status: str
    steps: list[dict[str, Any]]
    usage: dict[str, int]
    warnings: list[str]
    degraded: bool


def new_state(
    *,
    question: str,
    tenant_id: str,
    session_id: str,
    task_id: str,
    trace_id: str,
    locale: str = "zh-CN",
    channel: str = "amazon",
    product_sku: str | None = None,
    require_citations: bool = True,
) -> AgentState:
    return AgentState(
        trace_id=trace_id,
        tenant_id=tenant_id,
        session_id=session_id,
        task_id=task_id,
        question=question,
        locale=locale,
        channel=channel,
        product_sku=product_sku or "",
        require_citations=require_citations,
        rounds=0,
        status="running",
        steps=[],
        warnings=[],
        usage={"prompt_tokens": 0, "completion_tokens": 0},
        contexts=[],
        citations=[],
        draft="",
        review={},
        degraded=False,
    )

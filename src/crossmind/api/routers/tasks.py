"""Task inspection and human-in-the-loop approval."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from crossmind.api.deps import ContainerDep, PrincipalDep
from crossmind.domain.models import ApprovalRequest, ChatResponse, TaskRecord
from crossmind.services.chat import ChatService

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("", response_model=list[TaskRecord], summary="任务列表（按租户隔离）")
async def list_tasks(
    principal: PrincipalDep,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[TaskRecord]:
    return ChatService(container).list_tasks(principal, limit=limit)


@router.get("/{task_id}", response_model=TaskRecord, summary="任务详情（含待审批状态快照）")
async def get_task(task_id: str, principal: PrincipalDep, container: ContainerDep) -> TaskRecord:
    return ChatService(container).get_task(principal, task_id)


@router.post("/{task_id}/approve", response_model=ChatResponse, summary="人工审批并继续执行")
async def approve(
    task_id: str,
    payload: ApprovalRequest,
    principal: PrincipalDep,
    container: ContainerDep,
) -> ChatResponse:
    return await ChatService(container).approve(principal, task_id, payload)

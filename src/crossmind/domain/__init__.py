"""Domain package: enums, errors and shared models."""

from __future__ import annotations

from .enums import (
    AgentRole,
    ApprovalDecision,
    Channel,
    Intent,
    RiskLevel,
    TaskStatus,
    TaskType,
)
from .errors import ErrorCode

__all__ = [
    "AgentRole",
    "ApprovalDecision",
    "Channel",
    "ErrorCode",
    "Intent",
    "RiskLevel",
    "TaskStatus",
    "TaskType",
]

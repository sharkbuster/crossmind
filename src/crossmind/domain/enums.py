"""Domain vocabulary: intents, task types, statuses, risk levels."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Python 3.11+ native StrEnum, kept explicit for JSON friendliness."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class Intent(StrEnum):
    SUPPORT = "support"          # 客服：政策/物流/售后问答
    COPYWRITING = "copywriting"  # 文案：标题/五点描述/卖点
    TRANSLATION = "translation"  # 翻译与本地化
    SELECTION = "selection"      # 选品分析
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


class TaskType(StrEnum):
    QA = "qa"
    COPYWRITING = "copywriting"
    TRANSLATION = "translation"
    SELECTION = "selection"
    COMPOSITE = "composite"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentRole(StrEnum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    SUPPORT = "support"
    COPYWRITER = "copywriter"
    TRANSLATOR = "translator"
    SELECTOR = "selector"
    CRITIC = "critic"
    FINALIZER = "finalizer"


class Channel(StrEnum):
    """Marketplace the copy is written for; drives tone and length limits."""

    AMAZON = "amazon"
    SHOPEE = "shopee"
    TIKTOK = "tiktok"
    INDEPENDENT_SITE = "site"


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"

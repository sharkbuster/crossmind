"""Provider-agnostic chat model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from crossmind.domain.models import Usage

Role = str  # "system" | "user" | "assistant"


@dataclass(slots=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ChatResult:
    text: str
    usage: Usage
    model: str
    finish_reason: str = "stop"
    latency_ms: int = 0

    @property
    def tokens(self) -> int:
        return self.usage.total_tokens


class ChatModelError(Exception):
    """Raised by providers; the router decides whether to retry or degrade."""

    def __init__(self, message: str, *, retryable: bool = True, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class ChatModel(ABC):
    """A single completion call. Implementations must be safe for concurrent use."""

    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> ChatResult: ...


def estimate_tokens(text: str) -> int:
    """Cheap token estimate that is good enough for budgets and unit assertions."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk * 1.0 + other / 4.0))

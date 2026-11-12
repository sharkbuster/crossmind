"""Model routing: retries, circuit breaker and graceful degradation.

The platform must answer even when the upstream model is down.  A failed primary
model degrades to the deterministic local model instead of returning a 5xx, and
the degradation is recorded on the span + metrics so it is visible, not silent.
"""

from __future__ import annotations

import asyncio
import time

from crossmind.config import Settings
from crossmind.llm.base import ChatModel, ChatModelError, ChatResult, Message
from crossmind.llm.mock import MockChatModel
from crossmind.llm.openai_compat import OpenAICompatChatModel
from crossmind.obs.logging import get_logger
from crossmind.obs.metrics import LLM_CALLS, LLM_LATENCY, LLM_TOKENS

log = get_logger(__name__)


class CircuitBreaker:
    """Classic consecutive-failure breaker with a cooldown window."""

    def __init__(self, threshold: int = 5, cooldown_s: float = 30.0) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at = 0.0

    @property
    def is_open(self) -> bool:
        if self._failures < self.threshold:
            return False
        if time.monotonic() - self._opened_at > self.cooldown_s:
            self._failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures == self.threshold:
            self._opened_at = time.monotonic()
            log.warning("circuit breaker opened", extra={"failures": self._failures})


class ResilientChatModel(ChatModel):
    """Retry + breaker + fallback wrapper around a primary model."""

    def __init__(
        self,
        primary: ChatModel,
        fallback: ChatModel,
        *,
        max_retries: int = 2,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max_retries
        self.breaker = breaker or CircuitBreaker()
        self.name = f"resilient({primary.name})"

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> ChatResult:
        if not self.breaker.is_open:
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                started = time.perf_counter()
                try:
                    result = await self.primary.complete(
                        messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
                    )
                    self.breaker.record_success()
                    latency = int((time.perf_counter() - started) * 1000)
                    LLM_CALLS.inc(provider=self.primary.name, model=result.model, status="ok")
                    LLM_LATENCY.observe(latency, provider=self.primary.name)
                    LLM_TOKENS.inc(result.usage.prompt_tokens, direction="prompt")
                    LLM_TOKENS.inc(result.usage.completion_tokens, direction="completion")
                    return result
                except ChatModelError as exc:
                    last_error = exc
                    self.breaker.record_failure()
                    LLM_CALLS.inc(provider=self.primary.name, model="unknown", status="error")
                    if not exc.retryable or attempt == self.max_retries:
                        break
                    await asyncio.sleep(0.15 * (2**attempt))
            log.warning(
                "primary model failed, degrading to local model",
                extra={"error": str(last_error), "breaker_open": self.breaker.is_open},
            )

        result = await self.fallback.complete(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        LLM_CALLS.inc(provider=self.fallback.name, model=result.model, status="degraded")
        return result


def build_chat_model(settings: Settings) -> tuple[ChatModel, ChatModel]:
    """Return ``(effective_model, fallback_model)``.

    Both are returned so callers can report whether the answer came from the
    real provider or from the offline fixture.
    """
    fallback = MockChatModel(latency_ms=settings.llm_mock_latency_ms)
    if settings.llm_provider == "mock":
        return fallback, fallback
    primary = OpenAICompatChatModel(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    return ResilientChatModel(primary, fallback, max_retries=settings.llm_max_retries), fallback

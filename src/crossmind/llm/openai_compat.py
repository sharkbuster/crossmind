"""OpenAI-compatible chat client (DeepSeek / Qwen / vLLM / any gateway)."""

from __future__ import annotations

import time

import httpx

from crossmind.domain.models import Usage
from crossmind.llm.base import ChatModel, ChatModelError, ChatResult, Message


class OpenAICompatChatModel(ChatModel):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 30.0,
        name: str = "openai-compat",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ChatModelError("LLM_API_KEY is required when XM_LLM_PROVIDER=openai", retryable=False)
        self.name = name
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            transport=transport,
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> ChatResult:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ChatModelError(f"upstream timeout: {exc}", retryable=True, status=504) from exc
        except httpx.HTTPError as exc:
            raise ChatModelError(f"upstream transport error: {exc}", retryable=True, status=502) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code == 429:
            raise ChatModelError("upstream rate limited", retryable=True, status=429)
        if response.status_code >= 500:
            raise ChatModelError(
                f"upstream {response.status_code}", retryable=True, status=response.status_code
            )
        if response.status_code >= 400:
            raise ChatModelError(
                f"upstream rejected request: {response.text[:300]}",
                retryable=False,
                status=response.status_code,
            )

        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatModelError(f"malformed upstream payload: {data}", retryable=False) from exc

        raw_usage = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
        )
        return ChatResult(
            text=text,
            usage=usage,
            model=data.get("model", self.model),
            finish_reason=str(data["choices"][0].get("finish_reason", "stop")),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

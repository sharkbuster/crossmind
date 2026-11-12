"""Provider resilience: transport mapping, retries, breaker and degradation.

These paths are what make the platform survive an upstream outage, so they are
tested against a mock transport rather than a live endpoint.
"""

from __future__ import annotations

import json

import httpx
import pytest

from crossmind.llm.base import ChatModelError, Message
from crossmind.llm.mock import MockChatModel
from crossmind.llm.openai_compat import OpenAICompatChatModel
from crossmind.llm.router import CircuitBreaker, ResilientChatModel

MESSAGES = [Message(role="user", content="你好")]


def make_model(handler, **kwargs) -> OpenAICompatChatModel:
    return OpenAICompatChatModel(
        base_url="http://llm.test/v1",
        api_key="sk-test",
        model="test-model",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def ok_handler(captured: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["body"] = json.loads(request.content)
            captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "答复"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )

    return handler


async def test_success_parses_text_and_usage() -> None:
    captured: dict = {}
    model = make_model(ok_handler(captured))
    result = await model.complete(MESSAGES)
    assert result.text == "答复"
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 7
    assert captured["auth"] == "Bearer sk-test"
    await model.aclose()


async def test_json_mode_sets_response_format() -> None:
    captured: dict = {}
    model = make_model(ok_handler(captured))
    await model.complete(MESSAGES, json_mode=True)
    assert captured["body"]["response_format"] == {"type": "json_object"}
    await model.aclose()


@pytest.mark.parametrize("status", [500, 502, 503])
async def test_server_errors_are_retryable(status: int) -> None:
    model = make_model(lambda _request: httpx.Response(status, text="boom"))
    with pytest.raises(ChatModelError) as exc:
        await model.complete(MESSAGES)
    assert exc.value.retryable is True
    assert exc.value.status == status
    await model.aclose()


async def test_client_error_is_not_retryable() -> None:
    model = make_model(lambda _request: httpx.Response(400, text="bad request"))
    with pytest.raises(ChatModelError) as exc:
        await model.complete(MESSAGES)
    assert exc.value.retryable is False
    await model.aclose()


async def test_rate_limit_is_retryable() -> None:
    model = make_model(lambda _request: httpx.Response(429, text="slow down"))
    with pytest.raises(ChatModelError) as exc:
        await model.complete(MESSAGES)
    assert exc.value.retryable is True
    await model.aclose()


async def test_timeout_is_mapped_to_retryable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    model = make_model(handler)
    with pytest.raises(ChatModelError) as exc:
        await model.complete(MESSAGES)
    assert exc.value.retryable is True
    assert exc.value.status == 504
    await model.aclose()


async def test_malformed_payload_is_not_retryable() -> None:
    model = make_model(lambda _request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ChatModelError) as exc:
        await model.complete(MESSAGES)
    assert exc.value.retryable is False
    await model.aclose()


async def test_missing_api_key_fails_fast() -> None:
    with pytest.raises(ChatModelError):
        OpenAICompatChatModel(base_url="http://llm.test/v1", api_key="", model="m")


async def test_resilient_model_degrades_to_fallback() -> None:
    calls = {"n": 0}

    def failing(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="down")

    primary = make_model(failing)
    fallback = MockChatModel(model_label="fallback")
    model = ResilientChatModel(primary, fallback, max_retries=1)

    result = await model.complete(MESSAGES)
    assert result.model == "fallback"
    assert calls["n"] == 2  # one retry then degrade
    await primary.aclose()


async def test_breaker_opens_after_threshold() -> None:
    breaker = CircuitBreaker(threshold=2, cooldown_s=60)
    assert breaker.is_open is False
    breaker.record_failure()
    assert breaker.is_open is False
    breaker.record_failure()
    assert breaker.is_open is True
    breaker.record_success()
    assert breaker.is_open is False


async def test_open_breaker_skips_primary() -> None:
    calls = {"n": 0}

    def failing(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="down")

    primary = make_model(failing)
    breaker = CircuitBreaker(threshold=1, cooldown_s=60)
    model = ResilientChatModel(primary, MockChatModel(), max_retries=0, breaker=breaker)

    first = await model.complete(MESSAGES)
    assert first.model != "test-model"
    calls_after_first = calls["n"]

    second = await model.complete(MESSAGES)
    assert second.model != "test-model"
    assert calls["n"] == calls_after_first, "open breaker must not hit the primary again"
    await primary.aclose()

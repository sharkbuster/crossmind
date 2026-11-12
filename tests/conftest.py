"""Shared fixtures.

The container is built once per session through ``asyncio.run`` and reused: the
offline stack holds no loop-bound resources, so this keeps the suite fast while
still exercising the real wiring (index build, answerability gate, graph compile).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from crossmind.api.app import create_app
from crossmind.config import Settings, reset_settings_cache
from crossmind.container import build_container
from crossmind.obs.metrics import reset_metrics

TEST_KEY = "test-key"
TEST_TENANT = "tenant-test"


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("crossmind")
    reset_settings_cache()
    return Settings(
        env="test",
        api_keys=f"{TEST_KEY}:{TEST_TENANT},other-key:tenant-other",
        db_path=root / "test.db",
        trace_dir=root / "traces",
        data_dir=Path(__file__).resolve().parents[1] / "data",
        llm_provider="mock",
        embedding_provider="hash",
        rate_limit_rps=50,
        rate_limit_burst=100,
    )


@pytest.fixture(scope="session")
def container(settings: Settings):
    instance = asyncio.run(build_container(settings))
    yield instance
    asyncio.run(instance.aclose())


@pytest.fixture()
async def client(container) -> httpx.AsyncClient:
    container.limiter.reset()
    app = create_app(container.settings, container=container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://crossmind.test") as http_client:
        yield http_client


@pytest.fixture()
def headers() -> dict[str, str]:
    return {"X-API-Key": TEST_KEY}


@pytest.fixture(autouse=True)
def _clean_metrics() -> Iterator[None]:
    reset_metrics()
    yield

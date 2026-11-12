"""Application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from crossmind import __version__
from crossmind.api.errors import register_error_handlers
from crossmind.api.middleware import ObservabilityMiddleware
from crossmind.api.routers import catalog, chat, kb, ops, tasks
from crossmind.config import Settings, get_settings
from crossmind.container import REPO_ROOT, Container, build_container
from crossmind.obs.logging import configure_logging, get_logger

from crossmind.api.responses import UTF8JSONResponse

WEB_DIR = REPO_ROOT / "apps" / "web"


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    log = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.container is None:
            app.state.container = await build_container(settings)
            log.info(
                "platform ready",
                extra={
                    "offline": settings.is_offline,
                    "documents": len(app.state.container.documents),
                    "chunks": len(app.state.container.chunks),
                },
            )
        try:
            yield
        finally:
            if app.state.container is not None:
                await app.state.container.aclose()

    app = FastAPI(
        title="CrossMind · 跨境电商多智能体运营平台",
        version=__version__,
        description=(
            "四模块（客服 / 文案 / 翻译 / 选品）多智能体平台：RAG 可溯源问答、"
            "LangGraph 编排、HITL 审批、全链路可观测。默认离线运行，无需任何 API Key。"
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
        default_response_class=UTF8JSONResponse,
    )
    app.state.container = container
    app.state.settings = settings
    app.state.max_body_bytes = settings.max_body_bytes

    app.add_middleware(ObservabilityMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://127.0.0.1:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id", "X-Request-ID"],
    )
    register_error_handlers(app)

    prefix = settings.api_prefix
    app.include_router(chat.router, prefix=prefix)
    app.include_router(tasks.router, prefix=prefix)
    app.include_router(kb.router, prefix=prefix)
    app.include_router(catalog.router, prefix=prefix)
    app.include_router(ops.router, prefix=prefix)

    @app.get("/healthz", tags=["ops"], summary="存活探针")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"], summary="就绪探针")
    async def readyz() -> dict[str, object]:
        current: Container | None = app.state.container
        ready = current is not None and len(current.chunks) > 0
        return {
            "status": "ready" if ready else "starting",
            "chunks": len(current.chunks) if current else 0,
            "offline": settings.is_offline,
        }

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"], summary="Prometheus 指标")
    async def metrics() -> PlainTextResponse:
        from crossmind.obs.metrics import render_metrics

        return PlainTextResponse(render_metrics(), media_type="text/plain; version=0.0.4")

    if WEB_DIR.exists():
        app.mount("/console", StaticFiles(directory=WEB_DIR, html=True), name="console")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse("/console/")

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon() -> FileResponse:
            path = Path(WEB_DIR) / "favicon.svg"
            if path.exists():
                return FileResponse(path)
            return PlainTextResponse("", status_code=204)  # type: ignore[return-value]

    return app


app = create_app()

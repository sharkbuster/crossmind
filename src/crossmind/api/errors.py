"""Error rendering: one envelope for every failure."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from crossmind.api.responses import UTF8JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from crossmind.domain.errors import AppError, ErrorCode
from crossmind.obs.logging import get_logger
from crossmind.obs.tracing import current_trace_id

log = get_logger(__name__)


def _envelope(code: str, message: str, details: object = None, retryable: bool = False) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "retryable": retryable,
            "trace_id": current_trace_id() or None,
        }
    }


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        payload = exc.to_dict()
        headers = {}
        if exc.status_code == 429:
            headers["Retry-After"] = "1"
        return UTF8JSONResponse(
            status_code=exc.status_code,
            content=_envelope(payload["code"], payload["message"], payload["details"], payload["retryable"]),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return UTF8JSONResponse(
            status_code=422,
            content=_envelope(
                str(ErrorCode.BAD_REQUEST), "request validation failed", {"errors": exc.errors()[:5]}
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: ErrorCode.NOT_FOUND, 405: ErrorCode.BAD_REQUEST}.get(
            exc.status_code, ErrorCode.INTERNAL
        )
        return UTF8JSONResponse(
            status_code=exc.status_code, content=_envelope(str(code), str(exc.detail))
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled exception")
        return UTF8JSONResponse(
            status_code=500,
            content=_envelope(str(ErrorCode.INTERNAL), "internal server error", retryable=True),
        )


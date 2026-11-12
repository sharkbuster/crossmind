"""HTTP middleware: request id, trace lifecycle, access metrics."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from crossmind.obs.logging import bind_request_id, get_logger
from crossmind.obs.metrics import HTTP_LATENCY, HTTP_REQUESTS
from crossmind.obs.tracing import get_tracer, new_id

log = get_logger(__name__)

SKIP_METRICS = {"/healthz", "/readyz", "/metrics"}


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or new_id()
        bind_request_id(request_id)
        tracer = get_tracer()
        route = request.url.path
        trace = tracer.start("-", f"{request.method} {route}")
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            if route not in SKIP_METRICS:
                HTTP_REQUESTS.inc(route=route, status=str(status))
                HTTP_LATENCY.observe(duration_ms, route=route)
            tracer.finish(trace, "ok" if status < 500 else "error")
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": route,
                    "status": status,
                    "duration_ms": duration_ms,
                    "request_id": request_id,
                },
            )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-Id"] = trace.trace_id
        return response

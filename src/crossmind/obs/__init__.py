"""Observability primitives: structured logs, traces, metrics."""

from __future__ import annotations

from .logging import bind_request_id, bind_tenant, configure_logging, get_logger
from .metrics import (
    HTTP_LATENCY,
    HTTP_REQUESTS,
    LLM_CALLS,
    LLM_TOKENS,
    REGISTRY,
    render_metrics,
    reset_metrics,
)
from .tracing import (
    configure_tracer,
    current_trace,
    current_trace_id,
    get_tracer,
    new_id,
)

__all__ = [
    "HTTP_LATENCY",
    "HTTP_REQUESTS",
    "LLM_CALLS",
    "LLM_TOKENS",
    "REGISTRY",
    "bind_request_id",
    "bind_tenant",
    "configure_logging",
    "configure_tracer",
    "current_trace",
    "current_trace_id",
    "get_logger",
    "get_tracer",
    "new_id",
    "render_metrics",
    "reset_metrics",
]

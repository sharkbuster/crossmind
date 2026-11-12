"""Structured logging with request context.

Log lines are single-line JSON in every environment except local debugging, and
always carry ``trace_id`` / ``tenant_id`` when inside a request, which is what
makes the trace file joinable with the log file.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

from crossmind.obs.tracing import current_trace_id

_tenant: ContextVar[str] = ContextVar("crossmind_tenant", default="-")
_request_id: ContextVar[str] = ContextVar("crossmind_request_id", default="-")

_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


def bind_tenant(tenant_id: str) -> None:
    _tenant.set(tenant_id or "-")


def bind_request_id(request_id: str) -> None:
    _request_id.set(request_id or "-")


def current_tenant() -> str:
    return _tenant.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": current_trace_id() or "-",
            "tenant_id": _tenant.get(),
            "request_id": _request_id.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        return f"{base} trace={current_trace_id() or '-'} tenant={_tenant.get()}"


class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        extra = dict(kwargs.get("extra") or {})
        extra.setdefault("trace_id", current_trace_id() or "-")
        extra.setdefault("tenant_id", _tenant.get())
        kwargs["extra"] = extra
        return msg, kwargs


_CONFIGURED = False


def configure_logging(level: str = "INFO", json_mode: bool = True) -> None:
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    if json_mode:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> ContextAdapter:
    return ContextAdapter(logging.getLogger(name), {})

"""Request-scoped tracing.

A trace is a tree of spans stored in a :mod:`contextvars` context, so every
coroutine started from a request automatically joins the same trace.  Finished
traces are kept in a bounded in-memory ring (for the ops API) and appended to a
JSONL file (for offline analysis / evidence in docs).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from crossmind.domain.models import Span, TraceRecord, now_iso

_current: ContextVar[Trace | None] = ContextVar("crossmind_current_trace", default=None)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class Trace:
    def __init__(self, tenant_id: str, route: str) -> None:
        self.trace_id = new_id()
        self.tenant_id = tenant_id
        self.route = route
        self.started_at = now_iso()
        self._start = time.perf_counter()
        self.spans: list[Span] = []
        self.status = "ok"
        self.duration_ms: float = 0.0

    def finish(self, status: str = "ok") -> TraceRecord:
        self.status = status
        self.duration_ms = round((time.perf_counter() - self._start) * 1000, 3)
        return TraceRecord(
            trace_id=self.trace_id,
            tenant_id=self.tenant_id,
            route=self.route,
            started_at=self.started_at,
            duration_ms=self.duration_ms,
            status=self.status,
            spans=self.spans,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "route": self.route,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "spans": [s.model_dump() for s in self.spans],
        }


class SpanHandle:
    """Mutable handle returned by :meth:`Tracer.span`."""

    def __init__(self, span: Span, trace: Trace) -> None:
        self._span = span
        self._trace = trace

    @property
    def span_id(self) -> str:
        return self._span.span_id

    def set(self, **attrs: Any) -> SpanHandle:
        self._span.attributes.update(attrs)
        return self

    def add(self, key: str, value: Any) -> SpanHandle:
        self._span.attributes[key] = value
        return self


class Tracer:
    def __init__(self, trace_dir: Path | None = None, capacity: int = 512) -> None:
        self._recent: deque[TraceRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._trace_dir = trace_dir
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ lifecycle
    def start(self, tenant_id: str, route: str) -> Trace:
        trace = Trace(tenant_id=tenant_id, route=route)
        _current.set(trace)
        return trace

    def finish(self, trace: Trace, status: str = "ok") -> TraceRecord:
        record = trace.finish(status)
        with self._lock:
            self._recent.append(record)
        if self._trace_dir is not None:
            try:
                with (self._trace_dir / "traces.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
            except OSError:  # pragma: no cover - disk issues must not break requests
                pass
        _current.set(None)
        return record

    # ------------------------------------------------------------------ spans
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[SpanHandle]:
        trace = _current.get()
        parent = trace.spans[-1].span_id if trace and trace.spans else None
        span = Span(
            span_id=new_id(),
            parent_id=parent,
            name=name,
            start_ms=round(time.perf_counter() * 1000, 3),
            attributes=dict(attributes),
        )
        handle = SpanHandle(span, trace) if trace else None
        if trace:
            trace.spans.append(span)
        try:
            yield handle if handle is not None else _NULL_HANDLE
        except Exception as exc:
            span.status = "error"
            span.attributes["error"] = type(exc).__name__
            span.attributes["error_message"] = str(exc)[:300]
            raise
        finally:
            span.end_ms = round(time.perf_counter() * 1000, 3)

    # ------------------------------------------------------------------ queries
    def recent(self, limit: int = 50) -> list[TraceRecord]:
        with self._lock:
            items = list(self._recent)
        return list(reversed(items))[:limit]

    def get(self, trace_id: str) -> TraceRecord | None:
        with self._lock:
            for record in self._recent:
                if record.trace_id == trace_id:
                    return record
        return None


class _NullHandle:
    def set(self, **_attrs: Any) -> _NullHandle:
        return self

    def add(self, _key: str, _value: Any) -> _NullHandle:
        return self

    @property
    def span_id(self) -> str:
        return ""


_NULL_HANDLE = _NullHandle()
_DEFAULT_TRACER = Tracer()


def configure_tracer(trace_dir: Path) -> Tracer:
    global _DEFAULT_TRACER
    _DEFAULT_TRACER = Tracer(trace_dir=trace_dir)
    return _DEFAULT_TRACER


def get_tracer() -> Tracer:
    return _DEFAULT_TRACER


def current_trace() -> Trace | None:
    return _current.get()


def current_trace_id() -> str:
    trace = _current.get()
    return trace.trace_id if trace else ""

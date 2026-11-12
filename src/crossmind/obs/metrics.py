"""Minimal, dependency-free Prometheus-style metrics registry.

Only counters and histograms are needed by the platform, and keeping the
implementation in-tree avoids pulling a client library into every image.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import Any

DEFAULT_BUCKETS: tuple[float, ...] = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


def _sanitize(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() or ch == "_" else "_")
    cleaned = "".join(out)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def _labels_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


class Counter:
    def __init__(self, name: str, help_text: str = "") -> None:
        self.name = _sanitize(name)
        self.help = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        self._values[_labels_key(labels)] += amount

    def value(self, **labels: str) -> float:
        return self._values.get(_labels_key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for key, value in sorted(self._values.items()):
            label_str = ",".join(f'{k}="{v}"' for k, v in key)
            suffix = f"{{{label_str}}}" if label_str else ""
            lines.append(f"{self.name}{suffix} {value:g}")
        return lines


class Histogram:
    def __init__(self, name: str, help_text: str = "", buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        self.name = _sanitize(name)
        self.help = help_text
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[tuple[tuple[str, str], ...], list[int]] = defaultdict(
            lambda: [0] * (len(self.buckets) + 1)
        )
        self._sums: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._count: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)

    def observe(self, value: float, **labels: str) -> None:
        key = _labels_key(labels)
        counts = self._counts[key]
        idx = len(self.buckets)
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                idx = i
                break
        counts[idx] += 1
        self._sums[key] += value
        self._count[key] += 1

    def snapshot(self, **labels: str) -> dict[str, float]:
        key = _labels_key(labels)
        counts = self._counts.get(key)
        if counts is None:
            return {"count": 0, "sum": 0.0, "avg": 0.0}
        total = self._count[key]
        total_sum = self._sums[key]
        return {
            "count": float(total),
            "sum": total_sum,
            "avg": round(total_sum / total, 3) if total else 0.0,
        }

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key in sorted(self._counts):
            label_str = ",".join(f'{k}="{v}"' for k, v in key)
            counts = self._counts[key]
            cumulative = 0
            for i, bound in enumerate(self.buckets):
                cumulative += counts[i]
                labels = f'{label_str},le="{bound:g}"' if label_str else f'le="{bound:g}"'
                lines.append(f"{self.name}_bucket{{{labels}}} {cumulative}")
            cumulative += counts[-1]
            labels = f'{label_str},le="+Inf"' if label_str else 'le="+Inf"'
            lines.append(f"{self.name}_bucket{{{labels}}} {cumulative}")
            prefix = f"{{{label_str}}}" if label_str else ""
            lines.append(f"{self.name}_sum{prefix} {self._sums[key]:g}")
            lines.append(f"{self.name}_count{prefix} {self._count[key]}")
        return lines


class Gauge:
    def __init__(self, name: str, help_text: str = "") -> None:
        self.name = _sanitize(name)
        self.help = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        self._values[_labels_key(labels)] = float(value)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        for key, value in sorted(self._values.items()):
            label_str = ",".join(f'{k}="{v}"' for k, v in key)
            suffix = f"{{{label_str}}}" if label_str else ""
            lines.append(f"{self.name}{suffix} {value:g}")
        return lines


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, Counter] = {}
        self.histograms: dict[str, Histogram] = {}
        self.gauges: dict[str, Gauge] = {}

    def counter(self, name: str, help_text: str = "") -> Counter:
        with self._lock:
            if name not in self.counters:
                self.counters[name] = Counter(name, help_text)
            return self.counters[name]

    def histogram(
        self, name: str, help_text: str = "", buckets: tuple[float, ...] = DEFAULT_BUCKETS
    ) -> Histogram:
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(name, help_text, buckets)
            return self.histograms[name]

    def gauge(self, name: str, help_text: str = "") -> Gauge:
        with self._lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(name, help_text)
            return self.gauges[name]

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = list(self.counters.values())
            histograms = list(self.histograms.values())
            gauges = list(self.gauges.values())
        for obj in (counters, histograms, gauges):
            for metric in obj:
                lines.extend(metric.render())
                lines.append("")
        return "\n".join(lines) + "\n"


REGISTRY = Registry()

# ---- canonical metrics -------------------------------------------------------
HTTP_REQUESTS = REGISTRY.counter("crossmind_http_requests_total", "HTTP requests by route/status")
HTTP_LATENCY = REGISTRY.histogram(
    "crossmind_http_request_duration_ms", "HTTP latency in ms", buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500)
)
LLM_CALLS = REGISTRY.counter("crossmind_llm_calls_total", "LLM invocations by provider/model/status")
LLM_LATENCY = REGISTRY.histogram(
    "crossmind_llm_latency_ms", "LLM latency in ms", buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
)
LLM_TOKENS = REGISTRY.counter("crossmind_llm_tokens_total", "Token usage by direction")
RETRIEVAL_CALLS = REGISTRY.counter("crossmind_retrieval_total", "Retrieval calls by outcome")
RATE_LIMITED = REGISTRY.counter("crossmind_rate_limited_total", "Requests rejected by the rate limiter")
HITL_TASKS = REGISTRY.counter("crossmind_hitl_total", "Human-in-the-loop gates by decision")
CACHE_HITS = REGISTRY.counter("crossmind_idempotent_replays_total", "Idempotent replays served")
ACTIVE_TASKS = REGISTRY.gauge("crossmind_tasks_in_flight", "Tasks currently executing")


def percentage(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 2)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; small helper used by the load-test script."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return round(ordered[rank - 1], 3)


def render_metrics() -> str:
    return REGISTRY.render()


def reset_metrics() -> None:
    """Test helper: keep the module-level metric handles, drop their data."""
    for metric in list(REGISTRY.counters.values()):
        metric._values.clear()  # noqa: SLF001 - intentional test hook
    for hist in list(REGISTRY.histograms.values()):
        hist._counts.clear()  # noqa: SLF001
        hist._sums.clear()  # noqa: SLF001
        hist._count.clear()  # noqa: SLF001
    for gauge in list(REGISTRY.gauges.values()):
        gauge._values.clear()  # noqa: SLF001


def labels_preview(metric: Any) -> dict[str, Any]:  # pragma: no cover - debugging helper
    return {"name": getattr(metric, "name", "?")}

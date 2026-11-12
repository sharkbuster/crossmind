"""Metrics registry rendering."""

from __future__ import annotations

from crossmind.obs.metrics import REGISTRY, Registry, percentile


def test_counter_renders_with_labels() -> None:
    registry = Registry()
    counter = registry.counter("test_requests_total", "requests")
    counter.inc(route="/a", status="200")
    counter.inc(2, route="/a", status="200")
    rendered = registry.render()
    assert "# TYPE test_requests_total counter" in rendered
    assert 'test_requests_total{route="/a",status="200"} 3' in rendered


def test_histogram_renders_buckets_and_sum() -> None:
    registry = Registry()
    hist = registry.histogram("test_latency_ms", "latency", buckets=(10, 100))
    hist.observe(5)
    hist.observe(50)
    hist.observe(500)
    rendered = registry.render()
    assert 'test_latency_ms_bucket{le="10"} 1' in rendered
    assert 'test_latency_ms_bucket{le="100"} 2' in rendered
    assert 'test_latency_ms_bucket{le="+Inf"} 3' in rendered
    assert "test_latency_ms_count 3" in rendered


def test_histogram_snapshot_average() -> None:
    registry = Registry()
    hist = registry.histogram("h")
    hist.observe(10)
    hist.observe(30)
    snapshot = hist.snapshot()
    assert snapshot["count"] == 2
    assert snapshot["avg"] == 20.0


def test_gauge_renders_last_value() -> None:
    registry = Registry()
    gauge = registry.gauge("test_in_flight")
    gauge.set(3)
    gauge.set(1)
    assert "test_in_flight 1" in registry.render()


def test_global_registry_has_canonical_metrics() -> None:
    names = set(REGISTRY.counters) | set(REGISTRY.histograms) | set(REGISTRY.gauges)
    assert "crossmind_http_requests_total" in names
    assert "crossmind_llm_latency_ms" in names
    assert "crossmind_tasks_in_flight" in names


def test_percentile_nearest_rank() -> None:
    values = [float(x) for x in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile([], 95) == 0.0

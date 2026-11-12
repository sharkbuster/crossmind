"""In-process load test against the ASGI app.

Running over ASGITransport removes the network stack from the measurement, which
is what you want when the question is "how much of the latency is our code":
the remaining number is routing + retrieval + graph + serialisation.

Writes reports/load-test.json and prints p50/p95/p99.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossmind.api.app import create_app  # noqa: E402
from crossmind.config import Settings  # noqa: E402
from crossmind.container import build_container  # noqa: E402
from crossmind.obs.metrics import percentile  # noqa: E402

QUESTIONS = [
    "未拆封商品多久可以退货？",
    "标准专线几天送达，运费多少？",
    "退款多久到账？",
    "电池保修多久？",
    "美国免税额度是多少？",
    "积分怎么抵扣？",
    "投诉多久有响应？",
    "AirSound 续航多久？",
    "PowerGo 能带上飞机吗？",
    "给 AirSound 写一版 Amazon 文案",
    "帮我分析音频品类的选品机会",
    "FitTrack 手环支持哪些系统？",
]


async def worker(
    client: httpx.AsyncClient, headers: dict[str, str], count: int, latencies: list[float], errors: list[str]
) -> None:
    for _ in range(count):
        question = random.choice(QUESTIONS)
        started = time.perf_counter()
        try:
            response = await client.post("/api/v1/chat", json={"message": question}, headers=headers)
            if response.status_code != 200:
                errors.append(f"{response.status_code}: {response.text[:120]}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
        latencies.append((time.perf_counter() - started) * 1000)


async def main(total: int, concurrency: int) -> int:
    settings = Settings(rate_limit_rps=10000, rate_limit_burst=10000)
    container = await build_container(settings)
    try:
        app = create_app(settings, container=container)
        transport = httpx.ASGITransport(app=app)
        headers = {"X-API-Key": "demo-key-alpha"}
        per_worker = max(1, total // concurrency)
        latencies: list[float] = []
        errors: list[str] = []

        async with httpx.AsyncClient(transport=transport, base_url="http://load.test") as client:
            await client.post(
                "/api/v1/chat", json={"message": "预热"}, headers=headers
            )  # warm jieba/index caches
            started = time.perf_counter()
            await asyncio.gather(
                *[
                    worker(client, headers, per_worker, latencies, errors)
                    for _ in range(concurrency)
                ]
            )
            elapsed = time.perf_counter() - started

        summary = {
            "requests": len(latencies),
            "concurrency": concurrency,
            "errors": len(errors),
            "duration_s": round(elapsed, 3),
            "rps": round(len(latencies) / elapsed, 1),
            "p50_ms": percentile(latencies, 50),
            "p90_ms": percentile(latencies, 90),
            "p95_ms": percentile(latencies, 95),
            "p99_ms": percentile(latencies, 99),
            "mean_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        }
        out = ROOT / "reports"
        out.mkdir(exist_ok=True)
        (out / "load-test.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        if errors:
            print("\nfirst errors:", errors[:3], file=sys.stderr)
        return 0 if not errors else 1
    finally:
        await container.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.requests, args.concurrency)))

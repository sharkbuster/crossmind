"""Per-tenant token-bucket rate limiting (in-process).

Deliberately in-process: it protects a single instance and keeps the platform
dependency-free.  The interface takes a routing scope so that expensive routes
(chat) can be limited more tightly than cheap ones (health), and so a Redis
implementation can drop in behind it when the service scales horizontally.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    tokens: float
    updated_at: float = field(default_factory=time.monotonic)


class TokenBucketLimiter:
    def __init__(self, rate_per_s: float, burst: int, *, cost: float = 1.0) -> None:
        self.rate = rate_per_s
        self.burst = float(burst)
        self.cost = cost
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _take(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.burst)
                self._buckets[key] = bucket
            elapsed = now - bucket.updated_at
            bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.rate)
            bucket.updated_at = now
            if bucket.tokens >= self.cost:
                bucket.tokens -= self.cost
                return True, 0.0
            deficit = self.cost - bucket.tokens
            return False, round(deficit / self.rate, 3)

    def allow(self, tenant_id: str, scope: str = "default") -> tuple[bool, float]:
        return self._take(f"{tenant_id}:{scope}")

    def reset(self, tenant_id: str | None = None) -> None:
        with self._lock:
            if tenant_id is None:
                self._buckets.clear()
            else:
                for key in [k for k in self._buckets if k.startswith(f"{tenant_id}:")]:
                    self._buckets.pop(key, None)

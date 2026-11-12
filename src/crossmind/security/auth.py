"""API-key authentication mapped to tenants."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from crossmind.domain.errors import Unauthorized


@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: str
    key_fingerprint: str

    @property
    def log_label(self) -> str:
        return f"{self.tenant_id}#{self.key_fingerprint}"


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


class ApiKeyAuthenticator:
    def __init__(self, key_map: dict[str, str]) -> None:
        if not key_map:
            raise ValueError("no API keys configured")
        self._key_map = key_map

    def authenticate(self, api_key: str | None) -> Principal:
        if not api_key:
            raise Unauthorized("missing API key", details={"header": "X-API-Key"})
        for candidate, tenant in self._key_map.items():
            if hmac.compare_digest(candidate, api_key):
                return Principal(tenant_id=tenant, key_fingerprint=fingerprint(candidate))
        raise Unauthorized("invalid API key")

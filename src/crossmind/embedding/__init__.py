"""Embedding providers."""

from __future__ import annotations

import hashlib
import math
import re

import httpx

from crossmind.domain.errors import UpstreamUnavailable

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Mixed CJK/Latin tokenizer: Latin/digit words + CJK unigrams.

    Chinese has no whitespace, so unigrams plus character bigrams are the
    cheapest tokenisation that still carries enough signal for hashing
    embeddings and BM25 without pulling in a segmenter dependency.
    """
    lowered = text.lower()
    tokens = TOKEN_RE.findall(lowered)
    cjk = [t for t in tokens if len(t) == 1 and "\u4e00" <= t <= "\u9fff"]
    bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return tokens + bigrams


class HashingEmbedder:
    """Feature-hashing embedder: deterministic, offline, zero dependencies."""

    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            weight = 1.0 if len(token) > 1 else 0.5
            vector[index] += sign * weight
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]


class RemoteEmbedder:
    """OpenAI-compatible ``/embeddings`` client."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, *, timeout_s: float = 30.0) -> None:
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "/embeddings", json={"model": self.model, "input": texts}
            )
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable(f"embedding upstream error: {exc}") from exc
        if response.status_code >= 400:
            raise UpstreamUnavailable(f"embedding upstream {response.status_code}: {response.text[:200]}")
        data = response.json().get("data") or []
        data.sort(key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]

    async def aclose(self) -> None:
        await self._client.aclose()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

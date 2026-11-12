"""Centralised, validated configuration.

Every knob is env driven (prefix ``XM_``) and validated at import time so that a
mis-configured deployment fails fast instead of at the first request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Env = Literal["dev", "test", "prod"]
Provider = Literal["mock", "openai"]
EmbeddingProvider = Literal["hash", "openai"]

ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="XM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- runtime ----
    app_name: str = "CrossMind"
    env: Env = "dev"
    api_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    log_json: bool = True
    db_path: Path = Path("var/crossmind.db")
    trace_dir: Path = Path("var/traces")
    data_dir: Path = Path("data")

    # ---- auth: "key:tenant,key:tenant" ----
    api_keys: str = "demo-key-alpha:tenant-alpha,demo-key-beta:tenant-beta"

    # ---- rate limit (token bucket, per tenant) ----
    rate_limit_rps: float = Field(default=20.0, gt=0)
    rate_limit_burst: int = Field(default=40, ge=1)
    max_body_bytes: int = Field(default=1_000_000, ge=1024)

    # ---- llm ----
    llm_provider: Provider = "mock"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout_s: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_mock_latency_ms: int = Field(default=0, ge=0)

    # ---- embedding ----
    embedding_provider: EmbeddingProvider = "hash"
    embedding_dim: int = Field(default=256, ge=64, le=4096)

    # ---- retrieval ----
    rag_chunk_size: int = Field(default=260, ge=200, le=1200)
    rag_chunk_overlap: float = Field(default=0.15, ge=0.0, le=0.5)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_recall_k: int = Field(default=20, ge=5, le=100)
    rag_score_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    rag_embed_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    rag_answerability_gate: bool = True
    rag_answerability_threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    # ---- agent / business ----
    hitl_amount_threshold: float = Field(default=500.0, ge=0)
    max_review_rounds: int = Field(default=2, ge=0, le=5)
    idempotency_ttl_s: int = Field(default=86400, ge=60)

    # ---------------------------------------------------------------- validation
    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        level = v.upper()
        if level not in ALLOWED_LOG_LEVELS:
            raise ValueError(f"XM_LOG_LEVEL must be one of {sorted(ALLOWED_LOG_LEVELS)}, got {v!r}")
        return level

    @field_validator("api_keys")
    @classmethod
    def _check_api_keys(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("XM_API_KEYS is empty: at least one 'key:tenant' pair is required")
        for pair in v.split(","):
            if not pair.strip():
                continue
            if ":" not in pair:
                raise ValueError(f"malformed XM_API_KEYS entry {pair!r}, expected 'key:tenant'")
        return v

    @property
    def api_key_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for pair in self.api_keys.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, tenant = pair.split(":", 1)
            mapping[key.strip()] = tenant.strip()
        return mapping

    @property
    def is_offline(self) -> bool:
        return self.llm_provider == "mock" and self.embedding_provider == "hash"

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def reset_settings_cache() -> None:
    """Used by tests that need to rebuild settings from a patched environment."""
    get_settings.cache_clear()

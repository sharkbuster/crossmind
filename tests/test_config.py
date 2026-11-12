"""Configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crossmind.config import Settings


def test_offline_mode_needs_no_secrets() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "mock"
    assert settings.embedding_provider == "hash"
    assert settings.is_offline is True
    assert settings.llm_api_key == ""


def test_api_keys_are_parsed_into_tenant_map() -> None:
    settings = Settings(_env_file=None, api_keys="k1:tenant-a,k2:tenant-b")
    assert settings.api_key_map == {"k1": "tenant-a", "k2": "tenant-b"}


def test_malformed_api_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_keys="no-tenant-here")


def test_empty_api_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_keys="  ")


def test_log_level_is_normalised_and_validated() -> None:
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="verbose")


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_score_threshold_out_of_range_is_rejected(threshold: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rag_score_threshold=threshold)


def test_chunk_size_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rag_chunk_size=100)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rag_chunk_size=5000)


def test_ensure_dirs_creates_parents(tmp_path) -> None:
    settings = Settings(
        _env_file=None, db_path=tmp_path / "nested" / "db.sqlite", trace_dir=tmp_path / "nested" / "traces"
    )
    settings.ensure_dirs()
    assert settings.db_path.parent.exists()
    assert settings.trace_dir.exists()

"""Project metadata consistency.

A UTF-8 BOM in ``pyproject.toml`` breaks every TOML parser, and a version that
drifts between the package and the build config makes release notes lie.  Both
are cheap to check and were real incidents during development.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import crossmind

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_is_valid_toml_and_has_no_bom() -> None:
    raw = (ROOT / "pyproject.toml").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "pyproject.toml must not start with a UTF-8 BOM"
    data = tomllib.loads(raw.decode("utf-8"))
    assert data["project"]["name"] == "crossmind"
    assert data["project"]["requires-python"] == ">=3.11"


def test_package_version_matches_pyproject() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert crossmind.__version__ == data["project"]["version"]


def test_env_example_has_no_bom_and_covers_critical_keys() -> None:
    raw = (ROOT / ".env.example").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), ".env.example must not start with a UTF-8 BOM"
    text = raw.decode("utf-8")
    for key in (
        "XM_LLM_PROVIDER",
        "XM_API_KEYS",
        "XM_RAG_SCORE_THRESHOLD",
        "XM_RAG_ANSWERABILITY_THRESHOLD",
        "XM_MAX_BODY_BYTES",
    ):
        assert f"{key}=" in text, f"{key} missing from .env.example"


def test_web_console_assets_exist() -> None:
    web = ROOT / "apps" / "web"
    for name in ("index.html", "app.js", "styles.css"):
        assert (web / name).is_file(), f"missing console asset {name}"
    html = (web / "index.html").read_text(encoding="utf-8")
    assert "/console/" not in html  # served at mount root, not nested

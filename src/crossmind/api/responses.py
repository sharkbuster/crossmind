"""Shared response types."""

from __future__ import annotations

from fastapi.responses import JSONResponse


class UTF8JSONResponse(JSONResponse):
    """Explicit charset: this API is Chinese-first and some clients default to latin-1."""

    media_type = "application/json; charset=utf-8"

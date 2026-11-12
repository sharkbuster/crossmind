"""Error taxonomy.

Every failure that can reach a client is expressed as an :class:`AppError` with a
stable machine-readable ``code``.  The HTTP layer renders it into a single
envelope so that callers never have to parse prose.
"""

from __future__ import annotations

from typing import Any

from .enums import StrEnum


class ErrorCode(StrEnum):
    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    RATE_LIMITED = "rate_limited"
    GUARD_BLOCKED = "guard_blocked"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    NO_ANSWER = "no_answer"
    INTERNAL = "internal_error"


class AppError(Exception):
    """Base class for expected, client-visible failures."""

    code: ErrorCode = ErrorCode.INTERNAL
    status_code: int = 500
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        code: ErrorCode | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": str(self.code),
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


class BadRequest(AppError):
    code = ErrorCode.BAD_REQUEST
    status_code = 400


class Unauthorized(AppError):
    code = ErrorCode.UNAUTHORIZED
    status_code = 401


class Forbidden(AppError):
    code = ErrorCode.FORBIDDEN
    status_code = 403


class NotFound(AppError):
    code = ErrorCode.NOT_FOUND
    status_code = 404


class Conflict(AppError):
    code = ErrorCode.CONFLICT
    status_code = 409


class RateLimited(AppError):
    code = ErrorCode.RATE_LIMITED
    status_code = 429
    retryable = True


class GuardBlocked(AppError):
    code = ErrorCode.GUARD_BLOCKED
    status_code = 422


class UpstreamUnavailable(AppError):
    code = ErrorCode.UPSTREAM_UNAVAILABLE
    status_code = 503
    retryable = True


class UpstreamTimeout(AppError):
    code = ErrorCode.UPSTREAM_TIMEOUT
    status_code = 504
    retryable = True

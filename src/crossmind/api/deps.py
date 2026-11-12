"""FastAPI dependencies: container, auth, rate limiting."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from crossmind.container import Container
from crossmind.domain.errors import RateLimited
from crossmind.obs.logging import bind_tenant
from crossmind.obs.metrics import RATE_LIMITED
from crossmind.obs.tracing import current_trace
from crossmind.security.auth import Principal


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - only before startup
        raise RuntimeError("container is not initialised")
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_principal(
    container: ContainerDep,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    principal = container.auth.authenticate(api_key)
    bind_tenant(principal.tenant_id)
    trace = current_trace()
    if trace is not None:
        trace.tenant_id = principal.tenant_id
    return principal


PrincipalDep = Annotated[Principal, Depends(get_principal)]


def rate_limited(scope: str, cost: float = 1.0):
    """Dependency factory so each route declares its own budget."""

    def _dep(principal: PrincipalDep, container: ContainerDep) -> Principal:
        allowed, retry_after = container.limiter.allow(principal.tenant_id, scope)
        if not allowed:
            RATE_LIMITED.inc(tenant=principal.tenant_id, scope=scope)
            raise RateLimited(
                "rate limit exceeded",
                details={"scope": scope, "retry_after_s": retry_after},
            )
        return principal

    return _dep


ChatPrincipalDep = Annotated[Principal, Depends(rate_limited("chat", cost=1.0))]
EvalPrincipalDep = Annotated[Principal, Depends(rate_limited("eval", cost=5.0))]

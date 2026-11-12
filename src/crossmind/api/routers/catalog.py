"""Catalog + tool discovery."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from crossmind.agents.tools import ToolRegistry
from crossmind.api.deps import ContainerDep, PrincipalDep
from crossmind.domain.models import Product

router = APIRouter(tags=["catalog"])


@router.get("/catalog/products", response_model=list[Product], summary="商品列表")
async def products(
    _principal: PrincipalDep, container: ContainerDep, category: str | None = None
) -> list[Product]:
    return container.catalog.list_products(category)


@router.get("/catalog/categories", summary="类目与市场信号")
async def categories(_principal: PrincipalDep, container: ContainerDep) -> dict[str, Any]:
    return {
        "categories": container.catalog.categories(),
        "signals": container.catalog.market_signals,
        "version": container.catalog.version,
    }


@router.get("/tools", summary="智能体可用工具清单")
async def tools(_principal: PrincipalDep, container: ContainerDep) -> dict[str, Any]:
    registry = ToolRegistry(container.catalog)
    return {"tools": [spec.to_dict() for spec in registry.specs()]}

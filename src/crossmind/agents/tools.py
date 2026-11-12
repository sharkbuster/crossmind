"""Tools available to agents.

Each tool is a plain Python callable with an explicit spec, which keeps them
testable in isolation and makes it trivial to expose the same catalogue over an
MCP/HTTP boundary later.  Tools never raise raw exceptions: an agent that cannot
get data should degrade its plan, not crash the graph.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crossmind.services.catalog import Catalog

EXCHANGE_RATES = {
    ("USD", "CNY"): 7.18,
    ("USD", "EUR"): 0.92,
    ("USD", "JPY"): 151.4,
    ("USD", "GBP"): 0.79,
}

CARRIERS = ["YunExpress", "4PX", "燕文", "DHL eCommerce"]
STAGES = ["已揽收", "已出境", "到达目的国", "清关中", "派送中", "已签收"]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., dict[str, Any]]
    parameters: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._tools: dict[str, ToolSpec] = {}
        for spec in self._build(catalog):
            self._tools[spec.name] = spec

    # ------------------------------------------------------------------ specs
    def _build(self, catalog: Catalog) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="catalog_lookup",
                description="按 SKU 或商品名查询商品价格、成本、重量与卖点",
                parameters={"sku": "string", "name": "string"},
                handler=self._catalog_lookup,
            ),
            ToolSpec(
                name="market_signal",
                description="查询品类增长率、中位价与竞争强度",
                parameters={"category": "string"},
                handler=self._market_signal,
            ),
            ToolSpec(
                name="competitor_scan",
                description="查询品类头部竞品的价格、评分与核心优势",
                parameters={"category": "string"},
                handler=self._competitor_scan,
            ),
            ToolSpec(
                name="logistics_track",
                description="按订单号查询跨境物流轨迹（离线确定性数据）",
                parameters={"order_no": "string"},
                handler=self._logistics_track,
            ),
            ToolSpec(
                name="exchange_rate",
                description="查询美元对目标货币的参考汇率",
                parameters={"base": "string", "quote": "string"},
                handler=self._exchange_rate,
            ),
        ]

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def call(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        spec = self._tools.get(tool_name)
        if spec is None:
            return {"ok": False, "error": f"unknown tool {tool_name}"}
        try:
            return {"ok": True, "data": spec.handler(**kwargs)}
        except Exception as exc:  # noqa: BLE001 - tools degrade, they do not crash graphs
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------ handlers
    def _catalog_lookup(self, sku: str = "", name: str = "") -> dict[str, Any]:
        product = self.catalog.resolve(sku=sku or None, text=name)
        if product is None:
            return {"found": False}
        return {
            "found": True,
            "product": product.model_dump(),
            "margin": product.margin,
        }

    def _market_signal(self, category: str) -> dict[str, Any]:
        signal = self.catalog.signal_for(category)
        return {"category": category, **signal} if signal else {"category": category, "found": False}

    def _competitor_scan(self, category: str) -> dict[str, Any]:
        return {"category": category, "competitors": self.catalog.competitors_for(category)}

    def _logistics_track(self, order_no: str) -> dict[str, Any]:
        digest = hashlib.sha256(order_no.encode("utf-8")).digest()
        stage_index = digest[0] % len(STAGES)
        return {
            "order_no": order_no,
            "carrier": CARRIERS[digest[1] % len(CARRIERS)],
            "stage": STAGES[stage_index],
            "progress": round((stage_index + 1) / len(STAGES), 2),
            "eta_days": max(1, 12 - stage_index * 2),
            "note": "轨迹为离线示例数据，接入真实承运商 API 时替换本 handler",
        }

    def _exchange_rate(self, base: str = "USD", quote: str = "CNY") -> dict[str, Any]:
        pair = (base.upper(), quote.upper())
        rate = EXCHANGE_RATES.get(pair)
        if rate is None:
            return {"found": False, "pair": f"{pair[0]}/{pair[1]}"}
        return {"base": pair[0], "quote": pair[1], "rate": rate, "source": "offline-reference-table"}


"""Tool registry and catalog reference data."""

from __future__ import annotations

import httpx
import pytest

from crossmind.agents.tools import EXCHANGE_RATES, ToolRegistry
from crossmind.domain.errors import NotFound
from crossmind.services.catalog import Catalog


@pytest.fixture(scope="module")
def catalog(container) -> Catalog:
    return container.catalog


@pytest.fixture(scope="module")
def tools(catalog: Catalog) -> ToolRegistry:
    return ToolRegistry(catalog)


def test_specs_are_described_for_the_console(catalog: Catalog) -> None:
    registry = ToolRegistry(catalog)
    specs = registry.specs()
    names = {spec.name for spec in specs}
    assert names == {
        "catalog_lookup",
        "market_signal",
        "competitor_scan",
        "logistics_track",
        "exchange_rate",
    }
    payload = specs[0].to_dict()
    assert payload["description"]
    assert payload["parameters"]


def test_catalog_lookup_by_sku(tools: ToolRegistry) -> None:
    result = tools.call("catalog_lookup", sku="CB-AS-001")
    assert result["ok"] is True
    assert result["data"]["found"] is True
    assert result["data"]["product"]["name"].startswith("AirSound")
    assert 0 < result["data"]["margin"] < 1


def test_catalog_lookup_by_name(tools: ToolRegistry) -> None:
    result = tools.call("catalog_lookup", name="PowerGo")
    assert result["data"]["found"] is True
    assert result["data"]["product"]["sku"] == "CB-PG-003"


def test_catalog_lookup_miss_returns_not_found(tools: ToolRegistry) -> None:
    result = tools.call("catalog_lookup", sku="NOPE-1")
    assert result["ok"] is True
    assert result["data"]["found"] is False


def test_market_signal_and_competitors(tools: ToolRegistry) -> None:
    signal = tools.call("market_signal", category="音频")
    assert signal["data"]["growth_rate"]
    competitors = tools.call("competitor_scan", category="音频")
    assert len(competitors["data"]["competitors"]) >= 2
    missing = tools.call("market_signal", category="不存在")
    assert missing["data"]["found"] is False


def test_logistics_tracking_is_deterministic(tools: ToolRegistry) -> None:
    first = tools.call("logistics_track", order_no="CB202609150001")
    second = tools.call("logistics_track", order_no="CB202609150001")
    assert first == second
    assert first["data"]["stage"]
    assert 0 < first["data"]["progress"] <= 1


def test_exchange_rate_pairs(tools: ToolRegistry) -> None:
    hit = tools.call("exchange_rate", base="usd", quote="cny")
    assert hit["data"]["rate"] == EXCHANGE_RATES[("USD", "CNY")]
    miss = tools.call("exchange_rate", base="USD", quote="XXX")
    assert miss["data"]["found"] is False


def test_unknown_tool_is_reported_not_raised(tools: ToolRegistry) -> None:
    result = tools.call("teleport")
    assert result["ok"] is False
    assert "unknown tool" in result["error"]


def test_handler_error_is_converted_to_result(tools: ToolRegistry, monkeypatch) -> None:
    def boom(**_kwargs):
        raise RuntimeError("catalog exploded")

    monkeypatch.setattr(tools.catalog, "resolve", boom)
    result = tools.call("catalog_lookup", sku="X")
    assert result["ok"] is False
    assert "RuntimeError" in result["error"]


def test_catalog_queries(catalog: Catalog) -> None:
    assert len(catalog.list_products()) == 4
    assert all(p.category == "音频" for p in catalog.list_products("音频"))
    assert catalog.get("cb-as-001").sku == "CB-AS-001"
    assert catalog.find_by_name("AirSound") is not None
    assert catalog.resolve(text="完全不存在的商品") is None
    assert "音频" in catalog.categories()
    with pytest.raises(NotFound):
        catalog.get("missing")


async def test_catalog_endpoints(
    client: httpx.AsyncClient, headers: dict[str, str]
) -> None:
    products = (await client.get("/api/v1/catalog/products", headers=headers)).json()
    assert len(products) == 4

    audio = (
        await client.get("/api/v1/catalog/products?category=音频", headers=headers)
    ).json()
    assert audio and all(item["category"] == "音频" for item in audio)

    categories = (await client.get("/api/v1/catalog/categories", headers=headers)).json()
    assert "音频" in categories["categories"]
    assert categories["signals"]

    tools = (await client.get("/api/v1/tools", headers=headers)).json()["tools"]
    assert {tool["name"] for tool in tools} >= {"exchange_rate", "logistics_track"}

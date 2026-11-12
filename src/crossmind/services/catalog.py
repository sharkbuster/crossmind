"""Product catalog + market signals (read-only reference data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crossmind.domain.errors import NotFound
from crossmind.domain.models import Product


class Catalog:
    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.version: str = payload.get("version", "unknown")
        self.products: list[Product] = [Product(**item) for item in payload.get("products", [])]
        self.market_signals: dict[str, dict[str, Any]] = payload.get("market_signals", {})
        self.competitors: dict[str, list[dict[str, Any]]] = payload.get("competitors", {})

    def list_products(self, category: str | None = None) -> list[Product]:
        if category is None:
            return list(self.products)
        return [p for p in self.products if p.category == category]

    def get(self, sku: str) -> Product:
        for product in self.products:
            if product.sku.lower() == sku.lower():
                return product
        raise NotFound(f"product {sku} not found", details={"sku": sku})

    def find_by_name(self, name: str) -> Product | None:
        needle = name.strip().lower()
        if not needle:
            return None
        for product in self.products:
            if needle in product.name.lower() or product.name.lower() in needle:
                return product
        return None

    def resolve(self, *, sku: str | None = None, text: str = "") -> Product | None:
        if sku:
            try:
                return self.get(sku)
            except NotFound:
                return None
        return self.find_by_name(text)

    def signal_for(self, category: str) -> dict[str, Any]:
        return self.market_signals.get(category, {})

    def competitors_for(self, category: str) -> list[dict[str, Any]]:
        return self.competitors.get(category, [])

    def categories(self) -> list[str]:
        return sorted({p.category for p in self.products})

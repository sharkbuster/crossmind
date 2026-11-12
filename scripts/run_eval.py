"""Run the offline RAG evaluation and write reports/rag-eval.{json,md}."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossmind.config import Settings  # noqa: E402
from crossmind.container import build_container  # noqa: E402
from crossmind.rag.evaluator import to_markdown  # noqa: E402


async def main() -> int:
    settings = Settings()
    container = await build_container(settings)
    try:
        report = await container.evaluate(mode="offline" if settings.is_offline else "online")
        reports = ROOT / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "rag-eval.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        markdown = to_markdown(report, corpus_stats=container.corpus_stats)
        (reports / "rag-eval.md").write_text(markdown, encoding="utf-8")

        m = report.metrics
        print(f"mode={report.mode} cases={m.cases}")
        print(f"hit@{report.top_k}={m.hit_at_k:.4f}  MRR={m.mrr:.4f}")
        print(f"refusal_accuracy={m.refusal_accuracy:.4f}  citation_precision={m.citation_precision:.4f}")
        print(f"faithfulness={m.faithfulness:.4f}  key_point_coverage={m.key_point_coverage:.4f}")
        print(f"avg_latency_ms={m.avg_latency_ms:.2f}")
        print(f"report -> {reports / 'rag-eval.md'}")
        return 0 if m.refusal_accuracy >= 0.9 and m.hit_at_k >= 0.9 else 1
    finally:
        await container.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

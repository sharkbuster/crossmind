"""Command line entrypoint: ``crossmind serve|eval|smoke|version``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from crossmind import __version__


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "crossmind.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def _eval(args: argparse.Namespace) -> int:
    from crossmind.config import Settings
    from crossmind.container import REPO_ROOT, build_container
    from crossmind.rag.evaluator import to_markdown

    async def run() -> int:
        container = await build_container(Settings())
        try:
            report = await container.evaluate(mode="offline" if container.settings.is_offline else "online")
            out = REPO_ROOT / "reports"
            out.mkdir(exist_ok=True)
            (out / "rag-eval.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            (out / "rag-eval.md").write_text(
                to_markdown(report, corpus_stats=container.corpus_stats), encoding="utf-8"
            )
            print(json.dumps(report.metrics.model_dump(), ensure_ascii=False, indent=2))
            return 0
        finally:
            await container.aclose()

    return asyncio.run(run())


def _smoke(args: argparse.Namespace) -> int:
    from crossmind.config import Settings
    from crossmind.container import build_container

    async def run() -> int:
        container = await build_container(Settings())
        try:
            answer = await container.rag.answer("未拆封商品多久可以退货？")
            refusal = await container.rag.answer("你们线下门店在哪里？")
            checks = {
                "answer_cited": bool(answer.citations),
                "answer_grounded": answer.grounded,
                "refusal_works": refusal.no_answer,
                "corpus_chunks": len(container.chunks),
            }
            print(json.dumps(checks, ensure_ascii=False))
            return 0 if all(checks[key] for key in ("answer_cited", "refusal_works")) else 1
        finally:
            await container.aclose()

    return asyncio.run(run())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crossmind", description="CrossMind 跨境电商多智能体平台")
    parser.add_argument("--version", action="version", version=f"crossmind {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动 API 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=_serve)

    eval_cmd = sub.add_parser("eval", help="运行离线评测并写报告")
    eval_cmd.set_defaults(func=_eval)

    smoke = sub.add_parser("smoke", help="冒烟检查")
    smoke.set_defaults(func=_smoke)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

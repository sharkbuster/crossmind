"""Operational endpoints: traces, audit, metrics summary, evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from crossmind.api.deps import ContainerDep, EvalPrincipalDep, PrincipalDep
from crossmind.container import REPO_ROOT
from crossmind.domain.errors import NotFound
from crossmind.domain.models import EvalReport, TraceRecord
from crossmind.obs.metrics import REGISTRY
from crossmind.obs.tracing import get_tracer
from crossmind.rag.evaluator import to_markdown

router = APIRouter(tags=["ops"])
REPORT_DIR = REPO_ROOT / "reports"


@router.get("/ops/traces", response_model=list[TraceRecord], summary="最近的链路 Trace")
async def traces(
    _principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 30,
) -> list[TraceRecord]:
    return get_tracer().recent(limit)


@router.get("/ops/traces/{trace_id}", response_model=TraceRecord, summary="Trace 明细（span 树）")
async def trace_detail(trace_id: str, _principal: PrincipalDep) -> TraceRecord:
    record = get_tracer().get(trace_id)
    if record is None:
        raise NotFound(f"trace {trace_id} not found (ring buffer keeps the last 512)")
    return record


@router.get("/ops/audit", summary="审计日志")
async def audit(
    principal: PrincipalDep,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[dict[str, Any]]:
    return container.store.list_audit(principal.tenant_id, limit=limit)


@router.get("/ops/summary", summary="运行摘要（任务/指标/模型模式）")
async def summary(_principal: PrincipalDep, container: ContainerDep) -> dict[str, Any]:
    hist = REGISTRY.histograms["crossmind_http_request_duration_ms"]
    snapshot = hist.snapshot()
    return {
        "env": container.settings.env,
        "offline": container.settings.is_offline,
        "llm": {
            "provider": container.settings.llm_provider,
            "model": container.settings.llm_model,
            "primary": container.chat_model.name,
        },
        "corpus": container.corpus_stats,
        "rag": {
            "top_k": container.settings.rag_top_k,
            "score_threshold": container.settings.rag_score_threshold,
            "answerability_gate": container.settings.rag_answerability_gate,
            "answerability_threshold": container.settings.rag_answerability_threshold,
            "embed_weight": container.settings.rag_embed_weight,
        },
        "tasks": container.store.task_counts(),
        "http": {
            "requests": float(REGISTRY.counters["crossmind_http_requests_total"].value()),
            "avg_latency_ms": snapshot["avg"],
        },
        "traces_buffered": len(get_tracer().recent(limit=512)),
    }


@router.get("/eval/latest", response_model=EvalReport, summary="最近一次评测报告")
async def latest_report(_principal: PrincipalDep) -> EvalReport:
    path = REPORT_DIR / "rag-eval.json"
    if not path.exists():
        raise NotFound("no evaluation report yet; POST /eval/run first")
    return EvalReport(**json.loads(path.read_text("utf-8")))


@router.post("/eval/run", response_model=EvalReport, summary="重跑评测（离线约 1-3 秒）")
async def run_eval(principal: EvalPrincipalDep, container: ContainerDep) -> EvalReport:
    report = await container.evaluate(mode="offline" if container.settings.is_offline else "online")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "rag-eval.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "rag-eval.md").write_text(
        to_markdown(report, corpus_stats=container.corpus_stats), encoding="utf-8"
    )
    container.store.audit(
        principal.tenant_id,
        "eval.run",
        "rag-eval",
        f"hit@k={report.metrics.hit_at_k} refusal={report.metrics.refusal_accuracy}",
    )
    return report


@router.get("/eval/report.md", response_class=PlainTextResponse, summary="评测报告 Markdown")
async def report_markdown(_principal: PrincipalDep) -> PlainTextResponse:
    path = REPORT_DIR / "rag-eval.md"
    if not path.exists():
        raise NotFound("no evaluation report yet")
    return PlainTextResponse(path.read_text("utf-8"), media_type="text/markdown; charset=utf-8")


def metrics_payload() -> str:
    return REGISTRY.render()


__all__ = ["router", "metrics_payload", "REPORT_DIR"]

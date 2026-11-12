"""Offline evaluation harness.

Metrics are computed from the same pipeline the API serves, so the numbers in
docs/EVALUATION.md cannot drift from the product.  Two metric families:

* retrieval quality - hit@k, MRR (did the right document get retrieved at all);
* answer quality    - refusal correctness, citation precision, faithfulness,
                      key-point coverage (was the answer grounded and useful).

In offline (mock) mode these describe the *pipeline*, not a model.  That
distinction is stated in the generated report.
"""

from __future__ import annotations

from crossmind.domain.models import (
    EvalCase,
    EvalCaseResult,
    EvalMetrics,
    EvalReport,
)
from crossmind.obs.logging import get_logger
from crossmind.rag.citation import CITATION_RE, faithfulness, is_refusal
from crossmind.rag.pipeline import RagPipeline

log = get_logger(__name__)


def _normalise(text: str) -> str:
    return CITATION_RE.sub("", text).replace(" ", "").replace("\n", "")


def _key_point_coverage(answer: str, key_points: list[str]) -> float:
    if not key_points:
        return 1.0
    body = _normalise(answer)
    hits = sum(1 for point in key_points if _normalise(point) and _normalise(point) in body)
    return round(hits / len(key_points), 4)


def _citation_precision(cited_docs: list[str], expected_docs: list[str]) -> float:
    if not cited_docs:
        return 0.0
    if not expected_docs:
        return 0.0
    hits = sum(1 for doc in cited_docs if doc in expected_docs)
    return round(hits / len(cited_docs), 4)


def _reciprocal_rank(source_docs: list[str], expected_docs: list[str]) -> float:
    if not expected_docs:
        return 0.0
    for position, doc in enumerate(source_docs, start=1):
        if doc in expected_docs:
            return round(1.0 / position, 4)
    return 0.0


async def evaluate(pipeline: RagPipeline, cases: list[EvalCase], *, mode: str = "offline") -> EvalReport:
    results: list[EvalCaseResult] = []
    for case in cases:
        answer = await pipeline.answer(case.question)
        source_docs = [chunk.doc_id for chunk in answer.sources]
        cited_docs = [citation.doc_id for citation in answer.citations]
        hit = any(doc in case.expected_doc_ids for doc in source_docs) if case.expected_doc_ids else False
        results.append(
            EvalCaseResult(
                id=case.id,
                answerable=case.answerable,
                answered=not answer.no_answer and not is_refusal(answer.text),
                no_answer=answer.no_answer,
                grounded=answer.grounded,
                hit=hit,
                reciprocal_rank=_reciprocal_rank(source_docs, case.expected_doc_ids),
                citation_precision=_citation_precision(cited_docs, case.expected_doc_ids),
                faithfulness=faithfulness(answer.text, answer.citations)
                if answer.citations
                else 0.0,
                key_point_coverage=_key_point_coverage(answer.text, case.expected_key_points),
                latency_ms=answer.latency_ms,
            )
        )

    answerable = [r for r in results if r.answerable]
    evaluated = answerable or results
    metrics = EvalMetrics(
        cases=len(results),
        hit_at_k=round(sum(1 for r in answerable if r.hit) / len(answerable), 4) if answerable else 0.0,
        mrr=round(sum(r.reciprocal_rank for r in answerable) / len(answerable), 4) if answerable else 0.0,
        refusal_accuracy=round(
            sum(1 for r in results if r.no_answer == (not r.answerable)) / len(results), 4
        ),
        citation_precision=round(sum(r.citation_precision for r in answerable) / len(answerable), 4)
        if answerable
        else 0.0,
        faithfulness=round(sum(r.faithfulness for r in evaluated) / len(evaluated), 4),
        key_point_coverage=round(
            sum(r.key_point_coverage for r in evaluated) / len(evaluated), 4
        ),
        avg_latency_ms=round(sum(r.latency_ms for r in results) / len(results), 2),
    )
    report = EvalReport(
        mode=mode,
        top_k=pipeline.settings.rag_top_k,
        threshold=pipeline.settings.rag_score_threshold,
        metrics=metrics,
        results=results,
    )
    log.info("evaluation finished", extra={"mode": mode, **metrics.model_dump()})
    return report


def to_markdown(report: EvalReport, *, corpus_stats: dict[str, object] | None = None) -> str:
    m = report.metrics
    lines = [
        "# CrossMind RAG 评测报告",
        "",
        f"- 生成时间：{report.generated_at}",
        f"- 运行模式：`{report.mode}`",
        f"- 检索参数：top_k={report.top_k} · threshold={report.threshold}",
    ]
    if corpus_stats:
        lines.append(
            f"- 语料：{corpus_stats.get('documents')} 篇文档 / {corpus_stats.get('chunks')} 个知识块"
            f" / 向量维度 {corpus_stats.get('dim')}"
        )
    lines += [
        "",
        "## 汇总指标",
        "",
        "| 指标 | 值 | 含义 |",
        "|---|---|---|",
        f"| hit@{report.top_k} | {m.hit_at_k:.4f} | 正确答案文档出现在召回结果中的比例 |",
        f"| MRR | {m.mrr:.4f} | 正确文档的平均倒数排名 |",
        f"| 拒答正确率 | {m.refusal_accuracy:.4f} | 该答的答了、该拒的拒了 |",
        f"| 引用准确率 | {m.citation_precision:.4f} | 引用命中期望文档的比例 |",
        f"| 忠实度 | {m.faithfulness:.4f} | 答案 3-gram 被引用片段覆盖的比例 |",
        f"| 要点覆盖率 | {m.key_point_coverage:.4f} | 期望要点出现在答案中的比例 |",
        f"| 平均延迟 | {m.avg_latency_ms:.2f} ms | 单条问答端到端耗时 |",
        "",
        "## 逐条结果",
        "",
        "| 用例 | 可答 | 已答 | 拒答 | 命中 | RR | 引用准确 | 忠实度 | 要点覆盖 | 延迟(ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.results:
        lines.append(
            f"| {r.id} | {'是' if r.answerable else '否'} | {'是' if r.answered else '否'} | "
            f"{'是' if r.no_answer else '否'} | {'是' if r.hit else '否'} | {r.reciprocal_rank:.2f} | "
            f"{r.citation_precision:.2f} | {r.faithfulness:.2f} | {r.key_point_coverage:.2f} | "
            f"{r.latency_ms} |"
        )
    lines += [
        "",
        "## 口径说明",
        "",
        "- 离线（mock）模式下，生成器是确定性夹具，因此忠实度/要点覆盖率描述的是"
        "**流水线是否正确地把证据传下去**，不代表模型质量；对照真实模型时请切换 "
        "`XM_LLM_PROVIDER=openai` 后重跑。",
        "- 忠实度基于字符 3-gram 覆盖率：无法通过同义改写伪造支撑，但会轻微低估"
        "合理改写的答案，这是有意的保守口径。",
        "- 拒答正确性把「不可答问题被回答」计为错误，与「可答问题被拒答」同权。",
        "",
    ]
    return "\n".join(lines)

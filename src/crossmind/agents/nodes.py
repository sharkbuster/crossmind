"""Graph nodes.

Each node is a pure-ish async function of the state: it reads what it needs,
writes a partial update, and records a step so the console can replay the run.
Business logic lives here; the graph in ``graph.py`` only wires the edges.
"""

from __future__ import annotations

import json
import time
from typing import Any

from crossmind.agents.state import AgentState
from crossmind.agents.tools import ToolRegistry
from crossmind.container import Container
from crossmind.domain.models import SourceChunk
from crossmind.llm.base import Message
from crossmind.obs.logging import get_logger
from crossmind.obs.tracing import get_tracer
from crossmind.rag.citation import is_refusal

log = get_logger(__name__)

INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("translation", ("翻译", "translate", "本地化", "译成", "英文版", "日文版", "德语")),
    ("copywriting", ("文案", "标题", "卖点", "五点", "描述", "listing", "撰写", "copy")),
    ("selection", ("选品", "市场分析", "竞品", "品类", "定价策略", "值得做")),
]

LANG_BY_LOCALE = {"en-US": "en", "en": "en", "ja-JP": "ja", "de-DE": "de", "zh-CN": "zh"}


def classify(question: str) -> str:
    text = question.lower()
    for intent, keywords in INTENT_RULES:
        if any(keyword in text for keyword in keywords):
            return intent
    return "support"


def _task_prompt(task: str, payload: dict[str, Any], instructions: str) -> list[Message]:
    header = f"#TASK {task}\n#PAYLOAD {json.dumps(payload, ensure_ascii=False)}"
    return [
        Message(role="system", content="你是 CrossMind 平台的跨境电商运营专家，输出必须可直接交付使用。"),
        Message(role="user", content=f"{header}\n\n{instructions}"),
    ]


class Nodes:
    def __init__(self, container: Container) -> None:
        self.c = container
        self.tools = ToolRegistry(container.catalog)
        self.tracer = get_tracer()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _steps(state: AgentState, **step: Any) -> list[dict[str, Any]]:
        return [*(state.get("steps") or []), step]

    def _usage(self, state: AgentState, result: Any) -> dict[str, int]:
        current = state.get("usage") or {}
        return {
            "prompt_tokens": current.get("prompt_tokens", 0) + result.usage.prompt_tokens,
            "completion_tokens": current.get("completion_tokens", 0) + result.usage.completion_tokens,
        }

    # ------------------------------------------------------------------ nodes
    async def router(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        intent = classify(state["question"])
        latency = int((time.perf_counter() - started) * 1000)
        return {
            "intent": intent,
            "steps": self._steps(
                state, node="router", status="ok", detail=f"intent={intent}", latency_ms=latency
            ),
        }

    async def supervisor(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        payload = {
            "question": state["question"],
            "intent": state["intent"],
            "amount": state.get("amount", 0),
            "available_agents": ["researcher", "support", "copywriter", "translator", "selector", "critic"],
        }
        plan, risk, reason = self._fallback_plan(state)
        try:
            with self.tracer.span("agent.supervisor") as span:
                result = await self.c.chat_model.complete(
                    _task_prompt("supervisor", payload, "请输出 JSON：intent/steps/risk/reason"),
                    json_mode=True,
                    temperature=0.0,
                    max_tokens=300,
                )
                if span:
                    span.set(tokens=result.usage.total_tokens)
            parsed = json.loads(result.text)
            plan = parsed.get("steps") or plan
            risk = parsed.get("risk", risk)
            reason = parsed.get("reason", reason)
            usage = self._usage(state, result)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("supervisor output unparsable, using rule plan", extra={"error": str(exc)})
            usage = state.get("usage") or {}

        return {
            "plan": plan,
            "risk": risk,
            "approval_reason": reason,
            "usage": usage,
            "steps": self._steps(
                state,
                node="supervisor",
                status="ok",
                detail=f"plan={'→'.join(plan)} risk={risk}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            ),
        }

    def _fallback_plan(self, state: AgentState) -> tuple[list[str], str, str]:
        intent = state["intent"]
        plans = {
            "support": ["retrieve", "support", "critic"],
            "copywriting": ["catalog", "copywriting", "critic"],
            "translation": ["catalog", "translation", "critic"],
            "selection": ["retrieve", "selection", "critic"],
        }
        risk = "medium" if intent in {"copywriting", "selection"} else "low"
        return plans[intent], risk, f"rule plan for {intent}"

    async def research(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        intent = state["intent"]
        contexts: list[dict[str, Any]] = []
        detail = "no retrieval needed"
        if intent in {"support", "selection"}:
            with self.tracer.span("agent.research.retrieve", intent=intent) as span:
                context = await self.c.retriever.retrieve(state["question"])
                contexts = [chunk.model_dump() for chunk in context.chunks]
                if span:
                    span.set(chunks=len(contexts), best_score=context.best_score, no_answer=context.no_answer)
            detail = f"retrieved={len(contexts)} no_answer={context.no_answer}"
        elif intent in {"copywriting", "translation"}:
            product = self.c.catalog.resolve(
                sku=state.get("product_sku") or None, text=state["question"]
            )
            tool_result = self.tools.call(
                "catalog_lookup", sku=product.sku if product else "", name=state["question"]
            )
            detail = f"catalog_hit={bool(product)}"
            contexts = (
                [
                    {
                        "chunk_id": f"catalog::{product.sku}",
                        "doc_id": product.sku,
                        "doc_title": product.name,
                        "section": "商品主数据",
                        "uri": "data/catalog.json",
                        "text": json.dumps(product.model_dump(), ensure_ascii=False),
                        "score": 1.0,
                        "rank": 1,
                    }
                ]
                if tool_result.get("ok") and product
                else []
            )
        return {
            "contexts": contexts,
            "steps": self._steps(
                state,
                node="research",
                status="ok",
                detail=detail,
                latency_ms=int((time.perf_counter() - started) * 1000),
            ),
        }

    async def specialist(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        intent = state["intent"]
        chunks = [SourceChunk(**item) for item in state.get("contexts") or []]
        feedback = state.get("feedback") or ""
        citations: list[dict[str, Any]] = []
        text = ""
        usage = state.get("usage") or {}

        if intent == "support":
            answer = await self.c.rag.answer_with_context(
                state["question"],
                chunks,
                conversation=self._history(state),
                locale=state.get("locale", "zh-CN"),
                require_citations=state.get("require_citations", True),
            )
            text = answer.text
            citations = [citation.model_dump() for citation in answer.citations]
            usage = self._usage(state, answer)
            if answer.no_answer:
                text = answer.text
        elif intent == "copywriting":
            text, result = await self._copywriting(state, feedback)
            usage = self._usage(state, result)
        elif intent == "translation":
            text, result = await self._translation(state)
            usage = self._usage(state, result)
        else:
            text, result = await self._selection(state)
            usage = self._usage(state, result)

        return {
            "draft": text,
            "citations": citations,
            "usage": usage,
            "steps": self._steps(
                state,
                node=f"specialist.{intent}",
                status="ok",
                detail=f"chars={len(text)} citations={len(citations)}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            ),
        }

    async def _copywriting(self, state: AgentState, feedback: str) -> tuple[str, Any]:
        contexts = state.get("contexts") or []
        product = json.loads(contexts[0]["text"]) if contexts else {}
        payload = {
            "product": product,
            "channel": state.get("channel", "amazon"),
            "locale": state.get("locale", "en-US"),
            "features": product.get("tags", []),
            "feedback": feedback,
        }
        instructions = (
            "为商品撰写 Amazon 风格标题与五点描述。要求：标题不超过 200 字符；"
            "五点描述每条以动词开头；不得出现绝对化用词与未验证的认证。"
            + (f"\n上轮评审意见：{feedback}" if feedback else "")
        )
        result = await self.c.chat_model.complete(
            _task_prompt("copywriting", payload, instructions), temperature=0.4, max_tokens=700
        )
        return result.text, result

    async def _translation(self, state: AgentState) -> tuple[str, Any]:
        contexts = state.get("contexts") or []
        source = json.loads(contexts[0]["text"]) if contexts else {}
        text = source.get("description") or state["question"]
        target = LANG_BY_LOCALE.get(state.get("locale", "en-US"), "en")
        payload = {
            "text": text,
            "target_lang": target,
            "glossary": {"包邮": "free shipping", "保修": "warranty", "续航": "battery life"},
            "tone": "marketplace-listing",
        }
        instructions = f"将商品描述本地化为目标语言 {target}，保留型号与数字，术语按 glossary 统一。"
        result = await self.c.chat_model.complete(
            _task_prompt("translation", payload, instructions), temperature=0.1, max_tokens=500
        )
        return result.text, result

    async def _selection(self, state: AgentState) -> tuple[str, Any]:
        contexts = state.get("contexts") or []
        category = self._infer_category(state["question"], contexts)
        product = self.c.catalog.resolve(sku=state.get("product_sku") or None, text=state["question"])
        payload = {
            "category": category,
            "market": self.c.catalog.signal_for(category),
            "competitors": self.c.catalog.competitors_for(category),
            "candidate": product.model_dump() if product else None,
        }
        instructions = (
            "输出选品分析报告，包含三段：市场分析、竞品分析、选品建议。"
            "每段给出可执行结论，不要罗列空话。"
        )
        result = await self.c.chat_model.complete(
            _task_prompt("selection", payload, instructions), temperature=0.3, max_tokens=900
        )
        return result.text, result

    @staticmethod
    def _infer_category(question: str, contexts: list[dict[str, Any]]) -> str:
        for chunk in contexts:
            text = chunk.get("text", "")
            for category in ("音频", "可穿戴", "充电配件", "清洁电器"):
                if category in text:
                    return category
        for category in ("音频", "可穿戴", "充电配件", "清洁电器"):
            if category in question:
                return category
        return "音频"

    async def critic(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        payload = {
            "answer": state.get("draft", ""),
            "contexts": state.get("contexts") or [],
            "requires_citations": state.get("intent") == "support",
            "intent": state.get("intent"),
        }
        review: dict[str, Any] = {"ok": True, "issues": [], "score": 1.0}
        usage = state.get("usage") or {}
        try:
            result = await self.c.chat_model.complete(
                _task_prompt("critic", payload, "请输出 JSON：grounded/score/issues"),
                json_mode=True,
                temperature=0.0,
                max_tokens=300,
            )
            parsed = json.loads(result.text)
            review = {
                "ok": bool(parsed.get("grounded", False)),
                "score": float(parsed.get("score", 0.0)),
                "issues": parsed.get("issues") or [],
            }
            usage = self._usage(state, result)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            review = {"ok": False, "score": 0.0, "issues": [f"critic_output_error: {exc}"]}

        if is_refusal(state.get("draft", "")):
            review["ok"] = True
            review["issues"] = []

        rounds = int(state.get("rounds", 0)) + 1
        feedback = "; ".join(review["issues"])
        return {
            "review": review,
            "rounds": rounds,
            "feedback": feedback,
            "usage": usage,
            "steps": self._steps(
                state,
                node="critic",
                status="ok" if review["ok"] else "revise",
                detail=f"score={review['score']} issues={len(review['issues'])} round={rounds}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            ),
        }

    async def hitl_gate(self, state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        risk = state.get("risk", "low")
        reasons: list[str] = []
        if intent in {"copywriting", "selection"}:
            reasons.append(f"{intent} 产出将对外发布，需人工确认")
        if risk == "high":
            reasons.append("supervisor 判定为高风险任务")
        requires = bool(reasons) and not state.get("approved")

        if not requires:
            return {
                "requires_approval": False,
                "status": "running",
                "steps": self._steps(
                    state,
                    node="hitl_gate",
                    status="skip",
                    detail="no approval required",
                    latency_ms=0,
                ),
            }
        return {
            "requires_approval": True,
            "approval_reason": "; ".join(reasons),
            "status": "awaiting_approval",
            "steps": self._steps(
                state,
                node="hitl_gate",
                status="pending",
                detail="; ".join(reasons),
                latency_ms=0,
            ),
        }

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        draft = state.get("draft", "").strip()
        status = "rejected" if state.get("approval_decision") == "reject" else "succeeded"
        if status == "rejected":
            draft = "已按人工审核意见驳回本次产出，可调整需求后重新提交。"
        return {
            "status": status,
            "draft": draft,
            "steps": self._steps(
                state,
                node="finalize",
                status="ok",
                detail=f"status={status}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            ),
        }

    # ------------------------------------------------------------------ misc
    @staticmethod
    def _history(state: AgentState) -> list[Message]:
        return [
            Message(role=item["role"], content=item["content"])
            for item in (state.get("history") or [])
            if item.get("role") in {"user", "assistant"}
        ]

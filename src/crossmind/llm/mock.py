"""Deterministic offline model.

This is what makes the platform runnable with **zero API keys**: every node of
the graph can be exercised end-to-end, asserted in tests, and streamed to the
console without any network call.

It is deliberately *not* a language model.  It is a fixture renderer: prompts
carry a machine header (``#TASK`` / ``#PAYLOAD``) and this renderer turns the
payload into a plausible, deterministic artefact.  Quality metrics measured in
this mode describe the pipeline, never the model - see docs/EVALUATION.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from typing import Any

from crossmind.domain.models import Usage
from crossmind.llm.base import ChatModel, ChatResult, Message, estimate_tokens

HEADER_RE = re.compile(r"^#TASK\s+(?P<task>[a-z_]+)\s*$", re.MULTILINE)
PAYLOAD_RE = re.compile(r"^#PAYLOAD\s+(?P<payload>.+)$", re.MULTILINE)
CTX_RE = re.compile(
    r"^\[(?P<idx>\d+)\]\s+doc_id=(?P<doc>\S+)\s*\|\s*title=(?P<title>.*?)\s*\|\s*section=(?P<section>.*?)\s*$"
    r"\n(?P<text>.*?)(?=\n\[\d+\]|\Z)",
    re.MULTILINE | re.DOTALL,
)


def parse_task(prompt: str) -> tuple[str, dict[str, Any]]:
    task_match = HEADER_RE.search(prompt)
    task = task_match.group("task") if task_match else "generic"
    payload: dict[str, Any] = {}
    payload_match = PAYLOAD_RE.search(prompt)
    if payload_match:
        try:
            payload = json.loads(payload_match.group("payload"))
        except json.JSONDecodeError:
            payload = {}
    return task, payload


def parse_contexts(prompt: str) -> list[dict[str, str]]:
    contexts = []
    for match in CTX_RE.finditer(prompt):
        contexts.append(
            {
                "idx": match.group("idx"),
                "doc_id": match.group("doc"),
                "title": match.group("title"),
                "section": match.group("section"),
                "text": match.group("text").strip(),
            }
        )
    return contexts


def _first_sentence(text: str, limit: int = 90) -> str:
    text = text.strip().replace("\n", " ")
    for sep in ("。", "；", "，", ". "):
        idx = text.find(sep)
        if 8 <= idx <= limit:
            return text[: idx + len(sep)].strip()
    return text[:limit].strip()


def _stable_pick(options: list[str], seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


SENTENCE_SPLIT_RE = re.compile(r"[^。！？；\n]+[。！？；]?")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.findall(text) if s.strip()]


def _char_bigrams(text: str) -> set[str]:
    compact = re.sub(r"[\s，。；：！？、“”（）()\[\]【】]", "", text)
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


STOP_GRAMS = {
    "多少", "多久", "什么", "怎么", "可以", "是否", "支持", "需要", "时间", "时候",
    "一般", "大概", "几天", "多长", "哪些", "如何", "是不", "不是", "能否", "你们",
}


class MockChatModel(ChatModel):
    name = "mock"

    def __init__(self, latency_ms: int = 0, model_label: str = "mock-deterministic") -> None:
        self.latency_ms = latency_ms
        self.model_label = model_label

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> ChatResult:
        prompt = "\n".join(m.content for m in messages if m.role != "system")
        system = "\n".join(m.content for m in messages if m.role == "system")
        task, payload = parse_task(prompt)
        if task == "generic":
            task, payload = parse_task(system + "\n" + prompt)

        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)

        text = self._render(task, payload, prompt, system)
        if json_mode and not text.lstrip().startswith("{"):
            text = json.dumps({"text": text}, ensure_ascii=False)

        return ChatResult(
            text=text,
            usage=Usage(
                prompt_tokens=estimate_tokens(prompt + system),
                completion_tokens=estimate_tokens(text),
            ),
            model=self.model_label,
            finish_reason="stop",
        )

    # ------------------------------------------------------------------ renderers
    def _render(self, task: str, payload: dict[str, Any], prompt: str, system: str) -> str:
        renderer = getattr(self, f"_render_{task}", None)
        if renderer is not None:
            return renderer(payload, prompt, system)
        return self._render_generic(payload, prompt)

    def _render_support(self, payload: dict[str, Any], prompt: str, system: str) -> str:
        question = str(payload.get("question", "")).strip()
        contexts = payload.get("contexts") or parse_contexts(prompt)
        if not contexts:
            return "抱歉，我在知识库中没有找到能回答这个问题的依据，已为你转接人工客服。"

        pool: list[tuple[int, str, str, set[str]]] = []
        for position, ctx in enumerate(contexts, start=1):
            section = f"{ctx.get('title', '')}{ctx.get('section', '')}"
            for sentence in _sentences(ctx.get("text", "")):
                pool.append((position, section, sentence, _char_bigrams(sentence)))
        if not pool:
            return "抱歉，我在知识库中没有找到能回答这个问题的依据，已为你转接人工客服。"

        # IDF over the candidate sentences: an attribute the question asks about
        # ("吸力", "续航") is rare inside the context and therefore decisive,
        # while the product name (present in every sentence) is not.
        df: dict[str, int] = {}
        for _, _, _, grams in pool:
            for gram in grams:
                df[gram] = df.get(gram, 0) + 1
        total = len(pool)

        def weight(gram: str) -> float:
            return math.log((total + 1) / (df.get(gram, 0) + 1)) + 1.0

        q_grams = {g for g in _char_bigrams(question) if g not in STOP_GRAMS}
        # The document title is the subject of the question, not the ask: every
        # sentence in the document matches it, so it carries no discriminating
        # signal.  Removing it is what makes "续航多久" land on the spec sentence
        # instead of the marketing intro.
        topic_grams: set[str] = set()
        for ctx in contexts:
            topic_grams |= _char_bigrams(str(ctx.get("title", "")))
        q_grams -= topic_grams
        if not q_grams:
            q_grams = {g for g in _char_bigrams(question) if g not in STOP_GRAMS} or _char_bigrams(question)
        denominator = sum(weight(g) for g in q_grams) or 1.0
        asks_quantity = bool(re.search(r"多少|多久|几天|多长|几个|多重|多大", question))

        best: tuple[float, int, str] | None = None
        for position, section, sentence, grams in pool:
            matched = sum(weight(g) for g in (q_grams & grams))
            score = matched / denominator
            if section and (q_grams & _char_bigrams(section)):
                score += 0.1
            if asks_quantity and re.search(r"\d", sentence):
                score += 0.15
            if best is None or score > best[0]:
                best = (score, position, sentence)
        if best is None:
            return "抱歉，我在知识库中没有找到能回答这个问题的依据，已为你转接人工客服。"

        score, position, sentence = best
        picked = [sentence.strip("。； ")]
        for other_pos, _, other, grams in pool:
            if other_pos != position or other.strip() == sentence:
                continue
            if len("；".join(picked)) >= 90:
                break
            matched = sum(weight(g) for g in (q_grams & grams)) / denominator
            if matched >= max(0.2, score * 0.6):
                picked.append(other.strip("。； "))
                break
        body = "；".join(picked)
        return f"{body}【{position}】"

    def _render_copywriting(self, payload: dict[str, Any], prompt: str, system: str) -> str:
        product = payload.get("product") or {}
        name = product.get("name") or payload.get("name") or "这款商品"
        category = product.get("category", "")
        features = product.get("features") or payload.get("features") or []
        channel = payload.get("channel", "amazon")
        locale = payload.get("locale", "en")
        if not features:
            features = [f"适合{category}场景", "轻量便携", "一年质保"]

        title = f"{name} | {features[0]} | 适配{channel.upper()}" if locale.startswith("zh") else \
            f"{name} - {features[0]} for {category or 'everyday use'}"
        bullets = []
        for i, feat in enumerate(features[:5], start=1):
            bullets.append(f"{i}. {feat}：在真实使用场景中稳定可靠，减少售后沟通成本。")
        tail = "（文案由 CrossMind 生成，发布前请人工复核合规与商标风险。）"
        return "\n".join([f"标题：{title}", "", "五点描述：", *bullets, "", tail])

    def _render_translation(self, payload: dict[str, Any], prompt: str, system: str) -> str:
        text = str(payload.get("text", "")).strip()
        target = str(payload.get("target_lang", "en")).lower()
        glossary: dict[str, str] = payload.get("glossary") or {}
        out = text
        for src, dst in glossary.items():
            out = out.replace(src, dst)
        if target.startswith("en") and out == text:
            out = f"[{target}] {text}"
        return out

    def _render_selection(self, payload: dict[str, Any], prompt: str, system: str) -> str:
        category = payload.get("category", "目标品类")
        market = payload.get("market") or {}
        competitors = payload.get("competitors") or []
        growth = market.get("growth_rate", "—")
        median_price = market.get("median_price", "—")
        lines = [
            f"## 市场分析\n{category}近 12 个月需求增长 {growth}，价格带中位数 {median_price} 美元，"
            f"头部集中度中等，仍有差异化空间。",
            "## 竞品分析",
        ]
        for comp in competitors[:3]:
            lines.append(
                f"- {comp.get('name', '竞品')}：售价 {comp.get('price', '—')} 美元，"
                f"评分 {comp.get('rating', '—')}，主要优势 {comp.get('strength', '品牌')}。"
            )
        if not competitors:
            lines.append("- 暂无可比竞品数据，建议先补齐 3 个对标 ASIN。")
        lines.append(
            f"## 选品建议\n建议以 {median_price} 美元附近的差异化款切入，首单小批量测试，"
            "用 2 周广告数据验证转化后再放量。"
        )
        return "\n".join(lines)

    def _render_critic(self, payload: dict[str, Any], prompt: str, system: str) -> str:
        answer = str(payload.get("answer", ""))
        contexts = payload.get("contexts") or parse_contexts(prompt)
        cited = [int(x) for x in re.findall(r"【(\d+)】", answer)]
        grounded = bool(cited) and all(1 <= c <= len(contexts) for c in cited)
        issues: list[str] = []
        if not cited:
            issues.append("answer carries no citation marker")
        if any(c > len(contexts) for c in cited):
            issues.append("citation index out of range")
        if len(answer) > 1200:
            issues.append("answer exceeds channel length budget")
        score = 1.0 if grounded and not issues else 0.5
        return json.dumps(
            {"grounded": grounded and not issues, "score": score, "issues": issues},
            ensure_ascii=False,
        )

    def _render_supervisor(self, payload: dict[str, Any], prompt: str, system: str) -> str:
        question = str(payload.get("question", ""))
        intent = str(payload.get("intent") or _stable_pick(["support", "copywriting", "translation", "selection"], question))
        steps = {
            "support": ["retrieve", "support", "critic"],
            "copywriting": ["catalog", "copywriting", "critic"],
            "translation": ["catalog", "translation", "critic"],
            "selection": ["retrieve", "selection", "critic"],
        }[intent]
        risk = "high" if payload.get("amount", 0) and float(payload["amount"]) >= 500 else "medium"
        return json.dumps(
            {"intent": intent, "steps": steps, "risk": risk, "reason": f"deterministic plan for {intent}"},
            ensure_ascii=False,
        )

    def _render_generic(self, payload: dict[str, Any], prompt: str) -> str:
        text = str(payload.get("input") or payload.get("text") or "").strip()
        if text:
            return text
        return _stable_pick(
            ["已处理完成。", "已按你的要求生成结果。", "任务已完成，请查看结构化输出。"], prompt
        )

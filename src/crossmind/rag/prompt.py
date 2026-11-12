"""Prompt construction for the knowledge-grounded answer path.

The machine header (``#TASK`` / ``#PAYLOAD``) exists so the offline fixture
model can render deterministic output; a real model simply reads it as a
comment.  Instructions are kept in one place so the behaviour that evaluation
measures is the behaviour that ships.
"""

from __future__ import annotations

import json

from crossmind.domain.models import SourceChunk
from crossmind.llm.base import Message
from crossmind.rag.citation import build_context_block

SUPPORT_SYSTEM = """你是 CrossMind 平台的跨境电商客服专家。
规则（必须全部遵守）：
1. 只能依据【知识片段】回答，不得使用片段之外的知识，不得编造政策、时效、金额。
2. 每个结论后必须标注片段编号，格式为【1】【2】，编号必须来自给定片段。
3. 如果片段不足以回答，直接说明没有依据并转人工，不要猜测。
4. 用简体中文回答，控制在 120 字以内，语气专业、克制。"""


def build_support_messages(
    question: str,
    chunks: list[SourceChunk],
    *,
    locale: str = "zh-CN",
    conversation: list[Message] | None = None,
) -> list[Message]:
    payload = {
        "question": question,
        "locale": locale,
        "contexts": [
            {
                "idx": str(i),
                "doc_id": c.doc_id,
                "title": c.doc_title,
                "section": c.section,
                "text": c.text,
            }
            for i, c in enumerate(chunks, start=1)
        ],
    }
    header = f"#TASK support\n#PAYLOAD {json.dumps(payload, ensure_ascii=False)}\n"
    user = (
        f"{header}\n【知识片段】\n{build_context_block(chunks)}\n\n"
        f"【用户问题】\n{question}\n\n请依据片段作答，并在每个结论后标注【n】。"
    )
    messages: list[Message] = [Message(role="system", content=SUPPORT_SYSTEM)]
    if conversation:
        messages.extend(conversation[-4:])
    messages.append(Message(role="user", content=user))
    return messages

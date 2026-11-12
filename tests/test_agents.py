"""Agent graph behaviour: routing, plan, HITL and resume."""

from __future__ import annotations

from crossmind.agents.graph import resume_after_approval
from crossmind.agents.nodes import classify
from crossmind.agents.state import new_state


def test_intent_classification() -> None:
    assert classify("把这段描述翻译成英文") == "translation"
    assert classify("给这款音箱写个标题和卖点") == "copywriting"
    assert classify("帮我分析一下音频品类的选品机会") == "selection"
    assert classify("未拆封可以退货吗") == "support"


async def test_graph_produces_cited_support_answer(container) -> None:
    state = new_state(
        question="未拆封商品多久可以退货？",
        tenant_id="tenant-test",
        session_id="s1",
        task_id="t1",
        trace_id="trace-1",
    )
    result = await container.graph.ainvoke(state)
    assert result["intent"] == "support"
    assert result["status"] == "succeeded"
    assert result["draft"]
    assert result["citations"]
    nodes = [step["node"] for step in result["steps"]]
    assert nodes[0] == "router"
    assert "supervisor" in nodes
    assert "critic" in nodes
    assert "finalize" in nodes


async def test_graph_pauses_copywriting_for_approval(container) -> None:
    state = new_state(
        question="给 AirSound 写一版 Amazon 文案",
        tenant_id="tenant-test",
        session_id="s2",
        task_id="t2",
        trace_id="trace-2",
    )
    result = await container.graph.ainvoke(state)
    assert result["intent"] == "copywriting"
    assert result["requires_approval"] is True
    assert result["status"] == "awaiting_approval"
    assert "finalize" not in [step["node"] for step in result["steps"]]


async def test_resume_after_approval_finishes_the_task(container) -> None:
    state = new_state(
        question="给 AirSound 写一版 Amazon 文案",
        tenant_id="tenant-test",
        session_id="s3",
        task_id="t3",
        trace_id="trace-3",
    )
    paused = await container.graph.ainvoke(state)
    resumed = await resume_after_approval(container, paused, decision="approve")
    assert resumed["status"] == "succeeded"
    assert resumed["draft"]


async def test_resume_with_rejection(container) -> None:
    state = new_state(
        question="给我一份音频品类选品分析",
        tenant_id="tenant-test",
        session_id="s4",
        task_id="t4",
        trace_id="trace-4",
    )
    paused = await container.graph.ainvoke(state)
    assert paused["status"] == "awaiting_approval"
    resumed = await resume_after_approval(container, paused, decision="reject")
    assert resumed["status"] == "rejected"
    assert "驳回" in resumed["draft"]


async def test_selection_output_has_three_sections(container) -> None:
    state = new_state(
        question="音频品类值得做吗？给我选品分析",
        tenant_id="tenant-test",
        session_id="s5",
        task_id="t5",
        trace_id="trace-5",
    )
    result = await container.graph.ainvoke(state)
    assert "市场分析" in result["draft"]
    assert "竞品分析" in result["draft"]
    assert "选品建议" in result["draft"]


async def test_translation_uses_catalog_description(container) -> None:
    state = new_state(
        question="把 PowerGo 的描述翻译成英文",
        tenant_id="tenant-test",
        session_id="s6",
        task_id="t6",
        trace_id="trace-6",
        locale="en-US",
        product_sku="CB-PG-003",
    )
    result = await container.graph.ainvoke(state)
    assert result["intent"] == "translation"
    assert "65W" in result["draft"]
    assert result["draft"].startswith("[en]")


async def test_steps_record_latency(container) -> None:
    state = new_state(
        question="积分怎么用",
        tenant_id="tenant-test",
        session_id="s7",
        task_id="t7",
        trace_id="trace-7",
    )
    result = await container.graph.ainvoke(state)
    for step in result["steps"]:
        assert "latency_ms" in step
        assert step["latency_ms"] >= 0


async def test_usage_is_accumulated(container) -> None:
    state = new_state(
        question="退款多久到账",
        tenant_id="tenant-test",
        session_id="s8",
        task_id="t8",
        trace_id="trace-8",
    )
    result = await container.graph.ainvoke(state)
    assert result["usage"]["prompt_tokens"] > 0
    assert result["usage"]["completion_tokens"] > 0

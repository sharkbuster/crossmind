"""LangGraph wiring.

    router -> supervisor -> research -> specialist -> critic
                                            ^            |
                                            +-- revise --+
                                                         |
                                                    hitl_gate
                                                     |      |
                                                finalize   END (awaiting approval)

The revise loop is bounded by ``XM_MAX_REVIEW_ROUNDS`` so a critic that keeps
rejecting cannot spin forever; the graph always terminates in a persisted state.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from crossmind.agents.nodes import Nodes
from crossmind.agents.state import AgentState
from crossmind.container import Container

NODE_NAMES = ("router", "supervisor", "research", "specialist", "critic", "hitl_gate", "finalize")


def build_graph(container: Container) -> Any:
    nodes = Nodes(container)
    max_rounds = container.settings.max_review_rounds

    def route_after_critic(state: AgentState) -> str:
        review = state.get("review") or {}
        if review.get("ok", False):
            return "gate"
        if int(state.get("rounds", 0)) >= max_rounds:
            return "gate"
        return "specialist"

    def route_after_gate(state: AgentState) -> str:
        return "end" if state.get("requires_approval") else "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("router", nodes.router)
    graph.add_node("supervisor", nodes.supervisor)
    graph.add_node("research", nodes.research)
    graph.add_node("specialist", nodes.specialist)
    graph.add_node("critic", nodes.critic)
    graph.add_node("hitl_gate", nodes.hitl_gate)
    graph.add_node("finalize", nodes.finalize)

    graph.set_entry_point("router")
    graph.add_edge("router", "supervisor")
    graph.add_edge("supervisor", "research")
    graph.add_edge("research", "specialist")
    graph.add_edge("specialist", "critic")
    graph.add_conditional_edges(
        "critic", route_after_critic, {"specialist": "specialist", "gate": "hitl_gate"}
    )
    graph.add_conditional_edges("hitl_gate", route_after_gate, {"finalize": "finalize", "end": END})
    graph.add_edge("finalize", END)
    return graph.compile()


async def resume_after_approval(
    container: Container, state: AgentState, *, decision: str
) -> AgentState:
    """Continue a paused run without re-executing the expensive nodes.

    The state was persisted before pausing, so approval is a state transition on
    the stored snapshot - it survives a process restart, which a blocking
    interrupt would not.
    """
    nodes = Nodes(container)
    patched: AgentState = AgentState(**state)  # type: ignore[typeddict-item]
    patched["approved"] = decision == "approve"
    patched["approval_decision"] = decision
    patched["requires_approval"] = False
    update = await nodes.finalize(patched)
    patched.update(update)  # type: ignore[typeddict-item]
    return patched

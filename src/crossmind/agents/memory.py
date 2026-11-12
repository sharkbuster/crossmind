"""Conversation memory.

Short-term memory is the recent turn window (what the support agent needs to
resolve "那运费呢?"), and it is persisted per (tenant, session) so a session can
resume after a restart.  Long-term preference memory is intentionally out of
scope for v1.0 and listed in the roadmap rather than half-implemented.
"""

from __future__ import annotations

from crossmind.storage.db import Store

MAX_TURNS = 12


class ConversationMemory:
    def __init__(self, store: Store, max_turns: int = MAX_TURNS) -> None:
        self.store = store
        self.max_turns = max_turns

    def history(self, tenant_id: str, session_id: str) -> list[dict[str, str]]:
        rows = self.store.list_memory(tenant_id, session_id, limit=self.max_turns)
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append(self, tenant_id: str, session_id: str, role: str, content: str) -> None:
        if not content.strip():
            return
        self.store.add_memory(tenant_id, session_id, role, content[:4000])

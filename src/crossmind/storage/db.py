"""SQLite persistence.

One file, no server, WAL enabled: enough for tasks, idempotency keys, conversation
memory and the audit trail, and it keeps the whole platform startable with a
single command.  The access layer is deliberately thin (explicit SQL, dict rows)
so swapping in Postgres later is a repository change, not a rewrite.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from crossmind.domain.models import TaskRecord, TaskStatus, TaskType, now_iso

IDEMPOTENCY_PENDING = "__pending__"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    risk TEXT NOT NULL,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    trace_id TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant_id, created_at DESC);

CREATE TABLE IF NOT EXISTS approvals (
    task_id TEXT PRIMARY KEY,
    decision TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory(tenant_id, session_id, id);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    action TEXT NOT NULL,
    resource TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    trace_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit(tenant_id, id DESC);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- tasks
    def create_task(self, task: TaskRecord) -> TaskRecord:
        with self._lock:
            self._conn.execute(
                "INSERT INTO tasks (id, tenant_id, type, status, risk, input, output, trace_id, error,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task.id,
                    task.tenant_id,
                    str(task.type),
                    str(task.status),
                    str(task.risk),
                    json.dumps(task.input, ensure_ascii=False),
                    json.dumps(task.output, ensure_ascii=False),
                    task.trace_id,
                    task.error,
                    task.created_at,
                    task.updated_at,
                ),
            )
            self._conn.commit()
        return task

    def update_task(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        output: dict[str, Any] | None = None,
        error: str | None = None,
        task_type: TaskType | None = None,
    ) -> None:
        sets = ["updated_at = ?"]
        values: list[Any] = [now_iso()]
        if status is not None:
            sets.append("status = ?")
            values.append(str(status))
        if output is not None:
            sets.append("output = ?")
            values.append(json.dumps(output, ensure_ascii=False))
        if error is not None:
            sets.append("error = ?")
            values.append(error)
        if task_type is not None:
            sets.append("type = ?")
            values.append(str(task_type))
        values.append(task_id)
        with self._lock:
            self._conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", values)
            self._conn.commit()

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(self, tenant_id: str, limit: int = 20) -> list[TaskRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
                (tenant_id, limit),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            id=row["id"],
            tenant_id=row["tenant_id"],
            type=TaskType(row["type"]),
            status=TaskStatus(row["status"]),
            risk=row["risk"],
            input=json.loads(row["input"]),
            output=json.loads(row["output"]),
            trace_id=row["trace_id"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ---------------------------------------------------------------- approvals
    def save_approval(self, task_id: str, decision: str, reviewer: str, comment: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO approvals (task_id, decision, reviewer, comment, created_at)"
                " VALUES (?,?,?,?,?)",
                (task_id, decision, reviewer, comment, now_iso()),
            )
            self._conn.commit()

    def get_approval(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM approvals WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    # ---------------------------------------------------------------- idempotency
    def get_idempotent(self, key: str, tenant_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM idempotency WHERE key = ? AND tenant_id = ?", (key, tenant_id)
            ).fetchone()
        if row is None or row["response"] == IDEMPOTENCY_PENDING:
            return None
        return json.loads(row["response"])

    def reserve_idempotent(
        self, key: str, tenant_id: str, *, ttl_s: int = 0
    ) -> tuple[str, dict[str, Any] | None]:
        """Atomically claim an idempotency key.

        Returns ``("reserved", None)`` when the caller owns the key,
        ``("pending", None)`` when an identical request is still running, and
        ``("replay", response)`` when a completed response can be reused.
        """
        with self._lock:
            if ttl_s:
                self._purge_idempotent(ttl_s)
            row = self._conn.execute(
                "SELECT response FROM idempotency WHERE key = ? AND tenant_id = ?", (key, tenant_id)
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT OR IGNORE INTO idempotency (key, tenant_id, response, created_at)"
                    " VALUES (?,?,?,?)",
                    (key, tenant_id, IDEMPOTENCY_PENDING, now_iso()),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT response FROM idempotency WHERE key = ? AND tenant_id = ?",
                    (key, tenant_id),
                ).fetchone()
                if row is not None and row["response"] == IDEMPOTENCY_PENDING:
                    return "reserved", None
            if row is None:
                return "pending", None
            if row["response"] == IDEMPOTENCY_PENDING:
                return "pending", None
            return "replay", json.loads(row["response"])

    def complete_idempotent(self, key: str, tenant_id: str, response: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE idempotency SET response = ?, created_at = ? WHERE key = ? AND tenant_id = ?",
                (json.dumps(response, ensure_ascii=False), now_iso(), key, tenant_id),
            )
            self._conn.commit()

    def release_idempotent(self, key: str, tenant_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM idempotency WHERE key = ? AND tenant_id = ? AND response = ?",
                (key, tenant_id, IDEMPOTENCY_PENDING),
            )
            self._conn.commit()

    def _purge_idempotent(self, ttl_s: int) -> None:
        cutoff = (datetime.now(UTC) - timedelta(seconds=ttl_s)).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
        self._conn.execute("DELETE FROM idempotency WHERE created_at < ?", (cutoff,))

    def put_idempotent(self, key: str, tenant_id: str, response: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO idempotency (key, tenant_id, response, created_at) VALUES (?,?,?,?)",
                (key, tenant_id, json.dumps(response, ensure_ascii=False), now_iso()),
            )
            self._conn.commit()

    # ---------------------------------------------------------------- memory
    def add_memory(self, tenant_id: str, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory (tenant_id, session_id, role, content, created_at) VALUES (?,?,?,?,?)",
                (tenant_id, session_id, role, content, now_iso()),
            )
            self._conn.commit()

    def list_memory(self, tenant_id: str, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, created_at FROM memory WHERE tenant_id = ? AND session_id = ?"
                " ORDER BY id DESC LIMIT ?",
                (tenant_id, session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    # ---------------------------------------------------------------- audit
    def audit(self, tenant_id: str, action: str, resource: str, detail: str = "", trace_id: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit (tenant_id, action, resource, detail, trace_id, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (tenant_id, action, resource, detail[:1000], trace_id, now_iso()),
            )
            self._conn.commit()

    def list_audit(self, tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit WHERE tenant_id = ? ORDER BY id DESC LIMIT ?", (tenant_id, limit)
            ).fetchall()
        return [dict(row) for row in rows]

    def task_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall()
        return {row["status"]: row["n"] for row in rows}

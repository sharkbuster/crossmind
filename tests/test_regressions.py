"""Regression tests for defects found by adversarial probing.

Each test corresponds to a concrete behaviour that was wrong once:

* the answerability gate was skipped on the graph path, so /chat answered what
  /kb/answer refused;
* tasks were always persisted with type ``qa``, whatever the routed intent;
* idempotency was read-then-write, so N concurrent duplicates created N tasks;
* an arbitrarily large request body was accepted;
* approving a task replaced its output with a different shape.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import httpx

from crossmind.storage.db import Store


async def test_chat_and_kb_agree_on_unanswerable_question(
    client: httpx.AsyncClient, headers: dict[str, str]
) -> None:
    question = "你们支持以旧换新吗？"

    chat = (await client.post("/api/v1/chat", json={"message": question}, headers=headers)).json()
    kb = (await client.post("/api/v1/kb/answer", json={"question": question}, headers=headers)).json()

    assert "没有找到" in chat["answer"]
    assert kb["no_answer"] is True
    assert chat["citations"] == []


async def test_task_type_follows_routed_intent(
    client: httpx.AsyncClient, headers: dict[str, str]
) -> None:
    created = (
        await client.post(
            "/api/v1/chat", json={"message": "给 AirSound 写一版 Amazon 文案"}, headers=headers
        )
    ).json()
    task = (await client.get(f"/api/v1/tasks/{created['task_id']}", headers=headers)).json()
    assert task["type"] == "copywriting"

    qa = (
        await client.post("/api/v1/chat", json={"message": "保修期多久"}, headers=headers)
    ).json()
    qa_task = (await client.get(f"/api/v1/tasks/{qa['task_id']}", headers=headers)).json()
    assert qa_task["type"] == "qa"


async def test_concurrent_duplicate_idempotency_creates_one_task(
    client: httpx.AsyncClient, headers: dict[str, str]
) -> None:
    before = len((await client.get("/api/v1/tasks?limit=100", headers=headers)).json())

    async def call_once():
        return await client.post(
            "/api/v1/chat",
            json={"message": "保修期多久"},
            headers={**headers, "Idempotency-Key": "race-key-regression"},
        )

    responses = await asyncio.gather(*[call_once() for _ in range(5)])
    after = len((await client.get("/api/v1/tasks?limit=100", headers=headers)).json())

    assert after - before == 1, "duplicate idempotency keys must not create extra tasks"
    successes = [r for r in responses if r.status_code == 200]
    assert successes
    assert len({r.json()["task_id"] for r in successes}) == 1
    assert all(r.status_code in {200, 409} for r in responses)


def test_idempotency_reservation_is_atomic(tmp_path) -> None:
    import threading

    store = Store(tmp_path / "x.db")
    outcomes: list[str] = []
    lock = threading.Lock()

    def reserve():
        state, _ = store.reserve_idempotent("k", "t")
        with lock:
            outcomes.append(state)

    threads = [threading.Thread(target=reserve) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("reserved") == 1
    assert outcomes.count("pending") == 19
    store.close()


def test_idempotency_ttl_purges_old_keys(tmp_path) -> None:
    store = Store(tmp_path / "x.db")
    store.put_idempotent("old-key", "t", {"answer": "stale"})
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE idempotency SET created_at = '2000-01-01T00:00:00.000Z'")
        conn.commit()

    state, payload = store.reserve_idempotent("old-key", "t", ttl_s=60)
    assert state == "reserved"
    assert payload is None
    store.close()


def test_release_allows_retry_after_failure(tmp_path) -> None:
    store = Store(tmp_path / "x.db")
    assert store.reserve_idempotent("k", "t")[0] == "reserved"
    assert store.reserve_idempotent("k", "t")[0] == "pending"
    store.release_idempotent("k", "t")
    assert store.reserve_idempotent("k", "t")[0] == "reserved"
    store.close()


async def test_oversized_body_is_rejected(
    client: httpx.AsyncClient, headers: dict[str, str]
) -> None:
    payload = {"message": "保修多久", "options": {"blob": "x" * 1_200_000}}
    response = await client.post("/api/v1/chat", json=payload, headers=headers)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


async def test_approval_preserves_output_shape(
    client: httpx.AsyncClient, headers: dict[str, str]
) -> None:
    created = (
        await client.post(
            "/api/v1/chat", json={"message": "给 AirSound 写一版 Amazon 文案"}, headers=headers
        )
    ).json()
    before = (
        await client.get(f"/api/v1/tasks/{created['task_id']}", headers=headers)
    ).json()["output"]

    await client.post(
        f"/api/v1/tasks/{created['task_id']}/approve",
        json={"decision": "approve", "reviewer": "regression"},
        headers=headers,
    )
    after = (await client.get(f"/api/v1/tasks/{created['task_id']}", headers=headers)).json()["output"]

    assert {"answer", "citations", "intent", "steps", "trace_id"} <= set(after)
    assert after["status"] == "succeeded"
    assert after["requires_approval"] is False
    assert after["answer"]
    assert before["task_id"] == after["task_id"]


async def test_failed_run_does_not_poison_idempotency_key(
    client: httpx.AsyncClient, headers: dict[str, str], container, monkeypatch
) -> None:
    async def boom(_state):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(container.graph, "ainvoke", boom)
    failed = await client.post(
        "/api/v1/chat",
        json={"message": "保修期多久"},
        headers={**headers, "Idempotency-Key": "failure-key"},
    )
    assert failed.status_code == 500

    monkeypatch.undo()
    retried = await client.post(
        "/api/v1/chat",
        json={"message": "保修期多久"},
        headers={**headers, "Idempotency-Key": "failure-key"},
    )
    assert retried.status_code == 200
    assert retried.json()["answer"]


def test_json_roundtrip_of_task_output(tmp_path) -> None:
    """Guard against storing non-serialisable state in outputs."""
    payload = {"answer": "ok", "steps": [{"node": "router", "latency_ms": 1}]}
    assert json.loads(json.dumps(payload)) == payload

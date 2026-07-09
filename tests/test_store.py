"""Tests for the SQLite response store."""

from __future__ import annotations

from fastresponses.adapter import StateUpdate, TextDelta
from fastresponses.models import (
    FunctionCallItem,
    MessageItem,
    Response,
)
from fastresponses.store import SQLiteResponseStore, StoredResponse

from conftest import make_client


def make_stored(response_id: str, state=None) -> StoredResponse:
    return StoredResponse(
        response=Response(
            id=response_id,
            model="m",
            output=[
                MessageItem(role="assistant", content="hi there"),
                FunctionCallItem(call_id="c1", name="lookup", arguments="{}"),
            ],
        ),
        input_items=[MessageItem(role="user", content="hi")],
        adapter_state=state or {"history": "[]", "n": 1},
    )


async def test_round_trip(tmp_path):
    store = SQLiteResponseStore(tmp_path / "r.db")
    await store.put(make_stored("resp_1"))
    loaded = await store.get("resp_1")
    assert loaded is not None
    assert loaded.response.id == "resp_1"
    assert loaded.input_items[0].type == "message"
    assert loaded.response.output[1].type == "function_call"
    assert loaded.adapter_state == {"history": "[]", "n": 1}
    assert await store.get("resp_missing") is None
    store.close()


async def test_survives_reopen(tmp_path):
    path = tmp_path / "r.db"
    store = SQLiteResponseStore(path)
    await store.put(make_stored("resp_1"))
    store.close()

    reopened = SQLiteResponseStore(path)
    loaded = await reopened.get("resp_1")
    assert loaded is not None
    assert loaded.response.output[0].text() == "hi there"
    reopened.close()


async def test_delete(tmp_path):
    store = SQLiteResponseStore(tmp_path / "r.db")
    await store.put(make_stored("resp_1"))
    assert await store.delete("resp_1") is True
    assert await store.delete("resp_1") is False
    assert await store.get("resp_1") is None
    store.close()


async def test_upsert_and_prune(tmp_path):
    store = SQLiteResponseStore(tmp_path / "r.db", max_responses=2)
    await store.put(make_stored("resp_1"))
    await store.put(make_stored("resp_1", state={"v": 2}))  # upsert, no dup
    await store.put(make_stored("resp_2"))
    await store.put(make_stored("resp_3"))  # prunes resp_1 (oldest)
    assert await store.get("resp_1") is None
    assert await store.get("resp_2") is not None
    assert await store.get("resp_3") is not None
    store.close()


def test_continuation_through_server_with_sqlite_store(tmp_path):
    def script(run):
        turn = sum(1 for i in run.context_items if i.type == "message")
        yield TextDelta(f"turn-{turn}")
        yield StateUpdate({"turns": turn})

    store = SQLiteResponseStore(tmp_path / "r.db")
    client, adapter = make_client(script, store=store)
    first = client.post("/v1/responses", json={"input": "one"}).json()
    second = client.post(
        "/v1/responses",
        json={"input": "two", "previous_response_id": first["id"]},
    ).json()
    assert second["output"][0]["content"][0]["text"] == "turn-3"
    assert adapter.runs[1].previous_state == {"turns": 1}
    # retrieval endpoint reads from SQLite too
    assert client.get(f"/v1/responses/{first['id']}").status_code == 200
    store.close()

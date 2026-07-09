"""Tests for the WebSocket transport of /v1/responses."""

from __future__ import annotations

from fastresponses.adapter import (
    AdapterError,
    StateUpdate,
    TextDelta,
)
from fastresponses.models import FunctionCallOutputItem

from conftest import make_client


def echo_script(run):
    last = run.new_items[-1]
    if isinstance(last, FunctionCallOutputItem):
        raise AdapterError(
            "No matching tool call for the provided output.",
            type="invalid_request",
            code="invalid_value",
            param="input",
        )
    turn = sum(1 for item in run.context_items if item.type == "message")
    yield TextDelta(f"turn-{turn}")
    yield StateUpdate({"turns": turn})


def recv_turn(ws) -> tuple[list[dict], dict | None, dict | None]:
    """Read events until a terminal or error envelope; returns (events, final, error)."""
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] in ("response.completed", "response.failed", "response.incomplete"):
            return events, event["response"], None
        if event["type"] == "error" and "status" in event:
            return events, None, event


def create(ws, **fields):
    ws.send_json({"type": "response.create", **fields})
    return recv_turn(ws)


def test_websocket_basic_turn():
    client, _ = make_client(echo_script)
    with client.websocket_connect("/v1/responses") as ws:
        events, final, error = create(ws, model="m", input="hello", store=False)
    assert error is None
    types = [e["type"] for e in events]
    assert types[0] == "response.created"
    assert types[-1] == "response.completed"
    assert "response.output_text.delta" in types
    assert final["status"] == "completed"
    assert final["output"][0]["content"][0]["text"] == "turn-1"
    # sequence numbers restart per turn and increase monotonically
    assert [e["sequence_number"] for e in events] == list(range(len(events)))


def test_websocket_sequential_turns_and_store_false_continuation():
    client, adapter = make_client(echo_script)
    with client.websocket_connect("/v1/responses") as ws:
        _, first, _ = create(ws, input="one", store=False)
        _, second, _ = create(
            ws, input="two", store=False, previous_response_id=first["id"]
        )
    assert second["status"] == "completed"
    assert second["previous_response_id"] == first["id"]
    # continuation got connection-local state despite store=false
    assert adapter.runs[1].previous_state == {"turns": 1}
    assert [i.type for i in adapter.runs[1].context_items] == [
        "message",  # input "one"
        "message",  # assistant turn-1
        "message",  # input "two"
    ]


def test_websocket_store_false_is_not_in_global_store():
    client, _ = make_client(echo_script)
    with client.websocket_connect("/v1/responses") as ws:
        _, final, _ = create(ws, input="one", store=False)
    assert client.get(f"/v1/responses/{final['id']}").status_code == 404


def test_websocket_previous_response_not_found():
    client, _ = make_client(echo_script)
    with client.websocket_connect("/v1/responses") as ws:
        _, final, error = create(
            ws, input="hi", store=False, previous_response_id="resp_missing"
        )
    assert final is None
    assert error["status"] == 400
    assert error["error"]["code"] == "previous_response_not_found"
    assert error["error"]["param"] == "previous_response_id"


def test_websocket_failed_continuation_evicts_cache():
    client, _ = make_client(echo_script)
    with client.websocket_connect("/v1/responses") as ws:
        _, first, _ = create(ws, input="one", store=False)

        # continuation that fails (unknown tool output) -> error envelope
        _, final, error = create(
            ws,
            store=False,
            previous_response_id=first["id"],
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "call_missing",
                    "output": "nope",
                }
            ],
        )
        assert final is None
        assert error["status"] == 400
        assert error["error"]["code"] == "invalid_value"

        # the referenced id must now be evicted from connection-local state
        _, final, error = create(
            ws, input="retry", store=False, previous_response_id=first["id"]
        )
        assert final is None
        assert error["error"]["code"] == "previous_response_not_found"


def test_websocket_rejects_non_response_create():
    client, _ = make_client(echo_script)
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "something.else"})
        error = ws.receive_json()
    assert error["type"] == "error"
    assert error["status"] == 400
    assert error["error"]["param"] == "type"


def test_websocket_requires_api_key():
    client, _ = make_client(echo_script, api_key="sekret")
    with client.websocket_connect("/v1/responses") as ws:
        error = ws.receive_json()
        assert error["status"] == 401
        assert error["error"]["code"] == "invalid_api_key"

    with client.websocket_connect(
        "/v1/responses", headers={"Authorization": "Bearer sekret"}
    ) as ws:
        _, final, error = create(ws, input="hi", store=False)
    assert error is None
    assert final["status"] == "completed"


def test_websocket_store_true_hydrates_from_global_store_on_new_connection():
    client, _ = make_client(echo_script)
    with client.websocket_connect("/v1/responses") as ws:
        _, first, _ = create(ws, input="one", store=True)
    # new connection: continuation works because the response was persisted
    with client.websocket_connect("/v1/responses") as ws:
        _, second, error = create(
            ws, input="two", store=True, previous_response_id=first["id"]
        )
    assert error is None
    assert second["status"] == "completed"

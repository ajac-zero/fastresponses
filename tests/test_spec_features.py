"""Tests for the long tail of spec features: obfuscation, incomplete
responses, background mode, WS connection limit, and parameter echo."""

from __future__ import annotations

import time

from open_responses_server.adapter import (
    Incomplete,
    StateUpdate,
    TextDelta,
)

from conftest import make_client, read_sse


def simple_script(run):
    yield TextDelta("Hello")
    yield TextDelta(" world")
    yield StateUpdate({"ok": True})


# ---------------------------------------------------------------------------
# stream_options.include_obfuscation
# ---------------------------------------------------------------------------


def test_obfuscation_included_by_default_and_padded():
    client, _ = make_client(simple_script)
    with client.stream(
        "POST", "/v1/responses", json={"input": "hi", "stream": True}
    ) as r:
        events = read_sse(r)
    deltas = [
        e
        for e in events
        if isinstance(e, dict) and e["type"] == "response.output_text.delta"
    ]
    assert deltas
    for e in deltas:
        assert "obfuscation" in e
        assert (len(e["delta"]) + len(e["obfuscation"])) % 16 == 0


def test_obfuscation_disabled_via_stream_options():
    client, _ = make_client(simple_script)
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "input": "hi",
            "stream": True,
            "stream_options": {"include_obfuscation": False},
        },
    ) as r:
        events = read_sse(r)
    deltas = [
        e
        for e in events
        if isinstance(e, dict) and e["type"] == "response.output_text.delta"
    ]
    assert deltas
    assert all("obfuscation" not in e for e in deltas)


# ---------------------------------------------------------------------------
# incomplete responses
# ---------------------------------------------------------------------------


def incomplete_script(run):
    yield TextDelta("Partial answ")
    yield Incomplete("max_output_tokens")
    yield StateUpdate({"cut": True})


def test_incomplete_response_non_streaming():
    client, _ = make_client(incomplete_script)
    body = client.post("/v1/responses", json={"input": "hi"}).json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    message = body["output"][0]
    assert message["status"] == "incomplete"
    assert message["content"][0]["text"] == "Partial answ"


def test_incomplete_response_streaming_terminal_event():
    client, _ = make_client(incomplete_script)
    with client.stream(
        "POST", "/v1/responses", json={"input": "hi", "stream": True}
    ) as r:
        events = read_sse(r)
    payloads = [e for e in events if isinstance(e, dict)]
    assert payloads[-1]["type"] == "response.incomplete"
    assert payloads[-1]["response"]["status"] == "incomplete"
    item_done = next(
        e for e in payloads if e["type"] == "response.output_item.done"
    )
    assert item_done["item"]["status"] == "incomplete"


def test_incomplete_response_is_stored_for_continuation():
    client, adapter = make_client(incomplete_script)
    body = client.post("/v1/responses", json={"input": "hi"}).json()
    assert client.get(f"/v1/responses/{body['id']}").status_code == 200
    client.post(
        "/v1/responses",
        json={"input": "continue", "previous_response_id": body["id"]},
    )
    assert adapter.runs[1].previous_state == {"cut": True}


# ---------------------------------------------------------------------------
# background responses
# ---------------------------------------------------------------------------


def test_background_response_returns_queued_then_completes():
    client, _ = make_client(simple_script)
    queued = client.post(
        "/v1/responses", json={"input": "hi", "background": True}
    ).json()
    assert queued["status"] == "queued"
    assert queued["background"] is True

    deadline = time.time() + 5
    body = None
    while time.time() < deadline:
        body = client.get(f"/v1/responses/{queued['id']}").json()
        if body["status"] == "completed":
            break
        time.sleep(0.05)
    assert body is not None and body["status"] == "completed"
    assert body["id"] == queued["id"]
    assert body["output"][0]["content"][0]["text"] == "Hello world"


def test_background_requires_store():
    client, _ = make_client(simple_script)
    r = client.post(
        "/v1/responses", json={"input": "hi", "background": True, "store": False}
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "store"


def test_background_rejects_streaming():
    client, _ = make_client(simple_script)
    r = client.post(
        "/v1/responses", json={"input": "hi", "background": True, "stream": True}
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "stream"


# ---------------------------------------------------------------------------
# WebSocket connection limit
# ---------------------------------------------------------------------------


def test_websocket_connection_limit_reached():
    client, _ = make_client(simple_script, ws_connection_limit_seconds=0)
    with client.websocket_connect("/v1/responses") as ws:
        envelope = ws.receive_json()
    assert envelope["type"] == "error"
    assert envelope["error"]["code"] == "websocket_connection_limit_reached"


# ---------------------------------------------------------------------------
# parameter echo in the response resource
# ---------------------------------------------------------------------------


def test_request_parameters_are_echoed():
    client, _ = make_client(simple_script)
    body = client.post(
        "/v1/responses",
        json={
            "input": "hi",
            "temperature": 0.2,
            "top_p": 0.9,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.25,
            "top_logprobs": 3,
            "max_tool_calls": 4,
            "service_tier": "priority",
            "safety_identifier": "user-42",
            "prompt_cache_key": "cache-1",
            "reasoning": {"effort": "high", "summary": "auto"},
            "text": {"format": {"type": "json_object"}},
            "metadata": {"trace": "t1"},
        },
    ).json()
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["presence_penalty"] == 0.5
    assert body["frequency_penalty"] == 0.25
    assert body["top_logprobs"] == 3
    assert body["max_tool_calls"] == 4
    assert body["service_tier"] == "priority"
    assert body["safety_identifier"] == "user-42"
    assert body["prompt_cache_key"] == "cache-1"
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["text"]["format"]["type"] == "json_object"
    assert body["metadata"] == {"trace": "t1"}
    assert body["completed_at"] >= body["created_at"]

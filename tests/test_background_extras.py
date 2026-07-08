"""Tests for background-response extras: cancellation, progressive
snapshots, streaming, and resumable event streams."""

from __future__ import annotations

import asyncio
import time

from open_responses_server.adapter import ItemDone, StateUpdate, TextDelta
from open_responses_server.models import FunctionCallItem

from conftest import make_client, read_sse


def quick_script(run):
    yield TextDelta("Hello ")
    yield TextDelta("world")
    yield StateUpdate({"ok": True})


def slow_script(run):
    yield ItemDone(
        FunctionCallItem(call_id="c1", name="step_one", arguments="{}")
    )
    for _ in range(200):
        yield asyncio.sleep(0.05)  # awaited by FakeAdapter, paces the turn
        yield TextDelta("tick ")


def wait_for(client, response_id, statuses, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/responses/{response_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"response never reached {statuses}: {body['status']}")


# ---------------------------------------------------------------------------
# background + stream
# ---------------------------------------------------------------------------


def test_background_with_stream_returns_live_sse():
    client, _ = make_client(quick_script)
    with client.stream(
        "POST",
        "/v1/responses",
        json={"input": "hi", "background": True, "stream": True},
    ) as r:
        events = read_sse(r)
    payloads = [e for e in events if isinstance(e, dict)]
    assert payloads[0]["type"] == "response.created"
    assert payloads[-1]["type"] == "response.completed"
    assert events[-1] == "[DONE]"
    deltas = [
        e["delta"] for e in payloads if e["type"] == "response.output_text.delta"
    ]
    assert deltas == ["Hello ", "world"]


# ---------------------------------------------------------------------------
# resumable event stream
# ---------------------------------------------------------------------------


def test_events_endpoint_replays_full_stream_after_completion():
    client, _ = make_client(quick_script)
    queued = client.post(
        "/v1/responses", json={"input": "hi", "background": True}
    ).json()
    wait_for(client, queued["id"], {"completed"})

    with client.stream("GET", f"/v1/responses/{queued['id']}/events") as r:
        events = read_sse(r)
    payloads = [e for e in events if isinstance(e, dict)]
    assert payloads[0]["type"] == "response.created"
    assert payloads[-1]["type"] == "response.completed"


def test_events_endpoint_resumes_from_cursor():
    client, _ = make_client(quick_script)
    queued = client.post(
        "/v1/responses", json={"input": "hi", "background": True}
    ).json()
    wait_for(client, queued["id"], {"completed"})

    with client.stream("GET", f"/v1/responses/{queued['id']}/events") as r:
        all_events = [e for e in read_sse(r) if isinstance(e, dict)]
    cursor = all_events[2]["sequence_number"]

    with client.stream(
        "GET", f"/v1/responses/{queued['id']}/events?starting_after={cursor}"
    ) as r:
        resumed = [e for e in read_sse(r) if isinstance(e, dict)]
    assert resumed == all_events[3:]


def test_events_endpoint_unknown_id_is_404():
    client, _ = make_client(quick_script)
    r = client.get("/v1/responses/resp_missing/events")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# progressive snapshots
# ---------------------------------------------------------------------------


def test_background_snapshot_shows_progress_mid_run():
    # Use the client as a context manager: it keeps one event loop alive
    # across requests, which the long-running background task needs.
    client, _ = make_client(slow_script)
    with client:
        queued = client.post(
            "/v1/responses", json={"input": "hi", "background": True}
        ).json()
        body = wait_for(client, queued["id"], {"in_progress"})
        # the completed item is already visible while the run keeps going
        deadline = time.time() + 5
        while time.time() < deadline:
            body = client.get(f"/v1/responses/{queued['id']}").json()
            if body["output"]:
                break
            time.sleep(0.02)
        assert body["status"] == "in_progress"
        assert body["output"][0]["type"] == "function_call"
        client.post(f"/v1/responses/{queued['id']}/cancel")


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------


def test_cancel_background_response():
    client, _ = make_client(slow_script)
    with client:
        queued = client.post(
            "/v1/responses", json={"input": "hi", "background": True}
        ).json()
        wait_for(client, queued["id"], {"in_progress"})

        cancelled = client.post(f"/v1/responses/{queued['id']}/cancel").json()
        assert cancelled["status"] == "cancelled"
        assert cancelled["id"] == queued["id"]
        # sticks in the store
        assert (
            client.get(f"/v1/responses/{queued['id']}").json()["status"]
            == "cancelled"
        )


def test_cancel_completed_response_returns_it_unchanged():
    client, _ = make_client(quick_script)
    queued = client.post(
        "/v1/responses", json={"input": "hi", "background": True}
    ).json()
    wait_for(client, queued["id"], {"completed"})
    body = client.post(f"/v1/responses/{queued['id']}/cancel").json()
    assert body["status"] == "completed"


def test_cancel_non_background_response_is_rejected():
    client, _ = make_client(quick_script)
    body = client.post("/v1/responses", json={"input": "hi"}).json()
    r = client.post(f"/v1/responses/{body['id']}/cancel")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_value"


def test_cancel_unknown_response_is_404():
    client, _ = make_client(quick_script)
    assert client.post("/v1/responses/resp_missing/cancel").status_code == 404

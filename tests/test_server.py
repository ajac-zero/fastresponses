from __future__ import annotations

from fastresponses.adapter import (
    AdapterError,
    ItemDone,
    ReasoningDelta,
    StateUpdate,
    TextDelta,
    UsageDelta,
)
from fastresponses.models import FunctionCallItem
from fastresponses.server import _content_disposition

from conftest import make_client, read_sse


def simple_script(run):
    yield TextDelta("Hello")
    yield TextDelta(" world")
    yield UsageDelta(input_tokens=3, output_tokens=2, total_tokens=5)
    yield StateUpdate({"turn": len(run.context_items)})


def test_non_streaming_text_response():
    client, _ = make_client(simple_script)
    r = client.post("/v1/responses", json={"model": "m1", "input": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response"
    assert body["id"].startswith("resp_")
    assert body["status"] == "completed"
    assert body["model"] == "m1"
    message = body["output"][0]
    assert message["type"] == "message"
    assert message["role"] == "assistant"
    assert message["status"] == "completed"
    assert message["content"] == [
        {"type": "output_text", "text": "Hello world", "annotations": []}
    ]
    assert body["usage"]["input_tokens"] == 3
    assert body["usage"]["output_tokens"] == 2
    assert body["usage"]["total_tokens"] == 5


def test_default_model_used_when_request_omits_model():
    client, _ = make_client(simple_script)
    r = client.post("/v1/responses", json={"input": "hi"})
    assert r.json()["model"] == "fake-model"


def test_streaming_event_sequence():
    client, _ = make_client(simple_script)
    with client.stream(
        "POST", "/v1/responses", json={"input": "hi", "stream": True}
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = read_sse(r)

    assert events[-1] == "[DONE]"
    payloads = [e for e in events if isinstance(e, dict)]
    types = [e["type"] for e in payloads]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    # sequence numbers strictly increase from 0
    assert [e["sequence_number"] for e in payloads] == list(range(len(payloads)))
    deltas = [e["delta"] for e in payloads if e["type"] == "response.output_text.delta"]
    assert deltas == ["Hello", " world"]
    done = payloads[types.index("response.output_text.done")]
    assert done["text"] == "Hello world"
    completed = payloads[-1]
    assert completed["response"]["status"] == "completed"
    assert completed["response"]["output"][0]["content"][0]["text"] == "Hello world"
    # item ids are consistent across the message lifecycle
    item_id = payloads[2]["item"]["id"]
    assert all(
        e.get("item_id", item_id) == item_id
        for e in payloads
        if e["type"].startswith(("response.output_text", "response.content_part"))
    )


def function_call_script(run):
    if run.previous_state is None:
        yield TextDelta("Let me check.")
        yield ItemDone(
            FunctionCallItem(
                id="fc_1",
                call_id="call_abc",
                name="get_weather",
                arguments='{"city": "Tokyo"}',
                status="completed",
            )
        )
        yield StateUpdate({"pending": "call_abc"})
    else:
        yield TextDelta("It is sunny.")
        yield StateUpdate({"pending": None})


def test_function_call_and_continuation():
    client, adapter = make_client(function_call_script)
    r1 = client.post(
        "/v1/responses",
        json={
            "input": "weather in tokyo?",
            "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
        },
    )
    body1 = r1.json()
    assert body1["status"] == "completed"
    fc = body1["output"][1]
    assert fc["type"] == "function_call"
    assert fc["call_id"] == "call_abc"
    assert fc["name"] == "get_weather"

    r2 = client.post(
        "/v1/responses",
        json={
            "previous_response_id": body1["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": '{"temp": 20}',
                }
            ],
        },
    )
    body2 = r2.json()
    assert body2["status"] == "completed"
    assert body2["output"][0]["content"][0]["text"] == "It is sunny."
    assert body2["previous_response_id"] == body1["id"]

    # adapter got the stored state and the full logical context
    run2 = adapter.runs[1]
    assert run2.previous_state == {"pending": "call_abc"}
    context_types = [item.type for item in run2.context_items]
    assert context_types == [
        "message",  # original user input
        "message",  # assistant "Let me check."
        "function_call",
        "function_call_output",
    ]


def test_function_call_streaming_events():
    client, _ = make_client(function_call_script)
    with client.stream(
        "POST",
        "/v1/responses",
        json={"input": "weather?", "stream": True},
    ) as r:
        events = read_sse(r)
    types = [e["type"] for e in events if isinstance(e, dict)]
    assert "response.function_call_arguments.delta" in types
    assert "response.function_call_arguments.done" in types
    # message closes before the function_call item is added
    assert types.index("response.output_item.done") < types.index(
        "response.function_call_arguments.delta"
    )


def test_previous_response_not_found():
    client, _ = make_client(simple_script)
    r = client.post(
        "/v1/responses",
        json={"input": "hi", "previous_response_id": "resp_missing"},
    )
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["code"] == "previous_response_not_found"
    assert error["param"] == "previous_response_id"


def test_get_and_delete_stored_response():
    client, _ = make_client(simple_script)
    response_id = client.post("/v1/responses", json={"input": "hi"}).json()["id"]

    got = client.get(f"/v1/responses/{response_id}")
    assert got.status_code == 200
    assert got.json()["id"] == response_id

    deleted = client.delete(f"/v1/responses/{response_id}")
    assert deleted.json() == {
        "id": response_id,
        "object": "response",
        "deleted": True,
    }
    assert client.get(f"/v1/responses/{response_id}").status_code == 404


def test_store_false_is_not_persisted():
    client, _ = make_client(simple_script)
    response_id = client.post(
        "/v1/responses", json={"input": "hi", "store": False}
    ).json()["id"]
    assert client.get(f"/v1/responses/{response_id}").status_code == 404


def test_invalid_body_returns_error_envelope():
    client, _ = make_client(simple_script)
    r = client.post("/v1/responses", json={"input": 42})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request"


def test_api_key_auth():
    client, _ = make_client(simple_script, api_key="sekret")
    assert client.post("/v1/responses", json={"input": "hi"}).status_code == 401
    assert (
        client.post(
            "/v1/responses",
            json={"input": "hi"},
            headers={"Authorization": "Bearer wrong"},
        ).status_code
        == 401
    )
    ok = client.post(
        "/v1/responses",
        json={"input": "hi"},
        headers={"Authorization": "Bearer sekret"},
    )
    assert ok.status_code == 200


def reasoning_script(run):
    yield ReasoningDelta(delta="Considering ")
    yield ReasoningDelta(delta="the question.")
    yield ReasoningDelta(encrypted_content="c2lnbmF0dXJl")
    yield TextDelta("The answer is 42.")


def test_reasoning_item_non_streaming():
    client, _ = make_client(reasoning_script)
    body = client.post("/v1/responses", json={"input": "hi"}).json()
    assert body["status"] == "completed"

    reasoning = body["output"][0]
    assert reasoning["type"] == "reasoning"
    assert reasoning["id"].startswith("rs_")
    assert reasoning["status"] == "completed"
    assert reasoning["summary"] == [
        {"type": "summary_text", "text": "Considering the question."}
    ]
    assert reasoning["encrypted_content"] == "c2lnbmF0dXJl"

    message = body["output"][1]
    assert message["type"] == "message"
    assert message["content"][0]["text"] == "The answer is 42."


def test_reasoning_streaming_events():
    client, _ = make_client(reasoning_script)
    with client.stream(
        "POST", "/v1/responses", json={"input": "hi", "stream": True}
    ) as r:
        events = read_sse(r)
    payloads = [e for e in events if isinstance(e, dict)]
    types = [e["type"] for e in payloads]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",  # reasoning
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        "response.output_item.done",  # reasoning
        "response.output_item.added",  # message
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",  # message
        "response.completed",
    ]
    assert [e["sequence_number"] for e in payloads] == list(range(len(payloads)))

    deltas = [
        e["delta"]
        for e in payloads
        if e["type"] == "response.reasoning_summary_text.delta"
    ]
    assert deltas == ["Considering ", "the question."]
    summary_done = next(
        e for e in payloads if e["type"] == "response.reasoning_summary_text.done"
    )
    assert summary_done["text"] == "Considering the question."

    reasoning_done = payloads[8]
    assert reasoning_done["item"]["type"] == "reasoning"
    assert reasoning_done["item"]["encrypted_content"] == "c2lnbmF0dXJl"
    # ids consistent across the reasoning lifecycle
    rs_id = payloads[2]["item"]["id"]
    assert all(
        e["item_id"] == rs_id
        for e in payloads
        if e["type"].startswith("response.reasoning_summary")
    )
    # final response carries both items
    final = payloads[-1]["response"]
    assert [item["type"] for item in final["output"]] == ["reasoning", "message"]


def failing_script(run):
    yield TextDelta("partial")
    raise AdapterError("model exploded", type="model_error", code="boom")


def test_adapter_error_non_streaming():
    client, _ = make_client(failing_script)
    r = client.post("/v1/responses", json={"input": "hi"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "boom"


def test_adapter_error_streaming_emits_error_then_failed():
    client, _ = make_client(failing_script)
    with client.stream(
        "POST", "/v1/responses", json={"input": "hi", "stream": True}
    ) as r:
        events = read_sse(r)
    payloads = [e for e in events if isinstance(e, dict)]
    types = [e["type"] for e in payloads]
    assert types[-2:] == ["error", "response.failed"]
    error_event = payloads[-2]
    assert error_event["error"]["code"] == "boom"
    assert error_event["error"]["type"] == "model_error"
    assert error_event["error"]["message"] == "model exploded"
    failed = payloads[-1]["response"]
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "boom"
    assert events[-1] == "[DONE]"


def test_content_disposition_ascii_filename_has_no_extended_parameter():
    assert _content_disposition("report.txt") == 'attachment; filename="report.txt"'


def test_content_disposition_unicode_filename_adds_rfc5987_parameter():
    header = _content_disposition("résumé 报告.txt")
    assert header == (
        'attachment; filename="r_sum_ __.txt"; '
        "filename*=UTF-8''r%C3%A9sum%C3%A9%20%E6%8A%A5%E5%91%8A.txt"
    )


def test_content_disposition_quotes_and_separators_cannot_split_header():
    header = _content_disposition('a"b;c.txt')
    assert header == (
        'attachment; filename="a_b_c.txt"; filename*=UTF-8\'\'a%22b%3Bc.txt'
    )
    assert '"a_b_c.txt"' in header
    # No raw quote or separator survives outside the quoted fallback.
    assert '"b' not in header.replace('filename="a_b_c.txt"', "")


def test_content_disposition_strips_control_characters():
    header = _content_disposition("evil\r\nSet-Cookie: x=1\x00\x1b.txt")
    assert "\r" not in header
    assert "\n" not in header
    assert "\x00" not in header
    assert "\x1b" not in header
    assert header.startswith('attachment; filename="evilSet-Cookie_ x_1.txt"')
    assert header.endswith("filename*=UTF-8''evilSet-Cookie%3A%20x%3D1.txt")


def test_content_disposition_neutralizes_path_separators():
    header = _content_disposition("../../etc/passwd")
    assert header == 'attachment; filename="_.._etc_passwd"'


def test_content_disposition_backslash_paths_are_neutralized():
    header = _content_disposition("..\\..\\boot.ini")
    assert header == 'attachment; filename="_.._boot.ini"'


def test_content_disposition_empty_and_dot_only_names_fall_back():
    assert _content_disposition("") == 'attachment; filename="artifact"'
    assert _content_disposition("...") == 'attachment; filename="artifact"'
    assert _content_disposition("\r\n") == 'attachment; filename="artifact"'


def test_content_disposition_path_separator_only_sanitization_omits_extended():
    # Path separators are neutralized identically in both forms, so the
    # extended parameter adds no information and must be omitted.
    assert _content_disposition("a/b.txt") == 'attachment; filename="a_b.txt"'


def test_content_disposition_percent_is_encoded_not_double_decoded():
    header = _content_disposition("100%.txt")
    assert header == (
        'attachment; filename="100_.txt"; filename*=UTF-8\'\'100%25.txt'
    )


def test_content_disposition_drops_unicode_format_and_separator_characters():
    # RTL override (spoofing), line separator, and zero-width space are all
    # non-printable and must be dropped from both filename forms.
    header = _content_disposition("exe\u202etxt.gpj\u2028\u200b.txt")
    assert "\u202e" not in header
    assert "\u2028" not in header
    assert "\u200b" not in header
    assert header == 'attachment; filename="exetxt.gpj.txt"'


def test_content_disposition_header_value_is_always_latin1_encodable():
    for name in ("报告.txt", "a\u202eb", "café/№∞.pdf", "", "\x00\x1f"):
        _content_disposition(name).encode("latin-1")

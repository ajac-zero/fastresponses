"""Tests for the OpenAI Agents SDK adapter, using a scripted model (offline)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("agents")

import agents as agents_sdk  # noqa: E402
from agents import Agent, function_tool  # noqa: E402
from agents.models.interface import Model  # noqa: E402
from openai.types.responses import (  # noqa: E402
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_usage import (  # noqa: E402
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

from fastresponses.adapters.openai_agents import (  # noqa: E402
    OpenAIAgentsAdapter,
)
from fastresponses.server import create_app  # noqa: E402

agents_sdk.set_tracing_disabled(True)


def _usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=9,
        output_tokens=4,
        total_tokens=13,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def text_response(text: str) -> Response:
    return Response(
        id="resp_fake",
        created_at=0,
        model="scripted",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                role="assistant",
                status="completed",
                type="message",
                content=[
                    ResponseOutputText(type="output_text", text=text, annotations=[])
                ],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=_usage(),
    )


def call_response(name: str, args: dict, call_id: str) -> Response:
    return Response(
        id="resp_fake",
        created_at=0,
        model="scripted",
        object="response",
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                call_id=call_id,
                name=name,
                arguments=json.dumps(args),
                status="completed",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=_usage(),
    )


class ScriptedModel(Model):
    """Replays scripted Responses; records every model request."""

    def __init__(self, script: list[Response]) -> None:
        self.script = list(script)
        self.requests: list[dict] = []

    async def get_response(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def get_retry_advice(self, *args, **kwargs):  # pragma: no cover
        return None

    async def close(self) -> None:  # pragma: no cover
        return None

    async def stream_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        **kwargs,
    ):
        self.requests.append(
            {
                "system": system_instructions,
                "input": input,
                "tools": tools,
                "settings": model_settings,
            }
        )
        if not self.script:
            raise AssertionError("ScriptedModel ran out of turns")
        response = self.script.pop(0)
        seq = 0
        for item in response.output:
            if item.type == "message":
                text = item.content[0].text
                words = text.split(" ")
                for i, word in enumerate(words):
                    delta = word if i == len(words) - 1 else word + " "
                    yield ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        content_index=0,
                        item_id="msg_1",
                        output_index=0,
                        delta=delta,
                        logprobs=[],
                        sequence_number=seq,
                    )
                    seq += 1
        yield ResponseCompletedEvent(
            type="response.completed", response=response, sequence_number=seq
        )


@function_tool
def get_weather(city: str) -> str:
    """Returns the weather for a city."""
    return f"sunny in {city}"


def make_oa_client(script: list[Response], *, tools=(), **app_kwargs):
    model = ScriptedModel(script)
    agent = Agent(name="test_agent", model=model, tools=list(tools))
    adapter = OpenAIAgentsAdapter(agent)
    return TestClient(create_app(adapter, **app_kwargs)), model


CLIENT_TOOL = {
    "type": "function",
    "name": "lookup",
    "description": "Look something up",
    "parameters": {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    },
}


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_simple_text_response():
    client, _ = make_oa_client([text_response("Hello from the SDK.")])
    body = client.post("/v1/responses", json={"input": "hi"}).json()
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "Hello from the SDK."
    assert body["usage"]["total_tokens"] == 13


def test_internal_tool_surfaces_extension_item():
    client, _ = make_oa_client(
        [
            call_response("get_weather", {"city": "Paris"}, "c1"),
            text_response("Sunny in Paris."),
        ],
        tools=[get_weather],
    )
    body = client.post("/v1/responses", json={"input": "weather?"}).json()
    receipt = body["output"][0]
    assert receipt["type"] == "openai_agents:function_call"
    assert receipt["name"] == "get_weather"
    assert receipt["output"] == "sunny in Paris"
    assert body["output"][1]["content"][0]["text"] == "Sunny in Paris."


def test_multi_turn_continuation():
    client, model = make_oa_client(
        [text_response("First."), text_response("Second.")]
    )
    first = client.post("/v1/responses", json={"input": "one"}).json()
    second = client.post(
        "/v1/responses", json={"input": "two", "previous_response_id": first["id"]}
    ).json()
    assert second["output"][0]["content"][0]["text"] == "Second."
    replayed = json.dumps(model.requests[1]["input"])
    assert "one" in replayed and "First." in replayed and "two" in replayed


def test_instructions_and_settings_mapping():
    client, model = make_oa_client([text_response("Oui.")])
    client.post(
        "/v1/responses",
        json={
            "input": "hi",
            "instructions": "Answer only in French.",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 64,
        },
    )
    request = model.requests[0]
    assert request["system"] == "Answer only in French."
    assert request["settings"].temperature == 0.2
    assert request["settings"].top_p == 0.9
    assert request["settings"].max_tokens == 64


# ---------------------------------------------------------------------------
# Client tools: interruption + resume
# ---------------------------------------------------------------------------


def test_client_tool_interrupts_and_resumes():
    client, model = make_oa_client(
        [
            call_response("lookup", {"q": "answer"}, "call_x1"),
            text_response("The answer is 42."),
        ]
    )
    first = client.post(
        "/v1/responses", json={"input": "find it", "tools": [CLIENT_TOOL]}
    ).json()
    assert first["status"] == "completed"
    call = first["output"][-1]
    assert call["type"] == "function_call"
    assert call["name"] == "lookup"
    assert call["call_id"] == "call_x1"  # native SDK call id preserved
    assert json.loads(call["arguments"]) == {"q": "answer"}

    second = client.post(
        "/v1/responses",
        json={
            "previous_response_id": first["id"],
            "tools": [CLIENT_TOOL],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_x1",
                    "output": "42",
                }
            ],
        },
    ).json()
    assert second["status"] == "completed"
    assert second["output"][0]["content"][0]["text"] == "The answer is 42."
    # resumed model call saw the client-provided output in its input
    replayed = json.dumps(model.requests[-1]["input"])
    assert '"42"' in replayed


def test_stateless_replay_with_tool_transcript():
    client, model = make_oa_client([text_response("The answer is 42.")])
    body = client.post(
        "/v1/responses",
        json={
            "tools": [CLIENT_TOOL],
            "input": [
                {"type": "message", "role": "user", "content": "find it"},
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "lookup",
                    "arguments": '{"q": "answer"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": "42",
                },
            ],
        },
    ).json()
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "The answer is 42."


def test_unknown_call_id_is_invalid_request():
    client, _ = make_oa_client([call_response("lookup", {"q": "x"}, "call_x1")])
    first = client.post(
        "/v1/responses", json={"input": "go", "tools": [CLIENT_TOOL]}
    ).json()
    r = client.post(
        "/v1/responses",
        json={
            "previous_response_id": first["id"],
            "tools": [CLIENT_TOOL],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_openresponses_missing",
                    "output": "nope",
                }
            ],
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request"


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def test_allowed_tools_suppresses_execution():
    executed = {"n": 0}

    @function_tool
    def secret_tool() -> str:
        """Secret."""
        executed["n"] += 1
        return "ok"

    client, _ = make_oa_client(
        [
            call_response("secret_tool", {}, "c1"),
            text_response("Could not."),
        ],
        tools=[get_weather, secret_tool],
    )
    r = client.post(
        "/v1/responses",
        json={
            "input": "use the secret tool",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "get_weather"}],
            },
        },
    )
    assert r.status_code == 200
    assert executed["n"] == 0
    receipt = r.json()["output"][0]
    assert "not allowed" in receipt["output"]


def test_max_tool_calls_marks_response_incomplete():
    executed = {"n": 0}

    @function_tool
    def counter() -> str:
        """Counts."""
        executed["n"] += 1
        return str(executed["n"])

    client, _ = make_oa_client(
        [
            call_response("counter", {}, "c1"),
            call_response("counter", {}, "c2"),
            text_response("done"),
        ],
        tools=[counter],
    )
    body = client.post(
        "/v1/responses", json={"input": "count", "max_tool_calls": 1}
    ).json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_tool_calls"}

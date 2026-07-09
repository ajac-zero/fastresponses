"""Tests for the Pydantic AI adapter, using scripted FunctionModels (offline)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.function import (  # noqa: E402
    AgentInfo,
    DeltaThinkingPart,
    DeltaToolCall,
    FunctionModel,
)

from fastresponses.adapters.pydantic_ai import PydanticAIAdapter  # noqa: E402
from fastresponses.server import create_app  # noqa: E402

from conftest import read_sse  # noqa: E402


def tool_returns(messages) -> int:
    """How many tool results are in the conversation so far."""
    return sum(
        1
        for m in messages
        for p in getattr(m, "parts", [])
        if type(p).__name__ in ("ToolReturnPart", "RetryPromptPart")
    )


def make_pai_client(stream_fn, *, tools=(), agent_kwargs=None, **app_kwargs):
    agent = Agent(
        FunctionModel(stream_function=stream_fn, model_name="scripted"),
        name="test_agent",
        **(agent_kwargs or {}),
    )
    for tool in tools:
        agent.tool_plain(tool)
    adapter = PydanticAIAdapter(agent)
    return TestClient(create_app(adapter, **app_kwargs)), agent


def get_weather(city: str) -> dict:
    """Returns the weather for a city."""
    return {"city": city, "forecast": "sunny"}


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_simple_text_response():
    async def stream_fn(messages, info: AgentInfo):
        yield "Hello "
        yield "there!"

    client, _ = make_pai_client(stream_fn)
    body = client.post("/v1/responses", json={"input": "hi"}).json()
    assert body["status"] == "completed"
    assert body["model"] == "pydantic-ai/test_agent/scripted"
    assert body["output"][0]["content"][0]["text"] == "Hello there!"
    assert body["usage"]["total_tokens"] > 0


def test_streaming_deltas():
    async def stream_fn(messages, info: AgentInfo):
        yield "One "
        yield "two"

    client, _ = make_pai_client(stream_fn)
    with client.stream(
        "POST", "/v1/responses", json={"input": "hi", "stream": True}
    ) as r:
        events = read_sse(r)
    deltas = [
        e["delta"]
        for e in events
        if isinstance(e, dict) and e["type"] == "response.output_text.delta"
    ]
    assert deltas == ["One ", "two"]


def test_internal_tool_call_surfaces_extension_item():
    async def stream_fn(messages, info: AgentInfo):
        if tool_returns(messages) == 0:
            yield {
                0: DeltaToolCall(
                    name="get_weather",
                    json_args='{"city": "Paris"}',
                    tool_call_id="t1",
                )
            }
        else:
            yield "Sunny in Paris."

    client, _ = make_pai_client(stream_fn, tools=[get_weather])
    body = client.post("/v1/responses", json={"input": "weather?"}).json()
    receipt = body["output"][0]
    assert receipt["type"] == "pydantic_ai:function_call"
    assert receipt["name"] == "get_weather"
    assert json.loads(receipt["output"]) == {"city": "Paris", "forecast": "sunny"}
    assert body["output"][1]["content"][0]["text"] == "Sunny in Paris."


# ---------------------------------------------------------------------------
# Client tools (deferred) and resume
# ---------------------------------------------------------------------------

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


def lookup_script():
    async def stream_fn(messages, info: AgentInfo):
        # The client tool must be visible to the model.
        assert any(t.name == "lookup" for t in info.function_tools)
        if tool_returns(messages) == 0:
            yield {
                0: DeltaToolCall(
                    name="lookup", json_args='{"q": "answer"}', tool_call_id="c1"
                )
            }
        else:
            returns = [
                p
                for m in messages
                for p in getattr(m, "parts", [])
                if type(p).__name__ == "ToolReturnPart"
            ]
            yield f"The answer is {returns[-1].content}."

    return stream_fn


def test_client_tool_yields_control_and_resumes():
    client, _ = make_pai_client(lookup_script())
    first = client.post(
        "/v1/responses", json={"input": "find the answer", "tools": [CLIENT_TOOL]}
    ).json()
    assert first["status"] == "completed"
    call = first["output"][-1]
    assert call["type"] == "function_call"
    assert call["name"] == "lookup"
    assert json.loads(call["arguments"]) == {"q": "answer"}

    second = client.post(
        "/v1/responses",
        json={
            "previous_response_id": first["id"],
            "tools": [CLIENT_TOOL],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "42",
                }
            ],
        },
    ).json()
    assert second["status"] == "completed"
    assert second["output"][0]["content"][0]["text"] == "The answer is 42."


def test_stateless_resume_from_client_supplied_history():
    """No previous_response_id: full context echoed by the client."""
    client, _ = make_pai_client(lookup_script())
    body = client.post(
        "/v1/responses",
        json={
            "tools": [CLIENT_TOOL],
            "input": [
                {"type": "message", "role": "user", "content": "find the answer"},
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
    client, _ = make_pai_client(lookup_script())
    first = client.post(
        "/v1/responses", json={"input": "find the answer", "tools": [CLIENT_TOOL]}
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


def test_multi_turn_with_previous_response_id():
    async def stream_fn(messages, info: AgentInfo):
        user_turns = sum(
            1
            for m in messages
            for p in getattr(m, "parts", [])
            if type(p).__name__ == "UserPromptPart"
        )
        yield f"turn-{user_turns}"

    client, _ = make_pai_client(stream_fn)
    first = client.post("/v1/responses", json={"input": "one"}).json()
    second = client.post(
        "/v1/responses", json={"input": "two", "previous_response_id": first["id"]}
    ).json()
    assert first["output"][0]["content"][0]["text"] == "turn-1"
    assert second["output"][0]["content"][0]["text"] == "turn-2"


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


def test_thinking_becomes_reasoning_item():
    async def stream_fn(messages, info: AgentInfo):
        yield {0: DeltaThinkingPart(content="Considering... ")}
        yield {0: DeltaThinkingPart(content="decided.")}
        yield "The answer."

    client, _ = make_pai_client(stream_fn)
    body = client.post("/v1/responses", json={"input": "think"}).json()
    reasoning = body["output"][0]
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"][0]["text"] == "Considering... decided."
    assert body["output"][1]["content"][0]["text"] == "The answer."


# ---------------------------------------------------------------------------
# Parameter mapping
# ---------------------------------------------------------------------------


def test_model_settings_mapping():
    from pydantic_ai.profiles import ModelProfile

    captured = {}

    async def stream_fn(messages, info: AgentInfo):
        captured["settings"] = info.model_settings
        captured["params"] = info.model_request_parameters
        yield "ok"

    agent = Agent(
        FunctionModel(
            stream_function=stream_fn,
            model_name="scripted",
            profile=ModelProfile(supports_thinking=True),
        ),
        name="test_agent",
    )
    client = TestClient(create_app(PydanticAIAdapter(agent)))
    client.post(
        "/v1/responses",
        json={
            "input": "hi",
            "temperature": 0.3,
            "top_p": 0.7,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "max_output_tokens": 99,
            "reasoning": {"effort": "high"},
        },
    )
    settings = captured["settings"]
    assert settings["temperature"] == 0.3
    assert settings["top_p"] == 0.7
    assert settings["presence_penalty"] == 0.1
    assert settings["frequency_penalty"] == 0.2
    assert settings["max_tokens"] == 99
    # 'thinking' is resolved into request parameters by Pydantic AI
    assert captured["params"].thinking == "high"


def test_text_format_json_schema_uses_structured_output():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    async def stream_fn(messages, info: AgentInfo):
        assert info.output_tools, "structured output should register an output tool"
        yield {
            0: DeltaToolCall(
                name=info.output_tools[0].name,
                json_args='{"answer": "42"}',
                tool_call_id="o1",
            )
        }

    client, _ = make_pai_client(stream_fn)
    body = client.post(
        "/v1/responses",
        json={
            "input": "hi",
            "text": {"format": {"type": "json_schema", "name": "reply", "schema": schema}},
        },
    ).json()
    assert body["status"] == "completed"


def test_allowed_tools_suppresses_execution():
    executed = {"secret": False}

    def secret_tool() -> str:
        """Secret."""
        executed["secret"] = True
        return "ok"

    async def stream_fn(messages, info: AgentInfo):
        if tool_returns(messages) == 0:
            yield {
                0: DeltaToolCall(name="secret_tool", json_args="{}", tool_call_id="s1")
            }
        else:
            yield "Could not."

    client, _ = make_pai_client(stream_fn, tools=[get_weather, secret_tool])
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
    assert executed["secret"] is False
    receipt = r.json()["output"][0]
    assert "not allowed" in receipt["output"]


def test_max_tool_calls_marks_response_incomplete():
    async def stream_fn(messages, info: AgentInfo):
        n = tool_returns(messages)
        if n < 2:
            yield {
                0: DeltaToolCall(
                    name="get_weather",
                    json_args='{"city": "Paris"}',
                    tool_call_id=f"w{n}",
                )
            }
        else:
            yield "done"

    client, _ = make_pai_client(stream_fn, tools=[get_weather])
    body = client.post(
        "/v1/responses", json={"input": "go", "max_tool_calls": 1}
    ).json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_tool_calls"}


# ---------------------------------------------------------------------------
# Multimodal input
# ---------------------------------------------------------------------------

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAA="


def test_input_image_data_url_becomes_binary_content():
    captured = {}

    async def stream_fn(messages, info: AgentInfo):
        captured["messages"] = list(messages)
        yield "A pixel."

    client, _ = make_pai_client(stream_fn)
    r = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is this?"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{PNG_B64}",
                        },
                    ],
                }
            ]
        },
    )
    assert r.status_code == 200
    parts = captured["messages"][-1].parts
    content = parts[-1].content
    binary = [c for c in content if type(c).__name__ == "BinaryContent"]
    assert binary and binary[0].media_type == "image/png"


def test_instructions_are_passed_through():
    captured = {}

    async def stream_fn(messages, info: AgentInfo):
        captured["messages"] = list(messages)
        yield "Oui."

    client, _ = make_pai_client(stream_fn)
    client.post(
        "/v1/responses",
        json={"input": "hi", "instructions": "Answer only in French."},
    )
    instructions = captured["messages"][0].instructions
    assert "Answer only in French." in (instructions or "")

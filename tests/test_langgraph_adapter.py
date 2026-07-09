"""Tests for the LangGraph adapter, using a scripted chat model (offline)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("langgraph")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import (  # noqa: E402
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402

from fastresponses.adapters.langgraph import LangGraphAdapter  # noqa: E402
from fastresponses.server import create_app  # noqa: E402


class ScriptedChatModel(BaseChatModel):
    """Replays scripted AIMessages; records the prompts it received."""

    script: list
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next(self, messages) -> AIMessage:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("ScriptedChatModel ran out of turns")
        return self.script.pop(0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._next(messages)
        usage = {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13}
        if msg.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="", tool_calls=msg.tool_calls, usage_metadata=usage
                )
            )
            return
        words = str(msg.content).split(" ")
        for i, word in enumerate(words):
            text = word if i == len(words) - 1 else word + " "
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", usage_metadata=usage)
        )


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def get_weather(city: str) -> str:
    """Returns the weather for a city."""
    return f"sunny in {city}"


def flatten_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def make_lg_client(script: list, *, tools=(), factory=False, **app_kwargs):
    model = ScriptedChatModel(script=list(script), calls=[])

    if factory:

        def build(client_tools):
            return create_react_agent(
                model,
                tools=[*tools, *client_tools],
                checkpointer=InMemorySaver(),
            )

        adapter = LangGraphAdapter(build)
    else:
        graph = create_react_agent(
            model, tools=list(tools), checkpointer=InMemorySaver()
        )
        adapter = LangGraphAdapter(graph)
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
    client, _ = make_lg_client([AIMessage(content="Hello from LangGraph.")])
    body = client.post("/v1/responses", json={"input": "hi"}).json()
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "Hello from LangGraph."
    assert body["usage"]["total_tokens"] == 13


def test_internal_tool_call_surfaces_extension_item():
    client, _ = make_lg_client(
        [
            tool_call("get_weather", {"city": "Paris"}, "t1"),
            AIMessage(content="Sunny in Paris."),
        ],
        tools=[get_weather],
    )
    body = client.post("/v1/responses", json={"input": "weather?"}).json()
    receipt = body["output"][0]
    assert receipt["type"] == "langgraph:function_call"
    assert receipt["name"] == "get_weather"
    assert receipt["output"] == "sunny in Paris"
    assert body["output"][1]["content"][0]["text"] == "Sunny in Paris."


def test_multi_turn_continuation_uses_thread_state():
    client, model = make_lg_client(
        [AIMessage(content="First."), AIMessage(content="Second.")]
    )
    first = client.post("/v1/responses", json={"input": "one"}).json()
    second = client.post(
        "/v1/responses", json={"input": "two", "previous_response_id": first["id"]}
    ).json()
    assert second["output"][0]["content"][0]["text"] == "Second."
    # second model call sees the full thread history from the checkpointer
    prompts = [flatten_text(m.content) for m in model.calls[1]]
    assert prompts == ["one", "First.", "two"]


def test_stateless_replay():
    client, model = make_lg_client([AIMessage(content="Sure.")])
    client.post(
        "/v1/responses",
        json={
            "input": [
                {"type": "message", "role": "user", "content": "one"},
                {"type": "message", "role": "assistant", "content": "First."},
                {"type": "message", "role": "user", "content": "two"},
            ]
        },
    )
    prompts = [m.content for m in model.calls[0]]
    assert prompts == ["one", "First.", "two"]


def test_instructions_become_system_message():
    client, model = make_lg_client([AIMessage(content="Oui.")])
    client.post(
        "/v1/responses",
        json={"input": "hi", "instructions": "Answer only in French."},
    )
    first = model.calls[0][0]
    assert type(first).__name__ == "SystemMessage"
    assert "French" in first.content


# ---------------------------------------------------------------------------
# Client tools via interrupt / resume
# ---------------------------------------------------------------------------


def test_client_tool_interrupts_and_resumes():
    client, model = make_lg_client(
        [
            tool_call("lookup", {"q": "answer"}, "t1"),
            AIMessage(content="The answer is 42."),
        ],
        factory=True,
    )
    first = client.post(
        "/v1/responses", json={"input": "find it", "tools": [CLIENT_TOOL]}
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
    # the resumed model call must see the full thread: the original user
    # message, the tool call, and the client-provided tool result
    final_prompt = model.calls[-1]
    types_seen = [type(m).__name__ for m in final_prompt]
    assert "HumanMessage" in types_seen
    assert "ToolMessage" in types_seen
    tool_msg = next(m for m in final_prompt if type(m).__name__ == "ToolMessage")
    assert tool_msg.content == "42"


def test_unknown_call_id_is_invalid_request():
    client, _ = make_lg_client(
        [tool_call("lookup", {"q": "x"}, "t1")], factory=True
    )
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


def test_fixed_graph_rejects_request_tools():
    client, _ = make_lg_client([AIMessage(content="hi")])
    r = client.post("/v1/responses", json={"input": "x", "tools": [CLIENT_TOOL]})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "tools"


# ---------------------------------------------------------------------------
# Limits and enforcement
# ---------------------------------------------------------------------------


def test_max_tool_calls_marks_response_incomplete():
    executed = {"n": 0}

    def counter() -> str:
        """Counts."""
        executed["n"] += 1
        return str(executed["n"])

    client, _ = make_lg_client(
        [
            tool_call("counter", {}, "t1"),
            tool_call("counter", {}, "t2"),
            AIMessage(content="done"),
        ],
        tools=[counter],
    )
    body = client.post(
        "/v1/responses", json={"input": "count", "max_tool_calls": 1}
    ).json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_tool_calls"}
    assert executed["n"] == 1  # the second call never executed


def test_allowed_tools_is_rejected_loudly():
    client, _ = make_lg_client([AIMessage(content="hi")])
    r = client.post(
        "/v1/responses",
        json={
            "input": "x",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "get_weather"}],
            },
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_parameter"

"""End-to-end tests for the ADK adapter using a scripted (offline) LLM."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.genai import types

from fastresponses.adapters.adk import (
    ADKAdapter,
    ADKToolResponse,
    InputFileContent,
    InputFileReferenceContext,
)
from fastresponses.artifacts import ArtifactRecord, ArtifactRegistry
from fastresponses.models import CustomItem
from fastresponses.server import create_app

from conftest import read_sse


@asynccontextmanager
async def _async_context(value):
    yield value


def usage(prompt=10, output=5):
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=output,
        total_token_count=prompt + output,
    )


def text_turn(*chunks: str) -> list[LlmResponse]:
    """Partial chunks followed by the aggregated final response."""
    responses = [
        LlmResponse(
            partial=True,
            content=types.Content(role="model", parts=[types.Part(text=chunk)]),
        )
        for chunk in chunks
    ]
    responses.append(
        LlmResponse(
            partial=False,
            content=types.Content(
                role="model", parts=[types.Part(text="".join(chunks))]
            ),
            usage_metadata=usage(),
        )
    )
    return responses


def call_turn(name: str, args: dict) -> list[LlmResponse]:
    return [
        LlmResponse(
            partial=False,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(function_call=types.FunctionCall(name=name, args=args))
                ],
            ),
            usage_metadata=usage(),
        )
    ]


class ScriptedLlm(BaseLlm):
    """Replays canned turns; records the requests it received."""

    model: str = "scripted-model"
    turns: list[list[LlmResponse]] = []
    requests: list[LlmRequest] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        if not self.turns:
            raise AssertionError("ScriptedLlm ran out of turns")
        for response in self.turns.pop(0):
            yield response


def get_weather(city: str) -> dict:
    """Returns the weather for a city."""
    return {"city": city, "forecast": "sunny"}


def make_adk_client(turns: list[list[LlmResponse]], **agent_kwargs):
    llm = ScriptedLlm(turns=list(turns), requests=[])
    agent = Agent(
        name="test_agent",
        model=llm,
        instruction="Be helpful.",
        **agent_kwargs,
    )
    adapter = ADKAdapter(agent, app_name="test-app")
    return TestClient(create_app(adapter)), llm


def test_simple_text_response():
    client, _ = make_adk_client([text_turn("Hello", " from", " ADK")])
    r = client.post("/v1/responses", json={"input": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["model"] == "adk/test_agent"
    assert body["output"][0]["content"][0]["text"] == "Hello from ADK"
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["output_tokens"] == 5
    assert body["usage"]["total_tokens"] == 15


def test_streaming_deltas_are_not_duplicated_by_final_event():
    client, _ = make_adk_client([text_turn("Hello", " world")])
    with client.stream(
        "POST", "/v1/responses", json={"input": "hi", "stream": True}
    ) as r:
        events = read_sse(r)
    payloads = [e for e in events if isinstance(e, dict)]
    deltas = [e["delta"] for e in payloads if e["type"] == "response.output_text.delta"]
    assert deltas == ["Hello", " world"]
    final = payloads[-1]["response"]
    assert final["output"][0]["content"][0]["text"] == "Hello world"


def test_internal_tool_call_surfaces_standard_pair():
    client, _ = make_adk_client(
        [
            call_turn("get_weather", {"city": "Tokyo"}),
            text_turn("It is sunny in Tokyo."),
        ],
        tools=[get_weather],
    )
    r = client.post("/v1/responses", json={"input": "weather in tokyo?"})
    body = r.json()
    assert body["status"] == "completed"

    call, output = body["output"][:2]
    assert call["type"] == "function_call"
    assert output["type"] == "function_call_output"
    assert call["status"] == output["status"] == "completed"
    assert call["call_id"] == output["call_id"]
    assert call["name"] == "get_weather"
    assert '"city": "Tokyo"' in call["arguments"]
    assert "sunny" in output["output"]
    for item in (call, output):
        assert "provider" not in item
        assert "provider_call_id" not in item
        assert "agent" not in item

    message = body["output"][2]
    assert message["type"] == "message"
    assert message["content"][0]["text"] == "It is sunny in Tokyo."
    # usage summed across both LLM calls
    assert body["usage"]["total_tokens"] == 30


def test_internal_tool_response_mapper_receives_context_and_preserves_order():
    seen: list[ADKToolResponse] = []

    def mapper(response: ADKToolResponse):
        seen.append(response)
        return [
            CustomItem.model_validate(
                {
                    "type": "test:presentation",
                    "id": "presentation_1",
                    "status": "completed",
                    "call_id": response.call.call_id,
                }
            )
        ]

    llm = ScriptedLlm(
        turns=[call_turn("get_weather", {"city": "Tokyo"}), text_turn("Done.")],
        requests=[],
    )
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm, tools=[get_weather]),
        app_name="test-app",
        internal_tool_response_mapper=mapper,
    )
    body = TestClient(create_app(adapter)).post(
        "/v1/responses", json={"input": "weather?"}
    ).json()

    assert [item["type"] for item in body["output"]] == [
        "function_call",
        "function_call_output",
        "test:presentation",
        "message",
    ]
    assert seen[0].arguments == {"city": "Tokyo"}
    assert seen[0].response == {"city": "Tokyo", "forecast": "sunny"}


def test_mapper_error_completes_canonical_pair_before_response_fails():
    def mapper(_: ADKToolResponse):
        raise RuntimeError("secret mapper configuration")

    llm = ScriptedLlm(
        turns=[call_turn("get_weather", {"city": "Tokyo"}), text_turn("Done.")],
        requests=[],
    )
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm, tools=[get_weather]),
        app_name="test-app",
        internal_tool_response_mapper=mapper,
    )
    client = TestClient(create_app(adapter))
    with client.stream(
        "POST", "/v1/responses", json={"input": "weather?", "stream": True}
    ) as response:
        payloads = [event for event in read_sse(response) if isinstance(event, dict)]

    done_types = [
        event["item"]["type"]
        for event in payloads
        if event["type"] == "response.output_item.done"
    ]
    assert done_types == ["function_call", "function_call_output"]
    assert payloads[-1]["type"] == "response.failed"
    assert payloads[-1]["response"]["error"]["code"] == "internal_tool_response_mapper_error"
    assert payloads[-1]["response"]["error"]["message"] == (
        "Internal tool response mapper failed."
    )
    assert "secret" not in str(payloads[-1])


def test_client_tool_yields_control_and_resumes():
    client, llm = make_adk_client(
        [
            call_turn("get_time", {"timezone": "UTC"}),
            text_turn("It is noon."),
        ]
    )
    tools = [
        {
            "type": "function",
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        }
    ]

    r1 = client.post(
        "/v1/responses", json={"input": "what time is it?", "tools": tools}
    )
    body1 = r1.json()
    assert body1["status"] == "completed"
    fc = body1["output"][0]
    assert fc["type"] == "function_call"
    assert fc["name"] == "get_time"
    assert fc["call_id"].startswith("call_")
    assert '"timezone": "UTC"' in fc["arguments"]
    # the client tool declaration reached the model
    declared = llm.requests[0].config.tools[0].function_declarations[0]
    assert declared.name == "get_time"

    r2 = client.post(
        "/v1/responses",
        json={
            "previous_response_id": body1["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": fc["call_id"],
                    "output": '{"time": "12:00"}',
                }
            ],
            "tools": tools,
        },
    )
    body2 = r2.json()
    assert body2["status"] == "completed"
    assert body2["output"][0]["content"][0]["text"] == "It is noon."

    # the resumed LLM call saw the function response in context
    last_request = llm.requests[-1]
    fr_parts = [
        part.function_response
        for content in last_request.contents
        for part in (content.parts or [])
        if part.function_response
    ]
    assert fr_parts and fr_parts[0].response == {"time": "12:00"}


def thought_part(text: str, signature: bytes | None = None) -> types.Part:
    part = types.Part(text=text)
    part.thought = True
    if signature is not None:
        part.thought_signature = signature
    return part


def test_streamed_thoughts_become_reasoning_item():
    turn = [
        LlmResponse(
            partial=True,
            content=types.Content(role="model", parts=[thought_part("Let me think. ")]),
        ),
        LlmResponse(
            partial=True,
            content=types.Content(
                role="model", parts=[thought_part("Tokyo is in Japan.")]
            ),
        ),
        LlmResponse(
            partial=True,
            content=types.Content(
                role="model", parts=[types.Part(text="It is sunny.")]
            ),
        ),
        LlmResponse(
            partial=False,
            content=types.Content(
                role="model",
                parts=[
                    thought_part("Let me think. Tokyo is in Japan."),
                    types.Part(text="It is sunny."),
                ],
            ),
            usage_metadata=usage(),
        ),
    ]
    client, _ = make_adk_client([turn])
    with client.stream(
        "POST", "/v1/responses", json={"input": "weather?", "stream": True}
    ) as r:
        events = read_sse(r)
    payloads = [e for e in events if isinstance(e, dict)]

    reasoning_deltas = [
        e["delta"]
        for e in payloads
        if e["type"] == "response.reasoning_summary_text.delta"
    ]
    assert reasoning_deltas == ["Let me think. ", "Tokyo is in Japan."]
    text_deltas = [
        e["delta"] for e in payloads if e["type"] == "response.output_text.delta"
    ]
    assert text_deltas == ["It is sunny."]

    final = payloads[-1]["response"]
    assert [item["type"] for item in final["output"]] == ["reasoning", "message"]
    assert final["output"][0]["summary"] == [
        {"type": "summary_text", "text": "Let me think. Tokyo is in Japan."}
    ]
    assert final["output"][1]["content"][0]["text"] == "It is sunny."


def test_unstreamed_thoughts_and_signature_in_final_event():
    turn = [
        LlmResponse(
            partial=False,
            content=types.Content(
                role="model",
                parts=[
                    thought_part("Weighing options.", signature=b"\x01\x02"),
                    types.Part(text="Go with option A."),
                ],
            ),
            usage_metadata=usage(),
        ),
    ]
    client, _ = make_adk_client([turn])
    body = client.post("/v1/responses", json={"input": "which option?"}).json()

    reasoning = body["output"][0]
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"] == [
        {"type": "summary_text", "text": "Weighing options."}
    ]
    assert reasoning["encrypted_content"] == "AQI="  # base64 of \x01\x02
    assert body["output"][1]["content"][0]["text"] == "Go with option A."


def test_previous_response_id_reuses_session_history():
    client, llm = make_adk_client(
        [text_turn("Blue."), text_turn("Because of Rayleigh scattering.")]
    )
    body1 = client.post(
        "/v1/responses", json={"input": "what color is the sky?"}
    ).json()
    body2 = client.post(
        "/v1/responses",
        json={"previous_response_id": body1["id"], "input": "why?"},
    ).json()
    assert body2["output"][0]["content"][0]["text"] == "Because of Rayleigh scattering."

    # second LLM call sees prior user + assistant turns from the ADK session
    contents = llm.requests[-1].contents
    texts = [
        part.text for content in contents for part in (content.parts or []) if part.text
    ]
    assert "what color is the sky?" in texts
    assert "Blue." in texts
    assert "why?" in texts


def test_stateless_history_replay():
    client, llm = make_adk_client([text_turn("Paris has about 2 million people.")])
    body = client.post(
        "/v1/responses",
        json={
            "input": [
                {"type": "message", "role": "user", "content": "capital of France?"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Paris."}],
                },
                {"type": "message", "role": "user", "content": "population?"},
            ]
        },
    ).json()
    assert body["status"] == "completed"
    texts = [
        part.text
        for content in llm.requests[0].contents
        for part in (content.parts or [])
        if part.text
    ]
    assert texts == ["capital of France?", "Paris.", "population?"]


def test_stateless_history_replays_completed_internal_tool_pair():
    client, llm = make_adk_client([text_turn("It was sunny.")])
    body = client.post(
        "/v1/responses",
        json={
            "input": [
                {"type": "message", "role": "user", "content": "weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_weather",
                    "name": "get_weather",
                    "arguments": '{"city":"Tokyo"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": '{"forecast":"sunny"}',
                },
                {"type": "message", "role": "assistant", "content": "Sunny."},
                {"type": "message", "role": "user", "content": "and yesterday?"},
            ]
        },
    ).json()

    assert body["status"] == "completed"
    parts = [part for content in llm.requests[0].contents for part in content.parts or []]
    call = next(part.function_call for part in parts if part.function_call)
    output = next(part.function_response for part in parts if part.function_response)
    assert call.name == output.name == "get_weather"
    assert call.args == {"city": "Tokyo"}
    assert output.response == {"forecast": "sunny"}


def test_compaction_round_trip():
    client, llm = make_adk_client([text_turn("Paris.")])
    compacted = client.post(
        "/v1/responses/compact",
        json={
            "model": "compliance-model",
            "input": [
                {"type": "message", "role": "user", "content": "capital of France?"},
                {"type": "message", "role": "assistant", "content": "It is Paris."},
            ],
        },
    ).json()
    assert compacted["object"] == "response.compaction"
    item = compacted["output"][0]
    assert item["type"] == "compaction"
    assert item["id"].startswith("cmp_")
    assert item["encrypted_content"]

    # Use the compacted window as the base input of a new chain.
    body = client.post(
        "/v1/responses",
        json={
            "input": [
                item,
                {"type": "message", "role": "user", "content": "Repeat the capital."},
            ]
        },
    ).json()
    assert body["status"] == "completed"
    # The compacted history was expanded into the model's context.
    texts = [
        part.text
        for content in llm.requests[0].contents
        for part in (content.parts or [])
        if part.text
    ]
    assert texts == ["capital of France?", "It is Paris.", "Repeat the capital."]


def test_compact_requires_model():
    client, _ = make_adk_client([])
    r = client.post("/v1/responses/compact", json={"input": "hello"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_required_parameter"


def test_input_must_end_with_user_message_or_tool_output():
    client, _ = make_adk_client([])
    r = client.post(
        "/v1/responses",
        json={
            "input": [
                {"type": "message", "role": "assistant", "content": "hello there"}
            ]
        },
    )
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["type"] == "invalid_request"
    assert "must end with" in error["message"]


def test_request_instructions_override_agent_instruction():
    client, llm = make_adk_client([text_turn("OK")])
    client.post(
        "/v1/responses",
        json={"input": "hi", "instructions": "Answer only in French."},
    )
    system = llm.requests[0].config.system_instruction
    assert "Answer only in French." in str(system)


# ---------------------------------------------------------------------------
# Sampling / generation parameter mapping
# ---------------------------------------------------------------------------


def test_sampling_parameters_map_to_generate_content_config():
    client, llm = make_adk_client([text_turn("OK")])
    client.post(
        "/v1/responses",
        json={
            "input": "hi",
            "temperature": 0.1,
            "top_p": 0.8,
            "presence_penalty": 0.5,
            "frequency_penalty": -0.5,
            "top_logprobs": 5,
            "max_output_tokens": 128,
        },
    )
    config = llm.requests[0].config
    assert config.temperature == 0.1
    assert config.top_p == 0.8
    assert config.presence_penalty == 0.5
    assert config.frequency_penalty == -0.5
    assert config.response_logprobs is True
    assert config.logprobs == 5
    assert config.max_output_tokens == 128


def test_text_format_json_schema_maps_to_structured_output():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    client, llm = make_adk_client([text_turn('{"answer": "42"}')])
    r = client.post(
        "/v1/responses",
        json={
            "input": "hi",
            "text": {
                "format": {"type": "json_schema", "name": "reply", "schema": schema}
            },
        },
    )
    config = llm.requests[0].config
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema
    assert r.json()["text"]["format"]["type"] == "json_schema"


def test_text_format_json_object_sets_mime_type():
    client, llm = make_adk_client([text_turn("{}")])
    client.post(
        "/v1/responses",
        json={"input": "hi", "text": {"format": {"type": "json_object"}}},
    )
    config = llm.requests[0].config
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema is None


def test_reasoning_effort_maps_to_thinking_config():
    client, llm = make_adk_client([text_turn("OK")])
    client.post(
        "/v1/responses",
        json={"input": "hi", "reasoning": {"effort": "low", "summary": "auto"}},
    )
    thinking = llm.requests[0].config.thinking_config
    assert thinking is not None
    assert thinking.include_thoughts is True
    assert thinking.thinking_budget == 1024


def test_reasoning_effort_none_disables_thinking():
    client, llm = make_adk_client([text_turn("OK")])
    client.post("/v1/responses", json={"input": "hi", "reasoning": {"effort": "none"}})
    thinking = llm.requests[0].config.thinking_config
    assert thinking.include_thoughts is False
    assert thinking.thinking_budget == 0


# ---------------------------------------------------------------------------
# Multimodal input
# ---------------------------------------------------------------------------

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAA="


def test_input_image_data_url_becomes_inline_data():
    client, llm = make_adk_client([text_turn("A pixel.")])
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
    parts = llm.requests[0].contents[-1].parts
    assert parts[0].text == "What is this?"
    blob = parts[1].inline_data
    assert blob is not None
    assert blob.mime_type == "image/png"
    assert len(blob.data) > 0


def test_input_image_http_url_becomes_file_data():
    client, llm = make_adk_client([text_turn("A cat.")])
    client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/cat.jpg",
                        },
                    ],
                }
            ]
        },
    )
    file_data = llm.requests[0].contents[-1].parts[1].file_data
    assert file_data is not None
    assert file_data.file_uri == "https://example.com/cat.jpg"
    assert file_data.mime_type == "image/jpeg"


def test_input_file_base64_becomes_inline_data():
    llm = ScriptedLlm(turns=[text_turn("A doc.")], requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        input_file_routes={"inline": ".pdf"},
    )
    client = TestClient(create_app(adapter))
    client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Summarize"},
                        {
                            "type": "input_file",
                            "filename": "notes.pdf",
                            "file_data": PNG_B64,
                        },
                    ],
                }
            ]
        },
    )
    blob = llm.requests[0].contents[-1].parts[1].inline_data
    assert blob is not None
    assert blob.display_name == "notes.pdf"
    assert blob.mime_type == "application/pdf"


def test_input_file_url_is_fetched_and_hidden_from_model():
    llm = ScriptedLlm(turns=[text_turn("Read.")], requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        app_name="test-app",
        input_file_url_origins=["https://parley.example"],
        input_file_routes={"inline": ".pdf"},
    )
    client = TestClient(create_app(adapter))
    response = httpx.Response(
        200,
        headers={"content-type": "application/pdf", "content-length": "4"},
        content=b"data",
    )
    with patch("httpx.AsyncClient.stream", return_value=_async_context(response)):
        body = client.post(
            "/v1/responses",
            json={
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": "notes.pdf",
                                "file_url": "https://parley.example/capability",
                            }
                        ],
                    }
                ]
            },
        ).json()

    assert body["status"] == "completed"
    blob = llm.requests[0].contents[-1].parts[0].inline_data
    assert blob is not None
    assert blob.data == b"data"
    assert blob.display_name == "notes.pdf"
    assert blob.mime_type == "application/pdf"
    assert all(
        not part.file_data
        for content in llm.requests[0].contents
        for part in content.parts or []
    )


def _url_file_client(
    *turns,
    origins=("https://files.example",),
    routes={"inline": [".pdf", ".txt"]},
    default_action="inline",
):
    llm = ScriptedLlm(turns=list(turns) or [text_turn("Read.")], requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        app_name="test-app",
        input_file_url_origins=origins,
        input_file_routes=routes,
        default_input_file_action=default_action,
    )
    return TestClient(create_app(adapter)), llm


def _url_file_payload(*, filename: str | None = None):
    part = {
        "type": "input_file",
        "file_url": "https://files.example/download",
    }
    if filename is not None:
        part["filename"] = filename
    return {
        "input": [{"type": "message", "role": "user", "content": [part]}]
    }


def test_input_file_url_uses_response_mime_and_allows_missing_filename():
    client, llm = _url_file_client()
    response = httpx.Response(
        200,
        headers={"content-type": "application/pdf; charset=binary"},
        content=b"pdf",
    )
    with patch("httpx.AsyncClient.stream", return_value=_async_context(response)):
        body = client.post("/v1/responses", json=_url_file_payload()).json()

    assert body["status"] == "completed"
    blob = llm.requests[0].contents[-1].parts[0].inline_data
    assert blob.display_name == "attachment.pdf"
    assert blob.mime_type == "application/pdf"


def test_input_file_url_response_mime_overrides_filename_guess():
    client, llm = _url_file_client()
    response = httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=b"pdf"
    )
    with patch("httpx.AsyncClient.stream", return_value=_async_context(response)):
        body = client.post(
            "/v1/responses", json=_url_file_payload(filename="misleading.txt")
        ).json()

    assert body["status"] == "completed"
    blob = llm.requests[0].contents[-1].parts[0].inline_data
    assert blob.display_name == "misleading.txt"
    assert blob.mime_type == "application/pdf"


def test_input_file_url_follows_allowlisted_relative_redirect():
    client, llm = _url_file_client()
    redirect = httpx.Response(302, headers={"location": "/object"})
    final = httpx.Response(
        200, headers={"content-type": "text/plain"}, content=b"data"
    )
    with patch(
        "httpx.AsyncClient.stream",
        side_effect=[_async_context(redirect), _async_context(final)],
    ) as stream:
        body = client.post("/v1/responses", json=_url_file_payload()).json()

    assert body["status"] == "completed"
    assert str(stream.call_args_list[1].args[1]) == "https://files.example/object"
    assert llm.requests[0].contents[-1].parts[0].inline_data.mime_type == "text/plain"


def test_input_file_url_rejects_redirect_to_unlisted_origin():
    client, llm = _url_file_client()
    redirect = httpx.Response(
        302, headers={"location": "https://storage.example/object"}
    )
    with patch("httpx.AsyncClient.stream", return_value=_async_context(redirect)):
        response = client.post("/v1/responses", json=_url_file_payload())

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Input file URL origin is not allowed."
    assert llm.requests == []


def test_input_file_url_rejects_stored_url_when_session_is_lost():
    client, llm = _url_file_client(text_turn("Read."), text_turn("Again."))
    downloaded = httpx.Response(
        200, headers={"content-type": "text/plain"}, content=b"data"
    )
    with patch("httpx.AsyncClient.stream", return_value=_async_context(downloaded)):
        first = client.post("/v1/responses", json=_url_file_payload()).json()

    adapter = client.app.state.adapter
    adapter.session_service.sessions["test-app"]["default"].clear()
    adapter.input_file_url_origins = frozenset()
    response = client.post(
        "/v1/responses",
        json={"previous_response_id": first["id"], "input": "read it again"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Input file URL origin is not allowed."
    assert len(llm.requests) == 1


def test_input_file_url_accepts_allowlisted_ipv6_loopback():
    client, llm = _url_file_client(origins=("http://[::1]",))
    response = httpx.Response(
        200, headers={"content-type": "text/plain"}, content=b"data"
    )
    payload = _url_file_payload(filename="notes.txt")
    payload["input"][0]["content"][0]["file_url"] = "http://[::1]/download"
    with patch("httpx.AsyncClient.stream", return_value=_async_context(response)):
        body = client.post("/v1/responses", json=payload).json()

    assert body["status"] == "completed"
    blob = llm.requests[0].contents[-1].parts[0].inline_data
    assert blob.data == b"data"
    assert blob.display_name == "notes.txt"


async def save_report(tool_context: ToolContext) -> dict:
    """Save a generated report."""
    version = await tool_context.save_artifact(
        "report.txt",
        types.Part.from_bytes(data=b"generated report", mime_type="text/plain"),
    )
    return {"version": version}


def test_generated_artifact_non_streaming_and_download():
    client, _ = make_adk_client(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = next(
        item for item in body["output"] if item["type"] == "ajac-zero:artifact"
    )
    assert artifact["filename"] == "report.txt"
    assert artifact["mime_type"] == "text/plain"
    assert artifact["size"] == len(b"generated report")
    download = client.get(artifact["content_url"])
    assert download.content == b"generated report"
    assert download.headers["x-content-type-options"] == "nosniff"


async def save_unicode_report(tool_context: ToolContext) -> dict:
    """Save a generated report with an international filename."""
    version = await tool_context.save_artifact(
        "compte-rendu — 报告 café.txt",
        types.Part.from_bytes(data=b"generated report", mime_type="text/plain"),
    )
    return {"version": version}


def test_generated_artifact_download_preserves_unicode_filename():
    client, _ = make_adk_client(
        [call_turn("save_unicode_report", {}), text_turn("Report ready.")],
        tools=[save_unicode_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = next(
        item for item in body["output"] if item["type"] == "ajac-zero:artifact"
    )
    assert artifact["filename"] == "compte-rendu — 报告 café.txt"
    download = client.get(artifact["content_url"])
    assert download.content == b"generated report"
    assert download.headers["x-content-type-options"] == "nosniff"
    disposition = download.headers["content-disposition"]
    assert disposition.startswith('attachment; filename="compte-rendu _ __ caf_.txt"')
    assert disposition.endswith(
        "filename*=UTF-8''compte-rendu%20%E2%80%94%20%E6%8A%A5%E5%91%8A%20caf%C3%A9.txt"
    )


def test_inline_input_file_is_not_emitted_as_output_artifact():
    llm = ScriptedLlm(turns=[text_turn("Read it.")], requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        input_file_routes={"inline": ".txt"},
    )
    body = TestClient(create_app(adapter)).post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "input.txt",
                            "file_data": "aGVsbG8=",
                        }
                    ],
                }
            ]
        },
    ).json()
    assert all(item["type"] != "ajac-zero:artifact" for item in body["output"])


def test_generated_artifact_cannot_overwrite_referenced_input_or_break_replay():
    async def try_overwrite(tool_context: ToolContext) -> dict:
        """Try to overwrite the referenced input."""
        try:
            await tool_context.save_artifact(
                "attachment_1",
                types.Part.from_bytes(data=b"generated", mime_type="text/plain"),
            )
        except ValueError as exc:
            blocked = "reserved input namespace" in str(exc)
        else:
            blocked = False
        artifact = await tool_context.load_artifact("attachment_1")
        return {"blocked": blocked, "data": artifact.inline_data.data.decode()}

    llm = ScriptedLlm(
        turns=[
            call_turn("try_overwrite", {}),
            text_turn("Protected."),
            text_turn("Replayed."),
        ],
        requests=[],
    )
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm, tools=[try_overwrite]),
        app_name="test-app",
        input_file_routes={"reference": ".txt"},
    )
    client = TestClient(create_app(adapter))
    first = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "notes.txt",
                            "file_data": "b3JpZ2luYWw=",
                        }
                    ],
                }
            ]
        },
    ).json()

    assert first["status"] == "completed"
    assert '"blocked": true' in first["output"][1]["output"]
    assert '"data": "original"' in first["output"][1]["output"]
    assert all(item["type"] != "ajac-zero:artifact" for item in first["output"])

    adapter.session_service.sessions["test-app"]["default"].clear()
    replay = client.post(
        "/v1/responses",
        json={"previous_response_id": first["id"], "input": "read it again"},
    ).json()
    assert replay["status"] == "completed"


def test_input_file_routes_support_mixed_actions_and_longest_suffix():
    llm = ScriptedLlm(turns=[text_turn("Handled.")], requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        app_name="test-app",
        input_file_url_origins=["https://files.example"],
        input_file_routes={
            "inline": ["PDF", ".tar.gz"],
            "url": "mp4",
            "reference": [".docx", ".gz"],
        },
        default_input_file_action="reject",
    )
    client = TestClient(create_app(adapter))
    response = httpx.Response(
        200, headers={"content-type": "application/gzip"}, content=b"archive"
    )
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "bundle.TAR.GZ",
                        "file_url": "https://files.example/archive",
                    },
                    {
                        "type": "input_file",
                        "filename": "clip.mp4",
                        "file_url": "https://files.example/clip",
                    },
                    {
                        "type": "input_file",
                        "filename": "report.docx",
                        "file_data": "ZGF0YQ==",
                    },
                ],
            }
        ]
    }
    with patch("httpx.AsyncClient.stream", return_value=_async_context(response)) as stream:
        body = client.post("/v1/responses", json=payload).json()

    assert body["status"] == "completed"
    assert stream.call_count == 1
    parts = llm.requests[0].contents[-1].parts
    assert parts[0].inline_data.data == b"archive"
    assert parts[1].file_data.file_uri == "https://files.example/clip"
    assert parts[2].text == (
        '[Uploaded Artifact: {"artifact_id":"attachment_1",'
        '"filename":"report.docx","mime_type":'
        '"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}]'
    )
    stored = client.app.state.adapter.input_file_reference_store.artifact_service
    assert "attachment_1" in str(stored.artifacts)
    assert "report.docx" in str(stored.artifacts)


def test_input_file_reference_store_controls_model_text():
    class Store:
        calls: list[tuple[InputFileContent, InputFileReferenceContext]] = []

        async def create_reference(self, file, *, context):
            self.calls.append((file, context))
            return f"custom://{context.reference_id}/{file.filename}"

    store = Store()
    llm = ScriptedLlm(turns=[text_turn("Handled.")], requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        input_file_routes={"reference": ".zip"},
        input_file_reference_store=store,
    )
    client = TestClient(create_app(adapter))
    body = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "source.zip",
                            "file_data": "ZGF0YQ==",
                        }
                    ],
                }
            ]
        },
    ).json()

    assert body["status"] == "completed"
    assert llm.requests[0].contents[-1].parts[0].text == "custom://attachment_1/source.zip"
    assert store.calls[0][0].data == b"data"
    assert store.calls[0][1].app_name == "fastresponses"


def test_input_file_reference_ids_continue_across_turns():
    store_calls = []

    class Store:
        async def create_reference(self, file, *, context):
            store_calls.append(context.reference_id)
            return context.reference_id

    llm = ScriptedLlm(
        turns=[text_turn("First."), text_turn("Second.")], requests=[]
    )
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        input_file_routes={"reference": ".zip"},
        input_file_reference_store=Store(),
    )
    client = TestClient(create_app(adapter))

    first = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "first.zip",
                            "file_data": "MQ==",
                        }
                    ],
                }
            ]
        },
    ).json()
    second = client.post(
        "/v1/responses",
        json={
            "previous_response_id": first["id"],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "second.zip",
                            "file_data": "Mg==",
                        }
                    ],
                }
            ],
        },
    ).json()

    assert second["status"] == "completed"
    assert store_calls == ["attachment_1", "attachment_2"]
    assert llm.requests[1].contents[-1].parts[0].text == "attachment_2"


def test_default_reference_store_is_available_to_agent_tools():
    async def inspect_attachment(tool_context: ToolContext) -> dict:
        """Inspect the uploaded attachment."""
        artifact = await tool_context.load_artifact("attachment_1")
        return {"data": artifact.inline_data.data.decode()}

    llm = ScriptedLlm(
        turns=[
            call_turn("inspect_attachment", {}),
            text_turn("Loaded."),
        ],
        requests=[],
    )
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm, tools=[inspect_attachment]),
        input_file_routes={"reference": ".txt"},
    )
    client = TestClient(create_app(adapter))
    body = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "notes.txt",
                            "file_data": "aGVsbG8=",
                        }
                    ],
                }
            ]
        },
    ).json()
    assert body["status"] == "completed"
    assert '"data": "hello"' in body["output"][1]["output"]


def test_input_file_route_configuration_is_strict():
    agent = Agent(name="test_agent", model=ScriptedLlm(turns=[], requests=[]))
    with pytest.raises(ValueError, match="routed to both"):
        ADKAdapter(
            agent,
            input_file_routes={"inline": ".pdf", "reference": "PDF"},
        )
    with pytest.raises(ValueError, match="not both"):
        ADKAdapter(
            agent,
            input_file_routes={"inline": ".pdf"},
            input_file_router=lambda _: "inline",
        )
    with pytest.raises(ValueError, match="Unknown input file action"):
        ADKAdapter(agent, default_input_file_action="guess")
    with pytest.raises(ValueError, match="must not be empty"):
        ADKAdapter(agent, input_file_routes={"inline": ""})


def test_input_file_default_action_rejects_unknown_extension():
    client, llm = _url_file_client(
        routes={"inline": ".pdf"}, default_action="reject"
    )
    response = client.post(
        "/v1/responses", json=_url_file_payload(filename="payload.exe")
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_file_type"
    assert llm.requests == []
    sessions = client.app.state.adapter.session_service.sessions
    assert not sessions.get("test-app", {}).get("default", {})


def test_input_file_explicit_reject_route_and_url_source_validation():
    client, llm = _url_file_client(
        routes={"url": ".mp4", "reject": ".exe"}, default_action="inline"
    )
    rejected = client.post(
        "/v1/responses", json=_url_file_payload(filename="payload.exe")
    )
    missing_url = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "clip.mp4",
                            "file_data": "ZGF0YQ==",
                        }
                    ],
                }
            ]
        },
    )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "unsupported_file_type"
    assert missing_url.status_code == 400
    assert "must provide file_url" in missing_url.json()["error"]["message"]
    assert llm.requests == []


def test_input_file_rejects_dual_sources():
    client, llm = _url_file_client(routes={"url": ".mp4"})
    payload = _url_file_payload(filename="clip.mp4")
    payload["input"][0]["content"][0]["file_data"] = "ZGF0YQ=="
    response = client.post("/v1/responses", json=payload)

    assert response.status_code == 400
    assert "both file_data and file_url" in response.json()["error"]["message"]
    assert llm.requests == []


def test_input_file_rejects_malformed_url_as_invalid_request():
    client, llm = _url_file_client(routes={"url": ".mp4"})
    payload = _url_file_payload(filename="clip.mp4")
    payload["input"][0]["content"][0]["file_url"] = (
        "https://files.example:bad/clip"
    )
    response = client.post("/v1/responses", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Input file URL is invalid."
    assert llm.requests == []


def test_input_file_accepts_empty_base64_inline():
    llm = ScriptedLlm(turns=[text_turn("Empty.")], requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        input_file_routes={"inline": ".txt"},
    )
    client = TestClient(create_app(adapter))
    body = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "empty.txt",
                            "file_data": "",
                        }
                    ],
                }
            ]
        },
    ).json()

    assert body["status"] == "completed"
    assert llm.requests[0].contents[-1].parts[0].inline_data.data == b""


def test_input_file_rejects_empty_reference():
    llm = ScriptedLlm(turns=[text_turn("No.")], requests=[])
    client = TestClient(
        create_app(
            ADKAdapter(
                Agent(name="test_agent", model=llm),
                input_file_routes={"reference": ".txt"},
            )
        )
    )
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "empty.txt",
                            "file_data": "",
                        }
                    ],
                }
            ]
        },
    )

    assert response.status_code == 400
    assert "must not be empty" in response.json()["error"]["message"]
    assert llm.requests == []


def test_input_file_router_supports_async_and_rejects_invalid_action():
    async def inline(_):
        return "inline"

    llm = ScriptedLlm(turns=[text_turn("Read.")], requests=[])
    client = TestClient(
        create_app(
            ADKAdapter(
                Agent(name="test_agent", model=llm), input_file_router=inline
            )
        )
    )
    body = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "notes.txt",
                            "file_data": "ZGF0YQ==",
                        }
                    ],
                }
            ]
        },
    ).json()
    assert body["status"] == "completed"

    bad_client = TestClient(
        create_app(
            ADKAdapter(
                Agent(
                    name="bad_agent",
                    model=ScriptedLlm(turns=[text_turn("No.")], requests=[]),
                ),
                input_file_router=lambda _: "guess",
            )
        )
    )
    response = bad_client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "notes.txt",
                            "file_data": "ZGF0YQ==",
                        }
                    ],
                }
            ]
        },
    )
    assert response.status_code == 400
    assert "Unknown input file action" in response.json()["error"]["message"]


def test_input_file_rejects_file_id_explicitly():
    client, llm = _url_file_client()
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "notes.txt",
                            "file_id": "file_123",
                        }
                    ],
                }
            ]
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert llm.requests == []


def test_input_file_callback_failures_are_sanitized():
    def fail_router(_):
        raise RuntimeError("secret router detail")

    client = TestClient(
        create_app(
            ADKAdapter(
                Agent(
                    name="test_agent",
                    model=ScriptedLlm(turns=[text_turn("No.")], requests=[]),
                ),
                input_file_router=fail_router,
            )
        )
    )
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "notes.txt",
                            "file_data": "ZGF0YQ==",
                        }
                    ],
                }
            ]
        },
    )

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Input file router failed."
    assert "secret" not in response.text


def test_failed_fresh_run_cleans_up_default_input_artifacts():
    llm = ScriptedLlm(turns=[], requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        app_name="test-app",
        input_file_routes={"reference": ".txt"},
    )
    client = TestClient(create_app(adapter))
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "notes.txt",
                            "file_data": "ZGF0YQ==",
                        }
                    ],
                }
            ]
        },
    )

    assert response.status_code == 500
    assert not adapter.session_service.sessions.get("test-app", {}).get("default", {})
    assert "attachment_1" not in str(adapter.artifact_service.artifacts)


def test_artifact_registry_is_bounded_and_expires(monkeypatch):
    now = {"value": 0.0}
    monkeypatch.setattr("fastresponses.artifacts.time.monotonic", lambda: now["value"])
    registry = ArtifactRegistry(max_records=1, ttl_seconds=10)
    record = ArtifactRecord(object(), "app", "user", "session", "a.txt", 0, "text/plain")
    first = registry.register(record)
    second = registry.register(record)
    assert registry.get(first) is None
    assert registry.get(second) is record
    now["value"] = 11
    assert registry.get(second) is None


def test_multimodal_history_replay_preserves_images():
    client, llm = make_adk_client([text_turn("Still a pixel.")])
    client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "look"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{PNG_B64}",
                        },
                    ],
                },
                {"type": "message", "role": "assistant", "content": "A pixel."},
                {"type": "message", "role": "user", "content": "sure?"},
            ]
        },
    )
    contents = llm.requests[0].contents
    image_parts = [
        p for c in contents for p in (c.parts or []) if p.inline_data is not None
    ]
    assert len(image_parts) == 1


# ---------------------------------------------------------------------------
# max_tool_calls / allowed_tools enforcement / finish reasons
# ---------------------------------------------------------------------------


def test_max_tool_calls_marks_response_incomplete():
    calls = {"n": 0}

    def counter() -> dict:
        """Counts."""
        calls["n"] += 1
        return {"n": calls["n"]}

    client, _ = make_adk_client(
        [
            call_turn("counter", {}),
            call_turn("counter", {}),
            text_turn("done"),
        ],
        tools=[counter],
    )
    body = client.post(
        "/v1/responses", json={"input": "count twice", "max_tool_calls": 1}
    ).json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_tool_calls"}
    # first call went through, second was cut off
    assert calls["n"] >= 1
    tool_calls = [i for i in body["output"] if i["type"] == "function_call"]
    outputs = [i for i in body["output"] if i["type"] == "function_call_output"]
    assert len(tool_calls) == len(outputs) == 1


def test_allowed_tools_blocks_disallowed_internal_tool():
    executed = {"secret": False}

    def secret_tool() -> dict:
        """Does something secret."""
        executed["secret"] = True
        return {"ok": True}

    client, _ = make_adk_client(
        [call_turn("secret_tool", {}), text_turn("I could not use that tool.")],
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
    assert executed["secret"] is False  # execution was suppressed


def test_max_tokens_finish_reason_marks_incomplete():
    truncated = [
        LlmResponse(
            partial=False,
            content=types.Content(role="model", parts=[types.Part(text="Once upon a")]),
            finish_reason=types.FinishReason.MAX_TOKENS,
            usage_metadata=usage(),
        )
    ]
    client, _ = make_adk_client([truncated])
    body = client.post(
        "/v1/responses", json={"input": "tell a story", "max_output_tokens": 5}
    ).json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    assert body["output"][0]["content"][0]["text"] == "Once upon a"

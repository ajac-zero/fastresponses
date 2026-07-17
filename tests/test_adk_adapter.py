"""End-to-end tests for the ADK adapter using a scripted (offline) LLM."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from google.adk.agents import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import ValidationError

from fastresponses.adapter import AdapterError
from fastresponses.adapters.adk import (
    ADKAdapter,
    ADKToolResponse,
    InputFileContent,
    InputFileReferenceContext,
    _EventTranslator,
    _GeneratedArtifactPolicy,
)
from fastresponses.artifacts import (
    ArtifactItem,
    ArtifactRecord,
    ArtifactRegistry,
    parse_artifact_item,
)
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
    assert artifact["available"] is True
    assert isinstance(artifact["expires_at"], int)
    assert artifact["expires_at"] > time.time()
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


def _make_artifact_adapter(turns, *, adapter_kwargs=None, **agent_kwargs):
    llm = ScriptedLlm(turns=list(turns), requests=[])
    agent = Agent(name="test_agent", model=llm, instruction="Be helpful.", **agent_kwargs)
    adapter = ADKAdapter(agent, app_name="test-app", **(adapter_kwargs or {}))
    return TestClient(create_app(adapter)), adapter


def _generated_artifacts(body: dict) -> list[dict]:
    return [item for item in body["output"] if item["type"] == "ajac-zero:artifact"]


def test_revoked_artifact_immediately_returns_not_found():
    client, _ = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]

    revoke = client.delete(f"/v1/artifacts/{artifact['id']}")
    assert revoke.status_code == 200
    assert revoke.json() == {
        "id": artifact["id"],
        "object": "artifact",
        "deleted": True,
    }

    download = client.get(artifact["content_url"])
    assert download.status_code == 404
    assert download.json()["error"]["code"] == "artifact_not_found"

    second = client.delete(f"/v1/artifacts/{artifact['id']}")
    assert second.status_code == 404
    assert second.json()["error"]["code"] == "artifact_not_found"


def test_revoking_unknown_artifact_returns_not_found():
    client, _ = _make_artifact_adapter([text_turn("Hi.")])
    response = client.delete("/v1/artifacts/artifact_unknown")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "artifact_not_found"


def test_revocation_keeps_provider_content_by_default():
    client, adapter = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]

    assert client.delete(f"/v1/artifacts/{artifact['id']}").status_code == 200

    # The link is dead, but the artifact stays in the ADK session context so
    # later agent turns can still load it.
    assert client.get(artifact["content_url"]).status_code == 404
    assert "report.txt" in str(adapter.artifact_service.artifacts)


def test_revoking_last_reference_with_delete_content_deletes_provider_content():
    client, adapter = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]
    assert "report.txt" in str(adapter.artifact_service.artifacts)

    revoke = client.delete(f"/v1/artifacts/{artifact['id']}?delete_content=true")
    assert revoke.status_code == 200
    assert "report.txt" not in str(adapter.artifact_service.artifacts)


def test_revoking_one_version_preserves_other_versions():
    client, adapter = _make_artifact_adapter(
        [
            call_turn("save_report", {}),
            call_turn("save_report", {}),
            text_turn("Two reports saved."),
        ],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make two reports"}).json()
    first, second = _generated_artifacts(body)
    assert first["id"] != second["id"]

    revoke = client.delete(f"/v1/artifacts/{second['id']}?delete_content=true")
    assert revoke.status_code == 200

    # The other version keeps its public record and its provider content:
    # provider deletion is filename-wide, so it is skipped while a live
    # sibling record remains.
    assert client.get(second["content_url"]).status_code == 404
    download = client.get(first["content_url"])
    assert download.status_code == 200
    assert download.content == b"generated report"
    assert "report.txt" in str(adapter.artifact_service.artifacts)


def test_provider_cleanup_failure_does_not_restore_access(monkeypatch):
    client, adapter = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]

    async def failing_delete(self, **kwargs):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(
        type(adapter.artifact_service), "delete_artifact", failing_delete
    )

    revoke = client.delete(f"/v1/artifacts/{artifact['id']}?delete_content=true")
    assert revoke.status_code == 200
    assert revoke.json()["deleted"] is True
    assert client.get(artifact["content_url"]).status_code == 404


def test_endpoint_cleanup_failure_is_retryable_after_provider_recovers(monkeypatch):
    client, adapter = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]

    real_delete = type(adapter.artifact_service).delete_artifact
    provider_down = {"value": True}

    async def flaky_delete(self, **kwargs):
        if provider_down["value"]:
            raise RuntimeError("provider is down")
        return await real_delete(self, **kwargs)

    monkeypatch.setattr(type(adapter.artifact_service), "delete_artifact", flaky_delete)

    # The endpoint stays best-effort: the failure is swallowed, access is
    # revoked, and the bytes are still there.
    revoke = client.delete(f"/v1/artifacts/{artifact['id']}?delete_content=true")
    assert revoke.status_code == 200
    assert client.get(artifact["content_url"]).status_code == 404
    assert "report.txt" in str(adapter.artifact_service.artifacts)

    # Once the provider recovers, repeating the DELETE completes the pending
    # cleanup instead of orphaning the bytes forever.
    provider_down["value"] = False
    retry = client.delete(f"/v1/artifacts/{artifact['id']}?delete_content=true")
    assert retry.status_code == 200
    assert retry.json()["deleted"] is True
    assert "report.txt" not in str(adapter.artifact_service.artifacts)

    # With the cleanup completed, the ID behaves like any revoked ID again.
    done = client.delete(f"/v1/artifacts/{artifact['id']}?delete_content=true")
    assert done.status_code == 404
    assert done.json()["error"]["code"] == "artifact_not_found"


def test_endpoint_falls_back_to_registry_when_adapter_lacks_revoke_artifact():
    from fastresponses.adapter import AgentAdapter

    class RegistryOnlyAdapter(AgentAdapter):
        name = "registry-only"

        def __init__(self):
            self.artifact_registry = ArtifactRegistry(max_records=4, ttl_seconds=60)

        async def run(self, run):  # pragma: no cover - never invoked
            yield None

    class RecordingService:
        def __init__(self):
            self.deleted: list[str] = []

        async def delete_artifact(self, **kwargs):
            self.deleted.append(kwargs["filename"])

    adapter = RegistryOnlyAdapter()
    service = RecordingService()
    client = TestClient(create_app(adapter))

    def register(filename, version):
        return adapter.artifact_registry.register(
            ArtifactRecord(service, "app", "user", "session", filename, version, "text/plain")
        )

    # Default revocation removes the record without touching provider content.
    plain_id = register("a.txt", 0)
    assert client.delete(f"/v1/artifacts/{plain_id}").status_code == 200
    assert adapter.artifact_registry.get(plain_id) is None
    assert service.deleted == []
    assert client.delete(f"/v1/artifacts/{plain_id}").status_code == 404

    # delete_content honours the sibling-version guard in the fallback too.
    first_id = register("a.txt", 0)
    second_id = register("a.txt", 1)
    assert (
        client.delete(f"/v1/artifacts/{second_id}?delete_content=true").status_code
        == 200
    )
    assert service.deleted == []
    assert adapter.artifact_registry.get(first_id) is not None

    # With no live siblings left, delete_content deletes the provider bytes.
    assert (
        client.delete(f"/v1/artifacts/{first_id}?delete_content=true").status_code
        == 200
    )
    assert service.deleted == ["a.txt"]

    # Unknown IDs stay non-disclosing.
    unknown = client.delete("/v1/artifacts/artifact_unknown")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "artifact_not_found"


def test_pending_cleanup_map_is_bounded_by_registry_capacity():
    llm = ScriptedLlm(turns=[], requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        artifact_registry_max_records=1,
        artifact_registry_ttl_seconds=60,
    )

    class FlakyService:
        def __init__(self):
            self.fail = True
            self.deleted: list[str] = []

        async def delete_artifact(self, **kwargs):
            if self.fail:
                raise RuntimeError("provider is down")
            self.deleted.append(kwargs["filename"])

    service = FlakyService()
    id_a = adapter.artifact_registry.register(
        ArtifactRecord(service, "app", "user", "session", "a.txt", 0, "text/plain")
    )
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.revoke_artifact(id_a, delete_content=True))
    id_b = adapter.artifact_registry.register(
        ArtifactRecord(service, "app", "user", "session", "b.txt", 0, "text/plain")
    )
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.revoke_artifact(id_b, delete_content=True))

    # The bound (max_records=1) evicted the oldest pending cleanup, so only
    # the newest one stays retryable; evicted IDs behave like revoked IDs.
    service.fail = False
    assert asyncio.run(adapter.revoke_artifact(id_a, delete_content=True)) is False
    assert asyncio.run(adapter.revoke_artifact(id_b, delete_content=True)) is True
    assert service.deleted == ["b.txt"]


def test_adapter_revoke_artifact_removes_access_but_keeps_content():
    client, adapter = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]

    assert asyncio.run(adapter.revoke_artifact(artifact["id"])) is True
    assert client.get(artifact["content_url"]).status_code == 404
    assert "report.txt" in str(adapter.artifact_service.artifacts)
    # Unknown, expired, and already-revoked IDs are indistinguishable.
    assert asyncio.run(adapter.revoke_artifact(artifact["id"])) is False
    assert asyncio.run(adapter.revoke_artifact("artifact_unknown")) is False


def test_adapter_revoke_artifact_with_delete_content_removes_provider_bytes():
    client, adapter = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]

    revoked = asyncio.run(adapter.revoke_artifact(artifact["id"], delete_content=True))
    assert revoked is True
    assert client.get(artifact["content_url"]).status_code == 404
    assert "report.txt" not in str(adapter.artifact_service.artifacts)


def test_adapter_revoke_artifact_preserves_sibling_versions():
    client, adapter = _make_artifact_adapter(
        [
            call_turn("save_report", {}),
            call_turn("save_report", {}),
            text_turn("Two reports saved."),
        ],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make two reports"}).json()
    first, second = _generated_artifacts(body)

    # A live sibling download ID blocks filename-wide provider deletion.
    revoked = asyncio.run(adapter.revoke_artifact(second["id"], delete_content=True))
    assert revoked is True
    assert client.get(second["content_url"]).status_code == 404
    download = client.get(first["content_url"])
    assert download.status_code == 200
    assert download.content == b"generated report"
    assert "report.txt" in str(adapter.artifact_service.artifacts)

    # With no live siblings left, content deletion proceeds.
    revoked = asyncio.run(adapter.revoke_artifact(first["id"], delete_content=True))
    assert revoked is True
    assert "report.txt" not in str(adapter.artifact_service.artifacts)


def test_delete_endpoint_rejects_invalid_delete_content_value():
    client, adapter = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]

    response = client.delete(f"/v1/artifacts/{artifact['id']}?delete_content=banana")
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"
    assert response.json()["error"]["code"] == "invalid_value"

    # The invalid request neither revoked the link nor deleted the content.
    assert client.get(artifact["content_url"]).status_code == 200
    assert "report.txt" in str(adapter.artifact_service.artifacts)


def test_adapter_revoke_artifact_provider_failure_is_retryable(monkeypatch):
    client, adapter = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]

    real_delete = type(adapter.artifact_service).delete_artifact
    provider_down = {"value": True}

    async def flaky_delete(self, **kwargs):
        if provider_down["value"]:
            raise RuntimeError("provider is down")
        return await real_delete(self, **kwargs)

    monkeypatch.setattr(type(adapter.artifact_service), "delete_artifact", flaky_delete)

    # Unlike the HTTP endpoint, the adapter surfaces the cleanup failure so
    # callers can retry, but public access stays revoked either way.
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.revoke_artifact(artifact["id"], delete_content=True))
    assert client.get(artifact["content_url"]).status_code == 404
    assert "report.txt" in str(adapter.artifact_service.artifacts)

    # Without delete_content the ID counts as already revoked, and the
    # pending provider cleanup is left for a later delete_content retry.
    assert asyncio.run(adapter.revoke_artifact(artifact["id"])) is False

    # Once the provider recovers, retrying completes the deferred cleanup.
    provider_down["value"] = False
    revoked = asyncio.run(adapter.revoke_artifact(artifact["id"], delete_content=True))
    assert revoked is True
    assert "report.txt" not in str(adapter.artifact_service.artifacts)
    assert client.get(artifact["content_url"]).status_code == 404

    # The completed cleanup is not retryable again.
    assert (
        asyncio.run(adapter.revoke_artifact(artifact["id"], delete_content=True))
        is False
    )


def test_deleting_response_does_not_revoke_artifacts():
    client, _ = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]

    deleted = client.delete(f"/v1/responses/{body['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    # Artifact lifecycle is independent of stored responses: items may be
    # replayed into forked or continued conversations.
    download = client.get(artifact["content_url"])
    assert download.status_code == 200
    assert download.content == b"generated report"


def test_get_response_reports_revoked_artifact_as_unavailable():
    client, _ = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]
    assert client.delete(f"/v1/artifacts/{artifact['id']}").status_code == 200

    refetched = client.get(f"/v1/responses/{body['id']}").json()
    refreshed = _generated_artifacts(refetched)[0]
    assert refreshed["available"] is False
    # Revoking never mutates unrelated fields of the item.
    assert refreshed["filename"] == artifact["filename"]
    assert refreshed["content_url"] == artifact["content_url"]


def _wait_for_status(client, response_id, statuses, timeout=5.0):
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        body = client.get(f"/v1/responses/{response_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"response never reached {statuses}: {body}")


def test_cancel_response_reports_revoked_artifact_as_unavailable():
    client, _ = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    with client:
        queued = client.post(
            "/v1/responses", json={"input": "make a report", "background": True}
        ).json()
        body = _wait_for_status(client, queued["id"], {"completed"})
        artifact = _generated_artifacts(body)[0]
        assert client.delete(f"/v1/artifacts/{artifact['id']}").status_code == 200

        # POST .../cancel returns a full response snapshot too, so it must
        # reflect the same live artifact availability as GET, not the stale
        # creation-time values.
        cancelled = client.post(f"/v1/responses/{body['id']}/cancel").json()
        assert cancelled["status"] == "completed"
        refreshed = _generated_artifacts(cancelled)[0]
        assert refreshed["available"] is False


def test_cancel_response_reports_expired_artifact_as_unavailable(monkeypatch):
    now = {"value": 1_000.0}
    monkeypatch.setattr("fastresponses.artifacts.time.monotonic", lambda: now["value"])
    monkeypatch.setattr("fastresponses.artifacts.time.time", lambda: now["value"])
    registry = ArtifactRegistry(max_records=8, ttl_seconds=10)
    client, _ = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
        adapter_kwargs={"artifact_registry": registry},
    )
    with client:
        queued = client.post(
            "/v1/responses", json={"input": "make a report", "background": True}
        ).json()
        body = _wait_for_status(client, queued["id"], {"completed"})
        artifact = _generated_artifacts(body)[0]
        assert artifact["available"] is True

        now["value"] = 1_011.0
        cancelled = client.post(f"/v1/responses/{body['id']}/cancel").json()
        assert cancelled["status"] == "completed"
        refreshed = _generated_artifacts(cancelled)[0]
        assert refreshed["available"] is False


def test_events_endpoint_replays_frozen_artifact_state_after_revocation():
    """The resumable event log is a point-in-time record: it must not
    reflect a later revocation, unlike GET /v1/responses/{id}."""
    client, _ = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    with client:
        queued = client.post(
            "/v1/responses", json={"input": "make a report", "background": True}
        ).json()
        body = _wait_for_status(client, queued["id"], {"completed"})
        artifact = _generated_artifacts(body)[0]
        assert client.delete(f"/v1/artifacts/{artifact['id']}").status_code == 200

        with client.stream(
            "GET", f"/v1/responses/{body['id']}/events"
        ) as r:
            events = [e for e in read_sse(r) if isinstance(e, dict)]

    artifact_events = [
        e
        for e in events
        if e.get("type") == "response.output_item.done"
        and e.get("item", {}).get("type") == "ajac-zero:artifact"
    ]
    assert len(artifact_events) == 1
    assert artifact_events[0]["item"]["available"] is True

    refetched = client.get(f"/v1/responses/{body['id']}").json()
    assert _generated_artifacts(refetched)[0]["available"] is False


def test_get_response_reports_evicted_artifact_as_unavailable():
    registry = ArtifactRegistry(max_records=1, ttl_seconds=3600)
    client, _ = _make_artifact_adapter(
        [
            call_turn("save_report", {}),
            call_turn("save_report", {}),
            text_turn("Two reports saved."),
        ],
        tools=[save_report],
        adapter_kwargs={"artifact_registry": registry},
    )
    body = client.post("/v1/responses", json={"input": "make two reports"}).json()
    first, second = _generated_artifacts(body)
    assert first["id"] != second["id"]

    # max_records=1 evicted the first record as soon as the second was
    # registered, within the same turn and before the response was ever
    # returned to the client — the immediate creation response must not
    # advertise it as available.
    assert first["available"] is False
    assert second["available"] is True

    refetched = client.get(f"/v1/responses/{body['id']}").json()
    refreshed_first, refreshed_second = _generated_artifacts(refetched)
    assert refreshed_first["available"] is False


def test_streaming_response_completed_event_reports_evicted_artifact():
    registry = ArtifactRegistry(max_records=1, ttl_seconds=3600)
    client, _ = _make_artifact_adapter(
        [
            call_turn("save_report", {}),
            call_turn("save_report", {}),
            text_turn("Two reports saved."),
        ],
        tools=[save_report],
        adapter_kwargs={"artifact_registry": registry},
    )
    with client.stream(
        "POST",
        "/v1/responses",
        json={"input": "make two reports", "stream": True},
    ) as r:
        events = [e for e in read_sse(r) if isinstance(e, dict)]

    final = next(e for e in events if e["type"] == "response.completed")
    first, second = _generated_artifacts(final["response"])
    # Same same-turn eviction as the non-streaming case, but observed via
    # the terminal SSE event's embedded response snapshot instead of a
    # follow-up GET.
    assert first["available"] is False
    assert second["available"] is True


def test_websocket_response_completed_reports_evicted_artifact():
    registry = ArtifactRegistry(max_records=1, ttl_seconds=3600)
    client, _ = _make_artifact_adapter(
        [
            call_turn("save_report", {}),
            call_turn("save_report", {}),
            text_turn("Two reports saved."),
        ],
        tools=[save_report],
        adapter_kwargs={"artifact_registry": registry},
    )
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "input": "make two reports"})
        final = None
        while final is None:
            event = ws.receive_json()
            if event["type"] in (
                "response.completed",
                "response.failed",
                "response.incomplete",
            ):
                final = event["response"]

    # Same same-turn eviction as the non-streaming/SSE cases, but observed
    # via the terminal event on the WebSocket transport.
    first, second = _generated_artifacts(final)
    assert first["available"] is False
    assert second["available"] is True


def test_get_response_refreshes_and_expires_artifact_availability(monkeypatch):
    now = {"value": 1_000.0}
    monkeypatch.setattr("fastresponses.artifacts.time.monotonic", lambda: now["value"])
    monkeypatch.setattr("fastresponses.artifacts.time.time", lambda: now["value"])
    registry = ArtifactRegistry(max_records=8, ttl_seconds=10)
    client, _ = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
        adapter_kwargs={"artifact_registry": registry},
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    artifact = _generated_artifacts(body)[0]
    assert artifact["expires_at"] == 1_010

    # Retrieving the response before expiry is a live access: it slides the
    # download window forward instead of letting a fixed clock run out.
    now["value"] = 1_009.0
    refetched = client.get(f"/v1/responses/{body['id']}").json()
    refreshed = _generated_artifacts(refetched)[0]
    assert refreshed["available"] is True
    assert refreshed["expires_at"] == 1_019
    assert client.get(artifact["content_url"]).status_code == 200

    # Without another access before the (now later) expiry, the link dies.
    now["value"] = 1_020.0
    refetched = client.get(f"/v1/responses/{body['id']}").json()
    refreshed = _generated_artifacts(refetched)[0]
    assert refreshed["available"] is False
    assert client.get(artifact["content_url"]).status_code == 404


# ---------------------------------------------------------------------------
# Formal schema (issue #13): every ajac-zero:artifact item, from either
# construction path, must validate against the public `ArtifactItem` model.
# ---------------------------------------------------------------------------


def test_session_generated_artifact_matches_documented_schema():
    """save_artifact-triggered items still follow the same function_call/
    function_call_output pair as any other tool call (README "Item ordering
    examples"), but the artifact item itself never carries call_id."""
    client, _ = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()

    assert [item["type"] for item in body["output"]] == [
        "function_call",
        "function_call_output",
        "ajac-zero:artifact",
        "message",
    ]
    raw = _generated_artifacts(body)[0]

    artifact = parse_artifact_item(raw)

    assert artifact.filename == "report.txt"
    assert artifact.mime_type == "text/plain"
    assert artifact.size == len(b"generated report")
    assert artifact.status == "completed"
    assert artifact.content_url == raw["content_url"]
    assert artifact.available is True
    assert artifact.call_id is None
    assert "call_id" not in raw


def test_mapper_created_artifact_matches_documented_schema_and_carries_call_id():
    """ADKToolResponse.create_artifact-triggered items always carry call_id,
    linking back to the internal function_call/function_call_output pair,
    and land after that pair in item order."""

    async def mapper(response: ADKToolResponse):
        return [
            await response.create_artifact("summary.txt", b"weather summary", "text/plain")
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
        "ajac-zero:artifact",
        "message",
    ]
    call_id = body["output"][0]["call_id"]
    raw = _generated_artifacts(body)[0]

    artifact = parse_artifact_item(raw)

    assert artifact.call_id == call_id
    assert artifact.filename == "summary.txt"
    assert artifact.size == len(b"weather summary")


def test_streaming_mapper_created_artifact_matches_documented_ordering():
    """Backs the README "Item ordering examples" streaming case: the
    artifact's response.output_item.done event lands immediately after the
    function_call/function_call_output pair that produced it, and carries
    that pair's call_id."""

    async def mapper(response: ADKToolResponse):
        return [
            await response.create_artifact("summary.txt", b"weather summary", "text/plain")
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
    client = TestClient(create_app(adapter))
    with client.stream(
        "POST", "/v1/responses", json={"input": "weather?", "stream": True}
    ) as stream:
        payloads = [event for event in read_sse(stream) if isinstance(event, dict)]

    done_items = [
        event["item"]
        for event in payloads
        if event["type"] == "response.output_item.done"
    ]
    assert [item["type"] for item in done_items] == [
        "function_call",
        "function_call_output",
        "ajac-zero:artifact",
        "message",
    ]
    call_id = done_items[0]["call_id"]
    assert done_items[1]["call_id"] == call_id
    assert done_items[2]["call_id"] == call_id
    assert payloads[-1]["type"] == "response.completed"


def test_artifact_item_schema_tolerates_unknown_future_fields():
    """New, additive fields must round-trip rather than raise, so older
    typed consumers keep working against a future minor release."""
    payload = {
        "type": "ajac-zero:artifact",
        "id": "artifact_abc123",
        "status": "completed",
        "filename": "report.txt",
        "mime_type": "text/plain",
        "size": 12,
        "content_url": "/v1/artifacts/artifact_abc123/content",
        "available": True,
        "expires_at": 1_700_000_000,
        "checksum": "sha256:deadbeef",  # hypothetical future field
    }

    artifact = ArtifactItem.model_validate(payload)

    assert artifact.model_dump(mode="json", exclude_none=True)["checksum"] == (
        "sha256:deadbeef"
    )


def test_artifact_item_schema_tolerates_pre_expiration_items_missing_fields():
    """available/expires_at were added to this item in a later release
    (fastresponses#19); an item predating them (e.g. replayed from
    GET /v1/responses/{id}/events, or read back from a response store
    populated by an earlier release) must still parse, defaulting to the
    documented best-effort semantics rather than raising."""
    payload = {
        "type": "ajac-zero:artifact",
        "id": "artifact_abc123",
        "status": "completed",
        "filename": "report.txt",
        "mime_type": "text/plain",
        "size": 12,
        "content_url": "/v1/artifacts/artifact_abc123/content",
        # no "available", no "expires_at"
    }

    artifact = parse_artifact_item(payload)

    assert artifact.available is True
    assert artifact.expires_at is None


@pytest.mark.parametrize(
    "content_url",
    [
        "https://example.com/artifact_abc123",  # absolute, not a relative path
        "/v1/artifacts/content",  # missing the artifact_id segment entirely
        "/v1/artifacts//content",  # empty artifact_id segment
        "/v1/artifacts/abc/def/content",  # id segment must not contain a slash
    ],
)
def test_artifact_item_schema_rejects_malformed_content_url(content_url):
    payload = {
        "type": "ajac-zero:artifact",
        "id": "artifact_abc123",
        "filename": "report.txt",
        "mime_type": "text/plain",
        "size": 12,
        "content_url": content_url,
        "available": True,
        "expires_at": 1_700_000_000,
    }

    with pytest.raises(ValidationError):
        ArtifactItem.model_validate(payload)


def test_parse_artifact_item_rejects_non_artifact_type():
    with pytest.raises(ValidationError):
        parse_artifact_item({"type": "message", "id": "msg_1", "role": "assistant"})


def test_artifact_item_round_trips_from_stored_response_json():
    """An artifact item read back out of stored response JSON (e.g. via
    ResponseStore) must parse identically to the one served live."""
    client, _ = _make_artifact_adapter(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()
    raw = _generated_artifacts(body)[0]

    round_tripped = ArtifactItem.model_validate(raw).model_dump(
        mode="json", exclude_none=True
    )

    assert round_tripped == raw


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


def test_artifact_registry_refresh_slides_expiry_without_changing_id(monkeypatch):
    now = {"value": 0.0}
    monkeypatch.setattr("fastresponses.artifacts.time.monotonic", lambda: now["value"])
    registry = ArtifactRegistry(max_records=4, ttl_seconds=10)
    record = ArtifactRecord(object(), "app", "user", "session", "a.txt", 0, "text/plain")
    artifact_id = registry.register(record)

    now["value"] = 9
    assert registry.refresh(artifact_id) is record
    # Without the refresh this would have expired at t=10.
    now["value"] = 11
    assert registry.get(artifact_id) is record

    now["value"] = 22
    assert registry.refresh(artifact_id) is None


def test_artifact_registry_refresh_treats_expired_and_unknown_ids_as_none(monkeypatch):
    now = {"value": 0.0}
    monkeypatch.setattr("fastresponses.artifacts.time.monotonic", lambda: now["value"])
    registry = ArtifactRegistry(max_records=4, ttl_seconds=10)
    record = ArtifactRecord(object(), "app", "user", "session", "a.txt", 0, "text/plain")
    artifact_id = registry.register(record)

    now["value"] = 11
    assert registry.refresh(artifact_id) is None
    assert registry.refresh("artifact_unknown") is None
    # The expired entry was self-healed (removed) by the failed refresh.
    assert registry.get(artifact_id) is None


def test_artifact_registry_revoke_removes_only_the_target_record(monkeypatch):
    now = {"value": 0.0}
    monkeypatch.setattr("fastresponses.artifacts.time.monotonic", lambda: now["value"])
    registry = ArtifactRegistry(max_records=4, ttl_seconds=10)
    service = object()
    v0 = ArtifactRecord(service, "app", "user", "session", "a.txt", 0, "text/plain")
    v1 = ArtifactRecord(service, "app", "user", "session", "a.txt", 1, "text/plain")
    first = registry.register(v0)
    second = registry.register(v1)

    assert registry.revoke(second) is v1
    assert registry.get(second) is None
    assert registry.get(first) is v0
    assert registry.revoke(second) is None
    assert registry.revoke("artifact_unknown") is None


def test_artifact_registry_revoke_treats_expired_records_as_unknown(monkeypatch):
    now = {"value": 0.0}
    monkeypatch.setattr("fastresponses.artifacts.time.monotonic", lambda: now["value"])
    registry = ArtifactRegistry(max_records=4, ttl_seconds=10)
    record = ArtifactRecord(object(), "app", "user", "session", "a.txt", 0, "text/plain")
    artifact_id = registry.register(record)
    now["value"] = 11
    assert registry.revoke(artifact_id) is None


def test_artifact_registry_live_reference_is_version_insensitive(monkeypatch):
    now = {"value": 0.0}
    monkeypatch.setattr("fastresponses.artifacts.time.monotonic", lambda: now["value"])
    registry = ArtifactRegistry(max_records=4, ttl_seconds=10)
    service = object()
    v0 = ArtifactRecord(service, "app", "user", "session", "a.txt", 0, "text/plain")
    v1 = ArtifactRecord(service, "app", "user", "session", "a.txt", 1, "text/plain")
    other = ArtifactRecord(service, "app", "user", "session", "b.txt", 0, "text/plain")
    registry.register(v0)
    other_id = registry.register(other)

    # Another live record targets the same provider filename (any version).
    assert registry.has_live_reference(v1) is True
    # No live record targets a revoked, unrelated filename.
    registry.revoke(other_id)
    assert registry.has_live_reference(other) is False
    # Expired records do not count as live references.
    now["value"] = 11
    assert registry.has_live_reference(v1) is False


def test_artifact_registry_live_reference_requires_the_same_service():
    registry = ArtifactRegistry(max_records=4, ttl_seconds=10)
    record = ArtifactRecord(object(), "app", "user", "session", "a.txt", 0, "text/plain")
    twin = ArtifactRecord(object(), "app", "user", "session", "a.txt", 0, "text/plain")
    registry.register(record)
    assert registry.has_live_reference(twin) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_records": 0},
        {"max_records": -1},
        {"ttl_seconds": 0},
        {"ttl_seconds": -1},
        {"ttl_seconds": float("nan")},
        {"max_records": float("nan")},
    ],
)
def test_artifact_registry_rejects_non_positive_configuration(kwargs):
    with pytest.raises(ValueError):
        ArtifactRegistry(**kwargs)


def test_adk_adapter_defaults_to_standard_registry_limits():
    llm = ScriptedLlm(turns=[], requests=[])
    adapter = ADKAdapter(Agent(name="test_agent", model=llm))
    assert adapter.artifact_registry.max_records == 1024
    assert adapter.artifact_registry.ttl_seconds == 3600


def test_adk_adapter_forwards_custom_registry_limits():
    llm = ScriptedLlm(turns=[], requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm),
        artifact_registry_max_records=8,
        artifact_registry_ttl_seconds=30,
    )
    assert adapter.artifact_registry.max_records == 8
    assert adapter.artifact_registry.ttl_seconds == 30


def test_adk_adapter_accepts_injected_artifact_registry():
    llm = ScriptedLlm(turns=[], requests=[])
    registry = ArtifactRegistry(max_records=2, ttl_seconds=5)
    adapter = ADKAdapter(Agent(name="test_agent", model=llm), artifact_registry=registry)
    assert adapter.artifact_registry is registry


def test_adk_adapter_rejects_registry_and_limits_together():
    llm = ScriptedLlm(turns=[], requests=[])
    registry = ArtifactRegistry(max_records=2, ttl_seconds=5)
    with pytest.raises(ValueError):
        ADKAdapter(
            Agent(name="test_agent", model=llm),
            artifact_registry=registry,
            artifact_registry_max_records=8,
        )


def test_adk_adapter_rejects_registry_and_limits_matching_defaults():
    # Explicitly passing the same values as the defaults must still be
    # treated as "the caller configured both", not silently ignored.
    llm = ScriptedLlm(turns=[], requests=[])
    registry = ArtifactRegistry(max_records=2, ttl_seconds=5)
    with pytest.raises(ValueError):
        ADKAdapter(
            Agent(name="test_agent", model=llm),
            artifact_registry=registry,
            artifact_registry_max_records=1024,
        )
    with pytest.raises(ValueError):
        ADKAdapter(
            Agent(name="test_agent", model=llm),
            artifact_registry=registry,
            artifact_registry_ttl_seconds=3600,
        )


def test_adk_adapter_propagates_invalid_registry_limits():
    llm = ScriptedLlm(turns=[], requests=[])
    with pytest.raises(ValueError):
        ADKAdapter(Agent(name="test_agent", model=llm), artifact_registry_max_records=0)
    with pytest.raises(ValueError):
        ADKAdapter(Agent(name="test_agent", model=llm), artifact_registry_ttl_seconds=0)


def _policy_client(turns, tools, **adapter_kwargs):
    llm = ScriptedLlm(turns=list(turns), requests=[])
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm, tools=tools),
        app_name="test-app",
        **adapter_kwargs,
    )
    return TestClient(create_app(adapter)), adapter


def test_oversized_tool_artifact_is_rejected_before_storage_and_download():
    client, adapter = _policy_client(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
        max_generated_artifact_bytes=8,
    )
    response = client.post("/v1/responses", json={"input": "make a report"})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "artifact_too_large"
    assert "report.txt" in error["message"]
    # Nothing was stored and no public download ID was handed out.
    assert adapter.artifact_registry._records == {}
    assert all("report.txt" not in key for key in adapter.artifact_service.artifacts)


def test_compliant_tool_artifact_passes_configured_policies():
    client, _ = _policy_client(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
        max_generated_artifact_bytes=len(b"generated report"),
        allowed_generated_artifact_mime_types=["text/plain"],
    )
    body = client.post("/v1/responses", json={"input": "make a report"}).json()

    assert body["status"] == "completed"
    artifact = next(
        item for item in body["output"] if item["type"] == "ajac-zero:artifact"
    )
    download = client.get(artifact["content_url"])
    assert download.content == b"generated report"


def test_mime_denylist_rejects_tool_artifact_in_streaming_and_non_streaming():
    make = lambda: _policy_client(  # noqa: E731
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
        blocked_generated_artifact_mime_types=["text/*"],
    )

    client, adapter = make()
    response = client.post("/v1/responses", json={"input": "make a report"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "artifact_mime_type_rejected"
    assert adapter.artifact_registry._records == {}

    client, adapter = make()
    with client.stream(
        "POST", "/v1/responses", json={"input": "make a report", "stream": True}
    ) as stream:
        payloads = [event for event in read_sse(stream) if isinstance(event, dict)]
    assert payloads[-1]["type"] == "response.failed"
    assert payloads[-1]["response"]["error"]["code"] == "artifact_mime_type_rejected"
    assert adapter.artifact_registry._records == {}


def test_mime_allowlist_rejects_non_matching_tool_artifact():
    client, adapter = _policy_client(
        [call_turn("save_report", {}), text_turn("Report ready.")],
        tools=[save_report],
        allowed_generated_artifact_mime_types=["image/png", "application/pdf"],
    )
    response = client.post("/v1/responses", json={"input": "make a report"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "artifact_mime_type_rejected"
    assert adapter.artifact_registry._records == {}


async def save_text_notes(tool_context: ToolContext) -> dict:
    """Save a generated text-part artifact."""
    version = await tool_context.save_artifact("notes.txt", types.Part(text="x" * 1000))
    return {"version": version}


def test_text_part_tool_artifact_is_subject_to_size_policy():
    client, adapter = _policy_client(
        [call_turn("save_text_notes", {}), text_turn("Saved.")],
        tools=[save_text_notes],
        max_generated_artifact_bytes=8,
    )
    response = client.post("/v1/responses", json={"input": "save notes"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "artifact_too_large"
    # Rejected before storage: the text part was never persisted.
    assert all("notes.txt" not in key for key in adapter.artifact_service.artifacts)
    assert adapter.artifact_registry._records == {}


def test_text_part_tool_artifact_is_subject_to_mime_policy():
    client, adapter = _policy_client(
        [call_turn("save_text_notes", {}), text_turn("Saved.")],
        tools=[save_text_notes],
        blocked_generated_artifact_mime_types=["text/*"],
    )
    response = client.post("/v1/responses", json={"input": "save notes"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "artifact_mime_type_rejected"
    assert all("notes.txt" not in key for key in adapter.artifact_service.artifacts)
    assert adapter.artifact_registry._records == {}


def test_mapper_created_artifact_policy_rejection_non_streaming():
    async def mapper(response: ADKToolResponse):
        return [await response.create_artifact("summary.bin", b"0" * 32, "text/plain")]

    llm = ScriptedLlm(
        turns=[call_turn("get_weather", {"city": "Tokyo"}), text_turn("Done.")],
        requests=[],
    )
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm, tools=[get_weather]),
        app_name="test-app",
        internal_tool_response_mapper=mapper,
        max_generated_artifact_bytes=16,
    )
    client = TestClient(create_app(adapter))
    response = client.post("/v1/responses", json={"input": "weather?"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "artifact_too_large"
    assert adapter.artifact_registry._records == {}
    assert all("summary.bin" not in key for key in adapter.artifact_service.artifacts)


def test_mapper_created_artifact_policy_rejection_keeps_canonical_pair():
    async def mapper(response: ADKToolResponse):
        return [await response.create_artifact("summary.bin", b"0" * 32, "text/plain")]

    llm = ScriptedLlm(
        turns=[call_turn("get_weather", {"city": "Tokyo"}), text_turn("Done.")],
        requests=[],
    )
    adapter = ADKAdapter(
        Agent(name="test_agent", model=llm, tools=[get_weather]),
        app_name="test-app",
        internal_tool_response_mapper=mapper,
        max_generated_artifact_bytes=16,
    )
    client = TestClient(create_app(adapter))
    with client.stream(
        "POST", "/v1/responses", json={"input": "weather?", "stream": True}
    ) as stream:
        payloads = [event for event in read_sse(stream) if isinstance(event, dict)]

    done_types = [
        event["item"]["type"]
        for event in payloads
        if event["type"] == "response.output_item.done"
    ]
    assert done_types == ["function_call", "function_call_output"]
    assert payloads[-1]["type"] == "response.failed"
    assert payloads[-1]["response"]["error"]["code"] == "artifact_too_large"
    # Rejected before storage: no artifact bytes and no download ID exist.
    assert adapter.artifact_registry._records == {}
    assert all("summary.bin" not in key for key in adapter.artifact_service.artifacts)


async def test_exposure_gate_never_registers_policy_violating_artifact():
    service = InMemoryArtifactService()
    await service.save_artifact(
        app_name="app",
        user_id="user",
        session_id="session",
        filename="big.bin",
        artifact=types.Part.from_bytes(
            data=b"0123456789", mime_type="application/octet-stream"
        ),
    )
    registry = ArtifactRegistry()
    translator = _EventTranslator(
        set(),
        {},
        artifact_service=service,
        artifact_registry=registry,
        artifact_policy=_GeneratedArtifactPolicy(max_bytes=4),
        app_name="app",
        user_id="user",
        session_id="session",
        internal_tool_response_mapper=None,
    )

    with pytest.raises(AdapterError) as err:
        await translator._artifact_item("big.bin", 0)

    assert err.value.code == "artifact_too_large"
    assert err.value.type == "invalid_request"
    assert registry._records == {}


async def test_exposure_gate_never_registers_mime_violating_artifact():
    service = InMemoryArtifactService()
    await service.save_artifact(
        app_name="app",
        user_id="user",
        session_id="session",
        filename="page.html",
        artifact=types.Part.from_bytes(data=b"<html>", mime_type="text/html"),
    )
    registry = ArtifactRegistry()
    translator = _EventTranslator(
        set(),
        {},
        artifact_service=service,
        artifact_registry=registry,
        artifact_policy=_GeneratedArtifactPolicy(
            blocked_mime_types=frozenset({"text/html"})
        ),
        app_name="app",
        user_id="user",
        session_id="session",
        internal_tool_response_mapper=None,
    )

    with pytest.raises(AdapterError) as err:
        await translator._artifact_item("page.html", 0)

    assert err.value.code == "artifact_mime_type_rejected"
    assert err.value.type == "invalid_request"
    assert registry._records == {}


async def save_file_reference(tool_context: ToolContext) -> dict:
    """Save a generated file-reference artifact."""
    version = await tool_context.save_artifact(
        "remote.html",
        types.Part(
            file_data=types.FileData(
                file_uri="gs://bucket/remote.html", mime_type="text/html"
            )
        ),
    )
    return {"version": version}


def test_file_reference_tool_artifact_is_subject_to_mime_policy():
    client, adapter = _policy_client(
        [call_turn("save_file_reference", {}), text_turn("Saved.")],
        tools=[save_file_reference],
        blocked_generated_artifact_mime_types=["text/html"],
    )
    response = client.post("/v1/responses", json={"input": "save it"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "artifact_mime_type_rejected"
    assert all("remote.html" not in key for key in adapter.artifact_service.artifacts)
    assert adapter.artifact_registry._records == {}


def test_file_reference_tool_artifact_is_exempt_from_size_cap():
    # File references carry no local bytes to measure, so only the MIME
    # policy applies to them; the size cap must not reject them.
    client, adapter = _policy_client(
        [call_turn("save_file_reference", {}), text_turn("Saved.")],
        tools=[save_file_reference],
        max_generated_artifact_bytes=1,
        allowed_generated_artifact_mime_types=["text/html"],
    )
    response = client.post("/v1/responses", json={"input": "save it"})

    # The reference is stored, but it exposes no inline bytes, so turning it
    # into a downloadable artifact item still fails without a registry entry.
    assert response.json()["error"]["code"] == "artifact_not_found"
    assert any("remote.html" in key for key in adapter.artifact_service.artifacts)
    assert adapter.artifact_registry._records == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_generated_artifact_bytes": 0},
        {"max_generated_artifact_bytes": -1},
        {"max_generated_artifact_bytes": float("nan")},
        {"allowed_generated_artifact_mime_types": []},
        {"allowed_generated_artifact_mime_types": ["text"]},
        {"allowed_generated_artifact_mime_types": ["*/*"]},
        {"blocked_generated_artifact_mime_types": ["not a mime type"]},
        {"blocked_generated_artifact_mime_types": [""]},
    ],
)
def test_adk_adapter_rejects_invalid_artifact_policy_configuration(kwargs):
    llm = ScriptedLlm(turns=[], requests=[])
    with pytest.raises(ValueError):
        ADKAdapter(Agent(name="test_agent", model=llm), **kwargs)


def test_artifact_mime_policy_normalizes_case_parameters_and_wildcards():
    policy = _GeneratedArtifactPolicy(
        allowed_mime_types=frozenset({"text/*"}),
        blocked_mime_types=frozenset({"text/html"}),
    )
    policy.check("a.txt", 1, "Text/Plain; charset=utf-8")
    with pytest.raises(ValueError) as err:
        policy.check("a.html", 1, "text/html")
    assert getattr(err.value, "code") == "artifact_mime_type_rejected"
    with pytest.raises(ValueError):
        policy.check("a.png", 1, "image/png")
    with pytest.raises(ValueError):
        policy.check("mystery", 1, None)


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

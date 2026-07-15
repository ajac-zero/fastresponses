"""End-to-end tests for the ADK adapter using a scripted (offline) LLM."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi.testclient import TestClient
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from fastresponses.adapters.adk import ADKAdapter
from fastresponses.server import create_app

from conftest import read_sse


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
                    types.Part(
                        function_call=types.FunctionCall(name=name, args=args)
                    )
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
    deltas = [
        e["delta"] for e in payloads if e["type"] == "response.output_text.delta"
    ]
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
            content=types.Content(role="model", parts=[thought_part("Tokyo is in Japan.")]),
        ),
        LlmResponse(
            partial=True,
            content=types.Content(role="model", parts=[types.Part(text="It is sunny.")]),
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
        part.text
        for content in contents
        for part in (content.parts or [])
        if part.text
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
    client, llm = make_adk_client([text_turn("A doc.")])
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
    assert blob.mime_type == "application/pdf"


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
            content=types.Content(
                role="model", parts=[types.Part(text="Once upon a")]
            ),
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

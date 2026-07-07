# open-responses-server

Serve agent frameworks over the [Open Responses](https://www.openresponses.org) API.

Build your agent with the framework you like — **Google ADK** first, more to come —
and expose it as an Open Responses provider. Any Open Responses / OpenAI Responses
compatible client (SDKs, UIs, routers, eval harnesses) can then talk to it with
zero custom integration.

```
┌────────────────────┐   POST /v1/responses    ┌───────────────────────────┐
│ Open Responses     │ ──────────────────────▶ │ open-responses-server     │
│ client (any SDK)   │ ◀────────────────────── │  engine ─ AgentAdapter ─▶ │──▶ ADK agent
└────────────────────┘   JSON or SSE events    └───────────────────────────┘
```

## Features

- **`POST /v1/responses`** with JSON responses or spec-compliant SSE streaming
  (semantic events, `sequence_number`, item/content-part lifecycles, `data: [DONE]`).
- **WebSocket transport** at the same `/v1/responses` resource: sequential
  `response.create` turns, connection-local `previous_response_id` continuation
  (works with `store: false` / zero data retention), `previous_response_not_found`
  error envelopes, cache eviction on failed continuation turns, and the
  spec's 60-minute connection lifetime (`websocket_connection_limit_reached`).
- **`previous_response_id` continuation** — conversations map to persistent ADK
  sessions, so history is not re-sent to the model. Stateless replay (full
  transcript in `input`) also works.
- **Client-defined function tools**: declare `tools` in the request and the ADK
  agent can call them. Control yields back to your client as a standard
  `function_call` output item; answer with a `function_call_output` item to resume.
- **Agent-internal tools** (functions owned by the ADK agent) run server-side and
  are surfaced as `adk:function_call` extension items — a receipt of what happened,
  per the spec's guidance for internally-hosted tools.
- **Reasoning**: model "thought" parts (e.g. Gemini thought summaries) are
  surfaced as `reasoning` output items with streamed
  `response.reasoning_summary_text.delta` events; thought signatures are attached
  as `encrypted_content` when available.
- `instructions`, `temperature`, `top_p`, `presence_penalty`,
  `frequency_penalty`, `top_logprobs`, `max_output_tokens`, and `tool_choice`
  (`auto` / `required` / `none` / forced function / `allowed_tools`) mapped to
  ADK. `allowed_tools` is enforced server-side as a hard constraint: calls to
  tools outside the allowed set are suppressed before execution.
- **Multimodal input**: `input_image` and `input_file` parts (data URLs,
  base64 file data, or remote URLs) are translated to genai `inline_data` /
  `file_data` parts — in fresh input and in replayed history.
- **Structured output**: `text.format` `json_schema` / `json_object` map to
  the model's native JSON-schema-constrained decoding.
- **`reasoning.effort` / `reasoning.summary`** map to a genai `ThinkingConfig`
  (thinking budget by effort level, thought summaries on request).
- **`max_tool_calls`** is enforced mid-run: when the model exceeds the budget
  the turn stops with status `incomplete` and
  `incomplete_details.reason: "max_tool_calls"`; token exhaustion
  (`MAX_TOKENS`) likewise yields `incomplete` with reason
  `max_output_tokens`. Incomplete responses stay continuable.
- **`background: true`**: returns a `queued` response immediately and executes
  the turn asynchronously; poll `GET /v1/responses/{id}` for the result.
- **Obfuscation padding** on `response.output_text.delta` events (on by
  default, disable with `stream_options.include_obfuscation: false`).
- **`POST /v1/responses/compact`** — compacts a conversation into a single
  round-trippable `compaction` item that can seed a new response chain.
- `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`, `store: false`, usage
  accounting, structured error envelopes, optional bearer-token auth.
- **Passes the full official Open Responses acceptance suite** — all 17 tests
  (HTTP and WebSocket transports) of the
  [compliance suite](https://www.openresponses.org/compliance).

## Install

```bash
uv add 'open-responses-server[adk]'
```

## Quickstart

Write an ADK agent (`weather_agent.py`):

```python
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    """Returns the current weather for a city."""
    return {"city": city, "forecast": "sunny", "temperature_c": 21}

agent = Agent(
    name="weather_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful weather assistant.",
    tools=[get_weather],
)
```

Serve it:

```bash
open-responses-server serve weather_agent.py:agent --port 8080
```

Call it with any Open Responses client:

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input": "What is the weather in Tokyo?"}'
```

Or with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
response = client.responses.create(input="What is the weather in Tokyo?")
print(response.output_text)
```

Streaming works the same way (`"stream": true` / `client.responses.create(stream=True)`).

### Multi-turn conversations

```bash
curl http://127.0.0.1:8080/v1/responses -d '{
  "input": "And in Paris?",
  "previous_response_id": "resp_..."
}'
```

The server keeps the conversation in an ADK session keyed to the response chain.

### Client-side tools (yield control back to your app)

```bash
curl http://127.0.0.1:8080/v1/responses -d '{
  "input": "Book a table for two at 7pm",
  "tools": [{
    "type": "function",
    "name": "book_table",
    "description": "Books a restaurant table",
    "parameters": {"type": "object", "properties": {
      "time": {"type": "string"}, "people": {"type": "integer"}}}
  }]
}'
```

When the agent decides to call `book_table`, the response contains a
`function_call` output item and control returns to you. Execute the tool and
resume:

```bash
curl http://127.0.0.1:8080/v1/responses -d '{
  "previous_response_id": "resp_...",
  "input": [{"type": "function_call_output", "call_id": "call_...", "output": "{\"confirmed\": true}"}],
  "tools": [ ...same tools... ]
}'
```

## Embedding in your own app

```python
from open_responses_server import create_app
from open_responses_server.adapters.adk import ADKAdapter
from weather_agent import agent

app = create_app(ADKAdapter(agent), api_key="my-secret")
# uvicorn.run(app, ...) or mount it in an existing FastAPI project
```

## Writing an adapter for another framework

Implement `AgentAdapter` — an async generator that translates one turn into a
handful of simple events. The engine handles all protocol mechanics (sequence
numbers, item/content-part lifecycles, SSE framing, storage):

```python
from open_responses_server import AgentAdapter, AgentRun, TextDelta, ItemDone, StateUpdate

class MyAdapter(AgentAdapter):
    name = "myfw"            # implementor slug for extension item types
    default_model = "my-model"

    async def run(self, run: AgentRun):
        # run.new_items       -> new input items from this request
        # run.context_items   -> full logical context (prev input+output+new)
        # run.previous_state  -> your opaque state from the previous response
        yield TextDelta("Hello ")
        yield TextDelta("world")
        yield StateUpdate({"session": "abc"})
```

## Development

```bash
uv sync --extra adk
uv run pytest
```

Tests run fully offline: protocol tests use a scripted adapter, and ADK tests
drive a real ADK `Runner` with a scripted `BaseLlm` (no API key needed).

### Compliance suite

`tests/test_compliance.py` runs the official Open Responses acceptance tests
(the CLI runner from [openresponses/openresponses](https://github.com/openresponses/openresponses),
same suite as the [web tester](https://www.openresponses.org/compliance))
against a local server backed by a deterministic ADK agent:

```bash
uv run pytest -m compliance
```

It needs `bun` on PATH and network access on the first run (the spec repo is
pinned and cached under `.compliance/`; pin a different revision with
`OPENRESPONSES_SPEC_REF`). Without `bun` the test skips itself. All 17 tests
pass, covering both HTTP and WebSocket transports.

## Current limitations

- `background: true` returns a `queued` snapshot and runs the turn
  asynchronously; intermediate `in_progress` snapshots are not persisted, and
  `background` cannot be combined with `stream` (poll `GET /v1/responses/{id}`).
- `include: ["message.output_text.logprobs"]` is accepted but logprob arrays
  stay empty (ADK does not surface per-token logprobs in its event stream; the
  request maps to `response_logprobs` on the model call).
- The WebSocket connection lifetime cap is enforced between turns, not
  mid-turn.
- Thought signatures arriving after the reasoning block has closed in the stream
  are not attached to output (the full-fidelity trace lives in the ADK session,
  so continuation never depends on the client echoing `encrypted_content` back).
- The response store and ADK sessions are in-memory; horizontal scaling needs a
  shared `ResponseStore` and ADK `SessionService` implementation.
- If a request's `input` ends with `function_call_output` items, those resume the
  paused tool call; mixing them with a *later* user message in the same request
  seeds the outputs as history instead.

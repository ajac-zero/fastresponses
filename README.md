# fastresponses

Serve agent frameworks over the [Open Responses](https://www.openresponses.org) API.

Build your agent with the framework you like — **Google ADK**,
**Pydantic AI**, **LangGraph**, or the **OpenAI Agents SDK** — and expose it
as an Open Responses provider. Any Open Responses / OpenAI Responses
compatible client (SDKs, UIs, routers, eval harnesses) can then talk to it
with zero custom integration.

```
┌────────────────────┐   POST /v1/responses    ┌───────────────────────────┐──▶ ADK agent
│ Open Responses     │ ──────────────────────▶ │ fastresponses             │──▶ Pydantic AI agent
│ client (any SDK)   │ ◀────────────────────── │  engine ─ AgentAdapter ─▶ │──▶ LangGraph graph
└────────────────────┘   JSON or SSE events    └───────────────────────────┘──▶ OpenAI Agents SDK
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
  are surfaced as provider-neutral, canonical `function_call` /
  `function_call_output` pairs. This replaces the earlier `adk:function_call`
  extension wire format.
- **Routed input attachments**: configure grouped extension routes for `inline`,
  `url`, `reference`, or `reject` handling. For example,
  `{"inline": [".pdf", ".png"], "reference": [".docx", ".zip"]}`. Matching is
  case-insensitive and longest-suffix-first. Unknown extensions use the explicit
  `default_input_file_action` (`reject` by default).
- `inline` and `reference` URL inputs are fetched only from an explicit origin
  allowlist, bounded by timeout, redirects, 32 MiB per file, 16 files, and 64 MiB
  of decoded file data per request. Lost-session replay is separately bounded to
  64 files and 128 MiB. `url` inputs count toward file-count limits but not decoded
  byte limits. `url` inputs are validated
  against the allowlist and passed through without fetching; because the model
  provider performs that fetch, redirect enforcement is also provider-owned. The built-in
  `ADKArtifactInputFileStore` preserves `attachment_N` references; a custom
  `InputFileReferenceStore` controls storage and the exact text shown to the model.
  FastResponses never injects a loader, tool, plugin, or retrieval instructions.

  ```python
  adapter = ADKAdapter(
      agent,
      input_file_url_origins=["https://files.example"],
      input_file_routes={
          "inline": [".pdf", ".png", ".jpg"],
          "url": [".mp3", ".mp4"],
          "reference": [".docx", ".xlsx", ".zip", ".tar.gz"],
      },
      default_input_file_action="reject",
  )
  ```

  Supply `input_file_router` instead of `input_file_routes` for per-file dynamic
  decisions, or `input_file_reference_store` to own reference persistence and
  formatting. Routes use the client-asserted filename as dispatch metadata; they
  do not verify the file's actual type. A `url` route requires `file_url`; it never
  falls back for `file_data`. Routers and stores are trusted application code and
  must be deterministic and replay-safe; stores should treat `reference_id` writes
  as idempotent because lost-session recovery can recreate historical references.
  Sequential `attachment_N` allocation is coordinated within one adapter process;
  deployments sharing ADK sessions across workers should provide external
  serialization or a custom store/reference scheme with collision-resistant IDs.
- Pass an `internal_tool_response_mapper` to `ADKAdapter` to derive additional
  namespaced items from completed internal tools. Mappers may create downloadable
  `ajac-zero:artifact` items backed by the authenticated artifact endpoint.
- Generated artifact download links are tracked in an in-process
  `ArtifactRegistry`, defaulting to a 3,600 second TTL and 1,024-record
  capacity. Tune these with `artifact_registry_ttl_seconds` /
  `artifact_registry_max_records`, or inject a pre-built instance (e.g. a
  shared implementation) via `artifact_registry=`. Both values must be
  positive; configure the registry or the two size/TTL kwargs, not both.

  ```python
  adapter = ADKAdapter(
      agent,
      artifact_registry_ttl_seconds=900,
      artifact_registry_max_records=256,
  )
  ```
- **Artifact lifecycle**: a download ID stops resolving when it expires, when
  it is evicted by capacity, or when it is explicitly revoked with
  `DELETE /v1/artifacts/{artifact_id}` (or, programmatically,
  `await adapter.revoke_artifact(artifact_id)`); all three return the same
  non-disclosing `artifact_not_found` error afterwards. Expiration, eviction,
  and plain revocation remove only the public download record and leave
  provider content in place — generated artifacts are part of the ADK session
  context, and later agent turns may still load them. Pass
  `?delete_content=true` (or `delete_content=True` to `revoke_artifact`) to
  also delete the provider content, which happens only when no other live
  download ID references the same provider filename — provider deletion is
  filename-wide, so this keeps other registered versions downloadable. Public
  access is always removed before provider cleanup, so a cleanup failure never
  restores a revoked ID; the endpoint logs cleanup failures best-effort, while
  `revoke_artifact` raises them. Failed cleanups stay retryable either way —
  repeat the `DELETE` with `?delete_content=true`, or call `revoke_artifact`
  again with `delete_content=True`, to re-attempt the deletion. Deleting or
  cancelling a
  response does not revoke its artifacts, because artifact items may be
  replayed into forked or continued conversations; revoke IDs explicitly when
  they must stop resolving.
- Generated artifacts (tool-saved or mapper-created) can be constrained with
  size and MIME-type policies. `max_generated_artifact_bytes` caps individual
  artifact size (default: unlimited); `allowed_generated_artifact_mime_types`
  is an allowlist (default `None` = allow all; must be non-empty when set) and
  `blocked_generated_artifact_mime_types` is a denylist (deny wins). Entries
  are matched case-insensitively without parameters and support `type/*`
  wildcards (e.g. `image/*`). Sizes are measured from inline bytes (text
  parts as UTF-8); file-reference parts carry no local bytes, so only the
  MIME-type policy applies to them. Violations are rejected *before* storage
  where possible and always before a public download ID is issued, so
  rejected artifacts are never downloadable. Both streaming and non-streaming
  responses fail with the stable error codes `artifact_too_large` or
  `artifact_mime_type_rejected` (`invalid_request`, HTTP 400 when
  non-streaming).

  ```python
  adapter = ADKAdapter(
      agent,
      max_generated_artifact_bytes=8 * 1024 * 1024,
      allowed_generated_artifact_mime_types=["application/pdf", "image/*", "text/plain"],
  )
  ```
- **Artifact expiration is surfaced, not silent**: every `ajac-zero:artifact`
  item carries `expires_at` (Unix seconds, set when the item is created;
  advisory — a wall-clock estimate, not an exact deadline) and `available`
  (whether the download link is currently expected to work; the
  authoritative signal). `GET /v1/responses/{id}`, `POST
  /v1/responses/{id}/cancel`, and the final `response.completed` /
  `.incomplete` snapshot of a non-background request (streamed or not,
  including the WebSocket transport) all recompute these fields against
  the live registry, so a response never keeps advertising a dead link
  without saying so — including the narrow case where a same-turn
  registry eviction (another artifact generated later in the same turn
  pushing an earlier one out) would otherwise make the very first response a
  client sees already stale. A `.failed` outcome is refreshed the same way
  when streamed over SSE (its `response.failed` event still carries a full
  response snapshot); non-streaming HTTP and the WebSocket transport instead
  surface a `.failed` outcome as an error envelope with no response/artifact
  data at all, so there is nothing to refresh on those two. Every one of
  these is a live access: while a
  registered artifact is still live, it extends its download window to a
  fresh full TTL from that moment (sliding expiration) and advances
  `expires_at` to match; once an artifact has expired, been evicted, or been
  revoked, `available` flips to `false` and `content_url` returns
  `artifact_not_found` like any other dead ID. By contrast,
  `GET /v1/responses/{id}/events` and a background request's streamed
  events replay a frozen, point-in-time event log for resumable streaming —
  a replayed item's `available`/`expires_at` reflect its state when the
  event was recorded, not the current state; use `GET /v1/responses/{id}`
  for current availability. Response retention (how long the response
  object stays in the response store) and artifact retention (how long its
  download link keeps working) are independent: a response can outlive its
  artifacts, and `store: false` responses are never persisted regardless of
  artifact liveness.
- **Reasoning**: model "thought" parts (e.g. Gemini thought summaries) are
  surfaced as `reasoning` output items with streamed
  `response.reasoning_summary_text.delta` events; thought signatures are attached
  as `encrypted_content` when available.
- `instructions`, `temperature`, `top_p`, `presence_penalty`,
  `frequency_penalty`, `top_logprobs`, `max_output_tokens`, and `tool_choice`
  (`auto` / `required` / `none` / forced function / `allowed_tools`) mapped to
  ADK. `allowed_tools` is enforced server-side as a hard constraint: calls to
  tools outside the allowed set are suppressed before execution.
- **Multimodal input**: `input_image` and `input_file` parts are translated to
  genai parts in fresh input and replayed history. `url` routes delegate fetching
  to the provider; `inline` and `reference` routes use FastResponses' bounded fetcher.
  Remote inputs are disabled unless their origin is explicitly allowed.
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
uv add 'fastresponses[adk]'            # Google ADK agents
uv add 'fastresponses[pydantic-ai]'    # Pydantic AI agents
uv add 'fastresponses[langgraph]'      # LangGraph graphs
uv add 'fastresponses[openai-agents]'  # OpenAI Agents SDK agents
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
fastresponses serve weather_agent.py:agent --port 8080
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

### Pydantic AI

The same works for a Pydantic AI agent — the CLI detects the framework:

```python
# weather_agent.py
from pydantic_ai import Agent

agent = Agent("openai:gpt-5.2", instructions="You are a weather assistant.")

@agent.tool_plain
def get_weather(city: str) -> dict:
    """Returns the current weather for a city."""
    return {"city": city, "forecast": "sunny", "temperature_c": 21}
```

```bash
fastresponses serve weather_agent.py:agent --port 8080
```

Framework mapping notes: client-declared `tools` become an `ExternalToolset`
(deferred tool calls yield control back to your client), agent-internal tools
are surfaced as `pydantic_ai:function_call` receipt items, `ThinkingPart`s
become `reasoning` items (signatures map to `encrypted_content`),
`previous_response_id` continuation stores the serialized Pydantic AI message
history, `reasoning.effort` maps to the unified `thinking` setting, and
`text.format` JSON schemas map to `StructuredDict` output. All adapters pass
the full official compliance suite.

### LangGraph

Any compiled graph following the `MessagesState` convention works:

```python
# weather_agent.py
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"It is sunny in {city}."

agent = create_agent("openai:gpt-5.2", tools=[get_weather])
```

```bash
fastresponses serve weather_agent.py:agent --port 8080
```

Framework mapping notes: `previous_response_id` continuation maps to
checkpointer threads (the adapter attaches a shared `InMemorySaver` when the
graph has none); graph-internal tools surface as `langgraph:function_call`
receipts. Client-declared `tools` need a *graph factory* — construct
`LangGraphAdapter(lambda client_tools: create_agent(model, tools=[...,
*client_tools]))` — and yield control via `interrupt()`: the adapter turns
interrupts into `function_call` items and resumes the graph with
`Command(resume=...)` when the client answers. Human-in-the-loop interrupts
from your own graph surface the same way (dicts with `name`/`args` keep
their tool name; anything else becomes a `human_input` call).
`allowed_tools` requests are rejected loudly: an arbitrary compiled graph
offers no hook for the hard enforcement the spec requires. See
`examples/langgraph_weather_agent.py`.

### OpenAI Agents SDK

```python
# weather_agent.py
from agents import Agent, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"It is sunny in {city}."

agent = Agent(name="weather_agent", model="gpt-5.2", tools=[get_weather])
```

```bash
fastresponses serve weather_agent.py:agent --port 8080
```

Framework mapping notes: the SDK already speaks Responses items, so mapping
is nearly direct. Client-declared `tools` become `needs_approval`
`FunctionTool`s — the model calling one pauses the run with an interruption,
surfaced as a `function_call` item (native `call_id` preserved); answering
with `function_call_output` approves it and resumes from the serialized
`RunState`, handing your output back as the tool result. Agent-internal
tools surface as `openai_agents:function_call` receipts, and `allowed_tools`
is enforced hard by wrapping out-of-set tools to refuse execution.

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
from fastresponses import create_app
from fastresponses.adapters.adk import ADKAdapter
from weather_agent import agent

app = create_app(ADKAdapter(agent), api_key="my-secret")
# uvicorn.run(app, ...) or mount it in an existing FastAPI project
```

## Background responses

`"background": true` runs the turn asynchronously and immediately returns a
`queued` snapshot. Beyond the basics, the server implements the OpenAI-style
conveniences:

- **Progressive snapshots** — `GET /v1/responses/{id}` shows `in_progress`
  status and output items as they complete, not just the final state.
- **Cancellation** — `POST /v1/responses/{id}/cancel` stops a running
  background response (`status: "cancelled"`); completed ones are returned
  unchanged.
- **Streaming** — `"background": true, "stream": true` returns live SSE
  backed by a replayable buffer.
- **Resumable streams** — if the connection drops, resume from the last
  event you saw: `GET /v1/responses/{id}/events?starting_after=<sequence_number>`
  replays buffered events from the cursor and follows live until the run
  finishes.

## Observability

Every turn emits one structured log record on the
`fastresponses.turn` logger — status, duration, token usage, output
item count, and error code, all as fields (`record.turn`) that any
structured-logging formatter can serialize.

With the `otel` extra (`fastresponses[otel]`) and an OpenTelemetry
SDK configured, each turn is additionally wrapped in an
`open_responses.turn` span carrying the same attributes plus
`gen_ai.usage.*` token counts. Spans parent to whatever context is active
when the turn starts, so ASGI auto-instrumentation composes naturally.

## Persistence

By default responses live in an in-process LRU store, so
`previous_response_id` continuation does not survive restarts. For a durable
single-file store (stdlib SQLite, WAL mode, no extra dependencies):

```bash
fastresponses serve weather_agent.py:agent --store responses.db
```

```python
from fastresponses import SQLiteResponseStore, create_app

app = create_app(adapter, store=SQLiteResponseStore("responses.db"))
```

Custom backends (Redis, Postgres, ...) implement the three-method
`ResponseStore` ABC (`get`/`put`/`delete`). ADK users should pair this with a
persistent `SessionService` (e.g. `DatabaseSessionService`) passed to
`ADKAdapter` so framework-side conversation state survives restarts too; the
Pydantic AI adapter keeps all conversation state in the response store
already.

## Artifact item schema

`ajac-zero:artifact` is an `ajac-zero`-namespaced extension item (a
`CustomItem` in `fastresponses.models` terms) surfaced by the Google ADK
adapter for downloadable generated artifacts (reports, files, images, ...
saved during a turn). This section is the formal, field-by-field contract
for that item, so consumers can implement handling for it without reading
adapter source. A typed model matching this contract ships as
`fastresponses.artifacts.ArtifactItem`, together with a
`parse_artifact_item(item)` helper that validates and parses any response
output item as one:

```python
from fastresponses.artifacts import parse_artifact_item

for item in response["output"]:
    if item["type"] == "ajac-zero:artifact":
        artifact = parse_artifact_item(item)
        if artifact.available:
            download(artifact.content_url)
```

### Fields

| Field | Type | Presence | Notes |
| --- | --- | --- | --- |
| `type` | `str` | always | Always the literal `"ajac-zero:artifact"`. |
| `id` | `str` | always | Opaque `ArtifactRegistry` token. Treat as an opaque string — no format is guaranteed beyond uniqueness. |
| `status` | `str` | always | Always `"completed"` today; this item is only ever emitted once the artifact is fully saved. |
| `filename` | `str` | always | Provider-declared filename, arbitrary Unicode. Not guaranteed unique across an entire conversation (later turns may reuse a filename as a new version). |
| `mime_type` | `str` | always | E.g. `"application/pdf"`. Best-effort (declared by the tool/provider, or guessed from the filename extension, falling back to `"application/octet-stream"`); not schema-enforced against the served `Content-Type`, which applies its own stricter fallback at download time. |
| `size` | `int` | always | Byte length of the artifact content (not characters). |
| `content_url` | `str` | always | Always a **relative path** of the exact form `/v1/artifacts/{id}/content` — never an absolute URL. Resolve it against the same origin/base URL you used for the Responses API call. |
| `available` | `bool` | always* | The authoritative signal. `false` means: do not attempt the download, it will fail. `true` (or the field being absent on an older item predating this field) is **best-effort only, never a guarantee** — always attempt the download and handle failure regardless. See "Mutability" below. |
| `expires_at` | `int \| None` | always* | Unix seconds when present. Advisory — a predicted deadline (`now + ttl` at the time this copy was produced), not an exact guarantee. Use it to decide when a cached copy is worth re-verifying, not as a hard cutover instant. `None`/absent means unknown, not "never expires" — treat it the same as an unknown `available`. |
| `call_id` | `str` | conditional | Present **only** for mapper-created artifacts (see "Construction paths" below). Its absence is meaningful — it means the artifact was not produced in response to a specific internal tool call — not missing data. |

\* Always present on items produced by the current adapter. `ArtifactItem`/`parse_artifact_item` default a missing `available` to `True` and a missing `expires_at` to `None` rather than raising, so older items predating these fields (e.g. replayed via `GET /v1/responses/{id}/events`, or read back from a response store populated by an earlier release) still parse — consistent with treating an absent `available` as best-effort-true.

### Construction paths

The item is identical in shape regardless of path, but `call_id` presence
differs:

- **Session-generated** (no `call_id`): the agent's own tool code calls
  ADK's `tool_context.save_artifact(...)` directly; the adapter picks this
  up from the turn's `artifact_delta` with no linkage back to a specific
  tool call.
- **Mapper-created** (always has `call_id`): an `internal_tool_response_mapper`
  calls `ADKToolResponse.create_artifact(filename, data, mime_type)`, which
  threads the call_id of the internal `function_call`/`function_call_output`
  pair that triggered it through to the resulting item.

### Item ordering examples

Non-streaming, session-generated (a tool call's own `save_artifact` still
produces the same `function_call`/`function_call_output` pair as any other
tool call — the artifact item that follows just has no `call_id` linking
it back to that pair):

```json
{
  "output": [
    { "type": "function_call", "call_id": "call_...", "name": "save_report", "...": "..." },
    { "type": "function_call_output", "call_id": "call_...", "...": "..." },
    { "type": "ajac-zero:artifact", "id": "artifact_...", "filename": "report.txt", "...": "no call_id key present" },
    { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "Report ready." }] }
  ]
}
```

Streaming, mapper-created (`internal_tool_response_mapper` calling
`create_artifact`), showing `response.output_item.done` order — the
artifact always lands immediately after the internal call pair that
produced it:

```
response.output_item.done  { "type": "function_call", "call_id": "call_abc", ... }
response.output_item.done  { "type": "function_call_output", "call_id": "call_abc", ... }
response.output_item.done  { "type": "ajac-zero:artifact", "call_id": "call_abc", ... }
response.output_item.done  { "type": "message", ... }
response.completed         { ... }
```

### Mutability and backward compatibility

`available` is recomputed fresh on every live read (`GET
/v1/responses/{id}`, cancel, and the terminal event of a live create — see
"Artifact expiration is surfaced, not silent" above). `expires_at` is
refreshed alongside it only while the artifact is still available; once
`available` flips to `false`, `expires_at` keeps whatever value it last
had — which may or may not already be in the past, depending on whether
the cause was natural TTL expiry, capacity eviction, or explicit
revocation — rather than being recomputed. Ignore `expires_at` once
`available` is `false`; it carries no meaning at that point. Neither
field is ever pushed to a copy the client already holds. A client that
caches an item (e.g. its own database) owns
re-verifying it — a cached `available: true` can silently go stale, while
a cached `available: false` for the same `id` will not, since a
revoked/evicted ID never comes back.

Every field in the table above — including `available` and `expires_at`
(the `"always*"` rows) — will not be removed or repurposed without a
breaking (major) release; the `*` only means their *per-item presence* is
tolerant of older data, not that the fields themselves are any less
stable than the plain `"always"` ones. `call_id` will continue to be
present only when applicable. New, additive fields may appear in a future
minor release; `ArtifactItem` (and `CustomItem` generally) is permissive
(`extra="allow"`), so unrecognized fields are preserved through
round-tripping rather than rejected — forward-compatible consumers should
do the same rather than assuming the field list above is exhaustive.

## Writing an adapter for another framework

Implement `AgentAdapter` — an async generator that translates one turn into a
handful of simple events. The engine handles all protocol mechanics (sequence
numbers, item/content-part lifecycles, SSE framing, storage):

```python
from fastresponses import AgentAdapter, AgentRun, TextDelta, ItemDone, StateUpdate

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
against local servers backed by deterministic (offline) agents — once per
adapter, ADK and Pydantic AI:

```bash
uv run pytest -m compliance
```

It needs `bun` on PATH and network access on the first run (the spec repo is
pinned and cached under `.compliance/`; pin a different revision with
`OPENRESPONSES_SPEC_REF`). Without `bun` the test skips itself. All 17 tests
pass for both adapters, covering both HTTP and WebSocket transports.

## Current limitations

- Background event buffers (for `GET /v1/responses/{id}/events` resumption)
  are in-process and bounded to the 64 most recent background runs.
- Generated artifact download IDs are process-local, retained for one hour, and
  bounded to the 1,024 most recently used records by default. Restarts and
  requests routed to another worker invalidate those URLs; multi-worker
  deployments need sticky routing or a shared registry implementation. The
  retention window and capacity are configurable via `ADKAdapter`'s
  `artifact_registry_ttl_seconds` / `artifact_registry_max_records`, or by
  passing a pre-built `artifact_registry=ArtifactRegistry(...)` (e.g. a shared
  or differently tuned instance).
- `include: ["message.output_text.logprobs"]` is accepted but logprob arrays
  stay empty (ADK does not surface per-token logprobs in its event stream; the
  request maps to `response_logprobs` on the model call).
- The WebSocket connection lifetime cap is enforced between turns, not
  mid-turn.
- Thought signatures arriving after the reasoning block has closed in the stream
  are not attached to output (the full-fidelity trace lives in the ADK session,
  so continuation never depends on the client echoing `encrypted_content` back).
- Multi-host deployments need a shared `ResponseStore` implementation (the
  bundled `SQLiteResponseStore` is single-host; the interface is three async
  methods, so a Redis/Postgres store is straightforward). For the ADK adapter,
  pass a shared `SessionService` (e.g. `DatabaseSessionService`) for the same
  reason.
- If a request's `input` ends with `function_call_output` items, those resume the
  paused tool call; mixing them with a *later* user message in the same request
  seeds the outputs as history instead.

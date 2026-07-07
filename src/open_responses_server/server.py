"""HTTP server exposing an :class:`AgentAdapter` as an Open Responses API.

Implements the core Open Responses surface:

- ``POST /v1/responses`` — create a response (``application/json`` or,
  with ``"stream": true``, ``text/event-stream`` semantic events terminated
  by ``data: [DONE]``).
- ``WS /v1/responses`` — WebSocket transport: ``response.create`` messages
  answered with the same streaming events, one in-flight response at a time,
  with connection-local ``previous_response_id`` continuation (including
  ``store: false``) and eviction on failed continuation turns.
- ``POST /v1/responses/compact`` — compact a conversation into a
  round-trippable ``compaction`` item.
- ``GET /v1/responses/{response_id}`` — retrieve a stored response.
- ``DELETE /v1/responses/{response_id}`` — delete a stored response.
"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .adapter import AgentAdapter, AgentRun
from .compaction import compact_items
from .engine import ResponseEngine, collect_response
from .models import (
    ERROR_STATUS_CODES,
    CompactResource,
    ErrorBody,
    ErrorEnvelope,
    ErrorEvent,
    ErrorPayload,
    Item,
    Response,
    ResponseFailedEvent,
    ResponsesRequest,
    WebSocketErrorEvent,
)
from .store import InMemoryResponseStore, ResponseStore, StoredResponse

WS_LOCAL_CACHE_LIMIT = 32

# WebSocket connections are limited to 60 minutes per the specification.
WS_CONNECTION_LIMIT_SECONDS = 60 * 60

# HTTP/SSE transport-specific fields that must not be part of a WebSocket
# response.create message body.
_WS_STRIPPED_FIELDS = ("type", "stream", "stream_options", "background")


def _event_json(event) -> str:
    """Serialize a streaming event, omitting the obfuscation key when unset."""
    data = event.model_dump()
    if data.get("obfuscation") is None:
        data.pop("obfuscation", None)
    return json.dumps(data, separators=(",", ":"))


class ApiError(Exception):
    def __init__(
        self,
        message: str,
        *,
        type: str = "invalid_request",
        code: str | None = None,
        param: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.body = ErrorBody(type=type, code=code, message=message, param=param)
        self.status_code = status_code or ERROR_STATUS_CODES.get(type, 500)


def _error_response(exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorEnvelope(error=exc.body).model_dump(),
    )


def create_app(
    adapter: AgentAdapter,
    *,
    store: ResponseStore | None = None,
    api_key: str | None = None,
    title: str | None = None,
    ws_connection_limit_seconds: float = WS_CONNECTION_LIMIT_SECONDS,
) -> FastAPI:
    """Build a FastAPI app serving ``adapter`` over the Open Responses API.

    Args:
        adapter: The agent adapter to expose.
        store: Response store used for ``previous_response_id`` continuation.
            Defaults to an in-memory store.
        api_key: If set, requests must carry ``Authorization: Bearer <api_key>``.
        title: OpenAPI title for the app.
        ws_connection_limit_seconds: Maximum WebSocket connection lifetime;
            when reached the server sends a ``websocket_connection_limit_reached``
            error envelope and closes the socket (spec: 60 minutes).
    """
    response_store = store or InMemoryResponseStore()
    engine = ResponseEngine(adapter, response_store)
    app = FastAPI(title=title or f"Open Responses ({adapter.name})")
    app.state.adapter = adapter
    app.state.response_store = response_store
    app.state.background_tasks = set()

    @app.exception_handler(ApiError)
    async def _handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(part) for part in first.get("loc", []) if part != "body"]
        return _error_response(
            ApiError(
                first.get("msg", "Invalid request body."),
                type="invalid_request",
                code="invalid_value",
                param=".".join(loc) or None,
            )
        )

    async def _authorize(request: Request) -> None:
        if api_key is None:
            return
        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ").strip()
        if token != api_key:
            raise ApiError(
                "Incorrect or missing API key.",
                type="invalid_request",
                code="invalid_api_key",
                status_code=401,
            )

    async def _build_run(
        payload: ResponsesRequest,
        local_cache: OrderedDict[str, StoredResponse] | None = None,
    ) -> AgentRun:
        new_items: list[Item] = payload.input_items()
        context_items: list[Item] = list(new_items)
        previous_state = None
        if payload.previous_response_id:
            stored = None
            if local_cache is not None:
                stored = local_cache.get(payload.previous_response_id)
            if stored is None:
                stored = await response_store.get(payload.previous_response_id)
            if stored is None:
                raise ApiError(
                    f"Previous response with id '{payload.previous_response_id}' "
                    "not found.",
                    type="invalid_request",
                    code="previous_response_not_found",
                    param="previous_response_id",
                )
            context_items = [*stored.context_items(), *new_items]
            previous_state = dict(stored.adapter_state)
        return AgentRun(
            request=payload,
            new_items=new_items,
            context_items=context_items,
            previous_state=previous_state,
        )

    async def _start_background(run: AgentRun) -> Response:
        """Start a response in the background; returns the queued snapshot."""
        events = engine.events(run)
        first = await events.__anext__()  # response.created
        response: Response = first.response  # type: ignore[union-attr]
        queued = response.model_copy(deep=True)
        queued.status = "queued"
        await response_store.put(
            StoredResponse(response=queued, input_items=run.context_items)
        )

        async def drain() -> None:
            try:
                async for _ in events:
                    pass
            except Exception:  # pragma: no cover - defensive
                pass

        task = asyncio.create_task(drain())
        app.state.background_tasks.add(task)
        task.add_done_callback(app.state.background_tasks.discard)
        return queued

    @app.post("/v1/responses")
    async def create_response(payload: ResponsesRequest, request: Request):
        await _authorize(request)
        if payload.background:
            if payload.stream:
                raise ApiError(
                    "Streaming background responses are not supported; poll "
                    "GET /v1/responses/{id} instead.",
                    code="unsupported_parameter",
                    param="stream",
                )
            if payload.store is False:
                raise ApiError(
                    "Background responses require 'store' to be true.",
                    code="invalid_value",
                    param="store",
                )
            run = await _build_run(payload)
            queued = await _start_background(run)
            return JSONResponse(content=queued.model_dump())

        run = await _build_run(payload)

        if payload.stream:
            return StreamingResponse(
                _sse(engine.events(run)),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        response = await collect_response(engine.events(run))
        if response.status == "failed" and response.error is not None:
            error_type = getattr(response.error, "type", None) or "server_error"
            raise ApiError(
                response.error.message,
                type=error_type,
                code=response.error.code,
                param=getattr(response.error, "param", None),
            )
        return JSONResponse(content=response.model_dump(exclude_none=False))

    @app.post("/v1/responses/compact")
    async def compact_response(payload: ResponsesRequest, request: Request):
        await _authorize(request)
        if not payload.model:
            raise ApiError(
                "The 'model' parameter is required.",
                type="invalid_request",
                code="missing_required_parameter",
                param="model",
            )
        run = await _build_run(payload)
        compacted = CompactResource(output=[compact_items(run.context_items)])
        return JSONResponse(content=compacted.model_dump())

    @app.get("/v1/responses/{response_id}")
    async def get_response(response_id: str, request: Request):
        await _authorize(request)
        stored = await response_store.get(response_id)
        if stored is None:
            raise ApiError(
                f"Response with id '{response_id}' not found.",
                type="not_found",
                code="response_not_found",
                param="response_id",
            )
        return JSONResponse(content=stored.response.model_dump())

    @app.delete("/v1/responses/{response_id}")
    async def delete_response(response_id: str, request: Request):
        await _authorize(request)
        deleted = await response_store.delete(response_id)
        if not deleted:
            raise ApiError(
                f"Response with id '{response_id}' not found.",
                type="not_found",
                code="response_not_found",
                param="response_id",
            )
        return JSONResponse(
            content={"id": response_id, "object": "response", "deleted": True}
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "adapter": adapter.name}

    # ------------------------------------------------------------------
    # WebSocket transport
    # ------------------------------------------------------------------

    async def _ws_error(
        websocket: WebSocket,
        *,
        status: int,
        code: str,
        message: str,
        error_type: str | None = None,
        param: str | None = None,
    ) -> None:
        event = WebSocketErrorEvent(
            status=status,
            error=ErrorPayload(
                type=error_type or _type_for_status(status),
                code=code,
                message=message,
                param=param,
            ),
        )
        await websocket.send_text(
            json.dumps(event.model_dump(exclude_none=True), separators=(",", ":"))
        )

    async def _ws_turn(
        websocket: WebSocket,
        raw: str,
        local_cache: OrderedDict[str, StoredResponse],
    ) -> None:
        """Process a single response.create message on the connection."""
        try:
            payload = json.loads(raw)
        except ValueError:
            await _ws_error(
                websocket,
                status=400,
                code="invalid_json",
                message="WebSocket message was not valid JSON.",
            )
            return
        if not isinstance(payload, dict) or payload.get("type") != "response.create":
            await _ws_error(
                websocket,
                status=400,
                code="invalid_value",
                message="WebSocket messages must be 'response.create' events.",
                param="type",
            )
            return

        body = {k: v for k, v in payload.items() if k not in _WS_STRIPPED_FIELDS}
        try:
            request_model = ResponsesRequest.model_validate(body)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            await _ws_error(
                websocket,
                status=400,
                code="invalid_value",
                message=str(first.get("msg", "Invalid request body.")),
                param=".".join(str(p) for p in first.get("loc", [])) or None,
            )
            return

        previous_id = request_model.previous_response_id
        try:
            run = await _build_run(request_model, local_cache=local_cache)
        except ApiError as exc:
            await _ws_error(
                websocket,
                status=exc.status_code,
                code=exc.body.code or exc.body.type,
                message=exc.body.message,
                error_type=exc.body.type,
                param=exc.body.param,
            )
            return

        stored_holder: list[StoredResponse] = []

        async def on_stored(stored: StoredResponse) -> None:
            stored_holder.append(stored)

        failed = False
        async for event in engine.events(run, on_stored=on_stored):
            if isinstance(event, ErrorEvent):
                # WebSocket failures are sent as a single error envelope
                # instead of the SSE error + response.failed pair.
                failed = True
                status = ERROR_STATUS_CODES.get(event.error.type, 500)
                await _ws_error(
                    websocket,
                    status=status,
                    code=event.error.code or event.error.type,
                    message=event.error.message,
                    error_type=event.error.type,
                    param=event.error.param,
                )
                continue
            if failed and isinstance(event, ResponseFailedEvent):
                continue
            await websocket.send_text(_event_json(event))

        if failed:
            # A failed continuation turn must evict the referenced response
            # from the connection-local cache.
            if previous_id:
                local_cache.pop(previous_id, None)
            return

        if stored_holder:
            stored = stored_holder[-1]
            local_cache[stored.response.id] = stored
            local_cache.move_to_end(stored.response.id)
            while len(local_cache) > WS_LOCAL_CACHE_LIMIT:
                local_cache.popitem(last=False)

    @app.websocket("/v1/responses")
    async def responses_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        if api_key is not None:
            header = websocket.headers.get("authorization", "")
            token = header.removeprefix("Bearer ").strip()
            if token != api_key:
                await _ws_error(
                    websocket,
                    status=401,
                    code="invalid_api_key",
                    message="Incorrect or missing API key.",
                    error_type="invalid_request",
                )
                await websocket.close(code=1008)
                return

        # Connection-local continuation state: enables previous_response_id
        # with store=false on the same socket without persisting anything.
        local_cache: OrderedDict[str, StoredResponse] = OrderedDict()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + ws_connection_limit_seconds
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=remaining
                )
                await _ws_turn(websocket, raw, local_cache)
        except TimeoutError:
            await _ws_error(
                websocket,
                status=408,
                code="websocket_connection_limit_reached",
                message=(
                    "This WebSocket connection reached its maximum lifetime; "
                    "open a new connection to continue."
                ),
                error_type="invalid_request",
            )
            await websocket.close(code=1000)
        except WebSocketDisconnect:
            return

    return app


def _type_for_status(status: int) -> str:
    for error_type, code in ERROR_STATUS_CODES.items():
        if code == status:
            return error_type
    return "invalid_request" if 400 <= status < 500 else "server_error"


async def _sse(events: AsyncIterator) -> AsyncIterator[str]:
    """Frame streaming events as Server-Sent Events."""
    async for event in events:
        yield f"event: {event.type}\ndata: {_event_json(event)}\n\n"
    yield "data: [DONE]\n\n"

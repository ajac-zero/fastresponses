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
- ``POST /v1/responses/{response_id}/cancel`` — cancel a running
  background response.
- ``GET /v1/responses/{response_id}/events`` — replay/resume the event
  stream of a background response (``?starting_after=<sequence_number>``).
- ``GET /v1/artifacts/{artifact_id}/content`` — download a generated
  artifact.
- ``DELETE /v1/artifacts/{artifact_id}`` — revoke a generated artifact
  download ID (``?delete_content=true`` also deletes the provider bytes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from urllib.parse import quote

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response as HttpResponse, StreamingResponse
from pydantic import ValidationError

from .adapter import AgentAdapter, AgentRun
from .artifacts import ARTIFACT_TYPE, ArtifactRegistry
from .compaction import compact_items
from .engine import ResponseEngine, collect_response
from .models import (
    ERROR_STATUS_CODES,
    CompactResource,
    CustomItem,
    ErrorBody,
    ErrorEnvelope,
    ErrorEvent,
    ErrorPayload,
    Item,
    OutputItemDoneEvent,
    Response,
    ResponseFailedEvent,
    ResponseInProgressEvent,
    ResponsesRequest,
    WebSocketErrorEvent,
)
from .store import InMemoryResponseStore, ResponseStore, StoredResponse

logger = logging.getLogger(__name__)

WS_LOCAL_CACHE_LIMIT = 32

# WebSocket connections are limited to 60 minutes per the specification.
WS_CONNECTION_LIMIT_SECONDS = 60 * 60

# HTTP/SSE transport-specific fields that must not be part of a WebSocket
# response.create message body.
_WS_STRIPPED_FIELDS = ("type", "stream", "stream_options", "background")


# Characters allowed verbatim inside an RFC 5987 ``ext-value`` (attr-char),
# minus the characters :func:`urllib.parse.quote` already never encodes.
_RFC5987_ATTR_CHARS = "!#$&+^`|"


def _content_disposition(filename: str) -> str:
    """Build a forced-download ``Content-Disposition`` header value.

    Emits a sanitized ASCII ``filename`` fallback for legacy clients and,
    when the original name carries additional (e.g. Unicode) characters, an
    RFC 5987/6266 ``filename*`` parameter so modern clients preserve the
    intended international filename. Control characters are dropped and path
    separators neutralized so filenames can never inject or split headers.
    """
    cleaned = "".join(ch for ch in filename if ch.isprintable())
    cleaned = cleaned.replace("/", "_").replace("\\", "_").strip(" .")
    ascii_fallback = re.sub(r"[^A-Za-z0-9._ -]", "_", cleaned)
    ascii_fallback = ascii_fallback.strip(" .") or "artifact"
    header = f'attachment; filename="{ascii_fallback}"'
    if cleaned and cleaned != ascii_fallback:
        encoded = quote(cleaned, safe=_RFC5987_ATTR_CHARS)
        header += f"; filename*=UTF-8''{encoded}"
    return header


def _refresh_artifact_items(
    response: Response, registry: ArtifactRegistry | None
) -> Response:
    """Report current download availability on ``ajac-zero:artifact`` items.

    Retrieving a stored response is a live access, so still-registered
    artifacts have their download window extended (sliding TTL) and their
    ``expires_at`` advanced accordingly. Artifacts that are no longer
    registered — expired, evicted by capacity, or explicitly revoked — are
    reported with ``available: False`` rather than silently continuing to
    advertise a dead ``content_url``. The registry is the sole source of
    truth for availability and is never itself persisted, so this is
    recomputed fresh on every retrieval; response retention (how long the
    response object itself is stored) and artifact retention (how long its
    download link keeps working) are independent guarantees.
    """
    if registry is None:
        return response
    output: list[Item] = []
    changed = False
    for item in response.output:
        if (
            isinstance(item, CustomItem)
            and item.type == ARTIFACT_TYPE
            and isinstance(item.id, str)
        ):
            record = registry.refresh(item.id)
            update = (
                {
                    "available": True,
                    "expires_at": int(time.time() + registry.ttl_seconds),
                }
                if record is not None
                else {"available": False}
            )
            item = item.model_copy(update=update)
            changed = True
        output.append(item)
    return response.model_copy(update={"output": output}) if changed else response


def _event_json(event) -> str:
    """Serialize a streaming event, omitting the obfuscation key when unset."""
    data = event.model_dump()
    if data.get("obfuscation") is None:
        data.pop("obfuscation", None)
    return json.dumps(data, separators=(",", ":"))


# Buffered background runs kept for resumable streaming (most recent first
# to be evicted last).
BACKGROUND_RUN_BUFFER_LIMIT = 64


@dataclass
class _BackgroundRun:
    """A background response run with a replayable event buffer.

    Events are buffered as pre-framed SSE strings; ``stream`` replays from a
    cursor (``sequence_number``) and then follows live until the run ends.
    """

    response_id: str
    queued: Response
    events: list[tuple[int, str, str]] = field(default_factory=list)
    done: bool = False
    task: asyncio.Task | None = None

    def __post_init__(self) -> None:
        self._condition = asyncio.Condition()

    def append(self, event) -> None:
        self.events.append((event.sequence_number, event.type, _event_json(event)))

    async def notify(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    async def finish(self) -> None:
        self.done = True
        await self.notify()

    async def stream(self, *, starting_after: int) -> AsyncIterator[str]:
        """SSE frames for events with sequence_number > starting_after."""
        index = 0
        while True:
            while index < len(self.events):
                seq, event_type, payload = self.events[index]
                index += 1
                if seq > starting_after:
                    yield f"event: {event_type}\ndata: {payload}\n\n"
            if self.done:
                break
            async with self._condition:
                if self.done or index < len(self.events):
                    continue
                await self._condition.wait()
        yield "data: [DONE]\n\n"


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
    app.state.background_runs = OrderedDict()
    app.state.artifact_registry = getattr(adapter, "artifact_registry", None)

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

    async def _start_background(run: AgentRun) -> _BackgroundRun:
        """Start a response in the background; buffers its events for
        resumable streaming and persists progressive snapshots."""
        events = engine.events(run)
        first = await events.__anext__()  # response.created
        response: Response = first.response  # type: ignore[union-attr]
        queued = response.model_copy(deep=True)
        queued.status = "queued"
        await response_store.put(
            StoredResponse(response=queued, input_items=run.context_items)
        )

        bg = _BackgroundRun(response_id=queued.id, queued=queued)
        bg.append(first)
        app.state.background_runs[queued.id] = bg
        while len(app.state.background_runs) > BACKGROUND_RUN_BUFFER_LIMIT:
            app.state.background_runs.popitem(last=False)

        async def drain() -> None:
            snapshot = queued.model_copy(deep=True)
            snapshot.status = "in_progress"
            try:
                async for event in events:
                    bg.append(event)
                    await bg.notify()
                    # Persist progressive snapshots so GET shows progress.
                    if isinstance(event, OutputItemDoneEvent):
                        snapshot.output.append(event.item)
                        await response_store.put(
                            StoredResponse(
                                response=snapshot.model_copy(deep=True),
                                input_items=run.context_items,
                            )
                        )
                    elif isinstance(event, ResponseInProgressEvent):
                        await response_store.put(
                            StoredResponse(
                                response=snapshot.model_copy(deep=True),
                                input_items=run.context_items,
                            )
                        )
            except asyncio.CancelledError:
                cancelled = snapshot.model_copy(deep=True)
                cancelled.status = "cancelled"
                cancelled.completed_at = int(time.time())
                await response_store.put(
                    StoredResponse(response=cancelled, input_items=run.context_items)
                )
                raise
            except Exception:  # pragma: no cover - defensive
                pass
            finally:
                await bg.finish()

        bg.task = asyncio.create_task(drain())
        app.state.background_tasks.add(bg.task)
        bg.task.add_done_callback(app.state.background_tasks.discard)
        return bg

    @app.post("/v1/responses")
    async def create_response(payload: ResponsesRequest, request: Request):
        await _authorize(request)
        if payload.background:
            if payload.store is False:
                raise ApiError(
                    "Background responses require 'store' to be true.",
                    code="invalid_value",
                    param="store",
                )
            run = await _build_run(payload)
            bg = await _start_background(run)
            if payload.stream:
                return StreamingResponse(
                    bg.stream(starting_after=-1),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            return JSONResponse(content=bg.queued.model_dump())

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
        response = _refresh_artifact_items(stored.response, app.state.artifact_registry)
        return JSONResponse(content=response.model_dump())

    @app.get("/v1/artifacts/{artifact_id}/content")
    async def get_artifact_content(artifact_id: str, request: Request):
        await _authorize(request)
        registry = app.state.artifact_registry
        record = registry.get(artifact_id) if registry is not None else None
        if record is None:
            raise ApiError(
                f"Artifact with id '{artifact_id}' not found.",
                type="not_found",
                code="artifact_not_found",
                param="artifact_id",
            )
        part = await record.service.load_artifact(
            app_name=record.app_name,
            user_id=record.user_id,
            session_id=record.session_id,
            filename=record.filename,
            version=record.version,
        )
        blob = part.inline_data if part is not None else None
        if blob is None or blob.data is None:
            raise ApiError(
                f"Artifact with id '{artifact_id}' not found.",
                type="not_found",
                code="artifact_not_found",
                param="artifact_id",
            )
        mime_type = blob.mime_type or record.mime_type
        if not re.fullmatch(r"[\w.+-]+/[\w.+-]+", mime_type or ""):
            mime_type = "application/octet-stream"
        return HttpResponse(
            content=blob.data,
            media_type=mime_type,
            headers={
                "Content-Disposition": _content_disposition(record.filename),
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
            },
        )

    @app.delete("/v1/artifacts/{artifact_id}")
    async def delete_artifact(
        artifact_id: str, request: Request, delete_content: bool = False
    ):
        """Revoke a generated artifact download ID.

        By default only public access is removed: generated artifacts are
        part of the ADK session context (later agent turns may load them),
        so revoking a download link must not destroy the underlying bytes.
        With ``?delete_content=true`` the provider content is also deleted,
        best-effort, and only when no other live registry record references
        the same provider filename, because provider deletion is
        filename-wide and would otherwise remove unrelated versions. Public
        access is removed before provider cleanup is attempted, so a
        provider failure can never restore access to a revoked ID. When the
        adapter exposes ``revoke_artifact``, a failed cleanup stays
        retryable by repeating the request with ``?delete_content=true``.
        """
        await _authorize(request)
        registry = app.state.artifact_registry
        revoker = getattr(adapter, "revoke_artifact", None)
        if revoker is not None:
            try:
                revoked = await revoker(artifact_id, delete_content=delete_content)
            except Exception:
                # revoke_artifact removes public access before provider
                # cleanup, so a cleanup failure still counts as revoked and
                # remains retryable through the adapter's pending cleanups.
                revoked = True
                logger.warning(
                    "Provider cleanup failed for revoked artifact '%s'.",
                    artifact_id,
                    exc_info=True,
                )
        else:
            record = registry.revoke(artifact_id) if registry is not None else None
            revoked = record is not None
            if revoked and delete_content and not registry.has_live_reference(record):
                try:
                    await record.service.delete_artifact(
                        app_name=record.app_name,
                        user_id=record.user_id,
                        session_id=record.session_id,
                        filename=record.filename,
                    )
                except Exception:
                    logger.warning(
                        "Provider cleanup failed for revoked artifact '%s'.",
                        artifact_id,
                        exc_info=True,
                    )
        if not revoked:
            raise ApiError(
                f"Artifact with id '{artifact_id}' not found.",
                type="not_found",
                code="artifact_not_found",
                param="artifact_id",
            )
        return JSONResponse(
            content={"id": artifact_id, "object": "artifact", "deleted": True}
        )

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel_response(response_id: str, request: Request):
        await _authorize(request)
        stored = await response_store.get(response_id)
        if stored is None:
            raise ApiError(
                f"Response with id '{response_id}' not found.",
                type="not_found",
                code="response_not_found",
                param="response_id",
            )
        if not stored.response.background:
            raise ApiError(
                "Only background responses can be cancelled.",
                code="invalid_value",
                param="response_id",
            )
        bg: _BackgroundRun | None = app.state.background_runs.get(response_id)
        if (
            bg is not None
            and bg.task is not None
            and not bg.task.done()
            and stored.response.status in ("queued", "in_progress")
        ):
            bg.task.cancel()
            try:
                await bg.task
            except asyncio.CancelledError:
                pass
            stored = await response_store.get(response_id) or stored
        return JSONResponse(content=stored.response.model_dump())

    @app.get("/v1/responses/{response_id}/events")
    async def stream_response_events(
        response_id: str, request: Request, starting_after: int = -1
    ):
        """Resume the event stream of a background response from a cursor
        (``starting_after`` is the last ``sequence_number`` received)."""
        await _authorize(request)
        bg: _BackgroundRun | None = app.state.background_runs.get(response_id)
        if bg is None:
            raise ApiError(
                f"No streamable background response with id '{response_id}'.",
                type="not_found",
                code="response_not_found",
                param="response_id",
            )
        return StreamingResponse(
            bg.stream(starting_after=starting_after),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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

"""HTTP server exposing an :class:`AgentAdapter` as an Open Responses API.

Implements the core Open Responses surface:

- ``POST /v1/responses`` — create a response (``application/json`` or,
  with ``"stream": true``, ``text/event-stream`` semantic events terminated
  by ``data: [DONE]``).
- ``GET /v1/responses/{response_id}`` — retrieve a stored response.
- ``DELETE /v1/responses/{response_id}`` — delete a stored response.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .adapter import AgentAdapter, AgentRun
from .engine import ResponseEngine, collect_response
from .models import (
    ERROR_STATUS_CODES,
    ErrorBody,
    ErrorEnvelope,
    Item,
    Response,
    ResponsesRequest,
)
from .store import InMemoryResponseStore, ResponseStore


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
) -> FastAPI:
    """Build a FastAPI app serving ``adapter`` over the Open Responses API.

    Args:
        adapter: The agent adapter to expose.
        store: Response store used for ``previous_response_id`` continuation.
            Defaults to an in-memory store.
        api_key: If set, requests must carry ``Authorization: Bearer <api_key>``.
        title: OpenAPI title for the app.
    """
    response_store = store or InMemoryResponseStore()
    engine = ResponseEngine(adapter, response_store)
    app = FastAPI(title=title or f"Open Responses ({adapter.name})")
    app.state.adapter = adapter
    app.state.response_store = response_store

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

    async def _build_run(payload: ResponsesRequest) -> AgentRun:
        new_items: list[Item] = payload.input_items()
        context_items: list[Item] = list(new_items)
        previous_state = None
        if payload.previous_response_id:
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

    @app.post("/v1/responses")
    async def create_response(payload: ResponsesRequest, request: Request):
        await _authorize(request)
        if payload.background:
            raise ApiError(
                "Background responses are not supported by this server.",
                code="unsupported_parameter",
                param="background",
            )
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

    return app


async def _sse(events: AsyncIterator) -> AsyncIterator[str]:
    """Frame streaming events as Server-Sent Events."""
    async for event in events:
        data = json.dumps(event.model_dump(), separators=(",", ":"))
        yield f"event: {event.type}\ndata: {data}\n\n"
    yield "data: [DONE]\n\n"

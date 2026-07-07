"""Turns adapter events into Open Responses semantic events.

The engine owns everything protocol-related: response lifecycle
(``response.created`` → ``response.in_progress`` → terminal event), output
item and content part lifecycles, sequence numbers, usage aggregation, and
persistence into the response store.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .adapter import (
    AdapterError,
    AgentAdapter,
    AgentRun,
    ItemAdded,
    ItemDone,
    ReasoningDelta,
    StateUpdate,
    TextDelta,
    UsageDelta,
)
from .models import (
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    ErrorEvent,
    ErrorPayload,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    FunctionCallItem,
    MessageItem,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputText,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ReasoningItem,
    ReasoningSummaryPartAddedEvent,
    ReasoningSummaryPartDoneEvent,
    ReasoningSummaryTextDeltaEvent,
    ReasoningSummaryTextDoneEvent,
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseError,
    ResponseFailedEvent,
    ResponseInProgressEvent,
    StreamEvent,
    SummaryText,
    Usage,
    new_message_id,
    new_reasoning_id,
)
from .store import ResponseStore, StoredResponse

logger = logging.getLogger(__name__)


class ResponseEngine:
    def __init__(self, adapter: AgentAdapter, store: ResponseStore) -> None:
        self.adapter = adapter
        self.store = store

    async def events(
        self,
        run: AgentRun,
        *,
        on_stored: Callable[[StoredResponse], Awaitable[None]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute one turn, yielding Open Responses streaming events.

        ``on_stored`` is invoked with the completed :class:`StoredResponse`
        regardless of the request's ``store`` flag, letting transports keep
        connection-local continuation state (e.g. WebSocket ``store=false``).
        """
        state = _TurnState(self.adapter, run)

        yield state.stamp(ResponseCreatedEvent(response=state.snapshot()))
        yield state.stamp(ResponseInProgressEvent(response=state.snapshot()))

        try:
            async for adapter_event in self.adapter.run(run):
                match adapter_event:
                    case TextDelta(delta=delta):
                        if delta:
                            for ev in state.on_text_delta(delta):
                                yield ev
                    case ReasoningDelta() as reasoning:
                        for ev in state.on_reasoning_delta(reasoning):
                            yield ev
                    case ItemAdded(item=item):
                        for ev in state.on_item_added(item):
                            yield ev
                    case ItemDone(item=item):
                        for ev in state.on_item_done(item):
                            yield ev
                    case UsageDelta() as usage:
                        state.on_usage(usage)
                    case StateUpdate(state=adapter_state):
                        state.adapter_state.update(adapter_state)
        except AdapterError as exc:
            async for ev in self._fail(
                state, exc.message, error_type=exc.type, code=exc.code, param=exc.param
            ):
                yield ev
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("adapter %r raised", self.adapter.name)
            async for ev in self._fail(
                state, f"The provider encountered an error: {exc}"
            ):
                yield ev
            return

        for ev in state.close_open_items():
            yield ev

        response = state.snapshot(status="completed")
        stored = await self._persist(state, response)
        if on_stored is not None:
            await on_stored(stored)
        yield state.stamp(ResponseCompletedEvent(response=response))

    async def _fail(
        self,
        state: _TurnState,
        message: str,
        *,
        error_type: str = "server_error",
        code: str | None = None,
        param: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        yield state.stamp(
            ErrorEvent(
                error=ErrorPayload(
                    type=error_type, code=code, message=message, param=param
                )
            )
        )
        response = state.snapshot(status="failed")
        response.error = ResponseError.model_validate(
            {
                "code": code or error_type,
                "message": message,
                "type": error_type,
                "param": param,
            }
        )
        await self._persist(state, response)
        yield state.stamp(ResponseFailedEvent(response=response))

    async def _persist(self, state: _TurnState, response: Response) -> StoredResponse:
        stored = StoredResponse(
            response=response,
            input_items=state.run.context_items,
            adapter_state=state.adapter_state,
        )
        if state.run.request.store is not False:
            await self.store.put(stored)
        return stored


async def collect_response(events: AsyncIterator[StreamEvent]) -> Response:
    """Drain an event stream and return the terminal response object."""
    final: Response | None = None
    async for event in events:
        response = getattr(event, "response", None)
        if response is not None:
            final = response
    if final is None:  # pragma: no cover - engine always yields a terminal event
        raise RuntimeError("event stream ended without a terminal response event")
    return final


class _TurnState:
    """Mutable state for one response turn."""

    def __init__(self, adapter: AgentAdapter, run: AgentRun) -> None:
        self.run = run
        self.sequence = 0
        self.output: list[Any] = []
        self.adapter_state: dict[str, Any] = {}
        self.usage = Usage()
        self.has_usage = False

        request = run.request
        self.response_template = Response(
            model=request.model or adapter.default_model or adapter.name,
            instructions=request.instructions,
            previous_response_id=request.previous_response_id,
            store=request.store if request.store is not None else True,
            tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=(
                request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True
            ),
            temperature=request.temperature if request.temperature is not None else 1.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            max_output_tokens=request.max_output_tokens,
            truncation=request.truncation or "disabled",
            metadata=request.metadata or {},
            service_tier=request.service_tier or "default",
        )

        # Open assistant message being streamed, if any.
        self._message: MessageItem | None = None
        self._message_index = -1
        self._message_text = ""
        # Open reasoning item being streamed, if any.
        self._reasoning: ReasoningItem | None = None
        self._reasoning_index = -1
        self._reasoning_text = ""
        # Non-message items that were added but not yet done, keyed by id.
        self._open_items: dict[str, int] = {}

    # -- events -------------------------------------------------------------

    def stamp(self, event: StreamEvent) -> StreamEvent:
        event.sequence_number = self.sequence
        self.sequence += 1
        return event

    def snapshot(self, status: str = "in_progress") -> Response:
        response = self.response_template.model_copy(deep=True)
        response.status = status  # type: ignore[assignment]
        if status in ("completed", "failed", "incomplete", "cancelled"):
            response.completed_at = int(time.time())
        response.output = list(self.output)
        if self._reasoning is not None:
            response.output.append(self._reasoning.model_copy(deep=True))
        if self._message is not None:
            response.output.append(self._message.model_copy(deep=True))
        response.usage = self.usage if self.has_usage else None
        return response

    def on_text_delta(self, delta: str) -> list[StreamEvent]:
        events: list[StreamEvent] = self.close_reasoning()
        if self._message is None:
            self._message = MessageItem(
                id=new_message_id(), role="assistant", status="in_progress", content=[]
            )
            self._message_index = len(self.output)
            self._message_text = ""
            events.append(
                self.stamp(
                    OutputItemAddedEvent(
                        output_index=self._message_index,
                        item=self._message.model_copy(deep=True),
                    )
                )
            )
            events.append(
                self.stamp(
                    ContentPartAddedEvent(
                        item_id=self._message.id or "",
                        output_index=self._message_index,
                        content_index=0,
                        part=OutputText(text=""),
                    )
                )
            )
        self._message_text += delta
        events.append(
            self.stamp(
                OutputTextDeltaEvent(
                    item_id=self._message.id or "",
                    output_index=self._message_index,
                    content_index=0,
                    delta=delta,
                )
            )
        )
        return events

    def close_message(self) -> list[StreamEvent]:
        if self._message is None:
            return []
        message = self._message
        index = self._message_index
        text = self._message_text
        self._message = None
        self._message_text = ""

        message.status = "completed"
        message.content = [OutputText(text=text)]
        # While a message is open no other item can be appended, so the
        # reserved index is always the tail of the output list.
        self.output.append(message)

        item_id = message.id or ""
        return [
            self.stamp(
                OutputTextDoneEvent(
                    item_id=item_id, output_index=index, content_index=0, text=text
                )
            ),
            self.stamp(
                ContentPartDoneEvent(
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    part=OutputText(text=text),
                )
            ),
            self.stamp(
                OutputItemDoneEvent(
                    output_index=index, item=message.model_copy(deep=True)
                )
            ),
        ]

    def on_reasoning_delta(self, delta: ReasoningDelta) -> list[StreamEvent]:
        events: list[StreamEvent] = self.close_message()
        if self._reasoning is None:
            self._reasoning = ReasoningItem(
                id=new_reasoning_id(), status="in_progress", summary=[]
            )
            self._reasoning_index = len(self.output)
            self._reasoning_text = ""
            events.append(
                self.stamp(
                    OutputItemAddedEvent(
                        output_index=self._reasoning_index,
                        item=self._reasoning.model_copy(deep=True),
                    )
                )
            )
            events.append(
                self.stamp(
                    ReasoningSummaryPartAddedEvent(
                        item_id=self._reasoning.id or "",
                        output_index=self._reasoning_index,
                        summary_index=0,
                        part=SummaryText(text=""),
                    )
                )
            )
        if delta.encrypted_content is not None:
            self._reasoning.encrypted_content = delta.encrypted_content
        if delta.delta:
            self._reasoning_text += delta.delta
            events.append(
                self.stamp(
                    ReasoningSummaryTextDeltaEvent(
                        item_id=self._reasoning.id or "",
                        output_index=self._reasoning_index,
                        summary_index=0,
                        delta=delta.delta,
                    )
                )
            )
        return events

    def close_reasoning(self) -> list[StreamEvent]:
        if self._reasoning is None:
            return []
        reasoning = self._reasoning
        index = self._reasoning_index
        text = self._reasoning_text
        self._reasoning = None
        self._reasoning_text = ""

        reasoning.status = "completed"
        reasoning.summary = [SummaryText(text=text)] if text else []
        # While a reasoning item is open no other item can be appended, so
        # the reserved index is always the tail of the output list.
        self.output.append(reasoning)

        item_id = reasoning.id or ""
        events: list[StreamEvent] = []
        if text:
            events.append(
                self.stamp(
                    ReasoningSummaryTextDoneEvent(
                        item_id=item_id,
                        output_index=index,
                        summary_index=0,
                        text=text,
                    )
                )
            )
            events.append(
                self.stamp(
                    ReasoningSummaryPartDoneEvent(
                        item_id=item_id,
                        output_index=index,
                        summary_index=0,
                        part=SummaryText(text=text),
                    )
                )
            )
        events.append(
            self.stamp(
                OutputItemDoneEvent(
                    output_index=index, item=reasoning.model_copy(deep=True)
                )
            )
        )
        return events

    def close_open_items(self) -> list[StreamEvent]:
        """Close whichever streamed item (reasoning or message) is open."""
        return [*self.close_reasoning(), *self.close_message()]

    def on_item_added(self, item: Any) -> list[StreamEvent]:
        events = self.close_open_items()
        index = len(self.output)
        self.output.append(item)
        if getattr(item, "id", None):
            self._open_items[item.id] = index
        events.append(
            self.stamp(OutputItemAddedEvent(output_index=index, item=item))
        )
        return events

    def on_item_done(self, item: Any) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        item_id = getattr(item, "id", None)
        if item_id and item_id in self._open_items:
            index = self._open_items.pop(item_id)
            self.output[index] = item
        else:
            events.extend(self.close_open_items())
            index = len(self.output)
            self.output.append(item)
            events.append(
                self.stamp(OutputItemAddedEvent(output_index=index, item=item))
            )
        if isinstance(item, FunctionCallItem) and item.arguments:
            events.append(
                self.stamp(
                    FunctionCallArgumentsDeltaEvent(
                        item_id=item.id or "",
                        output_index=index,
                        delta=item.arguments,
                    )
                )
            )
            events.append(
                self.stamp(
                    FunctionCallArgumentsDoneEvent(
                        item_id=item.id or "",
                        output_index=index,
                        arguments=item.arguments,
                    )
                )
            )
        events.append(self.stamp(OutputItemDoneEvent(output_index=index, item=item)))
        return events

    def on_usage(self, usage: UsageDelta) -> None:
        self.has_usage = True
        self.usage.input_tokens += usage.input_tokens
        self.usage.output_tokens += usage.output_tokens
        self.usage.total_tokens += usage.total_tokens
        self.usage.input_tokens_details.cached_tokens += usage.cached_tokens
        self.usage.output_tokens_details.reasoning_tokens += usage.reasoning_tokens

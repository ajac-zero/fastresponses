"""Framework-agnostic agent adapter protocol.

An :class:`AgentAdapter` bridges an agent framework (ADK, LangGraph, ...) to
the Open Responses protocol. Adapters receive an :class:`AgentRun` describing
one turn and yield a stream of simple :data:`AdapterEvent` objects. The
engine (:mod:`open_responses_server.engine`) is responsible for turning those
into spec-compliant Open Responses semantic events and the final response
object, so adapters never have to deal with sequence numbers, content-part
lifecycles, or SSE framing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Union

from .models import Item, ResponsesRequest


@dataclass
class AgentRun:
    """A single Open Responses turn handed to an adapter."""

    request: ResponsesRequest
    """The raw (validated) request."""

    new_items: list[Item]
    """The new input items from this request only."""

    context_items: list[Item]
    """The full logical context: ``previous.input + previous.output + new``.

    Stateless adapters should use this; stateful adapters (e.g. ADK, which
    keeps its own session history) can rely on ``previous_state`` instead.
    """

    previous_state: dict[str, Any] | None = None
    """Opaque adapter state stored with the previous response, if any."""


# ---------------------------------------------------------------------------
# Adapter events
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    """A chunk of assistant output text for the current message."""

    delta: str


@dataclass
class ItemAdded:
    """A non-message output item has started (e.g. a tool call receipt).

    The engine emits ``response.output_item.added`` and keeps the item open
    until a matching :class:`ItemDone` (same ``item.id``) arrives.
    """

    item: Item


@dataclass
class ItemDone:
    """A non-message output item is complete.

    If an :class:`ItemAdded` with the same ``item.id`` was emitted earlier,
    this replaces/completes it; otherwise it is treated as an atomic item
    (added + done).
    """

    item: Item


@dataclass
class UsageDelta:
    """Token accounting; values are summed across events."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class StateUpdate:
    """Opaque adapter state persisted alongside the response.

    It is handed back as ``AgentRun.previous_state`` when a client continues
    the conversation with ``previous_response_id``.
    """

    state: dict[str, Any] = field(default_factory=dict)


AdapterEvent = Union[TextDelta, ItemAdded, ItemDone, UsageDelta, StateUpdate]


class AdapterError(Exception):
    """Raised by adapters to fail the response with a structured error."""

    def __init__(
        self,
        message: str,
        *,
        type: str = "server_error",
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.type = type
        self.code = code
        self.param = param


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


class AgentAdapter(ABC):
    """Wraps an agent framework as an Open Responses provider."""

    name: str = "agent"
    """Implementor slug, used to prefix extension item types."""

    default_model: str | None = None
    """Model name reported in responses when the request omits ``model``."""

    @abstractmethod
    def run(self, run: AgentRun) -> AsyncIterator[AdapterEvent]:
        """Execute one turn and yield adapter events.

        Implementations must be async generators (or return an async
        iterator). Raise :class:`AdapterError` to fail the response.
        """
        raise NotImplementedError

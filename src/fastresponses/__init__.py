"""fastresponses: serve agent frameworks over the Open Responses API.

Wraps agent frameworks (Google ADK first) as providers compatible with the
Open Responses specification (https://www.openresponses.org), so any Open
Responses / OpenAI Responses client can talk to your agents.
"""

from .adapter import (
    AdapterError,
    AdapterEvent,
    AgentAdapter,
    AgentRun,
    Incomplete,
    ItemAdded,
    ItemDone,
    ReasoningDelta,
    StateUpdate,
    TextDelta,
    UsageDelta,
)
from .cli import main
from .engine import ResponseEngine, collect_response
from .server import create_app
from .store import (
    InMemoryResponseStore,
    ResponseStore,
    SQLiteResponseStore,
    StoredResponse,
)

__all__ = [
    "AdapterError",
    "AdapterEvent",
    "AgentAdapter",
    "AgentRun",
    "InMemoryResponseStore",
    "Incomplete",
    "ItemAdded",
    "ItemDone",
    "ReasoningDelta",
    "ResponseEngine",
    "ResponseStore",
    "SQLiteResponseStore",
    "StateUpdate",
    "StoredResponse",
    "TextDelta",
    "UsageDelta",
    "collect_response",
    "create_app",
    "main",
]

"""open-responses-server: serve agent frameworks over the Open Responses API.

Wraps agent frameworks (Google ADK first) as providers compatible with the
Open Responses specification (https://www.openresponses.org), so any Open
Responses / OpenAI Responses client can talk to your agents.
"""

from .adapter import (
    AdapterError,
    AdapterEvent,
    AgentAdapter,
    AgentRun,
    ItemAdded,
    ItemDone,
    StateUpdate,
    TextDelta,
    UsageDelta,
)
from .cli import main
from .engine import ResponseEngine, collect_response
from .server import create_app
from .store import InMemoryResponseStore, ResponseStore, StoredResponse

__all__ = [
    "AdapterError",
    "AdapterEvent",
    "AgentAdapter",
    "AgentRun",
    "InMemoryResponseStore",
    "ItemAdded",
    "ItemDone",
    "ResponseEngine",
    "ResponseStore",
    "StateUpdate",
    "StoredResponse",
    "TextDelta",
    "UsageDelta",
    "collect_response",
    "create_app",
    "main",
]

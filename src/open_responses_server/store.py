"""Response storage for ``previous_response_id`` continuation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .models import Item, Response


@dataclass
class StoredResponse:
    """Everything needed to continue a conversation from a response."""

    response: Response
    input_items: list[Item] = field(default_factory=list)
    adapter_state: dict[str, Any] = field(default_factory=dict)

    def context_items(self) -> list[Item]:
        """``input + output`` of this response, in semantic order."""
        return [*self.input_items, *self.response.output]


class ResponseStore(ABC):
    @abstractmethod
    async def get(self, response_id: str) -> StoredResponse | None: ...

    @abstractmethod
    async def put(self, stored: StoredResponse) -> None: ...

    @abstractmethod
    async def delete(self, response_id: str) -> bool: ...


class InMemoryResponseStore(ResponseStore):
    """Simple LRU-bounded in-memory store."""

    def __init__(self, max_responses: int = 1000) -> None:
        self._max = max_responses
        self._responses: OrderedDict[str, StoredResponse] = OrderedDict()

    async def get(self, response_id: str) -> StoredResponse | None:
        stored = self._responses.get(response_id)
        if stored is not None:
            self._responses.move_to_end(response_id)
        return stored

    async def put(self, stored: StoredResponse) -> None:
        self._responses[stored.response.id] = stored
        self._responses.move_to_end(stored.response.id)
        while len(self._responses) > self._max:
            self._responses.popitem(last=False)

    async def delete(self, response_id: str) -> bool:
        return self._responses.pop(response_id, None) is not None

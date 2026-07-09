"""Response storage for ``previous_response_id`` continuation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

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


_ITEMS_ADAPTER: TypeAdapter[list[Item]] = TypeAdapter(list[Item])


class SQLiteResponseStore(ResponseStore):
    """Durable single-file store; safe across restarts and multiple workers
    on one host.

    Uses the stdlib ``sqlite3`` module (WAL mode) with blocking calls moved
    off the event loop via ``asyncio.to_thread``. Set ``max_responses`` to
    prune the oldest responses on insert; ``None`` keeps everything.
    """

    def __init__(
        self, path: str | Path, *, max_responses: int | None = None
    ) -> None:
        self._path = str(path)
        self._max = max_responses
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    stored_at REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS responses_stored_at"
                " ON responses (stored_at)"
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- serialization ---------------------------------------------------

    @staticmethod
    def _dumps(stored: StoredResponse) -> str:
        return json.dumps(
            {
                "response": stored.response.model_dump(),
                "input_items": [i.model_dump() for i in stored.input_items],
                "adapter_state": stored.adapter_state,
            },
            separators=(",", ":"),
        )

    @staticmethod
    def _loads(payload: str) -> StoredResponse:
        data = json.loads(payload)
        return StoredResponse(
            response=Response.model_validate(data["response"]),
            input_items=_ITEMS_ADAPTER.validate_python(data["input_items"]),
            adapter_state=data.get("adapter_state") or {},
        )

    # -- blocking primitives ---------------------------------------------

    def _get_sync(self, response_id: str) -> StoredResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM responses WHERE id = ?", (response_id,)
            ).fetchone()
        return self._loads(row[0]) if row else None

    def _put_sync(self, stored: StoredResponse) -> None:
        payload = self._dumps(stored)
        with self._lock:
            self._conn.execute(
                "INSERT INTO responses (id, stored_at, payload) VALUES (?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET stored_at = excluded.stored_at,"
                " payload = excluded.payload",
                (stored.response.id, time.time(), payload),
            )
            if self._max is not None:
                self._conn.execute(
                    "DELETE FROM responses WHERE id IN ("
                    " SELECT id FROM responses ORDER BY stored_at DESC"
                    " LIMIT -1 OFFSET ?)",
                    (self._max,),
                )
            self._conn.commit()

    def _delete_sync(self, response_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM responses WHERE id = ?", (response_id,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    # -- ResponseStore ----------------------------------------------------

    async def get(self, response_id: str) -> StoredResponse | None:
        return await asyncio.to_thread(self._get_sync, response_id)

    async def put(self, stored: StoredResponse) -> None:
        await asyncio.to_thread(self._put_sync, stored)

    async def delete(self, response_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, response_id)

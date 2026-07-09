"""Context-window compaction (``POST /v1/responses/compact``).

The compacted payload is opaque to clients per the spec; this implementation
encodes the full transcript, so using the compacted window as the base
``input`` of a new response chain is lossless.
"""

from __future__ import annotations

import base64
import json

from pydantic import TypeAdapter

from .models import CompactionItem, Item, new_compaction_id

_ITEM_LIST_ADAPTER: TypeAdapter[list[Item]] = TypeAdapter(list[Item])


def compact_items(items: list[Item]) -> CompactionItem:
    """Compact a list of items into a single, round-trippable item."""
    payload = json.dumps(
        [item.model_dump(exclude_none=True) for item in items],
        separators=(",", ":"),
    )
    return CompactionItem(
        id=new_compaction_id(),
        status="completed",
        encrypted_content=base64.b64encode(payload.encode("utf-8")).decode("ascii"),
        created_by="fastresponses",
    )


def expand_compaction_item(item: CompactionItem) -> list[Item]:
    """Decode a compaction item produced by :func:`compact_items`.

    Returns an empty list for payloads not produced by this server.
    """
    try:
        payload = base64.b64decode(item.encrypted_content)
        return _ITEM_LIST_ADAPTER.validate_json(payload)
    except Exception:
        return []

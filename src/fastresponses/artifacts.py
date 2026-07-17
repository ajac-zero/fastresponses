"""Opaque process-local registry for downloadable provider artifacts."""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactRecord:
    service: Any
    app_name: str
    user_id: str
    session_id: str | None
    filename: str
    version: int
    mime_type: str


class ArtifactRegistry:
    """Maps opaque public IDs to exact provider artifact scopes."""

    def __init__(self, *, max_records: int = 1024, ttl_seconds: float = 3600) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be a positive integer.")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive number.")
        self.max_records = max_records
        self.ttl_seconds = ttl_seconds
        self._records: OrderedDict[str, tuple[float, ArtifactRecord]] = OrderedDict()

    def register(self, record: ArtifactRecord) -> str:
        self._purge_expired()
        artifact_id = f"artifact_{secrets.token_urlsafe(24)}"
        self._records[artifact_id] = (time.monotonic() + self.ttl_seconds, record)
        while len(self._records) > self.max_records:
            self._records.popitem(last=False)
        return artifact_id

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        entry = self._records.get(artifact_id)
        if entry is None:
            return None
        expires_at, record = entry
        if expires_at <= time.monotonic():
            del self._records[artifact_id]
            return None
        self._records.move_to_end(artifact_id)
        return record

    def _purge_expired(self) -> None:
        now = time.monotonic()
        for artifact_id, (expires_at, _) in list(self._records.items()):
            if expires_at > now:
                continue
            del self._records[artifact_id]

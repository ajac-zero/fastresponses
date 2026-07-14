"""Opaque process-local registry for downloadable provider artifacts."""

from __future__ import annotations

import secrets
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

    def __init__(self) -> None:
        self._records: dict[str, ArtifactRecord] = {}

    def register(self, record: ArtifactRecord) -> str:
        artifact_id = f"artifact_{secrets.token_urlsafe(24)}"
        self._records[artifact_id] = record
        return artifact_id

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        return self._records.get(artifact_id)

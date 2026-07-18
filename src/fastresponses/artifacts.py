"""Opaque process-local registry for downloadable provider artifacts."""

from __future__ import annotations

import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

#: ``type`` of the ``CustomItem`` used to surface downloadable generated
#: artifacts. This item type is owned and versioned by
#: ajac-zero/openresponses-extensions (schemas/artifact.json), not by this
#: repo; ``ARTIFACT_TYPE`` is shared here between the ADK adapter (which
#: creates the item) and the server (which reports live download
#: availability on retrieval).
ARTIFACT_TYPE = "ajac-zero:artifact"

#: Matches ``content_url`` values of the exact form this implementation
#: emits, ``/v1/artifacts/{artifact_id}/content``, requiring a non-empty,
#: slash-free ``artifact_id`` segment. The shared ajac-zero:artifact spec
#: also permits an absolute URL; this implementation just never emits one.
_CONTENT_URL_PATTERN = re.compile(r"/v1/artifacts/[^/]+/content")


class ArtifactItem(BaseModel):
    """Typed schema for this implementation's ``ajac-zero:artifact`` output.

    ``ajac-zero:artifact`` is a shared extension item type owned and
    versioned by `ajac-zero/openresponses-extensions
    <https://github.com/ajac-zero/openresponses-extensions>`_ (see its
    ``schemas/artifact.json`` for the authoritative, cross-implementation
    JSON Schema contract). This model does **not** replace that contract —
    it validates the exact, narrower shape *this* implementation (the
    Google ADK adapter) actually emits, which in places is stricter than
    what the shared spec permits (e.g. ``content_url`` here is always a
    relative path, though the shared spec also permits absolute URLs).

    The Google ADK adapter surfaces generated artifacts through two
    distinct code paths, both producing this same item shape:

    - **Session-generated**: the agent's own tool code calls ADK's
      ``tool_context.save_artifact(...)`` directly, and the adapter picks
      it up from the turn's ``artifact_delta``. These items never carry
      ``call_id`` — ADK does not link the resulting delta back to a
      specific tool call.
    - **Mapper-created**: an ``internal_tool_response_mapper`` calls
      ``ADKToolResponse.create_artifact(...)``. These items always carry
      ``call_id``, tying the artifact back to the internal
      ``function_call`` / ``function_call_output`` pair that produced it.

    See the "Artifact item schema" section of ``README.md`` for the full
    field-by-field contract (including backward-compatibility guarantees).
    This model exists so consumers can parse and validate artifact items
    without reading adapter source code; it is intentionally permissive
    about unknown fields (``extra="allow"``) so that new, additive fields
    introduced in a future release still round-trip through it instead of
    raising.

    Stability summary:

    - ``type``, ``id``, ``status``, ``filename``, ``mime_type``,
      ``content_url``, and ``size`` are always present.
    - ``available`` and ``expires_at`` are always present on items produced
      by the current adapter, but default to ``True`` / ``None`` here so
      that older items predating these fields (e.g. replayed from
      ``GET /v1/responses/{id}/events``, or read back from a response
      store populated by an earlier release) still parse instead of
      raising — consistent with treating an absent ``available`` as
      best-effort-true, per the next point.
    - ``call_id`` is present only for mapper-created artifacts; its
      absence is meaningful, not missing data.
    - ``available: False`` is authoritative (never attempt the download).
      ``available: True`` or absent is best-effort only, never a
      guarantee — always attempt the download and handle failure.
    - ``expires_at`` is advisory (a predicted Unix-seconds deadline) when
      present, not an exact guarantee; absent means unknown, not "never
      expires."
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["ajac-zero:artifact"] = ARTIFACT_TYPE
    id: str
    status: Literal["completed"] = "completed"
    filename: str
    mime_type: str
    size: int
    content_url: str
    available: bool = True
    expires_at: int | None = None
    call_id: str | None = None

    @field_validator("content_url")
    @classmethod
    def _content_url_matches_this_implementations_download_path(
        cls, value: str
    ) -> str:
        """Validate against this implementation's own emitted shape.

        The shared ``ajac-zero:artifact`` contract (see class docstring)
        permits an absolute URL too; this implementation just never emits
        one, so this check is intentionally narrower than the shared spec.
        """
        if not _CONTENT_URL_PATTERN.fullmatch(value):
            raise ValueError(
                "content_url must be a relative path of the form "
                "'/v1/artifacts/{artifact_id}/content', with a non-empty id "
                "(this implementation never emits an absolute content_url, "
                "though the shared ajac-zero:artifact spec permits one)."
            )
        return value


def parse_artifact_item(item: Any) -> ArtifactItem:
    """Parse a response output item as a typed, validated ``ArtifactItem``.

    Accepts a pydantic model instance (e.g. the ``CustomItem`` fastresponses
    itself emits) or a plain ``dict`` (e.g. read back from stored JSON).
    Raises ``pydantic.ValidationError`` if ``item`` is not a well-formed
    ``ajac-zero:artifact`` item, including when its ``type`` does not
    match.
    """
    payload = item.model_dump(mode="json") if isinstance(item, BaseModel) else item
    return ArtifactItem.model_validate(payload)


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
        if not max_records > 0:
            raise ValueError("max_records must be a positive integer.")
        if not ttl_seconds > 0:
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

    def refresh(self, artifact_id: str) -> ArtifactRecord | None:
        """Extend a live record's expiry to a fresh full TTL from now.

        Retrieving a stored response is treated as a live access: as long as
        a client keeps fetching it, its artifact download links keep
        working (sliding expiration) instead of dying on a fixed clock from
        creation. Returns ``None`` for unknown, already-expired, or revoked
        IDs, self-healing expired entries the same way :meth:`get` does, so
        callers cannot distinguish those cases.
        """
        entry = self._records.get(artifact_id)
        if entry is None:
            return None
        expires_at, record = entry
        if expires_at <= time.monotonic():
            del self._records[artifact_id]
            return None
        self._records[artifact_id] = (time.monotonic() + self.ttl_seconds, record)
        self._records.move_to_end(artifact_id)
        return record

    def revoke(self, artifact_id: str) -> ArtifactRecord | None:
        """Remove public access to ``artifact_id`` immediately.

        Returns the revoked record, or ``None`` when the ID is unknown or
        already expired so callers cannot distinguish the two cases.
        """
        entry = self._records.pop(artifact_id, None)
        if entry is None:
            return None
        expires_at, record = entry
        if expires_at <= time.monotonic():
            return None
        return record

    def has_live_reference(self, record: ArtifactRecord) -> bool:
        """Whether any live record targets the same provider artifact filename.

        Provider artifact deletion is filename-wide (it removes every stored
        version), so callers must not delete provider content while another
        registered record still references the same scope and filename.
        """
        now = time.monotonic()
        return any(
            expires_at > now
            and other.service is record.service
            and other.app_name == record.app_name
            and other.user_id == record.user_id
            and other.session_id == record.session_id
            and other.filename == record.filename
            for expires_at, other in self._records.values()
        )

    def _purge_expired(self) -> None:
        now = time.monotonic()
        for artifact_id, (expires_at, _) in list(self._records.items()):
            if expires_at > now:
                continue
            del self._records[artifact_id]

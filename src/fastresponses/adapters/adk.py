"""Google ADK adapter.

Wraps a :class:`google.adk.agents.LlmAgent` (or any ``BaseAgent``) as an
Open Responses provider:

- Assistant text is streamed as ``output_text`` deltas (ADK ``StreamingMode.SSE``).
- Model "thought" parts (e.g. Gemini thought summaries) are surfaced as
  ``reasoning`` output items with streamed summary text; thought signatures
  are attached as ``encrypted_content`` when available.
- Tools owned by the ADK agent run *inside* the provider; each execution is
  surfaced as a standard ``function_call`` / ``function_call_output`` pair.
- Function tools declared by the *client* in ``request.tools`` are exposed to
  the agent as long-running ADK tools: when the model calls one, the run
  yields control back and the server emits a standard ``function_call``
  output item. The client then answers with a ``function_call_output`` item
  (optionally via ``previous_response_id``) and the conversation resumes.
- ``previous_response_id`` continuation maps to a persistent ADK session, so
  history is not re-sent to the model. Requests without it are replayed
  statelessly into a fresh session from the request ``input``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import uuid
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.artifacts import BaseArtifactService, InMemoryArtifactService
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, InMemorySessionService, Session
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

from ..adapter import (  # noqa: I001
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
from ..artifacts import ArtifactRecord, ArtifactRegistry
from ..compaction import expand_compaction_item
from ..models import (
    CompactionItem,
    CustomItem,
    FunctionCallItem,
    FunctionCallOutputItem,
    FunctionTool,
    InputFile,
    InputImage,
    InputText,
    Item,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    MessageItem,
    ToolChoiceAllowed,
    ToolChoiceFunction,
    new_call_id,
    new_function_call_id,
)

ARTIFACT_TYPE = "ajac-zero:artifact"


@dataclass(frozen=True)
class ADKToolResponse:
    """Stable context for deriving items from a completed internal ADK tool."""

    name: str
    arguments: dict[str, Any]
    response: Any
    output: str | None
    author: str
    call: FunctionCallItem
    output_item: FunctionCallOutputItem
    create_artifact: Callable[[str, bytes, str], Awaitable[Item]]


InternalToolResponseMapper = Callable[
    [ADKToolResponse], Iterable[Item] | Awaitable[Iterable[Item]]
]

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


_MAX_INPUT_FILE_BYTES = 32 * 1024 * 1024
_MAX_INPUT_FILES = 16
_MAX_INPUT_FILES_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_REPLAY_INPUT_FILES = 64
_MAX_REPLAY_INPUT_FILES_TOTAL_BYTES = 128 * 1024 * 1024
_MAX_INPUT_FILE_REDIRECTS = 3
_GENERIC_MIME_TYPES = {"", "application/octet-stream", "binary/octet-stream"}
_INPUT_COUNTER_KEY = "fastresponses:input_attachment_counter"
_INPUT_ARTIFACT_PREFIX = "attachment_"

InputFileAction = Literal["inline", "url", "reference", "reject"]
InputFileRouter = Callable[[InputFile], InputFileAction | Awaitable[InputFileAction]]


@dataclass(frozen=True)
class InputFileContent:
    """A downloaded or decoded input file passed to a reference store."""

    filename: str
    mime_type: str
    data: bytes


@dataclass(frozen=True)
class InputFileReferenceContext:
    """Framework context and adapter-allocated ID for an input reference."""

    app_name: str
    user_id: str
    session_id: str
    reference_id: str


class InputFileReferenceStore(Protocol):
    async def create_reference(
        self, file: InputFileContent, *, context: InputFileReferenceContext
    ) -> str:
        """Store a file and return the exact text shown to the model."""
        ...


class ADKArtifactInputFileStore:
    """Store referenced inputs in ADK's artifact service."""

    def __init__(self, artifact_service: BaseArtifactService | None = None) -> None:
        self.artifact_service = artifact_service or InMemoryArtifactService()

    async def create_reference(
        self, file: InputFileContent, *, context: InputFileReferenceContext
    ) -> str:
        existing = await self.artifact_service.load_artifact(
            app_name=context.app_name,
            user_id=context.user_id,
            session_id=context.session_id,
            filename=context.reference_id,
        )
        if existing is not None:
            blob = existing.inline_data
            if (
                blob is None
                or blob.data != file.data
                or blob.mime_type != file.mime_type
                or blob.display_name != file.filename
            ):
                raise ValueError(
                    f"Input artifact {context.reference_id!r} already contains "
                    "different content."
                )
        else:
            await self.artifact_service.save_artifact(
                app_name=context.app_name,
                user_id=context.user_id,
                session_id=context.session_id,
                filename=context.reference_id,
                artifact=types.Part(
                    inline_data=types.Blob(
                        data=file.data,
                        mime_type=file.mime_type,
                        display_name=file.filename,
                    )
                ),
            )
        reference = {
            "artifact_id": context.reference_id,
            "filename": file.filename,
            "mime_type": file.mime_type,
        }
        return "[Uploaded Artifact: " + json.dumps(
            reference, separators=(",", ":"), sort_keys=True
        ) + "]"


class _GeneratedArtifactService(BaseArtifactService):
    """Prevent agent tools from mutating adapter-owned input artifacts."""

    def __init__(self, service: BaseArtifactService) -> None:
        self.service = service

    async def save_artifact(self, **kwargs):
        _validate_generated_artifact_name(kwargs["filename"])
        return await self.service.save_artifact(**kwargs)

    async def load_artifact(self, **kwargs):
        return await self.service.load_artifact(**kwargs)

    async def list_artifact_keys(self, **kwargs):
        return await self.service.list_artifact_keys(**kwargs)

    async def delete_artifact(self, **kwargs):
        _validate_generated_artifact_name(kwargs["filename"])
        return await self.service.delete_artifact(**kwargs)

    async def list_versions(self, **kwargs):
        return await self.service.list_versions(**kwargs)

    async def list_artifact_versions(self, **kwargs):
        return await self.service.list_artifact_versions(**kwargs)

    async def get_artifact_version(self, **kwargs):
        return await self.service.get_artifact_version(**kwargs)


def _validate_generated_artifact_name(filename: str) -> None:
    if filename.startswith(_INPUT_ARTIFACT_PREFIX):
        raise ValueError("Generated artifact filename uses a reserved input namespace.")


def _validate_input_file_url(url: str, allowed_origins: frozenset[str]) -> httpx.URL:
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, ValueError) as exc:
        raise AdapterError(
            "Input file URL is invalid.",
            type="invalid_request",
            code="invalid_value",
            param="input",
        ) from exc
    if parsed.userinfo:
        raise AdapterError(
            "Input file URLs must not contain credentials.",
            type="invalid_request",
            code="invalid_value",
            param="input",
        )
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.host in {"localhost", "127.0.0.1", "::1"}
    ):
        raise AdapterError(
            "Input file URLs must use HTTPS.",
            type="invalid_request",
            code="invalid_value",
            param="input",
        )
    origin = str(parsed.copy_with(path="", query=None, fragment=None)).rstrip("/")
    if origin not in allowed_origins:
        raise AdapterError(
            "Input file URL origin is not allowed.",
            type="invalid_request",
            code="invalid_value",
            param="input",
        )
    return parsed


def _response_mime_type(response: httpx.Response, filename: str | None) -> str:
    content_type = response.headers.get("content-type", "")
    mime_type = content_type.partition(";")[0].strip().lower()
    if mime_type not in _GENERIC_MIME_TYPES and "/" in mime_type:
        return mime_type
    return _guess_mime(filename)


async def _fetch_input_file(
    url: str, filename: str | None, allowed_origins: frozenset[str]
) -> tuple[bytes, str]:
    current = _validate_input_file_url(url, allowed_origins)
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=30) as client:
            for redirects in range(_MAX_INPUT_FILE_REDIRECTS + 1):
                async with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise AdapterError(
                                "Input file redirect is missing Location.",
                                type="invalid_request",
                                code="invalid_value",
                                param="input",
                            )
                        if redirects == _MAX_INPUT_FILE_REDIRECTS:
                            raise AdapterError(
                                "Input file exceeded the redirect limit.",
                                type="invalid_request",
                                code="invalid_value",
                                param="input",
                            )
                        current = _validate_input_file_url(
                            str(current.join(location)), allowed_origins
                        )
                        continue
                    if response.status_code < 200 or response.status_code >= 300:
                        raise AdapterError(
                            f"Input file download returned HTTP {response.status_code}.",
                            type="invalid_request",
                            code="invalid_value",
                            param="input",
                        )
                    declared = response.headers.get("content-length")
                    if declared is not None and int(declared) > _MAX_INPUT_FILE_BYTES:
                        raise AdapterError(
                            "Input file exceeds the maximum size.",
                            type="invalid_request",
                            code="invalid_value",
                            param="input",
                        )
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > _MAX_INPUT_FILE_BYTES:
                            raise AdapterError(
                                "Input file exceeds the maximum size.",
                                type="invalid_request",
                                code="invalid_value",
                                param="input",
                            )
                        chunks.append(chunk)
                    if declared is not None and size != int(declared):
                        raise AdapterError(
                            "Input file response was truncated.",
                            type="invalid_request",
                            code="invalid_value",
                            param="input",
                        )
                    return b"".join(chunks), _response_mime_type(response, filename)
        raise AssertionError("unreachable")
    except AdapterError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise AdapterError(
            f"Input file download failed: {exc}",
            type="invalid_request",
            code="invalid_value",
            param="input",
        ) from exc


def _decode_data_url(url: str) -> types.Blob | None:
    """Decode a ``data:<mime>;base64,<data>`` URL into a genai Blob."""
    header, _, data = url.partition(",")
    if not data or "base64" not in header:
        return None
    mime = header.removeprefix("data:").split(";")[0] or "application/octet-stream"
    try:
        return types.Blob(mime_type=mime, data=base64.b64decode(data))
    except (ValueError, TypeError):
        return None


def _guess_mime(filename: str | None, default: str = "application/octet-stream") -> str:
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    return default


def _allowed_tools_guard(allowed: set[str]):
    """ADK before_tool_callback that blocks tools outside the allowed set."""

    def guard(*, tool: BaseTool, args: dict[str, Any], tool_context: ToolContext):
        if tool.name not in allowed:
            return {
                "error": (
                    f"Tool '{tool.name}' is not allowed for this request "
                    "(restricted by allowed_tools)."
                )
            }
        return None

    return guard


def _content_part_to_adk(part: Any) -> types.Part | None:
    """Translate an Open Responses user content part to a genai Part."""
    if isinstance(part, InputText):
        return types.Part(text=part.text) if part.text else None
    if isinstance(part, InputImage):
        if part.image_url:
            if part.image_url.startswith("data:"):
                blob = _decode_data_url(part.image_url)
                if blob is not None:
                    return types.Part(inline_data=blob)
                return None
            return types.Part(
                file_data=types.FileData(
                    file_uri=part.image_url,
                    mime_type=_guess_mime(part.image_url, "image/jpeg"),
                )
            )
        return None
    if isinstance(part, InputFile):
        if part.file_data is not None:
            try:
                data = base64.b64decode(part.file_data)
            except (ValueError, TypeError):
                return None
            return types.Part(
                inline_data=types.Blob(
                    mime_type=part._download_mime_type or _guess_mime(part.filename),
                    data=data,
                    display_name=part.filename,
                )
            )
        if part.file_url:
            return types.Part(
                file_data=types.FileData(
                    file_uri=part.file_url,
                    mime_type=_guess_mime(part.filename or part.file_url),
                )
            )
        return None
    # Unknown content parts: fall back to any text they carry.
    text = getattr(part, "text", None)
    if isinstance(text, str) and text:
        return types.Part(text=text)
    return None


class ClientFunctionTool(BaseTool):
    """A client-implemented function tool, declared from JSON Schema.

    Marked long-running so that when the model calls it, ADK ends the
    invocation and control yields back to the Open Responses client.
    """

    def __init__(self, tool: FunctionTool) -> None:
        super().__init__(
            name=tool.name,
            description=tool.description or "",
            is_long_running=True,
        )
        self._parameters = tool.parameters or {"type": "object", "properties": {}}

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._parameters,
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        # Returning a falsy value from a long-running tool makes ADK skip the
        # function response and end the invocation: control yields back to us.
        return None


class ADKAdapter(AgentAdapter):
    """Expose a Google ADK agent as an Open Responses provider."""

    name = "adk"

    def __init__(
        self,
        agent: BaseAgent,
        *,
        app_name: str = "fastresponses",
        session_service: BaseSessionService | None = None,
        artifact_service: BaseArtifactService | None = None,
        model_name: str | None = None,
        input_file_url_origins: Iterable[str] = (),
        input_file_routes: Mapping[InputFileAction, str | Iterable[str]] | None = None,
        default_input_file_action: InputFileAction = "reject",
        input_file_router: InputFileRouter | None = None,
        input_file_reference_store: InputFileReferenceStore | None = None,
        internal_tool_response_mapper: InternalToolResponseMapper | None = None,
    ) -> None:
        if input_file_routes is not None and input_file_router is not None:
            raise ValueError("Configure input_file_routes or input_file_router, not both.")
        self._validate_input_file_action(default_input_file_action)
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service or InMemorySessionService()
        self.artifact_service = artifact_service or InMemoryArtifactService()
        self.default_model = model_name or self._infer_model_name(agent)
        self.artifact_registry = ArtifactRegistry()
        self.internal_tool_response_mapper = internal_tool_response_mapper
        origins: set[str] = set()
        for origin in input_file_url_origins:
            parsed = httpx.URL(origin)
            canonical = str(
                parsed.copy_with(path="", query=None, fragment=None)
            ).rstrip("/")
            if (
                not parsed.host
                or parsed.userinfo
                or parsed.scheme not in {"http", "https"}
                or str(parsed).rstrip("/") != canonical
            ):
                raise ValueError(f"Input file URL origin is not canonical: {origin!r}.")
            origins.add(canonical)
        self.input_file_url_origins = frozenset(origins)
        self.input_file_routes = self._normalize_input_file_routes(input_file_routes)
        self.default_input_file_action = default_input_file_action
        self.input_file_router = input_file_router
        self.input_file_reference_store = input_file_reference_store or (
            ADKArtifactInputFileStore(self.artifact_service)
        )
        self._input_file_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    @staticmethod
    def _validate_input_file_action(action: str) -> None:
        if action not in {"inline", "url", "reference", "reject"}:
            raise ValueError(f"Unknown input file action: {action!r}.")

    @classmethod
    def _normalize_input_file_routes(
        cls, routes: Mapping[InputFileAction, str | Iterable[str]] | None
    ) -> tuple[tuple[str, InputFileAction], ...]:
        normalized: dict[str, InputFileAction] = {}
        for action, configured in (routes or {}).items():
            cls._validate_input_file_action(action)
            extensions = [configured] if isinstance(configured, str) else configured
            for extension in extensions:
                extension = extension.strip().lower()
                if not extension:
                    raise ValueError("Input file route extensions must not be empty.")
                if not extension.startswith("."):
                    extension = f".{extension}"
                previous = normalized.get(extension)
                if previous is not None:
                    raise ValueError(
                        f"Input file extension {extension!r} is routed to both "
                        f"{previous!r} and {action!r}."
                    )
                normalized[extension] = action
        return tuple(sorted(normalized.items(), key=lambda route: len(route[0]), reverse=True))

    @staticmethod
    def _infer_model_name(agent: BaseAgent) -> str:
        model = getattr(agent, "model", None)
        if isinstance(model, str) and model:
            return f"adk/{agent.name}/{model}"
        return f"adk/{agent.name}"

    # ------------------------------------------------------------------
    # AgentAdapter
    # ------------------------------------------------------------------

    async def run(self, run: AgentRun) -> AsyncIterator[AdapterEvent]:
        async for event in self._run_locked(run):
            yield event

    async def _run_locked(self, run: AgentRun) -> AsyncIterator[AdapterEvent]:
        state = dict(run.previous_state or {})
        user_id: str = state.get("user_id") or run.request.user or "default"

        session = await self._resolve_session(state, user_id)
        created_session = session is None
        if session is None:
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=f"or-{uuid.uuid4().hex}",
            )
        lock = self._input_file_locks.setdefault(session.id, asyncio.Lock())
        created_references: list[str] = []
        references_committed = [False]
        try:
            async with lock:
                refreshed = await self.session_service.get_session(
                    app_name=self.app_name, user_id=user_id, session_id=session.id
                )
                if refreshed is not None:
                    session = refreshed
                async for event in self._run_in_session(
                    run,
                    state,
                    user_id,
                    session,
                    created_references,
                    references_committed,
                ):
                    yield event
        except BaseException:
            if (created_session or not references_committed[0]) and isinstance(
                self.input_file_reference_store, ADKArtifactInputFileStore
            ):
                for reference_id in created_references:
                    try:
                        await self.artifact_service.delete_artifact(
                            app_name=self.app_name,
                            user_id=user_id,
                            session_id=session.id,
                            filename=reference_id,
                        )
                    except Exception:
                        pass
            if created_session:
                try:
                    await self.session_service.delete_session(
                        app_name=self.app_name,
                        user_id=user_id,
                        session_id=session.id,
                    )
                except Exception:
                    pass
            raise

    async def _run_in_session(
        self,
        run: AgentRun,
        state: dict[str, Any],
        user_id: str,
        session: Session,
        created_references: list[str],
        references_committed: list[bool],
    ) -> AsyncIterator[AdapterEvent]:
        call_map: dict[str, dict[str, str]] = dict(state.get("call_ids") or {})
        replay_items = run.context_items[: len(run.context_items) - len(run.new_items)]
        self._validate_input_shape(run.new_items)
        attachment_counter = int(session.state.get(_INPUT_COUNTER_KEY, 0))
        replay: list[Item] = []
        if not state.get("session_id") or state.get("session_id") != session.id:
            # Fresh conversation (or lost session): replay full context first.
            replay, attachment_counter = await self._prepare_input_files(
                replay_items,
                user_id,
                session.id,
                attachment_counter,
                [0, 0],
                max_files=_MAX_REPLAY_INPUT_FILES,
                max_bytes=_MAX_REPLAY_INPUT_FILES_TOTAL_BYTES,
                created_references=created_references,
            )
        input_file_budget = [0, 0]
        hydrated_new, attachment_counter = await self._prepare_input_files(
            run.new_items,
            user_id,
            session.id,
            attachment_counter,
            input_file_budget,
            max_files=_MAX_INPUT_FILES,
            max_bytes=_MAX_INPUT_FILES_TOTAL_BYTES,
            created_references=created_references,
        )
        history, new_message = self._split_input(hydrated_new, call_map)
        if replay:
            history = [*replay, *history]

        await self._seed_history(session, history, call_map)
        if attachment_counter != int(session.state.get(_INPUT_COUNTER_KEY, 0)):
            await self.session_service.append_event(
                session,
                Event(
                    invocation_id=f"or-state-{uuid.uuid4().hex}",
                    author="fastresponses",
                    actions=EventActions(
                        state_delta={_INPUT_COUNTER_KEY: attachment_counter}
                    ),
                ),
            )
        references_committed[0] = True

        client_tools = [ClientFunctionTool(t) for t in run.request.function_tools()]
        agent = self._configure_agent(run, client_tools)
        runner = Runner(
            agent=agent,
            app_name=self.app_name,
            session_service=self.session_service,
            artifact_service=_GeneratedArtifactService(self.artifact_service),
        )

        client_tool_names = {t.name for t in client_tools}
        translator = _EventTranslator(
            client_tool_names,
            call_map,
            artifact_service=self.artifact_service,
            artifact_registry=self.artifact_registry,
            app_name=self.app_name,
            user_id=user_id,
            session_id=session.id,
            internal_tool_response_mapper=self.internal_tool_response_mapper,
        )
        max_tool_calls = run.request.max_tool_calls
        tool_calls = 0

        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=new_message,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                if max_tool_calls is not None and not event.partial:
                    pending = len(event.get_function_calls())
                    if pending and tool_calls + pending > max_tool_calls:
                        yield Incomplete("max_tool_calls")
                        break
                    tool_calls += pending
                for adapter_event in await translator.translate(event):
                    yield adapter_event
                translator.raise_deferred_error()
        except AdapterError:
            raise
        except ValueError as exc:
            raise AdapterError(str(exc), type="invalid_request", code="invalid_value")
        except Exception as exc:
            raise AdapterError(
                f"ADK agent run failed: {exc}", type="model_error"
            ) from exc

        yield StateUpdate(
            {"session_id": session.id, "user_id": user_id, "call_ids": call_map}
        )

    # ------------------------------------------------------------------
    # Input translation
    # ------------------------------------------------------------------

    async def _prepare_input_files(
        self,
        items: list[Item],
        user_id: str,
        session_id: str,
        attachment_counter: int,
        input_file_budget: list[int],
        *,
        max_files: int,
        max_bytes: int,
        created_references: list[str],
    ) -> tuple[list[Item], int]:
        prepared: list[Item] = []
        pending_references: list[
            tuple[list[Any], int, InputFileContent, InputFileReferenceContext]
        ] = []
        for item in items:
            if not isinstance(item, MessageItem) or not isinstance(item.content, list):
                prepared.append(item)
                continue
            content: list[Any] = []
            for part in item.content:
                if not isinstance(part, InputFile):
                    content.append(part)
                    continue
                input_file_budget[0] += 1
                if input_file_budget[0] > max_files:
                    raise AdapterError(
                        "Input contains too many files.",
                        type="invalid_request",
                        code="invalid_value",
                        param="input",
                    )
                if part.file_data is not None and part.file_url is not None:
                    raise AdapterError(
                        "Input file must not provide both file_data and file_url.",
                        type="invalid_request",
                        code="invalid_value",
                        param="input",
                    )
                if part.file_id is not None:
                    raise AdapterError(
                        "ADK input-file routing does not support file_id.",
                        type="invalid_request",
                        code="unsupported_parameter",
                        param="input",
                    )
                try:
                    action = await self._route_input_file(part)
                except AdapterError:
                    raise
                except Exception as exc:
                    raise AdapterError(
                        "Input file router failed.",
                        code="input_file_router_error",
                    ) from exc
                if action == "reject":
                    raise AdapterError(
                        f"Input file {part.filename or '<unnamed>'!r} is not allowed.",
                        type="invalid_request",
                        code="unsupported_file_type",
                        param="input",
                    )
                if action == "url":
                    if not part.file_url:
                        raise AdapterError(
                            "URL-routed input files must provide file_url.",
                            type="invalid_request",
                            code="invalid_value",
                            param="input",
                        )
                    _validate_input_file_url(part.file_url, self.input_file_url_origins)
                    content.append(part)
                    continue
                file = await self._read_input_file(part)
                input_file_budget[1] += len(file.data)
                if input_file_budget[1] > max_bytes:
                    raise AdapterError(
                        "Input files exceed the aggregate size limit.",
                        type="invalid_request",
                        code="invalid_value",
                        param="input",
                    )
                if action == "inline":
                    inline = part.model_copy(
                        update={
                            "filename": file.filename,
                            "file_url": None,
                            "file_data": _b64(file.data),
                        }
                    )
                    inline._download_mime_type = file.mime_type
                    content.append(inline)
                    continue
                if not file.data:
                    raise AdapterError(
                        "Reference-routed input files must not be empty.",
                        type="invalid_request",
                        code="invalid_value",
                        param="input",
                    )
                attachment_counter += 1
                index = len(content)
                content.append(None)
                pending_references.append(
                    (
                        content,
                        index,
                        file,
                        InputFileReferenceContext(
                            app_name=self.app_name,
                            user_id=user_id,
                            session_id=session_id,
                            reference_id=f"{_INPUT_ARTIFACT_PREFIX}{attachment_counter}",
                        ),
                    )
                )
            prepared.append(item.model_copy(update={"content": content}))
        for content, index, file, context in pending_references:
            try:
                text = await self.input_file_reference_store.create_reference(
                    file, context=context
                )
                if not isinstance(text, str):
                    raise TypeError("create_reference() must return a string")
                created_references.append(context.reference_id)
            except Exception as exc:
                raise AdapterError(
                    "Input file reference store failed.",
                    code="input_file_reference_store_error",
                ) from exc
            content[index] = InputText(text=text)
        return prepared, attachment_counter

    async def _route_input_file(self, part: InputFile) -> InputFileAction:
        if self.input_file_router is not None:
            action = self.input_file_router(part)
            if isinstance(action, Awaitable):
                action = await action
            try:
                self._validate_input_file_action(action)
            except ValueError as exc:
                raise AdapterError(
                    str(exc),
                    type="invalid_request",
                    code="invalid_value",
                    param="input",
                ) from exc
            return action
        filename = (part.filename or "").lower()
        for extension, action in self.input_file_routes:
            if filename.endswith(extension):
                return action
        return self.default_input_file_action

    async def _read_input_file(self, part: InputFile) -> InputFileContent:
        if part.file_url:
            data, mime_type = await _fetch_input_file(
                part.file_url, part.filename, self.input_file_url_origins
            )
        elif part.file_data is not None:
            try:
                data = base64.b64decode(part.file_data, validate=True)
            except (ValueError, TypeError) as exc:
                raise AdapterError(
                    "Input file contains invalid base64 data.",
                    type="invalid_request",
                    code="invalid_value",
                    param="input",
                ) from exc
            if len(data) > _MAX_INPUT_FILE_BYTES:
                raise AdapterError(
                    "Input file exceeds the maximum size.",
                    type="invalid_request",
                    code="invalid_value",
                    param="input",
                )
            mime_type = _guess_mime(part.filename)
        else:
            raise AdapterError(
                "Input file must provide file_data or file_url.",
                type="invalid_request",
                code="invalid_value",
                param="input",
            )
        filename = part.filename
        if not filename:
            extension = mimetypes.guess_extension(mime_type) or ".bin"
            filename = f"attachment{extension}"
        return InputFileContent(filename=filename, mime_type=mime_type, data=data)

    @staticmethod
    def _validate_input_shape(items: list[Item]) -> None:
        if not items:
            raise AdapterError(
                "Request 'input' must contain at least one item.",
                type="invalid_request",
                code="invalid_value",
                param="input",
            )
        if isinstance(items[-1], MessageItem) and items[-1].role == "user":
            return
        if isinstance(items[-1], FunctionCallOutputItem):
            return
        raise AdapterError(
            "Request 'input' must end with a user message or with "
            "function_call_output items.",
            type="invalid_request",
            code="invalid_value",
            param="input",
        )

    def _split_input(
        self, items: list[Item], call_map: dict[str, dict[str, str]]
    ) -> tuple[list[Item], types.Content]:
        """Split new input into (history items, ADK new_message).

        The new message is either the trailing user message (anything before
        it, including function_call_output items, is seeded as session
        history) or, when the input ends with function call outputs, a
        function_response message that resumes the paused invocation.
        """
        if not items:
            raise AdapterError(
                "Request 'input' must contain at least one item.",
                type="invalid_request",
                code="invalid_value",
                param="input",
            )

        last = items[-1]
        if isinstance(last, MessageItem) and last.role == "user":
            return items[:-1], types.Content(
                role="user", parts=self._user_message_parts(last)
            )

        if isinstance(last, FunctionCallOutputItem):
            outputs: list[FunctionCallOutputItem] = []
            index = len(items)
            while index > 0 and isinstance(items[index - 1], FunctionCallOutputItem):
                index -= 1
                outputs.insert(0, items[index])  # type: ignore[arg-type]
            parts = [self._function_response_part(o, call_map) for o in outputs]
            return items[:index], types.Content(role="user", parts=parts)

        raise AdapterError(
            "Request 'input' must end with a user message or with "
            "function_call_output items.",
            type="invalid_request",
            code="invalid_value",
            param="input",
        )

    @staticmethod
    def _user_message_parts(message: MessageItem) -> list[types.Part]:
        parts: list[types.Part] = []
        if isinstance(message.content, str):
            if message.content:
                parts.append(types.Part(text=message.content))
        else:
            for part in message.content:
                adk_part = _content_part_to_adk(part)
                if adk_part is not None:
                    parts.append(adk_part)
        if not parts:
            parts.append(types.Part(text=""))
        return parts

    @staticmethod
    def _function_response_part(
        output_item: FunctionCallOutputItem, call_map: dict[str, dict[str, str]]
    ) -> types.Part:
        mapping = call_map.get(output_item.call_id, {})
        adk_call_id = mapping.get("id") or output_item.call_id
        name = mapping.get("name") or output_item.call_id
        output = output_item.output
        response: dict[str, Any]
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
            except (ValueError, TypeError):
                parsed = output
            response = parsed if isinstance(parsed, dict) else {"result": parsed}
        else:
            response = {"result": output}
        return types.Part(
            function_response=types.FunctionResponse(
                id=adk_call_id, name=name, response=response
            )
        )

    # ------------------------------------------------------------------
    # Session handling
    # ------------------------------------------------------------------

    async def _resolve_session(
        self, state: dict[str, Any], user_id: str
    ) -> Session | None:
        session_id = state.get("session_id")
        if not session_id:
            return None
        return await self.session_service.get_session(
            app_name=self.app_name, user_id=user_id, session_id=session_id
        )

    async def _seed_history(
        self,
        session: Session,
        items: list[Item],
        call_map: dict[str, dict[str, str]],
    ) -> None:
        """Append prior conversation items as ADK session events."""
        invocation_id = f"or-seed-{uuid.uuid4().hex}"
        expanded: list[Item] = []
        for item in items:
            if isinstance(item, CompactionItem):
                expanded.extend(expand_compaction_item(item))
            else:
                expanded.append(item)
        for item in expanded:
            event = self._history_event(item, call_map, invocation_id)
            if event is not None:
                await self.session_service.append_event(session, event)

    def _history_event(
        self,
        item: Item,
        call_map: dict[str, dict[str, str]],
        invocation_id: str,
    ) -> Event | None:
        agent_name = self.agent.name
        if isinstance(item, MessageItem):
            if item.role == "assistant":
                text = item.text()
                if not text:
                    return None
                return Event(
                    invocation_id=invocation_id,
                    author=agent_name,
                    content=types.Content(role="model", parts=[types.Part(text=text)]),
                )
            parts = self._user_message_parts(item)
            if all(p.text == "" for p in parts if p.text is not None) and not any(
                p.inline_data or p.file_data for p in parts
            ):
                return None
            return Event(
                invocation_id=invocation_id,
                author="user",
                content=types.Content(role="user", parts=parts),
            )
        if isinstance(item, FunctionCallItem):
            mapping = call_map.setdefault(
                item.call_id, {"id": item.call_id, "name": item.name}
            )
            try:
                args = json.loads(item.arguments) if item.arguments else {}
            except (ValueError, TypeError):
                args = {"_raw": item.arguments}
            return Event(
                invocation_id=invocation_id,
                author=agent_name,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id=mapping["id"], name=item.name, args=args
                            )
                        )
                    ],
                ),
            )
        if isinstance(item, FunctionCallOutputItem):
            return Event(
                invocation_id=invocation_id,
                author="user",
                content=types.Content(
                    role="user", parts=[self._function_response_part(item, call_map)]
                ),
            )
        # Reasoning items, item references, and unknown extension items are
        # not replayable into an ADK session; skip them.
        return None

    # ------------------------------------------------------------------
    # Per-request agent configuration
    # ------------------------------------------------------------------

    def _configure_agent(
        self, run: AgentRun, client_tools: list[ClientFunctionTool]
    ) -> BaseAgent:
        request = run.request
        update: dict[str, Any] = {}

        if client_tools:
            if not isinstance(self.agent, LlmAgent):
                raise AdapterError(
                    "Client-defined function tools require an LlmAgent root agent.",
                    type="invalid_request",
                    code="unsupported_parameter",
                    param="tools",
                )
            update["tools"] = [*self.agent.tools, *client_tools]

        if request.instructions is not None and isinstance(self.agent, LlmAgent):
            update["instruction"] = request.instructions

        if isinstance(self.agent, LlmAgent):
            config = self._generate_content_config(run)
            if config is not None:
                update["generate_content_config"] = config
            # `allowed_tools` is a hard constraint: block execution of any
            # tool outside the allowed set, not just hint the model.
            allowed = self._allowed_tool_names(request.tool_choice)
            if allowed is not None:
                update["before_tool_callback"] = _allowed_tools_guard(allowed)

        if not update:
            return self.agent
        return self.agent.clone(update=update)

    @staticmethod
    def _allowed_tool_names(choice: Any) -> set[str] | None:
        if isinstance(choice, ToolChoiceAllowed):
            return {t.get("name") for t in choice.tools if t.get("name")}
        if isinstance(choice, ToolChoiceFunction):
            return {choice.name}
        return None

    def _generate_content_config(
        self, run: AgentRun
    ) -> types.GenerateContentConfig | None:
        request = run.request
        base = getattr(self.agent, "generate_content_config", None)
        config = base.model_copy(deep=True) if base is not None else None

        def ensure() -> types.GenerateContentConfig:
            nonlocal config
            if config is None:
                config = types.GenerateContentConfig()
            return config

        if request.temperature is not None:
            ensure().temperature = request.temperature
        if request.top_p is not None:
            ensure().top_p = request.top_p
        if request.presence_penalty is not None:
            ensure().presence_penalty = request.presence_penalty
        if request.frequency_penalty is not None:
            ensure().frequency_penalty = request.frequency_penalty
        if request.max_output_tokens is not None:
            ensure().max_output_tokens = request.max_output_tokens
        if request.top_logprobs:
            ensure().response_logprobs = True
            ensure().logprobs = request.top_logprobs

        # text.format -> structured output
        fmt = request.text.format if request.text is not None else None
        if isinstance(fmt, JsonSchemaResponseFormat):
            ensure().response_mime_type = "application/json"
            if fmt.json_schema:
                ensure().response_json_schema = fmt.json_schema
        elif isinstance(fmt, JsonObjectResponseFormat):
            ensure().response_mime_type = "application/json"

        # reasoning -> thinking config
        thinking = self._thinking_config(run)
        if thinking is not None:
            ensure().thinking_config = thinking

        tool_config = self._tool_config(run)
        if tool_config is not None:
            ensure().tool_config = tool_config

        return config

    _THINKING_BUDGETS = {
        "none": 0,
        "minimal": 512,
        "low": 1024,
        "medium": 8192,
        "high": 24576,
        "xhigh": 32768,
    }

    def _thinking_config(self, run: AgentRun) -> types.ThinkingConfig | None:
        reasoning = run.request.reasoning
        if reasoning is None or (
            reasoning.effort is None and reasoning.summary is None
        ):
            return None
        include_thoughts = reasoning.summary is not None or (
            reasoning.effort is not None and reasoning.effort != "none"
        )
        config = types.ThinkingConfig(include_thoughts=include_thoughts)
        if reasoning.effort is not None:
            config.thinking_budget = self._THINKING_BUDGETS.get(reasoning.effort)
        return config

    def _tool_config(self, run: AgentRun) -> types.ToolConfig | None:
        choice = run.request.tool_choice
        fcc: types.FunctionCallingConfig | None = None
        if choice == "none":
            fcc = types.FunctionCallingConfig(mode="NONE")
        elif choice == "required":
            fcc = types.FunctionCallingConfig(mode="ANY")
        elif isinstance(choice, ToolChoiceFunction):
            fcc = types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=[choice.name]
            )
        elif isinstance(choice, ToolChoiceAllowed):
            names = [t.get("name") for t in choice.tools if t.get("name")]
            if names:
                fcc = types.FunctionCallingConfig(
                    mode="ANY" if choice.mode == "required" else "AUTO",
                    allowed_function_names=names,
                )
        if fcc is None:
            return None
        return types.ToolConfig(function_calling_config=fcc)


class _EventTranslator:
    """Translates a stream of ADK events into adapter events."""

    def __init__(
        self,
        client_tool_names: set[str],
        call_map: dict[str, dict[str, str]],
        *,
        artifact_service: BaseArtifactService,
        artifact_registry: ArtifactRegistry,
        app_name: str,
        user_id: str,
        session_id: str,
        internal_tool_response_mapper: InternalToolResponseMapper | None,
    ) -> None:
        self.client_tool_names = client_tool_names
        self.call_map = call_map
        self.artifact_service = artifact_service
        self.artifact_registry = artifact_registry
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id
        self.internal_tool_response_mapper = internal_tool_response_mapper
        self._deferred_error: AdapterError | None = None
        self._streamed_chars = 0
        self._streamed_thought_chars = 0
        self._open_calls: dict[str, FunctionCallItem] = {}

    async def translate(self, event: Event) -> list[AdapterEvent]:
        if event.error_code or event.error_message:
            raise AdapterError(
                event.error_message or f"ADK error: {event.error_code}",
                type="model_error",
                code=event.error_code,
            )

        out: list[AdapterEvent] = []
        parts = list(event.content.parts or []) if event.content else []

        if event.partial:
            for part in parts:
                if not part.text or part.function_call:
                    continue
                if part.thought:
                    self._streamed_thought_chars += len(part.text)
                    out.append(ReasoningDelta(delta=part.text))
                    if part.thought_signature:
                        out.append(
                            ReasoningDelta(
                                encrypted_content=_b64(part.thought_signature)
                            )
                        )
                else:
                    self._streamed_chars += len(part.text)
                    out.append(TextDelta(part.text))
            return out

        # Final (aggregated) event for this step.
        out.extend(self._final_reasoning(parts))
        text = "".join(
            p.text for p in parts if p.text and not p.thought and not p.function_call
        )
        if text:
            if self._streamed_chars == 0:
                out.append(TextDelta(text))
            self._streamed_chars = 0

        long_running_ids = event.long_running_tool_ids or set()
        for fc in event.get_function_calls():
            args = json.dumps(fc.args or {})
            if fc.id in long_running_ids and fc.name in self.client_tool_names:
                out.append(self._yield_client_call(fc, args))
            else:
                out.append(self._open_internal_call(fc, args))

        for fr in event.get_function_responses():
            call, output_item = self._close_internal_call(fr)
            out.append(ItemDone(call))
            out.append(ItemDone(output_item))
            if self.internal_tool_response_mapper is not None:
                try:
                    arguments = json.loads(call.arguments)
                    if not isinstance(arguments, dict):
                        arguments = {}
                    context = ADKToolResponse(
                        name=fr.name or "",
                        arguments=arguments,
                        response=fr.response,
                        output=output_item.output if isinstance(output_item.output, str) else None,
                        author=event.author,
                        call=call,
                        output_item=output_item,
                        create_artifact=lambda filename, data, mime_type: self._create_artifact(
                            filename, data, mime_type, call.call_id
                        ),
                    )
                    mapped = self.internal_tool_response_mapper(context)
                    if isinstance(mapped, Awaitable):
                        mapped = await mapped
                    out.extend(ItemDone(item) for item in mapped)
                except Exception as exc:
                    self._deferred_error = AdapterError(
                        "Internal tool response mapper failed.",
                        code="internal_tool_response_mapper_error",
                    )
                    self._deferred_error.__cause__ = exc

        if event.actions.artifact_delta:
            for filename, version in event.actions.artifact_delta.items():
                if not filename.startswith(_INPUT_ARTIFACT_PREFIX):
                    out.append(await self._artifact_item(filename, version))

        usage = event.usage_metadata
        if usage is not None:
            out.append(
                UsageDelta(
                    input_tokens=usage.prompt_token_count or 0,
                    output_tokens=(usage.candidates_token_count or 0)
                    + (usage.thoughts_token_count or 0),
                    total_tokens=usage.total_token_count or 0,
                    reasoning_tokens=usage.thoughts_token_count or 0,
                    cached_tokens=usage.cached_content_token_count or 0,
                )
            )

        if event.finish_reason == types.FinishReason.MAX_TOKENS:
            out.append(Incomplete("max_output_tokens"))
        return out

    def raise_deferred_error(self) -> None:
        if self._deferred_error is not None:
            error, self._deferred_error = self._deferred_error, None
            raise error

    def _final_reasoning(self, parts: list[types.Part]) -> list[AdapterEvent]:
        """Reasoning ("thought") handling for a final aggregated event.

        Thought text not already streamed via partial chunks is emitted now.
        A thought signature is attached as ``encrypted_content`` while the
        reasoning block is still open; if the block was already closed by
        streamed answer text, the signature is dropped (the full-fidelity
        trace lives in the ADK session, so continuation does not depend on
        the client echoing it back).
        """
        out: list[AdapterEvent] = []
        thought_text = "".join(p.text for p in parts if p.text and p.thought)
        signature = next(
            (p.thought_signature for p in parts if p.thought_signature), None
        )
        if thought_text and self._streamed_thought_chars == 0:
            out.append(ReasoningDelta(delta=thought_text))
            if signature:
                out.append(ReasoningDelta(encrypted_content=_b64(signature)))
        elif (
            signature and self._streamed_thought_chars > 0 and self._streamed_chars == 0
        ):
            out.append(ReasoningDelta(encrypted_content=_b64(signature)))
        self._streamed_thought_chars = 0
        return out

    def _yield_client_call(self, fc: types.FunctionCall, args: str) -> ItemDone:
        call_id = new_call_id()
        self.call_map[call_id] = {"id": fc.id or call_id, "name": fc.name or ""}
        return ItemDone(
            FunctionCallItem(
                id=new_function_call_id(),
                call_id=call_id,
                name=fc.name or "",
                arguments=args,
                status="completed",
            )
        )

    def _open_internal_call(self, fc: types.FunctionCall, args: str) -> ItemAdded:
        call_id = new_call_id()
        item = FunctionCallItem(
            id=new_function_call_id(),
            call_id=call_id,
            status="in_progress",
            name=fc.name or "",
            arguments=args,
        )
        if fc.id:
            self._open_calls[fc.id] = item
            self.call_map[call_id] = {"id": fc.id, "name": fc.name or ""}
        return ItemAdded(item)

    def _close_internal_call(
        self, fr: types.FunctionResponse
    ) -> tuple[FunctionCallItem, FunctionCallOutputItem]:
        opened = self._open_calls.pop(fr.id or "", None)
        call = opened or FunctionCallItem(
            id=new_function_call_id(),
            call_id=new_call_id(),
            name=fr.name or "",
            arguments="{}",
        )
        call = call.model_copy(update={"status": "completed"})
        self.call_map.setdefault(
            call.call_id, {"id": fr.id or call.call_id, "name": fr.name or ""}
        )
        output_item = FunctionCallOutputItem(
            id=f"fco_{uuid.uuid4().hex}",
            call_id=call.call_id,
            output=json.dumps(fr.response) if fr.response is not None else "null",
            status="completed",
        )
        return call, output_item

    async def _create_artifact(
        self, filename: str, data: bytes, mime_type: str, call_id: str | None = None
    ) -> Item:
        _validate_generated_artifact_name(filename)
        version = await self.artifact_service.save_artifact(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
            filename=filename,
            artifact=types.Part.from_bytes(data=data, mime_type=mime_type),
        )
        return (await self._artifact_item(filename, version, call_id)).item

    async def _artifact_item(
        self, filename: str, version: int, call_id: str | None = None
    ) -> ItemDone:
        part = await self.artifact_service.load_artifact(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
            filename=filename,
            version=version,
        )
        blob = part.inline_data if part is not None else None
        if blob is None or blob.data is None:
            raise AdapterError(
                f"Generated artifact '{filename}' could not be loaded.",
                code="artifact_not_found",
            )
        mime_type = blob.mime_type or _guess_mime(filename)
        artifact_id = self.artifact_registry.register(
            ArtifactRecord(
                service=self.artifact_service,
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session_id,
                filename=filename,
                version=version,
                mime_type=mime_type,
            )
        )
        return ItemDone(
            CustomItem.model_validate(
                {
                    "type": ARTIFACT_TYPE,
                    "id": artifact_id,
                    "status": "completed",
                    "filename": filename,
                    "mime_type": mime_type,
                    "size": len(blob.data),
                    "content_url": f"/v1/artifacts/{artifact_id}/content",
                    **({"call_id": call_id} if call_id else {}),
                }
            )
        )

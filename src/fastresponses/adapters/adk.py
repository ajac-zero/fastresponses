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

import base64
import json
import mimetypes
import uuid
from collections.abc import AsyncIterator
from typing import Any

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event
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
from ..compaction import expand_compaction_item
from ..models import (
    CompactionItem,
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

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


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
        if part.file_data:
            try:
                data = base64.b64decode(part.file_data)
            except (ValueError, TypeError):
                return None
            return types.Part(
                inline_data=types.Blob(
                    mime_type=_guess_mime(part.filename), data=data
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

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
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
        model_name: str | None = None,
    ) -> None:
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service or InMemorySessionService()
        self.default_model = model_name or self._infer_model_name(agent)

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
        state = dict(run.previous_state or {})
        call_map: dict[str, dict[str, str]] = dict(state.get("call_ids") or {})
        user_id: str = state.get("user_id") or run.request.user or "default"

        history, new_message = self._split_input(run.new_items, call_map)

        session = await self._resolve_session(state, user_id)
        if session is None:
            # Fresh conversation (or lost session): replay full context.
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=f"or-{uuid.uuid4().hex}",
            )
            replay = run.context_items[: len(run.context_items) - len(run.new_items)]
            history = [*replay, *history]

        await self._seed_history(session, history, call_map)

        client_tools = [ClientFunctionTool(t) for t in run.request.function_tools()]
        agent = self._configure_agent(run, client_tools)
        runner = Runner(
            agent=agent,
            app_name=self.app_name,
            session_service=self.session_service,
        )

        client_tool_names = {t.name for t in client_tools}
        translator = _EventTranslator(client_tool_names, call_map)
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
                for adapter_event in translator.translate(event):
                    yield adapter_event
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
        if reasoning is None or (reasoning.effort is None and reasoning.summary is None):
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
    ) -> None:
        self.client_tool_names = client_tool_names
        self.call_map = call_map
        self._streamed_chars = 0
        self._streamed_thought_chars = 0
        self._open_calls: dict[str, FunctionCallItem] = {}

    def translate(self, event: Event) -> list[AdapterEvent]:
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
        elif signature and self._streamed_thought_chars > 0 and self._streamed_chars == 0:
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

"""Pydantic AI adapter: expose a ``pydantic_ai.Agent`` as an Open Responses
provider.

Mapping notes (mirrors the design of :mod:`.adk`):

- **Client-declared tools** (``tools`` in the request) become an
  :class:`pydantic_ai.ExternalToolset`; when the model calls one, the run
  yields ``DeferredToolRequests`` and the adapter surfaces ``function_call``
  items, yielding control to the client. ``function_call_output`` items in a
  follow-up request resume the run via ``DeferredToolResults``.
- **Agent-internal tools** run inside Pydantic AI and are surfaced as
  ``pydantic_ai:function_call`` extension items (a receipt of what happened).
- **Continuation**: Pydantic AI is stateless, so the adapter state stores the
  serialized message history (``ModelMessagesTypeAdapter``) plus the set of
  pending deferred call ids. Clients that don't use ``previous_response_id``
  get full stateless replay from ``context_items``.
- **Reasoning**: ``ThinkingPart``/``ThinkingPartDelta`` become reasoning
  deltas; thinking signatures map to ``encrypted_content``.
"""

from __future__ import annotations

import base64
import contextlib
import json
import mimetypes
from collections.abc import AsyncIterator
from typing import Any

try:
    from pydantic_ai import (
        Agent,
        AgentRunResultEvent,
        DeferredToolRequests,
        DeferredToolResults,
        ExternalToolset,
    )
    from pydantic_ai import messages as pai_messages
    from pydantic_ai.exceptions import UsageLimitExceeded, UserError
    from pydantic_ai.messages import (
        BinaryContent,
        DocumentUrl,
        ImageUrl,
        ModelMessagesTypeAdapter,
        ModelRequest,
        ModelResponse,
        PartDeltaEvent,
        PartStartEvent,
        SystemPromptPart,
        TextPart,
        TextPartDelta,
        ThinkingPart,
        ThinkingPartDelta,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )
    from pydantic_ai.output import StructuredDict
    from pydantic_ai.settings import ModelSettings, ToolOrOutput
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.toolsets import WrapperToolset
    from pydantic_ai.usage import UsageLimits
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The Pydantic AI adapter requires the 'pydantic-ai' extra: "
        "pip install 'fastresponses[pydantic-ai]'"
    ) from exc

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
    ReasoningItem,
    ToolChoiceAllowed,
    ToolChoiceFunction,
    new_function_call_id,
)

EXTENSION_FUNCTION_CALL = "pydantic_ai:function_call"

_THINKING_LEVELS = {
    "none": False,
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}


def _output_str(output: str | list[Any]) -> str:
    if isinstance(output, str):
        return output
    texts = [getattr(p, "text", None) or str(p) for p in output]
    return "\n".join(texts)


def _user_content(message: MessageItem) -> list[Any]:
    """Translate an Open Responses user message to Pydantic AI UserContent."""
    if isinstance(message.content, str):
        return [message.content]
    content: list[Any] = []
    for part in message.content:
        if isinstance(part, InputText):
            if part.text:
                content.append(part.text)
        elif isinstance(part, InputImage):
            url = part.image_url
            if not url:
                continue
            if url.startswith("data:"):
                blob = _decode_data_url(url)
                if blob is not None:
                    content.append(blob)
            else:
                content.append(ImageUrl(url=url))
        elif isinstance(part, InputFile):
            if part.file_data:
                try:
                    data = base64.b64decode(part.file_data)
                except (ValueError, TypeError):
                    continue
                content.append(
                    BinaryContent(
                        data=data, media_type=_guess_mime(part.filename)
                    )
                )
            elif part.file_url:
                content.append(DocumentUrl(url=part.file_url))
        else:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                content.append(text)
    return content or [""]


def _decode_data_url(url: str) -> BinaryContent | None:
    header, _, data = url.partition(",")
    if not data or "base64" not in header:
        return None
    mime = header.removeprefix("data:").split(";")[0] or "application/octet-stream"
    try:
        return BinaryContent(data=base64.b64decode(data), media_type=mime)
    except (ValueError, TypeError):
        return None


def _guess_mime(filename: str | None, default: str = "application/octet-stream") -> str:
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    return default


class _AllowedToolsGuard(WrapperToolset):
    """Hard enforcement of ``allowed_tools``: execution of tools outside the
    allowed set is suppressed and the model receives an error result instead
    (``tool_choice`` alone is only a request-level hint to the model)."""

    def __init__(self, wrapped: Any, allowed: frozenset[str]) -> None:
        super().__init__(wrapped)
        self.allowed = allowed

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: Any, tool: Any
    ) -> Any:
        kind = getattr(getattr(tool, "tool_def", None), "kind", "function")
        if kind == "function" and name not in self.allowed:
            return {
                "error": (
                    f"Tool '{name}' is not allowed for this request "
                    "(restricted by allowed_tools)."
                )
            }
        return await super().call_tool(name, tool_args, ctx, tool)


class PydanticAIAdapter(AgentAdapter):
    """Expose a Pydantic AI agent as an Open Responses provider."""

    name = "pydantic_ai"

    def __init__(self, agent: Agent, *, model_name: str | None = None) -> None:
        self.agent = agent
        self.default_model = model_name or self._infer_model_name(agent)

    @staticmethod
    def _infer_model_name(agent: Agent) -> str:
        agent_name = agent.name or "agent"
        model = getattr(agent, "model", None)
        model_name = getattr(model, "model_name", None) or (
            model if isinstance(model, str) else None
        )
        if model_name:
            return f"pydantic-ai/{agent_name}/{model_name}"
        return f"pydantic-ai/{agent_name}"

    # ------------------------------------------------------------------
    # AgentAdapter
    # ------------------------------------------------------------------

    async def run(self, run: AgentRun) -> AsyncIterator[AdapterEvent]:
        state = dict(run.previous_state or {})
        pending_calls: set[str] = set(state.get("pending_calls") or [])

        items = run.new_items if state.get("history") else run.context_items
        trailing_outputs = self._trailing_outputs(items)
        history_items = items[: len(items) - len(trailing_outputs)]

        message_history = self._base_history(state) + self._items_to_messages(
            history_items
        )
        user_prompt: list[Any] | None = None
        if message_history and isinstance(message_history[-1], ModelRequest):
            # Pull a trailing user prompt out of the history so Pydantic AI
            # sees it as this run's prompt.
            last = message_history[-1]
            if len(last.parts) == 1 and isinstance(last.parts[0], UserPromptPart):
                content = last.parts[0].content
                user_prompt = list(content) if isinstance(content, list) else [content]
                message_history = message_history[:-1]

        deferred_results = self._deferred_results(
            trailing_outputs, pending_calls, message_history
        )

        client_tools = [t for t in run.request.tools if isinstance(t, FunctionTool)]
        toolsets = self._toolsets(client_tools)
        output_type = self._output_type(run)
        settings = self._model_settings(run)
        usage_limits = (
            UsageLimits(tool_calls_limit=run.request.max_tool_calls)
            if run.request.max_tool_calls is not None
            else None
        )

        translator = _EventTranslator({t.name for t in client_tools})
        incomplete_reason: str | None = None
        result = None

        allowed = self._allowed_tool_names(run.request.tool_choice)
        if allowed is not None:
            # Replace the agent's function toolset with a guarded wrapper.
            # `override(toolsets=...)` suppresses run-level toolsets, so the
            # client-tools ExternalToolset moves into the override too.
            guarded: list[Any] = [
                _AllowedToolsGuard(ts, frozenset(allowed))
                for ts in self.agent.toolsets
            ]
            if toolsets:
                guarded.extend(toolsets)
                toolsets = None
            override = self.agent.override(tools=[], toolsets=guarded)
        else:
            override = contextlib.nullcontext()

        try:
            with override:
                async with self.agent.run_stream_events(
                    user_prompt,
                    message_history=message_history or None,
                    deferred_tool_results=deferred_results,
                    instructions=run.request.instructions,
                    model_settings=settings,
                    usage_limits=usage_limits,
                    toolsets=toolsets,
                    output_type=output_type,
                ) as stream:
                    async for event in stream:
                        if isinstance(event, AgentRunResultEvent):
                            result = event.result
                            continue
                        for adapter_event in translator.translate(event):
                            yield adapter_event
        except UsageLimitExceeded as exc:
            if "tool_calls" not in str(exc):  # pragma: no cover - defensive
                raise AdapterError(str(exc), type="invalid_request") from exc
            incomplete_reason = "max_tool_calls"
        except UserError as exc:
            raise AdapterError(
                str(exc), type="invalid_request", code="invalid_value", param="input"
            ) from exc

        new_pending: set[str] = set()
        if result is not None:
            if isinstance(result.output, DeferredToolRequests):
                for call in result.output.calls:
                    new_pending.add(call.tool_call_id)
                    yield ItemDone(
                        FunctionCallItem(
                            id=new_function_call_id(),
                            call_id=call.tool_call_id,
                            name=call.tool_name,
                            arguments=call.args_as_json_str(),
                            status="completed",
                        )
                    )
            yield self._usage_delta(result)
            last = result.all_messages()[-1] if result.all_messages() else None
            if (
                isinstance(last, ModelResponse)
                and last.finish_reason == "length"
            ):
                incomplete_reason = "max_output_tokens"
            yield StateUpdate(
                {
                    "history": result.all_messages_json().decode(),
                    "pending_calls": sorted(new_pending),
                }
            )

        if incomplete_reason is not None:
            yield Incomplete(incomplete_reason)

    # ------------------------------------------------------------------
    # Input translation
    # ------------------------------------------------------------------

    @staticmethod
    def _trailing_outputs(items: list[Item]) -> list[FunctionCallOutputItem]:
        outputs: list[FunctionCallOutputItem] = []
        for item in reversed(items):
            if isinstance(item, FunctionCallOutputItem):
                outputs.append(item)
            else:
                break
        outputs.reverse()
        return outputs

    @staticmethod
    def _base_history(state: dict[str, Any]) -> list[Any]:
        raw = state.get("history")
        if not raw:
            return []
        try:
            return list(ModelMessagesTypeAdapter.validate_json(raw))
        except ValueError as exc:  # pragma: no cover - corrupted state
            raise AdapterError(
                "Stored conversation state could not be restored.",
                type="invalid_request",
                code="previous_response_not_found",
                param="previous_response_id",
            ) from exc

    def _items_to_messages(self, items: list[Item]) -> list[Any]:
        """Convert Open Responses items to Pydantic AI messages (stateless
        replay). ``function_call`` items keep their ``call_id`` as the
        ``tool_call_id`` so outputs can be matched without extra state."""
        expanded: list[Item] = []
        for item in items:
            if isinstance(item, CompactionItem):
                expanded.extend(expand_compaction_item(item))
            else:
                expanded.append(item)

        messages: list[Any] = []

        def response_parts() -> list[Any]:
            if messages and isinstance(messages[-1], ModelResponse):
                return messages[-1].parts
            messages.append(ModelResponse(parts=[]))
            return messages[-1].parts

        def request_parts() -> list[Any]:
            if messages and isinstance(messages[-1], ModelRequest):
                return messages[-1].parts
            messages.append(ModelRequest(parts=[]))
            return messages[-1].parts

        for item in expanded:
            if isinstance(item, MessageItem):
                if item.role == "assistant":
                    text = item.text()
                    if text:
                        response_parts().append(TextPart(content=text))
                elif item.role == "system" or item.role == "developer":
                    text = item.text()
                    if text:
                        request_parts().append(SystemPromptPart(content=text))
                else:
                    content = _user_content(item)
                    request_parts().append(
                        UserPromptPart(
                            content=content[0]
                            if len(content) == 1 and isinstance(content[0], str)
                            else content
                        )
                    )
            elif isinstance(item, FunctionCallItem):
                response_parts().append(
                    ToolCallPart(
                        tool_name=item.name,
                        args=item.arguments or "{}",
                        tool_call_id=item.call_id,
                    )
                )
            elif isinstance(item, FunctionCallOutputItem):
                tool_name = self._tool_name_for_call(messages, item.call_id)
                request_parts().append(
                    ToolReturnPart(
                        tool_name=tool_name,
                        content=_output_str(item.output),
                        tool_call_id=item.call_id,
                    )
                )
            elif isinstance(item, ReasoningItem):
                text = "".join(s.text for s in item.summary)
                if text or item.encrypted_content:
                    response_parts().append(
                        ThinkingPart(
                            content=text, signature=item.encrypted_content
                        )
                    )
            elif isinstance(item, CustomItem) and item.type == EXTENSION_FUNCTION_CALL:
                # Receipt of an internal tool call: replay as call + return.
                extra = item.model_extra or {}
                call_id = f"internal_{item.id or new_function_call_id()}"
                response_parts().append(
                    ToolCallPart(
                        tool_name=extra.get("name") or "tool",
                        args=extra.get("arguments") or "{}",
                        tool_call_id=call_id,
                    )
                )
                request_parts().append(
                    ToolReturnPart(
                        tool_name=extra.get("name") or "tool",
                        content=extra.get("output") or "",
                        tool_call_id=call_id,
                    )
                )
        return messages

    @staticmethod
    def _tool_name_for_call(messages: list[Any], call_id: str) -> str:
        for message in reversed(messages):
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if (
                        isinstance(part, ToolCallPart)
                        and part.tool_call_id == call_id
                    ):
                        return part.tool_name
        return "tool"

    def _deferred_results(
        self,
        outputs: list[FunctionCallOutputItem],
        pending_calls: set[str],
        message_history: list[Any],
    ) -> DeferredToolResults | None:
        if not outputs:
            return None
        known = set(pending_calls)
        for message in message_history:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        known.add(part.tool_call_id)
        results = DeferredToolResults()
        for output in outputs:
            if output.call_id not in known:
                raise AdapterError(
                    f"No tool call with call_id '{output.call_id}' is awaiting "
                    "output in this conversation.",
                    type="invalid_request",
                    code="invalid_value",
                    param="input",
                )
            results.calls[output.call_id] = _output_str(output.output)
        return results

    # ------------------------------------------------------------------
    # Run configuration
    # ------------------------------------------------------------------

    def _toolsets(self, client_tools: list[FunctionTool]) -> list[Any] | None:
        if not client_tools:
            return None
        return [
            ExternalToolset(
                [
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description or "",
                        parameters_json_schema=tool.parameters
                        or {"type": "object", "properties": {}},
                        strict=tool.strict,
                    )
                    for tool in client_tools
                ],
                id="open-responses-client-tools",
            )
        ]

    def _output_type(self, run: AgentRun) -> Any | None:
        fmt = run.request.text.format if run.request.text is not None else None
        structured = None
        if isinstance(fmt, JsonSchemaResponseFormat) and fmt.json_schema:
            structured = StructuredDict(
                fmt.json_schema,
                name=fmt.name or None,
                description=fmt.description or None,
            )
        elif isinstance(fmt, JsonObjectResponseFormat):
            structured = StructuredDict(
                {"type": "object", "additionalProperties": True}
            )
        has_client_tools = any(
            isinstance(t, FunctionTool) for t in run.request.tools
        )
        if structured is not None:
            return [structured, DeferredToolRequests]
        if has_client_tools:
            return [str, DeferredToolRequests]
        return None

    @staticmethod
    def _allowed_tool_names(choice: Any) -> set[str] | None:
        if isinstance(choice, ToolChoiceAllowed):
            return {t.get("name") for t in choice.tools if t.get("name")}
        if isinstance(choice, ToolChoiceFunction):
            return {choice.name}
        return None

    def _model_settings(self, run: AgentRun) -> ModelSettings | None:
        request = run.request
        settings: dict[str, Any] = {}
        if request.temperature is not None:
            settings["temperature"] = request.temperature
        if request.top_p is not None:
            settings["top_p"] = request.top_p
        if request.presence_penalty is not None:
            settings["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            settings["frequency_penalty"] = request.frequency_penalty
        if request.max_output_tokens is not None:
            settings["max_tokens"] = request.max_output_tokens
        if request.parallel_tool_calls is not None:
            settings["parallel_tool_calls"] = request.parallel_tool_calls

        choice = request.tool_choice
        if choice == "none":
            settings["tool_choice"] = "none"
        elif choice == "required":
            settings["tool_choice"] = "required"
        elif isinstance(choice, ToolChoiceFunction):
            settings["tool_choice"] = ToolOrOutput(function_tools=[choice.name])
        elif isinstance(choice, ToolChoiceAllowed):
            names = [t.get("name") for t in choice.tools if t.get("name")]
            if names:
                settings["tool_choice"] = ToolOrOutput(function_tools=names)

        if request.reasoning is not None and request.reasoning.effort is not None:
            level = _THINKING_LEVELS.get(request.reasoning.effort)
            if level is not None:
                settings["thinking"] = level

        if not settings:
            return None
        return ModelSettings(**settings)

    @staticmethod
    def _usage_delta(result: Any) -> UsageDelta:
        usage = result.usage() if callable(result.usage) else result.usage
        details = usage.details or {}
        return UsageDelta(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            total_tokens=(usage.input_tokens or 0) + (usage.output_tokens or 0),
            reasoning_tokens=details.get("reasoning_tokens", 0),
            cached_tokens=usage.cache_read_tokens or 0,
        )


class _EventTranslator:
    """Translates Pydantic AI stream events into adapter events."""

    def __init__(self, client_tool_names: set[str]) -> None:
        self.client_tool_names = client_tool_names
        self._open_calls: dict[str, CustomItem] = {}

    def translate(self, event: Any) -> list[AdapterEvent]:
        out: list[AdapterEvent] = []
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, TextPart):
                if part.content:
                    out.append(TextDelta(part.content))
            elif isinstance(part, ThinkingPart):
                if part.content or part.signature:
                    out.append(
                        ReasoningDelta(
                            delta=part.content or "",
                            encrypted_content=part.signature,
                        )
                    )
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, TextPartDelta):
                if delta.content_delta:
                    out.append(TextDelta(delta.content_delta))
            elif isinstance(delta, ThinkingPartDelta):
                if delta.content_delta or delta.signature_delta:
                    out.append(
                        ReasoningDelta(
                            delta=delta.content_delta or "",
                            encrypted_content=delta.signature_delta,
                        )
                    )
        elif isinstance(event, pai_messages.FunctionToolCallEvent):
            part = event.part
            if part.tool_name not in self.client_tool_names:
                item = CustomItem.model_validate(
                    {
                        "type": EXTENSION_FUNCTION_CALL,
                        "id": new_function_call_id(),
                        "status": "in_progress",
                        "name": part.tool_name,
                        "arguments": part.args_as_json_str(),
                        "output": None,
                    }
                )
                self._open_calls[part.tool_call_id] = item
                out.append(ItemAdded(item))
        elif isinstance(event, pai_messages.FunctionToolResultEvent):
            result_part = event.part
            opened = self._open_calls.pop(result_part.tool_call_id, None)
            if opened is not None:
                data = opened.model_dump()
                data["status"] = "completed"
                content = getattr(result_part, "content", None)
                if isinstance(result_part, ToolReturnPart):
                    data["output"] = (
                        content
                        if isinstance(content, str)
                        else json.dumps(content, default=str)
                    )
                else:  # RetryPromptPart and friends
                    data["output"] = json.dumps(
                        {"error": str(content)}, default=str
                    )
                out.append(ItemDone(CustomItem.model_validate(data)))
        return out

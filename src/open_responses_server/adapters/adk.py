"""Google ADK adapter.

Wraps a :class:`google.adk.agents.LlmAgent` (or any ``BaseAgent``) as an
Open Responses provider:

- Assistant text is streamed as ``output_text`` deltas (ADK ``StreamingMode.SSE``).
- Model "thought" parts (e.g. Gemini thought summaries) are surfaced as
  ``reasoning`` output items with streamed summary text; thought signatures
  are attached as ``encrypted_content`` when available.
- Tools owned by the ADK agent run *inside* the provider; each execution is
  surfaced as an ``adk:function_call`` extension item (a "receipt" per the
  Open Responses spec for internally-hosted tools).
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

from ..adapter import (
    AdapterError,
    AdapterEvent,
    AgentAdapter,
    AgentRun,
    ItemAdded,
    ItemDone,
    ReasoningDelta,
    StateUpdate,
    TextDelta,
    UsageDelta,
)
from ..models import (
    CustomItem,
    FunctionCallItem,
    FunctionCallOutputItem,
    FunctionTool,
    Item,
    MessageItem,
    ToolChoiceAllowed,
    ToolChoiceFunction,
    new_call_id,
    new_function_call_id,
)

EXTENSION_FUNCTION_CALL = "adk:function_call"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


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
        app_name: str = "open-responses-server",
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

        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=new_message,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
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
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    parts.append(types.Part(text=text))
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
        for item in items:
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
            text = item.text()
            if not text:
                return None
            if item.role == "assistant":
                return Event(
                    invocation_id=invocation_id,
                    author=agent_name,
                    content=types.Content(role="model", parts=[types.Part(text=text)]),
                )
            return Event(
                invocation_id=invocation_id,
                author="user",
                content=types.Content(role="user", parts=[types.Part(text=text)]),
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

        if not update:
            return self.agent
        return self.agent.clone(update=update)

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
        if request.max_output_tokens is not None:
            ensure().max_output_tokens = request.max_output_tokens

        tool_config = self._tool_config(run)
        if tool_config is not None:
            ensure().tool_config = tool_config

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
        self._open_calls: dict[str, CustomItem] = {}

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
                out.append(self._open_internal_call(event, fc, args))

        for fr in event.get_function_responses():
            out.append(self._close_internal_call(event, fr))

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

    def _open_internal_call(
        self, event: Event, fc: types.FunctionCall, args: str
    ) -> ItemAdded:
        item = CustomItem.model_validate(
            {
                "type": EXTENSION_FUNCTION_CALL,
                "id": new_function_call_id(),
                "status": "in_progress",
                "name": fc.name or "",
                "arguments": args,
                "agent": event.author,
                "output": None,
            }
        )
        if fc.id:
            self._open_calls[fc.id] = item
        return ItemAdded(item)

    def _close_internal_call(
        self, event: Event, fr: types.FunctionResponse
    ) -> ItemDone:
        opened = self._open_calls.pop(fr.id or "", None)
        data = opened.model_dump() if opened is not None else {
            "type": EXTENSION_FUNCTION_CALL,
            "id": new_function_call_id(),
            "name": fr.name or "",
            "arguments": "{}",
            "agent": event.author,
        }
        data["status"] = "completed"
        data["output"] = json.dumps(fr.response) if fr.response is not None else None
        return ItemDone(CustomItem.model_validate(data))

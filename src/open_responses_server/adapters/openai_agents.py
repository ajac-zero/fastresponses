"""OpenAI Agents SDK adapter: expose an ``agents.Agent`` as an Open
Responses provider.

The Agents SDK already speaks Responses-API-shaped items, so most of the
mapping is direct:

- **Client-declared tools** (``tools`` in the request) become
  ``FunctionTool``s with ``needs_approval=True``. When the model calls one,
  the run pauses with an interruption and the adapter surfaces a standard
  ``function_call`` item (keeping the SDK's native ``call_id``), yielding
  control. A follow-up request with ``function_call_output`` approves the
  interruption and resumes from the serialized ``RunState``; the tool's
  ``on_invoke_tool`` hands back the client-provided output.
- **Agent-internal tools** run inside the SDK and are surfaced as
  ``openai_agents:function_call`` extension items (receipts).
- **Continuation** stores ``result.to_input_list()`` (already Responses
  items) plus, when interrupted, the serialized ``RunState``.
- ``allowed_tools`` is enforced hard: internal tools outside the allowed set
  get their ``on_invoke_tool`` wrapped to refuse execution.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator
from typing import Any

try:
    from agents import Agent as OpenAIAgent
    from agents import FunctionTool as SDKFunctionTool
    from agents import ModelSettings as SDKModelSettings
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded, UserError
    from agents.items import ToolCallItem, ToolCallOutputItem
    from agents.run_state import RunState
    from agents.stream_events import (
        RawResponsesStreamEvent,
        RunItemStreamEvent,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The OpenAI Agents SDK adapter requires the 'openai-agents' extra: "
        "pip install 'open-responses-server[openai-agents]'"
    ) from exc

from ..adapter import (  # noqa: I001
    AdapterError,
    AdapterEvent,
    AgentAdapter,
    AgentRun,
    Incomplete,
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
    Item,
    ToolChoiceAllowed,
    ToolChoiceFunction,
    new_function_call_id,
)

EXTENSION_FUNCTION_CALL = "openai_agents:function_call"


def _client_tool(tool: FunctionTool, outputs: dict[str, str]) -> SDKFunctionTool:
    """A needs-approval FunctionTool whose execution hands back the
    client-provided output for its call_id (filled in on resume)."""

    async def on_invoke(ctx: Any, args: str) -> str:
        call_id = getattr(ctx, "tool_call_id", None)
        if call_id in outputs:
            return outputs[call_id]
        return ""  # pragma: no cover - approval without output

    return SDKFunctionTool(
        name=tool.name,
        description=tool.description or "",
        params_json_schema=tool.parameters or {"type": "object", "properties": {}},
        on_invoke_tool=on_invoke,
        strict_json_schema=bool(tool.strict),
        needs_approval=True,
    )


def _guard_tool(tool: SDKFunctionTool, allowed: frozenset[str]) -> SDKFunctionTool:
    if tool.name in allowed:
        return tool

    async def refuse(ctx: Any, args: str) -> str:
        return json.dumps(
            {
                "error": f"Tool '{tool.name}' is not allowed for this request "
                "(restricted by allowed_tools)."
            }
        )

    return dataclasses.replace(tool, on_invoke_tool=refuse)


class OpenAIAgentsAdapter(AgentAdapter):
    """Expose an OpenAI Agents SDK agent as an Open Responses provider."""

    name = "openai_agents"

    def __init__(
        self,
        agent: OpenAIAgent,
        *,
        model_name: str | None = None,
        max_turns: int = 25,
    ) -> None:
        self.agent = agent
        self.max_turns = max_turns
        model = getattr(agent, "model", None)
        inferred = model if isinstance(model, str) and model else None
        self.default_model = model_name or (
            f"openai-agents/{agent.name}/{inferred}"
            if inferred
            else f"openai-agents/{agent.name}"
        )

    # ------------------------------------------------------------------
    # AgentAdapter
    # ------------------------------------------------------------------

    async def run(self, run: AgentRun) -> AsyncIterator[AdapterEvent]:
        request = run.request
        state = dict(run.previous_state or {})

        outputs: dict[str, str] = {}
        client_tools = [
            _client_tool(t, outputs)
            for t in request.tools
            if isinstance(t, FunctionTool)
        ]
        client_tool_names = {t.name for t in client_tools}
        agent = self._configure_agent(run, client_tools)

        continuing = bool(state.get("items") or state.get("run_state"))
        items = run.new_items if continuing else run.context_items
        trailing_outputs = self._trailing_outputs(items)
        message_items = items[: len(items) - len(trailing_outputs)]

        run_input = await self._run_input(
            agent, state, message_items, trailing_outputs, outputs
        )

        translator = _EventTranslator(client_tool_names, request.max_tool_calls)
        result = Runner.run_streamed(agent, run_input, max_turns=self.max_turns)

        try:
            async for event in result.stream_events():
                for adapter_event in translator.translate(event):
                    yield adapter_event
                if translator.exceeded:
                    result.cancel()
                    break
        except MaxTurnsExceeded as exc:
            raise AdapterError(str(exc), type="server_error") from exc
        except UserError as exc:
            raise AdapterError(
                str(exc), type="invalid_request", code="invalid_value", param="input"
            ) from exc
        except AgentsException as exc:
            raise AdapterError(str(exc), type="server_error") from exc

        new_state: dict[str, Any] = {}
        pending: dict[str, str] = {}
        if result.interruptions and not translator.exceeded:
            for interruption in result.interruptions:
                raw = interruption.raw_item
                call_id = getattr(raw, "call_id", None) or new_function_call_id()
                pending[call_id] = interruption.tool_name or ""
                yield ItemDone(
                    FunctionCallItem(
                        id=new_function_call_id(),
                        call_id=call_id,
                        name=interruption.tool_name or "",
                        arguments=getattr(raw, "arguments", None) or "{}",
                        status="completed",
                    )
                )
            new_state["run_state"] = json.dumps(result.to_state().to_json())
        else:
            new_state["items"] = json.dumps(result.to_input_list(), default=str)

        yield translator.usage()
        new_state["pending"] = pending
        yield StateUpdate(new_state)
        if translator.exceeded:
            yield Incomplete("max_tool_calls")

    # ------------------------------------------------------------------
    # Input construction
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

    async def _run_input(
        self,
        agent: OpenAIAgent,
        state: dict[str, Any],
        message_items: list[Item],
        trailing_outputs: list[FunctionCallOutputItem],
        outputs: dict[str, str],
    ) -> Any:
        if trailing_outputs and state.get("run_state"):
            pending: dict[str, str] = state.get("pending") or {}
            try:
                run_state = RunState.from_json(agent, json.loads(state["run_state"]))
                if hasattr(run_state, "__await__"):
                    run_state = await run_state
            except Exception as exc:
                raise AdapterError(
                    "Stored conversation state could not be restored.",
                    type="invalid_request",
                    code="previous_response_not_found",
                    param="previous_response_id",
                ) from exc
            interruptions = {
                getattr(i.raw_item, "call_id", None): i
                for i in run_state.get_interruptions()
            }
            for output in trailing_outputs:
                interruption = interruptions.get(output.call_id)
                if interruption is None or output.call_id not in pending:
                    raise AdapterError(
                        f"No tool call with call_id '{output.call_id}' is "
                        "awaiting output in this conversation.",
                        type="invalid_request",
                        code="invalid_value",
                        param="input",
                    )
                outputs[output.call_id] = self._output_str(output.output)
                run_state.approve(interruption)
            return run_state

        if trailing_outputs and not state.get("run_state"):
            # Stateless replay: outputs must pair with function_call items
            # already present in the transcript.
            transcript = self._items_to_input(message_items, state)
            known = {
                entry.get("call_id")
                for entry in transcript
                if entry.get("type") == "function_call"
            }
            for output in trailing_outputs:
                if output.call_id not in known:
                    raise AdapterError(
                        f"No tool call with call_id '{output.call_id}' is "
                        "awaiting output in this conversation.",
                        type="invalid_request",
                        code="invalid_value",
                        param="input",
                    )
            return self._items_to_input(
                [*message_items, *trailing_outputs], state
            )

        return self._items_to_input(message_items, state)

    def _items_to_input(
        self, items: list[Item], state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        base: list[dict[str, Any]] = []
        if state.get("items"):
            base = json.loads(state["items"])

        expanded: list[Item] = []
        for item in items:
            if isinstance(item, CompactionItem):
                expanded.extend(expand_compaction_item(item))
            else:
                expanded.append(item)

        for item in expanded:
            if isinstance(item, CustomItem) and item.type == EXTENSION_FUNCTION_CALL:
                extra = item.model_extra or {}
                call_id = f"internal_{item.id or new_function_call_id()}"
                base.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": extra.get("name") or "tool",
                        "arguments": extra.get("arguments") or "{}",
                    }
                )
                base.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": extra.get("output") or "",
                    }
                )
            elif isinstance(item, CustomItem):
                continue  # other providers' extension items
            else:
                data = item.model_dump(exclude_none=True)
                data.pop("id", None)
                data.pop("status", None)
                base.append(data)
        return base

    @staticmethod
    def _output_str(output: str | list[Any]) -> str:
        if isinstance(output, str):
            return output
        return "\n".join(getattr(p, "text", None) or str(p) for p in output)

    # ------------------------------------------------------------------
    # Agent configuration
    # ------------------------------------------------------------------

    def _configure_agent(
        self, run: AgentRun, client_tools: list[SDKFunctionTool]
    ) -> OpenAIAgent:
        request = run.request
        update: dict[str, Any] = {}

        tools = [*self.agent.tools, *client_tools]
        allowed = self._allowed_tool_names(request.tool_choice)
        if allowed is not None:
            tools = [
                _guard_tool(t, frozenset(allowed))
                if isinstance(t, SDKFunctionTool) and not t.needs_approval
                else t
                for t in tools
            ]
        update["tools"] = tools

        if request.instructions is not None:
            update["instructions"] = request.instructions

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
        if request.top_logprobs:
            settings["top_logprobs"] = request.top_logprobs
        choice = request.tool_choice
        if choice in ("none", "required"):
            settings["tool_choice"] = choice
        elif isinstance(choice, ToolChoiceFunction):
            settings["tool_choice"] = choice.name
        if settings:
            base = self.agent.model_settings or SDKModelSettings()
            update["model_settings"] = dataclasses.replace(base, **settings)

        return self.agent.clone(**update)

    @staticmethod
    def _allowed_tool_names(choice: Any) -> set[str] | None:
        if isinstance(choice, ToolChoiceAllowed):
            return {t.get("name") for t in choice.tools if t.get("name")}
        if isinstance(choice, ToolChoiceFunction):
            return {choice.name}
        return None


class _EventTranslator:
    """Translates Agents SDK stream events into adapter events."""

    def __init__(self, client_tool_names: set[str], max_tool_calls: int | None) -> None:
        self.client_tool_names = client_tool_names
        self.max_tool_calls = max_tool_calls
        self.tool_calls = 0
        self.exceeded = False
        self._open_calls: dict[str, dict[str, Any]] = {}
        self._usage = UsageDelta()

    def translate(self, event: Any) -> list[AdapterEvent]:
        if isinstance(event, RawResponsesStreamEvent):
            return self._on_raw(event.data)
        if isinstance(event, RunItemStreamEvent):
            return self._on_item(event)
        return []

    def _on_raw(self, data: Any) -> list[AdapterEvent]:
        kind = getattr(data, "type", "")
        if kind == "response.output_text.delta":
            return [TextDelta(data.delta)] if data.delta else []
        if kind in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            return [ReasoningDelta(delta=data.delta)] if data.delta else []
        if kind == "response.completed":
            usage = getattr(data.response, "usage", None)
            if usage is not None:
                self._usage.input_tokens += usage.input_tokens or 0
                self._usage.output_tokens += usage.output_tokens or 0
                self._usage.total_tokens += usage.total_tokens or 0
                details = getattr(usage, "output_tokens_details", None)
                if details is not None:
                    self._usage.reasoning_tokens += (
                        getattr(details, "reasoning_tokens", 0) or 0
                    )
                cached = getattr(usage, "input_tokens_details", None)
                if cached is not None:
                    self._usage.cached_tokens += (
                        getattr(cached, "cached_tokens", 0) or 0
                    )
        return []

    def _on_item(self, event: RunItemStreamEvent) -> list[AdapterEvent]:
        item = event.item
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            name = getattr(raw, "name", "") or ""
            if name in self.client_tool_names:
                return []  # surfaced via the interruption instead
            if self.max_tool_calls is not None:
                if self.tool_calls + 1 > self.max_tool_calls:
                    self.exceeded = True
                    return []
            self.tool_calls += 1
            call_id = getattr(raw, "call_id", "") or ""
            self._open_calls[call_id] = {
                "type": EXTENSION_FUNCTION_CALL,
                "id": new_function_call_id(),
                "name": name,
                "arguments": getattr(raw, "arguments", None) or "{}",
            }
            return []
        if isinstance(item, ToolCallOutputItem):
            raw = item.raw_item
            call_id = (
                raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", "")
            ) or ""
            opened = self._open_calls.pop(call_id, None)
            if opened is None:
                return []
            opened["status"] = "completed"
            opened["output"] = (
                item.output if isinstance(item.output, str) else json.dumps(item.output, default=str)
            )
            return [ItemDone(CustomItem.model_validate(opened))]
        return []

    def usage(self) -> UsageDelta:
        return self._usage

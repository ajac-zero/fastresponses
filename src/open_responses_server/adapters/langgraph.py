"""LangGraph adapter: expose a compiled LangGraph graph as an Open Responses
provider.

Works with any compiled graph that follows the ``MessagesState`` convention
(a ``messages`` key holding LangChain messages), e.g. graphs built with
``langchain.agents.create_agent`` / ``create_react_agent``.

Mapping notes:

- **Client-declared tools** (``tools`` in the request) require a *graph
  factory*: pass a callable ``(client_tools) -> compiled graph`` instead of a
  compiled graph, and register the provided tools (see
  :func:`interrupt_tool`, which builds a LangChain tool that calls
  ``langgraph.types.interrupt``). When the graph hits such an interrupt, the
  adapter surfaces a ``function_call`` item and yields control; a follow-up
  request with ``function_call_output`` resumes via
  ``Command(resume={interrupt_id: output})``.
- **Human-in-the-loop interrupts** whose value is a dict with ``name``/
  ``args`` keys surface as ``function_call`` items for that tool name; any
  other interrupt value surfaces as a ``function_call`` named
  ``human_input`` with the value JSON-encoded in ``arguments``.
- **Graph-internal tools** surface as ``langgraph:function_call`` extension
  items (receipts).
- **Continuation** uses the graph's checkpointer (``thread_id`` per
  conversation). If the graph has no checkpointer, the adapter attaches a
  shared ``InMemorySaver`` (required for interrupt/resume). Fresh
  conversations replay the full item context into ``messages``.
- ``allowed_tools`` requests are rejected: an arbitrary compiled graph offers
  no hook to suppress tool execution, and the spec requires hard enforcement,
  so failing loudly is the only honest behavior.
"""

from __future__ import annotations

import json
import mimetypes
import uuid
from collections.abc import AsyncIterator
from typing import Any, Callable

try:
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.tools import StructuredTool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.errors import GraphRecursionError
    from langgraph.types import Command, Interrupt, interrupt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The LangGraph adapter requires the 'langgraph' extra: "
        "pip install 'open-responses-server[langgraph]'"
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
    InputFile,
    InputImage,
    InputText,
    Item,
    MessageItem,
    ToolChoiceAllowed,
    ToolChoiceFunction,
    new_function_call_id,
)

EXTENSION_FUNCTION_CALL = "langgraph:function_call"
GENERIC_INTERRUPT_TOOL = "human_input"


def interrupt_tool(tool: FunctionTool) -> StructuredTool:
    """Build a LangChain tool for a client-declared function tool.

    The tool pauses the graph via ``interrupt`` with a ``{"name", "args"}``
    payload; the adapter surfaces it as an Open Responses ``function_call``
    and resumes the graph with the client-provided output.
    """

    def _run(**kwargs: Any) -> Any:
        return interrupt({"name": tool.name, "args": kwargs})

    return StructuredTool.from_function(
        func=_run,
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.parameters
        or {"type": "object", "properties": {}},
    )


def _decode_output(output: str | list[Any]) -> str:
    if isinstance(output, str):
        return output
    return "\n".join(getattr(p, "text", None) or str(p) for p in output)


def _guess_mime(name: str | None, default: str = "application/octet-stream") -> str:
    if name:
        guessed, _ = mimetypes.guess_type(name)
        if guessed:
            return guessed
    return default


def _user_content(message: MessageItem) -> str | list[dict[str, Any]]:
    if isinstance(message.content, str):
        return message.content
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, InputText):
            if part.text:
                blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, InputImage):
            url = part.image_url
            if not url:
                continue
            if url.startswith("data:"):
                header, _, data = url.partition(",")
                mime = header.removeprefix("data:").split(";")[0] or "image/png"
                blocks.append({"type": "image", "base64": data, "mime_type": mime})
            else:
                blocks.append({"type": "image", "url": url})
        elif isinstance(part, InputFile):
            if part.file_data:
                blocks.append(
                    {
                        "type": "file",
                        "base64": part.file_data,
                        "mime_type": _guess_mime(part.filename),
                    }
                )
            elif part.file_url:
                blocks.append({"type": "file", "url": part.file_url})
        else:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
    return blocks or ""


class LangGraphAdapter(AgentAdapter):
    """Expose a LangGraph graph as an Open Responses provider."""

    name = "langgraph"

    def __init__(
        self,
        graph: Any | Callable[[list[FunctionTool]], Any],
        *,
        model_name: str | None = None,
    ) -> None:
        """``graph`` is a compiled graph, or a factory taking the request's
        client-declared tools (as LangChain tools built by
        :func:`interrupt_tool`) and returning a compiled graph."""
        if callable(graph) and not hasattr(graph, "astream"):
            self._factory = graph
            self._graph = None
        else:
            self._factory = None
            self._graph = graph
        self._shared_checkpointer = InMemorySaver()
        inferred = getattr(self._graph, "name", None) or "graph"
        self.default_model = model_name or f"langgraph/{inferred}"

    # ------------------------------------------------------------------
    # AgentAdapter
    # ------------------------------------------------------------------

    async def run(self, run: AgentRun) -> AsyncIterator[AdapterEvent]:
        request = run.request
        if isinstance(request.tool_choice, (ToolChoiceAllowed, ToolChoiceFunction)):
            raise AdapterError(
                "The LangGraph adapter cannot enforce allowed_tools/forced "
                "tool choice on an arbitrary compiled graph.",
                type="invalid_request",
                code="unsupported_parameter",
                param="tool_choice",
            )

        client_tools = [t for t in request.tools if isinstance(t, FunctionTool)]
        graph = self._resolve_graph(client_tools)

        state = dict(run.previous_state or {})
        pending: dict[str, str] = dict(state.get("pending") or {})
        thread_id = state.get("thread_id") or f"or-{uuid.uuid4().hex}"
        continuing = bool(state.get("thread_id"))

        items = run.new_items if continuing else run.context_items
        trailing_outputs = self._trailing_outputs(items)
        message_items = items[: len(items) - len(trailing_outputs)]

        graph_input = self._graph_input(
            run, message_items, trailing_outputs, pending, continuing
        )

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
        }
        translator = _EventTranslator(request.max_tool_calls)
        interrupts: list[Interrupt] = []

        try:
            async for mode, chunk in graph.astream(
                graph_input, config, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    for event in translator.on_message_chunk(chunk):
                        yield event
                else:
                    for event in translator.on_update(chunk, interrupts):
                        yield event
                if translator.exceeded:
                    break
        except GraphRecursionError as exc:
            raise AdapterError(
                f"The graph exceeded its recursion limit: {exc}",
                type="server_error",
            ) from exc

        new_pending: dict[str, str] = {}
        for intr in interrupts:
            call_id, item = self._interrupt_to_call(intr)
            new_pending[call_id] = intr.id
            yield ItemDone(item)

        yield translator.usage()
        yield StateUpdate({"thread_id": thread_id, "pending": new_pending})
        if translator.exceeded:
            yield Incomplete("max_tool_calls")

    # ------------------------------------------------------------------
    # Graph resolution and input construction
    # ------------------------------------------------------------------

    def _resolve_graph(self, client_tools: list[FunctionTool]) -> Any:
        if self._factory is not None:
            graph = self._factory([interrupt_tool(t) for t in client_tools])
            # Factory graphs are rebuilt per request, so any checkpointer
            # compiled into them would lose thread state between turns. The
            # adapter owns persistence: always attach the shared saver.
            try:
                graph.checkpointer = self._shared_checkpointer
            except (AttributeError, TypeError):  # pragma: no cover
                pass
            return graph

        if client_tools:
            raise AdapterError(
                "This LangGraph adapter serves a fixed compiled graph; "
                "request-declared tools require constructing the adapter "
                "with a graph factory (see LangGraphAdapter docs).",
                type="invalid_request",
                code="unsupported_parameter",
                param="tools",
            )
        graph = self._graph
        if getattr(graph, "checkpointer", None) is None:
            # interrupt/resume and continuation need a checkpointer.
            try:
                graph.checkpointer = self._shared_checkpointer
            except (AttributeError, TypeError):  # pragma: no cover
                pass
        return graph

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

    def _graph_input(
        self,
        run: AgentRun,
        message_items: list[Item],
        trailing_outputs: list[FunctionCallOutputItem],
        pending: dict[str, str],
        continuing: bool,
    ) -> Any:
        if trailing_outputs:
            resume: dict[str, Any] = {}
            for output in trailing_outputs:
                interrupt_id = pending.get(output.call_id)
                if interrupt_id is None:
                    raise AdapterError(
                        f"No tool call with call_id '{output.call_id}' is "
                        "awaiting output in this conversation.",
                        type="invalid_request",
                        code="invalid_value",
                        param="input",
                    )
                resume[interrupt_id] = _decode_output(output.output)
            return Command(resume=resume)

        messages = self._items_to_messages(message_items)
        if not continuing and run.request.instructions:
            messages.insert(0, SystemMessage(content=run.request.instructions))
        return {"messages": messages}

    def _items_to_messages(self, items: list[Item]) -> list[BaseMessage]:
        expanded: list[Item] = []
        for item in items:
            if isinstance(item, CompactionItem):
                expanded.extend(expand_compaction_item(item))
            else:
                expanded.append(item)

        messages: list[BaseMessage] = []
        for item in expanded:
            if isinstance(item, MessageItem):
                if item.role == "assistant":
                    text = item.text()
                    if text:
                        messages.append(AIMessage(content=text))
                elif item.role in ("system", "developer"):
                    text = item.text()
                    if text:
                        messages.append(SystemMessage(content=text))
                else:
                    messages.append(HumanMessage(content=_user_content(item)))
            elif isinstance(item, FunctionCallItem):
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                messages.append(
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": item.name, "args": args, "id": item.call_id}
                        ],
                    )
                )
            elif isinstance(item, FunctionCallOutputItem):
                messages.append(
                    ToolMessage(
                        content=_decode_output(item.output),
                        tool_call_id=item.call_id,
                    )
                )
            elif isinstance(item, CustomItem) and item.type == EXTENSION_FUNCTION_CALL:
                extra = item.model_extra or {}
                call_id = f"internal_{item.id or new_function_call_id()}"
                try:
                    args = json.loads(extra.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                messages.append(
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": extra.get("name") or "tool",
                                "args": args,
                                "id": call_id,
                            }
                        ],
                    )
                )
                messages.append(
                    ToolMessage(
                        content=extra.get("output") or "", tool_call_id=call_id
                    )
                )
        return messages

    @staticmethod
    def _interrupt_to_call(intr: Interrupt) -> tuple[str, FunctionCallItem]:
        value = intr.value
        if isinstance(value, dict) and "name" in value:
            name = str(value["name"])
            arguments = json.dumps(value.get("args") or {})
        else:
            name = GENERIC_INTERRUPT_TOOL
            arguments = json.dumps({"value": value}, default=str)
        call_id = intr.id
        return call_id, FunctionCallItem(
            id=new_function_call_id(),
            call_id=call_id,
            name=name,
            arguments=arguments,
            status="completed",
        )


class _EventTranslator:
    """Translates LangGraph stream chunks into adapter events."""

    def __init__(self, max_tool_calls: int | None) -> None:
        self.max_tool_calls = max_tool_calls
        self.tool_calls = 0
        self.exceeded = False
        self._open_calls: dict[str, CustomItem] = {}
        self._usage = UsageDelta()

    # -- stream_mode="messages": token stream ---------------------------

    def on_message_chunk(self, chunk: tuple[Any, dict]) -> list[AdapterEvent]:
        message, _meta = chunk
        if not isinstance(message, AIMessageChunk):
            return []
        out: list[AdapterEvent] = []
        usage = getattr(message, "usage_metadata", None)
        if usage:
            self._usage.input_tokens += usage.get("input_tokens", 0)
            self._usage.output_tokens += usage.get("output_tokens", 0)
            self._usage.total_tokens += usage.get("total_tokens", 0)

        content = message.content
        if isinstance(content, str):
            if content:
                out.append(TextDelta(content))
            return out
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and block.get("text"):
                out.append(TextDelta(block["text"]))
            elif kind == "reasoning":
                delta = block.get("reasoning") or block.get("text") or ""
                if delta:
                    out.append(ReasoningDelta(delta=delta))
        return out

    # -- stream_mode="updates": node completions ------------------------

    def on_update(
        self, update: dict[str, Any], interrupts: list[Interrupt]
    ) -> list[AdapterEvent]:
        out: list[AdapterEvent] = []
        for node, payload in update.items():
            if node == "__interrupt__":
                interrupts.extend(payload)
                continue
            if not isinstance(payload, dict):
                continue
            for message in payload.get("messages") or []:
                if isinstance(message, AIMessage) and message.tool_calls:
                    out.extend(self._on_tool_calls(message))
                elif isinstance(message, ToolMessage):
                    out.extend(self._on_tool_result(message))
        return out

    def _on_tool_calls(self, message: AIMessage) -> list[AdapterEvent]:
        """Record pending calls. Receipts are emitted atomically when the
        matching ToolMessage arrives: at this point we cannot know which
        calls will pause the graph via interrupt (those surface as
        ``function_call`` items instead of receipts)."""
        out: list[AdapterEvent] = []
        if self.max_tool_calls is not None:
            if self.tool_calls + len(message.tool_calls) > self.max_tool_calls:
                self.exceeded = True
                return out
        self.tool_calls += len(message.tool_calls)
        for call in message.tool_calls:
            item = CustomItem.model_validate(
                {
                    "type": EXTENSION_FUNCTION_CALL,
                    "id": new_function_call_id(),
                    "status": "in_progress",
                    "name": call["name"],
                    "arguments": json.dumps(call.get("args") or {}),
                    "output": None,
                }
            )
            self._open_calls[call.get("id") or ""] = item
        return out

    def _on_tool_result(self, message: ToolMessage) -> list[AdapterEvent]:
        opened = self._open_calls.pop(message.tool_call_id, None)
        if opened is None:
            return []
        data = opened.model_dump()
        data["status"] = "completed"
        content = message.content
        data["output"] = (
            content if isinstance(content, str) else json.dumps(content, default=str)
        )
        return [ItemDone(CustomItem.model_validate(data))]

    def usage(self) -> UsageDelta:
        return self._usage

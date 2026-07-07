"""Pydantic models for the Open Responses specification.

Covers the core surface of https://www.openresponses.org/specification :
request bodies, items, content parts, the response object, and the
semantic streaming events.

The models are intentionally permissive (``extra="allow"``) so that
provider-specific extensions (e.g. ``adk:function_call`` items) round-trip
without loss.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def new_response_id() -> str:
    return _id("resp")


def new_message_id() -> str:
    return _id("msg")


def new_function_call_id() -> str:
    return _id("fc")


def new_call_id() -> str:
    return _id("call")


# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------


class InputText(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["input_text"] = "input_text"
    text: str


class InputImage(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    file_id: str | None = None
    detail: str | None = None


class InputFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["input_file"] = "input_file"
    file_id: str | None = None
    filename: str | None = None
    file_data: str | None = None
    file_url: str | None = None


class OutputText(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)


class Refusal(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["refusal"] = "refusal"
    refusal: str


class SummaryText(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["summary_text"] = "summary_text"
    text: str


UserContent = Union[InputText, InputImage, InputFile]
ModelContent = Union[OutputText, Refusal]

ContentPart = Annotated[
    Union[InputText, InputImage, InputFile, OutputText, Refusal],
    Field(union_mode="left_to_right"),
]


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

ItemStatus = Literal["in_progress", "completed", "incomplete", "failed"]


class MessageItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["message"] = "message"
    id: str | None = None
    role: Literal["user", "assistant", "system", "developer"]
    status: ItemStatus | None = None
    content: str | list[ContentPart] = Field(default_factory=list)

    def text(self) -> str:
        """Concatenated text of all text-bearing content parts."""
        if isinstance(self.content, str):
            return self.content
        chunks: list[str] = []
        for part in self.content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks)


class FunctionCallItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function_call"] = "function_call"
    id: str | None = None
    call_id: str
    name: str
    arguments: str = ""
    status: ItemStatus | None = None


class FunctionCallOutputItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function_call_output"] = "function_call_output"
    id: str | None = None
    call_id: str
    output: str | list[Any]
    status: ItemStatus | None = None


class ReasoningItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["reasoning"] = "reasoning"
    id: str | None = None
    status: ItemStatus | None = None
    summary: list[SummaryText] = Field(default_factory=list)
    content: list[Any] | None = None
    encrypted_content: str | None = None


class ItemReference(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["item_reference"] = "item_reference"
    id: str


class CustomItem(BaseModel):
    """Fallback for provider-specific extension items (e.g. ``adk:function_call``)."""

    model_config = ConfigDict(extra="allow")

    type: str
    id: str | None = None
    status: ItemStatus | None = None


Item = Annotated[
    Union[
        MessageItem,
        FunctionCallItem,
        FunctionCallOutputItem,
        ReasoningItem,
        ItemReference,
        CustomItem,
    ],
    Field(union_mode="left_to_right"),
]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class FunctionTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class CustomTool(BaseModel):
    """Fallback for hosted / provider-specific tool definitions."""

    model_config = ConfigDict(extra="allow")

    type: str


Tool = Annotated[Union[FunctionTool, CustomTool], Field(union_mode="left_to_right")]


class ToolChoiceFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    name: str


class ToolChoiceAllowed(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["allowed_tools"] = "allowed_tools"
    mode: Literal["auto", "required"] = "auto"
    tools: list[dict[str, Any]] = Field(default_factory=list)


ToolChoice = Union[
    Literal["auto", "required", "none"], ToolChoiceFunction, ToolChoiceAllowed
]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str | list[Item] = Field(default_factory=list)
    instructions: str | None = None
    previous_response_id: str | None = None
    stream: bool | None = False
    store: bool | None = True
    background: bool | None = False
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: ToolChoice = "auto"
    parallel_tool_calls: bool | None = True
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    truncation: Literal["auto", "disabled"] | None = None
    metadata: dict[str, str] | None = None
    user: str | None = None
    service_tier: str | None = None

    def input_items(self) -> list[Item]:
        """Normalize ``input`` into a list of items."""
        if isinstance(self.input, str):
            return [MessageItem(role="user", content=[InputText(text=self.input)])]
        return list(self.input)

    def function_tools(self) -> list[FunctionTool]:
        return [t for t in self.tools if isinstance(t, FunctionTool)]


# ---------------------------------------------------------------------------
# Response object
# ---------------------------------------------------------------------------

ResponseStatus = Literal[
    "queued", "in_progress", "completed", "incomplete", "failed", "cancelled"
]


class InputTokensDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    reasoning_tokens: int = 0


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
    output_tokens_details: OutputTokensDetails = Field(
        default_factory=OutputTokensDetails
    )


class ResponseError(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str | None = None
    message: str = ""


class IncompleteDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    reason: str | None = None


class Response(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=new_response_id)
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: ResponseStatus = "in_progress"
    model: str | None = None
    output: list[Item] = Field(default_factory=list)
    error: ResponseError | None = None
    incomplete_details: IncompleteDetails | None = None
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool | None = True
    background: bool | None = False
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: ToolChoice = "auto"
    parallel_tool_calls: bool | None = True
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    truncation: Literal["auto", "disabled"] | None = None
    metadata: dict[str, str] | None = None
    usage: Usage | None = None

    @property
    def output_text(self) -> str:
        """Concatenated text of all assistant message output items."""
        chunks: list[str] = []
        for item in self.output:
            if isinstance(item, MessageItem) and item.role == "assistant":
                chunks.append(item.text())
        return "".join(chunks)


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


class ResponseCreatedEvent(BaseModel):
    type: Literal["response.created"] = "response.created"
    sequence_number: int = 0
    response: Response


class ResponseInProgressEvent(BaseModel):
    type: Literal["response.in_progress"] = "response.in_progress"
    sequence_number: int = 0
    response: Response


class ResponseCompletedEvent(BaseModel):
    type: Literal["response.completed"] = "response.completed"
    sequence_number: int = 0
    response: Response


class ResponseIncompleteEvent(BaseModel):
    type: Literal["response.incomplete"] = "response.incomplete"
    sequence_number: int = 0
    response: Response


class ResponseFailedEvent(BaseModel):
    type: Literal["response.failed"] = "response.failed"
    sequence_number: int = 0
    response: Response


class OutputItemAddedEvent(BaseModel):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    sequence_number: int = 0
    output_index: int
    item: Item


class OutputItemDoneEvent(BaseModel):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    sequence_number: int = 0
    output_index: int
    item: Item


class ContentPartAddedEvent(BaseModel):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    sequence_number: int = 0
    item_id: str
    output_index: int
    content_index: int
    part: ContentPart


class ContentPartDoneEvent(BaseModel):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    sequence_number: int = 0
    item_id: str
    output_index: int
    content_index: int
    part: ContentPart


class OutputTextDeltaEvent(BaseModel):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    sequence_number: int = 0
    item_id: str
    output_index: int
    content_index: int
    delta: str
    logprobs: list[Any] = Field(default_factory=list)


class OutputTextDoneEvent(BaseModel):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    sequence_number: int = 0
    item_id: str
    output_index: int
    content_index: int
    text: str
    logprobs: list[Any] = Field(default_factory=list)


class FunctionCallArgumentsDeltaEvent(BaseModel):
    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    sequence_number: int = 0
    item_id: str
    output_index: int
    delta: str


class FunctionCallArgumentsDoneEvent(BaseModel):
    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    sequence_number: int = 0
    item_id: str
    output_index: int
    arguments: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    sequence_number: int = 0
    code: str | None = None
    message: str = ""
    param: str | None = None


StreamEvent = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponseFailedEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    ErrorEvent,
]


# ---------------------------------------------------------------------------
# Error envelope (non-streaming HTTP errors)
# ---------------------------------------------------------------------------

ErrorType = Literal[
    "server_error", "invalid_request", "not_found", "model_error", "too_many_requests"
]

ERROR_STATUS_CODES: dict[str, int] = {
    "server_error": 500,
    "invalid_request": 400,
    "not_found": 404,
    "model_error": 500,
    "too_many_requests": 429,
}


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "server_error"
    code: str | None = None
    message: str = ""
    param: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody

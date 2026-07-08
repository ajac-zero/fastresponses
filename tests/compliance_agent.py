"""Deterministic ADK agent used to run the Open Responses compliance suite.

The backing "model" is rule-based (no network, no API key):

- If the request declares function tools, it calls the first one with dummy
  arguments derived from the tool's JSON schema.
- Otherwise it streams a short text answer.

This exercises the full stack the compliance suite validates: the ADK
adapter, the engine, and the HTTP/SSE server.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

MODEL_NAME = "compliance-model"


def _dummy_args(declaration: types.FunctionDeclaration) -> dict[str, str]:
    schema = declaration.parameters_json_schema or {}
    if not isinstance(schema, dict):
        return {}
    required = schema.get("required") or []
    return {name: "San Francisco, CA" for name in required}


class RuleBasedLlm(BaseLlm):
    model: str = MODEL_NAME

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=7, candidates_token_count=5, total_token_count=12
        )

        declarations: list[types.FunctionDeclaration] = []
        for tool in llm_request.config.tools or []:
            declarations.extend(getattr(tool, "function_declarations", None) or [])
        if declarations:
            declaration = declarations[0]
            yield LlmResponse(
                partial=False,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name=declaration.name,
                                args=_dummy_args(declaration),
                            )
                        )
                    ],
                ),
                usage_metadata=usage,
            )
            return

        chunks = ["Hello from the ", "compliance agent."]
        for chunk in chunks:
            yield LlmResponse(
                partial=True,
                content=types.Content(role="model", parts=[types.Part(text=chunk)]),
            )
        yield LlmResponse(
            partial=False,
            content=types.Content(
                role="model", parts=[types.Part(text="".join(chunks))]
            ),
            usage_metadata=usage,
        )


def create_agent() -> Agent:
    return Agent(
        name="compliance_agent",
        model=RuleBasedLlm(),
        instruction="Answer deterministically.",
    )


def create_pydantic_ai_agent():
    """Deterministic Pydantic AI agent with the same rule-based behavior."""
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models.function import (
        AgentInfo,
        DeltaToolCall,
        FunctionModel,
    )

    async def rule_based(messages, info: AgentInfo):
        pending = any(
            type(p).__name__ in ("ToolReturnPart", "RetryPromptPart")
            for m in messages
            for p in getattr(m, "parts", [])
        )
        if info.function_tools and not pending:
            tool = info.function_tools[0]
            schema = tool.parameters_json_schema or {}
            required = schema.get("required") or []
            args = {name: "San Francisco, CA" for name in required}
            yield {
                0: DeltaToolCall(
                    name=tool.name,
                    json_args=__import__("json").dumps(args),
                    tool_call_id="compliance_call_1",
                )
            }
            return
        yield "Hello from the "
        yield "compliance agent."

    return PydanticAgent(
        FunctionModel(stream_function=rule_based, model_name=MODEL_NAME),
        name="compliance_agent",
        instructions="Answer deterministically.",
    )


def create_langgraph_adapter():
    """Deterministic LangGraph adapter (graph factory) with the same rules."""
    import json

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    from open_responses_server.adapters.langgraph import LangGraphAdapter

    class RuleBasedChatModel(BaseChatModel):
        tool_specs: list = []

        @property
        def _llm_type(self) -> str:
            return MODEL_NAME

        def bind_tools(self, tools, **kwargs):
            self.tool_specs = list(tools)
            return self

        def _rule_based(self, messages) -> AIMessage:
            answered = any(isinstance(m, ToolMessage) for m in messages)
            if self.tool_specs and not answered:
                spec = self.tool_specs[0]
                schema = getattr(spec, "args_schema", None) or {}
                if hasattr(schema, "model_json_schema"):
                    schema = schema.model_json_schema()
                required = (schema or {}).get("required") or []
                args = {name: "San Francisco, CA" for name in required}
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": spec.name, "args": args, "id": "compliance_call_1"}
                    ],
                )
            return AIMessage(content="Hello from the compliance agent.")

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(
                generations=[ChatGeneration(message=self._rule_based(messages))]
            )

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            msg = self._rule_based(messages)
            usage = {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12}
            if msg.tool_calls:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="", tool_calls=msg.tool_calls, usage_metadata=usage
                    )
                )
                return
            for i, chunk in enumerate(["Hello from the ", "compliance agent."]):
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", usage_metadata=usage)
            )

    def build(client_tools):
        # Fresh model per request: the compliance CLI runs tests
        # concurrently and bind_tools state must not leak between runs.
        return create_react_agent(
            RuleBasedChatModel(tool_specs=[]),
            tools=client_tools,
            checkpointer=InMemorySaver(),
        )

    return LangGraphAdapter(build, model_name=MODEL_NAME)


def create_openai_agents_adapter():
    """Deterministic OpenAI Agents SDK adapter with the same rules."""
    import json

    import agents as agents_sdk
    from agents import Agent as SDKAgent
    from agents.models.interface import Model
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseTextDeltaEvent,
    )
    from openai.types.responses.response_usage import (
        InputTokensDetails,
        OutputTokensDetails,
        ResponseUsage,
    )

    from open_responses_server.adapters.openai_agents import OpenAIAgentsAdapter

    agents_sdk.set_tracing_disabled(True)

    class RuleBasedModel(Model):
        async def get_response(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        async def get_retry_advice(self, *args, **kwargs):  # pragma: no cover
            return None

        async def close(self) -> None:  # pragma: no cover
            return None

        async def stream_response(
            self,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            **kwargs,
        ):
            usage = ResponseUsage(
                input_tokens=7,
                output_tokens=5,
                total_tokens=12,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            )
            answered = any(
                isinstance(i, dict) and i.get("type") == "function_call_output"
                for i in (input if isinstance(input, list) else [])
            )
            if tools and not answered:
                tool = tools[0]
                schema = getattr(tool, "params_json_schema", None) or {}
                required = schema.get("required") or []
                args = {name: "San Francisco, CA" for name in required}
                output = [
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="compliance_call_1",
                        name=tool.name,
                        arguments=json.dumps(args),
                        status="completed",
                    )
                ]
            else:
                text = "Hello from the compliance agent."
                for i, chunk in enumerate(["Hello from the ", "compliance agent."]):
                    yield ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        content_index=0,
                        item_id="msg_1",
                        output_index=0,
                        delta=chunk,
                        logprobs=[],
                        sequence_number=i,
                    )
                output = [
                    ResponseOutputMessage(
                        id="msg_1",
                        role="assistant",
                        status="completed",
                        type="message",
                        content=[
                            ResponseOutputText(
                                type="output_text", text=text, annotations=[]
                            )
                        ],
                    )
                ]
            yield ResponseCompletedEvent(
                type="response.completed",
                response=Response(
                    id="resp_fake",
                    created_at=0,
                    model=MODEL_NAME,
                    object="response",
                    output=output,
                    parallel_tool_calls=False,
                    tool_choice="auto",
                    tools=[],
                    usage=usage,
                ),
                sequence_number=99,
            )

    agent = SDKAgent(name="compliance_agent", model=RuleBasedModel())
    return OpenAIAgentsAdapter(agent, model_name=MODEL_NAME)

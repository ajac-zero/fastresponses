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

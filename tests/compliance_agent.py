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

"""Example: a Pydantic AI agent served over the Open Responses API.

Run it (requires the pydantic-ai extra and a configured model provider,
e.g. ``OPENAI_API_KEY``)::

    open-responses-server serve examples/pydantic_ai_weather_agent.py:agent

Then talk to it with any Open Responses / OpenAI Responses client::

    curl http://127.0.0.1:8080/v1/responses \
        -H 'Content-Type: application/json' \
        -d '{"input": "What is the weather in Paris?"}'
"""

from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-5.2",
    name="weather_agent",
    instructions="You are a helpful weather assistant.",
)


@agent.tool_plain
def get_weather(city: str) -> dict:
    """Returns the current weather for a city."""
    return {"city": city, "forecast": "sunny", "temperature_c": 21}

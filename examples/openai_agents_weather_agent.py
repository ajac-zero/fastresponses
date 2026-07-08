"""Example: an OpenAI Agents SDK agent served over the Open Responses API.

Run it (requires the openai-agents extra and ``OPENAI_API_KEY``)::

    open-responses-server serve examples/openai_agents_weather_agent.py:agent
"""

from agents import Agent, function_tool


@function_tool
def get_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"It is sunny in {city}, 21C."


agent = Agent(
    name="weather_agent",
    model="gpt-5.2",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)

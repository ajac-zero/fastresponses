"""Example: a LangGraph agent served over the Open Responses API.

Run it (requires the langgraph extra plus a LangChain model package, e.g.
``langchain[openai]`` and ``OPENAI_API_KEY``)::

    fastresponses serve examples/langgraph_weather_agent.py:agent

To let clients declare their own tools per request, serve an adapter built
from a graph *factory* instead (``:adapter`` below) — the adapter passes
client-declared tools in as interrupt-based LangChain tools and surfaces
their calls as ``function_call`` items::

    fastresponses serve examples/langgraph_weather_agent.py:adapter
"""

from langchain.agents import create_agent

from fastresponses.adapters.langgraph import LangGraphAdapter


def get_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"It is sunny in {city}, 21C."


MODEL = "openai:gpt-5.2"

# Fixed graph: server-side tools only.
agent = create_agent(MODEL, tools=[get_weather])

# Graph factory: also accepts client-declared tools from each request.
adapter = LangGraphAdapter(
    lambda client_tools: create_agent(MODEL, tools=[get_weather, *client_tools])
)

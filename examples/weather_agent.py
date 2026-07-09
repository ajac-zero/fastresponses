"""Example: expose a Google ADK agent over the Open Responses API.

Run (requires GOOGLE_API_KEY or Vertex AI credentials):

    uv run fastresponses serve examples/weather_agent.py:agent --port 8080

Then talk to it with any Open Responses / OpenAI Responses client:

    curl http://127.0.0.1:8080/v1/responses \
      -H 'Content-Type: application/json' \
      -d '{"input": "What is the weather in Tokyo?", "stream": true}'
"""

from google.adk.agents import Agent


def get_weather(city: str) -> dict:
    """Returns the current weather for a city.

    Args:
        city: Name of the city.
    """
    # Replace with a real weather API call.
    return {"city": city, "forecast": "sunny", "temperature_c": 21}


agent = Agent(
    name="weather_agent",
    model="gemini-2.5-flash",
    description="Answers questions about the weather.",
    instruction="You are a helpful weather assistant. Use the get_weather tool "
    "to answer questions about current conditions.",
    tools=[get_weather],
)

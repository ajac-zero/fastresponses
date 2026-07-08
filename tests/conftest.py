from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterable

from fastapi.testclient import TestClient

from open_responses_server.adapter import (
    AdapterEvent,
    AgentAdapter,
    AgentRun,
)
from open_responses_server.server import create_app


class FakeAdapter(AgentAdapter):
    """Adapter driven by a script function for protocol-level tests."""

    name = "fake"
    default_model = "fake-model"

    def __init__(
        self, script: Callable[[AgentRun], Iterable[AdapterEvent]]
    ) -> None:
        self.script = script
        self.runs: list[AgentRun] = []

    async def run(self, run: AgentRun) -> AsyncIterator[AdapterEvent]:
        self.runs.append(run)
        for event in self.script(run):
            if hasattr(event, "__await__"):
                # scripts may yield awaitables (e.g. asyncio.sleep) to pace
                # long-running turns in background/cancellation tests
                await event
                continue
            yield event


def make_client(
    script: Callable[[AgentRun], Iterable[AdapterEvent]],
    **app_kwargs,
) -> tuple[TestClient, FakeAdapter]:
    adapter = FakeAdapter(script)
    app = create_app(adapter, **app_kwargs)
    return TestClient(app), adapter


def read_sse(response) -> list[dict | str]:
    """Parse an SSE body into a list of event dicts (and the '[DONE]' marker).

    Also asserts that every ``event:`` field matches the payload ``type``.
    """
    events: list[dict | str] = []
    event_name: str | None = None
    for line in response.iter_lines():
        if line.startswith("event: "):
            event_name = line[len("event: ") :]
        elif line.startswith("data: "):
            data = line[len("data: ") :]
            if data == "[DONE]":
                events.append("[DONE]")
                continue
            payload = json.loads(data)
            assert event_name == payload["type"], (event_name, payload["type"])
            events.append(payload)
        elif line == "":
            event_name = None
    return events

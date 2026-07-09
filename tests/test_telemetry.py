"""Tests for turn-level observability: structured logs and OTel spans."""

from __future__ import annotations

import logging

import pytest

from fastresponses.adapter import AdapterError, TextDelta, UsageDelta

from conftest import make_client


def script(run):
    yield TextDelta("hello")
    yield UsageDelta(input_tokens=7, output_tokens=3, total_tokens=10)


def failing_script(run):
    raise AdapterError("boom", type="server_error", code="model_error")
    yield  # pragma: no cover


def test_turn_log_record_has_structured_fields(caplog):
    client, _ = make_client(script)
    with caplog.at_level(logging.INFO, logger="fastresponses.turn"):
        body = client.post("/v1/responses", json={"input": "hi"}).json()

    records = [r for r in caplog.records if hasattr(r, "turn")]
    assert len(records) == 1
    turn = records[0].turn
    assert turn["response_id"] == body["id"]
    assert turn["adapter"] == "fake"
    assert turn["status"] == "completed"
    assert turn["input_tokens"] == 7
    assert turn["output_tokens"] == 3
    assert turn["output_items"] == 1
    assert turn["duration_ms"] >= 0


def test_failed_turn_logs_warning_with_error_code(caplog):
    client, _ = make_client(failing_script)
    with caplog.at_level(logging.INFO, logger="fastresponses.turn"):
        client.post("/v1/responses", json={"input": "hi"})

    records = [r for r in caplog.records if hasattr(r, "turn")]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].turn["status"] == "failed"
    assert records[0].turn["error_code"] == "model_error"


def test_otel_span_per_turn():
    otel_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    # Install an SDK tracer provider (no-op API otherwise). This is global,
    # so re-fetch the module-level tracer used by telemetry.
    import fastresponses.telemetry as telemetry

    exporter = InMemorySpanExporter()
    provider = otel_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    telemetry._tracer = trace.get_tracer("fastresponses")

    client, _ = make_client(script)
    body = client.post("/v1/responses", json={"input": "hi"}).json()

    spans = exporter.get_finished_spans()
    turn_spans = [s for s in spans if s.name == "open_responses.turn"]
    assert len(turn_spans) == 1
    attrs = dict(turn_spans[0].attributes)
    assert attrs["open_responses.response_id"] == body["id"]
    assert attrs["open_responses.status"] == "completed"
    assert attrs["gen_ai.usage.input_tokens"] == 7

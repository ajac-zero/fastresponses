"""Turn-level observability: structured logs and optional OpenTelemetry spans.

Every turn emits one structured log record on the
``fastresponses.turn`` logger with the outcome and token accounting
in ``record.__dict__`` (works with any structured-logging formatter).

If ``opentelemetry-api`` is installed (the ``otel`` extra), each turn is
additionally wrapped in an ``open_responses.turn`` span with the same
attributes. Spans are started without touching the active context (safe
inside async generators); they parent to whatever context is current when
the turn starts, e.g. an ASGI instrumentation span.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("fastresponses.turn")

try:  # pragma: no cover - trivial import branch
    from opentelemetry import trace as _trace

    _tracer = _trace.get_tracer("fastresponses")
except ImportError:  # pragma: no cover
    _trace = None
    _tracer = None


class TurnTelemetry:
    """Tracks one response turn; emits a log record and an optional span."""

    def __init__(self, *, adapter: str, model: str, response_id: str) -> None:
        self.adapter = adapter
        self.model = model
        self.response_id = response_id
        self._started = time.monotonic()
        self._span = None
        if _tracer is not None:
            self._span = _tracer.start_span(
                "open_responses.turn",
                attributes={
                    "open_responses.response_id": response_id,
                    "open_responses.adapter": adapter,
                    "gen_ai.request.model": model,
                },
            )

    def finish(
        self,
        *,
        status: str,
        usage: Any = None,
        output_items: int = 0,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        duration_ms = round((time.monotonic() - self._started) * 1000, 2)
        fields: dict[str, Any] = {
            "response_id": self.response_id,
            "adapter": self.adapter,
            "model": self.model,
            "status": status,
            "duration_ms": duration_ms,
            "output_items": output_items,
        }
        if usage is not None:
            fields["input_tokens"] = usage.input_tokens
            fields["output_tokens"] = usage.output_tokens
            fields["total_tokens"] = usage.total_tokens
        if error_code:
            fields["error_code"] = error_code

        level = logging.INFO if status in ("completed", "incomplete") else logging.WARNING
        logger.log(
            level,
            "response %s %s (%.0f ms)",
            self.response_id,
            status,
            duration_ms,
            extra={"turn": fields},
        )

        if self._span is not None:
            for key, value in fields.items():
                if key in ("response_id", "adapter", "model"):
                    continue
                self._span.set_attribute(f"open_responses.{key}", value)
            if usage is not None:
                self._span.set_attribute(
                    "gen_ai.usage.input_tokens", usage.input_tokens
                )
                self._span.set_attribute(
                    "gen_ai.usage.output_tokens", usage.output_tokens
                )
            if status == "failed" and _trace is not None:
                self._span.set_status(
                    _trace.status.Status(
                        _trace.status.StatusCode.ERROR, error_message or ""
                    )
                )
            self._span.end()
            self._span = None

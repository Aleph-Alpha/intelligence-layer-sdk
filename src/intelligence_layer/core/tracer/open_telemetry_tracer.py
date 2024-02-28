from datetime import datetime
from typing import Optional

from opentelemetry.context import attach, detach
from opentelemetry.trace import Span as OpenTSpan
from opentelemetry.trace import Tracer as OpenTTracer
from opentelemetry.trace import set_span_in_context

from intelligence_layer.core.tracer.tracer import (
    PydanticSerializable,
    Span,
    TaskSpan,
    Tracer,
    _serialize,
)


class OpenTelemetryTracer(Tracer):
    """A `Tracer` that uses open telemetry."""

    def __init__(self, tracer: OpenTTracer) -> None:
        self._tracer = tracer

    def span(
        self,
        name: str,
        timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> "OpenTelemetrySpan":
        trace_id = self.ensure_id(trace_id)
        tracer_span = self._tracer.start_span(
            name,
            attributes={"trace_id": trace_id},
            start_time=None if not timestamp else _open_telemetry_timestamp(timestamp),
        )
        token = attach(set_span_in_context(tracer_span))
        return OpenTelemetrySpan(tracer_span, self._tracer, token, trace_id)

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> "OpenTelemetryTaskSpan":
        trace_id = self.ensure_id(trace_id)

        tracer_span = self._tracer.start_span(
            task_name,
            attributes={"input": _serialize(input), "trace_id": trace_id},
            start_time=None if not timestamp else _open_telemetry_timestamp(timestamp),
        )
        token = attach(set_span_in_context(tracer_span))
        return OpenTelemetryTaskSpan(tracer_span, self._tracer, token, trace_id)


class OpenTelemetrySpan(Span, OpenTelemetryTracer):
    """A `Span` created by `OpenTelemetryTracer.span`."""

    end_timestamp: Optional[datetime] = None

    def id(self) -> str:
        return self._trace_id

    def __init__(
        self, span: OpenTSpan, tracer: OpenTTracer, token: object, trace_id: str
    ) -> None:
        super().__init__(tracer)
        self.open_ts_span = span
        self._token = token
        self._trace_id = trace_id

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.open_ts_span.add_event(
            message,
            {"value": _serialize(value), "trace_id": self.id()},
            None if not timestamp else _open_telemetry_timestamp(timestamp),
        )

    def end(self, timestamp: Optional[datetime] = None) -> None:
        detach(self._token)
        self.open_ts_span.end(
            _open_telemetry_timestamp(timestamp) if timestamp is not None else None
        )


class OpenTelemetryTaskSpan(TaskSpan, OpenTelemetrySpan):
    """A `TaskSpan` created by `OpenTelemetryTracer.task_span`."""

    output: Optional[PydanticSerializable] = None

    def __init__(
        self, span: OpenTSpan, tracer: OpenTTracer, token: object, trace_id: str
    ) -> None:
        super().__init__(span, tracer, token, trace_id)

    def record_output(self, output: PydanticSerializable) -> None:
        self.open_ts_span.set_attribute("output", _serialize(output))


def _open_telemetry_timestamp(t: datetime) -> int:
    # Open telemetry expects *nanoseconds* since epoch
    t_float = t.timestamp() * 1e9
    return int(t_float)

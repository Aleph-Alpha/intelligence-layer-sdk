from datetime import datetime
from typing import Optional, Sequence

from opentelemetry.context import attach, detach
from opentelemetry.trace import Span as OpenTSpan
from opentelemetry.trace import StatusCode, set_span_in_context
from opentelemetry.trace import Tracer as OpenTTracer
from pydantic import BaseModel, SerializeAsAny

from intelligence_layer.core.tracer.tracer import (
    Context,
    ExportedSpan,
    JsonSerializer,
    PydanticSerializable,
    Span,
    SpanStatus,
    SpanType,
    TaskSpan,
    Tracer,
)


class OpenTelemetryTracer(Tracer):
    """A `Tracer` that uses open telemetry."""

    def __init__(self, tracer: OpenTTracer) -> None:
        self._tracer = tracer

    def span(
        self,
        name: str,
        timestamp: Optional[datetime] = None,
    ) -> "OpenTelemetrySpan":
        tracer_span = self._tracer.start_span(
            name,
            attributes={"type": SpanType.SPAN.value},
            start_time=None if not timestamp else _open_telemetry_timestamp(timestamp),
        )
        token = attach(set_span_in_context(tracer_span))
        return OpenTelemetrySpan(tracer_span, self._tracer, token, self.context)

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "OpenTelemetryTaskSpan":
        tracer_span = self._tracer.start_span(
            task_name,
            attributes={"input": _serialize(input), "type": SpanType.TASK_SPAN.value},
            start_time=None if not timestamp else _open_telemetry_timestamp(timestamp),
        )
        token = attach(set_span_in_context(tracer_span))
        return OpenTelemetryTaskSpan(tracer_span, self._tracer, token, self.context)

    def export_for_viewing(self) -> Sequence[ExportedSpan]:
        raise NotImplementedError(
            "The OpenTelemetryTracer does not support export for viewing, as it can not access its own traces."
        )


class OpenTelemetrySpan(Span, OpenTelemetryTracer):
    """A `Span` created by `OpenTelemetryTracer.span`."""

    end_timestamp: Optional[datetime] = None

    def __init__(
        self,
        span: OpenTSpan,
        tracer: OpenTTracer,
        token: object,
        context: Optional[Context] = None,
    ) -> None:
        OpenTelemetryTracer.__init__(self, tracer)
        Span.__init__(self, context=context)
        self.open_ts_span = span
        self._token = token

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.open_ts_span.add_event(
            message,
            {"value": _serialize(value)},
            None if not timestamp else _open_telemetry_timestamp(timestamp),
        )

    def end(self, timestamp: Optional[datetime] = None) -> None:
        super().end(timestamp)
        self.open_ts_span.set_status(
            StatusCode.OK if self.status_code == SpanStatus.OK else StatusCode.ERROR
        )
        detach(self._token)
        self.open_ts_span.end(
            _open_telemetry_timestamp(timestamp) if timestamp is not None else None
        )


class OpenTelemetryTaskSpan(TaskSpan, OpenTelemetrySpan):
    """A `TaskSpan` created by `OpenTelemetryTracer.task_span`."""

    output: Optional[PydanticSerializable] = None

    def record_output(self, output: PydanticSerializable) -> None:
        self.open_ts_span.set_attribute("output", _serialize(output))


def _open_telemetry_timestamp(t: datetime) -> int:
    # Open telemetry expects *nanoseconds* since epoch
    t_float = t.timestamp() * 1e9
    return int(t_float)


def _serialize(s: SerializeAsAny[PydanticSerializable]) -> str:
    value = s if isinstance(s, BaseModel) else JsonSerializer(root=s)
    return value.model_dump_json()

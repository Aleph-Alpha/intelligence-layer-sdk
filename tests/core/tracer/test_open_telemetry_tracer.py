import json
import time
from typing import Any
from uuid import uuid4

import pytest
import requests
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from pytest import fixture

from intelligence_layer.core import OpenTelemetryTracer, Task


class DummyExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: trace.Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.spans = []

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@fixture()
def exporter() -> DummyExporter:
    return DummyExporter()


@fixture(scope="module")
def service_name() -> str:
    return "test-service"


@fixture(scope="module")
def trace_provider(service_name: str) -> TracerProvider:
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    return provider


@fixture()
def test_opentelemetry_tracer(
    exporter: DummyExporter, trace_provider: TracerProvider
) -> OpenTelemetryTracer:
    processor = SimpleSpanProcessor(exporter)
    trace_provider.add_span_processor(processor)
    tracer = OpenTelemetryTracer(trace.get_tracer("intelligence-layer"))
    return tracer


@fixture
def jaeger_compatible_tracer(trace_provider: TracerProvider) -> OpenTelemetryTracer:
    processor = SimpleSpanProcessor(OTLPSpanExporter())
    trace_provider.add_span_processor(processor)
    tracer = OpenTelemetryTracer(trace.get_tracer("intelligence-layer"))

    return tracer


def test_open_telemetry_tracer_has_consistent_trace_id(
    test_opentelemetry_tracer: OpenTelemetryTracer,
    exporter: DummyExporter,
    test_task: Task[str, str],
) -> None:
    test_task.run("test-input", test_opentelemetry_tracer)
    spans = exporter.spans
    assert len(spans) == 4
    assert len(set(span.context.trace_id for span in spans)) == 1


def test_open_telemetry_tracer_sets_attributes_correctly(
    test_opentelemetry_tracer: OpenTelemetryTracer,
    exporter: DummyExporter,
    test_task: Task[str, str],
) -> None:
    test_task.run("test-input", test_opentelemetry_tracer)
    spans = exporter.spans
    assert len(spans) == 4
    spans_sorted_by_start: list[ReadableSpan] = sorted(
        spans, key=lambda span: span.start_time
    )

    assert spans_sorted_by_start[0].name == "TestTask"
    assert spans_sorted_by_start[0].attributes["input"] == '"test-input"'
    assert spans_sorted_by_start[0].attributes["output"] == '"output"'

    assert spans_sorted_by_start[1].name == "span"
    assert "input" not in spans_sorted_by_start[1].attributes.keys()

    assert spans_sorted_by_start[2].name == "TestSubTask"
    assert spans_sorted_by_start[2].attributes["input"] == "null"
    assert spans_sorted_by_start[2].attributes["output"] == "null"

    assert spans_sorted_by_start[3].name == "TestSubTask"
    assert spans_sorted_by_start[3].attributes["input"] == "null"
    assert spans_sorted_by_start[3].attributes["output"] == "null"

    spans_sorted_by_end: list[ReadableSpan] = sorted(
        spans_sorted_by_start, key=lambda span: span.end_time
    )

    assert spans_sorted_by_end[0] == spans_sorted_by_start[2]
    assert spans_sorted_by_end[1] == spans_sorted_by_start[1]
    assert spans_sorted_by_end[2] == spans_sorted_by_start[3]
    assert spans_sorted_by_end[3] == spans_sorted_by_start[0]


def has_span_with_input(trace: Any, input_value: str) -> bool:
    return any(
        tag["key"] == "input" and tag["value"] == f'"{input_value}"'
        for span in trace["spans"]
        for tag in span["tags"]
    )


def get_current_traces(tracing_service: str) -> Any:
    response = requests.get(tracing_service)
    response_text = json.loads(response.text)
    return response_text["data"]


@pytest.mark.docker
def test_open_telemetry_tracer_works_with_jaeger(
    jaeger_compatible_tracer: OpenTelemetryTracer,
    test_task: Task[str, str],
    service_name: str,
) -> None:
    url = "http://localhost:16686/api/traces?service=" + service_name
    input_value = str(uuid4())
    test_task.run(input_value, jaeger_compatible_tracer)
    # the processor needs time to submit the trace to jaeger
    time.sleep(0.3)

    res = get_current_traces(url)

    test_res = [trace_ for trace_ in res if has_span_with_input(trace_, input_value)]

    assert len(test_res) == 1

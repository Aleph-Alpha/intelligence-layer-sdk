import json
import time
from typing import Any, Optional

import pytest
import requests
from aleph_alpha_client import Prompt
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pytest import fixture

from intelligence_layer.core import (
    CompleteInput,
    CompleteOutput,
    OpenTelemetryTracer,
    Task,
)


@fixture
def open_telemetry_tracer() -> tuple[str, OpenTelemetryTracer]:
    service_name = "test-service"
    url = "http://localhost:16686/api/traces?service=" + service_name
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    openTracer = OpenTelemetryTracer(trace.get_tracer("intelligence-layer"))
    return (url, openTracer)


def _get_trace_by_id(tracing_service: str, wanted_trace_id: str) -> Optional[Any]:
    request_timeout_in_seconds = 10
    traces = _get_current_traces(tracing_service)
    if traces:
        for current_trace in traces:
            trace_id = _get_trace_id_from_trace(current_trace)
            if trace_id == wanted_trace_id:
                return trace

    request_start = time.time()
    while time.time() - request_start < request_timeout_in_seconds:
        traces = _get_current_traces(tracing_service)
        if traces:
            for current_trace in traces:
                trace_id = _get_trace_id_from_trace(current_trace)
                if trace_id == wanted_trace_id:
                    return current_trace
        time.sleep(0.1)
    return None


def _get_current_traces(tracing_service: str) -> Any:
    response = requests.get(tracing_service)
    response_text = json.loads(response.text)
    return response_text["data"]


def _get_trace_id_from_trace(trace: Any) -> Optional[str]:
    spans = trace["spans"]
    if not spans:
        return None
    return _get_trace_id_from_span(spans[0])


def _get_trace_id_from_span(span: Any) -> Optional[str]:
    tags = span["tags"]
    if not tags:
        return None
    trace_id_tag = next(tag for tag in tags if tag["key"] == "trace_id")
    return str(trace_id_tag["value"])


@pytest.mark.docker
def test_open_telemetry_tracer_check_consistency_in_trace_ids(
    open_telemetry_tracer: tuple[str, OpenTelemetryTracer],
    test_task: Task[str, str],
) -> None:
    tracing_service, tracer = open_telemetry_tracer
    expected_trace_id = tracer.ensure_id(None)
    test_task.run("test-input", tracer, trace_id=expected_trace_id)
    trace = _get_trace_by_id(tracing_service, expected_trace_id)

    assert trace is not None
    assert _get_trace_id_from_trace(trace) == expected_trace_id
    spans = trace["spans"]
    assert len(spans) == 4
    for span in spans:
        assert _get_trace_id_from_span(span) == expected_trace_id


@pytest.mark.docker
def test_open_telemetry_tracer_loggs_input_and_output(
    open_telemetry_tracer: tuple[str, OpenTelemetryTracer],
    complete: Task[CompleteInput, CompleteOutput],
) -> None:
    tracing_service, tracer = open_telemetry_tracer
    input = CompleteInput(prompt=Prompt.from_text("test"))
    trace_id = tracer.ensure_id(None)
    complete.run(input, tracer, trace_id)
    trace = _get_trace_by_id(tracing_service, trace_id)

    assert trace is not None

    spans = trace["spans"]
    assert spans is not []

    task_span = next((span for span in spans if span["references"] == []), None)
    assert task_span is not None

    tags = task_span["tags"]
    open_tel_input_tag = [tag for tag in tags if tag["key"] == "input"]
    assert len(open_tel_input_tag) == 1

    open_tel_output_tag = [tag for tag in tags if tag["key"] == "output"]
    assert len(open_tel_output_tag) == 1

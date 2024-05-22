import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from unittest.mock import Mock

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
    CompositeTracer,
    FileTracer,
    InMemorySpan,
    InMemoryTaskSpan,
    InMemoryTracer,
    LogEntry,
    LuminousControlModel,
    OpenTelemetryTracer,
    Task,
    TaskSpan,
    utc_now,
)
from intelligence_layer.core.tracer.persistent_tracer import TracerLogEntryFailed
from intelligence_layer.core.tracer.tracer import ErrorValue

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


def test_temp_1(open_telemetry_tracer):
    print("test_1")

def test_temp_2(open_telemetry_tracer):
    print("test_2")

def test_temp_3(open_telemetry_tracer):
    print("test_3")
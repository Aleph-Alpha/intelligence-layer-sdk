import logging
import time
from typing import Sequence
from uuid import uuid4

import requests
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from tqdm import tqdm

from intelligence_layer.core import OpenTelemetryTracer, Task
from intelligence_layer.core.tracer.tracer import TaskSpan

logging.basicConfig(level=logging.DEBUG)


class StudioSpanExporter(SpanExporter):
    def __init__(self):
        pass

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        url = "http://127.0.0.1:8000/print_json"
        json_spans = [span.to_json() for span in spans]
        print(json_spans)
        requests.post(url, "[" + ",".join(json_spans) + "]")

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        pass


class TracerTestSubTask(Task[None, None]):
    def do_run(self, input: None, task_span: TaskSpan) -> None:
        task_span.log("subtask", "value")


class TracerTestTask(Task[str, str]):
    sub_task = TracerTestSubTask()

    def do_run(self, input: str, task_span: TaskSpan) -> str:
        sleep = 0.1
        time.sleep(sleep)
        with task_span.span("span") as sub_span:
            time.sleep(sleep)
            sub_span.log("message", "a value")
            time.sleep(sleep)
            self.sub_task.run(None, sub_span)
            time.sleep(sleep)
        self.sub_task.run(None, task_span)
        time.sleep(sleep)
        return "output"


project_name = str(uuid4())
service_name = "test-service"

resource = Resource.create({SERVICE_NAME: service_name})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

url = "http://localhost:4318/v1/traces"
processor = BatchSpanProcessor(OTLPSpanExporter(url), max_export_batch_size=10)

provider.add_span_processor(processor)
tracer = OpenTelemetryTracer(trace.get_tracer("intelligence-layer"))

for i in tqdm(range(1000), total=1000):
    input_value = {"key": "value"}
    TracerTestTask().run(input_value, tracer)

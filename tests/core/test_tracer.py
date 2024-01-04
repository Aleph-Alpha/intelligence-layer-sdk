from json import loads
from pathlib import Path
from uuid import UUID

from aleph_alpha_client import Prompt
from aleph_alpha_client.completion import CompletionRequest
from opentelemetry.trace import get_tracer
from pydantic import BaseModel, Field
from pytest import fixture, mark

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core import (
    Complete,
    CompleteInput,
    CompositeTracer,
    FileTracer,
    InMemorySpan,
    InMemoryTaskSpan,
    InMemoryTracer,
    LogEntry,
    OpenTelemetryTracer,
    Task,
    TaskSpan,
    utc_now,
)
from intelligence_layer.core.tracer import (
    EndSpan,
    EndTask,
    LogLine,
    PlainEntry,
    StartSpan,
    StartTask,
)


def test_tracer_id_exists_for_all_children_of_task_span() -> None:
    tracer = InMemoryTracer()
    parent_span = tracer.task_span("child", "input", id="ID")
    parent_span.span("child2")

    assert tracer.entries[0].id() == "ID"
    assert tracer.entries[0].entries[0].id() == tracer.entries[0].id()


    parent_span.task_span("child3", "input")
    assert tracer.entries[0].entries[1].id() == tracer.entries[0].id()


def test_tracer_id_exists_for_all_children_of_span() -> None:
    tracer = InMemoryTracer()
    parent_span = tracer.span("child", id="ID")
    parent_span.span("child2")

    assert tracer.entries[0].id() == "ID"
    assert tracer.entries[0].entries[0].id() == tracer.entries[0].id()

    parent_span.task_span("child3", "input")
    assert tracer.entries[0].entries[1].id() == tracer.entries[0].id()


def test_can_add_child_tracer() -> None:
    tracer = InMemoryTracer()
    tracer.span("child")

    assert len(tracer.entries) == 1

    log = tracer.entries[0]
    assert isinstance(log, InMemoryTracer)
    assert log.name == "child"
    assert len(log.entries) == 0


def test_can_add_parent_and_child_entries() -> None:
    parent = InMemoryTracer()
    with parent.span("child") as child:
        child.log("Two", 2)

    assert isinstance(parent.entries[0], InMemoryTracer)
    assert isinstance(parent.entries[0].entries[0], LogEntry)


def test_task_automatically_logs_input_and_output(
    client: AlephAlphaClientProtocol,
) -> None:
    tracer = InMemoryTracer()
    input = CompleteInput(
        request=CompletionRequest(prompt=Prompt.from_text("test")),
        model="luminous-base",
    )
    output = Complete(client=client).run(input=input, tracer=tracer)

    assert len(tracer.entries) == 1
    task_span = tracer.entries[0]
    assert isinstance(task_span, InMemoryTaskSpan)
    assert task_span.name == "Complete"
    assert task_span.input == input
    assert task_span.output == output
    assert task_span.start_timestamp and task_span.end_timestamp
    assert task_span.start_timestamp < task_span.end_timestamp


def test_tracer_can_set_custom_start_time_for_log_entry() -> None:
    tracer = InMemoryTracer()
    timestamp = utc_now()

    with tracer.span("span") as span:
        span.log("log", "message", timestamp)

    assert isinstance(tracer.entries[0], InMemorySpan)
    assert isinstance(tracer.entries[0].entries[0], LogEntry)
    assert tracer.entries[0].entries[0].timestamp == timestamp


def test_tracer_can_set_custom_start_time_for_span() -> None:
    tracer = InMemoryTracer()
    start = utc_now()

    span = tracer.span("span", start)

    assert span.start_timestamp == start


def test_span_sets_end_timestamp() -> None:
    tracer = InMemoryTracer()
    start = utc_now()

    span = tracer.span("span", start)
    span.end()

    assert span.end_timestamp and span.start_timestamp <= span.end_timestamp


def test_span_only_updates_end_timestamp_once() -> None:
    tracer = InMemoryTracer()

    span = tracer.span("span")
    end = utc_now()
    span.end(end)
    span.end()

    assert span.end_timestamp == end


def test_composite_tracer(client: AlephAlphaClientProtocol) -> None:
    tracer1 = InMemoryTracer()
    tracer2 = InMemoryTracer()
    input = CompleteInput(
        request=CompletionRequest(prompt=Prompt.from_text("test")),
        model="luminous-base",
    )
    Complete(client=client).run(input=input, tracer=CompositeTracer([tracer1, tracer2]))

    assert tracer1 == tracer2


class TestSubTask(Task[None, None]):
    def do_run(self, input: None, task_span: TaskSpan) -> None:
        task_span.log("subtask", "value")


class TestTask(Task[str, str]):
    sub_task = TestSubTask()

    def do_run(self, input: str, task_span: TaskSpan) -> str:
        with task_span.span("span") as sub_span:
            sub_span.log("message", "a value")
            self.sub_task.run(None, sub_span)
        self.sub_task.run(None, task_span)

        return "output"


@fixture
def file_tracer(tmp_path: Path) -> FileTracer:
    return FileTracer(tmp_path / "log.log")


def test_file_tracer(file_tracer: FileTracer) -> None:
    input = "input"
    expected = InMemoryTracer()

    TestTask().run(input, CompositeTracer([expected, file_tracer]))

    log_tree = parse_log(file_tracer._log_file_path)
    assert log_tree == expected


def parse_log(log_path: Path) -> InMemoryTracer:
    tree_builder = TreeBuilder()
    with log_path.open("r") as f:
        for line in f:
            json_line = loads(line)
            log_line = LogLine.model_validate(json_line)
            if log_line.entry_type == StartTask.__name__:
                tree_builder.start_task(log_line)
            elif log_line.entry_type == EndTask.__name__:
                tree_builder.end_task(log_line)
            elif log_line.entry_type == StartSpan.__name__:
                tree_builder.start_span(log_line)
            elif log_line.entry_type == EndSpan.__name__:
                tree_builder.end_span(log_line)
            elif log_line.entry_type == PlainEntry.__name__:
                tree_builder.plain_entry(log_line)
            else:
                raise RuntimeError(f"Unexpected entry_type in {log_line}")
    assert tree_builder.root
    return tree_builder.root


class TreeBuilder(BaseModel):
    root: InMemoryTracer = InMemoryTracer()
    tracers: dict[UUID, InMemoryTracer] = Field(default_factory=dict)
    tasks: dict[UUID, InMemoryTaskSpan] = Field(default_factory=dict)
    spans: dict[UUID, InMemorySpan] = Field(default_factory=dict)

    def start_task(self, log_line: LogLine) -> None:
        start_task = StartTask.model_validate(log_line.entry)
        child = InMemoryTaskSpan(
            name=start_task.name,
            input=start_task.input,
            start_timestamp=start_task.start,
        )
        self.tracers[start_task.uuid] = child
        self.tasks[start_task.uuid] = child
        self.tracers.get(start_task.parent, self.root).entries.append(child)

    def end_task(self, log_line: LogLine) -> None:
        end_task = EndTask.model_validate(log_line.entry)
        task_span = self.tasks[end_task.uuid]
        task_span.end_timestamp = end_task.end
        task_span.record_output(end_task.output)

    def start_span(self, log_line: LogLine) -> None:
        start_span = StartSpan.model_validate(log_line.entry)
        child = InMemorySpan(name=start_span.name, start_timestamp=start_span.start)
        self.tracers[start_span.uuid] = child
        self.spans[start_span.uuid] = child
        self.tracers.get(start_span.parent, self.root).entries.append(child)

    def end_span(self, log_line: LogLine) -> None:
        end_span = EndSpan.model_validate(log_line.entry)
        span = self.spans[end_span.uuid]
        span.end_timestamp = end_span.end

    def plain_entry(self, log_line: LogLine) -> None:
        plain_entry = PlainEntry.model_validate(log_line.entry)
        entry = LogEntry(
            message=plain_entry.message,
            value=plain_entry.value,
            timestamp=plain_entry.timestamp,
        )
        self.tracers[plain_entry.parent].entries.append(entry)


@mark.skip(
    "Does not assert anything, here to show how you can use the OpenTelemetry Tracer."
)
def test_open_telemetry_tracer() -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    # Service name is required for most backends,
    # and although it's not necessary for console export,
    # it's good to set service name anyways.
    resource = Resource(attributes={SERVICE_NAME: "your-service-name"})

    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    input = "input"
    openTracer = get_tracer("intelligence-layer")
    TestTask().run(input, CompositeTracer([OpenTelemetryTracer(openTracer)]))
    provider.force_flush()

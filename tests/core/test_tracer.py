from pathlib import Path

from aleph_alpha_client import Prompt
from opentelemetry.trace import get_tracer
from pytest import fixture, mark

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


@fixture
def complete(
    luminous_control_model: LuminousControlModel,
) -> Task[CompleteInput, CompleteOutput]:
    return luminous_control_model.complete_task()


def test_composite_tracer_id_consistent_across_children(
    file_tracer: FileTracer,
) -> None:
    input = "input"
    tracer1 = InMemoryTracer()

    TestTask().run(input, CompositeTracer([tracer1]))
    assert isinstance(tracer1.entries[0], InMemorySpan)
    assert tracer1.entries[0].id() == tracer1.entries[0].entries[0].id()


def test_tracer_id_exists_for_all_children_of_task_span() -> None:
    tracer = InMemoryTracer()
    parent_span = tracer.task_span("child", "input", id="ID")
    parent_span.span("child2")

    assert isinstance(tracer.entries[0], InMemorySpan)
    assert tracer.entries[0].id() == "ID"

    assert tracer.entries[0].entries[0].id() == tracer.entries[0].id()

    parent_span.task_span("child3", "input")
    assert tracer.entries[0].entries[1].id() == tracer.entries[0].id()


def test_tracer_id_exists_for_all_children_of_span() -> None:
    tracer = InMemoryTracer()
    parent_span = tracer.span("child", id="ID")
    parent_span.span("child2")

    assert isinstance(tracer.entries[0], InMemorySpan)
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
    complete: Task[CompleteInput, CompleteOutput]
) -> None:
    tracer = InMemoryTracer()
    input = CompleteInput(prompt=Prompt.from_text("test"))
    output = complete.run(input=input, tracer=tracer)

    assert len(tracer.entries) == 1
    task_span = tracer.entries[0]
    assert isinstance(task_span, InMemoryTaskSpan)
    assert task_span.name == type(complete).__name__
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


def test_composite_tracer(complete: Task[CompleteInput, CompleteOutput]) -> None:
    tracer1 = InMemoryTracer()
    tracer2 = InMemoryTracer()
    input = CompleteInput(prompt=Prompt.from_text("test"))
    complete.run(input=input, tracer=CompositeTracer([tracer1, tracer2]))

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

    log_tree = file_tracer.trace()
    assert log_tree == expected


def test_file_tracer_retrieves_correct_trace(file_tracer: FileTracer) -> None:
    input = "input"
    expected = InMemoryTracer()
    compositeTracer = CompositeTracer([expected, file_tracer])
    TestTask().run(input, compositeTracer, "ID1")
    TestTask().run(input, file_tracer, "ID2")
    log_tree = file_tracer.trace("ID1")
    assert log_tree == expected


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

import time

import pytest
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    CompositeTracer,
    FileTracer,
    InMemoryTracer,
    SpanStatus,
    SpanType,
    TaskSpanAttributes,
    Tracer,
    utc_now,
)
from intelligence_layer.core.task import Task
from tests.core.tracer.conftest import SpecificTestException


class DummyObject(BaseModel):
    content: str


@fixture
def composite_tracer(
    in_memory_tracer: InMemoryTracer, file_tracer: FileTracer
) -> CompositeTracer[Tracer]:
    return CompositeTracer(tracers=[in_memory_tracer, file_tracer])


tracer_fixtures = ["in_memory_tracer", "file_tracer", "composite_tracer"]


def delay() -> None:
    time.sleep(0.001)


@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_tracer_exports_spans_to_unified_format(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)
    dummy_object = DummyObject(content="cool")
    with tracer.span("name") as temp_span:
        delay()
        temp_span.log("test", dummy_object)
        delay()
    delay()

    unified_format = tracer.export_for_viewing()

    assert len(unified_format) == 1
    span = unified_format[0]
    assert span.name == "name"
    assert span.start_time < span.end_time < utc_now()
    assert span.attributes.type == SpanType.SPAN
    assert span.status == SpanStatus.OK

    assert len(span.events) == 1
    log = span.events[0]
    assert log.message == "test"
    assert (
        log.body == dummy_object or DummyObject.model_validate(log.body) == dummy_object
    )
    assert span.start_time < log.timestamp < span.end_time


@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_tracer_exports_task_spans_to_unified_format(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    with tracer.task_span("name", "input") as task_span:
        delay()
        task_span.record_output("output")
    delay()

    unified_format = tracer.export_for_viewing()

    assert len(unified_format) == 1
    span = unified_format[0]
    assert span.name == "name"
    assert span.parent_id is None
    assert span.start_time < span.end_time < utc_now()
    assert span.attributes.type == SpanType.TASK_SPAN
    assert isinstance(span.attributes, TaskSpanAttributes)  # for mypy
    assert span.attributes.input == "input"
    assert span.attributes.output == "output"
    assert span.status == SpanStatus.OK
    assert span.context.trace_id == span.context.span_id


@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_tracer_exports_error_correctly_on_span(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    try:
        with tracer.span("name"):
            delay()
            raise SpecificTestException
    except SpecificTestException:
        pass
    delay()

    unified_format = tracer.export_for_viewing()

    assert len(unified_format) == 1
    span = unified_format[0]
    assert span.name == "name"
    assert span.parent_id is None
    assert span.start_time < span.end_time < utc_now()
    assert span.attributes.type == SpanType.SPAN
    assert span.status == SpanStatus.ERROR


@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_tracer_exports_error_correctly_on_taskspan(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    try:
        with tracer.task_span("name", input="input"):
            delay()
            raise SpecificTestException
    except SpecificTestException:
        pass
    delay()

    unified_format = tracer.export_for_viewing()

    assert len(unified_format) == 1
    span = unified_format[0]
    assert span.name == "name"
    assert span.parent_id is None
    assert span.start_time < span.end_time < utc_now()
    assert span.attributes.type == SpanType.TASK_SPAN
    assert span.status == SpanStatus.ERROR
    assert span.attributes.output is None


@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_tracer_export_nests_correctly(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    with tracer.span("name") as parent_span:
        delay()
        with parent_span.span("name-2") as child_span:
            delay()
            child_span.log("", value="")
            delay()
        delay()
    delay()

    unified_format = tracer.export_for_viewing()

    assert len(unified_format) == 2
    parent, child = unified_format[0], unified_format[1]
    if parent.parent_id is not None:
        parent, child = child, parent
    assert parent.name == "name"
    assert parent.parent_id is None
    assert parent.end_time >= child.end_time
    assert parent.start_time <= child.start_time
    assert child.name == "name-2"
    assert child.parent_id == parent.context.span_id
    assert len(child.events) == 1
    assert len(parent.events) == 0
    assert child.context.trace_id == parent.context.trace_id
    assert child.context.span_id != parent.context.span_id


@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_tracer_exports_unrelated_spans_correctly(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    tracer.span("name").end()
    tracer.span("name-2").end()

    unified_format = tracer.export_for_viewing()

    assert len(unified_format) == 2
    span_1, span_2 = unified_format[0], unified_format[1]

    assert span_1.parent_id is None
    assert span_2.parent_id is None

    assert span_1.context.trace_id != span_2.context.trace_id


@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_tracer_raises_if_open_span_is_exported(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    with tracer.span("name") as root_span:
        child_span = root_span.span("name-2")
        child_span.log("test_message", "test_body")

        with pytest.raises(RuntimeError):
            child_span.export_for_viewing()


@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_spans_cannot_be_used_as_context_twice(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    span = tracer.span("name")
    with span:
        pass
    with pytest.raises(ValueError):  # noqa: SIM117
        with span:
            pass


@pytest.mark.docker
@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_tracer_can_be_submitted_to_trace_viewer(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
    tracer_test_task: Task[str, str],
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    tracer_test_task.run(input="input", tracer=tracer)

    assert tracer.submit_to_trace_viewer()


@pytest.mark.skip("Not yet implemented")
@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_spans_cannot_be_closed_twice(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    span = tracer.span("name")
    span.end()
    span.end()


@pytest.mark.skip("Not yet implemented")
@pytest.mark.parametrize(
    "tracer_fixture",
    tracer_fixtures,
)
def test_tracer_can_not_log_on_closed_span(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    span = tracer.span("name")
    # ok
    span.log("test_message", "test_body")
    span.end()
    # not ok
    with pytest.raises(RuntimeError):
        span.log("test_message", "test_body")

    span = tracer.span("name")
    # ok
    with span:
        span.log("test_message", "test_body")
    # not ok
    with pytest.raises(RuntimeError):
        span.log("test_message", "test_body")

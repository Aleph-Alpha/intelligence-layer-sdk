import pytest
from pydantic import BaseModel

from intelligence_layer.core.tracer.tracer import (
    SpanStatus,
    SpanType,
    TaskSpanAttributes,
    Tracer,
    utc_now,
)
from tests.core.tracer.conftest import TestException


class DummyObject(BaseModel):
    content: str


tracer_fixtures = ["in_memory_tracer", "file_tracer"]


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
        temp_span.log("test", dummy_object)

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
    assert log.body == dummy_object
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
        task_span.record_output("output")

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
def test_tracer_exports_error_correctly(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    try:
        with tracer.span("name"):
            raise TestException
    except TestException:
        pass
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
def test_tracer_export_nests_correctly(
    tracer_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    tracer: Tracer = request.getfixturevalue(tracer_fixture)

    with tracer.span("name") as parent_span:
        with parent_span.span("name-2") as child_span:
            child_span.log("", value="")

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

    tracer.span("name")
    tracer.span("name-2")

    unified_format = tracer.export_for_viewing()

    assert len(unified_format) == 2
    span_1, span_2 = unified_format[0], unified_format[1]

    assert span_1.parent_id is None
    assert span_2.parent_id is None

    assert span_1.context.trace_id != span_2.context.trace_id

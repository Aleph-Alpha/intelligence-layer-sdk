import pytest

from intelligence_layer.core import CompositeTracer, InMemoryTracer, SpanStatus, Task
from tests.core.tracer.conftest import SpecificTestException


def test_composite_tracer(tracer_test_task: Task[str, str]) -> None:
    tracer_1 = InMemoryTracer()
    tracer_2 = InMemoryTracer()
    tracer_test_task.run(input="input", tracer=CompositeTracer([tracer_1, tracer_2]))

    trace_1 = tracer_1.export_for_viewing()[0]
    trace_2 = tracer_2.export_for_viewing()[0]
    assert trace_1.name == trace_2.name
    assert trace_1.attributes == trace_2.attributes
    assert trace_1.status == trace_2.status
    assert trace_1.context.trace_id != trace_2.context.trace_id
    assert trace_1.context.span_id != trace_2.context.span_id


def test_composite_tracer_can_get_span_status(
    tracer_test_task: Task[str, str],
) -> None:
    tracer_1 = InMemoryTracer()
    tracer_2 = InMemoryTracer()

    composite_tracer = CompositeTracer([tracer_1, tracer_2])

    with composite_tracer.span("test_name") as composite_span:
        composite_span.status_code == SpanStatus.OK


def test_composite_tracer_raises_for_inconsistent_span_status(
    tracer_test_task: Task[str, str],
) -> None:
    tracer_1 = InMemoryTracer()
    tracer_2 = InMemoryTracer()

    composite_tracer = CompositeTracer([tracer_1, tracer_2])

    with composite_tracer.span("test_name") as composite_span:
        spans = composite_span.tracers
        single_span = spans[0]
        try:
            with single_span:
                raise SpecificTestException
        except SpecificTestException:
            pass

        with pytest.raises(ValueError):
            composite_span.status_code

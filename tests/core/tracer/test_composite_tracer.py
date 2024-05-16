from intelligence_layer.core import (
    CompositeTracer,
    FileTracer,
    InMemorySpan,
    InMemoryTracer,
    Task,
)


def test_composite_tracer(test_task: Task[str, str]) -> None:
    tracer_1 = InMemoryTracer()
    tracer_2 = InMemoryTracer()
    test_task.run(input="input", tracer=CompositeTracer([tracer_1, tracer_2]))

    trace_1 = tracer_1.export_for_viewing()[0]
    trace_2 = tracer_2.export_for_viewing()[0]
    assert trace_1.name == trace_2.name
    assert trace_1.attributes == trace_2.attributes
    assert trace_1.status == trace_2.status
    assert trace_1.context.trace_id != trace_2.context.trace_id
    assert trace_1.context.span_id != trace_2.context.span_id
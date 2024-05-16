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
    assert trace_1.attributes = trace_2.attributes


def test_composite_tracer_id_consistent_across_children(
    file_tracer: FileTracer, test_task: Task[str, str]
) -> None:
    input = "input"
    tracer1 = InMemoryTracer()

    test_task.run(input, CompositeTracer([tracer1]))
    assert isinstance(tracer1.entries[0], InMemorySpan)
    assert tracer1.entries[0].id() == tracer1.entries[0].entries[0].id()

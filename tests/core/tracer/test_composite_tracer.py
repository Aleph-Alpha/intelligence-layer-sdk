from intelligence_layer.core import (
    CompositeTracer,
    FileTracer,
    InMemorySpan,
    InMemoryTracer,
    Task,
)


def test_composite_tracer_id_consistent_across_children(
    file_tracer: FileTracer, test_task: Task[str, str]
) -> None:
    input = "input"
    tracer1 = InMemoryTracer()

    test_task.run(input, CompositeTracer([tracer1]))
    assert isinstance(tracer1.entries[0], InMemorySpan)
    assert tracer1.entries[0].id() == tracer1.entries[0].entries[0].id()

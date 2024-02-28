from datetime import datetime
from typing import Generic, Optional, Sequence

from intelligence_layer.core.tracer.tracer import (
    PydanticSerializable,
    Span,
    SpanVar,
    TaskSpan,
    Tracer,
    TracerVar,
    utc_now,
)


class CompositeTracer(Tracer, Generic[TracerVar]):
    """A :class:`Tracer` that allows for recording to multiple tracers simultaneously.

    Each log-entry and span will be forwarded to all subtracers.

    Args:
        tracers: tracers that will be forwarded all subsequent log and span calls.

    Example:
        >>> from intelligence_layer.core import InMemoryTracer, FileTracer, CompositeTracer, Chunk
        >>> from intelligence_layer.use_cases import PromptBasedClassify, ClassifyInput

        >>> tracer_1 = InMemoryTracer()
        >>> tracer_2 = InMemoryTracer()
        >>> tracer = CompositeTracer([tracer_1, tracer_2])
        >>> task = PromptBasedClassify()
        >>> response = task.run(ClassifyInput(chunk=Chunk("Cool"), labels=frozenset({"label", "other label"})), tracer)
    """

    def __init__(self, tracers: Sequence[TracerVar]) -> None:
        assert len(tracers) > 0
        self.tracers = tracers

    def span(
        self,
        name: str,
        timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> "CompositeSpan[Span]":
        timestamp = timestamp or utc_now()
        trace_id = self.ensure_id(trace_id)
        return CompositeSpan(
            [tracer.span(name, timestamp, trace_id) for tracer in self.tracers]
        )

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> "CompositeTaskSpan":
        timestamp = timestamp or utc_now()
        trace_id = self.ensure_id(trace_id)
        return CompositeTaskSpan(
            [
                tracer.task_span(task_name, input, timestamp, trace_id)
                for tracer in self.tracers
            ]
        )


class CompositeSpan(Generic[SpanVar], CompositeTracer[SpanVar], Span):
    """A :class:`Span` that allows for recording to multiple spans simultaneously.

    Each log-entry and span will be forwarded to all subspans.

    Args:
        tracers: spans that will be forwarded all subsequent log and span calls.
    """

    def id(self) -> str:
        return self.tracers[0].id()

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        timestamp = timestamp or utc_now()
        for tracer in self.tracers:
            tracer.log(message, value, timestamp)

    def end(self, timestamp: Optional[datetime] = None) -> None:
        timestamp = timestamp or utc_now()
        for tracer in self.tracers:
            tracer.end(timestamp)


class CompositeTaskSpan(CompositeSpan[TaskSpan], TaskSpan):
    """A :class:`TaskSpan` that allows for recording to multiple TaskSpans simultaneously.

    Each log-entry and span will be forwarded to all subspans.

    Args:
        tracers: task spans that will be forwarded all subsequent log and span calls.
    """

    def record_output(self, output: PydanticSerializable) -> None:
        for tracer in self.tracers:
            tracer.record_output(output)

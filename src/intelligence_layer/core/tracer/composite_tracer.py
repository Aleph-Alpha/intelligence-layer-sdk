from collections.abc import Sequence
from datetime import datetime
from typing import Generic, Optional, TypeVar

from intelligence_layer.core.tracer.tracer import (
    Context,
    ExportedSpan,
    PydanticSerializable,
    Span,
    SpanStatus,
    TaskSpan,
    Tracer,
    utc_now,
)

TracerVar = TypeVar("TracerVar", bound=Tracer)

SpanVar = TypeVar("SpanVar", bound=Span)


class CompositeTracer(Tracer, Generic[TracerVar]):
    """A :class:`Tracer` that allows for recording to multiple tracers simultaneously.

    Each log-entry and span will be forwarded to all subtracers.

    Args:
        tracers: tracers that will be forwarded all subsequent log and span calls.

    Example:
        >>> from intelligence_layer.core import InMemoryTracer, FileTracer, CompositeTracer, TextChunk
        >>> from intelligence_layer.examples import PromptBasedClassify, ClassifyInput

        >>> tracer_1 = InMemoryTracer()
        >>> tracer_2 = InMemoryTracer()
        >>> tracer = CompositeTracer([tracer_1, tracer_2])
        >>> task = PromptBasedClassify()
        >>> response = task.run(ClassifyInput(chunk=TextChunk("Cool"), labels=frozenset({"label", "other label"})), tracer)
    """

    def __init__(self, tracers: Sequence[TracerVar]) -> None:
        assert len(tracers) > 0
        self.tracers = tracers
        self.context = tracers[0].context

    def span(
        self,
        name: str,
        timestamp: Optional[datetime] = None,
    ) -> "CompositeSpan[Span]":
        timestamp = timestamp or utc_now()
        return CompositeSpan([tracer.span(name, timestamp) for tracer in self.tracers])

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "CompositeTaskSpan":
        timestamp = timestamp or utc_now()
        return CompositeTaskSpan(
            [tracer.task_span(task_name, input, timestamp) for tracer in self.tracers]
        )

    def export_for_viewing(self) -> Sequence[ExportedSpan]:
        if len(self.tracers) > 0:
            return self.tracers[0].export_for_viewing()
        return []


class CompositeSpan(Generic[SpanVar], CompositeTracer[SpanVar], Span):
    """A :class:`Span` that allows for recording to multiple spans simultaneously.

    Each log-entry and span will be forwarded to all subspans.

    Args:
        tracers: spans that will be forwarded all subsequent log and span calls.
        context: Context of the parent. Defaults to None.
    """

    def __init__(
        self, tracers: Sequence[SpanVar], context: Optional[Context] = None
    ) -> None:
        CompositeTracer.__init__(self, tracers)
        Span.__init__(self, context=context)

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

    @property
    def status_code(self) -> SpanStatus:
        status_codes = {tracer.status_code for tracer in self.tracers}
        if len(status_codes) > 1:
            raise ValueError(
                "Inconsistent status of traces in composite tracer. Status of all traces should be the same but they are different."
            )
        return next(iter(status_codes))

    @status_code.setter
    def status_code(self, span_status: SpanStatus) -> None:
        for tracer in self.tracers:
            tracer.status_code = span_status


class CompositeTaskSpan(CompositeSpan[TaskSpan], TaskSpan):
    """A :class:`TaskSpan` that allows for recording to multiple TaskSpans simultaneously.

    Each log-entry and span will be forwarded to all subspans.

    Args:
        tracers: task spans that will be forwarded all subsequent log and span calls.
    """

    def record_output(self, output: PydanticSerializable) -> None:
        for tracer in self.tracers:
            tracer.record_output(output)

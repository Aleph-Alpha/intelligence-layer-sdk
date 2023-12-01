from datetime import datetime
from json import dumps
from typing import Generic, Iterable, Optional, Protocol, Sequence, TypeVar, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from intelligence_layer.connectors import JsonSerializable
from intelligence_layer.core.task import Input, Output
from intelligence_layer.core.tracer import (
    InMemorySpan,
    InMemoryTaskSpan,
    JsonSerializer,
    LogEntry,
    PydanticSerializable,
)

ExpectedOutput = TypeVar("ExpectedOutput", bound=PydanticSerializable)
Evaluation = TypeVar("Evaluation", bound=BaseModel)
AggregatedEvaluation = TypeVar("AggregatedEvaluation", bound=BaseModel)


class EvaluationException(BaseModel):
    """Captures an exception raised during evaluating a :class:`Task`.

    Attributes:
        error_message: String-representation of the exception.
    """

    error_message: str


Trace = Union["TaskSpanTrace", "SpanTrace", "LogTrace"]


class SpanTrace(BaseModel):
    """Represents traces contained by :class:`Span`

    Attributes:
        traces: The child traces.
        start: Start time of the span.
        end: End time of the span.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    traces: Sequence[Trace]
    start: datetime
    end: Optional[datetime]

    @staticmethod
    def from_span(span: InMemorySpan) -> "SpanTrace":
        return SpanTrace(
            name=span.name,
            traces=[_to_trace_entry(t) for t in span.entries],
            start=span.start_timestamp,
            end=span.end_timestamp,
        )

    def _rich_render_(self) -> Tree:
        tree = Tree(label="span")
        for log in self.traces:
            tree.add(log._rich_render_())
        return tree


class TaskSpanTrace(SpanTrace):
    """Represents traces contained by :class:`TaskSpan`

    Attributes:
        input: Input from the traced :class:`Task`.
        output: Output of the traced :class:`Task`.
    """

    model_config = ConfigDict(frozen=True)

    input: SerializeAsAny[JsonSerializable]
    output: SerializeAsAny[JsonSerializable]

    @staticmethod
    def from_task_span(task_span: InMemoryTaskSpan) -> "TaskSpanTrace":
        return TaskSpanTrace(
            name=task_span.name,
            traces=[_to_trace_entry(t) for t in task_span.entries],
            start=task_span.start_timestamp,
            end=task_span.end_timestamp,
            # RootModel.model_dump is declared to return the type of root, but actually returns
            # a JSON-like structure that fits to the JsonSerializable type
            input=JsonSerializer(root=task_span.input).model_dump(mode="json"),  # type: ignore
            output=JsonSerializer(root=task_span.output).model_dump(mode="json"),  # type: ignore
        )

    def _rich_render_(self) -> Tree:
        tree = Tree(label="task")
        tree.add(_render_log_value(self.input, "Input"))
        for log in self.traces:
            tree.add(log._rich_render_())
        tree.add(_render_log_value(self.output, "Output"))
        return tree

    def _ipython_display_(self) -> None:
        """Default rendering for Jupyter notebooks"""
        from rich import print

        print(self._rich_render_())


class LogTrace(BaseModel):
    """Represents a :class:`LogEntry`.

    Attributes:
        message: A description of the value that is being logged, such as the step in the
            :class:`Task` this is related to.
        value: The logged data. Can be anything that is serializable by Pydantic,
            which gives the tracers flexibility in how they store and emit the logs.
    """

    model_config = ConfigDict(frozen=True)

    message: str
    value: SerializeAsAny[JsonSerializable]

    @staticmethod
    def from_log_entry(entry: LogEntry) -> "LogTrace":
        return LogTrace(
            message=entry.message,
            # RootModel.model_dump is declared to return the type of root, but actually returns
            # a JSON-like structure that fits to the JsonSerializable type
            value=JsonSerializer(root=entry.value).model_dump(mode="json"),  # type: ignore
        )

    def _rich_render_(self) -> Panel:
        return _render_log_value(self.value, self.message)


def _render_log_value(value: JsonSerializable, title: str) -> Panel:
    return Panel(
        Syntax(dumps(value, indent=2), "json", word_wrap=True),
        title=title,
    )


def _to_trace_entry(entry: InMemoryTaskSpan | InMemorySpan | LogEntry) -> Trace:
    if isinstance(entry, InMemoryTaskSpan):
        return TaskSpanTrace.from_task_span(entry)
    elif isinstance(entry, InMemorySpan):
        return SpanTrace.from_span(entry)
    else:
        return LogTrace.from_log_entry(entry)


class ExampleOutput(BaseModel, Generic[Output]):
    example_id: str
    output: Output


class ExampleResult(BaseModel, Generic[Evaluation]):
    """Result of a single evaluated :class:`Example`

    Created to persist the evaluation result in the repository.

    Attributes:
        example_id: Identifier of the :class:`Example` evaluated.
        result: If the evaluation was successful, evaluation's result,
            otherwise the exception raised during running or evaluating
            the :class:`Task`.
    """

    example_id: str
    result: SerializeAsAny[Evaluation | EvaluationException]


class ExampleTrace(BaseModel):
    """Result of a single evaluated :class:`Example`

    Created to persist the evaluation result in the repository.

    Attributes:
        example_id: Identifier of the :class:`Example` evaluated.
        trace: Generated when running the :class:`Task`.
    """

    example_id: str
    trace: TaskSpanTrace


class EvaluationFailed(Exception):
    def __init__(self, run_id: str, failed_count: int) -> None:
        super().__init__(
            f"Evaluation run {run_id} failed with {failed_count} failed examples."
        )


class EvaluationRunOverview(BaseModel, Generic[AggregatedEvaluation]):
    """Overview of the results of evaluating a :class:`Task` on a :class:`Dataset`.

    Created when running :meth:`Evaluator.evaluate_dataset`. Contains high-level information and statistics.

    Attributes:
        id: Identifier of the run.
        dataset_name: The name of the evaluated :class:`Dataset`, i.e. :attr:`Dataset.name`
        failed_evaluation_count: The number of examples where an exception was raised when running the task or
            evaluating the output
        successful_evaluation_count: The number of examples that where successfully evaluated
        start: The time when the evaluation run was started
        end: The time when the evaluation run ended
        statistics: Aggregated statistics of the run. Whatever is returned by :meth:`Evaluator.aggregate`
    """

    id: str
    dataset_name: str
    failed_evaluation_count: int
    successful_evaluation_count: int
    start: Optional[datetime]
    end: Optional[datetime]
    statistics: SerializeAsAny[AggregatedEvaluation]

    def raise_on_evaluation_failure(self) -> None:
        if self.failed_evaluation_count > 0:
            raise EvaluationFailed(self.id, self.failed_evaluation_count)


class Example(BaseModel, Generic[Input, ExpectedOutput]):
    """Example case used for evaluations.

    Attributes:
        input: Input for the :class:`Task`. Has to be same type as the input for the task used.
        expected_output: The expected output from a given example run.
            This will be used by the evaluator to compare the received output with.
        id: Identifier for the example, defaults to uuid.
    """

    input: Input
    expected_output: ExpectedOutput
    id: str = Field(default_factory=lambda: str(uuid4()))


class Dataset(Protocol[Input, ExpectedOutput]):
    """A dataset of examples used for evaluation of a :class:`Task`.

    Attributes:
        name: This a human readable identifier for a dataset.
        examples: The actual examples that a :class:`Task` will be evaluated on.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def examples(self) -> Iterable[Example[Input, ExpectedOutput]]:
        ...

    def example(self, example_id: str) -> Optional[Example[Input, ExpectedOutput]]:
        ...


class SequenceDataset(BaseModel, Dataset[Input, ExpectedOutput]):
    """A :class:`Dataset` that contains all examples in a sequence.

    We recommend using this when it is certain that all examples
    fit in memory.

    Attributes:
        name: This a human readable identifier for a :class:`Dataset`.
        examples: The actual examples that a :class:`Task` will be evaluated on.
    """

    name: str
    examples: Sequence[Example[Input, ExpectedOutput]]

    def example(self, example_id: str) -> Optional[Example[Input, ExpectedOutput]]:
        return next(
            (example for example in self.examples if example.id == example_id), None
        )

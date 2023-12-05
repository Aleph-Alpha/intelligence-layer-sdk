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
Evaluation = TypeVar("Evaluation", bound=BaseModel, covariant=True)
AggregatedEvaluation = TypeVar("AggregatedEvaluation", bound=BaseModel, covariant=True)


class FailedExampleRun(BaseModel):
    """Captures an exception raised when running a single example with a :class:`Task`.

    Attributes:
        error_message: String-representation of the exception.
    """

    error_message: str

    @staticmethod
    def from_exception(exception: Exception) -> "FailedExampleRun":
        return FailedExampleRun(error_message=f"{type(exception)}: {str(exception)}")


class FailedExampleEvaluation(BaseModel):
    """Captures an exception raised when evaluating an :class:`ExampleOutput`.

    Attributes:
        error_message: String-representation of the exception.
    """

    error_message: str

    @staticmethod
    def from_exception(exception: Exception) -> "FailedExampleEvaluation":
        return FailedExampleEvaluation(
            error_message=f"{type(exception)}: {str(exception)}"
        )


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
    """Output of a single evaluated :class:`Example`

    Created to persist the output of an individual example in the repository.

    Attributes:
        example_id: Identifier of the :class:`Example`.
        output: Generated when running the :class:`Task`.

    Generics:
        Output: Interface of the output returned by the task.
    """

    example_id: str
    output: Output | FailedExampleRun


class ExampleTrace(BaseModel):
    """Trace of a single evaluated :class:`Example`

    Created to persist the trace of an individual example in the repository.

    Attributes:
        example_id: Identifier of the :class:`Example`.
        trace: Generated when running the :class:`Task`.
    """

    example_id: str
    trace: TaskSpanTrace


class RunOverview(BaseModel):
    """Overview of the run of a :class:`Task` on a :class:`Dataset`.

    Attributes:
        dataset_name: Identifier of the dataset run.
        id: The unique identifier of this run.
        start: The time when the run was started
        end: The time when the run ended
        failed_example_count: The number of examples where an exception was raised when running the task.
        successful_example_count: The number of examples that where successfully run.
    """

    dataset_name: str
    id: str
    start: datetime
    end: datetime
    failed_example_count: int
    successful_example_count: int


class ExampleEvaluation(BaseModel, Generic[Evaluation]):
    """Evaluation of a single evaluated :class:`Example`

    Created to persist the evaluation result in the repository.

    Attributes:
        example_id: Identifier of the :class:`Example` evaluated.
        result: If the evaluation was successful, evaluation's result,
            otherwise the exception raised during running or evaluating
            the :class:`Task`.

    Generics:
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
    """

    example_id: str
    result: SerializeAsAny[Evaluation | FailedExampleEvaluation]


class EvaluationOverview(BaseModel):
    """Overview of the unaggregated results of evaluating a :class:`Task` on a :class:`Dataset`.

    Attributes:
        run_overview: Overview of the run that was evaluated.
        id: The unique identifier of this evaluation.
        start: The time when the evaluation run was started
        end: The time when the evaluation run ended
        failed_evaluation_count: The number of examples where an exception was raised when evaluating the output.
        successful_evaluation_count: The number of examples that where successfully evaluated.
    """

    run_overview: RunOverview
    id: str
    start: Optional[datetime]
    end: Optional[datetime]
    failed_evaluation_count: int
    successful_evaluation_count: int


class EvaluationFailed(Exception):
    def __init__(self, eval_id: str, failed_count: int) -> None:
        super().__init__(
            f"Evaluation {eval_id} failed with {failed_count} failed examples."
        )


class EvaluationRunOverview(BaseModel, Generic[AggregatedEvaluation]):
    """Complete overview of the results of evaluating a :class:`Task` on a :class:`Dataset`.

    Created when running :meth:`Evaluator.evaluate_dataset`. Contains high-level information and statistics.

    Attributes:
        evaluation_overview: Overview of the evaluation of one run, includes evaluation id.
        statistics: Aggregated statistics of the run. Whatever is returned by :meth:`Evaluator.aggregate`
    """

    evaluation_overview: EvaluationOverview
    statistics: SerializeAsAny[AggregatedEvaluation]

    @property
    def id(self) -> str:
        return self.evaluation_overview.id

    @property
    def run_id(self) -> str:
        return self.evaluation_overview.run_overview.id

    @property
    def failed_count(self) -> int:
        return (
            self.evaluation_overview.failed_evaluation_count
            + self.evaluation_overview.run_overview.failed_example_count
        )

    @property
    def successful_count(self) -> int:
        return self.evaluation_overview.successful_evaluation_count

    def raise_on_evaluation_failure(self) -> None:
        if self.failed_count > 0:
            raise EvaluationFailed(self.id, self.failed_count)


class Example(BaseModel, Generic[Input, ExpectedOutput]):
    """Example case used for evaluations.

    Attributes:
        input: Input for the :class:`Task`. Has to be same type as the input for the task used.
        expected_output: The expected output from a given example run.
            This will be used by the evaluator to compare the received output with.
        id: Identifier for the example, defaults to uuid.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
    """

    input: Input
    expected_output: ExpectedOutput
    id: str = Field(default_factory=lambda: str(uuid4()))


class Dataset(Protocol[Input, ExpectedOutput]):
    """A dataset of examples used for evaluation of a :class:`Task`.

    Attributes:
        name: This a human readable identifier for a dataset.
        examples: The actual examples that a :class:`Task` will be evaluated on.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def examples(self) -> Iterable[Example[Input, ExpectedOutput]]:
        ...

    def example(self, example_id: str) -> Optional[Example[Input, ExpectedOutput]]:
        ...


class SequenceDataset(BaseModel, Generic[Input, ExpectedOutput]):
    """A :class:`Dataset` that contains all examples in a sequence.

    We recommend using this when it is certain that all examples
    fit in memory.

    Attributes:
        name: This a human readable identifier for a :class:`Dataset`.
        examples: The actual examples that a :class:`Task` will be evaluated on.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
    """

    name: str
    examples: Sequence[Example[Input, ExpectedOutput]]

    def example(self, example_id: str) -> Optional[Example[Input, ExpectedOutput]]:
        return next(
            (example for example in self.examples if example.id == example_id), None
        )

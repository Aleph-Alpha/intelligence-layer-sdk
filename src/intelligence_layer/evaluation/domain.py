from datetime import datetime
from json import dumps
from typing import Generic, Iterable, Optional, Sequence, TypeVar, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from intelligence_layer.connectors import JsonSerializable
from intelligence_layer.core import (
    InMemorySpan,
    InMemoryTaskSpan,
    Input,
    JsonSerializer,
    LogEntry,
    Output,
    PydanticSerializable,
)

ExpectedOutput = TypeVar("ExpectedOutput", bound=PydanticSerializable)
Evaluation = TypeVar("Evaluation", bound=BaseModel, covariant=True)
AggregatedEvaluation = TypeVar("AggregatedEvaluation", bound=BaseModel, covariant=True)


class Dataset(BaseModel):
    """Represents a dataset linked to multiple examples

    Attributes:
        id: Dataset ID.
        name: A short name of the dataset.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str


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

    Created to persist the output (including failures) of an individual example in the repository.

    Attributes:
        run_id: Identifier of the run that created the output.
        example_id: Identifier of the :class:`Example` that provided the input for generating the output.
        output: Generated when running the :class:`Task`. When the running the task
            failed this is an :class:`FailedExampleRun`.

    Generics:
        Output: Interface of the output returned by the task.
    """

    run_id: str
    example_id: str
    output: Output | FailedExampleRun

    def __str__(self) -> str:
        return (
            f"Run ID={self.run_id}\n"
            f"Example ID={self.example_id}\n"
            f'Output="{self.output}"\n'
        )


class SuccessfulExampleOutput(BaseModel, Generic[Output]):
    """Successful output of a single evaluated :class:`Example`

    Attributes:
        run_id: Identifier of the run that created the output.
        example_id: Identifier of the :class:`Example`.
        output: Generated when running the :class:`Task`. This represent only
            the output of an successful run.

    Generics:
        Output: Interface of the output returned by the task.
    """

    run_id: str
    example_id: str
    output: Output

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"Run ID={self.run_id}\n"
            f"Example ID={self.example_id}\n"
            f'Output="{self.output}"\n'
        )


class ExampleTrace(BaseModel):
    """Trace of a single evaluated :class:`Example`

    Created to persist the trace of an individual example in the repository.

    Attributes:
        run_id: The unique identifier of the run.
        example_id: Identifier of the :class:`Example`.
        trace: Generated when running the :class:`Task`.
    """

    run_id: str
    example_id: str
    trace: TaskSpanTrace


class RunOverview(BaseModel, frozen=True):
    """Overview of the run of a :class:`Task` on a dataset.

    Attributes:
        dataset_id: Identifier of the dataset run.
        id: The unique identifier of this run.
        start: The time when the run was started
        end: The time when the run ended
        failed_example_count: The number of examples where an exception was raised when running the task.
        successful_example_count: The number of examples that where successfully run.
        description: Human-readable of the runner that run the task.
    """

    dataset_id: str
    id: str
    start: datetime
    end: datetime
    failed_example_count: int
    successful_example_count: int
    description: str

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"Run Overview ID={self.id}\n"
            f"Dataset ID={self.dataset_id}\n"
            f"Start time={self.start}\n"
            f"End time={self.end}\n"
            f"Failed example count={self.failed_example_count}\n"
            f"Successful example count={self.successful_example_count}\n"
            f'Description="{self.description}"\n'
        )


class ExampleEvaluation(BaseModel, Generic[Evaluation]):
    """Evaluation of a single evaluated :class:`Example`

    Created to persist the evaluation result in the repository.

    Attributes:
        evaluation_id: Identifier of the run the evaluated example belongs to.
        example_id: Identifier of the :class:`Example` evaluated.
        result: If the evaluation was successful, evaluation's result,
            otherwise the exception raised during running or evaluating
            the :class:`Task`.

    Generics:
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
    """

    evaluation_id: str
    example_id: str
    result: SerializeAsAny[Evaluation | FailedExampleEvaluation]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"Evaluation ID={self.evaluation_id}\n"
            f"Example ID={self.example_id}\n"
            f"Result={self.result}\n"
        )


class EvaluationOverview(BaseModel, frozen=True):
    """Overview of the un-aggregated results of evaluating a :class:`Task` on a dataset.

    Attributes:
        run_overviews: Overviews of the runs that were evaluated.
        id: The unique identifier of this evaluation.
        start: The time when the evaluation run was started
        description: human-readable for the evaluator that created the evaluation
    """

    run_overviews: frozenset[RunOverview]
    id: str
    start: Optional[datetime]
    description: str

    def __str__(self) -> str:
        run_overview_str: str = "Run Overviews={\n"
        for overview in self.run_overviews:
            run_overview_str += f"{overview}"
        run_overview_str += "}\n"

        return (
            f"Evaluation Overview ID={self.id}\n"
            f"Start time={self.start}\n"
            f'Description="{self.description}"\n'
            f"{run_overview_str}"
        )


class EvaluationFailed(Exception):
    def __init__(self, evaluation_id: str, failed_count: int) -> None:
        super().__init__(
            f"Evaluation {evaluation_id} failed with {failed_count} failed examples."
        )

    end: datetime
    failed_example_count: int
    successful_example_count: int
    description: str


class AggregationOverview(BaseModel, Generic[AggregatedEvaluation], frozen=True):
    """Complete overview of the results of evaluating a :class:`Task` on a dataset.

    Created when running :meth:`Evaluator.eval_and_aggregate_runs`. Contains high-level information and statistics.

    Attributes:
        evaluation_overviews: :class:`EvaluationOverview`s used for aggregation.
        id: Aggregation overview ID.
        start: Start timestamp of the aggregation.
        end: End timestamp of the aggregation.
        end: The time when the evaluation run ended
        successful_evaluation_count: The number of examples that where successfully evaluated.
        crashed_during_evaluation_count: The number of examples that crashed during evaluation.
        failed_evaluation_count: The number of examples that crashed during evaluation
            plus the number of examples that failed to produce an output for evaluation.
        run_ids: IDs of all :class:`RunOverview`s from all linked :class:`EvaluationOverview`s.
        description: A short description.
        statistics: Aggregated statistics of the run. Whatever is returned by :meth:`Evaluator.aggregate`
    """

    evaluation_overviews: frozenset[EvaluationOverview]
    id: str
    start: datetime
    end: datetime
    successful_evaluation_count: int
    crashed_during_evaluation_count: int
    description: str
    statistics: SerializeAsAny[AggregatedEvaluation]

    @property
    def run_ids(self) -> Sequence[str]:
        return [overview.id for overview in self.run_overviews()]

    def run_overviews(self) -> Iterable[RunOverview]:
        return set(
            run_overview
            for evaluation_overview in self.evaluation_overviews
            for run_overview in evaluation_overview.run_overviews
        )

    @property
    def failed_evaluation_count(self) -> int:
        return self.crashed_during_evaluation_count + sum(
            run_overview.failed_example_count for run_overview in self.run_overviews()
        )

    def raise_on_evaluation_failure(self) -> None:
        if self.crashed_during_evaluation_count > 0:
            raise EvaluationFailed(self.id, self.crashed_during_evaluation_count)


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

    def __str__(self) -> str:
        return (
            f"Example ID={self.id}\n"
            f"Input={self.input}\n"
            f'Expected output="{self.expected_output}"\n'
        )

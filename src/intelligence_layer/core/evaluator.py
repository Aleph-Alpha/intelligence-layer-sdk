from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from json import dumps
from typing import (
    Generic,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    cast,
    final,
)
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from tqdm import tqdm

from intelligence_layer.connectors.document_index.document_index import JsonSerializable
from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer import (
    InMemorySpan,
    InMemoryTaskSpan,
    InMemoryTracer,
    JsonSerializer,
    LogEntry,
    PydanticSerializable,
    Tracer,
)

ExpectedOutput = TypeVar("ExpectedOutput", bound=PydanticSerializable)
Evaluation = TypeVar("Evaluation", bound=BaseModel)
AggregatedEvaluation = TypeVar("AggregatedEvaluation", bound=PydanticSerializable)


class Example(BaseModel, Generic[Input, ExpectedOutput]):
    """Example case used for evaluations.

    Attributes:
        input: Input for the task. Has to be same type as the input for the task used.
        expected_output: The expected output from a given example run.
            This will be used by the evaluator to compare the received output with.
        ident: Identifier for the example, defaults to uuid.
    """

    input: Input
    expected_output: ExpectedOutput
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))


class Dataset(Protocol, Generic[Input, ExpectedOutput]):
    """A dataset of examples used for evaluation of a task.

    Attributes:
        name: This a human readable identifier for a dataset.
        examples: The actual examples that a task will be evaluated on.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def examples(self) -> Iterable[Example[Input, ExpectedOutput]]:
        ...


class SequenceDataset(BaseModel, Generic[Input, ExpectedOutput]):
    name: str
    examples: Sequence[Example[Input, ExpectedOutput]]


class EvaluationException(BaseModel):
    error_message: str


TraceEntry = Union["TaskTrace", "SpanTrace", "TraceLog"]


class SpanTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    traces: Sequence[TraceEntry]
    start: datetime
    end: Optional[datetime]

    @staticmethod
    def from_span(span: InMemorySpan) -> "SpanTrace":
        return SpanTrace(
            traces=[to_trace_entry(t) for t in span.entries],
            start=span.start_timestamp,
            end=span.end_timestamp,
        )

    def _rich_render_(self) -> Tree:
        tree = Tree(label="span")
        for log in self.traces:
            tree.add(log._rich_render_())
        return tree


class TaskTrace(SpanTrace):
    model_config = ConfigDict(frozen=True)

    input: SerializeAsAny[JsonSerializable]
    output: SerializeAsAny[JsonSerializable]

    @staticmethod
    def from_task_span(task_span: InMemoryTaskSpan) -> "TaskTrace":
        return TaskTrace(
            traces=[to_trace_entry(t) for t in task_span.entries],
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


class TraceLog(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    value: SerializeAsAny[JsonSerializable]

    @staticmethod
    def from_log_entry(entry: LogEntry) -> "TraceLog":
        return TraceLog(
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


def to_trace_entry(entry: InMemoryTaskSpan | InMemorySpan | LogEntry) -> TraceEntry:
    if isinstance(entry, InMemoryTaskSpan):
        return TaskTrace.from_task_span(entry)
    elif isinstance(entry, InMemorySpan):
        return SpanTrace.from_span(entry)
    else:
        return TraceLog.from_log_entry(entry)


class ExampleResult(BaseModel, Generic[Evaluation]):
    result: SerializeAsAny[Evaluation | EvaluationException]
    trace: TaskTrace


class EvaluationRepository(ABC):
    @abstractmethod
    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        ...

    @abstractmethod
    def store_example_result(
        self, run_id: str, result: ExampleResult[Evaluation]
    ) -> None:
        ...


class InMemoryEvaluationRepository(EvaluationRepository):
    class SerializedExampleResult(BaseModel):
        is_exception: bool
        json_result: str
        trace: TaskTrace

    _example_results: dict[str, list[str]] = defaultdict(list)

    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        def to_example_result(
            serialized_example: InMemoryEvaluationRepository.SerializedExampleResult,
        ) -> ExampleResult[Evaluation]:
            return (
                ExampleResult(
                    result=evaluation_type.model_validate_json(
                        serialized_example.json_result
                    ),
                    trace=serialized_example.trace,
                )
                if not serialized_example.is_exception
                else ExampleResult(
                    result=EvaluationException.model_validate_json(
                        serialized_example.json_result
                    ),
                    trace=serialized_example.trace,
                )
            )

        result_jsons = self._example_results.get(run_id, [])
        return [
            to_example_result(
                self.SerializedExampleResult.model_validate_json(json_str)
            )
            for json_str in result_jsons
        ]

    def store_example_result(
        self, run_id: str, result: ExampleResult[Evaluation]
    ) -> None:
        json_result = self.SerializedExampleResult(
            json_result=JsonSerializer(root=result.result).model_dump_json(),
            is_exception=isinstance(result.result, EvaluationException),
            trace=result.trace,
        )
        self._example_results[run_id].append(json_result.model_dump_json())


class EvaluationRunOverview(BaseModel, Generic[AggregatedEvaluation]):
    id: str
    statistics: SerializeAsAny[AggregatedEvaluation]


class Evaluator(
    ABC, Generic[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]
):
    """Base evaluator interface. This should run certain evaluation steps for some job.

    We suggest supplying a `Task` in the `__init__` method and running it in the `evaluate` method.

    Generics:
        Input: Interface to be passed to the task that shall be evaluated.
        ExpectedOutput: Output that is expected from the task run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated task.
        AggregatedEvaluation: The aggregated results of an evaluation run with a dataset.
    """

    def __init__(
        self, task: Task[Input, Output], repository: EvaluationRepository
    ) -> None:
        self.task = task
        self.repository = repository

    @abstractmethod
    def do_evaluate(
        self,
        input: Input,
        output: Output,
        expected_output: ExpectedOutput,
    ) -> Evaluation:
        """Executes the evaluation for this use-case.

        The implementation of this method is responsible for running a task (usually supplied by the __init__ method)
        and making any comparisons relevant to the evaluation.
        Based on the results, it should create an `Evaluation` class with all the metrics and return it.

        Args:
            input: The input that was passed to the task to produce the output
            output: Output of the task that shall be evaluated
            expected_output: Output that is expected from the task.
        Returns:
            Interface of the metrics that come from the evaluated task.
        """
        pass

    def _evaluate(
        self,
        run_id: str,
        input: Input,
        tracer: Tracer,
        expected_output: ExpectedOutput,
    ) -> Evaluation | EvaluationException:
        eval_tracer = InMemoryTracer()
        result = self.evaluate(input, expected_output, eval_tracer)
        example_result = ExampleResult(
            result=result,
            trace=TaskTrace.from_task_span(
                cast(InMemoryTaskSpan, eval_tracer.entries[0])
            ),
        )
        self.repository.store_example_result(run_id=run_id, result=example_result)
        return example_result.result

    @final
    def evaluate(
        self, input: Input, expected_output: ExpectedOutput, tracer: Tracer
    ) -> Evaluation | EvaluationException:
        output = self.task.run(input, tracer)
        try:
            return self.do_evaluate(input, output, expected_output)
        except Exception as e:
            return EvaluationException(error_message=str(e))

    @final
    def evaluate_dataset(
        self, dataset: Dataset[Input, ExpectedOutput], tracer: Tracer
    ) -> EvaluationRunOverview[AggregatedEvaluation]:
        """Evaluates an entire datasets in a threaded manner and aggregates the results into an `AggregatedEvaluation`.

        This will call the `run` method for each example in the dataset.
        Finally, it will call the `aggregate` method and return the aggregated results.

        Args:
            dataset: Dataset that will be used to evaluate a task.
            tracer: tracer used for tracing.
        Returns:
            The aggregated results of an evaluation run with a dataset.
        """

        run_id = str(uuid4())
        with ThreadPoolExecutor(max_workers=10) as executor:
            evaluations = tqdm(
                executor.map(
                    lambda idx_example: self._evaluate(
                        run_id,
                        idx_example.input,
                        tracer,
                        idx_example.expected_output,
                    ),
                    dataset.examples,
                ),
                desc="Evaluating",
            )

        # collect errors with debug log
        statistics = self.aggregate(
            evaluation
            for evaluation in evaluations
            if not isinstance(evaluation, EvaluationException)
        )

        return EvaluationRunOverview(id=run_id, statistics=statistics)

    @abstractmethod
    def aggregate(self, evaluations: Iterable[Evaluation]) -> AggregatedEvaluation:
        """`Evaluator`-specific method for aggregating individual `Evaluations` into report-like `Aggregated Evaluation`.

        This method is responsible for taking the results of an evaluation run and aggregating all the results.
        It should create an `AggregatedEvaluation` class and return it at the end.

        Args:
            evalautions: The results from running `evaluate_dataset` with a task.
        Returns:
            The aggregated results of an evaluation run with a dataset.
        """
        pass

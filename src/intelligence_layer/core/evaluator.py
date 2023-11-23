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

from intelligence_layer.connectors import JsonSerializable
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
AggregatedEvaluation = TypeVar("AggregatedEvaluation", bound=BaseModel)


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


class Dataset(Protocol, Generic[Input, ExpectedOutput]):
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


class SequenceDataset(BaseModel, Generic[Input, ExpectedOutput]):
    """A :class:`Dataset` that contains all examples in a sequence.

    We recommend using this when it is certain that all examples
    fit in memory.

    Attributes:
        name: This a human readable identifier for a :class:`Dataset`.
        examples: The actual examples that a :class:`Task` will be evaluated on.
    """

    name: str
    examples: Sequence[Example[Input, ExpectedOutput]]


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

    traces: Sequence[Trace]
    start: datetime
    end: Optional[datetime]

    @staticmethod
    def from_span(span: InMemorySpan) -> "SpanTrace":
        return SpanTrace(
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


class ExampleResult(BaseModel, Generic[Evaluation]):
    """Result of a single evaluated :class:`Example`

    Created to persist the evaluation result in the repository.

    Attributes:
        example_id: Identifier of the :class:`Example` evaluated.
        result: If the evaluation was successful, evaluation's result,
            otherwise the exception raised during running or evaluating
            the :class:`Task`.
        trace: Generated when running the :class:`Task`.
    """

    example_id: str
    result: SerializeAsAny[Evaluation | EvaluationException]
    trace: TaskSpanTrace


class EvaluationRunOverview(BaseModel, Generic[AggregatedEvaluation]):
    """Overview of the results of evaluating a :class:`Task` on a :class:`Dataset`.

    Created when running :func:`Evaluator.evaluate_dataset`. Contains high-level information and statistics.

    Attributes:
        id: Identifier of the run.
        statistics: Aggregated statistics of the run.
    """

    id: str
    # dataset_id: str
    # failed_evaluation_count: int
    # successful_evaluation_count: int
    # start: datetime
    # end: datetime
    statistics: SerializeAsAny[AggregatedEvaluation]


class EvaluationRepository(ABC):
    """Base evaluation repository interface.

    Provides methods to store and load evaluation results for individual examples
    of a run and the aggregated evaluation of said run.
    """

    @abstractmethod
    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        """Returns all :class:`ExampleResult` instances of a given run

        Args:
            run_id: Identifier of the run to obtain the results for.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            All :class:`ExampleResult` of the run. Will return an empty list if there's none.
        """
        ...

    @abstractmethod
    def evaluation_example_result(
        self, run_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleResult[Evaluation]]:
        """Returns an :class:`ExampleResult` of a given run by its id.

        Args:
            run_id: Identifier of the run to obtain the results for.
            example_id: Identifier of the :class:`ExampleResult` to be retrieved.
            evaluation_type: Type of evaluations that the `Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            :class:`ExampleResult` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_example_result(
        self, run_id: str, result: ExampleResult[Evaluation]
    ) -> None:
        """Stores an :class:`ExampleResult` for a run in the repository.

        Args:
            run_id: Identifier of the run.
            result: The result to be persisted.
        """
        ...

    @abstractmethod
    def evaluation_run_overview(
        self, run_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> Optional[EvaluationRunOverview[AggregatedEvaluation]]:
        """Returns an :class:`EvaluationRunOverview` of a given run by its id.

        Args:
            run_id: Identifier of the run to obtain the overview for.
            aggregation_type: Type of aggregations that the :class:`Evaluator` returned
                in :func:`Evaluator.aggregate`

        Returns:
            :class:`EvaluationRunOverview` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_evaluation_run_overview(
        self, overview: EvaluationRunOverview[AggregatedEvaluation]
    ) -> None:
        """Stores an :class:`EvaluationRunOverview` in the repository.

        Args:
            overview: The overview to be persisted.
        """
        ...


class InMemoryEvaluationRepository(EvaluationRepository):
    class SerializedExampleResult(BaseModel):
        example_id: str
        is_exception: bool
        json_result: str
        trace: TaskSpanTrace

    _example_results: dict[str, list[str]] = defaultdict(list)

    _run_overviews: dict[str, str] = dict()

    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        def to_example_result(
            serialized_example: InMemoryEvaluationRepository.SerializedExampleResult,
        ) -> ExampleResult[Evaluation]:
            return (
                ExampleResult(
                    example_id=serialized_example.example_id,
                    result=evaluation_type.model_validate_json(
                        serialized_example.json_result
                    ),
                    trace=serialized_example.trace,
                )
                if not serialized_example.is_exception
                else ExampleResult(
                    example_id=serialized_example.example_id,
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

    def evaluation_example_result(
        self, run_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> ExampleResult[Evaluation] | None:
        return next(
            (
                result
                for result in self.evaluation_run_results(run_id, evaluation_type)
                if result.example_id == example_id
            ),
            None,
        )

    def store_example_result(
        self, run_id: str, result: ExampleResult[Evaluation]
    ) -> None:
        json_result = self.SerializedExampleResult(
            json_result=JsonSerializer(root=result.result).model_dump_json(),
            is_exception=isinstance(result.result, EvaluationException),
            trace=result.trace,
            example_id=result.example_id,
        )
        self._example_results[run_id].append(json_result.model_dump_json())

    def store_evaluation_run_overview(
        self, overview: EvaluationRunOverview[AggregatedEvaluation]
    ) -> None:
        self._run_overviews[overview.id] = overview.model_dump_json()

    def evaluation_run_overview(
        self, run_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> EvaluationRunOverview[AggregatedEvaluation] | None:
        loaded_json = self._run_overviews.get(run_id)
        # mypy doesn't accept dynamic types as type parameter
        return (
            EvaluationRunOverview[aggregation_type].model_validate_json(loaded_json)  # type: ignore
            if loaded_json
            else None
        )


class Evaluator(
    ABC, Generic[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]
):
    """Base evaluator interface. This should run certain evaluation steps for some job.

    We suggest supplying a :class:`Task` in the `__init__` method and running it in the :func:`Evaluator.evaluate` method.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        Output: Type of the output of the :class:`Task` to be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
        AggregatedEvaluation: The aggregated results of an evaluation run with a :class:`Dataset`.
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

        The implementation of this method is responsible for running a :class:`Task` (usually
        supplied by the __init__ method) and making any comparisons relevant to the evaluation.
        Based on the results, it should create an `Evaluation` class with all the metrics and return it.

        Args:
            input: The input that was passed to the :class:`Task` to produce the output
            output: Output of the :class:`Task` that shall be evaluated
            expected_output: Output that is compared to the generated output.

        Returns:
            The metrics that come from the evaluated :class:`Task`.
        """
        pass

    def _evaluate_example(
        self,
        run_id: str,
        example: Example[Input, ExpectedOutput],
        tracer: Tracer,
    ) -> Evaluation | EvaluationException:
        eval_tracer = InMemoryTracer()
        result = self.evaluate(example.input, example.expected_output, eval_tracer)
        example_result = ExampleResult(
            example_id=example.id,
            result=result,
            trace=TaskSpanTrace.from_task_span(
                cast(InMemoryTaskSpan, eval_tracer.entries[0])
            ),
        )
        self.repository.store_example_result(run_id=run_id, result=example_result)
        return example_result.result

    @final
    def evaluate(
        self, input: Input, expected_output: ExpectedOutput, tracer: Tracer
    ) -> Evaluation | EvaluationException:
        """Evaluates a single example and returns an `Evaluation` or `EvaluationException`.

        This will call the `run` method for the :class:`Task` defined in the `__init__` method.
        Catches any errors that occur during :func:`Task.run` or :func:`Evaluator.do_evaluate`.

        Args:
            input: Input for the :class:`Task`. Has to be same type as the input for the task used.
            expected_output: The expected output for the run.
                This will be used by the evaluator to compare the received output with.
            tracer: :class:`Tracer` used for tracing.
        Returns:
            Result of the evaluation or exception in case an error during running
            the :class:`Task` or evaluation.
        """

        try:
            output = self.task.run(input, tracer)
            return self.do_evaluate(input, output, expected_output)
        except Exception as e:
            return EvaluationException(error_message=str(e))

    @final
    def evaluate_dataset(
        self, dataset: Dataset[Input, ExpectedOutput], tracer: Tracer
    ) -> EvaluationRunOverview[AggregatedEvaluation]:
        """Evaluates an entire :class:`Dataset` in a threaded manner and aggregates the results into an `AggregatedEvaluation`.

        This will call the `run` method for each example in the :class:`Dataset`.
        Finally, it will call the `aggregate` method and return the aggregated results.

        Args:
            dataset: Dataset that will be used to evaluate a :class:`Task`.
            tracer: tracer used for tracing.
        Returns:
            The aggregated results of an evaluation run with a dataset.
        """

        run_id = str(uuid4())
        with ThreadPoolExecutor(max_workers=10) as executor:
            evaluations = tqdm(
                executor.map(
                    lambda example: self._evaluate_example(
                        run_id,
                        example,
                        tracer,
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

        run_overview = EvaluationRunOverview(id=run_id, statistics=statistics)
        self.repository.store_evaluation_run_overview(run_overview)
        return run_overview

    @abstractmethod
    def aggregate(self, evaluations: Iterable[Evaluation]) -> AggregatedEvaluation:
        """`Evaluator`-specific method for aggregating individual `Evaluations` into report-like `Aggregated Evaluation`.

        This method is responsible for taking the results of an evaluation run and aggregating all the results.
        It should create an `AggregatedEvaluation` class and return it at the end.

        Args:
            evalautions: The results from running `evaluate_dataset` with a :class:`Task`.
        Returns:
            The aggregated results of an evaluation run with a :class:`Dataset`.
        """
        pass

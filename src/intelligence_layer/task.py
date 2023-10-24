from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import functools
from itertools import islice
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Mapping,
    NewType,
    Optional,
    Sequence,
    TypeVar,
    Protocol,
    Union,
    runtime_checkable,
    Callable,
    Self,
)
from uuid import uuid4
from pydantic import (
    BaseModel,
    RootModel,
    Field,
    SerializeAsAny,
)
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from typing_extensions import TypeAliasType

from tqdm import tqdm  # type: ignore


if TYPE_CHECKING:
    PydanticSerializable = (
        int
        | float
        | str
        | Sequence["PydanticSerializable"]
        | Mapping[str, "PydanticSerializable"]
        | None
        | bool
        | BaseModel
    )
else:
    PydanticSerializable = TypeAliasType(
        "PydanticSerializable",
        int
        | float
        | str
        | Sequence["PydanticSerializable"]
        | Mapping[str, "PydanticSerializable"]
        | None
        | bool
        | BaseModel,
    )

Chunk = NewType("Chunk", str)
"""Segment of a larger text.

This type infers that the string is smaller than the context size of the model where it is used.

LLMs can't process documents larger than their context size.
To handle this, documents have to be split up into smaller segments that fit within their context size.
These smaller segments are referred to as chunks.

"""


@runtime_checkable
class DebugLogger(Protocol):
    """A protocol for instrumenting `Task`s with structured logging.

    A logger needs to provide a way to collect an individual log, which should be serializable, and
    a way to generate nested loggers, so that sub-tasks can emit logs that are grouped together.

    Each `DebugLogger` is given a `name` to distinguish them from each other, and for nested logs.

    Implementations of how logs are collected and stored may differ. Refer to the individual
    documentation of each implementation to see how to use the resulting logger.
    """

    def log(self, message: str, value: PydanticSerializable) -> None:
        """Record a log of relevant information as part of a step within a task.

        By default, the `Input` and `Output` of each `Task` are logged automatically, but you can
        log anything else that seems relevant to understanding the output of a given task.

        Args:
            message: A description of the value you are logging, such as the step in the task this
                is related to.
            value: The relevant data you want to log. Can be anything that is serializable by
                Pydantic, which gives the loggers flexibility in how they store and emit the logs.
        """
        ...

    def span(self, name: str) -> "Span":
        """Generate a span from the current logging instance.

        Each logger implementation can decide on how it wants to represent this, but they should
        all allow for representing logs of a child task within the scope of the current task.

        Args:
            name: A descriptive name of what this span will contain logs about.

        Returns:
            An instance of something that meets the protocol of Span.
        """
        ...

    def task_span(self, task_name: str, input: PydanticSerializable) -> "TaskSpan":
        """Generate a task-specific span from the current logging instance.

        Each logger implementation can decide on how it wants to represent this, but they should
        all allow for representing logs of a span within the context of a parent span.

        Args:
            task_name: The name of the task that is being logged
            input: The input for the task that is being logged.

        Returns:
            An instance of something that also meets the protocol of DebugLogger. Most likely, it
            will create an instance of the same type, but this is dependent on the actual
            implementation.
        """
        ...


@runtime_checkable
class Span(AbstractContextManager["Span"], DebugLogger, Protocol):
    """A protocol for instrumenting logs nested within a span of time. Groups logs by some logical
    step.

    The implementation should also be a Context Manager, to capture the span of duration of
    execution.

    Implementations of how logs are collected and stored may differ. Refer to the individual
    documentation of each implementation to see how to use the resulting logger.
    """

    ...


@runtime_checkable
class TaskSpan(AbstractContextManager["TaskSpan"], DebugLogger, Protocol):
    """A protocol for instrumenting a `Task`'s input, output, and nested logs.

    Most likely, generating this task logger will capture the `Task`'s input, as well as the task
    name.

    The implementation should also be a Context Manager, to capture the span of duration of
    task execution.

    Implementations of how logs are collected and stored may differ. Refer to the individual
    documentation of each implementation to see how to use the resulting logger.
    """

    def record_output(self, output: PydanticSerializable) -> None:
        """Record a `Task`'s output. Since a Context Manager can't provide this in the `__exit__`
        method, output should be captured once it is generated.

        This should be handled automatically within the execution of the task.

        Args:
            output: The output of the task that is being logged.
        """
        ...


class NoOpDebugLogger:
    """A no-op logger. Useful for cases, like testing, where a logger is needed for a task, but you
    don't have a need to collect or inspect the actual logs.

    All calls to `log` won't actually do anything.
    """

    def log(self, message: str, value: PydanticSerializable) -> None:
        """Record a log of relevant information as part of a step within a task.

        By default, the `Input` and `Output` of each `Task` are logged automatically, but you can
        log anything else that seems relevant to understanding the output of a given task.

        Args:
            message: A description of the value you are logging, such as the step in the task this
                is related to.
            value: The relevant data you want to log. Can be anything that is serializable by
                Pydantic, which gives the loggers flexibility in how they store and emit the logs.
        """
        pass

    def span(self, name: str) -> "NoOpTaskSpan":
        """Generate a sub-logger from the current logging instance.

        Args:
            name: A descriptive name of what this child logger will contain logs about.

        Returns:
            Another `NoOpDebugLogger`
        """
        return NoOpTaskSpan()

    def task_span(self, task_name: str, input: PydanticSerializable) -> "NoOpTaskSpan":
        """Generate a task-specific span from the current logging instance.


        Args:
            task_name: The name of the task that is being logged
            input: The input for the task that is being logged.

        Returns:
            A `NoOpTaskSpan`
        """

        return NoOpTaskSpan()


class NoOpTaskSpan(NoOpDebugLogger, AbstractContextManager["NoOpTaskSpan"]):
    def record_output(self, output: PydanticSerializable) -> None:
        """Record a `Task`'s output. Since a Context Manager can't provide this in the `__exit__`
        method, output should be captured once it is generated.

        This should be handled automatically within the execution of the task.

        Args:
            output: The output of the task that is being logged.
        """
        pass

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        pass


class JsonSerializer(RootModel[PydanticSerializable]):
    root: SerializeAsAny[PydanticSerializable]


def _render_log_value(value: PydanticSerializable, title: str) -> Panel:
    value = value if isinstance(value, BaseModel) else JsonSerializer(root=value)
    return Panel(
        Syntax(
            value.model_dump_json(indent=2, exclude_defaults=True),
            "json",
            word_wrap=True,
        ),
        title=title,
    )


class LogEntry(BaseModel):
    """An individual log entry, currently used to represent individual logs by the
    `InMemoryDebugLogger`.

    Attributes:
        message: A description of the value you are logging, such as the step in the task this
            is related to.
        value: The relevant data you want to log. Can be anything that is serializable by
            Pydantic, which gives the loggers flexibility in how they store and emit the logs.
        timestamp: The time that the log was emitted.
    """

    message: str
    value: SerializeAsAny[PydanticSerializable]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def _rich_render_(self) -> Panel:
        """Renders the debug log via classes in the `rich` package"""
        return _render_log_value(self.value, self.message)

    def _ipython_display_(self) -> None:
        """Default rendering for Jupyter notebooks"""
        from rich import print

        print(self._rich_render_())


class InMemoryDebugLogger(BaseModel):
    """Collects log entries in a nested structure, and keeps them in memory.

    If desired, the structure is serializable with Pydantic, so you can write out the JSON
    representation to a file, or return via an API, or something similar.

    Attributes:
        name: A descriptive name of what the logger contains log entries about.
        logs: A sequential list of log entries and/or nested InMemoryDebugLoggers with their own
            log entries.
    """

    name: str
    logs: list[Union[LogEntry, "InMemoryDebugLogger", "InMemoryTaskSpan"]] = []

    def log(self, message: str, value: PydanticSerializable) -> None:
        """Record a log of relevant information as part of a step within a task.

        By default, the `Input` and `Output` of each `Task` are logged automatically, but you can
        log anything else that seems relevant to understanding the output of a given task.

        Args:
            message: A description of the value you are logging, such as the step in the task this
                is related to.
            value: The relevant data you want to log. Can be anything that is serializable by
                Pydantic, which gives the loggers flexibility in how they store and emit the logs.
        """
        self.logs.append(LogEntry(message=message, value=value))

    def span(self, name: str) -> "InMemorySpan":
        """Generate a sub-logger from the current logging instance.

        Args:
            name: A descriptive name of what this child logger will contain logs about.

        Returns:
            A nested `InMemoryDebugLogger` that is stored in a nested position as part of the parent
            logger.
        """
        child = InMemorySpan(name=name)
        self.logs.append(child)
        return child

    def task_span(
        self, task_name: str, input: PydanticSerializable
    ) -> "InMemoryTaskSpan":
        """Generate a task-specific span from the current logging instance.


        Args:
            task_name: The name of the task that is being logged
            input: The input for the task that is being logged.

        Returns:
            A nested `InMemoryTaskSpan` that is stored in a nested position as part of the parent
                logger
        """

        child = InMemoryTaskSpan(name=task_name, input=input)
        self.logs.append(child)
        return child

    def _rich_render_(self) -> Tree:
        """Renders the debug log via classes in the `rich` package"""
        tree = Tree(label=self.name)

        for log in self.logs:
            tree.add(log._rich_render_())

        return tree

    def _ipython_display_(self) -> None:
        """Default rendering for Jupyter notebooks"""
        from rich import print

        print(self._rich_render_())


class InMemorySpan(AbstractContextManager["InMemorySpan"], InMemoryDebugLogger):
    start_timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    end_timestamp: Optional[datetime] = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.end_timestamp = datetime.utcnow()

    def _rich_render_(self) -> Tree:
        """Renders the debug log via classes in the `rich` package"""
        tree = Tree(label=self.name)

        for log in self.logs:
            tree.add(log._rich_render_())

        return tree


class InMemoryTaskSpan(InMemorySpan):
    input: PydanticSerializable
    output: Optional[PydanticSerializable] = None

    def record_output(self, output: PydanticSerializable) -> None:
        """Record a `Task`'s output. Since a Context Manager can't provide this in the `__exit__`
        method, output should be captured once it is generated.

        This should be handled automatically within the execution of the task.

        Args:
            output: The output of the task that is being logged.
        """
        self.output = output

    def _rich_render_(self) -> Tree:
        """Renders the debug log via classes in the `rich` package"""
        tree = Tree(label=self.name)

        tree.add(_render_log_value(self.input, "Input"))

        for log in self.logs:
            tree.add(log._rich_render_())

        tree.add(_render_log_value(self.output, "Output"))

        return tree


Input = TypeVar("Input", bound=PydanticSerializable)
"""Interface to be passed to the task with all data needed to run the process.
Ideally, these are specified in terms related to the use-case, rather than lower-level
configuration options."""
Output = TypeVar("Output", bound=PydanticSerializable)
"""Interface of the output returned by the task."""


MAX_CONCURRENCY = 20
global_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)


class Task(ABC, Generic[Input, Output]):
    """Base task interface. This may consist of several sub-tasks to accomplish the given task.

    Generics:
        Input: Interface to be passed to the task with all data needed to run the process.
            Ideally, these are specified in terms related to the use-case, rather than lower-level
            configuration options.
        Output: Interface of the output returned by the task.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Decorates run method to auto log input and output for the task"""
        super().__init_subclass__(**kwargs)

        def log_run_input_output(
            func: Callable[["Task[Input, Output]", Input, DebugLogger], Output]
        ) -> Callable[["Task[Input, Output]", Input, DebugLogger], Output]:
            @functools.wraps(func)
            def inner(
                self: "Task[Input, Output]",
                input: Input,
                logger: DebugLogger,
            ) -> Output:
                with logger.task_span(type(self).__name__, input) as task_span:
                    output = func(self, input, task_span)
                    task_span.record_output(output)
                    return output

            return inner

        cls.run = log_run_input_output(cls.run)  # type: ignore

    @abstractmethod
    def run(self, input: Input, logger: DebugLogger) -> Output:
        """Executes the process for this use-case."""
        ...

    def run_concurrently(
        self,
        inputs: Iterable[Input],
        debug_logger: DebugLogger,
        concurrency_limit: int = MAX_CONCURRENCY,
    ) -> Sequence[Output]:
        """Executes multiple processes of this task concurrently.

        Each provided input is potentially executed concurrently to the others. There is a global limit
        on the number of concurrently executed tasks that is shared by all tasks of all types.

        Args:
            inputs: The inputs that are potentially processed concurrently.
            debug_logger: The logger passed on the `run` method when executing a task.
            concurrency_limit: An optional additional limit for the number of concurrently executed task for
                this method call. This can be used to prevent queue-full or similar error of downstream APIs
                when the global concurrency limit is too high for a certain task.
        Returns:
            The `Output`s generated by calling `run` for each given `Input`. The order of `Output`s
                corresponds to the order of the `Input`s.
        """

        with debug_logger.span(f"Concurrent {type(self).__name__} tasks") as span:

            def run_batch(inputs: Iterable[Input]) -> Iterable[Output]:
                return global_executor.map(
                    lambda input: self.run(input, span),
                    inputs,
                )

            return [
                output
                for batch in batched(inputs, concurrency_limit)
                for output in run_batch(batch)
            ]


T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterable[Iterable[T]]:
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


ExpectedOutput = TypeVar("ExpectedOutput", bound=PydanticSerializable)
Evaluation = TypeVar("Evaluation", bound=PydanticSerializable)
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
    ident: Optional[str] = Field(default_factory=lambda: str(uuid4()))


class Dataset(BaseModel, Generic[Input, ExpectedOutput]):
    """A dataset of examples used for evaluation of a task.

    Attributes:
        name: This a human readable identifier for a dataset.
        examples: The actual examples that a task will be evaluated on.
    """

    name: str
    examples: Sequence[Example[Input, ExpectedOutput]]


class Evaluator(ABC, Generic[Input, ExpectedOutput, Evaluation, AggregatedEvaluation]):
    """Base evaluator interface. This should run certain evaluation steps for some job.

    Generics:
        Input: Interface to be passed to the task that shall be evaluated.
        ExpectedOutput: Output that is expected from the task run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated task.
        AggregatedEvaluation: The aggregated results of an evaluation run with a dataset.

    We suggest supplying a `Task` in the `__init__` method and running it in the `evaluate` method.
    """

    @abstractmethod
    def evaluate(
        self,
        input: Input,
        logger: DebugLogger,
        expected_output: ExpectedOutput,
    ) -> Evaluation:
        """Executes the evaluation for this use-case."""
        pass

    def evaluate_dataset(
        self, dataset: Dataset[Input, ExpectedOutput], logger: DebugLogger
    ) -> AggregatedEvaluation:
        """Evaluates an entire datasets in a threaded manner and aggregates the results into an `AggregatedEvaluation`."""
        with ThreadPoolExecutor(max_workers=10) as executor:
            evaluations = list(
                tqdm(
                    executor.map(
                        lambda idx_example: self.evaluate(
                            idx_example.input,
                            logger,
                            idx_example.expected_output,
                        ),
                        dataset.examples,
                    ),
                    total=len(dataset.examples),
                    desc="Evaluating",
                )
            )
        return self.aggregate(evaluations)

    @abstractmethod
    def aggregate(self, evaluations: Sequence[Evaluation]) -> AggregatedEvaluation:
        """`Evaluator`-specific method for aggregating individual `Evaluations` into report-like `Aggregated Evaluation`."""
        pass

from contextlib import AbstractContextManager
from datetime import datetime
from types import TracebackType

from pydantic import BaseModel, Field, RootModel, SerializeAsAny
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from typing_extensions import TypeAliasType, Self

from typing import (
    TYPE_CHECKING,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

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
    logs: list[Union[LogEntry, "InMemorySpan", "InMemoryTaskSpan"]] = []

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
    input: SerializeAsAny[PydanticSerializable]
    output: Optional[SerializeAsAny[PydanticSerializable]] = None

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

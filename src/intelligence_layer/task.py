from abc import abstractmethod
from datetime import datetime
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Mapping,
    NewType,
    Optional,
    Sequence,
    TypeVar,
    Protocol,
    Union,
    runtime_checkable,
    Callable,
    ParamSpec,
)
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


class JsonSerializer(RootModel[PydanticSerializable]):
    root: SerializeAsAny[PydanticSerializable]


class LogEntry(BaseModel):
    message: str
    value: SerializeAsAny[PydanticSerializable]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def _render_(self) -> Panel:
        value = (
            self.value
            if isinstance(self.value, BaseModel)
            else JsonSerializer(root=self.value)
        )
        return Panel(
            Syntax(
                value.model_dump_json(indent=2),
                "json",
                word_wrap=True,
            ),
            title=self.message,
        )

    def _ipython_display_(self) -> None:
        from rich import print

        print(self._render_())


@runtime_checkable
class DebugLogger(Protocol):
    name: str

    def log(self, message: str, value: PydanticSerializable) -> None:
        ...

    def child_logger(self, name: str) -> "DebugLogger":
        ...


class NoOpDebugLogger:
    name = "NoOp"

    def log(self, message: str, value: PydanticSerializable) -> None:
        pass

    def child_logger(self, name: str) -> "NoOpDebugLogger":
        return NoOpDebugLogger()


class JsonDebugLogger(BaseModel):
    name: str
    logs: list[Union[LogEntry, "JsonDebugLogger"]] = []

    def log(self, message: str, value: PydanticSerializable) -> None:
        self.logs.append(LogEntry(message=message, value=value))

    def child_logger(self, name: str) -> "JsonDebugLogger":
        child = JsonDebugLogger(name=name)
        self.logs.append(child)
        return child

    def _render_(self) -> Tree:
        tree = Tree(label=self.name)

        for log in self.logs:
            tree.add(log._render_())

        return tree

    def _ipython_display_(self) -> None:
        from rich import print

        print(self._render_())


Input = TypeVar("Input", bound=PydanticSerializable)
"""Interface to be passed to the task with all data needed to run the process.
Ideally, these are specified in terms related to the use-case, rather than lower-level
configuration options."""
Output = TypeVar("Output", bound=PydanticSerializable)
"""Interface of the output returned by the task."""
P = ParamSpec("P")


def log_run_input_output(
    func: Callable[P, Output],
) -> Callable[P, Output]:
    @functools.wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> Output:
        input = args[1]
        logger = args[2]
        assert isinstance(input, BaseModel)
        assert isinstance(logger, DebugLogger)

        logger.log("Input", input)
        output = func(*args, **kwargs)
        logger.log("Output", output)
        return output

    return inner


class Task(Generic[Input, Output]):
    """Base task interface. This may consist of several sub-tasks to accomplish the given task.

    Generics:
        Input: Interface to be passed to the task with all data needed to run the process.
            Ideally, these are specified in terms related to the use-case, rather than lower-level
            configuration options.
        Output: Interface of the output returned by the task.
    """

    @abstractmethod
    @log_run_input_output
    def run(self, input: Input, logger: DebugLogger) -> Output:
        """Executes the process for this use-case."""
        raise NotImplementedError


ExpectedOutput = TypeVar("ExpectedOutput")
Evaluation = NewType("Evaluation", Mapping[str, int | float | bool | str])


class Evaluator(Generic[Input, ExpectedOutput]):
    """Base evaluator interface. This should run certain evaluation steps for some job.

    Generics:
        Input: Interface to be passed to the task that shall be evaluated.
        ExpectedOutput: Output that is expected from the task run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated task.

    We suggest supplying a `Task` and a number of `Grader`s in the `__init__` method.
    """

    @abstractmethod
    def evaluate(
        self,
        input: Input,
        logger: DebugLogger,
        expected_output: ExpectedOutput,
    ) -> Evaluation:
        """Executes the evaluation for this use-case."""
        raise NotImplementedError

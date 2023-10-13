from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
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
)
import uuid
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
                self: "Task[Input, Output]", input: Input, logger: DebugLogger
            ) -> Output:
                logger.log("Input", input)
                output = func(self, input, logger)
                logger.log("Output", output)
                return output

            return inner

        cls.run = log_run_input_output(cls.run)  # type: ignore

    @abstractmethod
    def run(self, input: Input, logger: DebugLogger) -> Output:
        """Executes the process for this use-case."""
        ...


ExpectedOutput = TypeVar("ExpectedOutput", bound=PydanticSerializable)
Evaluation = TypeVar("Evaluation", bound=PydanticSerializable)
AggregatedEvaluation = TypeVar("AggregatedEvaluation", bound=PydanticSerializable)

class Evaluator(
    ABC, Generic[Input, ExpectedOutput, Evaluation, AggregatedEvaluation]
):
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
        pass


    def evaluate_dataset(
        self, dataset: Sequence[tuple[Input, ExpectedOutput]], logger: DebugLogger
    ) -> AggregatedEvaluation:
        with ThreadPoolExecutor(max_workers=10) as executor:
            evaluations = list(
                tqdm(
                    executor.map(
                        lambda req: self.evaluate(
                            req[0], logger.child_logger(str(uuid.uuid4())), req[1]
                        ),
                        dataset,
                    ),
                    total=len(dataset),
                    desc="Evaluating",
                )
            )
        return self.aggregate(evaluations)

    @abstractmethod
    def aggregate(self, evaluations: Sequence[Evaluation]) -> AggregatedEvaluation:
        pass

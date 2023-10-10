from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Protocol,
    Union,
    runtime_checkable,
)
from aleph_alpha_client import (
    CompletionRequest,
    CompletionResponse,
    ExplanationRequest,
    ExplanationResponse,
    Prompt,
)
from pydantic import (
    BaseModel,
    RootModel,
    Field,
    SerializeAsAny,
)
from rich.panel import Panel
from rich.tree import Tree
from typing_extensions import TypeAliasType
from uuid import uuid4


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


class LogEntry(BaseModel):
    message: str
    value: SerializeAsAny[PydanticSerializable]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def _render_(self) -> Panel:
        return Panel(str(self.value), title=self.message)

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


Input = TypeVar("Input")
"""Interface to be passed to the task with all data needed to run the process.
Ideally, these are specified in terms related to the use-case, rather than lower-level
configuration options."""
Output = TypeVar("Output")
"""Interface of the output returned by the task."""


class Task(Generic[Input, Output]):
    """Base task interface. This may consist of several sub-tasks to accomplish the given task.

    Generics:
        Input: Interface to be passed to the task with all data needed to run the process.
            Ideally, these are specified in terms related to the use-case, rather than lower-level
            configuration options.
        Output: Interface of the output returned by the task.
    """

    @abstractmethod
    def run(self, input: Input, logger: DebugLogger) -> Output:
        """Executes the process for this use-case."""
        raise NotImplementedError

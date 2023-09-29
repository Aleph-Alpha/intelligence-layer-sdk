from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Protocol,
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
    SerializeAsAny,
)
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
        | Prompt
        | CompletionRequest
        | CompletionResponse
        | ExplanationRequest
        | ExplanationResponse
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
        | BaseModel
        | Prompt
        | CompletionRequest
        | CompletionResponse
        | ExplanationRequest
        | ExplanationResponse,
    )

LogLevel = Literal["info", "debug"]


class LogEntry(BaseModel):
    message: str
    level: LogLevel
    value: SerializeAsAny[PydanticSerializable]


class DebugLog(ABC, RootModel[list[LogEntry]]):
    @abstractmethod
    def info(self, message: str, value: PydanticSerializable) -> None:
        pass

    def debug(self, message: str, value: PydanticSerializable) -> None:
        pass

    @staticmethod
    def enabled(level: LogLevel) -> "DebugLog":
        return DebugEnabledLog() if level == "debug" else InfoEnabledLog()


class InfoEnabledLog(DebugLog):
    root: list[LogEntry] = []

    def info(self, message: str, value: PydanticSerializable) -> None:
        self.root.append(LogEntry(message=message, level="info", value=value))

    def _ipython_display_(self) -> None:
        from IPython.display import display_javascript, display_html  # type: ignore

        uuid = uuid4()
        display_html(
            f'<script src="https://rawgit.com/caldwell/renderjson/master/renderjson.js"></script><div id="{uuid}" style="height: 600px; width:100%;"></div>',
            raw=True,
        )
        display_javascript(
            f"""
        renderjson.set_show_to_level(2);
        document.getElementById('{uuid}').appendChild(renderjson({self.model_dump_json()}));
        """,
            raw=True,
        )


class DebugEnabledLog(InfoEnabledLog):
    def debug(self, message: str, value: PydanticSerializable) -> None:
        self.root.append(LogEntry(message=message, level="debug", value=value))


@runtime_checkable
class OutputProtocol(Protocol):
    """Minimum interface for a `Task`'s output."""

    debug_log: DebugLog
    """Provides key steps, decisions, and intermediate outputs of a task's process."""


Input = TypeVar("Input")
"""Interface to be passed to the task with all data needed to run the process.
Ideally, these are specified in terms related to the use-case, rather than lower-level
configuration options."""
Output = TypeVar("Output", bound=OutputProtocol)
"""Interface of the output returned by the task.
It is required to adhere to the `OutputProtocol` and provide a `DebugLog` of key steps
and decisions made in the process of generating the output."""


class Task(Generic[Input, Output]):
    """Base task interface. This may consist of several sub-tasks to accomplish the given task.

    Generics:
        Input: Interface to be passed to the task with all data needed to run the process.
            Ideally, these are specified in terms related to the use-case, rather than lower-level
            configuration options.
        Output: Interface of the output returned by the task.
            It is required to adhere to the `OutputProtocol` and provide a `DebugLog` of key steps
            and decisions made in the process of generating the output.
    """

    @abstractmethod
    def run(self, input: Input) -> Output:
        """Executes the process for this use-case."""
        raise NotImplementedError

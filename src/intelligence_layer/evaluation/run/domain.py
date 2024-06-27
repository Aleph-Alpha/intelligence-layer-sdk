import traceback
from datetime import datetime
from typing import Generic

from pydantic import BaseModel
from rich.tree import Tree

from intelligence_layer.connectors.base.json_serializable import SerializableDict
from intelligence_layer.core.task import Output


class FailedExampleRun(BaseModel):
    """Captures an exception raised when running a single example with a :class:`Task`.

    Attributes:
        error_message: String-representation of the exception.
    """

    error_message: str

    @staticmethod
    def from_exception(exception: Exception) -> "FailedExampleRun":
        return FailedExampleRun(
            error_message=f"{type(exception).__qualname__}: {exception}\n{traceback.format_exc()}"
        )


class ExampleOutput(BaseModel, Generic[Output]):
    """Output of a single evaluated :class:`Example`.

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

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"Example ID={self.example_id}\n"
            f"Related Run ID={self.run_id}\n"
            f'Output="{self.output}"\n'
        )

    def _rich_render(self, skip_example_id: bool = False) -> Tree:
        tree = Tree(f"Output: {self.run_id}")
        if not skip_example_id:
            tree.add(f"Example ID: {self.example_id}")
        tree.add(str(self.output))
        return tree


class SuccessfulExampleOutput(BaseModel, Generic[Output]):
    """Successful output of a single evaluated :class:`Example`.

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
            f"Run ID = {self.run_id}\n"
            f"Example ID = {self.example_id}\n"
            f'Output = "{self.output}"\n'
        )


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
        labels: Labels for filtering runs. Defaults to empty list.
        metadata: Additional information about the run. Defaults to empty dict.
    """

    dataset_id: str
    id: str
    start: datetime
    end: datetime
    failed_example_count: int
    successful_example_count: int
    description: str
    labels: set[str] = set()
    metadata: SerializableDict = dict()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"Run Overview ID = {self.id}\n"
            f"Dataset ID = {self.dataset_id}\n"
            f"Start time = {self.start}\n"
            f"End time = {self.end}\n"
            f"Failed example count = {self.failed_example_count}\n"
            f"Successful example count = {self.successful_example_count}\n"
            f'Description = "{self.description}"\n'
            f'Labels = "{self.labels}"\n'
            f'Metadata = "{self.metadata}"\n'
        )

    def __hash__(self) -> int:
        return hash(self.id)

from typing import Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field
from rich.tree import Tree

from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.core.task import Input
from intelligence_layer.core.tracer.tracer import PydanticSerializable

ExpectedOutput = TypeVar("ExpectedOutput", bound=PydanticSerializable)
"""Dataset-specific type that defines characteristics that an :class:`Output` can be checked against.

Traditional names for this are `label` or `y` in classification."""


class Example(BaseModel, Generic[Input, ExpectedOutput]):
    """Example case used for evaluations.

    Attributes:
        input: Input for the :class:`Task`. Has to be same type as the input for the task used.
        expected_output: The expected output from a given example run.
            This will be used by the evaluator to compare the received output with.
        id: Identifier for the example, defaults to uuid.
        metadata: Optional dictionary of custom key-value pairs.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
    """

    input: Input
    expected_output: ExpectedOutput
    id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: Optional[SerializableDict] = None

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"Example ID = {self.id}\n"
            f"Input = {self.input}\n"
            f"Expected output = {self.expected_output}\n"
            f"Metadata = {self.metadata}\n"
        )

    def _rich_render(self) -> Tree:
        example_tree = Tree(f"Example: {self.id}")
        example_tree.add("Input").add(str(self.input))
        example_tree.add("Expected Output").add(str(self.expected_output))
        if self.metadata:
            example_tree.add("Metadata").add(str(self.metadata))
        return example_tree


class Dataset(BaseModel):
    """Represents a dataset linked to multiple examples.

    Attributes:
        id: Dataset ID.
        name: A short name of the dataset.
        label: Labels for filtering datasets. Defaults to empty list.
        metadata: Additional information about the dataset. Defaults to empty dict.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    labels: set[str] = set()
    metadata: SerializableDict = dict()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"Dataset ID = {self.id}\n"
            f"Name = {self.name}\n"
            f"Labels = {self.labels}\n"
            f"Metadata = {self.metadata}"
        )

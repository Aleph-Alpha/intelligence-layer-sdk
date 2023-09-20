from abc import abstractmethod
from typing import Generic, Sequence, Any, TypeVar, Mapping
from pydantic import BaseModel, Field

Input = TypeVar("Input")
Output = TypeVar("Output")


class BaseTask(Generic[Input, Output]):
    @abstractmethod
    def definition(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def examples(self) -> Sequence[Input]:
        raise NotImplementedError

    @abstractmethod
    def run(self, input: Input) -> Output:
        raise NotImplementedError

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "definition": self.definition(),
            "examples": self.examples(),
        }

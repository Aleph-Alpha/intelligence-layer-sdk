from abc import abstractmethod
from typing import Generic, List, Dict, Any, TypeVar
from pydantic import BaseModel, Field

Input = TypeVar('Input')
Output = TypeVar('Output')

class BaseTask(BaseModel, Generic[Input, Output]):
    @abstractmethod
    def definition(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def examples(self) -> List[Input]:
        raise NotImplementedError

    @abstractmethod
    def run(self, input: Input) -> Output:
        raise NotImplementedError

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self)
        d["definition"] = self.definition()
        d["examples"] = self.examples()
        return d

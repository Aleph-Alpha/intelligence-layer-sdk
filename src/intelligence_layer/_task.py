from abc import abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class BaseTask(BaseModel):
    @abstractmethod
    def definition(self):
        raise NotImplementedError

    @abstractmethod
    def examples(self):
        raise NotImplementedError

    def evaluation_examples(self) -> [Any, Any]:  # TODO?
        raise NotImplementedError

    @abstractmethod
    def run():
        raise NotImplementedError

    def as_dict(self):
        d = dict(self)
        d["definition"] = self.definition()
        d["examples"] = self.examples()
        return d

from abc import abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Task:
    @abstractmethod
    def definition():
        raise NotImplementedError

    @abstractmethod
    def examples():
        raise NotImplementedError

    @abstractmethod
    def run():
        raise NotImplementedError

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional

from sqlalchemy import ColumnElement

from intelligence_layer.learning.models import (
    InstructionFinetuningSample,
)


class InstructionFinetuningDataRepository(ABC):
    @abstractmethod
    def store_sample(self, sample: InstructionFinetuningSample) -> str:
        pass

    @abstractmethod
    def store_samples(
        self, samples: Iterable[InstructionFinetuningSample]
    ) -> list[str]:
        pass

    @abstractmethod
    def head(self, limit: Optional[int] = 100) -> Iterable[InstructionFinetuningSample]:
        pass

    @abstractmethod
    def sample(self, id: str) -> Optional[InstructionFinetuningSample]:
        pass

    @abstractmethod
    def samples(self, ids: Iterable[str]) -> Iterable[InstructionFinetuningSample]:
        pass

    @abstractmethod
    def samples_with_filter(
        self, filter_expression: ColumnElement[bool], limit: Optional[int] = 100
    ) -> Iterable[InstructionFinetuningSample]:
        pass

    @abstractmethod
    def delete_sample(self, id: str) -> None:
        pass

    @abstractmethod
    def delete_samples(self, ids: Iterable[str]) -> None:
        pass

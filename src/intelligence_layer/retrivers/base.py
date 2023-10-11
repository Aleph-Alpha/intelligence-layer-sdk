from abc import ABC, abstractmethod
from typing import Sequence
from pydantic import BaseModel


class SearchResult(BaseModel):
    score: float
    chunk: str


class BaseRetriver(ABC):
    @abstractmethod
    def get_relevant_documents_with_scores(
        self, query: str, *, k: int
    ) -> Sequence[SearchResult]:
        pass

    @abstractmethod
    def add_documents(self, texts: Sequence[str]) -> None:
        pass

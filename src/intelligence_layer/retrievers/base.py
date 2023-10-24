from abc import ABC, abstractmethod
from typing import Sequence
from pydantic import BaseModel


class SearchResult(BaseModel):
    score: float
    text: str


class BaseRetriever(ABC):
    @abstractmethod
    def get_relevant_documents_with_scores(self, query: str) -> Sequence[SearchResult]:
        pass

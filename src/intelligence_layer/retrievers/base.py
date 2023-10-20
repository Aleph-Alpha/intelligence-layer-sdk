from abc import ABC, abstractmethod
import functools
from typing import Callable, Sequence, Any
from pydantic import BaseModel
from intelligence_layer.task import DebugLogger


class SearchResult(BaseModel):
    score: float
    chunk: str


class BaseRetriever(ABC):
    @abstractmethod
    def get_relevant_documents_with_scores(
        self, query: str, logger: DebugLogger, *, k: int
    ) -> Sequence[SearchResult]:
        pass

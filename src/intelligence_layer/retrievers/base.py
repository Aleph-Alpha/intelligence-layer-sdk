from abc import ABC, abstractmethod
from typing import Sequence
from pydantic import BaseModel


class SearchResult(BaseModel):
    """returns a text alongside its similarity score with a query.

    Attributes:
        score: The similarity score between the document and the query.
            Will be between 0 and 1, where 0 means no similarity and 1 perfect similarity.
        text: The text found by search.
    """

    score: float
    text: str


class BaseRetriever(ABC):
    @abstractmethod
    def get_relevant_documents_with_scores(self, query: str) -> Sequence[SearchResult]:
        pass

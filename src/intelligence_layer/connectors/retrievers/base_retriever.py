from abc import ABC, abstractmethod
from typing import Any, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors.document_index.document_index import DocumentPath


class Document(BaseModel):
    """Document abstraction, specifically for retrieval use cases.
    Attributes:
        text: The document's text.
        metadata: Any json-serializable object.
    """

    text: str
    metadata: Any = None


class SearchResult(BaseModel):
    """Contains a text alongside its search score.

    Attributes:
        score: The similarity score between the text and the query that was searched with.
            Will be between 0 and 1, where 0 means no similarity and 1 perfect similarity.
        document: The document found by search.
    """

    id: str 
    score: float
    document: Document


class BaseRetriever(ABC):
    """General interface for any retriever.

    Retrievers are used to find texts given a user query.
    Each Retriever implementation owns its own logic for retrieval.
    For comparison purposes, we assume scores in the `SearchResult` instances to be between 0 and 1.
    """

    @abstractmethod
    def get_relevant_documents_with_scores(self, query: str) -> Sequence[SearchResult]:
        pass

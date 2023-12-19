from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

from pydantic import BaseModel

from intelligence_layer.core.chunk import Chunk


class Document(BaseModel):
    """A document.

    Attributes:
        text: The document's text.
        metadata: Any metadata added to the document.
    """

    text: str
    metadata: Any = None


class DocumentChunk(BaseModel):
    """Part of a :class:`Document`, specifically for retrieval use cases.

    Attributes:
        chunk: Chunk of the document that matched the search query.
        metadata: Any metadata added to the document.
    """

    text: Chunk
    metadata: Any = None


ID = TypeVar("ID")


class SearchResult(BaseModel, Generic[ID]):
    """Contains a text alongside its search score.

    Attributes:
        score: The similarity score between the text and the query that was searched with.
            Will be between 0 and 1, where 0 means no similarity and 1 perfect similarity.
        document: The document found by search.
    """

    id: ID
    score: float
    document_chunk: DocumentChunk


class BaseRetriever(ABC, Generic[ID]):
    """General interface for any retriever.

    Retrievers are used to find texts given a user query.
    Each Retriever implementation owns its own logic for retrieval.
    For comparison purposes, we assume scores in the `SearchResult` instances to be between 0 and 1.
    """

    @abstractmethod
    def get_relevant_documents_with_scores(
        self, query: str
    ) -> Sequence[SearchResult[ID]]:
        pass

    @abstractmethod
    def get_full_document(self, id: ID) -> Document:
        pass

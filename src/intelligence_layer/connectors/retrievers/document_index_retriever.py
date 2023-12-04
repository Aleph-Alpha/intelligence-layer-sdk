from typing import Sequence

from intelligence_layer.connectors.document_index.document_index import (
    CollectionPath,
    DocumentIndexClient,
    SearchQuery,
)
from intelligence_layer.connectors.retrievers.base_retriever import (
    BaseRetriever,
    Document,
    SearchResult,
)


class DocumentIndexRetriever(BaseRetriever):
    """Search through documents within collections in the `DocumentIndexClient`.

    We initialize this Retriever with a collection & namespace names, and we can find the documents in the collection
    most semanticly similar to our query.

    Args:
        document_index: Client offering functionality for search.
        namespace: The namespace within the `DocumentIndexClient` where all collections are stored.
        collection: The collection within the namespace that holds the desired documents.
        k: The (top) number of documents to be returned by search.
        threshold: The mimumum value of cosine similarity between the query vector and the document vector.

    Example:
    >>> import os
    >>> from intelligence_layer.connectors import DocumentIndexClient, DocumentIndexRetriever
    >>> document_index = DocumentIndexClient(os.getenv("AA_TOKEN"))
    >>> retriever = DocumentIndexRetriever(document_index, "aleph-alpha", "wikipedia-de", 3)
    >>> documents = retriever.get_relevant_documents_with_scores("Who invented the airplane?")
    """

    def __init__(
        self,
        document_index: DocumentIndexClient,
        namespace: str,
        collection: str,
        k: int,
        threshold: float = 0.5,
    ) -> None:
        self._document_index = document_index
        self._collection_path = CollectionPath(
            namespace=namespace, collection=collection
        )
        self._k = k
        self._threshold = threshold

    def get_relevant_documents_with_scores(self, query: str) -> Sequence[SearchResult]:
        search_query = SearchQuery(
            query=query, max_results=self._k, min_score=self._threshold
        )
        response = self._document_index.asymmetric_search(
            self._collection_path, search_query
        )
        relevant_chunks = [
            SearchResult(
                score=result.score,
                document=Document(text=result.section),
            )
            for result in response
        ]
        return relevant_chunks

from collections.abc import Sequence
from typing import Optional

from intelligence_layer.connectors.document_index.document_index import (
    CollectionPath,
    DocumentIndexClient,
    DocumentPath,
    DocumentTextPosition,
    Filters,
    SearchQuery,
)
from intelligence_layer.connectors.retrievers.base_retriever import (
    BaseRetriever,
    Document,
    DocumentChunk,
    SearchResult,
)


class DocumentIndexRetriever(BaseRetriever[DocumentPath]):
    """Search through documents within collections in the `DocumentIndexClient`.

    This retriever lets you search for relevant documents in the given Document Index collection.

    Args:
        document_index: The Document Index client.
        index_name: The name of the Document Index index to use.
        namespace: The Document Index namespace.
        collection: The Document Index collection to use. This is the search context for the retriever.
        k: The number of most-relevant documents to return when searching.
        threshold: The minimum score for search results. For semantic indexes, this is the cosine
            similarity between the query and the document chunk. For hybrid indexes, this corresponds
            to fusion rank.

    Example:
    >>> import os
    >>> from intelligence_layer.connectors import DocumentIndexClient, DocumentIndexRetriever
    >>> document_index = DocumentIndexClient(os.getenv("AA_TOKEN"))
    >>> retriever = DocumentIndexRetriever(document_index, "asymmetric", "aleph-alpha", "wikipedia-de", 3)
    >>> documents = retriever.get_relevant_documents_with_scores("Who invented the airplane?")
    """

    def __init__(
        self,
        document_index: DocumentIndexClient,
        index_name: str,
        namespace: str,
        collection: str,
        k: int = 1,
        threshold: float = 0.0,
    ) -> None:
        self._document_index = document_index
        self._index_name = index_name
        self._collection_path = CollectionPath(
            namespace=namespace, collection=collection
        )
        self._k = k
        self._threshold = threshold

    def _get_absolute_position(
        self, id: DocumentPath, document_text_position: DocumentTextPosition
    ) -> dict[str, int]:
        doc = self._document_index.document(id)

        previous_item_length = sum(
            len(text) for text in doc.contents[0 : document_text_position.item]
        )

        start = previous_item_length + document_text_position.start_position
        end = previous_item_length + document_text_position.end_position

        return {"start": start, "end": end}

    def get_relevant_documents_with_scores(
        self, query: str, filters: Optional[list[Filters]] = None
    ) -> Sequence[SearchResult[DocumentPath]]:
        search_query = SearchQuery(
            query=query, max_results=self._k, min_score=self._threshold, filters=filters
        )
        response = self._document_index.search(
            self._collection_path, self._index_name, search_query
        )
        relevant_chunks = [
            SearchResult(
                id=result.document_path,
                score=result.score,
                document_chunk=DocumentChunk(
                    text=result.section,
                    **self._get_absolute_position(
                        id=result.document_path,
                        document_text_position=result.chunk_position,
                    ),
                ),
            )
            for result in response
        ]
        return relevant_chunks

    def get_full_document(self, id: DocumentPath) -> Document:
        contents = self._document_index.document(id)
        return Document(text="\n".join(contents.contents), metadata=contents.metadata)

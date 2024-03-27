from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional, Sequence

from aleph_alpha_client import Prompt, SemanticEmbeddingRequest, SemanticRepresentation
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.http.models import Distance, PointStruct, VectorParams, models

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
    LimitedConcurrencyClient,
)
from intelligence_layer.connectors.retrievers.base_retriever import (
    BaseRetriever,
    Document,
    DocumentChunk,
    SearchResult,
)


class RetrieverType(Enum):
    """Specify the type of retriever to instantiate.

    Attributes:
        ASYMMETRIC: Query is embedded as `Query` and each document as `Document`.
        SYMMETRIC: Both query and documents will be embedded as `Symmetric`.
    """

    ASYMMETRIC = (SemanticRepresentation.Query, SemanticRepresentation.Document)
    SYMMETRIC = (SemanticRepresentation.Symmetric, SemanticRepresentation.Symmetric)


class QdrantInMemoryRetriever(BaseRetriever[int]):
    """Search through documents stored in memory using semantic search.

    This retriever uses a [Qdrant](https://github.com/qdrant/qdrant)-in-Memory vector store instance to store documents and their asymmetric embeddings.
    When run, the given query is embedded and scored against the document embeddings to retrieve the k-most similar matches by cosine similarity.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        texts: The sequence of texts to be made searchable.
        k: The (top) number of documents to be returned by search.
        threshold: The mimumum value of cosine similarity between the query vector and the document vector.
        retriever_type: The type of retriever to be instantiated.
            Should be `ASYMMETRIC` for most query-document retrieveal use cases, `SYMMETRIC` is optimized
            for similar document retrieval.
        distance_metric: The distance metric to be used for vector comparison.

    Example:
        >>> from intelligence_layer.connectors import LimitedConcurrencyClient, Document, QdrantInMemoryRetriever
        >>> client = LimitedConcurrencyClient.from_env()
        >>> documents = [Document(text=t) for t in ["I do not like rain.", "Summer is warm.", "We are so back."]]
        >>> retriever = QdrantInMemoryRetriever(documents, 5, client=client)
        >>> query = "Do you like summer?"
        >>> documents = retriever.get_relevant_documents_with_scores(query)
    """

    MAX_WORKERS = 10

    def __init__(
        self,
        documents: Sequence[Document],
        k: int,
        client: AlephAlphaClientProtocol | None = None,
        threshold: float = 0.5,
        retriever_type: RetrieverType = RetrieverType.ASYMMETRIC,
        distance_metric: Distance = Distance.COSINE,
    ) -> None:
        self._client = client or LimitedConcurrencyClient.from_env()
        self._search_client = QdrantClient(":memory:")
        self._collection_name = "in_memory_collection"
        self._k = k
        self._threshold = threshold
        self._query_representation, self._document_representation = retriever_type.value
        self._distance_metric = distance_metric

        self._search_client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=128, distance=self._distance_metric),
        )
        self._add_texts_to_memory(documents)

    def get_relevant_documents_with_scores(
        self, query: str
    ) -> Sequence[SearchResult[int]]:
        query_embedding = self._embed(query, self._query_representation)
        search_result = self._search_client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            score_threshold=self._threshold,
            limit=self._k,
        )
        return [self._point_to_search_result(point) for point in search_result]

    def _embed(self, text: str, representation: SemanticRepresentation) -> list[float]:
        embedding_request = SemanticEmbeddingRequest(
            prompt=Prompt.from_text(text),
            representation=representation,
            compress_to_size=128,
            normalize=True,
        )
        return self._client.semantic_embed(
            request=embedding_request, model="luminous-base"
        ).embedding

    @staticmethod
    def _point_to_search_result(point: ScoredPoint) -> SearchResult[int]:
        assert point.payload
        assert isinstance(point.id, int)
        return SearchResult(
            id=point.id,
            score=point.score,
            document_chunk=DocumentChunk(**point.payload),
        )

    def _add_texts_to_memory(self, documents: Sequence[Document]) -> None:
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            embeddings = list(
                executor.map(
                    lambda c: self._embed(c.text, self._document_representation),
                    documents,
                )
            )
        self._search_client.upsert(
            collection_name=self._collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=idx,
                    vector=text_embedding,
                    payload=document.model_dump(),
                )
                for idx, (text_embedding, document) in enumerate(
                    zip(embeddings, documents)
                )
            ],
        )

    def get_filtered_documents_with_scores(
        self, query: str, filter: models.Filter
    ) -> Sequence[SearchResult[int]]:
        """Specific method for `InMemoryRetriever` to support filtering search results.

        Args:
            query: The text to be searched with.
            filter: Conditions to filter by.
        """
        query_embedding = self._embed(query, self._query_representation)
        search_result = self._search_client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=self._k,
            query_filter=filter,
        )
        return [self._point_to_search_result(point) for point in search_result]

    def get_full_document(self, id: int) -> Optional[Document]:
        records = self._search_client.retrieve(self._collection_name, [id], True, False)
        if not records:
            return None
        record = records.pop()
        assert record.payload
        return Document(**record.payload)

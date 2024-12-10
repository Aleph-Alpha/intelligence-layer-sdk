from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams, models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
    LimitedConcurrencyClient,
)
from intelligence_layer.connectors.retrievers.base_retriever import (
    Document,
    DocumentChunk,
    SearchResult,
)
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
    RetrieverType,
)


class HybridQdrantInMemoryRetriever(QdrantInMemoryRetriever):
    """Search through documents stored in memory using hybrid (keyword + semantic) search.

    This retriever uses a [Qdrant](https://github.com/qdrant/qdrant)-in-Memory vector store instance to store documents and their asymmetric embeddings.
    When run, the given query is embedded using both a dense and sparse embedding model and scored against the documents in the collection to find the most relevant documents.
    Finally, the retrievals are fused using the Reciprocal Rank Fusion algorithm.

    Args:
        documents: The sequence of documents to be made searchable.
        k: The (top) number of documents to be returned by search.
        client: Aleph Alpha client instance for running model related API calls. Defaults to `LimitedConcurrencyClient.from_env()`.
        threshold: The minimum value of the fusion rank score (combined cosine similarity and keyword similarity). Defaults to 0.0.
        retriever_type: The type of retriever to be instantiated. Should be `ASYMMETRIC` for most query-document retrieveal use cases, `SYMMETRIC` is optimized
            for similar document retrieval. Defaults to `ASYMMETRIC`.
        distance_metric: The distance metric to be used for vector comparison. Defaults to `Distance.COSINE`.
        sparse_model_name: The name of the sparse embedding model from `fastemebed` to be used. Defaults to `"bm25"`.
        max_workers: The maximum number of workers to use for concurrent processing. Defaults to 10.

    Example:
        >>> from intelligence_layer.connectors import LimitedConcurrencyClient, Document, HybridQdrantInMemoryRetriever
        >>> client = LimitedConcurrencyClient.from_env()
        >>> documents = [Document(text=t) for t in ["I do not like rain.", "Summer is warm.", "We are so back."]]
        >>> retriever = HybridQdrantInMemoryRetriever(documents, 5, client=client)
        >>> query = "Do you like summer?"
        >>> documents = retriever.get_relevant_documents_with_scores(query)
    """

    def __init__(
        self,
        documents: Sequence[Document],
        k: int,
        client: AlephAlphaClientProtocol | None = None,
        threshold: float = 0.0,
        retriever_type: RetrieverType = RetrieverType.ASYMMETRIC,
        distance_metric: Distance = Distance.COSINE,
        sparse_model_name: str = "bm25",
        max_workers: int = 10,
    ) -> None:
        self._client = client or LimitedConcurrencyClient.from_env()
        self._search_client = QdrantClient(":memory:")
        self._collection_name = "in_memory_collection"
        self._k = k
        self._threshold = threshold
        self._query_representation, self._document_representation = retriever_type.value
        self._distance_metric = distance_metric
        self._max_workers = max_workers

        self._search_client.set_sparse_model(sparse_model_name)
        self._sparse_vector_field_name = "text-sparse"
        self._dense_vector_field_name = "text-dense"

        if self._search_client.collection_exists(collection_name=self._collection_name):
            self._search_client.delete_collection(
                collection_name=self._collection_name,
            )

        self._search_client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=128, distance=self._distance_metric),
            sparse_vectors_config=self._search_client.get_fastembed_sparse_vector_params(),
        )

        self._add_texts_to_memory(documents)

    def _sparse_embed_query(self, query: str) -> models.SparseVector:
        if self._search_client.sparse_embedding_model_name is None:
            raise ValueError("Sparse embedding model is not set!")
        sparse_embedder = self._search_client._get_or_init_sparse_model(
            model_name=self._search_client.sparse_embedding_model_name
        )

        sparse_vector = next(sparse_embedder.query_embed(query=query))
        sparse_query_vector = models.SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist(),
        )

        return sparse_query_vector

    def get_filtered_documents_with_scores(
        self, query: str, filter: Optional[models.Filter]
    ) -> Sequence[SearchResult[int]]:
        """Retrieves documents that match the given query and filter conditions, using hybrid search.

        This method performs a hybrid search by embedding the query into dense and sparse vectors.
        It then executes search requests for both vector types and combines the results using the
        Reciprocal Rank Fusion algorithm.

        Args:
            query: The text query to search for.
            filter: An optional filter to apply to the search results.

        Returns:
            All documents that correspond to the query and pass the filter,
            sorted by their reciprocal rank fusion score.
        """
        dense_query_vector = self._embed(query, self._query_representation)
        sparse_query_vector = self._sparse_embed_query(query)

        dense_request = models.SearchRequest(
            vector=models.NamedVector(
                name=self._dense_vector_field_name,
                vector=dense_query_vector,
            ),
            limit=self._k,
            filter=filter,
            with_payload=True,
        )
        sparse_request = models.SearchRequest(
            vector=models.NamedSparseVector(
                name=self._sparse_vector_field_name,
                vector=sparse_query_vector,
            ),
            limit=self._k,
            filter=filter,
            with_payload=True,
        )

        dense_request_response, sparse_request_response = (
            self._search_client.search_batch(
                collection_name=self._collection_name,
                requests=[dense_request, sparse_request],
            )
        )
        search_result = reciprocal_rank_fusion(
            [dense_request_response, sparse_request_response], limit=self._k
        )

        return [
            self._point_to_search_result(point)
            for point in search_result
            if point.score >= self._threshold
        ]

    def get_relevant_documents_with_scores(
        self, query: str
    ) -> Sequence[SearchResult[int]]:
        """Search for relevant documents given a query using hybrid search (dense + sparse retrieval).

        This method performs a hybrid search by embedding the query into dense and sparse vectors.
        It then executes search requests for both vector types and combines the results using the
        Reciprocal Rank Fusion algorithm.

        Args:
            query: The text to be searched with.

        Returns:
            All documents that correspond to the query,
            sorted by their reciprocal rank fusion score.
        """
        return self.get_filtered_documents_with_scores(query, filter=None)

    def _add_texts_to_memory(self, documents: Sequence[Document]) -> None:
        with ThreadPoolExecutor(
            max_workers=min(len(documents), self._max_workers)
        ) as executor:
            dense_embeddings = list(
                executor.map(
                    lambda doc: self._embed(doc.text, self._document_representation),
                    documents,
                )
            )
        if self._search_client.sparse_embedding_model_name is None:
            raise ValueError("Sparse embedding model is not set!")
        sparse_embeddings = list(
            self._search_client._sparse_embed_documents(
                documents=[document.text for document in documents],
                embedding_model_name=self._search_client.sparse_embedding_model_name,
            )
        )
        self._search_client.upsert(
            collection_name=self._collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=idx,
                    vector={
                        self._dense_vector_field_name: dense_vector,
                        self._sparse_vector_field_name: sparse_vector,
                    },
                    payload=DocumentChunk(
                        text=document.text,
                        start=0,
                        end=len(document.text) - 1,
                        metadata=document.metadata,
                    ).model_dump(),
                )
                for idx, (dense_vector, sparse_vector, document) in enumerate(
                    zip(dense_embeddings, sparse_embeddings, documents, strict=True)
                )
            ],
        )

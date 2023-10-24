from typing import Sequence
from intelligence_layer.retrievers.base import BaseRetriever, SearchResult
from qdrant_client import QdrantClient
from aleph_alpha_client import (
    Client,
    Prompt,
    SemanticRepresentation,
    SemanticEmbeddingRequest,
)
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from intelligence_layer.task import Chunk, DebugLogger


class InMemoryRetriever(BaseRetriever):
    """Retrieve top k documents using in memory semantic search

    This retriever uses a [Qdrant](https://github.com/qdrant/qdrant) in memory vector store instance to store the chunks and their asymmetric (SemanticRepresentation.Document) embeddings.
    When running "get_relevant_documents_with_scores", the query is embedded asymmetrically (SemanticRepresentation.Query) and scored against the embeddings in the vector store to retrieve the k-most similar matches by cosine similarity.

    Args:
        client: An instance of the Aleph Alpha client.
        chunks: A sequence of texts you want to embed and put in memory as your documents
        threshold: A mimumum value of the cosine similarity between the query vector and the document vector

    Example:
        >>> chunks = ["I do not like rain", "Summer is warm", "We are so back"]
        >>> query = "Do you like summer?"
        >>> retriever = InMemoryRetriever(client, chunks)
        >>> documents = retriever.get_relevant_documents_with_scores(query, NoOpDebugLogger(), k=2)
    """

    def __init__(
        self,
        client: Client,
        chunks: Sequence[Chunk],
        threshold: float = 0.5,
    ) -> None:
        self._client = client
        self._search_client = QdrantClient(":memory:")
        self._collection_name = "in_memory_collection"
        self._threshold = threshold

        self._search_client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )
        self._add_chunks_to_memory(chunks)

    def get_relevant_documents_with_scores(
        self, query: str, logger: DebugLogger, *, k: int
    ) -> Sequence[SearchResult]:
        def _point_to_search_result(point: ScoredPoint) -> SearchResult:
            assert point.payload
            return SearchResult(score=point.score, text=point.payload["text"])

        query_embedding = self._embed(query, SemanticRepresentation.Query)

        logger.log("Query", query)
        logger.log("k", k)

        search_result = self._search_client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            score_threshold=self._threshold,
            limit=k,
        )
        logger.log("output", search_result)

        return [_point_to_search_result(point) for point in search_result]

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

    def _add_chunks_to_memory(self, chunks: Sequence[Chunk]) -> None:
        embeddings = [self._embed(c, SemanticRepresentation.Document) for c in chunks]

        self._search_client.upsert(
            collection_name=self._collection_name,
            wait=True,
            points=[
                PointStruct(id=idx, vector=text_embedding, payload={"text": text})
                for idx, (text_embedding, text) in enumerate(zip(embeddings, chunks))
            ],
        )

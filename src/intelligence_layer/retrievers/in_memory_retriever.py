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

from intelligence_layer.task import DebugLogger


class InMemoryRetriever(BaseRetriever):
    def __init__(
        self,
        client: Client,
        chunks: Sequence[str],
        threshold: float = 0.5,
        collection_name: str = "default_collection",
    ) -> None:
        self.client = client
        self.search_client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.threshold = threshold

        self.search_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )
        self._add_chunks_to_memory(chunks)

    def get_relevant_documents_with_scores(
        self, query: str, logger: DebugLogger, *, k: int
    ) -> Sequence[SearchResult]:
        def _point_to_search_result(point: ScoredPoint) -> SearchResult:
            assert point.payload
            return SearchResult(score=point.score, chunk=point.payload["text"])

        query_embedding = self._embed(query, SemanticRepresentation.Query)

        logger.log("Query", query)
        logger.log("k", k)

        search_result = self.search_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            score_threshold=self.threshold,
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

        return self.client.semantic_embed(
            request=embedding_request, model="luminous-base"
        ).embedding

    def _add_chunks_to_memory(self, chunks: Sequence[str]) -> None:
        embeddings = [self._embed(c, SemanticRepresentation.Document) for c in chunks]

        self.search_client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                PointStruct(id=idx, vector=text_embedding, payload={"text": text})
                for idx, (text_embedding, text) in enumerate(zip(embeddings, chunks))
            ],
        )

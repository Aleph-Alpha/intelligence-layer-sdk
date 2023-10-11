from typing import Sequence
from intelligence_layer.retrivers.base import BaseRetriver, SearchResult
from qdrant_client import QdrantClient
from aleph_alpha_client import (
    Client,
    Prompt,
    SemanticRepresentation,
    SemanticEmbeddingRequest,
)
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.http.models import Distance, VectorParams, PointStruct


class QdrantRetriver(BaseRetriver):
    def __init__(
        self,
        client: Client,
        location: str = ":memory:",  # follows the default qdrant setting for location
        threshold: float = 0.5,
        collection_name: str = "default_collection",
    ) -> None:
        self.client = client
        self.search_client = QdrantClient(location)
        self.collection_name = collection_name

        self.search_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )
        self.threshold = threshold

    def get_relevant_documents_with_scores(
        self, query: str, k: int
    ) -> Sequence[SearchResult]:
        def _point_to_search_result(point: ScoredPoint) -> SearchResult:
            assert point.payload
            return SearchResult(score=point.score, chunk=point.payload["text"])

        query_embedding = self._embed(query, SemanticRepresentation.Query)

        search_result = self.search_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            score_threshold=self.threshold,
            limit=k,
        )

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

    def add_documents(self, texts: Sequence[str]) -> None:
        embeddings = [
            self._embed(text, SemanticRepresentation.Document) for text in texts
        ]

        self.search_client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                PointStruct(id=idx, vector=text_embedding, payload={"text": text})
                for idx, (text_embedding, text) in enumerate(zip(embeddings, texts))
            ],
        )

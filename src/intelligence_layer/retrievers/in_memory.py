from typing import Sequence

from aleph_alpha_client import (
    Client,
    Prompt,
    SemanticRepresentation,
    SemanticEmbeddingRequest,
)
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from intelligence_layer.retrievers.base import BaseRetriever, SearchResult


class InMemoryRetriever(BaseRetriever):
    """Search through documents stored in memory using semantic search.

    This retriever uses a [Qdrant](https://github.com/qdrant/qdrant)-in-Memory vector store instance to store documents and their asymmetric embeddings.
    When run, the given query is embedded and scored against the document embeddings to retrieve the k-most similar matches by cosine similarity.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        texts: The sequence of texts to be made searchable.
        k: The (top) number of documents to be returned by search.
        threshold: The mimumum value of cosine similarity between the query vector and the document vector.

    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> texts = ["I do not like rain.", "Summer is warm.", "We are so back."]
        >>> retriever = InMemoryRetriever(client, texts)
        >>> query = "Do you like summer?"
        >>> documents = retriever.get_relevant_documents_with_scores(query)
    """

    def __init__(
        self,
        client: Client,
        texts: Sequence[str],
        k: int,
        threshold: float = 0.5,
    ) -> None:
        self._client = client
        self._search_client = QdrantClient(":memory:")
        self._collection_name = "in_memory_collection"
        self._k = k
        self._threshold = threshold

        self._search_client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )
        self._add_texts_to_memory(texts)

    def get_relevant_documents_with_scores(self, query: str) -> Sequence[SearchResult]:
        def _point_to_search_result(point: ScoredPoint) -> SearchResult:
            assert point.payload
            return SearchResult(score=point.score, text=point.payload["text"])

        query_embedding = self._embed(query, SemanticRepresentation.Query)
        search_result = self._search_client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            score_threshold=self._threshold,
            limit=self._k,
        )
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

    def _add_texts_to_memory(self, texts: Sequence[str]) -> None:
        embeddings = [self._embed(c, SemanticRepresentation.Document) for c in texts]
        self._search_client.upsert(
            collection_name=self._collection_name,
            wait=True,
            points=[
                PointStruct(id=idx, vector=text_embedding, payload={"text": text})
                for idx, (text_embedding, text) in enumerate(zip(embeddings, texts))
            ],
        )

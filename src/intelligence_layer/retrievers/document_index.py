from typing import Sequence
from intelligence_layer.retrievers.base import BaseRetriever, SearchResult
from aleph_alpha_client import (
    Client,
    Prompt,
    SemanticRepresentation,
    SemanticEmbeddingRequest,
)

from intelligence_layer.task import DebugLogger

import json
import requests
from typing import Union


BASE_DOCUMENT_INDEX_URL = "https://knowledge.aleph-alpha.com"
HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {TOKEN}'
}



def search(namespace: str, collection: str, query: str, max_results: int):
    url = f"{BASE_DOCUMENT_INDEX_URL}/collections/{namespace}/{collection}/search"
    data = {
        "query": [
            {
                "modality": "text",
                "text": query
            }
        ],
        "max_results": max_results
    }
    response = requests.post(url, data=json.dumps(data), headers=HEADERS)
    if response.status_code == 200:
        print(f"Successfully searched for documents with query in collection {collection} in namespace {namespace}.")
        return response.json()
    else:
        print(f"Failed to search for documents with query. Status code: {response.status_code}.")






class DocumentIndexRetriever(BaseRetriever):
    def __init__(
        self,
        client: Client,
        threshold: float = 0.5,
    ) -> None:
        self._token = client.token
        self.threshold = threshold


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

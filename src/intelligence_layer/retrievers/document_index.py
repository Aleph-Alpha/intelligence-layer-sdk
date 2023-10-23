from typing import Sequence
from intelligence_layer.retrievers.base import BaseRetriever, SearchResult
from aleph_alpha_client import (
    Client,
)

from intelligence_layer.task import DebugLogger

import json

# types-requests introduces dependency conflict with qdrant.
import requests  # type: ignore


class DocumentIndexRetriever(BaseRetriever):
    def __init__(
        self,
        client: Client,
        namespace: str,
        collection: str,
        base_document_index_url: str = "https://knowledge.aleph-alpha.com",
        threshold: float = 0.5,
    ) -> None:
        self._threshold = threshold
        self._namespace = namespace
        self._collection = collection
        self._base_document_index_url = base_document_index_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {client.token}",
        }

    def get_relevant_documents_with_scores(
        self, query: str, logger: DebugLogger, *, k: int
    ) -> Sequence[SearchResult]:
        logger.log("Query", query)
        logger.log("k", k)

        url = f"{self._base_document_index_url}/collections/{self._namespace}/{self._collection}/search"
        data = {
            "query": [{"modality": "text", "text": query}],
            "max_results": k,
            "min_score": self._threshold,
            "filter": [{"with": [{"modality": "text"}]}],
        }

        response = requests.post(url, data=json.dumps(data), headers=self.headers)
        if response.status_code != 200:
            raise Exception(
                f"Failed to search for documents with query. Status code: {response.status_code}."
            )
        return [
            SearchResult(score=result["score"], chunk=result["section"][0]["text"])
            for result in response.json()
        ]

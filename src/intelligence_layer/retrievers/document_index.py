import json
from typing import Any, Sequence

import requests  # type: ignore

from intelligence_layer.retrievers.base import BaseRetriever, SearchResult


class DocumentIndex:
    """Client for the Document Index allowing handling documents and search.

    Document Index is a tool for managing collections of documents, enabling operations such as creation, deletion, listing, and searching.
    Documents can be stored either in the cloud or in a local deployment.

    Args:
        token: A valid token for the document index API.
        base_document_index_url: The url of the document index' API.

    Example:
        >>> document_index = DocumentIndex(os.getenv("AA_TOKEN"))
        >>> documents = document_index.search(
        >>>     namespace="my_namespace",
        >>>     collection="germany_facts_collection",
        >>>     query: "What is the capital of Germany",
        >>>     max_results=4,
        >>>     min_score: 0.5
        >>> )
    """

    def __init__(
        self,
        token: str,
        base_document_index_url: str = "https://knowledge.aleph-alpha.com",
    ) -> None:
        self._base_document_index_url = base_document_index_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def create_collection(self, namespace: str, collection: str) -> None:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}"
        response = requests.put(url, headers=self.headers)
        response.raise_for_status()

    def delete_collection(self, namespace: str, collection: str) -> None:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()

    def add_document(
        self,
        namespace: str,
        collection: str,
        name: str,
        content: str,
    ) -> None:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}"
        data = {
            "schema_version": "V1",
            "contents": [{"modality": "text", "text": content}],
        }
        response = requests.put(url, data=json.dumps(data), headers=self.headers)
        response.raise_for_status()

    def delete_document(self, namespace: str, collection: str, name: str) -> None:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()

    def get_document(
        self, namespace: str, collection: str, name: str, get_chunks: bool = False
    ) -> Any:
        if not get_chunks:
            url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}"
        else:
            url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}/chunks"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_documents(self, namespace: str, collection: str) -> Any:
        url = (
            f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs"
        )
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def search(
        self,
        namespace: str,
        collection: str,
        query: str,
        max_results: int,
        min_score: float,
    ) -> Any:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/search"
        data = {
            "query": [{"modality": "text", "text": query}],
            "max_results": max_results,
            "min_score": min_score,
            "filter": [{"with": [{"modality": "text"}]}],
        }
        response = requests.post(url, data=json.dumps(data), headers=self.headers)
        response.raise_for_status()
        return response.json()


class DocumentIndexRetriever(BaseRetriever):
    """Search through documents within collections in the `DocumentIndex`.

    We initialize this Retriever with a collection & namespace names, and we can find the documents in the collection
    most semanticly similar to our query.

    Args:
        document_index: Client offering functionality for search.
        namespace: The namespace within the `DocumentIndex` where all collections are stored.
        collection: The collection within the namespace that holds the desired documents.
        k: The (top) number of documents to be returned by search.
        threshold: The mimumum value of cosine similarity between the query vector and the document vector.

    Example:
        >>> document_index = DocumentIndex(os.getenv("AA_TOKEN"))
        >>> retriever = DocumentIndexRetriever(document_index, "my_namespace", "airplane_facts_collection", 3)
        >>> query = "Who invented the airplane?"
        >>> documents = retriever.get_relevant_documents_with_scores(query)
    """

    def __init__(
        self,
        document_index: DocumentIndex,
        namespace: str,
        collection: str,
        k: int,
        threshold: float = 0.5,
    ) -> None:
        self._document_index = document_index
        self._namespace = namespace
        self._collection = collection
        self._k = k
        self._threshold = threshold

    def get_relevant_documents_with_scores(self, query: str) -> Sequence[SearchResult]:
        response = self._document_index.search(
            self._namespace, self._collection, query, self._k, self._threshold
        )
        relevant_chunks = [
            SearchResult(score=result["score"], text=result["section"][0]["text"])
            for result in response
        ]
        return relevant_chunks

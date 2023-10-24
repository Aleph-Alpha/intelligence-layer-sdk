from typing import Sequence, Union
from intelligence_layer.retrievers.base import BaseRetriever, SearchResult
import json

# types-requests introduces dependency conflict with qdrant.
import requests  # type: ignore

Metadata = Union[dict[str, str], list[dict[str, str]]]


class DocumentIndex:
    """Upload and search through the documents stored on the Aleph Alpha Document Index.

    With the document index we give users the ability to create collections and upload the documents to a cloud database.

    After that you can search through this documents based on the semantic similarity with a query.

    Args:
        client: An instance of the Aleph Alpha client.
        base_document_index_url: the url address of the document index


    Example:
        >>> query = "Do you like summer?"
        >>> document_index = DocumentIndex(client)
        >>> documents = document_index.search(namespace="my_namespace",
                                              collection="my_collection",
                                              query: "What is the capital of Germany",
                                              max_results=4,
                                              min_score: 0.5)
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
        metadata: Metadata,
    ) -> None:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}"
        data = {
            "schema_version": "V1",
            "contents": [{"modality": "text", "text": content}],
            "metadata": metadata,
        }
        response = requests.put(url, data=json.dumps(data), headers=self.headers)
        response.raise_for_status()

    def delete_document(self, namespace: str, collection: str, name: str) -> None:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()

    def get_document(
        self, namespace: str, collection: str, name: str, get_chunks: bool = False
    ) -> requests.Response:
        if not get_chunks:
            url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}"
        else:
            url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs/{name}/chunks"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response

    def list_documents(self, namespace: str, collection: str) -> requests.Response:
        url = (
            f"{self._base_document_index_url}/collections/{namespace}/{collection}/docs"
        )
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response

    def search(
        self,
        namespace: str,
        collection: str,
        query: str,
        max_results: int,
        min_score: float,
    ) -> requests.Response:
        url = f"{self._base_document_index_url}/collections/{namespace}/{collection}/search"
        data = {
            "query": [{"modality": "text", "text": query}],
            "max_results": max_results,
            "min_score": min_score,
            "filter": [{"with": [{"modality": "text"}]}],
        }
        response = requests.post(url, data=json.dumps(data), headers=self.headers)
        response.raise_for_status()
        return response


class DocumentIndexRetriever(BaseRetriever):
    """Search through the documents stored in the Document Index

    We initialize this Retriever with a collection & namespace names, and we can find the documents in the collection
    most semanticly similar to our query.

    Args:
        client: An instance of the Aleph Alpha client.
        namespace: Your user namespace where all the collections are stored.
        collection: The specyfic collection where you want to search through the documents.
        base_document_index_url: the url address of the document index
        threshold: A mimumum value of the cosine similarity between the query vector and the document vector

    Example:
        >>> query = "Do you like summer?"
        >>> retriever = DocumentIndexRetriever(client)
        >>> documents = retriever.get_relevant_documents_with_scores(query, NoOpDebugLogger(), k=2)
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
            for result in response.json()
        ]
        return relevant_chunks
